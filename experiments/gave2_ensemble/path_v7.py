from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import remove_small_objects, skeletonize

from .data import derive_av3_target, list_case_ids, read_png_float
from .data_v6 import GAVE2DatasetV6
from .metrics_v7 import challenge_selection_score_v7
from .predict_v6 import FloatProbabilityStore, oof_case_ownership
from .submission import save_probability_png


def _logit(value: np.ndarray | float) -> np.ndarray:
    clipped = np.clip(value, 1e-4, 1.0 - 1e-4)
    return np.log(clipped) - np.log1p(-clipped)


def calibrate_threshold(probability: np.ndarray, threshold: float) -> np.ndarray:
    """Monotonic piecewise calibration mapping a selected threshold to 0.5."""

    value = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    threshold = float(np.clip(threshold, 1e-3, 1.0 - 1e-3))
    lower = 0.5 * value / threshold
    upper = 0.5 + 0.5 * (value - threshold) / (1.0 - threshold)
    return np.where(value <= threshold, lower, upper).astype(np.float32)


def _branch_probability(
    support: np.ndarray,
    evidence: np.ndarray,
    local_probability: np.ndarray,
    *,
    strength: float,
    margin: float,
    minimum_length: int,
) -> np.ndarray:
    skeleton = skeletonize(support)
    if not skeleton.any() or strength <= 0:
        return local_probability
    degree = ndimage.convolve(skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant")
    degree = degree - skeleton.astype(np.uint8)
    junctions = skeleton & (degree != 2)
    edge_labels, edge_count = ndimage.label(skeleton & ~junctions, structure=np.ones((3, 3), dtype=np.uint8))
    skeleton_probability = local_probability.copy()
    assigned = np.zeros_like(support, dtype=bool)
    for edge_index in range(1, edge_count + 1):
        edge = edge_labels == edge_index
        if int(edge.sum()) < minimum_length:
            continue
        score = float(np.median(evidence[edge]))
        if abs(score) < margin:
            continue
        skeleton_probability[edge] = 0.995 if score > 0 else 0.005
        assigned[edge] = True
    if not assigned.any():
        return local_probability
    _, indices = ndimage.distance_transform_edt(~assigned, return_indices=True)
    propagated = skeleton_probability[indices[0], indices[1]]
    output = local_probability.copy()
    output[support] = (1.0 - strength) * output[support] + strength * propagated[support]
    return np.clip(output, 1e-4, 1.0 - 1e-4)


def reconstruct_probability(probability: np.ndarray, roi: np.ndarray, parameters: dict[str, object]) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W] probability, got {value.shape}")
    roi_mask = np.asarray(roi, dtype=np.float32) > 0.5
    vessel_threshold = float(parameters["vessel_threshold"])
    low = vessel_threshold * float(parameters["low_ratio"])
    vessel_raw = value[1] * roi_mask
    support = apply_hysteresis_threshold(vessel_raw, low, vessel_threshold) & roi_mask
    minimum_size = int(parameters["minimum_size"])
    if minimum_size > 0:
        support = remove_small_objects(support, min_size=minimum_size)

    evidence = _logit(value[0]) - _logit(value[2])
    sigma = float(parameters["sigma"])
    if sigma > 0:
        weight = np.maximum(vessel_raw, 0.05) * support
        numerator = ndimage.gaussian_filter(evidence * weight, sigma=sigma, mode="nearest")
        denominator = ndimage.gaussian_filter(weight, sigma=sigma, mode="nearest")
        evidence = numerator / np.maximum(denominator, 1e-4)
    av_threshold = float(parameters["av_threshold"])
    temperature = float(parameters["temperature"])
    local = 1.0 / (1.0 + np.exp(-np.clip((evidence - float(_logit(av_threshold))) / temperature, -20, 20)))
    local = np.where(local >= 0.5, np.maximum(local, 0.5001), np.minimum(local, 0.4999))
    conditional = _branch_probability(
        support,
        evidence - float(_logit(av_threshold)),
        local,
        strength=float(parameters["branch_strength"]),
        margin=float(parameters["branch_margin"]),
        minimum_length=int(parameters["minimum_branch_length"]),
    )

    vessel = calibrate_threshold(vessel_raw, vessel_threshold)
    bridge_floor = float(parameters["bridge_floor"])
    vessel[support & (vessel < 0.5)] = max(bridge_floor, 0.5001)
    vessel *= roi_mask
    artery = np.minimum(vessel, conditional)
    vein = np.minimum(vessel, 1.0 - conditional)
    output = np.stack((artery, vessel, vein), axis=0).astype(np.float32)
    output *= roi_mask[None]
    if np.any((output[0] >= 0.5) & (output[2] >= 0.5)):
        raise RuntimeError("V7 reconstruction violated A/V exclusivity")
    return np.ascontiguousarray(output)


def _load_targets(data_root: Path, case_ids: list[str]):
    targets = {}
    roi = {}
    for case_id in case_ids:
        targets[case_id] = derive_av3_target(
            read_png_float(data_root / "training" / "av" / f"{case_id}.png", channels=3)
        )
        roi[case_id] = read_png_float(
            data_root / "training" / "masks" / f"{case_id}.png", channels=1
        )[..., 0]
    return targets, roi


def _average_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    keys = (
        "dice_mean",
        "av_sensitivity",
        "av_specificity",
        "av_accuracy",
        "classification_mean",
        "path_recall",
        "path_precision",
        "path_score",
        "selection_score",
    )
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def _evaluate(
    case_ids: list[str],
    probabilities: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    roi: dict[str, np.ndarray],
    parameters: dict[str, object] | None,
) -> dict[str, float]:
    rows = []
    for case_id in case_ids:
        probability = probabilities[case_id]
        if parameters is not None:
            probability = reconstruct_probability(probability, roi[case_id], parameters)
        rows.append(
            challenge_selection_score_v7(
                probability[None], targets[case_id][None], roi[case_id][None, None]
            )
        )
    return _average_metrics(rows)


def _fit_vessel_threshold(case_ids, probabilities, targets, roi) -> float:
    best = (float("-inf"), 0.5)
    for threshold in np.linspace(0.30, 0.90, 13):
        true_positive = false_positive = false_negative = 0.0
        for case_id in case_ids:
            valid = roi[case_id] > 0.5
            pred = probabilities[case_id][1] >= threshold
            target = targets[case_id][1] > 0.5
            true_positive += float((pred & target & valid).sum())
            false_positive += float((pred & ~target & valid).sum())
            false_negative += float((~pred & target & valid).sum())
        dice = 2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1.0)
        sensitivity = true_positive / max(true_positive + false_negative, 1.0)
        score = 0.55 * dice + 0.45 * sensitivity
        if score > best[0]:
            best = (score, float(threshold))
    return best[1]


def _fit_av_threshold(case_ids, probabilities, targets, roi) -> float:
    samples = []
    labels = []
    for case_id in case_ids:
        target = targets[case_id]
        exclusive = np.logical_xor(target[0] > 0.5, target[2] > 0.5) & (roi[case_id] > 0.5)
        # Deterministic stride bounds tuning memory while retaining every vessel tree.
        selected = np.flatnonzero(exclusive.ravel())[::4]
        evidence = (_logit(probabilities[case_id][0]) - _logit(probabilities[case_id][2])).ravel()[selected]
        samples.append(1.0 / (1.0 + np.exp(-np.clip(evidence, -20, 20))))
        labels.append((target[2].ravel()[selected] < 0.5).astype(np.uint8))
    probability = np.concatenate(samples)
    label = np.concatenate(labels).astype(bool)
    best = (float("-inf"), 0.5)
    for threshold in np.linspace(0.35, 0.65, 13):
        prediction = probability >= threshold
        artery_recall = float((prediction & label).sum() / max(int(label.sum()), 1))
        vein_recall = float((~prediction & ~label).sum() / max(int((~label).sum()), 1))
        score = 0.5 * artery_recall + 0.5 * vein_recall
        if score > best[0]:
            best = (score, float(threshold))
    return best[1]


def candidate_parameters(case_ids, probabilities, targets, roi) -> list[dict[str, object]]:
    vessel = _fit_vessel_threshold(case_ids, probabilities, targets, roi)
    av = _fit_av_threshold(case_ids, probabilities, targets, roi)
    candidates = []
    # Two pre-registered profiles keep native-resolution cross-fitting affordable.
    for vessel_delta, sigma, strength, bridge in (
        (0.00, 1.0, 0.55, 0.505),
        (-0.03, 1.8, 0.75, 0.515),
    ):
        candidates.append(
            {
                "vessel_threshold": float(np.clip(vessel + vessel_delta, 0.2, 0.95)),
                "low_ratio": 0.72,
                "av_threshold": av,
                "temperature": 0.70,
                "sigma": sigma,
                "branch_strength": strength,
                "branch_margin": 0.12,
                "minimum_branch_length": 6,
                "minimum_size": 4,
                "bridge_floor": bridge,
            }
        )
    return candidates


def _select_parameters(case_ids, probabilities, targets, roi):
    scored = []
    for parameters in candidate_parameters(case_ids, probabilities, targets, roi):
        metrics = _evaluate(case_ids, probabilities, targets, roi, parameters)
        scored.append({"parameters": parameters, "metrics": metrics})
    scored.sort(key=lambda row: row["metrics"]["selection_score"], reverse=True)
    return scored[0], scored


def fit_cross_fitted(args: argparse.Namespace) -> dict[str, object]:
    manifest = json.loads(args.fold_manifest.read_text(encoding="utf-8"))
    ownership = oof_case_ownership(manifest)
    case_ids = sorted(ownership)
    source = FloatProbabilityStore(args.source_store_root, task=args.task, split="oof")
    missing = [case_id for case_id in case_ids if not source.is_case_complete(case_id)]
    if missing:
        raise FileNotFoundError(f"OOF source store is incomplete: {missing[:10]}")
    probabilities = {case_id: source.read_case(case_id) for case_id in case_ids}
    targets, roi = _load_targets(args.data_root, case_ids)
    output = FloatProbabilityStore(args.crossfit_output_store_root, task=args.task, split="oof")
    fold_reports = []
    for fold_index in range(3):
        apply_ids = sorted(case_id for case_id in case_ids if ownership[case_id] == fold_index)
        fit_ids = sorted(set(case_ids) - set(apply_ids))
        selected, _ = _select_parameters(fit_ids, probabilities, targets, roi)
        fold_reports.append(
            {"fold_index": fold_index, "fit_case_ids": fit_ids, "apply_case_ids": apply_ids, **selected}
        )
        for case_id in apply_ids:
            transformed = reconstruct_probability(probabilities[case_id], roi[case_id], selected["parameters"])
            output.write_case(
                case_id,
                transformed,
                provenance={"version": 7, "mode": "crossfit_path", "fold_index": fold_index},
            )

    crossfit_probabilities = {case_id: output.read_case(case_id) for case_id in case_ids}
    raw_metrics = _evaluate(case_ids, probabilities, targets, roi, None)
    crossfit_metrics = _evaluate(case_ids, crossfit_probabilities, targets, roi, None)
    deployment, deployment_candidates = _select_parameters(case_ids, probabilities, targets, roi)
    accepted = (
        crossfit_metrics["selection_score"] >= raw_metrics["selection_score"] + args.minimum_selection_gain
        and crossfit_metrics["path_score"] >= raw_metrics["path_score"] + args.minimum_path_gain
        and crossfit_metrics["dice_mean"] >= raw_metrics["dice_mean"] - args.maximum_dice_drop
        and crossfit_metrics["av_sensitivity"] >= raw_metrics["av_sensitivity"] - args.maximum_sensitivity_drop
    )
    report = {
        "version": 7,
        "task": args.task,
        "source_store_root": str(args.source_store_root),
        "crossfit_output_store_root": str(args.crossfit_output_store_root),
        "raw_metrics": raw_metrics,
        "crossfit_metrics": crossfit_metrics,
        "gate": {
            "accepted": bool(accepted),
            "minimum_selection_gain": args.minimum_selection_gain,
            "minimum_path_gain": args.minimum_path_gain,
            "maximum_dice_drop": args.maximum_dice_drop,
            "maximum_sensitivity_drop": args.maximum_sensitivity_drop,
        },
        "fold_reports": fold_reports,
        "deployment_parameters": deployment["parameters"],
        "deployment_oof_metrics": deployment["metrics"],
        "deployment_candidates": deployment_candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def apply_report(args: argparse.Namespace) -> dict[str, object]:
    report = json.loads(args.report.read_text(encoding="utf-8"))
    source = FloatProbabilityStore(args.source_store_root, task=args.task, split=args.source_split)
    output = FloatProbabilityStore(args.output_store_root, task=args.task, split=args.source_split)
    case_ids = source.list_complete_cases()
    dataset_split = "training" if args.source_split == "oof" else "validation"
    dataset = GAVE2DatasetV6(args.data_root, split=dataset_split, task=args.task, case_ids=case_ids, require_target=False)
    use_path = bool(report["gate"]["accepted"]) or args.force
    written = []
    for sample in dataset:
        probability = source.read_case(sample.case_id)
        if use_path:
            probability = reconstruct_probability(probability, sample.mask[0], report["deployment_parameters"])
        else:
            probability = np.ascontiguousarray(probability * (sample.mask[0] > 0.5)[None], dtype=np.float32)
        output.write_case(
            sample.case_id,
            probability,
            provenance={"version": 7, "mode": "deployment_path" if use_path else "raw_gate_fallback"},
        )
        written.append(sample.case_id)
    return {"task": args.task, "used_path": use_path, "written_cases": written}


def promote(args: argparse.Namespace) -> dict[str, object]:
    store = FloatProbabilityStore(args.source_store_root, task=args.task, split="validation")
    case_ids = list_case_ids(args.data_root, split="validation")
    task_name = "Task1" if args.task == "task1" else "Task2"
    output_dir = args.output_root / args.team_id / task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    for case_id in case_ids:
        if not store.is_case_complete(case_id):
            raise FileNotFoundError(f"Missing V7 validation probability {case_id}")
        save_probability_png(store.read_case(case_id), output_dir / f"{case_id}.png")
    return {"task": args.task, "output_dir": str(output_dir), "count": len(case_ids)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-fitted V7 path reconstruction and promotion.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--data-root", type=Path, required=True)
    fit.add_argument("--fold-manifest", type=Path, required=True)
    fit.add_argument("--source-store-root", type=Path, required=True)
    fit.add_argument("--crossfit-output-store-root", type=Path, required=True)
    fit.add_argument("--task", choices=("task1", "task2"), required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--minimum-selection-gain", type=float, default=0.005)
    fit.add_argument("--minimum-path-gain", type=float, default=0.01)
    fit.add_argument("--maximum-dice-drop", type=float, default=0.02)
    fit.add_argument("--maximum-sensitivity-drop", type=float, default=0.03)
    apply = subparsers.add_parser("apply")
    apply.add_argument("--data-root", type=Path, required=True)
    apply.add_argument("--source-store-root", type=Path, required=True)
    apply.add_argument("--output-store-root", type=Path, required=True)
    apply.add_argument("--report", type=Path, required=True)
    apply.add_argument("--task", choices=("task1", "task2"), required=True)
    apply.add_argument("--source-split", choices=("oof", "validation"), default="validation")
    apply.add_argument("--force", action="store_true")
    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--data-root", type=Path, required=True)
    promote_parser.add_argument("--source-store-root", type=Path, required=True)
    promote_parser.add_argument("--output-root", type=Path, required=True)
    promote_parser.add_argument("--team-id", required=True)
    promote_parser.add_argument("--task", choices=("task1", "task2"), required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.command == "fit":
        result = fit_cross_fitted(args)
    elif args.command == "apply":
        result = apply_report(args)
    else:
        result = promote(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
