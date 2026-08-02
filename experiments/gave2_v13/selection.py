from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

from experiments.gave2_ensemble.data import derive_av3_target
from experiments.gave2_v8.metrics import PathCounts, path_counts
from experiments.gave2_v8.store import ProbabilityStore
from experiments.gave2_v12.constants import (
    LIVE_CLASSIFICATION_WEIGHT,
    LIVE_DICE_WEIGHT,
    LIVE_TOPOLOGY_WEIGHT,
)
from experiments.gave2_v12.folds import validate_manifest
from experiments.gave2_v12.metrics import evaluate_cases
from experiments.gave2_v12.utils import atomic_json, case_ids


@dataclass(frozen=True)
class PathSettings:
    alpha: float
    seed_threshold: float
    grow_threshold: float
    teacher_seed_threshold: float
    support_threshold: float = 0.15
    support_radius: int = 2
    crossing_threshold: float = 0.46


@dataclass(frozen=True)
class TopologySafeSettings:
    alpha: float
    decision_threshold: float
    temperature: float
    corridor_radius: int = 2


def _teacher_corridor(teacher: np.ndarray, radius: int) -> np.ndarray:
    teacher = np.asarray(teacher, dtype=np.float32)
    protected = np.zeros_like(teacher, dtype=bool)
    for channel in range(teacher.shape[0]):
        path = skeletonize(teacher[channel] >= 0.5)
        if radius > 0:
            path = ndimage.binary_dilation(path, iterations=radius)
        protected[channel] = path & (teacher[channel] >= 0.5)
    return protected


def topology_safe_prune(
    raw: np.ndarray,
    teacher: np.ndarray,
    settings: TopologySafeSettings,
    *,
    protected: np.ndarray | None = None,
) -> np.ndarray:
    """Use V13 only to prune teacher positives while preserving teacher paths."""

    raw = np.clip(np.asarray(raw, dtype=np.float32), 0.0, 1.0)
    teacher = np.clip(np.asarray(teacher, dtype=np.float32), 0.0, 1.0)
    if raw.shape != teacher.shape or raw.ndim != 3 or raw.shape[0] != 3:
        raise ValueError("raw and teacher must be matching [3,H,W] probabilities")
    if not 0.0 <= settings.alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    if not 0.0 < settings.decision_threshold < 1.0 or settings.temperature <= 0.0:
        raise ValueError("invalid calibration parameters")
    if settings.corridor_radius < 0:
        raise ValueError("corridor_radius must be non-negative")

    blended = np.clip(
        teacher + settings.alpha * (raw - teacher),
        1e-5,
        1.0 - 1e-5,
    )
    decision_bias = np.log(settings.decision_threshold / (1.0 - settings.decision_threshold))
    logits = np.log(blended / (1.0 - blended))
    calibrated = 1.0 / (
        1.0 + np.exp(-np.clip((logits - decision_bias) / settings.temperature, -20.0, 20.0))
    )

    teacher_positive = teacher >= 0.5
    calibrated = np.where(teacher_positive, calibrated, np.minimum(calibrated, teacher))
    if protected is None:
        protected = _teacher_corridor(teacher, settings.corridor_radius)
    elif protected.shape != teacher.shape:
        raise ValueError("protected corridor must match teacher shape")
    calibrated[protected] = np.maximum(calibrated[protected], teacher[protected])
    calibrated[1] = np.maximum.reduce((calibrated[1], calibrated[0], calibrated[2]))
    calibrated[:, ~np.isfinite(calibrated).all(axis=0)] = 0.0
    return np.clip(calibrated, 0.0, 1.0).astype(np.float32)


def _apply_settings(
    raw: np.ndarray,
    teacher: np.ndarray,
    settings: dict[str, object],
    *,
    protected: np.ndarray | None = None,
) -> np.ndarray:
    method = str(settings.get("method", "geodesic_hysteresis"))
    values = {key: value for key, value in settings.items() if key != "method"}
    if method == "topology_safe_prune":
        return topology_safe_prune(raw, teacher, TopologySafeSettings(**values), protected=protected)
    if method == "geodesic_hysteresis":
        return geodesic_hysteresis(raw, teacher, PathSettings(**values))
    raise ValueError(f"Unknown V13 selection method: {method}")


def _open_store(path: Path | str) -> ProbabilityStore:
    root = Path(path)
    manifest = json.loads((root / "completion_manifest.json").read_text(encoding="utf-8"))
    return ProbabilityStore(root, namespace=str(manifest["namespace"]), split=str(manifest["split"]))


def _propagate(seed: np.ndarray, grow: np.ndarray) -> np.ndarray:
    seed = np.asarray(seed, dtype=bool) & np.asarray(grow, dtype=bool)
    if not seed.any():
        return seed
    return ndimage.binary_propagation(seed, mask=grow)


def geodesic_hysteresis(
    raw: np.ndarray,
    teacher: np.ndarray,
    settings: PathSettings,
) -> np.ndarray:
    """Calibrate class paths while keeping all corrections on trusted vessel support."""

    raw = np.clip(np.asarray(raw, dtype=np.float32), 0.0, 1.0)
    teacher = np.clip(np.asarray(teacher, dtype=np.float32), 0.0, 1.0)
    if raw.shape != teacher.shape or raw.ndim != 3 or raw.shape[0] != 3:
        raise ValueError("raw and teacher must be matching [3,H,W] probabilities")
    if not 0.0 <= settings.alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    if not 0.0 < settings.grow_threshold < settings.seed_threshold < 1.0:
        raise ValueError("expected 0 < grow_threshold < seed_threshold < 1")

    blended = teacher + settings.alpha * (raw - teacher)
    support = teacher[1] >= settings.support_threshold
    if settings.support_radius > 0:
        support = ndimage.binary_dilation(support, iterations=settings.support_radius)

    selected: list[np.ndarray] = []
    for channel in (0, 2):
        strong_teacher = teacher[channel] >= settings.teacher_seed_threshold
        seed = support & ((blended[channel] >= settings.seed_threshold) | strong_teacher)
        grow = support & (
            (blended[channel] >= settings.grow_threshold)
            | (teacher[channel] >= settings.grow_threshold)
        )
        selected.append(_propagate(seed, grow))

    artery, vein = selected
    overlap = artery & vein
    crossing = overlap & (raw[0] >= settings.crossing_threshold) & (raw[2] >= settings.crossing_threshold)
    exclusive_overlap = overlap & ~crossing
    artery[exclusive_overlap & (raw[0] < raw[2])] = False
    vein[exclusive_overlap & (raw[2] <= raw[0])] = False

    result = blended.copy()
    for channel, binary in ((0, artery), (2, vein)):
        result[channel] = np.where(binary, np.maximum(result[channel], 0.55), np.minimum(result[channel], 0.45))
        result[channel, ~support] = np.minimum(result[channel, ~support], teacher[channel, ~support])
    result[1] = np.maximum.reduce((blended[1], result[0], result[2]))
    result[:, ~np.isfinite(result).all(axis=0)] = 0.0
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _training_arrays(data_root: Path, ids: list[str]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    targets: list[np.ndarray] = []
    rois: list[np.ndarray] = []
    for case_id in ids:
        label = np.asarray(
            Image.open(data_root / "training" / "av" / f"{case_id}.png").convert("RGB"),
            dtype=np.float32,
        ) / 255.0
        targets.append(derive_av3_target(label).astype(np.float32))
        rois.append(
            np.asarray(Image.open(data_root / "training" / "masks" / f"{case_id}.png").convert("L")) > 127
        )
    return targets, rois


def _compact(report: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in report.items() if key != "case_metrics"}


def _fold_reports(
    ids: list[str],
    manifest: dict[str, object],
    candidate: list[np.ndarray],
    teacher: list[np.ndarray],
    targets: list[np.ndarray],
    rois: list[np.ndarray],
) -> list[dict[str, object]]:
    index = {case_id: position for position, case_id in enumerate(ids)}
    reports: list[dict[str, object]] = []
    for fold in manifest["folds"]:
        positions = [index[case_id] for case_id in fold["validation"]]
        baseline = evaluate_cases(
            [teacher[position] for position in positions],
            [targets[position] for position in positions],
            [rois[position] for position in positions],
        )
        trial = evaluate_cases(
            [candidate[position] for position in positions],
            [targets[position] for position in positions],
            [rois[position] for position in positions],
        )
        reports.append(
            {
                "fold": int(fold["fold"]),
                "cases": len(positions),
                "score_gain": float(trial["score"] - baseline["score"]),
                "topology_gain": float(trial["topology"] - baseline["topology"]),
                "dice_gain": float(trial["dice"] - baseline["dice"]),
                "sensitivity_drop": float(baseline["sensitivity"] - trial["sensitivity"]),
                "candidate_score": float(trial["score"]),
            }
        )
    return reports


def _path_records(
    ids: list[str],
    probabilities: list[np.ndarray],
    targets: list[np.ndarray],
    rois: list[np.ndarray],
    *,
    paths_per_case: int,
    seed: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for sequence, (case_id, probability, target, roi) in enumerate(
        zip(ids, probabilities, targets, rois, strict=True), 1
    ):
        channels: dict[int, dict[str, object]] = {}
        for channel in (0, 2):
            prediction = (probability[channel] >= 0.5) & roi
            truth = (target[channel] > 0.5) & roi
            channels[channel] = {
                "tp": int((prediction & truth).sum()),
                "fp": int((prediction & ~truth & roi).sum()),
                "fn": int((~prediction & truth).sum()),
                "tn": int((~prediction & ~truth & roi).sum()),
                "paths": path_counts(
                    prediction,
                    truth,
                    paths=paths_per_case,
                    seed=seed,
                    case_id=case_id,
                    channel=channel,
                ),
            }
        records.append({"case_id": case_id, "channels": channels})
        print(f"[{sequence:02d}/{len(ids):02d}] sampled path audit {case_id}", flush=True)
    return records


def _aggregate_path_records(records: list[dict[str, object]], positions: list[int] | None = None) -> dict[str, float]:
    if positions is None:
        positions = list(range(len(records)))
    channel_reports: list[dict[str, float]] = []
    for channel in (0, 2):
        totals = {name: 0 for name in ("tp", "fp", "fn", "tn")}
        paths = PathCounts()
        for position in positions:
            values = records[position]["channels"][channel]
            for name in totals:
                totals[name] += int(values[name])
            paths = paths + values["paths"]
        tp, fp, fn, tn = (totals[name] for name in ("tp", "fp", "fn", "tn"))
        correct = paths.correct / max(paths.total, 1)
        infeasible = paths.infeasible / max(paths.total, 1)
        channel_reports.append(
            {
                "dice": 2.0 * tp / max(2 * tp + fp + fn, 1),
                "sensitivity": tp / max(tp + fn, 1),
                "specificity": tn / max(tn + fp, 1),
                "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
                "correct": correct,
                "infeasible": infeasible,
                "topology": 0.5 * correct + 0.5 * (1.0 - infeasible),
            }
        )
    report = {
        name: float(np.mean([channel[name] for channel in channel_reports]))
        for name in ("dice", "sensitivity", "specificity", "accuracy", "correct", "infeasible", "topology")
    }
    report["classification"] = (
        0.3 * report["sensitivity"] + 0.3 * report["specificity"] + 0.4 * report["accuracy"]
    )
    report["score"] = 10.0 * (
        LIVE_CLASSIFICATION_WEIGHT * report["classification"]
        + LIVE_DICE_WEIGHT * report["dice"]
        + LIVE_TOPOLOGY_WEIGHT * report["topology"]
    )
    return report


def _path_fold_reports(
    ids: list[str],
    manifest: dict[str, object],
    candidate_records: list[dict[str, object]],
    baseline_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    index = {case_id: position for position, case_id in enumerate(ids)}
    output: list[dict[str, object]] = []
    for fold in manifest["folds"]:
        positions = [index[case_id] for case_id in fold["validation"]]
        baseline = _aggregate_path_records(baseline_records, positions)
        candidate = _aggregate_path_records(candidate_records, positions)
        output.append(
            {
                "fold": int(fold["fold"]),
                "cases": len(positions),
                "score_gain": float(candidate["score"] - baseline["score"]),
                "topology_gain": float(candidate["topology"] - baseline["topology"]),
                "correct_gain": float(candidate["correct"] - baseline["correct"]),
                "infeasible_reduction": float(baseline["infeasible"] - candidate["infeasible"]),
                "candidate_score": float(candidate["score"]),
            }
        )
    return output


def _reassignment_fraction(candidate: list[np.ndarray], teacher: list[np.ndarray]) -> float:
    additions = positives = 0
    for trial, baseline in zip(candidate, teacher, strict=True):
        trial_av = trial[(0, 2), ...] >= 0.5
        teacher_av = baseline[(0, 2), ...] >= 0.5
        additions += int((trial_av & ~teacher_av).sum())
        positives += int(teacher_av.sum())
    return float(additions / max(positives, 1))


def _removal_fraction(candidate: list[np.ndarray], teacher: list[np.ndarray]) -> float:
    removals = positives = 0
    for trial, baseline in zip(candidate, teacher, strict=True):
        trial_av = trial[(0, 2), ...] >= 0.5
        teacher_av = baseline[(0, 2), ...] >= 0.5
        removals += int((teacher_av & ~trial_av).sum())
        positives += int(teacher_av.sum())
    return float(removals / max(positives, 1))


def search(args: argparse.Namespace) -> dict[str, object]:
    ids = case_ids(args.data_root, "training")
    manifest = json.loads(args.fold_manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest, ids)
    teacher_store = _open_store(args.teacher_store)
    raw_store = _open_store(args.raw_store)
    if teacher_store.split != "training" or raw_store.split != "training":
        raise RuntimeError("OOF selection requires training stores")
    if teacher_store.list_cases() != ids or raw_store.list_cases() != ids:
        raise RuntimeError("Teacher or V13 OOF store is incomplete")

    teachers = [teacher_store.read_case(case_id) for case_id in ids]
    raw = [raw_store.read_case(case_id) for case_id in ids]
    targets, rois = _training_arrays(args.data_root, ids)
    baseline = evaluate_cases(teachers, targets, rois)
    protected = [
        _teacher_corridor(teacher, args.corridor_radius)
        for teacher in teachers
    ]
    settings_grid: list[dict[str, object]] = []
    if args.search_mode in {"topology_safe", "both"}:
        if any(not 0.0 <= value <= 1.0 for value in args.alpha_values):
            raise ValueError("all alpha values must be in [0,1]")
        if any(not 0.0 < value < 1.0 for value in args.decision_threshold_values):
            raise ValueError("all decision thresholds must be in (0,1)")
        if any(value <= 0.0 for value in args.temperature_values):
            raise ValueError("all temperatures must be positive")
        for alpha, threshold, temperature in itertools.product(
            args.alpha_values,
            args.decision_threshold_values,
            args.temperature_values,
        ):
            settings_grid.append(
                {
                    "method": "topology_safe_prune",
                    "alpha": alpha,
                    "decision_threshold": threshold,
                    "temperature": temperature,
                    "corridor_radius": args.corridor_radius,
                }
            )
    if args.search_mode in {"geodesic", "both"}:
        for alpha, seed, grow, teacher_seed in itertools.product(
            (0.75, 1.0),
            (0.62, 0.70),
            (0.30, 0.38),
            (0.75, 0.85),
        ):
            path_settings = PathSettings(
                alpha=alpha,
                seed_threshold=seed,
                grow_threshold=grow,
                teacher_seed_threshold=teacher_seed,
                support_threshold=args.support_threshold,
                support_radius=args.support_radius,
                crossing_threshold=args.crossing_threshold,
            )
            settings_grid.append({"method": "geodesic_hysteresis", **asdict(path_settings)})

    candidates: list[dict[str, object]] = []
    for settings in settings_grid:
        probabilities = [
            _apply_settings(prediction, teacher, settings, protected=corridor)
            for prediction, teacher, corridor in zip(raw, teachers, protected, strict=True)
        ]
        metrics = evaluate_cases(probabilities, targets, rois)
        folds = _fold_reports(ids, manifest, probabilities, teachers, targets, rois)
        score_gain = float(metrics["score"] - baseline["score"])
        topology_gain = float(metrics["topology"] - baseline["topology"])
        sensitivity_drop = float(baseline["sensitivity"] - metrics["sensitivity"])
        min_fold_score_gain = min(float(fold["score_gain"]) for fold in folds)
        min_fold_topology_gain = min(float(fold["topology_gain"]) for fold in folds)
        max_fold_sensitivity_drop = max(float(fold["sensitivity_drop"]) for fold in folds)
        reassignment = _reassignment_fraction(probabilities, teachers)
        removal = _removal_fraction(probabilities, teachers)
        fast_accepted = (
            score_gain >= min(args.minimum_score_gain, 0.15)
            and min_fold_score_gain >= min(args.minimum_fold_score_gain, 0.0)
            and sensitivity_drop <= args.maximum_sensitivity_drop
            and max_fold_sensitivity_drop <= args.maximum_fold_sensitivity_drop
            and reassignment <= args.maximum_reassignment_fraction
        )
        candidates.append(
            {
                "settings": settings,
                "accepted": False,
                "fast_accepted": bool(fast_accepted),
                "path_audited": False,
                "fast_score": float(metrics["score"]),
                "fast_score_gain": score_gain,
                "fast_topology_gain": topology_gain,
                "fast_minimum_fold_score_gain": min_fold_score_gain,
                "fast_minimum_fold_topology_gain": min_fold_topology_gain,
                "sensitivity_drop": sensitivity_drop,
                "maximum_fold_sensitivity_drop": max_fold_sensitivity_drop,
                "class_reassignment_fraction": reassignment,
                "class_removal_fraction": removal,
                "fast_metrics": _compact(metrics),
                "fast_fold_reports": folds,
            }
        )

    shortlist_pool = [candidate for candidate in candidates if candidate["fast_accepted"]] or candidates
    shortlist = sorted(shortlist_pool, key=lambda value: float(value["fast_score"]), reverse=True)[: args.path_shortlist]
    print("Auditing R2-V2 baseline with sampled COR/INF paths", flush=True)
    baseline_path_records = _path_records(
        ids,
        teachers,
        targets,
        rois,
        paths_per_case=args.paths_per_case,
        seed=args.path_seed,
    )
    baseline_path = _aggregate_path_records(baseline_path_records)
    for rank, candidate in enumerate(shortlist, 1):
        print(f"Path-auditing shortlist {rank}/{len(shortlist)}: {candidate['settings']}", flush=True)
        probabilities = [
            _apply_settings(prediction, teacher, candidate["settings"], protected=corridor)
            for prediction, teacher, corridor in zip(raw, teachers, protected, strict=True)
        ]
        records = _path_records(
            ids,
            probabilities,
            targets,
            rois,
            paths_per_case=args.paths_per_case,
            seed=args.path_seed,
        )
        metrics = _aggregate_path_records(records)
        folds = _path_fold_reports(ids, manifest, records, baseline_path_records)
        score_gain = float(metrics["score"] - baseline_path["score"])
        topology_gain = float(metrics["topology"] - baseline_path["topology"])
        min_fold_score_gain = min(float(fold["score_gain"]) for fold in folds)
        min_fold_topology_gain = min(float(fold["topology_gain"]) for fold in folds)
        accepted = (
            bool(candidate["fast_accepted"])
            and score_gain >= args.minimum_score_gain
            and min_fold_score_gain >= args.minimum_fold_score_gain
            and topology_gain >= args.minimum_topology_gain
            and min_fold_topology_gain >= args.minimum_fold_topology_gain
        )
        candidate.update(
            {
                "accepted": bool(accepted),
                "path_audited": True,
                "score": float(metrics["score"]),
                "score_gain": score_gain,
                "topology_gain": topology_gain,
                "minimum_fold_score_gain": min_fold_score_gain,
                "minimum_fold_topology_gain": min_fold_topology_gain,
                "metrics": metrics,
                "fold_reports": folds,
            }
        )

    passing = [candidate for candidate in candidates if candidate["accepted"]]
    if passing:
        selected = max(
            passing,
            key=lambda value: (
                float(value["minimum_fold_score_gain"]),
                float(value["score"]),
            ),
        )
        accepted = True
    else:
        selected = {
            "settings": None,
            "accepted": False,
            "score": float(baseline_path["score"]),
            "score_gain": 0.0,
            "topology_gain": 0.0,
            "minimum_fold_score_gain": 0.0,
            "minimum_fold_topology_gain": 0.0,
            "sensitivity_drop": 0.0,
            "maximum_fold_sensitivity_drop": 0.0,
            "class_reassignment_fraction": 0.0,
            "class_removal_fraction": 0.0,
            "metrics": baseline_path,
            "fold_reports": [],
        }
        accepted = False
    report = {
        "version": 13,
        "task": args.task,
        "accepted": accepted,
        "selection_basis": (
            "three-fold OOF topology-safe pruning with live-weighted classification, "
            "Dice, sampled COR, and sampled INF"
        ),
        "search_mode": args.search_mode,
        "baseline": baseline_path,
        "fast_baseline": _compact(baseline),
        "selected": selected,
        "thresholds": {
            "minimum_score_gain": args.minimum_score_gain,
            "minimum_fold_score_gain": args.minimum_fold_score_gain,
            "minimum_topology_gain": args.minimum_topology_gain,
            "minimum_fold_topology_gain": args.minimum_fold_topology_gain,
            "maximum_sensitivity_drop": args.maximum_sensitivity_drop,
            "maximum_fold_sensitivity_drop": args.maximum_fold_sensitivity_drop,
            "maximum_reassignment_fraction": args.maximum_reassignment_fraction,
        },
        "path_audit": {
            "shortlist": args.path_shortlist,
            "paths_per_case": args.paths_per_case,
            "seed": args.path_seed,
        },
        "candidates": sorted(candidates, key=lambda value: float(value.get("score", value["fast_score"])), reverse=True),
    }
    atomic_json(args.output_config, report)
    return report


def apply(args: argparse.Namespace) -> dict[str, object]:
    report = json.loads(args.selection.read_text(encoding="utf-8"))
    if report.get("task") != args.task or not report.get("accepted"):
        raise RuntimeError(f"V13 {args.task} OOF selection was not accepted")
    settings = report["selected"]["settings"]
    ids = case_ids(args.data_root, args.split)
    teacher_store = _open_store(args.teacher_store)
    raw_store = _open_store(args.raw_store)
    if teacher_store.list_cases() != ids or raw_store.list_cases() != ids:
        raise RuntimeError("Teacher or V13 raw store is incomplete")
    output = ProbabilityStore(
        args.output_store,
        namespace=f"gave2_v13_selected_{args.task}",
        split=args.split,
    )
    settings_payload = dict(settings)
    written = 0
    for case_id in ids:
        provenance = {
            "version": 13,
            "task": args.task,
            "settings": settings_payload,
            "teacher_sha256": teacher_store.case_record(case_id)["sha256"],
            "raw_sha256": raw_store.case_record(case_id)["sha256"],
        }
        if output.is_complete(case_id, provenance):
            continue
        value = _apply_settings(
            raw_store.read_case(case_id),
            teacher_store.read_case(case_id),
            settings,
        )
        output.write_case(case_id, value, provenance)
        written += 1
    return {
        "version": 13,
        "task": args.task,
        "split": args.split,
        "cases": len(ids),
        "new_cases": written,
        "output": str(args.output_store),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select and apply V13 channel-path postprocessing.")
    commands = parser.add_subparsers(dest="command", required=True)
    search_parser = commands.add_parser("search")
    search_parser.add_argument("--data-root", type=Path, required=True)
    search_parser.add_argument("--fold-manifest", type=Path, required=True)
    search_parser.add_argument("--teacher-store", type=Path, required=True)
    search_parser.add_argument("--raw-store", type=Path, required=True)
    search_parser.add_argument("--output-config", type=Path, required=True)
    search_parser.add_argument("--task", choices=("task1", "task2"), required=True)
    search_parser.add_argument(
        "--search-mode",
        choices=("topology_safe", "geodesic", "both"),
        default="topology_safe",
    )
    search_parser.add_argument("--corridor-radius", type=int, default=2)
    search_parser.add_argument(
        "--alpha-values",
        type=float,
        nargs="+",
        default=(0.10, 0.25, 0.50, 0.75, 1.00),
    )
    search_parser.add_argument(
        "--decision-threshold-values",
        type=float,
        nargs="+",
        default=(0.50, 0.525, 0.55, 0.575),
    )
    search_parser.add_argument(
        "--temperature-values",
        type=float,
        nargs="+",
        default=(0.90, 1.00),
    )
    search_parser.add_argument("--support-threshold", type=float, default=0.15)
    search_parser.add_argument("--support-radius", type=int, default=2)
    search_parser.add_argument("--crossing-threshold", type=float, default=0.46)
    search_parser.add_argument("--minimum-score-gain", type=float, default=0.30)
    search_parser.add_argument("--minimum-fold-score-gain", type=float, default=0.12)
    search_parser.add_argument("--minimum-topology-gain", type=float, default=0.015)
    search_parser.add_argument("--minimum-fold-topology-gain", type=float, default=0.0)
    search_parser.add_argument("--maximum-sensitivity-drop", type=float, default=0.03)
    search_parser.add_argument("--maximum-fold-sensitivity-drop", type=float, default=0.05)
    search_parser.add_argument("--maximum-reassignment-fraction", type=float, default=0.15)
    search_parser.add_argument("--path-shortlist", type=int, default=3)
    search_parser.add_argument("--paths-per-case", type=int, default=100)
    search_parser.add_argument("--path-seed", type=int, default=77)

    apply_parser = commands.add_parser("apply")
    apply_parser.add_argument("--data-root", type=Path, required=True)
    apply_parser.add_argument("--teacher-store", type=Path, required=True)
    apply_parser.add_argument("--raw-store", type=Path, required=True)
    apply_parser.add_argument("--output-store", type=Path, required=True)
    apply_parser.add_argument("--selection", type=Path, required=True)
    apply_parser.add_argument("--task", choices=("task1", "task2"), required=True)
    apply_parser.add_argument("--split", choices=("training", "validation"), required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = search(args) if args.command == "search" else apply(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
