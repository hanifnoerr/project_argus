from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

from experiments.gave2_ensemble.data import derive_av3_target
from experiments.gave2_v8.store import ProbabilityStore

from .folds import validate_manifest
from .metrics import evaluate_cases, pixel_score, teacher_path_recall
from .utils import atomic_json, case_ids


def calibrated_residual(
    candidate: np.ndarray,
    teacher: np.ndarray,
    *,
    alpha: float,
    decision_threshold: float,
    temperature: float,
    corridor_radius: int = 2,
    correction_mode: str = "prune",
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    if not 0.0 < decision_threshold < 1.0 or temperature <= 0.0:
        raise ValueError("Invalid calibration parameters")
    teacher = np.asarray(teacher, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if correction_mode not in {"prune", "vessel_support"}:
        raise ValueError(f"Unknown correction mode: {correction_mode}")
    blended = np.clip(teacher + alpha * (candidate - teacher), 1e-5, 1.0 - 1e-5)
    bias = np.log(decision_threshold / (1.0 - decision_threshold))
    logits = np.log(blended / (1.0 - blended))
    calibrated = 1.0 / (1.0 + np.exp(-((logits - bias) / temperature)))
    teacher_positive = teacher >= 0.5
    if correction_mode == "prune":
        allowed_positive = teacher_positive
    else:
        allowed_positive = np.broadcast_to(teacher[1:2] >= 0.5, teacher.shape)
    calibrated = np.where(allowed_positive, calibrated, np.minimum(calibrated, teacher))
    if corridor_radius >= 0:
        for channel in range(3):
            corridor = skeletonize(teacher[channel] >= 0.5)
            if corridor_radius > 0:
                corridor = ndimage.binary_dilation(corridor, iterations=corridor_radius)
            protected = corridor & teacher_positive[channel]
            calibrated[channel, protected] = np.maximum(calibrated[channel, protected], teacher[channel, protected])
    calibrated[1] = np.maximum.reduce((calibrated[1], calibrated[0], calibrated[2]))
    return np.clip(calibrated, 0.0, 1.0).astype(np.float32)


def _training_arrays(data_root: Path, ids: list[str]):
    targets, rois = [], []
    for case_id in ids:
        raw = np.asarray(Image.open(data_root / "training" / "av" / f"{case_id}.png").convert("RGB"), dtype=np.float32) / 255.0
        targets.append(derive_av3_target(raw))
        rois.append(np.asarray(Image.open(data_root / "training" / "masks" / f"{case_id}.png").convert("L")) > 127)
    return targets, rois


def _fold_reports(
    ids: list[str],
    manifest: dict[str, object],
    probabilities: list[np.ndarray],
    teachers: list[np.ndarray],
    targets: list[np.ndarray],
    rois: list[np.ndarray],
) -> list[dict[str, object]]:
    index = {case_id: position for position, case_id in enumerate(ids)}
    reports = []
    for fold in manifest["folds"]:
        positions = [index[case_id] for case_id in fold["validation"]]
        baseline = evaluate_cases(
            [teachers[position] for position in positions],
            [targets[position] for position in positions],
            [rois[position] for position in positions],
        )
        candidate = evaluate_cases(
            [probabilities[position] for position in positions],
            [targets[position] for position in positions],
            [rois[position] for position in positions],
        )
        reports.append(
            {
                "fold": int(fold["fold"]),
                "cases": len(positions),
                "pixel_score_gain": pixel_score(candidate) - pixel_score(baseline),
                "dice_gain": float(candidate["dice"]) - float(baseline["dice"]),
                "sensitivity_drop": float(baseline["sensitivity"]) - float(candidate["sensitivity"]),
            }
        )
    return reports


def search(args: argparse.Namespace) -> dict[str, object]:
    ids = case_ids(args.data_root, "training")
    manifest = json.loads(args.fold_manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest, ids)
    teacher_store = ProbabilityStore(args.teacher_store, namespace="r2v2_direct", split="training")
    raw_store = ProbabilityStore(args.raw_store, namespace=f"gave2_v12_raw_{args.task}", split="training")
    if teacher_store.list_cases() != ids or raw_store.list_cases() != ids:
        raise RuntimeError("Teacher or OOF raw store is incomplete")
    teachers = [teacher_store.read_case(case_id) for case_id in ids]
    raw = [raw_store.read_case(case_id) for case_id in ids]
    targets, rois = _training_arrays(args.data_root, ids)
    baseline = evaluate_cases(teachers, targets, rois)
    correction_mode = args.correction_mode
    candidates = []
    for alpha, threshold, temperature in itertools.product(
        (0.25, 0.50, 0.75, 1.00),
        (0.45, 0.50, 0.55),
        (0.90, 1.00, 1.10),
    ):
        probabilities = [
            calibrated_residual(
                prediction,
                teacher,
                alpha=alpha,
                decision_threshold=threshold,
                temperature=temperature,
                corridor_radius=args.corridor_radius,
                correction_mode=correction_mode,
            )
            for prediction, teacher in zip(raw, teachers)
        ]
        report = evaluate_cases(probabilities, targets, rois)
        path_recall = float(np.mean([teacher_path_recall(value, teacher) for value, teacher in zip(probabilities, teachers)]))
        folds = _fold_reports(ids, manifest, probabilities, teachers, targets, rois)
        score_gain = pixel_score(report) - pixel_score(baseline)
        minimum_fold_gain = min(float(fold["pixel_score_gain"]) for fold in folds)
        sensitivity_drop = float(baseline["sensitivity"]) - float(report["sensitivity"])
        maximum_fold_sensitivity_drop = max(float(fold["sensitivity_drop"]) for fold in folds)
        accepted = (
            path_recall >= args.minimum_teacher_path_recall
            and score_gain >= args.minimum_pixel_score_gain
            and minimum_fold_gain >= args.minimum_fold_pixel_score_gain
            and sensitivity_drop <= args.maximum_sensitivity_drop
            and maximum_fold_sensitivity_drop <= args.maximum_fold_sensitivity_drop
        )
        candidates.append(
            {
                "alpha": alpha,
                "decision_threshold": threshold,
                "temperature": temperature,
                "corridor_radius": args.corridor_radius,
                "correction_mode": correction_mode,
                "teacher_path_recall": path_recall,
                "pixel_score": pixel_score(report),
                "pixel_score_gain": score_gain,
                "minimum_fold_pixel_score_gain": minimum_fold_gain,
                "sensitivity_drop": sensitivity_drop,
                "maximum_fold_sensitivity_drop": maximum_fold_sensitivity_drop,
                "fold_reports": folds,
                "accepted": accepted,
                "metrics": report,
            }
        )
    passing = [candidate for candidate in candidates if candidate["accepted"]]
    if passing:
        selected = max(
            passing,
            key=lambda candidate: (
                float(candidate["minimum_fold_pixel_score_gain"]),
                float(candidate["pixel_score_gain"]),
            ),
        )
        accepted = True
    else:
        selected = {
            "alpha": 0.0,
            "decision_threshold": 0.5,
            "temperature": 1.0,
            "corridor_radius": args.corridor_radius,
            "correction_mode": correction_mode,
            "teacher_path_recall": 1.0,
            "pixel_score": pixel_score(baseline),
            "pixel_score_gain": 0.0,
            "minimum_fold_pixel_score_gain": 0.0,
            "sensitivity_drop": 0.0,
            "maximum_fold_sensitivity_drop": 0.0,
            "fold_reports": [],
            "accepted": False,
            "metrics": baseline,
        }
        accepted = False
    report = {
        "version": 12,
        "task": args.task,
        "accepted": accepted,
        "baseline": baseline,
        "selected": selected,
        "selection_basis": "OOF pixel gain plus fold stability; sampled path proxy is not an acceptance criterion",
        "minimum_pixel_score_gain": args.minimum_pixel_score_gain,
        "minimum_fold_pixel_score_gain": args.minimum_fold_pixel_score_gain,
        "maximum_sensitivity_drop": args.maximum_sensitivity_drop,
        "maximum_fold_sensitivity_drop": args.maximum_fold_sensitivity_drop,
        "minimum_teacher_path_recall": args.minimum_teacher_path_recall,
        "candidates": sorted(candidates, key=lambda candidate: float(candidate["pixel_score_gain"]), reverse=True),
    }
    atomic_json(args.output_config, report)
    return report


def apply(args: argparse.Namespace) -> dict[str, object]:
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    selected = selection["selected"]
    ids = case_ids(args.data_root, args.split)
    teacher_store = ProbabilityStore(args.teacher_store, namespace="r2v2_direct", split=args.split)
    raw_store = ProbabilityStore(args.raw_store, namespace=f"gave2_v12_raw_{args.task}", split=args.split)
    output = ProbabilityStore(args.output_store, namespace=f"gave2_v12_selected_{args.task}", split=args.split)
    if teacher_store.list_cases() != ids or raw_store.list_cases() != ids:
        raise RuntimeError("Teacher or raw store is incomplete")
    settings = {
        key: selected[key]
        for key in ("alpha", "decision_threshold", "temperature", "corridor_radius", "correction_mode")
    }
    new_cases = 0
    for case_id in ids:
        provenance = {
            "version": 12,
            "task": args.task,
            "settings": settings,
            "teacher_sha256": teacher_store.case_record(case_id)["sha256"],
            "raw_sha256": raw_store.case_record(case_id)["sha256"],
        }
        if output.is_complete(case_id, provenance):
            continue
        probability = calibrated_residual(
            raw_store.read_case(case_id),
            teacher_store.read_case(case_id),
            **settings,
        )
        output.write_case(case_id, probability, provenance)
        new_cases += 1
    return {"task": args.task, "split": args.split, "cases": len(ids), "new_cases": new_cases, "output": str(args.output_store)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select or apply topology-safe V12 OOF calibration.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("--data-root", type=Path, required=True)
    search_parser.add_argument("--fold-manifest", type=Path, required=True)
    search_parser.add_argument("--teacher-store", type=Path, required=True)
    search_parser.add_argument("--raw-store", type=Path, required=True)
    search_parser.add_argument("--output-config", type=Path, required=True)
    search_parser.add_argument("--task", choices=("task1", "task2"), required=True)
    search_parser.add_argument("--correction-mode", choices=("prune", "vessel_support"), default="prune")
    search_parser.add_argument("--corridor-radius", type=int, default=2)
    search_parser.add_argument("--minimum-pixel-score-gain", type=float, default=0.10)
    search_parser.add_argument("--minimum-fold-pixel-score-gain", type=float, default=0.0)
    search_parser.add_argument("--maximum-sensitivity-drop", type=float, default=0.025)
    search_parser.add_argument("--maximum-fold-sensitivity-drop", type=float, default=0.05)
    search_parser.add_argument("--minimum-teacher-path-recall", type=float, default=1.0)
    apply_parser = subparsers.add_parser("apply")
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
    report = search(args) if args.command == "search" else apply(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
