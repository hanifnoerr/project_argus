from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .probability_calibration_v2 import apply_threshold_calibration, fit_channel_threshold
from .predict_v6 import FloatProbabilityStore, oof_case_ownership


def _case_ids_for_fold(ownership: dict[str, int], fold_index: int) -> list[str]:
    return sorted(case_id for case_id, fold in ownership.items() if int(fold) == int(fold_index))


def _fit_thresholds(
    case_ids: list[str],
    probabilities: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    roi: dict[str, np.ndarray],
) -> tuple[list[float], list[float]]:
    thresholds: list[float] = []
    dice_scores: list[float] = []
    for channel in range(3):
        channel_probability = []
        channel_target = []
        for case_id in case_ids:
            case_probability = np.asarray(probabilities[case_id], dtype=np.float32)[channel]
            case_target = np.asarray(targets[case_id], dtype=np.float32)[channel] > 0.5
            case_roi = np.asarray(roi[case_id], dtype=np.float32) > 0.5
            channel_probability.append(case_probability[case_roi])
            channel_target.append(case_target[case_roi])
        probability_vector = np.concatenate(channel_probability, axis=0)
        target_vector = np.concatenate(channel_target, axis=0)
        threshold, dice = fit_channel_threshold(probability_vector, target_vector)
        thresholds.append(float(threshold))
        dice_scores.append(float(dice))
    return thresholds, dice_scores


def cross_fit_calibrators(
    ownership: dict[str, int],
    probabilities: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    roi: dict[str, np.ndarray],
    *,
    task: str,
) -> dict[str, object]:
    fold_ids = sorted(set(int(value) for value in ownership.values()))
    if fold_ids != [0, 1, 2]:
        raise ValueError("Cross-fitted V6 calibration requires exactly three folds")
    case_ids = sorted(ownership)
    report: dict[str, object] = {
        "version": 6,
        "task": str(task),
        "fit_phase": "cross_fit",
        "oof_locked": False,
        "case_ids": case_ids,
        "fold_calibrators": [],
        "case_calibrations": {},
    }
    for fold_index in fold_ids:
        apply_case_ids = _case_ids_for_fold(ownership, fold_index)
        fitted_on_case_ids = sorted(case_id for case_id in case_ids if case_id not in apply_case_ids)
        thresholds, dice_scores = _fit_thresholds(fitted_on_case_ids, probabilities, targets, roi)
        calibrator = {
            "fold_index": fold_index,
            "apply_case_ids": apply_case_ids,
            "fitted_on_case_ids": fitted_on_case_ids,
            "thresholds": thresholds,
            "dice_scores": dice_scores,
        }
        report["fold_calibrators"].append(calibrator)
        for case_id in apply_case_ids:
            report["case_calibrations"][case_id] = {
                "fold_index": fold_index,
                "thresholds": thresholds,
                "fitted_on_case_ids": fitted_on_case_ids,
            }
    return report


def fit_final_deployment_calibrator(
    probabilities: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    roi: dict[str, np.ndarray],
    *,
    task: str,
    oof_report: dict[str, object],
) -> dict[str, object]:
    if not bool(oof_report.get("oof_locked")):
        raise RuntimeError("Final deployment calibration requires locked OOF evaluation")
    case_ids = sorted(probabilities)
    thresholds, dice_scores = _fit_thresholds(case_ids, probabilities, targets, roi)
    return {
        "version": 6,
        "task": str(task),
        "fit_phase": "deployment",
        "oof_locked": True,
        "fitted_on_case_ids": case_ids,
        "thresholds": thresholds,
        "dice_scores": dice_scores,
    }


def apply_calibration(probability: np.ndarray, calibration: dict[str, object], *, case_id: str | None = None) -> np.ndarray:
    if calibration.get("fit_phase") == "cross_fit":
        if case_id is None:
            raise ValueError("Cross-fitted calibration requires a case_id")
        thresholds = calibration["case_calibrations"][case_id]["thresholds"]
    else:
        thresholds = calibration["thresholds"]
    return apply_threshold_calibration(probability, thresholds)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V6 cross-fitted and deployment calibration helper.")
    parser.add_argument("--task", choices=("task1", "task2"), required=True)
    parser.add_argument("--mode", choices=("cross-fit", "deployment"), required=True)
    parser.add_argument("--fold-manifest", type=Path)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oof-report", type=Path)
    return parser.parse_args(argv)


def run_calibration(args: argparse.Namespace) -> dict[str, object]:
    from .data import derive_av3_target, read_png_float

    store = FloatProbabilityStore(args.prediction_root, task=args.task, split="oof")

    if args.mode == "cross-fit":
        if args.fold_manifest is None:
            raise ValueError("Cross-fit calibration requires --fold-manifest")
        manifest = json.loads(Path(args.fold_manifest).read_text(encoding="utf-8"))
        ownership = oof_case_ownership(manifest)
        probabilities = {case_id: store.read_case(case_id) for case_id in sorted(ownership)}
        targets = {
            case_id: derive_av3_target(read_png_float(Path(args.data_root) / "training" / "av" / f"{case_id}.png", channels=3))
            for case_id in sorted(ownership)
        }
        roi = {
            case_id: read_png_float(Path(args.data_root) / "training" / "masks" / f"{case_id}.png", channels=1)[..., 0]
            for case_id in sorted(ownership)
        }
        report = cross_fit_calibrators(ownership, probabilities, targets, roi, task=args.task)
    else:
        if args.oof_report is None:
            raise ValueError("Deployment calibration requires --oof-report")
        oof_report = json.loads(Path(args.oof_report).read_text(encoding="utf-8"))
        case_ids = sorted(path.stem for path in (Path(args.data_root) / "training" / "images").glob("*.png"))
        probabilities = {case_id: store.read_case(case_id) for case_id in case_ids}
        targets = {
            case_id: derive_av3_target(read_png_float(Path(args.data_root) / "training" / "av" / f"{case_id}.png", channels=3))
            for case_id in case_ids
        }
        roi = {
            case_id: read_png_float(Path(args.data_root) / "training" / "masks" / f"{case_id}.png", channels=1)[..., 0]
            for case_id in case_ids
        }
        report = fit_final_deployment_calibrator(probabilities, targets, roi, task=args.task, oof_report=oof_report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    print(json.dumps(run_calibration(parse_args()), indent=2))


if __name__ == "__main__":
    main()
