from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .data import derive_av3_target, list_case_ids, read_png_float
from .predict_v2 import project_probabilities
from .submission import load_probability_png, save_probability_png


def fit_channel_threshold(probability: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    probability = np.asarray(probability, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=bool).ravel()
    if probability.shape != target.shape or probability.size == 0:
        raise ValueError("Probability and target must have the same nonempty shape")
    thresholds = np.linspace(0.10, 0.90, 161)
    best_threshold = 0.5
    best_dice = -1.0
    positives = float(target.sum())
    for threshold in thresholds:
        prediction = probability >= threshold
        true_positive = float(np.count_nonzero(prediction & target))
        dice = (2.0 * true_positive + 1e-8) / (float(prediction.sum()) + positives + 1e-8)
        if dice > best_dice:
            best_dice = dice
            best_threshold = float(threshold)
    return best_threshold, best_dice


def apply_threshold_calibration(probability_chw: np.ndarray, thresholds: list[float]) -> np.ndarray:
    probability = np.asarray(probability_chw, dtype=np.float32)
    if probability.ndim != 3 or probability.shape[0] != 3 or len(thresholds) != 3:
        raise ValueError("Expected 3xHxW probabilities and three thresholds")
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped) - np.log1p(-clipped)
    threshold = np.clip(np.asarray(thresholds, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    bias = np.log(threshold) - np.log1p(-threshold)
    calibrated = 1.0 / (1.0 + np.exp(-(logits - bias[:, None, None])))
    return calibrated.astype(np.float32)


def _dice_from_histogram(positive: np.ndarray, total: np.ndarray, threshold_bin: int) -> float:
    true_positive = float(positive[threshold_bin:].sum())
    predicted = float(total[threshold_bin:].sum())
    target = float(positive.sum())
    return (2.0 * true_positive + 1e-8) / (predicted + target + 1e-8)


def fit_oof_calibration(
    data_root: Path,
    probability_dir: Path,
    task: str,
    output_path: Path,
    calibrated_dir: Path | None = None,
) -> dict[str, object]:
    case_ids = list_case_ids(data_root, split="training")
    positive_hist = np.zeros((3, 256), dtype=np.float64)
    total_hist = np.zeros((3, 256), dtype=np.float64)
    for case_id in case_ids:
        probability = load_probability_png(probability_dir / f"{case_id}.png")
        target = derive_av3_target(read_png_float(data_root / "training" / "av" / f"{case_id}.png", channels=3))
        roi = read_png_float(data_root / "training" / "masks" / f"{case_id}.png", channels=1)[..., 0] > 0.5
        for channel in range(3):
            bins = np.rint(probability[channel][roi] * 255.0).astype(np.int16)
            total_hist[channel] += np.bincount(bins, minlength=256)
            positive_hist[channel] += np.bincount(bins, weights=target[channel][roi], minlength=256)

    thresholds = []
    before = []
    after = []
    for channel in range(3):
        candidates = range(int(round(0.15 * 255)), int(round(0.80 * 255)) + 1)
        scores = [_dice_from_histogram(positive_hist[channel], total_hist[channel], value) for value in candidates]
        best_index = int(np.argmax(scores))
        threshold_bin = list(candidates)[best_index]
        thresholds.append(float(threshold_bin / 255.0))
        before.append(_dice_from_histogram(positive_hist[channel], total_hist[channel], 128))
        after.append(float(scores[best_index]))
    report: dict[str, object] = {
        "version": 2,
        "task": task,
        "case_ids": case_ids,
        "thresholds": thresholds,
        "dice_t05_before_channels": before,
        "calibrated_dice_channels": after,
        "dice_t05_before_mean": float(np.mean(before)),
        "calibrated_dice_mean": float(np.mean(after)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if calibrated_dir is not None:
        if calibrated_dir.exists():
            shutil.rmtree(calibrated_dir)
        calibrated_dir.mkdir(parents=True)
        for case_id in case_ids:
            probability = load_probability_png(probability_dir / f"{case_id}.png")
            roi = read_png_float(data_root / "training" / "masks" / f"{case_id}.png", channels=1)[..., 0]
            calibrated = project_probabilities(apply_threshold_calibration(probability, thresholds), roi)
            save_probability_png(calibrated, calibrated_dir / f"{case_id}.png")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit channel thresholds from honest GAVE2 OOF predictions.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--probability-dir", type=Path, required=True)
    parser.add_argument("--task", choices=("task1", "task2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibrated-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = fit_oof_calibration(args.data_root, args.probability_dir, args.task, args.output, args.calibrated_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
