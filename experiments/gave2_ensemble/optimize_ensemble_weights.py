from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .data import derive_av3_target, list_case_ids, read_png_float
from .submission import load_probability_png


def dice_score(prob: np.ndarray, target: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> float:
    pred = (prob >= threshold) & (mask > 0.5)
    gt = (target >= 0.5) & (mask > 0.5)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return float((2 * inter + 1e-6) / (denom + 1e-6))


def weight_grid(step: float):
    values = np.arange(0.0, 1.0 + step / 2, step)
    for a, b in itertools.product(values, repeat=2):
        c = 1.0 - a - b
        if c < -1e-8:
            continue
        if c < 0:
            c = 0.0
        yield {"cmrrwnet": float(a), "sam3": float(b), "yolo_native": float(c)}


def optimize_channel_weights(
    data_root: Path,
    submission_root: Path,
    team_id: str,
    task_name: str,
    split: str,
    step: float,
    threshold: float,
) -> dict[str, list[float]]:
    case_ids = list_case_ids(data_root, split=split)
    best_by_channel = []
    for channel in range(3):
        best_score = -1.0
        best_weights = None
        for weights in weight_grid(step):
            scores = []
            for case_id in case_ids:
                target = derive_av3_target(read_png_float(data_root / split / "av" / f"{case_id}.png", channels=3))
                mask = read_png_float(data_root / split / "masks" / f"{case_id}.png", channels=1)[..., 0]
                combined = 0.0
                for branch, weight in weights.items():
                    pred = load_probability_png(submission_root / branch / team_id / task_name / f"{case_id}.png")
                    combined = combined + weight * pred[channel]
                scores.append(dice_score(combined, target[channel], mask, threshold=threshold))
            score = float(np.mean(scores))
            if score > best_score:
                best_score = score
                best_weights = weights
        best_by_channel.append(best_weights)

    output = {branch: [] for branch in ("cmrrwnet", "sam3", "yolo_native")}
    for weights in best_by_channel:
        for branch, value in weights.items():
            output[branch].append(float(value))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search per-channel GAVE2 ensemble weights.")
    parser.add_argument("--data-root", type=Path, default=Path("GAVE2_preliminary"))
    parser.add_argument("--submission-root", type=Path, default=Path("submissions"))
    parser.add_argument("--team-id", type=str, default="team_id")
    parser.add_argument("--task-name", choices=("Task1", "Task2"), default="Task2")
    parser.add_argument("--split", choices=("training",), default="training")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=Path("runs") / "gave2_ensemble" / "ensemble_weights.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = optimize_channel_weights(
        data_root=args.data_root,
        submission_root=args.submission_root,
        team_id=args.team_id,
        task_name=args.task_name,
        split=args.split,
        step=args.step,
        threshold=args.threshold,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(weights, indent=2), encoding="utf-8")
    print(json.dumps(weights, indent=2))


if __name__ == "__main__":
    main()
