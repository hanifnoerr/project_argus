from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from .data import list_case_ids, read_png_float
from .submission import load_probability_png

BIOMARKER_KEYS = (
    "CRAE",
    "CRVE",
    "AVR",
    "artery_density",
    "vein_density",
    "artery_fractal_dimension",
    "vein_fractal_dimension",
)


def box_count_fractal_dimension(binary: np.ndarray) -> float:
    binary = np.asarray(binary, dtype=bool)
    if not binary.any():
        return 0.0
    h, w = binary.shape
    max_power = int(math.floor(math.log2(max(2, min(h, w) // 4))))
    sizes = [2**p for p in range(1, max_power + 1)]
    xs = []
    ys = []
    for size in sizes:
        h_pad = int(math.ceil(h / size) * size)
        w_pad = int(math.ceil(w / size) * size)
        padded = np.zeros((h_pad, w_pad), dtype=bool)
        padded[:h, :w] = binary
        blocks = padded.reshape(h_pad // size, size, w_pad // size, size)
        count = np.count_nonzero(blocks.any(axis=(1, 3)))
        if count > 0:
            xs.append(math.log(1.0 / size))
            ys.append(math.log(float(count)))
    if len(xs) < 2:
        return 0.0
    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope)


def estimate_biomarkers_from_probabilities(
    probability_chw: np.ndarray,
    roi_mask_hw: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    if probability_chw.shape[0] != 3:
        raise ValueError(f"Expected 3 probability channels, got {probability_chw.shape}")
    roi = roi_mask_hw > 0.5
    denom = max(int(roi.sum()), 1)
    artery = (probability_chw[0] >= threshold) & roi
    vein = (probability_chw[2] >= threshold) & roi
    artery_density = float(artery.sum() / denom)
    vein_density = float(vein.sum() / denom)

    # Empirical scale anchors from the preliminary training EDA. These keep
    # values in the expected pixel-unit range until a disc-aware caliber module
    # or validation calibration is plugged in.
    crae = 64.8 * math.sqrt(max(artery_density, 0.0))
    crve = 94.9 * math.sqrt(max(vein_density, 0.0))
    avr = crae / crve if crve > 1e-8 else 0.0

    return {
        "CRAE": float(crae),
        "CRVE": float(crve),
        "AVR": float(avr),
        "artery_density": artery_density,
        "vein_density": vein_density,
        "artery_fractal_dimension": box_count_fractal_dimension(artery),
        "vein_fractal_dimension": box_count_fractal_dimension(vein),
    }


def write_biomarker_txt(values: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key} {values.get(key, 0.0):.6f}" for key in BIOMARKER_KEYS]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_task3_for_branch(
    data_root: Path,
    branch_submission_root: Path,
    split: str = "validation",
    threshold: float = 0.5,
) -> None:
    task2_dir = branch_submission_root / "Task2"
    task3_dir = branch_submission_root / "Task3"
    case_ids = list_case_ids(data_root, split=split)
    for case_id in case_ids:
        prob_path = task2_dir / f"{case_id}.png"
        if not prob_path.exists():
            raise FileNotFoundError(prob_path)
        prob = load_probability_png(prob_path)
        roi = read_png_float(data_root / split / "masks" / f"{case_id}.png", channels=1)[..., 0]
        values = estimate_biomarkers_from_probabilities(prob, roi, threshold=threshold)
        write_biomarker_txt(values, task3_dir / f"{case_id}.txt")
        print(f"saved {task3_dir / f'{case_id}.txt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GAVE2 Task3 TXT files from Task2 predictions.")
    parser.add_argument("--data-root", type=Path, default=Path("GAVE2_preliminary"))
    parser.add_argument("--submission-root", type=Path, default=Path("submissions"))
    parser.add_argument("--team-id", type=str, default="team_id")
    parser.add_argument("--branch", choices=("cmrrwnet", "sam3", "yolo_native", "ensemble"), required=True)
    parser.add_argument("--split", choices=("training", "validation"), default="validation")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_task3_for_branch(
        data_root=args.data_root,
        branch_submission_root=args.submission_root / args.branch / args.team_id,
        split=args.split,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
