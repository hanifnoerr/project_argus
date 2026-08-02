from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import list_case_ids, read_png_float, to_chw
from .submission import load_probability_png, save_probability_png

DEFAULT_WEIGHTS = {"cmrrwnet": 0.45, "sam3": 0.35, "yolo_native": 0.20}


def weighted_average_probabilities(probabilities, weights=None, roi_mask=None):
    import torch

    weights = DEFAULT_WEIGHTS if weights is None else weights
    missing = sorted(set(weights) - set(probabilities))
    if missing:
        raise ValueError(f"Missing branch probabilities for {missing}")
    weighted = None
    weight_sum = None
    for name, weight in weights.items():
        tensor = probabilities[name]
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, dtype=torch.float32)
        tensor = tensor.float()
        if isinstance(weight, (list, tuple)):
            weight_tensor = torch.as_tensor(weight, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
        else:
            weight_tensor = torch.as_tensor(float(weight), dtype=tensor.dtype, device=tensor.device)
        weighted = tensor * weight_tensor if weighted is None else weighted + tensor * weight_tensor
        weight_sum = weight_tensor if weight_sum is None else weight_sum + weight_tensor
    out = weighted / torch.clamp(weight_sum, min=1e-8)
    if roi_mask is not None:
        if not isinstance(roi_mask, torch.Tensor):
            roi_mask = torch.as_tensor(roi_mask, dtype=torch.float32)
        out = out * roi_mask.float()
    return out.clamp(0.0, 1.0)


def ensemble_case(
    branch_task_dirs: dict[str, Path],
    case_id: str,
    mask_path: Path | None,
    weights: dict[str, float] | None = None,
):
    import torch

    probs = {
        branch: torch.from_numpy(load_probability_png(task_dir / f"{case_id}.png"))
        for branch, task_dir in branch_task_dirs.items()
    }
    roi = None
    if mask_path is not None and mask_path.exists():
        roi = torch.from_numpy(to_chw(read_png_float(mask_path, channels=1)))
    return weighted_average_probabilities(probs, weights=weights, roi_mask=roi)


def ensemble_task_outputs(
    branch_submission_roots: dict[str, Path],
    output_task_dir: Path,
    data_root: Path,
    task_name: str,
    split: str = "validation",
    weights: dict[str, float] | None = None,
) -> None:
    case_ids = list_case_ids(data_root, split=split)
    branch_task_dirs = {
        branch: root / task_name for branch, root in branch_submission_roots.items()
    }
    output_task_dir.mkdir(parents=True, exist_ok=True)
    for case_id in case_ids:
        mask_path = data_root / split / "masks" / f"{case_id}.png"
        prob = ensemble_case(branch_task_dirs, case_id, mask_path, weights=weights)
        save_probability_png(prob, output_task_dir / f"{case_id}.png")


def parse_weights(items: list[str] | None) -> dict[str, float]:
    if not items:
        return dict(DEFAULT_WEIGHTS)
    weights = {}
    for item in items:
        name, value = item.split("=", 1)
        if "," in value:
            weights[name] = [float(part) for part in value.split(",")]
        else:
            weights[name] = float(value)
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble GAVE2 branch probability PNGs.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--submission-root", type=Path, default=Path("submissions"))
    parser.add_argument("--team-id", type=str, default="team_id")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--weights", nargs="*", default=None, help="Branch weights, e.g. cmrrwnet=0.45")
    parser.add_argument("--weights-json", type=Path, default=None)
    args = parser.parse_args()

    weights = json.loads(args.weights_json.read_text(encoding="utf-8")) if args.weights_json else parse_weights(args.weights)
    branch_roots = {
        branch: args.submission_root / branch / args.team_id
        for branch in ("cmrrwnet", "sam3", "yolo_native")
    }
    ensemble_root = args.submission_root / "ensemble" / args.team_id
    for task in ("Task1", "Task2"):
        ensemble_task_outputs(
            branch_roots,
            ensemble_root / task,
            args.data_root,
            task_name=task,
            split=args.split,
            weights=weights,
        )


if __name__ == "__main__":
    main()
