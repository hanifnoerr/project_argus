from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import GAVE2Dataset, torch_collate
from .losses import probabilities_from_logits
from .models import create_model
from .submission import save_probability_png


def _import_torch():
    try:
        import torch
        from torch.utils.data import DataLoader

        return torch, DataLoader
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("OOF prediction requires PyTorch.") from exc


def predict_with_tta(model, image, mask, tta: str):
    torch, _ = _import_torch()
    transforms = [(False, False)]
    if tta == "flips":
        transforms += [(True, False), (False, True), (True, True)]
    probs = []
    with torch.no_grad():
        for flip_h, flip_w in transforms:
            x = image
            if flip_h:
                x = torch.flip(x, dims=[-2])
            if flip_w:
                x = torch.flip(x, dims=[-1])
            prob = probabilities_from_logits(model(x)).float()
            if flip_w:
                prob = torch.flip(prob, dims=[-1])
            if flip_h:
                prob = torch.flip(prob, dims=[-2])
            probs.append(prob)
    return torch.stack(probs, dim=0).mean(dim=0) * mask


def load_model(path: Path, device: str):
    torch, _ = _import_torch()
    checkpoint = torch.load(path, map_location=device)
    model = create_model(
        checkpoint["branch"],
        checkpoint["task"],
        base_channels=int(checkpoint["base_channels"]),
        num_iterations=int(checkpoint.get("num_iterations", 2)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def run(args) -> None:
    torch, DataLoader = _import_torch()
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    fold_dirs = sorted((args.run_dir / args.branch / args.task).glob("fold_*"))
    if not fold_dirs:
        raise FileNotFoundError(args.run_dir / args.branch / args.task)
    task_dir = args.output_root / args.branch / args.team_id / ("Task1" if args.task == "task1" else "Task2")
    task_dir.mkdir(parents=True, exist_ok=True)

    for fold_dir in fold_dirs:
        checkpoint_path = fold_dir / "best.pt"
        if not checkpoint_path.exists():
            checkpoint_path = fold_dir / "last.pt"
        config_path = fold_dir / "config.json"
        if not checkpoint_path.exists() or not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        val_ids = config["val_ids"]
        preprocess = config.get("preprocess", args.preprocess if args.preprocess != "auto" else "none")
        model = load_model(checkpoint_path, device)
        dataset = GAVE2Dataset(
            args.data_root,
            split="training",
            task=args.task,
            case_ids=val_ids,
            require_target=False,
            preprocess=preprocess,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=torch_collate)
        for batch in loader:
            image = batch["images"].to(device)
            mask = batch["masks"].to(device)
            prob = predict_with_tta(model, image, mask, args.tta)[0].clamp(0.0, 1.0)
            case_id = batch["case_ids"][0]
            save_probability_png(prob, task_dir / f"{case_id}.png")
            print(f"saved OOF {task_dir / f'{case_id}.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate out-of-fold training predictions for ensemble tuning.")
    parser.add_argument("--data-root", type=Path, default=Path("GAVE2_preliminary"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs") / "gave2_ensemble")
    parser.add_argument("--output-root", type=Path, default=Path("submissions_oof"))
    parser.add_argument("--team-id", type=str, default="team_id")
    parser.add_argument("--branch", choices=("cmrrwnet", "sam3", "yolo_native"), required=True)
    parser.add_argument("--task", choices=("task1", "task2"), default="task2")
    parser.add_argument("--tta", choices=("none", "flips"), default="flips")
    parser.add_argument("--preprocess", choices=("auto", "none", "clahe", "gray_clahe", "vessel_enhance"), default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
