from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import GAVE2Dataset, list_case_ids, torch_collate
from .losses import probabilities_from_logits
from .models import create_model
from .submission import save_probability_png


def _import_torch():
    try:
        import torch
        from torch.utils.data import DataLoader

        return torch, DataLoader
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Prediction requires PyTorch. Use the gave2-main or gave2-sam3 environment.") from exc


def discover_checkpoints(run_dir: Path, branch: str, task: str) -> list[Path]:
    base = run_dir / branch / task
    checkpoints = sorted(base.glob("fold_*/best.pt"))
    if not checkpoints:
        checkpoints = sorted(base.glob("fold_*/last.pt"))
    return checkpoints


def load_model_from_checkpoint(path: Path, device: str):
    torch, _ = _import_torch()
    checkpoint = torch.load(path, map_location=device)
    model = create_model(
        branch=checkpoint["branch"],
        task=checkpoint["task"],
        base_channels=int(checkpoint["base_channels"]),
        num_iterations=int(checkpoint.get("num_iterations", 2)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def resolve_preprocess_mode(requested: str, checkpoints: list[Path]) -> str:
    if requested != "auto":
        return requested
    for checkpoint_path in checkpoints:
        config_path = Path(checkpoint_path).with_name("config.json")
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            return config.get("preprocess", "none")
    return "none"


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
    out = torch.stack(probs, dim=0).mean(dim=0)
    return (out * mask).clamp(0.0, 1.0)


def run_prediction(args) -> None:
    torch, DataLoader = _import_torch()
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoints = args.checkpoint or discover_checkpoints(args.run_dir, args.branch, args.task)
    if not checkpoints and not args.allow_untrained:
        raise FileNotFoundError(f"No checkpoints found for {args.branch}/{args.task} under {args.run_dir}")

    if checkpoints:
        models = [load_model_from_checkpoint(Path(path), device) for path in checkpoints]
    else:
        models = [
            create_model(
                branch=args.branch,
                task=args.task,
                base_channels=args.base_channels,
                num_iterations=args.num_iterations,
            ).to(device).eval()
        ]
    preprocess = resolve_preprocess_mode(args.preprocess, [Path(path) for path in checkpoints])

    case_ids = list_case_ids(args.data_root, split=args.split)
    if args.limit_cases is not None:
        case_ids = case_ids[: args.limit_cases]
    dataset = GAVE2Dataset(
        args.data_root,
        split=args.split,
        task=args.task,
        case_ids=case_ids,
        require_target=False,
        preprocess=preprocess,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=torch_collate)
    task_dir = args.output_root / args.branch / args.team_id / ("Task1" if args.task == "task1" else "Task2")
    task_dir.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        image = batch["images"].to(device)
        mask = batch["masks"].to(device)
        branch_probs = [predict_with_tta(model, image, mask, args.tta) for model in models]
        prob = torch.stack(branch_probs, dim=0).mean(dim=0)[0]
        case_id = batch["case_ids"][0]
        expected_h, expected_w = batch["sizes"][0]
        if tuple(prob.shape[-2:]) != (expected_h, expected_w):
            raise RuntimeError(f"{case_id}: prediction shape {tuple(prob.shape[-2:])} != {(expected_h, expected_w)}")
        save_probability_png(prob, task_dir / f"{case_id}.png")
        print(f"saved {task_dir / f'{case_id}.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict full-resolution GAVE2 branch probability maps.")
    parser.add_argument("--data-root", type=Path, default=Path("GAVE2_preliminary"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs") / "gave2_ensemble")
    parser.add_argument("--output-root", type=Path, default=Path("submissions"))
    parser.add_argument("--team-id", type=str, default="team_id")
    parser.add_argument("--branch", choices=("cmrrwnet", "sam3", "yolo_native"), required=True)
    parser.add_argument("--task", choices=("task1", "task2"), default="task2")
    parser.add_argument("--split", choices=("training", "validation"), default="validation")
    parser.add_argument("--checkpoint", type=Path, action="append", default=None)
    parser.add_argument("--allow-untrained", action="store_true")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--tta", choices=("none", "flips"), default="flips")
    parser.add_argument("--preprocess", choices=("auto", "none", "clahe", "gray_clahe", "vessel_enhance"), default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit-cases", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    run_prediction(parse_args())


if __name__ == "__main__":
    main()
