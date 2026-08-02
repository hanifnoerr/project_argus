from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .data import GAVE2Dataset, list_case_ids, make_folds, torch_collate


def _import_torch():
    try:
        import torch
        from torch.utils.data import DataLoader

        return torch, DataLoader
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Training requires PyTorch. Use the gave2-main or gave2-sam3 environment.") from exc


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch, _ = _import_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_fullres_augment(batch: dict, enabled: bool = True) -> dict:
    if not enabled:
        return batch
    torch, _ = _import_torch()
    images = batch["images"]
    masks = batch["masks"]
    targets = batch["targets"]
    if random.random() < 0.5:
        images = torch.flip(images, dims=[-1])
        masks = torch.flip(masks, dims=[-1])
        targets = torch.flip(targets, dims=[-1])
    if random.random() < 0.5:
        images = torch.flip(images, dims=[-2])
        masks = torch.flip(masks, dims=[-2])
        targets = torch.flip(targets, dims=[-2])
    if random.random() < 0.8:
        gain = torch.empty(images.shape[0], images.shape[1], 1, 1).uniform_(0.90, 1.10)
        bias = torch.empty(images.shape[0], images.shape[1], 1, 1).uniform_(-0.03, 0.03)
        images = (images * gain + bias).clamp(0.0, 1.0)
    batch["images"] = images
    batch["masks"] = masks
    batch["targets"] = targets
    return batch


def autocast_context(torch, device: str, amp: str):
    if device.startswith("cuda") and amp == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.startswith("cuda") and amp == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def move_batch(batch: dict, device: str) -> dict:
    batch["images"] = batch["images"].to(device, non_blocking=True)
    batch["masks"] = batch["masks"].to(device, non_blocking=True)
    if batch["targets"] is not None:
        batch["targets"] = batch["targets"].to(device, non_blocking=True)
    return batch


def run_validation(model, loader, criterion, device: str, amp: str) -> dict[str, float]:
    from .losses import probabilities_from_logits
    from .metrics import best_threshold_dice, mean_dice, roi_channel_means, soft_dice

    torch, _ = _import_torch()
    model.eval()
    total_loss = 0.0
    total_dice_t05 = 0.0
    total_soft_dice = 0.0
    total_best_dice = 0.0
    total_best_threshold = 0.0
    total_prob_means = [0.0, 0.0, 0.0]
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            with autocast_context(torch, device, amp):
                predictions = model(batch["images"])
                loss = criterion(predictions, batch["targets"], batch["masks"])
            probs = probabilities_from_logits(predictions).float()
            dice_t05 = mean_dice(probs, batch["targets"], batch["masks"], threshold=0.5)
            soft = soft_dice(probs, batch["targets"], batch["masks"])
            best_dice, best_threshold = best_threshold_dice(probs, batch["targets"], batch["masks"])
            prob_means = roi_channel_means(probs, batch["masks"])
            total_loss += float(loss.detach().cpu())
            total_dice_t05 += dice_t05
            total_soft_dice += soft
            total_best_dice += best_dice
            total_best_threshold += best_threshold
            for i, value in enumerate(prob_means):
                total_prob_means[i] += value
            n += 1
    denom = max(n, 1)
    return {
        "loss": total_loss / denom,
        "dice": total_dice_t05 / denom,
        "dice_t05": total_dice_t05 / denom,
        "soft_dice": total_soft_dice / denom,
        "best_dice": total_best_dice / denom,
        "best_threshold": total_best_threshold / denom,
        "prob_mean_artery": total_prob_means[0] / denom,
        "prob_mean_vessel": total_prob_means[1] / denom,
        "prob_mean_vein": total_prob_means[2] / denom,
    }


def train_fold(args, fold_index: int, train_ids: list[str], val_ids: list[str]) -> Path:
    from .losses import GAVE2SegmentationLoss, OfficialRRLoss
    from .models import create_model
    from .training_control import EarlyStopping

    torch, DataLoader = _import_torch()
    set_seeds(args.seed + fold_index)
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    train_ds = GAVE2Dataset(
        args.data_root,
        split="training",
        task=args.task,
        case_ids=train_ids,
        preprocess=args.preprocess,
    )
    val_ds = GAVE2Dataset(
        args.data_root,
        split="training",
        task=args.task,
        case_ids=val_ids,
        preprocess=args.preprocess,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.startswith("cuda"),
        collate_fn=torch_collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, min(args.workers, 2)),
        pin_memory=device.startswith("cuda"),
        collate_fn=torch_collate,
        drop_last=False,
    )

    model = create_model(
        branch=args.branch,
        task=args.task,
        base_channels=args.base_channels,
        num_iterations=args.num_iterations,
        use_official_cmrrwnet=not args.no_official_cmrrwnet,
    ).to(device)
    if args.loss_mode == "official_bce3":
        criterion = OfficialRRLoss()
    else:
        criterion = GAVE2SegmentationLoss(
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            consistency_weight=args.consistency_weight,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda") and args.amp == "fp16")
    early_stopping_mode = "min" if args.early_stopping_metric == "loss" else "max"
    early_stopper = EarlyStopping(
        mode=early_stopping_mode,
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
    )

    fold_dir = args.out_dir / args.branch / args.task / f"fold_{fold_index}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["data_root"] = str(args.data_root)
    config["out_dir"] = str(args.out_dir)
    config["fold_index"] = fold_index
    config["train_ids"] = train_ids
    config["val_ids"] = val_ids
    (fold_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        steps = 0
        for step, batch in enumerate(train_loader, start=1):
            batch = safe_fullres_augment(batch, enabled=not args.no_augment)
            batch = move_batch(batch, device)
            with autocast_context(torch, device, args.amp):
                predictions = model(batch["images"])
                loss = criterion(predictions, batch["targets"], batch["masks"])
                loss = loss / args.grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if step % args.grad_accum == 0 or step == len(train_loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach().cpu()) * args.grad_accum
            steps += 1

        val = run_validation(model, val_loader, criterion, device, args.amp)
        monitor_value = val[args.early_stopping_metric]
        is_best = early_stopper.update(monitor_value, epoch=epoch)
        row = {
            "epoch": epoch,
            "train_loss": running / max(steps, 1),
            **val,
            "monitor": monitor_value,
            "best_monitor": early_stopper.best,
            "best_epoch": early_stopper.best_epoch,
            "stale_epochs": early_stopper.stale_epochs,
        }
        history.append(row)
        print(json.dumps(row))

        checkpoint = {
            "state_dict": model.state_dict(),
            "branch": args.branch,
            "task": args.task,
            "base_channels": args.base_channels,
            "num_iterations": args.num_iterations,
            "epoch": epoch,
            "val": val,
            "monitor_metric": args.early_stopping_metric,
            "monitor": monitor_value,
            "preprocess": args.preprocess,
            "loss_mode": args.loss_mode,
        }
        torch.save(checkpoint, fold_dir / "last.pt")
        if is_best:
            torch.save(checkpoint, fold_dir / "best.pt")
        (fold_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if early_stopper.should_stop:
            stop_report = {
                "stopped_epoch": epoch,
                "best_epoch": early_stopper.best_epoch,
                "best_monitor": early_stopper.best,
                "monitor_metric": args.early_stopping_metric,
                "patience": args.early_stopping_patience,
                "min_delta": args.early_stopping_min_delta,
            }
            (fold_dir / "early_stop.json").write_text(json.dumps(stop_report, indent=2), encoding="utf-8")
            break

    return fold_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train full-resolution GAVE2 branch models.")
    parser.add_argument("--data-root", type=Path, default=Path("GAVE2_preliminary"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs") / "gave2_ensemble")
    parser.add_argument("--branch", choices=("cmrrwnet", "sam3", "yolo_native"), required=True)
    parser.add_argument("--task", choices=("task1", "task2"), default="task2")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold", type=str, default="all", help="'all' or a zero-based fold index")
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--consistency-weight", type=float, default=0.15)
    parser.add_argument("--preprocess", choices=("none", "clahe", "gray_clahe", "vessel_enhance"), default="none")
    parser.add_argument("--loss-mode", choices=("official_bce3", "hybrid"), default="official_bce3")
    parser.add_argument("--early-stopping-patience", type=int, default=25)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--early-stopping-metric",
        choices=("best_dice", "soft_dice", "dice", "loss"),
        default="best_dice",
    )
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-official-cmrrwnet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_ids = list_case_ids(args.data_root, split="training")
    if args.limit_cases is not None:
        case_ids = case_ids[: args.limit_cases]
    if len(case_ids) < 2:
        raise ValueError("Need at least two training cases for fold validation")
    folds = make_folds(case_ids, n_folds=min(args.folds, len(case_ids)), seed=args.seed)
    selected = range(len(folds)) if args.fold == "all" else [int(args.fold)]
    for fold_idx in selected:
        fold = folds[fold_idx]
        train_fold(args, fold_idx, fold["training"], fold["validation"])


if __name__ == "__main__":
    main()
