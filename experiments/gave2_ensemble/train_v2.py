from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .data import GAVE2Dataset, compute_roi_channel_stats, list_case_ids, torch_collate
from .integrity_v2 import (
    fold_manifest_sha256,
    hash_file,
    hash_path_metadata,
    load_fold_manifest,
    validate_fold_manifest,
    write_run_manifest,
)
from .training_utils_v2 import average_precision_from_histograms, positive_weights_from_counts


def wall_time_exhausted(
    started_at: float,
    max_wall_minutes: float | None,
    now: float | None = None,
) -> bool:
    if max_wall_minutes is None:
        return False
    if max_wall_minutes <= 0:
        raise ValueError("max_wall_minutes must be positive")
    current = time.monotonic() if now is None else float(now)
    return current - float(started_at) >= float(max_wall_minutes) * 60.0


def _import_torch():
    try:
        import torch
        from torch.utils.data import DataLoader

        return torch, DataLoader
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("CMRRWNet v2 training requires PyTorch") from exc


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch, _ = _import_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(torch, device: str, amp: str):
    if device.startswith("cuda") and amp == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def augment_full_canvas(batch: dict, task: str) -> dict:
    torch, _ = _import_torch()
    images = batch["images"]
    masks = batch["masks"]
    targets = batch["targets"]
    if random.random() < 0.5:
        images = torch.flip(images, dims=(-1,))
        masks = torch.flip(masks, dims=(-1,))
        targets = torch.flip(targets, dims=(-1,))
    if random.random() < 0.5:
        images = torch.flip(images, dims=(-2,))
        masks = torch.flip(masks, dims=(-2,))
        targets = torch.flip(targets, dims=(-2,))

    rgb = images[:, :3]
    if random.random() < 0.8:
        gain = random.uniform(0.90, 1.10)
        bias = random.uniform(-0.03, 0.03)
        gamma = random.uniform(0.90, 1.10)
        rgb = (rgb * gain + bias).clamp(0.0, 1.0).pow(gamma)
    if random.random() < 0.25:
        rgb = (rgb + torch.randn_like(rgb) * random.uniform(0.0, 0.01)).clamp(0.0, 1.0)
    images = torch.cat((rgb, images[:, 3:]), dim=1) if images.shape[1] > 3 else rgb
    if task == "task2" and random.random() < 0.10:
        channel = 3 if random.random() < 0.5 else 4
        images[:, channel : channel + 1] = 0.0

    batch["images"] = images
    batch["masks"] = masks
    batch["targets"] = targets
    return batch


def normalize_batch(images, masks, normalization: dict[str, list[float]]):
    torch, _ = _import_torch()
    mean = torch.as_tensor(normalization["mean"], dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    std = torch.as_tensor(normalization["std"], dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    return ((images - mean) / std.clamp_min(1e-6)) * masks


def compute_fold_target_statistics(dataset: GAVE2Dataset) -> tuple[np.ndarray, np.ndarray]:
    positive = np.zeros(3, dtype=np.float64)
    total = np.zeros(3, dtype=np.float64)
    for sample in dataset:
        roi = sample.mask[0] > 0.5
        positive += sample.target[:, roi].sum(axis=1)
        total += float(roi.sum())
    return positive_weights_from_counts(positive, total), (positive / total).astype(np.float32)


def compute_fold_positive_weights(dataset: GAVE2Dataset) -> np.ndarray:
    weights, _ = compute_fold_target_statistics(dataset)
    return weights


def initialize_output_biases(model, target_priors: np.ndarray) -> None:
    torch, _ = _import_torch()
    priors = np.clip(np.asarray(target_priors, dtype=np.float32), 1e-4, 1.0 - 1e-4)
    challenge_logits = np.log(priors / (1.0 - priors))
    internal_logits = challenge_logits[[0, 2, 1]]
    with torch.no_grad():
        model.first_u.outconv.bias.copy_(
            torch.as_tensor(internal_logits, dtype=model.first_u.outconv.bias.dtype, device=model.first_u.outconv.bias.device)
        )
        model.refiner.outconv.bias.copy_(
            torch.as_tensor(internal_logits[:2], dtype=model.refiner.outconv.bias.dtype, device=model.refiner.outconv.bias.device)
        )


def _move_batch(batch: dict, device: str, normalization: dict[str, list[float]]) -> dict:
    batch["images"] = normalize_batch(batch["images"], batch["masks"], normalization)
    batch["images"] = batch["images"].to(device, non_blocking=True)
    batch["masks"] = batch["masks"].to(device, non_blocking=True)
    batch["targets"] = batch["targets"].to(device, non_blocking=True)
    return batch


def run_validation(model, loader, criterion, device: str, amp: str, normalization) -> dict[str, object]:
    from .losses import probabilities_from_logits

    torch, _ = _import_torch()
    model.eval()
    losses = []
    soft_intersection = np.zeros(3, dtype=np.float64)
    probability_sum = np.zeros(3, dtype=np.float64)
    target_sum = np.zeros(3, dtype=np.float64)
    hard_intersection = np.zeros(3, dtype=np.float64)
    hard_sum = np.zeros(3, dtype=np.float64)
    positive_histogram = np.zeros((3, 256), dtype=np.float64)
    total_histogram = np.zeros((3, 256), dtype=np.float64)
    roi_sum = 0.0
    with torch.inference_mode():
        for batch in loader:
            batch = _move_batch(batch, device, normalization)
            with autocast_context(torch, device, amp):
                predictions = model(batch["images"])
                loss = criterion(predictions, batch["targets"], batch["masks"])
            probability = probabilities_from_logits(predictions).float() * batch["masks"]
            target = batch["targets"].float() * batch["masks"]
            dims = (0, 2, 3)
            intersection = (probability * target).sum(dims)
            binary = (probability >= 0.5).float()
            losses.append(float(loss.detach().cpu()))
            soft_intersection += intersection.detach().cpu().numpy()
            probability_sum += probability.sum(dims).detach().cpu().numpy()
            target_sum += target.sum(dims).detach().cpu().numpy()
            hard_intersection += (binary * target).sum(dims).detach().cpu().numpy()
            hard_sum += binary.sum(dims).detach().cpu().numpy()
            roi_sum += float(batch["masks"].sum().detach().cpu())
            roi = batch["masks"][:, 0] > 0.5
            for channel in range(3):
                selected_probability = probability[:, channel][roi]
                selected_target = target[:, channel][roi]
                bins = torch.clamp(torch.round(selected_probability * 255.0), 0, 255).long()
                total_histogram[channel] += torch.bincount(bins, minlength=256).cpu().numpy()
                positive_histogram[channel] += torch.bincount(
                    bins, weights=selected_target, minlength=256
                ).cpu().numpy()
    if not losses:
        raise RuntimeError("Validation loader is empty")
    soft = (2.0 * soft_intersection + 1e-6) / (probability_sum + target_sum + 1e-6)
    hard = (2.0 * hard_intersection + 1e-6) / (hard_sum + target_sum + 1e-6)
    means = probability_sum / max(roi_sum, 1.0)
    foreground_baseline = (2.0 * target_sum + 1e-6) / (roi_sum + target_sum + 1e-6)
    ranking = average_precision_from_histograms(positive_histogram, total_histogram)
    selection_score = 0.30 * float(np.mean(soft)) + 0.70 * float(ranking["average_precision"])
    return {
        "loss": float(np.mean(losses)),
        "soft_dice": float(np.mean(soft)),
        "soft_dice_channels": [float(value) for value in soft],
        "dice_t05": float(np.mean(hard)),
        "dice_t05_channels": [float(value) for value in hard],
        "probability_means": [float(value) for value in means],
        "all_foreground_dice": float(np.mean(foreground_baseline)),
        "all_foreground_dice_channels": [float(value) for value in foreground_baseline],
        "selection_score": selection_score,
        **ranking,
    }


def _load_checkpoint(torch, path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch before weights_only
        return torch.load(path, map_location=device)


def train_fold(args: argparse.Namespace) -> Path:
    fold_started_at = time.monotonic()
    from .cmrrwnet_v2 import create_cmrrwnet_v2, default_official_source
    from .losses import BalancedRecursiveLoss
    from .training_control import EarlyStopping

    torch, DataLoader = _import_torch()
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    manifest = load_fold_manifest(args.fold_manifest)
    all_ids = list_case_ids(args.data_root, split="training")
    validate_fold_manifest(manifest, all_ids, n_folds=int(manifest["n_folds"]))
    if args.fold < 0 or args.fold >= int(manifest["n_folds"]):
        raise ValueError(f"Invalid fold {args.fold}")
    fold = manifest["folds"][args.fold]
    train_ids = fold["training"]
    val_ids = fold["validation"]

    set_seeds(args.seed + args.fold)
    train_dataset = GAVE2Dataset(
        args.data_root,
        split="training",
        task=args.task,
        case_ids=train_ids,
        preprocess=args.preprocess,
    )
    val_dataset = GAVE2Dataset(
        args.data_root,
        split="training",
        task=args.task,
        case_ids=val_ids,
        preprocess=args.preprocess,
    )
    normalization = compute_roi_channel_stats(args.data_root, args.task, train_ids, args.preprocess)
    positive_weights, target_priors = compute_fold_target_statistics(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.startswith("cuda"),
        collate_fn=torch_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=min(args.workers, 2),
        pin_memory=device.startswith("cuda"),
        collate_fn=torch_collate,
    )

    model = create_cmrrwnet_v2(
        task=args.task,
        base_channels=args.base_channels,
        num_refinements=args.num_refinements,
        activation_checkpointing=not args.no_activation_checkpointing,
    ).to(device)
    initialize_output_biases(model, target_priors)
    criterion = BalancedRecursiveLoss(
        positive_weights=positive_weights,
        topology_weight=args.topology_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=args.lr_patience,
        threshold=args.early_stopping_min_delta,
        min_lr=args.min_lr,
    )
    early_stopper = EarlyStopping(
        mode="max",
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
    )

    fold_sha = fold_manifest_sha256(manifest)
    code_root = Path(__file__).resolve().parent
    official_sha = hash_file(default_official_source())
    code_sha = hash_path_metadata(code_root)
    write_run_manifest(
        args.run_dir,
        {
            "version": 3,
            "seed": args.seed,
            "fold_manifest_sha256": fold_sha,
            "code_metadata_sha256": code_sha,
            "official_source_sha256": official_sha,
        },
    )

    fold_dir = args.run_dir / "cmrrwnet_v2" / args.task / f"fold_{args.fold}"
    config = {
        "version": 3,
        "task": args.task,
        "fold_index": args.fold,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "fold_manifest_sha256": fold_sha,
        "model_class": type(model).__name__,
        "base_channels": args.base_channels,
        "num_refinements": args.num_refinements,
        "activation_checkpointing": not args.no_activation_checkpointing,
        "preprocess": args.preprocess,
        "normalization": normalization,
        "positive_weights": positive_weights.tolist(),
        "target_priors": target_priors.tolist(),
        "loss": {
            "name": "balanced_bce_dice_focal_tversky",
            "bce_weight": 0.25,
            "dice_weight": 0.15,
            "tversky_weight": 0.60,
            "classification_weight": 0.15,
            "hierarchy_weight": 0.05,
        },
        "code_metadata_sha256": code_sha,
        "official_source_sha256": official_sha,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "amp": args.amp,
        "learning_rate": args.lr,
        "minimum_learning_rate": args.min_lr,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.grad_clip,
        "lr_patience": args.lr_patience,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "topology_weight": args.topology_weight,
    }
    config_path = fold_dir / "config.json"
    if fold_dir.exists() and any(fold_dir.iterdir()):
        if not args.resume:
            raise RuntimeError(f"Fold directory is not empty; pass --resume only for this exact run: {fold_dir}")
        if not config_path.is_file() or json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise RuntimeError(f"Existing fold config does not match requested run: {fold_dir}")
    else:
        fold_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    history_path = fold_dir / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    start_epoch = 1
    last_path = fold_dir / "last.pt"
    if args.resume and last_path.exists():
        state = _load_checkpoint(torch, last_path, "cpu")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        early_stopper.best = state.get("best_monitor")
        early_stopper.best_epoch = state.get("best_epoch")
        early_stopper.stale_epochs = int(state.get("stale_epochs", 0))
        if early_stopper.should_stop:
            print(f"Fold already early-stopped at epoch {state['epoch']}; keeping certified best.pt")
            return fold_dir
        del state
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        steps = 0
        for step, batch in enumerate(train_loader, start=1):
            batch = augment_full_canvas(batch, args.task)
            batch = _move_batch(batch, device, normalization)
            with autocast_context(torch, device, args.amp):
                predictions = model(batch["images"])
                loss = criterion(predictions, batch["targets"], batch["masks"])
                scaled_loss = loss / args.grad_accum
            scaled_loss.backward()
            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.detach().cpu())
            steps += 1

        validation = run_validation(model, val_loader, criterion, device, args.amp, normalization)
        monitor = float(validation["selection_score"])
        scheduler.step(monitor)
        is_best = early_stopper.update(monitor, epoch=epoch)
        if not math.isfinite(monitor):
            raise RuntimeError(f"Non-finite validation score at epoch {epoch}")
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(steps, 1),
            **validation,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "best_monitor": early_stopper.best,
            "best_epoch": early_stopper.best_epoch,
            "stale_epochs": early_stopper.stale_epochs,
        }
        history.append(row)
        print(json.dumps(row))
        checkpoint_state = {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_monitor": early_stopper.best,
            "best_epoch": early_stopper.best_epoch,
            "stale_epochs": early_stopper.stale_epochs,
            "validation": validation,
            "config": config,
        }
        torch.save(checkpoint_state, last_path)
        if is_best:
            torch.save(checkpoint_state, fold_dir / "best.pt")
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        if early_stopper.should_stop:
            (fold_dir / "early_stop.json").write_text(
                json.dumps(
                    {
                        "stopped_epoch": epoch,
                        "best_epoch": early_stopper.best_epoch,
                        "best_soft_dice": early_stopper.best,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            break
        if wall_time_exhausted(fold_started_at, args.max_wall_minutes):
            elapsed_minutes = (time.monotonic() - fold_started_at) / 60.0
            report = {
                "stopped_epoch": epoch,
                "best_epoch": early_stopper.best_epoch,
                "best_monitor": early_stopper.best,
                "elapsed_minutes": elapsed_minutes,
                "max_wall_minutes": args.max_wall_minutes,
            }
            (fold_dir / "time_stop.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Fold wall-time limit reached after checkpointing epoch {epoch}: {elapsed_minutes:.2f} minutes")
            break
    return fold_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train corrected full-resolution CMRRWNet v2.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--task", choices=("task1", "task2"), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--num-refinements", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--max-wall-minutes", type=float)
    parser.add_argument("--amp", choices=("none", "bf16"), default="bf16")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lr-patience", type=int, default=12)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--preprocess", choices=("none", "clahe", "gray_clahe", "vessel_enhance"), default="none")
    parser.add_argument("--topology-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    return parser.parse_args()


def main() -> None:
    train_fold(parse_args())


if __name__ == "__main__":
    main()
