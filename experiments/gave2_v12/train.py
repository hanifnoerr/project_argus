from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.gave2_ensemble.data import derive_av3_target

from .data import V12Dataset, collate
from .folds import validate_manifest
from .losses import ResidualChallengeLoss
from .metrics import evaluate_cases, pixel_score
from .model import ModelConfig, build_model
from .utils import atomic_json, atomic_torch_save, case_ids, set_seed


def _device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _amp(device: str, requested: str):
    import torch

    if not device.startswith("cuda") or requested == "fp32":
        return None
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested but this CUDA device does not support BF16")
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    raise ValueError(requested)


def _positive_weights(data_root: Path, training_ids: list[str]) -> np.ndarray:
    positives = np.zeros(3, dtype=np.float64)
    total = 0.0
    for case_id in training_ids:
        label = np.asarray(Image.open(data_root / "training" / "av" / f"{case_id}.png").convert("RGB"), dtype=np.float32) / 255.0
        target = derive_av3_target(label)
        roi = np.asarray(Image.open(data_root / "training" / "masks" / f"{case_id}.png").convert("L")) > 127
        positives += (target * roi[None]).sum(axis=(1, 2))
        total += float(roi.sum())
    negatives = total - positives
    return np.clip(negatives / np.maximum(positives, 1.0), 1.0, 40.0).astype(np.float32)


def _validation(model, loader, device: str, amp_dtype) -> dict[str, object]:
    import torch

    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    rois: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            teacher = batch["teacher"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda" if device.startswith("cuda") else "cpu",
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                corridor = batch["corridor"].to(device, non_blocking=True)
                output = model(features, teacher, mask, corridor)
            for index in range(len(batch["case_ids"])):
                probabilities.append(output["probability"][index].float().cpu().numpy())
                targets.append(batch["target"][index].numpy())
                rois.append(batch["mask"][index, 0].numpy() > 0.5)
    reports = [evaluate_cases(probabilities, targets, rois, threshold=value) for value in (0.45, 0.50, 0.55)]
    for report in reports:
        report["pixel_score"] = pixel_score(report)
    selected = max(reports, key=lambda report: float(report["pixel_score"]))
    return {
        "selection_score": float(selected["pixel_score"]),
        "selection_basis": "classification_and_dice_only",
        "selection_threshold": float(selected["threshold"]),
        "at_0_5": next(report for report in reports if report["threshold"] == 0.5),
        "threshold_reports": reports,
    }


def _checkpoint(
    model,
    optimizer,
    scheduler,
    *,
    epoch: int,
    config: dict[str, object],
    history: list[dict[str, object]],
    validation: dict[str, object],
    best_score: float,
    stale_epochs: int,
) -> dict[str, object]:
    return {
        "version": 12,
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
        "history": history,
        "validation": validation,
        "best_score": float(best_score),
        "stale_epochs": int(stale_epochs),
    }


def run_training(args: argparse.Namespace) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader

    device = _device(args.device)
    amp_dtype = _amp(device, args.amp)
    set_seed(args.seed + args.fold)
    manifest = json.loads(args.fold_manifest.read_text(encoding="utf-8"))
    all_ids = case_ids(args.data_root, "training")
    validate_manifest(manifest, all_ids)
    fold = manifest["folds"][args.fold]
    training_ids, validation_ids = list(fold["training"]), list(fold["validation"])
    model_config = ModelConfig(
        input_channels=13 if args.task == "task2" else 8,
        base_channels=args.base_channels,
        max_logit_delta=args.max_logit_delta,
        corridor_radius=args.corridor_radius,
        correction_mode="prune",
        activation_checkpointing=args.activation_checkpointing,
    )
    config = {
        "task": args.task,
        "fold": args.fold,
        "training_ids": training_ids,
        "validation_ids": validation_ids,
        "model": model_config.to_dict(),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "minimum_learning_rate": args.min_lr,
        "weight_decay": args.weight_decay,
        "amp": args.amp,
        "seed": args.seed,
        "early_stopping_patience": args.early_stopping_patience,
        "minimum_epochs": args.minimum_epochs,
    }
    fold_dir = args.run_dir / "models" / args.task / f"fold_{args.fold}"
    config_path = fold_dir / "config.json"
    if fold_dir.exists() and any(fold_dir.iterdir()):
        if not args.resume:
            raise RuntimeError(f"Fold directory is not empty; use --resume: {fold_dir}")
        if not config_path.exists() or json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise RuntimeError(f"Resume configuration mismatch at {fold_dir}")
    else:
        fold_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(config_path, config)

    complete_path = fold_dir / "complete.json"
    if args.resume and complete_path.exists():
        completed = json.loads(complete_path.read_text(encoding="utf-8"))
        if (fold_dir / "best.pt").exists() and (fold_dir / "last.pt").exists():
            print(json.dumps({"reused_complete_fold": str(fold_dir), **completed}), flush=True)
            return completed

    training_dataset = V12Dataset(
        args.data_root,
        "training",
        args.task,
        training_ids,
        args.teacher_store,
        args.prepared_root,
        augment=True,
        seed=args.seed + args.fold,
        corridor_radius=args.corridor_radius,
    )
    validation_dataset = V12Dataset(
        args.data_root,
        "training",
        args.task,
        validation_ids,
        args.teacher_store,
        args.prepared_root,
        augment=False,
        seed=args.seed,
        corridor_radius=args.corridor_radius,
    )
    generator = torch.Generator().manual_seed(args.seed + args.fold)
    training_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=args.workers > 0,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=min(args.workers, 2),
        pin_memory=device.startswith("cuda"),
        persistent_workers=args.workers > 0,
        collate_fn=collate,
    )
    model = build_model(model_config).to(device)
    positive_weights = torch.from_numpy(_positive_weights(args.data_root, training_ids)).to(device)
    criterion = ResidualChallengeLoss(positive_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.min_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    history: list[dict[str, object]] = []
    start_epoch = 1
    stale_epochs = 0
    baseline = _validation(model, validation_loader, device, amp_dtype)
    best_score = float(baseline["selection_score"])
    best_validation = baseline
    initial = _checkpoint(
        model,
        optimizer,
        scheduler,
        epoch=0,
        config=config,
        history=history,
        validation=baseline,
        best_score=best_score,
        stale_epochs=0,
    )
    best_path = fold_dir / "best.pt"
    last_path = fold_dir / "last.pt"
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = list(checkpoint["history"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        stale_epochs = int(checkpoint["stale_epochs"])
        if best_path.exists():
            best_validation = torch.load(best_path, map_location="cpu", weights_only=False)["validation"]
    else:
        atomic_torch_save(best_path, initial)
        atomic_torch_save(last_path, initial)

    for epoch in range(start_epoch, args.epochs + 1):
        training_dataset.set_epoch(epoch)
        model.train()
        totals: dict[str, float] = {name: 0.0 for name in ("loss", "classification", "dice", "topology", "crossing", "teacher", "residual")}
        batches = 0
        for batch in training_loader:
            features = batch["features"].to(device, non_blocking=True)
            teacher = batch["teacher"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda" if device.startswith("cuda") else "cpu",
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                corridor = batch["corridor"].to(device, non_blocking=True)
                output = model(features, teacher, mask, corridor)
                loss, components = criterion(output, target, teacher, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(loss.detach())
            for name, value in components.items():
                totals[name] += float(value.detach())
            batches += 1
        scheduler.step()
        validation = _validation(model, validation_loader, device, amp_dtype)
        score = float(validation["selection_score"])
        improved = score > best_score + args.min_delta
        if improved:
            best_score = score
            best_validation = validation
            stale_epochs = 0
        elif epoch >= args.minimum_epochs:
            stale_epochs += 1
        record = {
            "epoch": epoch,
            **{name: value / max(batches, 1) for name, value in totals.items()},
            "validation": validation,
            "baseline_score": float(baseline["selection_score"]),
            "best_score": best_score,
            "stale_epochs": stale_epochs,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        state = _checkpoint(
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            config=config,
            history=history,
            validation=validation,
            best_score=best_score,
            stale_epochs=stale_epochs,
        )
        atomic_torch_save(last_path, state)
        if improved:
            atomic_torch_save(best_path, state)
        atomic_json(fold_dir / "history.json", history)
        print(json.dumps(record), flush=True)
        if epoch >= args.minimum_epochs and stale_epochs >= args.early_stopping_patience:
            break
    result = {
        "task": args.task,
        "fold": args.fold,
        "completed_epochs": int(history[-1]["epoch"]) if history else start_epoch - 1,
        "baseline_score": float(baseline["selection_score"]),
        "best_score": best_score,
        "best_validation": best_validation,
        "best_checkpoint": str(best_path),
    }
    atomic_json(complete_path, result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one full-canvas GAVE2 V12 residual fold.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--teacher-store", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--task", choices=("task1", "task2"), required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--base-channels", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--minimum-epochs", type=int, default=25)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-logit-delta", type=float, default=2.5)
    parser.add_argument("--corridor-radius", type=int, default=2)
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    print(json.dumps(run_training(parse_args()), indent=2))


if __name__ == "__main__":
    main()
