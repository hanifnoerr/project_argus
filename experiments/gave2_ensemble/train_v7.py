from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from .data_v6 import GAVE2DatasetV6
from .metrics_v7 import challenge_selection_score_v7
from .train_v6 import (
    BatchProfile,
    ClosedFormWarmupCosineScheduler,
    _import_torch,
    _load_checkpoint,
    _move_batch,
    accumulation_plan,
    augment_full_canvas,
    autocast_context,
    compute_fold_positive_weights_v6,
    compute_roi_channel_stats_v6,
    ensure_fold_directory_ready,
    hash_config_payload,
    hash_official_source_contents,
    load_fold_manifest,
    normalize_batch,
    resolve_resume_artifacts,
    set_seeds,
    torch_collate_v6,
    validate_fold_manifest,
)
from .training_state_v6 import (
    DualBestEarlyStopping,
    atomic_torch_save,
    atomic_write_json,
    capture_rng_state,
    certify_checkpoint,
    hash_file_bytes,
    hash_path_contents,
    restore_rng_state,
)


def task_defaults(task: str) -> dict[str, int]:
    if task == "task1":
        return {"input_channels": 4, "epochs": 30, "minimum_epochs": 12}
    if task == "task2":
        return {"input_channels": 6, "epochs": 40, "minimum_epochs": 15}
    raise ValueError(f"Unsupported task {task!r}")


def build_model_from_config(config: dict[str, object]):
    from .cmrrwnet_v7 import create_cmrrwnet_v7

    return create_cmrrwnet_v7(
        task=str(config["task"]),
        base_channels=int(config["base_channels"]),
        num_refinements=int(config["num_refinements"]),
        activation_checkpointing=bool(config["activation_checkpointing"]),
        official_source=config.get("official_source"),
    )


def _smoke_inference(model, checkpoint, config) -> None:
    from .losses_v7 import conditional_probabilities_from_logits

    torch, _ = _import_torch()
    model.eval()
    value = torch.zeros(1, int(config["input_channels"]), 32, 32)
    with torch.inference_mode():
        probability = conditional_probabilities_from_logits(model(value))
    if tuple(probability.shape) != (1, 3, 32, 32) or not torch.isfinite(probability).all():
        raise RuntimeError("V7 checkpoint smoke inference failed")
    overlap = (probability[:, 0] >= 0.5) & (probability[:, 2] >= 0.5)
    if bool(overlap.any()):
        raise RuntimeError("V7 checkpoint produced overlapping A/V decisions")


def certify_best_checkpoint(fold_dir: Path) -> dict[str, object]:
    checkpoint_path = fold_dir / "best.pt"
    checkpoint = certify_checkpoint(checkpoint_path, build_model_from_config, _smoke_inference)
    report = {
        "version": 7,
        "checkpoint": str(checkpoint_path),
        "selection_score": float(checkpoint["validation"]["selection_score"]),
        "epoch": int(checkpoint["epoch"]),
        "config_content_sha256": checkpoint["config"]["config_content_sha256"],
    }
    atomic_write_json(fold_dir / "best.certified.json", report)
    return report


def run_validation(model, loader, criterion, device: str, amp: str, normalization) -> dict[str, object]:
    from .losses_v7 import conditional_probabilities_from_logits

    torch, _ = _import_torch()
    model.eval()
    losses: list[float] = []
    metrics: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in loader:
            batch = _move_batch(batch, device, normalization)
            with autocast_context(torch, device, amp):
                predictions = model(batch["images"])
                loss = criterion(predictions, batch["targets"], batch["masks"])
            probability = conditional_probabilities_from_logits(predictions).float().cpu().numpy()
            target = batch["targets"].float().cpu().numpy()
            mask = batch["masks"].float().cpu().numpy()
            losses.append(float(loss.detach().cpu()))
            metrics.append(challenge_selection_score_v7(probability, target, mask))
    if not losses:
        raise RuntimeError("Validation loader is empty")
    scalar_keys = (
        "dice_mean",
        "av_sensitivity",
        "av_specificity",
        "av_accuracy",
        "classification_mean",
        "path_recall",
        "path_precision",
        "path_score",
        "selection_score",
    )
    return {
        "loss": float(np.mean(losses)),
        **{key: float(np.mean([float(row[key]) for row in metrics])) for key in scalar_keys},
    }


def _prepare_config(args, defaults, train_ids, val_ids, normalization, positive_weights, manifest_hash, warm_path):
    config = {
        "version": 7,
        "task": args.task,
        "fold_index": int(args.fold),
        "fold_manifest_path": str(Path(args.fold_manifest)),
        "fold_manifest_sha256": manifest_hash,
        "train_ids": list(train_ids),
        "val_ids": list(val_ids),
        "input_channels": defaults["input_channels"],
        "base_channels": int(args.base_channels),
        "num_refinements": int(args.num_refinements),
        "activation_checkpointing": not args.no_activation_checkpointing,
        "normalization": normalization,
        "positive_weights": positive_weights.astype(np.float32).tolist(),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "grad_accum": int(args.grad_accum),
        "amp": args.amp,
        "epochs": int(args.epochs),
        "minimum_epochs": int(args.minimum_epochs),
        "early_stopping_patience": int(args.early_stopping_patience),
        "early_stopping_min_delta": float(args.early_stopping_min_delta),
        "learning_rate": float(args.lr),
        "minimum_learning_rate": float(args.min_lr),
        "weight_decay": float(args.weight_decay),
        "gradient_clip": float(args.grad_clip),
        "warm_start_checkpoint": str(warm_path),
        "warm_start_sha256": hash_file_bytes(warm_path),
        "official_source": str(args.official_source) if args.official_source else None,
        "official_source_sha256": hash_official_source_contents(args.official_source),
        "code_sha256": hash_path_contents(Path(__file__).resolve().parent),
        "output_semantics": "vessel_sigmoid_plus_conditional_av_softmax",
    }
    config["config_content_sha256"] = hash_config_payload(config)
    return config


def train_fold(args: argparse.Namespace) -> Path:
    from .losses_v7 import ConditionalPathLossV7

    torch, DataLoader = _import_torch()
    device = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else str(args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = "cpu"
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    defaults = task_defaults(args.task)
    manifest = load_fold_manifest(args.fold_manifest)
    case_ids = [path.stem for path in sorted((Path(args.data_root) / "training" / "images").glob("*.png"))]
    validate_fold_manifest(manifest, case_ids)
    fold = manifest["folds"][int(args.fold)]
    train_ids = list(fold["training"])
    val_ids = list(fold["validation"])
    manifest_hash = hash_file_bytes(args.fold_manifest)
    warm_path = Path(args.warm_start_run_dir) / "cmrrwnet_v6" / args.task / f"fold_{args.fold}" / "best.pt"
    if not warm_path.is_file():
        raise FileNotFoundError(f"V7 requires the matching certified V6 warm start: {warm_path}")

    set_seeds(args.seed + args.fold)
    normalization = compute_roi_channel_stats_v6(args.data_root, args.task, train_ids)
    train_dataset = GAVE2DatasetV6(args.data_root, split="training", task=args.task, case_ids=train_ids)
    val_dataset = GAVE2DatasetV6(args.data_root, split="training", task=args.task, case_ids=val_ids)
    positive_weights = compute_fold_positive_weights_v6(train_dataset)
    loader_generator = torch.Generator().manual_seed(args.seed + args.fold)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.startswith("cuda"),
        collate_fn=torch_collate_v6,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=min(args.workers, 2),
        pin_memory=device.startswith("cuda"),
        collate_fn=torch_collate_v6,
    )

    config = _prepare_config(
        args, defaults, train_ids, val_ids, normalization, positive_weights, manifest_hash, warm_path
    )
    fold_dir = Path(args.run_dir) / "cmrrwnet_v7" / args.task / f"fold_{args.fold}"
    config_path = fold_dir / "config.json"
    history_path = fold_dir / "history.json"
    history_prev_path = fold_dir / "history.prev.json"
    last_path = fold_dir / "last.pt"
    last_prev_path = fold_dir / "last.prev.pt"
    best_path = fold_dir / "best.pt"
    fold_populated = ensure_fold_directory_ready(
        fold_dir=fold_dir,
        config_path=config_path,
        requested_config={key: value for key, value in config.items() if key != "config_content_sha256"},
        manifest_hash=manifest_hash,
        resume=args.resume,
    )
    fold_dir.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        atomic_write_json(config_path, config)

    model = build_model_from_config(config).to(device)
    criterion = ConditionalPathLossV7(positive_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    profile = BatchProfile(args.batch_size, args.grad_accum)
    steps_per_epoch = max((len(train_loader) + profile.grad_accum - 1) // profile.grad_accum, 1)
    scheduler = ClosedFormWarmupCosineScheduler(
        optimizer,
        total_steps=max(args.epochs * steps_per_epoch, 1),
        warmup_steps=min(2 * steps_per_epoch, max(args.epochs * steps_per_epoch, 1)),
        initial_lr=args.lr,
        min_lr=args.min_lr,
    )
    stopper = DualBestEarlyStopping(
        task=args.task,
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        minimum_epoch=args.minimum_epochs,
    )
    history: list[dict[str, object]] = []
    start_epoch = 1

    if args.resume and fold_populated:
        artifacts = resolve_resume_artifacts(
            last_path, last_prev_path, history_path, lambda path: _load_checkpoint(torch, path, "cpu")
        )
        state = artifacts["checkpoint_state"]
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        for key in ("best_value", "best_epoch", "reference_value", "reference_epoch", "stale_epochs", "last_epoch"):
            setattr(stopper, key, state["early_stopping"][key])
        restore_rng_state(state["rng_state"], loader_generator)
        start_epoch = int(state["epoch"]) + 1
        history = [dict(row) for row in artifacts["history"] if int(row["epoch"]) < start_epoch]
        del state
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    else:
        warm_state = _load_checkpoint(torch, warm_path, "cpu")
        model.load_state_dict(warm_state["state_dict"], strict=True)
        del warm_state

    if start_epoch > args.epochs:
        if best_path.exists() and not (fold_dir / "best.certified.json").exists():
            certify_best_checkpoint(fold_dir)
        return fold_dir

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        steps = 0
        plan = accumulation_plan(len(train_loader), profile.grad_accum)
        for batch_state, batch in zip(plan, train_loader):
            batch = augment_full_canvas(batch, args.task)
            batch = _move_batch(batch, device, normalization)
            with autocast_context(torch, device, args.amp):
                predictions = model(batch["images"])
                loss = criterion(predictions, batch["targets"], batch["masks"])
                scaled_loss = loss / int(batch_state["window_size"])
            scaled_loss.backward()
            if batch_state["should_step"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running_loss += float(loss.detach().cpu())
            steps += 1

        validation = run_validation(model, val_loader, criterion, device, args.amp, normalization)
        update = stopper.update(validation["selection_score"], epoch)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(steps, 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "best_selection_score": stopper.best_value,
            "best_epoch": stopper.best_epoch,
            "stale_epochs": stopper.stale_epochs,
            **validation,
        }
        history.append(row)
        state = {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "history": history,
            "early_stopping": {
                "best_value": stopper.best_value,
                "best_epoch": stopper.best_epoch,
                "reference_value": stopper.reference_value,
                "reference_epoch": stopper.reference_epoch,
                "stale_epochs": stopper.stale_epochs,
                "last_epoch": stopper.last_epoch,
            },
            "rng_state": capture_rng_state(loader_generator),
            "validation": validation,
            "config": config,
        }
        atomic_torch_save(last_path, state, previous_path=last_prev_path)
        atomic_write_json(history_path, history, previous_path=history_prev_path)
        if update.absolute_improved or not best_path.exists():
            atomic_torch_save(best_path, state)
            certify_best_checkpoint(fold_dir)
        print(json.dumps(row), flush=True)
        if update.should_stop:
            atomic_write_json(
                fold_dir / "early_stop.json",
                {"stopped_epoch": epoch, "best_epoch": stopper.best_epoch, "best_selection_score": stopper.best_value},
            )
            break
    return fold_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm-start path-consistent CMRRWNet V7 from a V6 fold.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--warm-start-run-dir", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--task", choices=("task1", "task2"), required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--base-channels", type=int, required=True)
    parser.add_argument("--num-refinements", type=int, default=2)
    parser.add_argument("--batch-size", type=int, choices=(2, 4, 6), default=2)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--minimum-epochs", type=int)
    parser.add_argument("--amp", choices=("none", "bf16"), default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--min-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--early-stopping-min-delta", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    parser.add_argument("--official-source", type=Path)
    args = parser.parse_args(argv)
    defaults = task_defaults(args.task)
    args.epochs = defaults["epochs"] if args.epochs is None else args.epochs
    args.minimum_epochs = defaults["minimum_epochs"] if args.minimum_epochs is None else args.minimum_epochs
    if args.grad_accum < 1:
        parser.error("--grad-accum must be at least 1")
    return args


def main() -> None:
    train_fold(parse_args())


if __name__ == "__main__":
    main()
