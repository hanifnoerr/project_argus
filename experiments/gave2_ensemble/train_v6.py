from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data_v6 import GAVE2DatasetV6
from .metrics_v6 import challenge_selection_score
from .training_state_v6 import (
    DualBestEarlyStopping,
    atomic_torch_save,
    atomic_write_json,
    capture_rng_state,
    certify_checkpoint,
    hash_file_bytes,
    hash_path_contents,
    restore_rng_state,
    warmup_cosine_lr,
)
from .training_utils_v2 import positive_weights_from_counts


@dataclass(frozen=True)
class TaskSpec:
    task: str
    input_channels: int
    epoch_cap: int
    minimum_epochs: int


@dataclass(frozen=True)
class BatchProfile:
    batch_size: int
    grad_accum: int


def _import_torch():
    try:
        import torch
        from torch.utils.data import DataLoader

        return torch, DataLoader
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("CMRRWNet v6 training requires PyTorch") from exc


def task_spec(task: str) -> TaskSpec:
    normalized = str(task).lower()
    if normalized == "task1":
        return TaskSpec(task="task1", input_channels=4, epoch_cap=50, minimum_epochs=25)
    if normalized == "task2":
        return TaskSpec(task="task2", input_channels=6, epoch_cap=100, minimum_epochs=40)
    raise ValueError(f"Unsupported task {task!r}")


def preferred_batch_profiles() -> list[BatchProfile]:
    return [BatchProfile(batch_size=6, grad_accum=1), BatchProfile(batch_size=4, grad_accum=1), BatchProfile(batch_size=2, grad_accum=2)]


def accumulation_window_size(total_batches: int, step_index: int, grad_accum: int) -> int:
    if total_batches < 1:
        raise ValueError("total_batches must be at least 1")
    if grad_accum < 1:
        raise ValueError("grad_accum must be at least 1")
    if step_index < 1 or step_index > total_batches:
        raise ValueError(f"step_index must be in [1, {total_batches}]")
    remainder = total_batches % grad_accum
    if remainder == 0:
        return grad_accum
    last_window_start = total_batches - remainder + 1
    if step_index >= last_window_start:
        return remainder
    return grad_accum


def accumulation_plan(total_batches: int, grad_accum: int) -> list[dict[str, object]]:
    plan = []
    for step_index in range(1, total_batches + 1):
        window_size = accumulation_window_size(total_batches=total_batches, step_index=step_index, grad_accum=grad_accum)
        should_step = step_index % grad_accum == 0 or step_index == total_batches
        is_window_start = step_index == 1 or plan[-1]["should_step"]
        plan.append(
            {
                "step_index": step_index,
                "window_size": window_size,
                "should_step": should_step,
                "is_window_start": is_window_start,
            }
        )
    return plan


def is_cuda_oom(exc: BaseException) -> bool:
    if exc.__class__.__name__ == "OutOfMemoryError":
        return True
    message = str(exc).lower()
    return "cuda out of memory" in message or "cuda error: out of memory" in message


def select_batch_profile(attempt, profiles: list[BatchProfile] | None = None):
    candidates = preferred_batch_profiles() if profiles is None else list(profiles)
    last_oom = None
    for profile in candidates:
        try:
            return profile, attempt(profile)
        except Exception as exc:
            if not is_cuda_oom(exc):
                raise
            last_oom = exc
    if last_oom is None:
        raise RuntimeError("No batch profiles were attempted")
    raise last_oom


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


def hash_config_payload(payload: dict[str, object]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def default_official_source_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "knowledge_base"
        / "sources"
        / "github"
        / "Peng2004_CMRRWNet"
        / "train"
        / "models.py"
    )


def hash_official_source_contents(path: Path | str | None = None) -> str:
    source_path = default_official_source_path() if path is None else Path(path)
    return hash_file_bytes(source_path)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_resume_config(
    config_path: Path | str,
    requested_config: dict[str, object],
    requested_manifest_hash: str,
) -> dict[str, object]:
    stored = _load_json(Path(config_path))
    if stored.get("fold_manifest_sha256") != requested_manifest_hash:
        raise RuntimeError("Existing fold manifest hash does not match requested resume manifest")
    stored_hash = stored.get("config_content_sha256")
    requested_hash = hash_config_payload(requested_config)
    if stored_hash != requested_hash:
        raise RuntimeError("Existing fold config hash does not match requested resume config")
    return stored


def ensure_fold_directory_ready(
    *,
    fold_dir: Path | str,
    config_path: Path | str,
    requested_config: dict[str, object],
    manifest_hash: str,
    resume: bool,
) -> bool:
    fold_path = Path(fold_dir)
    config_file = Path(config_path)
    fold_populated = fold_path.exists() and any(fold_path.iterdir())
    if config_file.exists():
        validate_resume_config(config_file, requested_config, manifest_hash)
        if not resume:
            raise RuntimeError(f"Fold directory is not empty; pass --resume only for this exact run: {fold_path}")
    elif fold_populated:
        raise RuntimeError(f"Fold directory is populated but missing immutable config.json: {fold_path}")
    return fold_populated


def load_fold_manifest(path: Path | str) -> dict[str, object]:
    manifest = _load_json(Path(path))
    if manifest.get("seed") != 77:
        raise ValueError("V6 requires a seed-77 three-fold manifest")
    folds = manifest.get("folds")
    if not isinstance(folds, list) or len(folds) != 3:
        raise ValueError("V6 requires exactly three folds")
    return manifest


def validate_fold_manifest(manifest: dict[str, object], case_ids: list[str]) -> None:
    known = set(case_ids)
    ownership: dict[str, int] = {}
    for expected_index, fold in enumerate(manifest["folds"]):
        if int(fold["fold_index"]) != expected_index:
            raise ValueError(f"Fold index mismatch at position {expected_index}")
        training = list(fold["training"])
        validation = list(fold["validation"])
        if any(case_id not in known for case_id in training + validation):
            raise ValueError("Fold manifest references unknown case ids")
        if set(training) & set(validation):
            raise ValueError("Fold training and validation splits overlap")
        for case_id in validation:
            if case_id in ownership:
                raise ValueError(f"Validation case {case_id} appears in multiple folds")
            ownership[case_id] = expected_index
    if set(ownership) != known:
        missing = sorted(known - set(ownership))
        raise ValueError(f"Every case must appear in validation exactly once; missing={missing}")


def normalize_batch(images, masks, normalization: dict[str, list[float]]):
    torch, _ = _import_torch()
    mean = torch.as_tensor(normalization["mean"], dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    std = torch.as_tensor(normalization["std"], dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    return ((images - mean) / std.clamp_min(1e-6)) * masks


def compute_roi_channel_stats_v6(data_root: Path | str, task: str, case_ids: list[str]) -> dict[str, list[float]]:
    dataset = GAVE2DatasetV6(data_root=data_root, split="training", task=task, case_ids=case_ids)
    total = None
    total_sq = None
    count = 0.0
    for sample in dataset:
        mask = sample.mask[0] > 0.5
        values = sample.image[:, mask].astype(np.float64)
        if total is None:
            total = np.zeros(values.shape[0], dtype=np.float64)
            total_sq = np.zeros(values.shape[0], dtype=np.float64)
        total += values.sum(axis=1)
        total_sq += np.square(values).sum(axis=1)
        count += float(values.shape[1])
    if total is None or total_sq is None or count <= 0:
        raise ValueError("Cannot compute normalization statistics from an empty ROI")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-8)
    return {"mean": mean.astype(np.float32).tolist(), "std": np.sqrt(variance).astype(np.float32).tolist()}


def compute_fold_positive_weights_v6(dataset: GAVE2DatasetV6) -> np.ndarray:
    positive = np.zeros(3, dtype=np.float64)
    total = np.zeros(3, dtype=np.float64)
    for sample in dataset:
        if sample.target is None:
            raise ValueError("Training targets are required to compute positive weights")
        roi = sample.mask[0] > 0.5
        positive += sample.target[:, roi].sum(axis=1)
        total += float(roi.sum())
    return positive_weights_from_counts(positive, total)


def torch_collate_v6(samples):
    torch, _ = _import_torch()
    images = torch.from_numpy(np.stack([sample.image for sample in samples], axis=0))
    masks = torch.from_numpy(np.stack([sample.mask for sample in samples], axis=0))
    targets = torch.from_numpy(np.stack([sample.target for sample in samples], axis=0))
    case_ids = [sample.case_id for sample in samples]
    sizes = [sample.original_size for sample in samples]
    return {"case_ids": case_ids, "images": images, "masks": masks, "targets": targets, "sizes": sizes}


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
    images = torch.cat((rgb, images[:, 3:]), dim=1)
    if task == "task2" and images.shape[1] >= 6 and random.random() < 0.10:
        channel = 4 if random.random() < 0.5 else 5
        images[:, channel : channel + 1] = 0.0
    batch["images"] = images
    batch["masks"] = masks
    batch["targets"] = targets
    return batch


def _move_batch(batch: dict, device: str, normalization: dict[str, list[float]]) -> dict:
    batch["images"] = normalize_batch(batch["images"], batch["masks"], normalization)
    batch["images"] = batch["images"].to(device, non_blocking=True)
    batch["masks"] = batch["masks"].to(device, non_blocking=True)
    batch["targets"] = batch["targets"].to(device, non_blocking=True)
    return batch


def _load_checkpoint(torch, path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        return torch.load(path, map_location=device)


def resolve_resume_artifacts(last_path: Path | str, last_prev_path: Path | str, history_path: Path | str, loader) -> dict[str, object]:
    primary = Path(last_path)
    fallback = Path(last_prev_path)
    history_file = Path(history_path)
    errors: list[str] = []
    for candidate in (primary, fallback):
        if not candidate.exists():
            errors.append(f"{candidate.name} missing")
            continue
        try:
            state = loader(candidate)
        except Exception as exc:
            errors.append(f"{candidate.name} invalid: {exc}")
            continue
        history = state.get("history")
        if not isinstance(history, list):
            disk_hint = " and stale history.json exists" if history_file.exists() else ""
            errors.append(f"{candidate.name} invalid: checkpoint history missing{disk_hint}")
            continue
        normalized_history = [dict(row) for row in history]
        return {"checkpoint_path": candidate, "checkpoint_state": state, "history": normalized_history}
    raise RuntimeError(f"Resume requires a valid last.pt or last.prev.pt; {'; '.join(errors)}")


class ClosedFormWarmupCosineScheduler:
    def __init__(self, optimizer, total_steps: int, warmup_steps: int, initial_lr: float, min_lr: float, current_step: int = 0):
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.initial_lr = float(initial_lr)
        self.min_lr = float(min_lr)
        self.current_step = int(current_step)
        self._set_lr_for_upcoming_step()

    def _set_lr_for_upcoming_step(self) -> None:
        next_step = min(self.current_step, self.total_steps - 1)
        lr = warmup_cosine_lr(
            step=next_step,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            initial_lr=self.initial_lr,
            min_lr=self.min_lr,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self) -> None:
        if self.current_step < self.total_steps - 1:
            self.current_step += 1
        self._set_lr_for_upcoming_step()

    def state_dict(self) -> dict[str, object]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "initial_lr": self.initial_lr,
            "min_lr": self.min_lr,
            "current_step": self.current_step,
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.total_steps = int(state_dict["total_steps"])
        self.warmup_steps = int(state_dict["warmup_steps"])
        self.initial_lr = float(state_dict["initial_lr"])
        self.min_lr = float(state_dict["min_lr"])
        self.current_step = int(state_dict["current_step"])
        self._set_lr_for_upcoming_step()


def _topology_weight_for_epoch(args: argparse.Namespace, epoch: int) -> float:
    if args.cpu_smoke or args.topology_weight <= 0:
        return 0.0 if args.cpu_smoke else float(args.topology_weight)
    return float(args.topology_weight) * min(float(epoch) / 10.0, 1.0)


def run_validation(model, loader, criterion, topology_loss, device: str, amp: str, normalization, topology_weight: float) -> dict[str, object]:
    from .losses import probabilities_from_logits

    torch, _ = _import_torch()
    model.eval()
    losses = []
    probabilities = []
    targets = []
    roi = []
    with torch.inference_mode():
        for batch in loader:
            batch = _move_batch(batch, device, normalization)
            with autocast_context(torch, device, amp):
                predictions = model(batch["images"])
                loss = criterion(predictions, batch["targets"], batch["masks"])
                if topology_weight > 0:
                    loss = loss + topology_weight * topology_loss(predictions, batch["targets"], batch["masks"])
            losses.append(float(loss.detach().cpu()))
            probabilities.append(probabilities_from_logits(predictions).float().cpu().numpy())
            targets.append(batch["targets"].float().cpu().numpy())
            roi.append(batch["masks"].float().cpu().numpy())
    if not losses:
        raise RuntimeError("Validation loader is empty")
    metrics = challenge_selection_score(
        np.concatenate(probabilities, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(roi, axis=0),
    )
    return {"loss": float(np.mean(losses)), **metrics}


def build_model_from_config(config: dict[str, object]):
    from .cmrrwnet_v6 import create_cmrrwnet_v6

    return create_cmrrwnet_v6(
        task=str(config["task"]),
        base_channels=int(config["base_channels"]),
        num_refinements=int(config["num_refinements"]),
        activation_checkpointing=bool(config["activation_checkpointing"]),
        official_source=config.get("official_source"),
    )


def _smoke_inference(model, checkpoint, config):
    torch, _ = _import_torch()
    height, width = (32, 32)
    if "smoke_input_size" in config:
        height, width = [int(value) for value in config["smoke_input_size"]]
    x = torch.zeros(1, int(config["input_channels"]), height, width, dtype=torch.float32)
    with torch.inference_mode():
        output = model(x)
    final = output[-1] if isinstance(output, (list, tuple)) else output
    if not torch.isfinite(final).all():
        raise RuntimeError("Checkpoint smoke inference produced non-finite values")


def certify_best_checkpoint(fold_dir: Path) -> dict[str, object]:
    best_path = fold_dir / "best.pt"
    checkpoint = certify_checkpoint(best_path, build_model_from_config, _smoke_inference)
    report = {
        "checkpoint": str(best_path),
        "selection_score": float(checkpoint["validation"]["selection_score"]),
        "epoch": int(checkpoint["epoch"]),
        "config_content_sha256": checkpoint["config"]["config_content_sha256"],
    }
    atomic_write_json(fold_dir / "best.certified.json", report)
    return report


def _prepare_config(
    args: argparse.Namespace,
    spec: TaskSpec,
    batch_profile: BatchProfile,
    train_ids: list[str],
    val_ids: list[str],
    normalization: dict[str, list[float]],
    positive_weights: np.ndarray,
    manifest_hash: str,
) -> dict[str, object]:
    config = {
        "version": 6,
        "task": spec.task,
        "fold_index": int(args.fold),
        "fold_manifest_path": str(Path(args.fold_manifest)),
        "fold_manifest_sha256": manifest_hash,
        "train_ids": list(train_ids),
        "val_ids": list(val_ids),
        "input_channels": spec.input_channels,
        "base_channels": int(args.base_channels),
        "num_refinements": int(args.num_refinements),
        "activation_checkpointing": not args.no_activation_checkpointing,
        "normalization": normalization,
        "positive_weights": positive_weights.astype(np.float32).tolist(),
        "seed": int(args.seed),
        "batch_size": int(batch_profile.batch_size),
        "grad_accum": int(batch_profile.grad_accum),
        "amp": str(args.amp),
        "epochs": int(args.epochs),
        "minimum_epochs": int(args.minimum_epochs),
        "early_stopping_patience": int(args.early_stopping_patience),
        "early_stopping_min_delta": float(args.early_stopping_min_delta),
        "learning_rate": float(args.lr),
        "minimum_learning_rate": float(args.min_lr),
        "weight_decay": float(args.weight_decay),
        "gradient_clip": float(args.grad_clip),
        "topology_weight": float(args.topology_weight),
        "cpu_smoke": bool(args.cpu_smoke),
        "official_source": str(args.official_source) if args.official_source is not None else None,
        "official_source_sha256": hash_official_source_contents(args.official_source),
        "code_sha256": hash_path_contents(Path(__file__).resolve().parent),
        "smoke_input_size": [32, 32],
    }
    config["config_content_sha256"] = hash_config_payload(config)
    return config


def train_fold(args: argparse.Namespace) -> Path:
    torch, DataLoader = _import_torch()
    device = str(args.device)
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    spec = task_spec(args.task)
    batch_profile = BatchProfile(
        batch_size=int(args.batch_size if args.batch_size is not None else preferred_batch_profiles()[0].batch_size),
        grad_accum=2 if int(args.batch_size if args.batch_size is not None else 6) == 2 else 1,
    )

    manifest = load_fold_manifest(args.fold_manifest)
    case_ids = sorted((Path(args.data_root) / "training" / "images").glob("*.png"))
    known_case_ids = [path.stem for path in case_ids]
    validate_fold_manifest(manifest, known_case_ids)
    fold = manifest["folds"][int(args.fold)]
    train_ids = list(fold["training"])
    val_ids = list(fold["validation"])
    manifest_hash = hash_file_bytes(args.fold_manifest)

    set_seeds(args.seed + args.fold)
    normalization = compute_roi_channel_stats_v6(args.data_root, args.task, train_ids)
    train_dataset = GAVE2DatasetV6(args.data_root, split="training", task=args.task, case_ids=train_ids)
    val_dataset = GAVE2DatasetV6(args.data_root, split="training", task=args.task, case_ids=val_ids)
    positive_weights = compute_fold_positive_weights_v6(train_dataset)
    loader_generator = torch.Generator().manual_seed(args.seed + args.fold)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_profile.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        pin_memory=device.startswith("cuda"),
        collate_fn=torch_collate_v6,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=min(int(args.workers), 2),
        pin_memory=device.startswith("cuda"),
        collate_fn=torch_collate_v6,
    )

    config = _prepare_config(args, spec, batch_profile, train_ids, val_ids, normalization, positive_weights, manifest_hash)
    fold_dir = Path(args.run_dir) / "cmrrwnet_v6" / spec.task / f"fold_{args.fold}"
    config_path = fold_dir / "config.json"
    history_path = fold_dir / "history.json"
    last_path = fold_dir / "last.pt"
    best_path = fold_dir / "best.pt"
    last_prev_path = fold_dir / "last.prev.pt"
    history_prev_path = fold_dir / "history.prev.json"

    from .cmrrwnet_v6 import create_cmrrwnet_v6
    from .losses import BalancedRecursiveLoss
    from .losses_v6 import VesselTopologyLoss

    model = create_cmrrwnet_v6(
        task=args.task,
        base_channels=args.base_channels,
        num_refinements=args.num_refinements,
        activation_checkpointing=not args.no_activation_checkpointing,
        official_source=args.official_source,
    ).to(device)
    criterion = BalancedRecursiveLoss(positive_weights=positive_weights, topology_weight=0.0).to(device)
    topology_loss = VesselTopologyLoss(from_logits=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max((len(train_loader) + batch_profile.grad_accum - 1) // batch_profile.grad_accum, 1)
    total_steps = max(int(args.epochs) * steps_per_epoch, 1)
    warmup_steps = min(total_steps, 3 * steps_per_epoch)
    scheduler = ClosedFormWarmupCosineScheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        initial_lr=args.lr,
        min_lr=args.min_lr,
    )
    early_stopper = DualBestEarlyStopping(
        task=args.task,
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        minimum_epoch=args.minimum_epochs,
    )

    fold_dir.mkdir(parents=True, exist_ok=True)
    fold_populated = ensure_fold_directory_ready(
        fold_dir=fold_dir,
        config_path=config_path,
        requested_config={k: v for k, v in config.items() if k != "config_content_sha256"},
        manifest_hash=manifest_hash,
        resume=bool(args.resume),
    )
    if not config_path.exists():
        atomic_write_json(config_path, config)

    history: list[dict[str, object]] = []
    start_epoch = 1
    if args.resume and fold_populated:
        resume_artifacts = resolve_resume_artifacts(
            last_path,
            last_prev_path,
            history_path,
            lambda checkpoint_path: _load_checkpoint(torch, checkpoint_path, "cpu"),
        )
        state = resume_artifacts["checkpoint_state"]
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        stopper_state = state["early_stopping"]
        early_stopper.best_value = stopper_state["best_value"]
        early_stopper.best_epoch = stopper_state["best_epoch"]
        early_stopper.reference_value = stopper_state["reference_value"]
        early_stopper.reference_epoch = stopper_state["reference_epoch"]
        early_stopper.stale_epochs = int(stopper_state["stale_epochs"])
        early_stopper.last_epoch = stopper_state["last_epoch"]
        restore_rng_state(state["rng_state"], loader_generator)
        start_epoch = int(state["epoch"]) + 1
        history = [row for row in resume_artifacts["history"] if int(row.get("epoch", 0)) <= int(state["epoch"])]
        atomic_write_json(history_path, history, previous_path=history_prev_path)
        del state
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        if start_epoch > int(args.epochs):
            if best_path.exists() and not (fold_dir / "best.certified.json").exists():
                certify_best_checkpoint(fold_dir)
            return fold_dir
    elif history_path.exists():
        history = list(_load_json(history_path))

    for epoch in range(start_epoch, int(args.epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        steps = 0
        topology_weight = _topology_weight_for_epoch(args, epoch)
        total_batches = len(train_loader)
        batch_plan = accumulation_plan(total_batches=total_batches, grad_accum=batch_profile.grad_accum)
        for batch_state, batch in zip(batch_plan, train_loader):
            batch = augment_full_canvas(batch, args.task)
            batch = _move_batch(batch, device, normalization)
            divisor = int(batch_state["window_size"])
            with autocast_context(torch, device, args.amp):
                predictions = model(batch["images"])
                loss = criterion(predictions, batch["targets"], batch["masks"])
                if topology_weight > 0:
                    loss = loss + topology_weight * topology_loss(predictions, batch["targets"], batch["masks"])
                scaled_loss = loss / divisor
            scaled_loss.backward()
            if bool(batch_state["should_step"]):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running_loss += float(loss.detach().cpu())
            steps += 1

        validation = run_validation(model, val_loader, criterion, topology_loss, device, args.amp, normalization, topology_weight)
        update = early_stopper.update(float(validation["selection_score"]), epoch=epoch)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(steps, 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "topology_weight": topology_weight,
            "best_selection_score": early_stopper.best_value,
            "best_epoch": early_stopper.best_epoch,
            "reference_selection_score": early_stopper.reference_value,
            "reference_epoch": early_stopper.reference_epoch,
            "stale_epochs": early_stopper.stale_epochs,
            **validation,
        }
        history.append(row)
        checkpoint_state = {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "history": history,
            "early_stopping": {
                "best_value": early_stopper.best_value,
                "best_epoch": early_stopper.best_epoch,
                "reference_value": early_stopper.reference_value,
                "reference_epoch": early_stopper.reference_epoch,
                "stale_epochs": early_stopper.stale_epochs,
                "last_epoch": early_stopper.last_epoch,
            },
            "rng_state": capture_rng_state(loader_generator),
            "validation": validation,
            "config": config,
        }
        atomic_torch_save(last_path, checkpoint_state, previous_path=last_prev_path)
        atomic_write_json(history_path, history, previous_path=history_prev_path)
        if update.absolute_improved or not best_path.exists():
            atomic_torch_save(best_path, checkpoint_state)
            certify_best_checkpoint(fold_dir)
        if update.should_stop:
            atomic_write_json(
                fold_dir / "early_stop.json",
                {
                    "stopped_epoch": epoch,
                    "best_epoch": early_stopper.best_epoch,
                    "best_selection_score": early_stopper.best_value,
                    "reference_epoch": early_stopper.reference_epoch,
                    "reference_selection_score": early_stopper.reference_value,
                },
            )
            break
    if best_path.exists() and not (fold_dir / "best.certified.json").exists():
        certify_best_checkpoint(fold_dir)
    return fold_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train corrected full-resolution CMRRWNet v6.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--task", choices=("task1", "task2"), required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--num-refinements", type=int, default=2)
    parser.add_argument("--batch-size", type=int, choices=(2, 4, 6))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--amp", choices=("none", "bf16"), default="bf16")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--topology-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--official-source", type=Path)
    args = parser.parse_args(argv)
    spec = task_spec(args.task)
    args.input_channels = spec.input_channels
    args.minimum_epochs = 0 if args.cpu_smoke else spec.minimum_epochs
    if args.epochs is None:
        args.epochs = 1 if args.cpu_smoke else spec.epoch_cap
    return args


def main() -> None:
    train_fold(parse_args())


if __name__ == "__main__":
    main()
