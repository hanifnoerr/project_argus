from __future__ import annotations

import hashlib
import json
import math
import os
import random
import uuid
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np


def _import_torch():
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("PyTorch is required for this operation") from exc


@dataclass(frozen=True)
class EarlyStoppingUpdate:
    absolute_improved: bool
    significant_improved: bool
    should_stop: bool


@dataclass
class DualBestEarlyStopping:
    task: str
    mode: str = "max"
    patience: int = 7
    min_delta: float = 1e-4
    minimum_epoch: int | None = None
    best_value: float | None = None
    best_epoch: int | None = None
    reference_value: float | None = None
    reference_epoch: int | None = None
    stale_epochs: int = 0
    last_epoch: int | None = None

    def __post_init__(self) -> None:
        task = self.task.lower()
        if task not in {"task1", "task2"}:
            raise ValueError(f"Unsupported task {self.task!r}")
        self.task = task
        if self.mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        if self.minimum_epoch is None:
            self.minimum_epoch = 40 if self.task == "task2" else 25

    @property
    def should_stop(self) -> bool:
        return self.stale_epochs >= self.patience

    def _is_better(self, value: float, baseline: float, delta: float = 0.0) -> bool:
        if self.mode == "max":
            return value > baseline + delta
        return value < baseline - delta

    def update(self, value: float, epoch: int) -> EarlyStoppingUpdate:
        score = float(value)
        current_epoch = int(epoch)
        self.last_epoch = current_epoch
        if self.best_value is None:
            self.best_value = score
            self.best_epoch = current_epoch
            self.reference_value = score
            self.reference_epoch = current_epoch
            self.stale_epochs = 0
            return EarlyStoppingUpdate(True, True, self.should_stop)

        absolute_improved = self._is_better(score, self.best_value)
        significant_improved = self._is_better(score, self.reference_value, delta=self.min_delta)
        if absolute_improved:
            self.best_value = score
            self.best_epoch = current_epoch
        if significant_improved:
            self.reference_value = score
            self.reference_epoch = current_epoch
            self.stale_epochs = 0
        elif current_epoch > self.minimum_epoch:
            self.stale_epochs += 1
        return EarlyStoppingUpdate(absolute_improved, significant_improved, self.should_stop)


def warmup_cosine_lr(step: int, total_steps: int, warmup_steps: int, initial_lr: float, min_lr: float) -> float:
    if total_steps < 1:
        raise ValueError("total_steps must be at least 1")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if step < 0:
        raise ValueError("step must be non-negative")
    if initial_lr < 0 or min_lr < 0:
        raise ValueError("Learning rates must be non-negative")
    capped_step = min(int(step), int(total_steps) - 1)
    if capped_step >= total_steps - 1:
        return float(min_lr)
    if warmup_steps > 0 and capped_step < warmup_steps:
        return float(initial_lr) * float(capped_step + 1) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return float(initial_lr)
    decay_steps = total_steps - warmup_steps - 1
    if decay_steps <= 0:
        return float(initial_lr)
    progress = (capped_step - warmup_steps) / float(decay_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr + (initial_lr - min_lr) * cosine)


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")


def _atomic_write_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _atomic_backup_existing(source: Path, destination: Path) -> Path:
    with source.open("rb") as handle:
        payload = handle.read()
    return _atomic_write_bytes(destination, payload)


def atomic_write_json(path: Path | str, payload: object, *, indent: int = 2, previous_path: Path | str | None = None) -> Path:
    destination = Path(path)
    if previous_path is not None and destination.exists():
        _atomic_backup_existing(destination, Path(previous_path))
    data = json.dumps(payload, indent=indent, ensure_ascii=False).encode("utf-8")
    return _atomic_write_bytes(destination, data)


def atomic_torch_save(path: Path | str, payload: object, *, previous_path: Path | str | None = None) -> Path:
    torch = _import_torch()
    destination = Path(path)
    if previous_path is not None and destination.exists():
        _atomic_backup_existing(destination, Path(previous_path))
    temporary = _temporary_sibling(destination)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def capture_rng_state(data_loader_generator=None) -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        torch = _import_torch()
    except RuntimeError:
        return state
    state["torch_cpu"] = torch.get_rng_state()
    state["torch_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    if data_loader_generator is not None:
        state["data_loader_generator"] = data_loader_generator.get_state()
    return state


def restore_rng_state(state: dict[str, object], data_loader_generator=None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    try:
        torch = _import_torch()
    except RuntimeError:
        return
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if data_loader_generator is not None and "data_loader_generator" in state:
        data_loader_generator.set_state(state["data_loader_generator"])


def hash_file_bytes(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_path_contents(root: Path | str) -> str:
    root_path = Path(root)
    if root_path.is_file():
        return hash_file_bytes(root_path)
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root_path.rglob("*") if candidate.is_file() and "__pycache__" not in candidate.parts):
        digest.update(path.relative_to(root_path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _ensure_finite(value: object, label: str) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _ensure_finite(nested, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _ensure_finite(nested, f"{label}[{index}]")
        return
    if isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite value found in {label}")
        return
    if isinstance(value, Real) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite value found in {label}")


def _cpu_load(path: Path):
    torch = _import_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        return torch.load(path, map_location="cpu")


def certify_checkpoint(path: Path | str, build_model, smoke_inference):
    checkpoint_path = Path(path)
    checkpoint = _cpu_load(checkpoint_path)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a mapping, got {type(checkpoint)!r}")
    if "state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint is missing state_dict: {checkpoint_path}")
    if "config" not in checkpoint:
        raise KeyError(f"Checkpoint is missing config: {checkpoint_path}")
    config = checkpoint["config"]
    score = checkpoint.get("validation", {}).get("selection_score", checkpoint.get("best_monitor"))
    if score is None:
        raise KeyError(f"Checkpoint is missing a selection score: {checkpoint_path}")
    _ensure_finite(config, "config")
    _ensure_finite(score, "selection score")
    model = build_model(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.cpu()
    model.eval()
    smoke_inference(model, checkpoint, config)
    return checkpoint
