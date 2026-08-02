from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from experiments.gave2_ensemble.data import derive_av3_target
from experiments.gave2_v8.store import ProbabilityStore

from .prepare import PreparedFFACache


@dataclass(frozen=True)
class V12Sample:
    case_id: str
    features: np.ndarray
    teacher: np.ndarray
    corridor: np.ndarray
    mask: np.ndarray
    target: np.ndarray | None


def _rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _roi(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127


def _green_contrast(cfp: np.ndarray, roi: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    green = cfp[..., 1]
    local = green - ndimage.gaussian_filter(green * roi, sigma=12.0)
    scale = max(float(np.std(local[roi])), 1e-3)
    return np.clip(local / (3.0 * scale), -1.0, 1.0).astype(np.float32)


def build_features(cfp: np.ndarray, teacher: np.ndarray, roi: np.ndarray, ffa: np.ndarray | None) -> np.ndarray:
    if teacher.shape != (3, *roi.shape) or cfp.shape != (*roi.shape, 3):
        raise ValueError(f"Incompatible full-canvas inputs: {cfp.shape}, {teacher.shape}, {roi.shape}")
    rgb = (2.0 * cfp.transpose(2, 0, 1) - 1.0) * roi[None]
    green = _green_contrast(cfp, roi)[None]
    teacher_features = (2.0 * np.clip(teacher, 0.0, 1.0) - 1.0) * roi[None]
    channels = [rgb, green, teacher_features, roi[None].astype(np.float32)]
    if ffa is not None:
        if ffa.shape != (5, *roi.shape):
            raise ValueError(f"Invalid FFA feature shape: {ffa.shape}")
        channels.append(ffa * roi[None])
    return np.concatenate(channels, axis=0).astype(np.float32)


def build_teacher_skeleton_corridor(teacher: np.ndarray, roi: np.ndarray, radius: int = 2) -> np.ndarray:
    from scipy import ndimage
    from skimage.morphology import skeletonize

    probability = np.asarray(teacher, dtype=np.float32)
    mask = np.asarray(roi, dtype=bool)
    if probability.shape != (3, *mask.shape):
        raise ValueError(f"Teacher/ROI shape mismatch: {probability.shape}, {mask.shape}")
    channels = []
    for channel in range(3):
        skeleton = skeletonize((probability[channel] >= 0.5) & mask)
        if radius > 0:
            skeleton = ndimage.binary_dilation(skeleton, iterations=radius)
        channels.append(skeleton & mask)
    return np.stack(channels, axis=0).astype(np.float32)


class V12Dataset:
    """Full-resolution residual dataset. No crop or resize is performed."""

    def __init__(
        self,
        data_root: Path | str,
        split: str,
        task: str,
        case_ids: Sequence[str],
        teacher_root: Path | str,
        prepared_root: Path | str | None = None,
        augment: bool = False,
        seed: int = 77,
        corridor_radius: int = 2,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.task = task.lower()
        if self.task not in {"task1", "task2"}:
            raise ValueError(self.task)
        self.case_ids = list(case_ids)
        self.teacher = ProbabilityStore(teacher_root, namespace="r2v2_direct", split=split)
        self.ffa = PreparedFFACache(prepared_root, split) if self.task == "task2" and prepared_root else None
        if self.task == "task2" and self.ffa is None:
            raise ValueError("Task 2 requires --prepared-root")
        self.augment = bool(augment)
        self.seed = int(seed)
        self.corridor_radius = int(corridor_radius)
        self._corridor_cache: dict[str, np.ndarray] = {}
        self.epoch = 0

    @property
    def input_channels(self) -> int:
        return 13 if self.task == "task2" else 8

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> V12Sample:
        case_id = self.case_ids[index]
        root = self.data_root / self.split
        cfp = _rgb(root / "images" / f"{case_id}.png")
        roi = _roi(root / "masks" / f"{case_id}.png")
        teacher = self.teacher.read_case(case_id)
        corridor = self._corridor_cache.get(case_id)
        if corridor is None:
            corridor = build_teacher_skeleton_corridor(teacher, roi, radius=self.corridor_radius)
            self._corridor_cache[case_id] = corridor
        ffa = self.ffa.read_case(case_id) if self.ffa is not None else None
        features = build_features(cfp, teacher, roi, ffa)
        target = None
        label_path = root / "av" / f"{case_id}.png"
        if label_path.exists():
            target = derive_av3_target(_rgb(label_path))
        if self.augment:
            rng = np.random.default_rng(self.seed + self.epoch * 1009 + index * 9176)
            flip_y, flip_x = bool(rng.integers(2)), bool(rng.integers(2))
            axes = tuple(axis for flag, axis in ((flip_y, -2), (flip_x, -1)) if flag)
            if axes:
                features = np.flip(features, axis=axes).copy()
                teacher = np.flip(teacher, axis=axes).copy()
                corridor = np.flip(corridor, axis=axes).copy()
                roi = np.flip(roi, axis=tuple(axis + 2 for axis in axes)).copy()
                if target is not None:
                    target = np.flip(target, axis=axes).copy()
        return V12Sample(
            case_id=case_id,
            features=np.ascontiguousarray(features),
            teacher=np.ascontiguousarray(teacher.astype(np.float32)),
            corridor=np.ascontiguousarray(corridor.astype(np.float32)),
            mask=np.ascontiguousarray(roi[None].astype(np.float32)),
            target=None if target is None else np.ascontiguousarray(target.astype(np.float32)),
        )


def collate(samples: Sequence[V12Sample]) -> dict[str, object]:
    import torch

    return {
        "case_ids": [sample.case_id for sample in samples],
        "features": torch.from_numpy(np.stack([sample.features for sample in samples])),
        "teacher": torch.from_numpy(np.stack([sample.teacher for sample in samples])),
        "corridor": torch.from_numpy(np.stack([sample.corridor for sample in samples])),
        "mask": torch.from_numpy(np.stack([sample.mask for sample in samples])),
        "target": None
        if samples[0].target is None
        else torch.from_numpy(np.stack([sample.target for sample in samples])),
    }
