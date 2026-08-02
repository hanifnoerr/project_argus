from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from experiments.gave2_ensemble.data import derive_av3_target
from experiments.gave2_v8.store import ProbabilityStore
from experiments.gave2_v12.data import build_features
from experiments.gave2_v12.prepare import PreparedFFACache


@dataclass(frozen=True)
class V13Sample:
    case_id: str
    features: np.ndarray
    teacher: np.ndarray
    mask: np.ndarray
    target: np.ndarray | None
    state_target: np.ndarray | None
    centerline: np.ndarray | None


def _rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _roi(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127


def state_target(target: np.ndarray) -> np.ndarray:
    if target.ndim != 3 or target.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W] target, got {target.shape}")
    artery = target[0] > 0.5
    vessel = target[1] > 0.5
    vein = target[2] > 0.5
    result = np.zeros(vessel.shape, dtype=np.int64)
    result[vessel & artery & ~vein] = 1
    result[vessel & vein & ~artery] = 2
    result[vessel & artery & vein] = 3
    result[vessel & ~artery & ~vein] = 4
    return result


def target_centerlines(target: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize

    return np.stack(
        (skeletonize(target[0] > 0.5), skeletonize(target[2] > 0.5)),
        axis=0,
    ).astype(np.float32)


def _photometric(cfp: np.ndarray, ffa: np.ndarray | None, rng: np.random.Generator):
    gamma = float(rng.uniform(0.88, 1.12))
    gain = rng.uniform(0.92, 1.08, size=(1, 1, 3)).astype(np.float32)
    offset = float(rng.uniform(-0.025, 0.025))
    cfp = np.clip(np.power(np.clip(cfp, 0.0, 1.0), gamma) * gain + offset, 0.0, 1.0)
    if ffa is not None:
        ffa_gain = float(rng.uniform(0.92, 1.08))
        ffa = np.clip(ffa * ffa_gain, -3.0, 3.0)
    return cfp.astype(np.float32), None if ffa is None else ffa.astype(np.float32)


class V13Dataset:
    """Native-canvas V13 dataset with safe flips and photometric augmentation."""

    def __init__(
        self,
        data_root: Path | str,
        split: str,
        task: str,
        case_ids: Sequence[str],
        teacher_root: Path | str,
        prepared_root: Path | str | None = None,
        *,
        augment: bool = False,
        seed: int = 77,
        include_targets: bool = True,
        preload_targets: bool = False,
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
        self.include_targets = bool(include_targets)
        self.epoch = 0
        self._target_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if preload_targets and self.include_targets:
            for case_id in self.case_ids:
                path = self.data_root / self.split / "av" / f"{case_id}.png"
                if path.exists():
                    self._target_cache[case_id] = self._read_target(path)

    @staticmethod
    def _read_target(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        target = derive_av3_target(_rgb(path)).astype(np.float32)
        return (
            (target > 0.5).astype(np.uint8),
            state_target(target).astype(np.uint8),
            target_centerlines(target).astype(np.uint8),
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> V13Sample:
        case_id = self.case_ids[index]
        root = self.data_root / self.split
        cfp = _rgb(root / "images" / f"{case_id}.png")
        roi = _roi(root / "masks" / f"{case_id}.png")
        teacher = self.teacher.read_case(case_id).astype(np.float32)
        ffa = self.ffa.read_case(case_id) if self.ffa is not None else None

        target = states = centerline = None
        label_path = root / "av" / f"{case_id}.png"
        if self.include_targets and label_path.exists():
            cached = self._target_cache.get(case_id)
            if cached is None:
                cached = self._read_target(label_path)
                self._target_cache[case_id] = cached
            target = cached[0].astype(np.float32, copy=True)
            states = cached[1].astype(np.int64, copy=True)
            centerline = cached[2].astype(np.float32, copy=True)

        rng = np.random.default_rng(self.seed + self.epoch * 1009 + index * 9176)
        if self.augment:
            cfp, ffa = _photometric(cfp, ffa, rng)
        features = build_features(cfp, teacher, roi, ffa)

        if self.augment:
            axes = tuple(axis for flag, axis in ((bool(rng.integers(2)), -2), (bool(rng.integers(2)), -1)) if flag)
            if axes:
                features = np.flip(features, axis=axes).copy()
                teacher = np.flip(teacher, axis=axes).copy()
                roi = np.flip(roi, axis=tuple(axis + 2 for axis in axes)).copy()
                if target is not None and states is not None and centerline is not None:
                    target = np.flip(target, axis=axes).copy()
                    states = np.flip(states, axis=tuple(axis + 2 for axis in axes)).copy()
                    centerline = np.flip(centerline, axis=axes).copy()

        return V13Sample(
            case_id=case_id,
            features=np.ascontiguousarray(features, dtype=np.float32),
            teacher=np.ascontiguousarray(teacher, dtype=np.float32),
            mask=np.ascontiguousarray(roi[None], dtype=np.float32),
            target=None if target is None else np.ascontiguousarray(target, dtype=np.float32),
            state_target=None if states is None else np.ascontiguousarray(states, dtype=np.int64),
            centerline=None if centerline is None else np.ascontiguousarray(centerline, dtype=np.float32),
        )


def collate(samples: Sequence[V13Sample]) -> dict[str, object]:
    import torch

    labelled = samples[0].target is not None
    return {
        "case_ids": [sample.case_id for sample in samples],
        "features": torch.from_numpy(np.stack([sample.features for sample in samples])),
        "teacher": torch.from_numpy(np.stack([sample.teacher for sample in samples])),
        "mask": torch.from_numpy(np.stack([sample.mask for sample in samples])),
        "target": torch.from_numpy(np.stack([sample.target for sample in samples])) if labelled else None,
        "state_target": torch.from_numpy(np.stack([sample.state_target for sample in samples])) if labelled else None,
        "centerline": torch.from_numpy(np.stack([sample.centerline for sample in samples])) if labelled else None,
    }
