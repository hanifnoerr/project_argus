from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from .preprocessing import PREPROCESS_MODES, preprocess_modalities


@dataclass(frozen=True)
class GAVE2CaseSample:
    case_id: str
    image: np.ndarray
    mask: np.ndarray
    target: np.ndarray | None
    original_size: tuple[int, int]
    biomarker_path: Path | None = None


def read_png_float(path: Path, channels: int | None = None) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    image = Image.open(path)
    if channels == 1:
        image = image.convert("L")
    elif channels == 3:
        image = image.convert("RGB")
    array = np.asarray(image).astype(np.float32)
    if array.ndim == 2:
        array = array[..., None]
    if array.max(initial=0) > 1.0:
        array /= 255.0
    return array


def to_chw(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"Expected HWC array, got shape {array.shape}")
    return np.ascontiguousarray(array.transpose(2, 0, 1).astype(np.float32))


def derive_av3_target(raw_av_rgb: np.ndarray) -> np.ndarray:
    if raw_av_rgb.ndim != 3 or raw_av_rgb.shape[2] < 3:
        raise ValueError(f"Expected RGB AV label, got shape {raw_av_rgb.shape}")
    raw = raw_av_rgb[..., :3] > 0.5
    artery = np.logical_or(raw[..., 0], raw[..., 1])
    vessel = np.logical_or(np.logical_or(raw[..., 0], raw[..., 1]), raw[..., 2])
    vein = np.logical_or(raw[..., 2], raw[..., 1])
    return np.stack([artery, vessel, vein], axis=0).astype(np.float32)


def list_case_ids(data_root: Path | str, split: str = "training") -> list[str]:
    root = Path(data_root)
    image_dir = root / split / "images"
    if not image_dir.exists():
        raise FileNotFoundError(image_dir)
    return [path.stem for path in sorted(image_dir.glob("*.png"))]


def make_folds(case_ids: Sequence[str], n_folds: int = 5, seed: int = 77) -> list[dict[str, list[str]]]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    ids = list(case_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    chunks = [list(chunk) for chunk in np.array_split(ids, n_folds)]
    folds: list[dict[str, list[str]]] = []
    for i, val_ids in enumerate(chunks):
        val_set = set(val_ids)
        train_ids = [case_id for case_id in ids if case_id not in val_set]
        folds.append({"fold": [str(i)], "training": train_ids, "validation": list(val_ids)})
    return folds


class GAVE2Dataset:
    """Native-resolution GAVE2 loader for Task 1 and Task 2.

    The loader never crops or resizes. It returns CHW float32 arrays in [0, 1].
    Targets follow the challenge output convention:
    R/0 = artery, G/1 = all vessels, B/2 = vein.
    """

    def __init__(
        self,
        data_root: Path | str,
        split: str,
        task: str,
        case_ids: Iterable[str] | None = None,
        require_target: bool | None = None,
        preprocess: str = "none",
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        task = task.lower()
        if task not in {"task1", "task2"}:
            raise ValueError(f"Unsupported task {task!r}; expected task1 or task2")
        self.task = task
        preprocess = preprocess.lower()
        if preprocess not in PREPROCESS_MODES:
            raise ValueError(f"Unsupported preprocess mode {preprocess!r}; expected one of {PREPROCESS_MODES}")
        self.preprocess = preprocess
        self.case_ids = list(case_ids) if case_ids is not None else list_case_ids(self.data_root, split)
        self.require_target = split == "training" if require_target is None else require_target

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> GAVE2CaseSample:
        case_id = self.case_ids[index]
        split_root = self.data_root / self.split
        cfp = read_png_float(split_root / "images" / f"{case_id}.png", channels=3)
        roi = read_png_float(split_root / "masks" / f"{case_id}.png", channels=1)
        h, w = cfp.shape[:2]
        if roi.shape[:2] != (h, w):
            raise ValueError(f"ROI mask shape mismatch for {case_id}: {roi.shape[:2]} vs {(h, w)}")

        if self.task == "task2":
            ffa_early = read_png_float(split_root / "FFA_A" / f"{case_id}.png", channels=1)
            ffa_late = read_png_float(split_root / "FFA_AV" / f"{case_id}.png", channels=1)
            for name, img in (("FFA_A", ffa_early), ("FFA_AV", ffa_late)):
                if img.shape[:2] != (h, w):
                    raise ValueError(f"{name} shape mismatch for {case_id}: {img.shape[:2]} vs {(h, w)}")
            image = preprocess_modalities(cfp, roi, ffa_early, ffa_late, mode=self.preprocess)
        else:
            image = preprocess_modalities(cfp, roi, mode=self.preprocess)

        target = None
        biomarker_path = None
        if self.split == "training":
            biomarker_path = split_root / "biomarker" / f"{case_id}.txt"
            av_path = split_root / "av" / f"{case_id}.png"
            if av_path.exists():
                target = derive_av3_target(read_png_float(av_path, channels=3))
            elif self.require_target:
                raise FileNotFoundError(av_path)
        elif self.require_target:
            raise ValueError("Validation split has no public AV labels")

        return GAVE2CaseSample(
            case_id=case_id,
            image=to_chw(image),
            mask=to_chw(roi),
            target=target,
            original_size=(h, w),
            biomarker_path=biomarker_path,
        )


def torch_collate(samples: Sequence[GAVE2CaseSample]):
    import torch

    images = torch.from_numpy(np.stack([sample.image for sample in samples], axis=0))
    masks = torch.from_numpy(np.stack([sample.mask for sample in samples], axis=0))
    targets = None
    if samples[0].target is not None:
        targets = torch.from_numpy(np.stack([sample.target for sample in samples], axis=0))
    case_ids = [sample.case_id for sample in samples]
    sizes = [sample.original_size for sample in samples]
    return {"case_ids": case_ids, "images": images, "masks": masks, "targets": targets, "sizes": sizes}


def compute_roi_channel_stats(
    data_root: Path | str,
    task: str,
    case_ids: Iterable[str],
    preprocess: str = "none",
) -> dict[str, list[float]]:
    dataset = GAVE2Dataset(
        data_root=data_root,
        split="training",
        task=task,
        case_ids=case_ids,
        preprocess=preprocess,
    )
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
