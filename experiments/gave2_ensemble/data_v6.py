from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
from skimage.registration import phase_cross_correlation

from .data import GAVE2CaseSample, derive_av3_target, list_case_ids, read_png_float, to_chw
from .preprocessing_v6 import build_vessel_enhanced_channel, preprocess_v6_modalities


def _clip01(array: np.ndarray) -> np.ndarray:
    return np.clip(array.astype(np.float32, copy=False), 0.0, 1.0)


def _normalize_single_channel(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return _clip01(array)
    if array.ndim == 3 and array.shape[2] == 1:
        return _clip01(array[..., 0])
    raise ValueError(f"Expected a single-channel image, got shape {array.shape}")


def _roi_mask(roi: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    roi_channel = _normalize_single_channel(roi)
    if roi_channel.shape != shape:
        raise ValueError(f"ROI shape {roi_channel.shape} does not match image shape {shape}")
    return roi_channel > 0.5


def _hash_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array.astype(np.float32, copy=False))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _save_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _save_overlay(path: Path, reference: np.ndarray, aligned: np.ndarray, roi_mask: np.ndarray) -> None:
    masked_ref = np.where(roi_mask, _clip01(reference), 0.0)
    masked_aligned = np.where(roi_mask, _clip01(aligned), 0.0)
    overlay = np.stack([masked_ref, masked_aligned, masked_ref], axis=2)
    Image.fromarray(np.round(overlay * 255.0).astype(np.uint8), mode="RGB").save(path)


def _shift_zero_fill(array: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    out = np.zeros_like(array)
    height, width = array.shape[:2]
    src_y0 = max(0, -shift_y)
    src_y1 = min(height, height - shift_y) if shift_y >= 0 else height
    dst_y0 = max(0, shift_y)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
    src_x0 = max(0, -shift_x)
    src_x1 = min(width, width - shift_x) if shift_x >= 0 else width
    dst_x0 = max(0, shift_x)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
    if dst_y1 > dst_y0 and dst_x1 > dst_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return out


def _cache_paths(cache_dir: Path, case_id: str, modality: str) -> tuple[Path, Path]:
    return (
        cache_dir / f"{case_id}_{modality}_transform.json",
        cache_dir / f"{case_id}_{modality}_qa.png",
    )


def _max_shift_pixels(shape: tuple[int, int], max_shift_ratio: float) -> tuple[int, int]:
    return (
        max(1, int(np.floor(shape[0] * max_shift_ratio))),
        max(1, int(np.floor(shape[1] * max_shift_ratio))),
    )


def _shift_is_plausible(shift: Sequence[int], shape: tuple[int, int], max_shift_ratio: float) -> bool:
    max_shift_y, max_shift_x = _max_shift_pixels(shape, max_shift_ratio)
    return abs(int(shift[0])) <= max_shift_y and abs(int(shift[1])) <= max_shift_x


def _registration_reference(cfp_rgb: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    reference = build_vessel_enhanced_channel(cfp_rgb)
    return np.where(roi_mask, reference, 0.0).astype(np.float32, copy=False)


def _estimate_translation(
    reference: np.ndarray,
    moving: np.ndarray,
    max_shift_ratio: float,
) -> tuple[list[int], bool, str]:
    if not np.isfinite(reference).all() or not np.isfinite(moving).all():
        return [0, 0], True, "non-finite intensities"
    if float(reference.std()) < 1e-6 or float(moving.std()) < 1e-6:
        return [0, 0], True, "flat intensities"
    shift, _, _ = phase_cross_correlation(reference, moving, upsample_factor=1)
    if not np.isfinite(shift).all():
        return [0, 0], True, "non-finite transform"
    shift_y = int(np.rint(float(shift[0])))
    shift_x = int(np.rint(float(shift[1])))
    if not _shift_is_plausible((shift_y, shift_x), reference.shape, max_shift_ratio):
        return [0, 0], True, f"implausible shift {shift_y},{shift_x}"
    return [shift_y, shift_x], False, "ok"


def register_ffa_to_cfp(
    cfp_rgb: np.ndarray,
    ffa_image: np.ndarray,
    roi: np.ndarray,
    *,
    cache_dir: Path | str | None = None,
    case_id: str = "case",
    modality: str = "ffa",
    max_shift_ratio: float = 0.25,
) -> tuple[np.ndarray, dict[str, object]]:
    """Register one FFA image to CFP using deterministic integer translation."""

    cfp_rgb = _clip01(cfp_rgb)
    moving = _normalize_single_channel(ffa_image)
    roi_mask = _roi_mask(roi, cfp_rgb.shape[:2])
    reference = _registration_reference(cfp_rgb, roi_mask)

    hashes = {
        "reference_sha256": _hash_array(reference),
        "moving_sha256": _hash_array(moving),
        "roi_sha256": _hash_array(roi_mask.astype(np.float32)),
    }

    metadata_path = None
    overlay_path = None
    if cache_dir is not None:
        metadata_path, overlay_path = _cache_paths(Path(cache_dir), case_id, modality)
        if metadata_path.exists():
            cached = json.loads(metadata_path.read_text(encoding="utf-8"))
            if all(cached.get(key) == value for key, value in hashes.items()):
                cached_shift = [int(cached["shift"][0]), int(cached["shift"][1])]
                cached_ratio = float(cached.get("max_shift_ratio", -1.0))
                if cached_ratio == float(max_shift_ratio) and _shift_is_plausible(cached_shift, reference.shape, max_shift_ratio):
                    aligned = _shift_zero_fill(moving, cached_shift[0], cached_shift[1])[..., None]
                    cached["cache_hit"] = True
                    if overlay_path is not None and not overlay_path.exists():
                        _save_overlay(overlay_path, reference, aligned[..., 0], roi_mask)
                    return aligned.astype(np.float32, copy=False), cached

    shift, identity_fallback, reason = _estimate_translation(reference, moving, max_shift_ratio)
    aligned = _shift_zero_fill(moving, shift[0], shift[1])[..., None]
    metadata = {
        "case_id": case_id,
        "modality": modality,
        "method": "phase_cross_correlation_integer_translation",
        "shift": shift,
        "max_shift_ratio": float(max_shift_ratio),
        "identity_fallback": identity_fallback,
        "reason": reason,
        "cache_hit": False,
        **hashes,
    }
    if metadata_path is not None and overlay_path is not None:
        _save_json(metadata_path, metadata)
        _save_overlay(overlay_path, reference, aligned[..., 0], roi_mask)
    return aligned.astype(np.float32, copy=False), metadata


def _record_vector(record: Mapping[str, object]) -> np.ndarray:
    laterality = str(record["laterality"]).upper()
    if laterality not in {"L", "R"}:
        raise ValueError(f"Unsupported laterality {record['laterality']!r}")
    return np.asarray(
        [
            float(record["artery_prevalence"]),
            float(record["vessel_prevalence"]),
            float(record["vein_prevalence"]),
            1.0 if laterality == "L" else 0.0,
        ],
        dtype=np.float32,
    )


def build_balanced_three_fold_manifest(
    records: Sequence[Mapping[str, object]],
    seed: int = 77,
) -> dict[str, object]:
    if seed != 77:
        raise ValueError("V6 requires seed 77")
    if len(records) != 50:
        raise ValueError(f"Expected 50 training records, got {len(records)}")

    normalized_records = [dict(record) for record in records]
    case_ids = [str(record["case_id"]) for record in normalized_records]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case_id values must be unique")

    vectors = np.stack([_record_vector(record) for record in normalized_records], axis=0)
    global_mean = vectors.mean(axis=0)
    std = np.maximum(vectors.std(axis=0), 1e-6)
    rng = np.random.default_rng(seed)
    noise = rng.random(len(normalized_records)) * 1e-6
    order = sorted(
        range(len(normalized_records)),
        key=lambda idx: (
            -float(np.linalg.norm((vectors[idx] - global_mean) / std)),
            float(noise[idx]),
            case_ids[idx],
        ),
    )

    target_sizes = [17, 17, 16]
    fold_records: list[list[dict[str, object]]] = [[] for _ in target_sizes]
    fold_sums = [np.zeros(vectors.shape[1], dtype=np.float64) for _ in target_sizes]

    for idx in order:
        record = normalized_records[idx]
        vector = vectors[idx].astype(np.float64)
        best_fold = None
        best_score = None
        for fold_index, target_size in enumerate(target_sizes):
            if len(fold_records[fold_index]) >= target_size:
                continue
            candidate_sum = fold_sums[fold_index] + vector
            target_sum = global_mean.astype(np.float64) * float(target_size)
            residual = candidate_sum - target_sum
            score = float(np.dot(residual, residual))
            if best_score is None or score < best_score - 1e-12 or (
                abs(score - best_score) <= 1e-12 and fold_index < int(best_fold)
            ):
                best_score = score
                best_fold = fold_index
        assert best_fold is not None
        fold_records[best_fold].append(record)
        fold_sums[best_fold] += vector

    sorted_all_ids = sorted(case_ids)
    folds = []
    for fold_index, records_in_fold in enumerate(fold_records):
        validation = sorted(str(record["case_id"]) for record in records_in_fold)
        validation_set = set(validation)
        training = [case_id for case_id in sorted_all_ids if case_id not in validation_set]
        folds.append(
            {
                "fold_index": fold_index,
                "training": training,
                "validation": validation,
                "validation_records": sorted(records_in_fold, key=lambda record: str(record["case_id"])),
            }
        )
    return {"seed": seed, "folds": folds}


class GAVE2DatasetV6:
    """Native-resolution V6 loader for Task 1 and Task 2."""

    def __init__(
        self,
        data_root: Path | str,
        split: str,
        task: str,
        case_ids: Iterable[str] | None = None,
        require_target: bool | None = None,
        registration_cache_dir: Path | str | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        task = task.lower()
        if task not in {"task1", "task2"}:
            raise ValueError(f"Unsupported task {task!r}; expected task1 or task2")
        self.task = task
        self.case_ids = list(case_ids) if case_ids is not None else list_case_ids(self.data_root, split)
        self.require_target = split == "training" if require_target is None else require_target
        default_cache_dir = self.data_root / split / "registration_cache"
        self.registration_cache_dir = Path(registration_cache_dir) if registration_cache_dir is not None else default_cache_dir

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> GAVE2CaseSample:
        case_id = self.case_ids[index]
        split_root = self.data_root / self.split
        cfp = read_png_float(split_root / "images" / f"{case_id}.png", channels=3)
        roi = read_png_float(split_root / "masks" / f"{case_id}.png", channels=1)
        height, width = cfp.shape[:2]
        if roi.shape[:2] != (height, width):
            raise ValueError(f"ROI mask shape mismatch for {case_id}: {roi.shape[:2]} vs {(height, width)}")

        if self.task == "task2":
            ffa_early = read_png_float(split_root / "FFA_A" / f"{case_id}.png", channels=1)
            ffa_late = read_png_float(split_root / "FFA_AV" / f"{case_id}.png", channels=1)
            registered_early, _ = register_ffa_to_cfp(
                cfp,
                ffa_early,
                roi,
                cache_dir=self.registration_cache_dir,
                case_id=case_id,
                modality="FFA_A",
            )
            registered_late, _ = register_ffa_to_cfp(
                cfp,
                ffa_late,
                roi,
                cache_dir=self.registration_cache_dir,
                case_id=case_id,
                modality="FFA_AV",
            )
            image = preprocess_v6_modalities(cfp, roi, registered_early, registered_late)
        else:
            image = preprocess_v6_modalities(cfp, roi)

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
            original_size=(height, width),
            biomarker_path=biomarker_path,
        )
