from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RegistrationQA:
    accepted: bool
    model: str
    matches: int
    inliers: int
    inlier_ratio: float
    median_error: float
    p95_error: float
    scale_min: float
    scale_max: float
    determinant: float
    translation_fraction: float
    coverage: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def robust_zero_one(image: np.ndarray, roi: np.ndarray | None = None) -> np.ndarray:
    value = np.asarray(image, dtype=np.float32)
    if value.ndim == 3:
        value = value[..., 0]
    selected = value[np.asarray(roi, dtype=bool)] if roi is not None else value.reshape(-1)
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return np.zeros_like(value, dtype=np.float32)
    low, high = np.percentile(selected, (1.0, 99.0))
    if high <= low + 1e-6:
        return np.zeros_like(value, dtype=np.float32)
    return np.clip((value - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _coverage(points: np.ndarray, shape_hw: tuple[int, int]) -> float:
    if len(points) < 2:
        return 0.0
    span = np.ptp(points, axis=0)
    height, width = shape_hw
    return float(np.clip((span[0] / max(width, 1)) * (span[1] / max(height, 1)), 0.0, 1.0))


def _matrix_diagnostics(matrix: np.ndarray, shape_hw: tuple[int, int]) -> tuple[float, float, float, float]:
    linear = np.asarray(matrix, dtype=np.float64)[:2, :2]
    singular = np.linalg.svd(linear, compute_uv=False)
    determinant = float(np.linalg.det(linear))
    height, width = shape_hw
    diagonal = math.hypot(height, width)
    translation = float(np.linalg.norm(np.asarray(matrix)[:2, 2]) / max(diagonal, 1.0))
    return float(singular.min()), float(singular.max()), determinant, translation


def _failed_qa(matches: int, reason: str) -> RegistrationQA:
    return RegistrationQA(
        accepted=False,
        model="identity",
        matches=int(matches),
        inliers=0,
        inlier_ratio=0.0,
        median_error=float("inf"),
        p95_error=float("inf"),
        scale_min=1.0,
        scale_max=1.0,
        determinant=1.0,
        translation_fraction=0.0,
        coverage=0.0,
        reason=reason,
    )


def fit_registration(
    moving_xy: np.ndarray,
    fixed_xy: np.ndarray,
    confidence: np.ndarray | None,
    shape_hw: tuple[int, int],
    *,
    seed: int = 77,
) -> tuple[np.ndarray, RegistrationQA]:
    """Fit the simplest MINIMA transform that passes conservative retinal QA.

    Returned matrices map FFA (moving) pixel coordinates into CFP (fixed)
    pixel coordinates. Rejected fits return identity and cannot alter data.
    """

    from skimage.measure import ransac
    from skimage.transform import AffineTransform, ProjectiveTransform, SimilarityTransform

    moving = np.asarray(moving_xy, dtype=np.float64).reshape(-1, 2)
    fixed = np.asarray(fixed_xy, dtype=np.float64).reshape(-1, 2)
    if moving.shape != fixed.shape:
        raise ValueError(f"Point shape mismatch: {moving.shape} vs {fixed.shape}")
    finite = np.isfinite(moving).all(axis=1) & np.isfinite(fixed).all(axis=1)
    if confidence is not None:
        scores = np.asarray(confidence, dtype=np.float64).reshape(-1)
        if len(scores) != len(moving):
            raise ValueError("Confidence length does not match correspondences")
        finite &= np.isfinite(scores)
        if finite.any():
            cutoff = max(0.05, float(np.quantile(scores[finite], 0.25)))
            finite &= scores >= cutoff
    moving, fixed = moving[finite], fixed[finite]
    if len(moving) < 12:
        return np.eye(3, dtype=np.float64), _failed_qa(len(moving), "fewer than 12 usable matches")

    candidates = (
        ("similarity", SimilarityTransform, 2),
        ("affine", AffineTransform, 3),
        ("projective", ProjectiveTransform, 4),
    )
    rng = np.random.default_rng(seed)
    for name, transform_class, minimum_samples in candidates:
        try:
            model, inliers = ransac(
                (moving, fixed),
                transform_class,
                min_samples=minimum_samples,
                residual_threshold=4.0,
                max_trials=3000,
                rng=rng,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        if model is None or inliers is None or not np.asarray(inliers).any():
            continue
        matrix = np.asarray(model.params, dtype=np.float64)
        projected = model(moving)
        errors = np.linalg.norm(projected - fixed, axis=1)
        selected = errors[np.asarray(inliers, dtype=bool)]
        inlier_count = int(np.count_nonzero(inliers))
        ratio = inlier_count / len(moving)
        median_error = float(np.median(selected))
        p95_error = float(np.percentile(selected, 95.0))
        scale_min, scale_max, determinant, translation = _matrix_diagnostics(matrix, shape_hw)
        coverage = min(_coverage(moving[inliers], shape_hw), _coverage(fixed[inliers], shape_hw))
        perspective_ok = name != "projective" or float(np.linalg.norm(matrix[2, :2])) < 2e-3
        accepted = (
            inlier_count >= 18
            and ratio >= 0.25
            and median_error <= 3.5
            and p95_error <= 8.0
            and 0.72 <= scale_min <= scale_max <= 1.38
            and 0.60 <= abs(determinant) <= 1.60
            and translation <= 0.28
            and coverage >= 0.035
            and perspective_ok
        )
        qa = RegistrationQA(
            accepted=accepted,
            model=name,
            matches=len(moving),
            inliers=inlier_count,
            inlier_ratio=float(ratio),
            median_error=median_error,
            p95_error=p95_error,
            scale_min=scale_min,
            scale_max=scale_max,
            determinant=determinant,
            translation_fraction=translation,
            coverage=coverage,
            reason="accepted" if accepted else "geometric QA rejected fit",
        )
        if accepted:
            return matrix, qa
    return np.eye(3, dtype=np.float64), _failed_qa(len(moving), "no candidate transform passed QA")


def warp_to_reference(image: np.ndarray, matrix_moving_to_fixed: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    from skimage.transform import ProjectiveTransform, warp

    value = np.asarray(image, dtype=np.float32)
    transform = ProjectiveTransform(matrix=np.asarray(matrix_moving_to_fixed, dtype=np.float64))
    warped = warp(
        value,
        inverse_map=transform.inverse,
        output_shape=shape_hw,
        order=1,
        mode="constant",
        cval=0.0,
        preserve_range=True,
    )
    return np.asarray(warped, dtype=np.float32)


def ffa_feature_stack(early: np.ndarray, late: np.ndarray, roi: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    mask = np.asarray(roi, dtype=bool)
    early01 = robust_zero_one(early, mask) * mask
    late01 = robust_zero_one(late, mask) * mask
    difference = np.clip(late01 - early01, -1.0, 1.0)
    early_local = early01 - ndimage.gaussian_filter(early01, sigma=9.0)
    late_local = late01 - ndimage.gaussian_filter(late01, sigma=9.0)
    scale = max(float(np.std(np.concatenate((early_local[mask], late_local[mask])))), 1e-3)
    artery_cue = np.clip(early_local / (3.0 * scale), -1.0, 1.0)
    vein_cue = np.clip(late_local / (3.0 * scale), -1.0, 1.0)
    return np.stack((early01, late01, difference, artery_cue, vein_cue), axis=0).astype(np.float32)


def load_matches(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as payload:
        moving = np.asarray(payload["moving_xy"], dtype=np.float32)
        fixed = np.asarray(payload["fixed_xy"], dtype=np.float32)
        confidence = np.asarray(payload["confidence"], dtype=np.float32) if "confidence" in payload else None
    return moving, fixed, confidence

