from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

FEATURE_SCHEMA_VERSION = 11
THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)


def zone_c_mask(
    shape_hw: tuple[int, int],
    center_xy: tuple[float, float],
    disc_diameter: float,
    roi_hw: np.ndarray | None = None,
) -> np.ndarray:
    h, w = shape_hw
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    distance = np.sqrt(np.square(x - center_xy[0]) + np.square(y - center_xy[1]))
    zone = (distance >= 1.5 * disc_diameter) & (distance <= 2.5 * disc_diameter)
    if roi_hw is not None:
        roi = np.asarray(roi_hw)
        if roi.shape != (h, w):
            raise ValueError(f"ROI shape {roi.shape} does not match {(h, w)}")
        zone &= roi > 0.5
    return zone


def skeleton_box_dimension(binary_hw: np.ndarray) -> float:
    binary = np.asarray(binary_hw, dtype=bool)
    if not binary.any():
        return 0.0
    skeleton = skeletonize(binary)
    h, w = skeleton.shape
    sizes = [2**power for power in range(1, int(math.log2(min(h, w))) - 1)]
    x_values: list[float] = []
    y_values: list[float] = []
    for size in sizes:
        h_pad = int(math.ceil(h / size) * size)
        w_pad = int(math.ceil(w / size) * size)
        padded = np.zeros((h_pad, w_pad), dtype=bool)
        padded[:h, :w] = skeleton
        blocks = padded.reshape(h_pad // size, size, w_pad // size, size)
        count = int(np.count_nonzero(blocks.any(axis=(1, 3))))
        if count > 0:
            x_values.append(math.log(1.0 / size))
            y_values.append(math.log(float(count)))
    return float(np.polyfit(x_values, y_values, 1)[0]) if len(x_values) >= 2 else 0.0


def knudtson_equivalent(widths: list[float] | np.ndarray, is_artery: bool) -> float:
    values = sorted(
        (float(value) for value in widths if value > 0 and math.isfinite(value)), reverse=True
    )[:6]
    if not values:
        return 0.0
    coefficient = 0.88 if is_artery else 0.95
    while len(values) > 1:
        values = sorted(values, reverse=True)
        combined = [
            coefficient * math.sqrt(values[index] ** 2 + values[-index - 1] ** 2)
            for index in range(len(values) // 2)
        ]
        if len(values) % 2:
            combined.append(values[len(values) // 2])
        values = combined
    return float(values[0])


def locate_optic_disc(
    cfp_rgb: np.ndarray,
    vessel_probability_hw: np.ndarray,
    roi_hw: np.ndarray,
) -> tuple[tuple[float, float], float, float]:
    cfp = np.asarray(cfp_rgb, dtype=np.float32)
    roi = np.asarray(roi_hw) > 0.5
    if cfp.ndim != 3 or cfp.shape[2] != 3 or roi.shape != cfp.shape[:2]:
        raise ValueError("Invalid CFP or ROI shape for optic-disc localization")
    if int(roi.sum()) < 100:
        raise ValueError("ROI is too small for optic-disc localization")
    luminance = 0.30 * cfp[..., 0] + 0.60 * cfp[..., 1] + 0.10 * cfp[..., 2]
    brightness = ndimage.gaussian_filter(luminance, sigma=18.0)
    convergence = ndimage.gaussian_filter(np.asarray(vessel_probability_hw, dtype=np.float32), sigma=22.0)
    distance_to_edge = ndimage.distance_transform_edt(roi)
    candidate = roi & (distance_to_edge >= max(20.0, min(roi.shape) * 0.035))
    if not candidate.any():
        candidate = roi

    def standardized(value: np.ndarray) -> np.ndarray:
        selected = value[candidate]
        return (value - float(np.median(selected))) / max(float(np.std(selected)), 1e-6)

    saliency = 0.75 * standardized(brightness) + 0.25 * standardized(convergence)
    saliency[~candidate] = -np.inf
    center_y, center_x = np.unravel_index(int(np.argmax(saliency)), saliency.shape)
    confidence = float(
        np.clip((saliency[center_y, center_x] - np.median(saliency[candidate])) / 8.0, 0.0, 1.0)
    )
    y, x = np.ogrid[: roi.shape[0], : roi.shape[1]]
    local = ((x - center_x) ** 2 + (y - center_y) ** 2 <= 110**2) & roi
    local_values = brightness[local]
    threshold = (
        float(np.percentile(local_values, 68.0))
        if local_values.size
        else float(brightness[center_y, center_x])
    )
    disc_candidate = local & (brightness >= threshold)
    labels, _ = ndimage.label(disc_candidate)
    label = int(labels[center_y, center_x])
    diameter = (
        2.0 * math.sqrt(int(np.count_nonzero(labels == label)) / math.pi)
        if label > 0
        else 120.0
    )
    return (float(center_x), float(center_y)), float(np.clip(diameter, 70.0, 180.0)), confidence


def quantize_probability(probability_chw: np.ndarray) -> np.ndarray:
    """Match the round-to-uint8 conversion used by the V8 submission writer."""

    value = np.asarray(probability_chw, dtype=np.float32)
    if value.ndim != 3 or value.shape[0] != 3 or not np.isfinite(value).all():
        raise ValueError(f"Expected finite [3,H,W] probability, got {value.shape}")
    return np.round(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8).astype(np.float32) / 255.0


def load_rgb_float(path: Path | str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_roi(path: Path | str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127


def load_probability_png(path: Path | str) -> np.ndarray:
    return load_rgb_float(path).transpose(2, 0, 1)


def _append(values: list[float], names: list[str], name: str, value: float) -> None:
    names.append(name)
    values.append(float(value))


def _safe_log_ratio(numerator: float, denominator: float) -> float:
    return float(math.log(max(numerator, 1e-8) / max(denominator, 1e-8)))


def _segment_widths(binary_hw: np.ndarray) -> list[float]:
    """Measure vessel widths on branch-free skeleton segments."""

    binary = np.asarray(binary_hw, dtype=bool)
    if not binary.any():
        return []
    skeleton = skeletonize(binary)
    neighbors = ndimage.convolve(
        skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant"
    ) - skeleton.astype(np.uint8)
    junctions = skeleton & (neighbors != 2)
    branch_free = skeleton & ~ndimage.binary_dilation(junctions, iterations=1)
    segments, count = ndimage.label(branch_free, structure=np.ones((3, 3), dtype=np.uint8))
    distance = ndimage.distance_transform_edt(binary)
    widths: list[float] = []
    for index in range(1, count + 1):
        selected = 2.0 * distance[segments == index]
        if selected.size >= 8:
            widths.append(float(np.percentile(selected, 75.0)))
    return sorted((value for value in widths if value > 0 and math.isfinite(value)), reverse=True)


def _width_features(binary_zone: np.ndarray, *, artery: bool) -> dict[str, float]:
    widths = _segment_widths(binary_zone)
    top = widths[:6]
    padded = top + [0.0] * (6 - len(top))
    return {
        "segment_count": float(len(widths)),
        "width_p50": float(np.median(widths)) if widths else 0.0,
        "width_p90": float(np.percentile(widths, 90.0)) if widths else 0.0,
        "width_max": float(widths[0]) if widths else 0.0,
        **{f"top_width_{index + 1}": value for index, value in enumerate(padded)},
        "knudtson": knudtson_equivalent(top, is_artery=artery),
    }


def extract_task3_features(
    probability_chw: np.ndarray,
    cfp_rgb: np.ndarray,
    roi_hw: np.ndarray,
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    """Extract compact, interpretable biomarker predictors from a full canvas."""

    probability = quantize_probability(probability_chw)
    cfp = np.asarray(cfp_rgb, dtype=np.float32)
    roi = np.asarray(roi_hw, dtype=bool)
    if probability.shape != (3, *roi.shape):
        raise ValueError(f"Probability shape {probability.shape} does not match ROI {roi.shape}")
    if cfp.shape != (*roi.shape, 3):
        raise ValueError(f"CFP shape {cfp.shape} does not match ROI {roi.shape}")
    if int(roi.sum()) < 100:
        raise ValueError("ROI is empty")

    center, diameter, confidence = locate_optic_disc(cfp, probability[1], roi)
    zone = zone_c_mask(roi.shape, center, diameter, roi)
    if int(zone.sum()) < 100:
        raise ValueError("Estimated Zone C is too small")

    values: list[float] = []
    names: list[str] = []
    _append(values, names, "od_x", center[0] / roi.shape[1])
    _append(values, names, "od_y", center[1] / roi.shape[0])
    _append(values, names, "od_diameter", diameter / min(roi.shape))
    _append(values, names, "od_confidence", confidence)
    _append(values, names, "roi_fraction", roi.mean())
    _append(values, names, "zone_fraction", zone.sum() / roi.sum())

    channel_measurements: dict[str, dict[str, float]] = {}
    for channel_index, prefix in ((0, "artery"), (2, "vein")):
        channel = probability[channel_index]
        measurements: dict[str, float] = {}
        for region_name, region in (("roi", roi), ("zone", zone)):
            selected = channel[region]
            for statistic, statistic_value in (
                ("soft", selected.mean()),
                ("soft2", np.square(selected).mean()),
                ("p90", np.percentile(selected, 90.0)),
            ):
                name = f"{prefix}_{region_name}_{statistic}"
                measurements[name] = float(statistic_value)
                _append(values, names, name, statistic_value)

        for threshold in THRESHOLDS:
            suffix = str(threshold).replace(".", "p")
            binary_roi = (channel >= threshold) & roi
            binary_zone = (channel >= threshold) & zone
            density_roi = binary_roi.sum() / roi.sum()
            density_zone = binary_zone.sum() / zone.sum()
            fractal_roi = skeleton_box_dimension(binary_roi)
            for metric_name, metric_value in (
                (f"{prefix}_roi_density_t{suffix}", density_roi),
                (f"{prefix}_zone_density_t{suffix}", density_zone),
                (f"{prefix}_roi_fractal_t{suffix}", fractal_roi),
            ):
                measurements[metric_name] = float(metric_value)
                _append(values, names, metric_name, metric_value)

            width = _width_features(binary_zone, artery=channel_index == 0)
            for width_name, width_value in width.items():
                name = f"{prefix}_{width_name}_t{suffix}"
                measurements[name] = width_value
                _append(values, names, name, width_value)
        channel_measurements[prefix] = measurements

    artery = channel_measurements["artery"]
    vein = channel_measurements["vein"]
    for threshold in THRESHOLDS:
        suffix = str(threshold).replace(".", "p")
        paired = {
            f"log_zone_density_ratio_t{suffix}": _safe_log_ratio(
                artery[f"artery_zone_density_t{suffix}"], vein[f"vein_zone_density_t{suffix}"]
            ),
            f"log_knudtson_ratio_t{suffix}": _safe_log_ratio(
                artery[f"artery_knudtson_t{suffix}"], vein[f"vein_knudtson_t{suffix}"]
            ),
            f"fractal_difference_t{suffix}": (
                artery[f"artery_roi_fractal_t{suffix}"] - vein[f"vein_roi_fractal_t{suffix}"]
            ),
        }
        for name, value in paired.items():
            _append(values, names, name, value)

    vector = np.asarray(values, dtype=np.float64)
    if not np.isfinite(vector).all() or len(set(names)) != len(names):
        raise ValueError("Task 3 feature vector is invalid")
    metadata = {
        "od_x": center[0],
        "od_y": center[1],
        "od_diameter": diameter,
        "od_confidence": confidence,
        "zone_pixels": float(zone.sum()),
    }
    return vector, names, metadata
