from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from .biomarkers import BIOMARKER_KEYS, write_biomarker_txt
from .data import list_case_ids, read_png_float
from .submission import load_probability_png


THRESHOLDS = (0.25, 0.35, 0.45, 0.55, 0.65)


def zone_c_mask(
    shape_hw: tuple[int, int],
    center_xy: tuple[float, float],
    disc_diameter: float,
    roi_hw: np.ndarray | None = None,
) -> np.ndarray:
    """Return the SIVA Zone C annulus, 1.5 to 2.5 DD from the OD center."""

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
    if len(x_values) < 2:
        return 0.0
    return float(np.polyfit(x_values, y_values, 1)[0])


def knudtson_equivalent(widths: list[float] | np.ndarray, is_artery: bool) -> float:
    values = sorted((float(value) for value in widths if value > 0 and math.isfinite(value)), reverse=True)[:6]
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
    """Estimate OD center/diameter from brightness and vessel convergence."""

    cfp = np.asarray(cfp_rgb, dtype=np.float32)
    roi = np.asarray(roi_hw) > 0.5
    if cfp.ndim != 3 or cfp.shape[2] != 3 or roi.shape != cfp.shape[:2]:
        raise ValueError("Invalid CFP or ROI shape for optic-disc localization")
    if roi.sum() < 100:
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
    confidence = float(np.clip((saliency[center_y, center_x] - np.median(saliency[candidate])) / 8.0, 0.0, 1.0))

    y, x = np.ogrid[: roi.shape[0], : roi.shape[1]]
    local = ((x - center_x) ** 2 + (y - center_y) ** 2 <= 110**2) & roi
    local_values = brightness[local]
    threshold = float(np.percentile(local_values, 68.0)) if local_values.size else float(brightness[center_y, center_x])
    disc_candidate = local & (brightness >= threshold)
    labels, _ = ndimage.label(disc_candidate)
    label = int(labels[center_y, center_x])
    if label > 0:
        area = int(np.count_nonzero(labels == label))
        diameter = 2.0 * math.sqrt(area / math.pi)
    else:
        diameter = 120.0
    diameter = float(np.clip(diameter, 70.0, 180.0))
    return (float(center_x), float(center_y)), diameter, confidence


def _component_and_width_features(binary: np.ndarray) -> list[float]:
    labels, count = ndimage.label(binary)
    areas = np.bincount(labels.ravel())[1:] if count else np.array([], dtype=np.int64)
    skeleton = skeletonize(binary)
    distance = ndimage.distance_transform_edt(binary)
    widths = 2.0 * distance[skeleton]
    component_widths: list[float] = []
    for label_index in range(1, count + 1):
        selected = widths[labels[skeleton] == label_index]
        if selected.size >= 3:
            component_widths.append(float(np.percentile(selected, 90.0)))
    top = sorted(component_widths, reverse=True)[:6]
    width_quantiles = np.percentile(widths, [50, 75, 90, 95, 100]).tolist() if widths.size else [0.0] * 5
    return [
        float(count),
        float(skeleton.sum()),
        float(areas.max(initial=0)),
        float(np.median(areas)) if areas.size else 0.0,
        *[float(value) for value in width_quantiles],
        *top,
        *([0.0] * (6 - len(top))),
    ]


def extract_biomarker_features(
    probability_chw: np.ndarray,
    cfp_rgb: np.ndarray,
    roi_hw: np.ndarray,
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    probability = np.asarray(probability_chw, dtype=np.float32)
    roi = np.asarray(roi_hw) > 0.5
    if probability.shape != (3, *roi.shape):
        raise ValueError(f"Probability shape {probability.shape} does not match ROI {roi.shape}")
    center, diameter, confidence = locate_optic_disc(cfp_rgb, probability[1], roi)
    zone = zone_c_mask(roi.shape, center, diameter, roi)
    if zone.sum() < 100:
        raise ValueError("Estimated Zone C is too small")

    values: list[float] = [center[0] / roi.shape[1], center[1] / roi.shape[0], diameter / min(roi.shape), confidence]
    names = ["od_x", "od_y", "od_diameter", "od_confidence"]
    for channel_index, prefix in ((0, "artery"), (2, "vein")):
        channel = probability[channel_index]
        values.extend(
            [
                float(channel[roi].mean()),
                float(np.square(channel[roi]).mean()),
                float(channel[zone].mean()),
                float(np.square(channel[zone]).mean()),
            ]
        )
        names.extend([f"{prefix}_roi_soft", f"{prefix}_roi_soft2", f"{prefix}_zone_soft", f"{prefix}_zone_soft2"])
        for threshold in THRESHOLDS:
            binary_roi = (channel >= threshold) & roi
            binary_zone = (channel >= threshold) & zone
            values.extend([float(binary_roi.sum() / roi.sum()), float(binary_zone.sum() / zone.sum())])
            suffix = str(threshold).replace(".", "p")
            names.extend([f"{prefix}_roi_density_{suffix}", f"{prefix}_zone_density_{suffix}"])
        binary = (channel >= 0.45) & roi
        values.append(skeleton_box_dimension(binary))
        names.append(f"{prefix}_fractal_t0p45")
        component_values = _component_and_width_features(binary & zone)
        component_names = ["components", "skeleton_length", "largest_area", "median_area"]
        component_names += ["width_p50", "width_p75", "width_p90", "width_p95", "width_max"]
        component_names += [f"top_width_{index}" for index in range(1, 7)]
        values.extend(component_values)
        names.extend([f"{prefix}_{name}" for name in component_names])
        values.append(knudtson_equivalent(component_values[-6:], is_artery=channel_index == 0))
        names.append(f"{prefix}_knudtson_proxy")
    vector = np.asarray(values, dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("Biomarker features contain non-finite values")
    return vector, names, {"od_x": center[0], "od_y": center[1], "od_diameter": diameter, "od_confidence": confidence}


def read_biomarker_txt(path: Path | str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"Invalid biomarker line in {path}: {line!r}")
        values[fields[0]] = float(fields[1])
    if set(values) != set(BIOMARKER_KEYS) or not np.isfinite(list(values.values())).all():
        raise ValueError(f"Invalid biomarker keys or values in {path}")
    return values


def _fit_standardized_ridge(features: np.ndarray, log_target: np.ndarray, alpha: float):
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-8] = 1.0
    standardized = (features - mean) / std
    intercept = float(log_target.mean())
    centered = log_target - intercept
    regularizer = np.eye(features.shape[1], dtype=np.float64) * alpha
    beta = np.linalg.solve(standardized.T @ standardized + regularizer, standardized.T @ centered)
    return mean, std, intercept, beta


def _metrics(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    mae = float(np.mean(np.abs(target - prediction)) / max(float(np.median(target)), 1e-8))
    smape = float(np.mean(2.0 * np.abs(target - prediction) / np.maximum(target + prediction, 1e-8)))
    return mae, smape


def fit_calibrator(
    features: np.ndarray,
    targets: dict[str, np.ndarray],
    feature_names: list[str],
    folds: int = 5,
    repeats: int = 10,
    seed: int = 77,
) -> dict[str, object]:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] < folds or features.shape[1] != len(feature_names):
        raise ValueError("Invalid feature matrix or fold count")
    if not np.isfinite(features).all():
        raise ValueError("Features contain non-finite values")
    alphas = (0.1, 1.0, 10.0, 100.0)
    shrinkages = (0.25, 0.5, 0.75, 1.0)
    rng = np.random.default_rng(seed)
    repeat_folds = []
    for _ in range(repeats):
        indices = rng.permutation(features.shape[0])
        repeat_folds.append([np.asarray(chunk, dtype=int) for chunk in np.array_split(indices, folds)])

    output_targets: dict[str, object] = {}
    for key in BIOMARKER_KEYS:
        target = np.asarray(targets[key], dtype=np.float64)
        if target.shape != (features.shape[0],) or np.any(target <= 0) or not np.isfinite(target).all():
            raise ValueError(f"Invalid target {key}")
        best = None
        for alpha in alphas:
            model_sum = np.zeros_like(target)
            baseline_sum = np.zeros_like(target)
            counts = np.zeros_like(target)
            for chunks in repeat_folds:
                for validation in chunks:
                    training = np.setdiff1d(np.arange(features.shape[0]), validation, assume_unique=True)
                    mean, std, intercept, beta = _fit_standardized_ridge(features[training], np.log(target[training]), alpha)
                    model_sum[validation] += np.exp(intercept + ((features[validation] - mean) / std) @ beta)
                    baseline_sum[validation] += float(np.median(target[training]))
                    counts[validation] += 1.0
            model_prediction = model_sum / counts
            baseline_prediction = baseline_sum / counts
            baseline_metrics = _metrics(target, baseline_prediction)
            for shrinkage in shrinkages:
                prediction = np.exp(
                    shrinkage * np.log(np.maximum(model_prediction, 1e-8))
                    + (1.0 - shrinkage) * np.log(np.maximum(baseline_prediction, 1e-8))
                )
                metrics = _metrics(target, prediction)
                candidate = (sum(metrics), metrics, alpha, shrinkage, baseline_metrics)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        assert best is not None
        _, metrics, alpha, shrinkage, baseline_metrics = best
        accepted = metrics[0] < baseline_metrics[0] and metrics[1] < baseline_metrics[1]
        mean, std, intercept, beta = _fit_standardized_ridge(features, np.log(target), float(alpha))
        span = float(target.max() - target.min())
        output_targets[key] = {
            "accepted": bool(accepted),
            "alpha": float(alpha),
            "shrinkage": float(shrinkage) if accepted else 0.0,
            "median": float(np.median(target)),
            "minimum": float(max(1e-8, target.min() - 0.05 * span)),
            "maximum": float(target.max() + 0.05 * span),
            "intercept": intercept,
            "beta": beta.tolist(),
            "cv_mae": float(metrics[0]),
            "cv_smape": float(metrics[1]),
            "baseline_mae": float(baseline_metrics[0]),
            "baseline_smape": float(baseline_metrics[1]),
        }
    return {
        "version": 2,
        "feature_names": feature_names,
        "feature_mean": features.mean(axis=0).tolist(),
        "feature_std": np.maximum(features.std(axis=0), 1e-8).tolist(),
        "targets": output_targets,
        "ridge_mean": {key: _fit_standardized_ridge(features, np.log(np.asarray(targets[key])), float(output_targets[key]["alpha"]))[0].tolist() for key in BIOMARKER_KEYS},
        "ridge_std": {key: _fit_standardized_ridge(features, np.log(np.asarray(targets[key])), float(output_targets[key]["alpha"]))[1].tolist() for key in BIOMARKER_KEYS},
    }


def predict_calibrated(calibrator: dict[str, object], feature_vector: np.ndarray) -> dict[str, float]:
    vector = np.asarray(feature_vector, dtype=np.float64)
    if vector.shape != (len(calibrator["feature_names"]),):
        raise ValueError("Feature vector does not match calibrator")
    global_mean = np.asarray(calibrator["feature_mean"], dtype=np.float64)
    global_std = np.asarray(calibrator["feature_std"], dtype=np.float64)
    out_of_distribution = not np.isfinite(vector).all() or bool(np.any(np.abs((vector - global_mean) / global_std) > 10.0))
    values: dict[str, float] = {}
    for key in BIOMARKER_KEYS:
        model = calibrator["targets"][key]
        value = float(model["median"])
        if bool(model["accepted"]) and not out_of_distribution:
            mean = np.asarray(calibrator["ridge_mean"][key], dtype=np.float64)
            std = np.asarray(calibrator["ridge_std"][key], dtype=np.float64)
            beta = np.asarray(model["beta"], dtype=np.float64)
            learned = math.exp(float(model["intercept"]) + float(((vector - mean) / std) @ beta))
            shrinkage = float(model["shrinkage"])
            value = math.exp(shrinkage * math.log(max(learned, 1e-8)) + (1.0 - shrinkage) * math.log(value))
        values[key] = float(np.clip(value, model["minimum"], model["maximum"]))
    return enforce_avr_consistency(values)


def enforce_avr_consistency(values: dict[str, float]) -> dict[str, float]:
    output = dict(values)
    crae = max(float(output["CRAE"]), 1e-8)
    crve = max(float(output["CRVE"]), 1e-8)
    avr = max(float(output["AVR"]), 1e-8)
    geometric_mean = math.sqrt(crae * crve)
    output["CRAE"] = geometric_mean * math.sqrt(avr)
    output["CRVE"] = geometric_mean / math.sqrt(avr)
    output["AVR"] = output["CRAE"] / output["CRVE"]
    return output


def _case_features(data_root: Path, split: str, probability_dir: Path, case_id: str):
    probability = load_probability_png(probability_dir / f"{case_id}.png")
    cfp = read_png_float(data_root / split / "images" / f"{case_id}.png", channels=3)
    roi = read_png_float(data_root / split / "masks" / f"{case_id}.png", channels=1)[..., 0]
    return extract_biomarker_features(probability, cfp, roi)


def fit_from_oof(data_root: Path, probability_dir: Path, output_path: Path) -> Path:
    case_ids = list_case_ids(data_root, split="training")
    vectors = []
    names = None
    targets = {key: [] for key in BIOMARKER_KEYS}
    localization = {}
    for case_id in case_ids:
        vector, current_names, metadata = _case_features(data_root, "training", probability_dir, case_id)
        names = current_names if names is None else names
        if current_names != names:
            raise RuntimeError("Biomarker feature names changed between cases")
        vectors.append(vector)
        localization[case_id] = metadata
        values = read_biomarker_txt(data_root / "training" / "biomarker" / f"{case_id}.txt")
        for key in BIOMARKER_KEYS:
            targets[key].append(values[key])
    calibrator = fit_calibrator(
        np.stack(vectors),
        {key: np.asarray(value) for key, value in targets.items()},
        names or [],
    )
    calibrator["training_case_ids"] = case_ids
    calibrator["localization"] = localization
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(calibrator, indent=2), encoding="utf-8")
    return output_path


def generate_task3(
    data_root: Path,
    probability_dir: Path,
    output_dir: Path,
    calibrator_path: Path,
    split: str = "validation",
) -> Path:
    calibrator = json.loads(calibrator_path.read_text(encoding="utf-8"))
    case_ids = list_case_ids(data_root, split=split)
    if output_dir.exists():
        for path in output_dir.glob("*.txt"):
            path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    for case_id in case_ids:
        try:
            vector, names, metadata = _case_features(data_root, split, probability_dir, case_id)
            if names != calibrator["feature_names"] or metadata["od_confidence"] < 0.01:
                raise ValueError("Unreliable optic-disc localization or feature schema")
            values = predict_calibrated(calibrator, vector)
        except (ValueError, RuntimeError):
            values = {key: float(calibrator["targets"][key]["median"]) for key in BIOMARKER_KEYS}
            values = enforce_avr_consistency(values)
        write_biomarker_txt(values, output_dir / f"{case_id}.txt")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit and run OOF-calibrated GAVE2 Task 3 biomarkers.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--data-root", type=Path, required=True)
    fit.add_argument("--oof-task2-dir", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--data-root", type=Path, required=True)
    predict.add_argument("--task2-dir", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--calibrator", type=Path, required=True)
    predict.add_argument("--split", choices=("training", "validation"), default="validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "fit":
        path = fit_from_oof(args.data_root, args.oof_task2_dir, args.output)
    else:
        path = generate_task3(args.data_root, args.task2_dir, args.output_dir, args.calibrator, args.split)
    print(f"Task 3 complete: {path}")


if __name__ == "__main__":
    main()
