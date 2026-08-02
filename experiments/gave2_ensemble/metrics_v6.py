from __future__ import annotations

import numpy as np


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _expand_roi(roi: np.ndarray, expected_shape: tuple[int, int, int, int]) -> np.ndarray:
    expected_roi_shape = (expected_shape[0], 1, expected_shape[2], expected_shape[3])
    if roi.shape != expected_roi_shape:
        raise ValueError(f"ROI must have shape {expected_roi_shape}, got {roi.shape}")
    channels = expected_shape[1]
    return np.repeat(roi > 0.5, channels, axis=1)


def _pool2d(value: np.ndarray, kernel_size: int, mode: str) -> np.ndarray:
    if kernel_size % 2 != 1:
        raise ValueError("kernel_size must be odd")
    if mode not in {"max", "avg"}:
        raise ValueError(f"Unsupported pooling mode {mode!r}")
    pad = kernel_size // 2
    padded = np.pad(value, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size), axis=(2, 3))
    if mode == "max":
        return windows.max(axis=(-2, -1))
    return windows.mean(axis=(-2, -1))


def _topology_scores(prediction: np.ndarray, target: np.ndarray, roi: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    pred = prediction * roi
    tgt = target * roi
    pooled_target = _pool2d(tgt, kernel_size=3, mode="max")
    pooled_prediction = _pool2d(pred, kernel_size=3, mode="avg")
    dims = (-2, -1)
    precision = (pred * pooled_target).sum(axis=dims) / np.clip(pred.sum(axis=dims), eps, None)
    recall = (tgt * pooled_prediction).sum(axis=dims) / np.clip(tgt.sum(axis=dims), eps, None)
    return (2.0 * precision * recall) / np.clip(precision + recall, eps, None)


def challenge_selection_score(probabilities, targets, roi, threshold: float = 0.5) -> dict[str, object]:
    probability = _to_numpy(probabilities)
    target = _to_numpy(targets)
    roi_mask = _to_numpy(roi)
    if probability.shape != target.shape:
        raise ValueError(f"Probability and target shapes must match, got {probability.shape} vs {target.shape}")
    if probability.ndim != 4:
        raise ValueError(f"Expected [N, C, H, W] tensors, got {probability.shape}")
    roi_channels = _expand_roi(roi_mask, probability.shape)

    binary = (probability >= float(threshold)).astype(np.float32)
    target_binary = (target > 0.5).astype(np.float32)
    pred = binary * roi_channels
    tgt = target_binary * roi_channels
    negative = (1.0 - tgt) * roi_channels
    dims = (-2, -1)

    true_positive = (pred * tgt).sum(axis=dims)
    false_positive = (pred * negative).sum(axis=dims)
    false_negative = ((1.0 - pred) * tgt).sum(axis=dims)
    true_negative = (((1.0 - pred) * negative)).sum(axis=dims)

    eps = 1e-6
    dice = (2.0 * true_positive + eps) / (2.0 * true_positive + false_positive + false_negative + eps)
    sensitivity = (true_positive + eps) / (true_positive + false_negative + eps)
    specificity = (true_negative + eps) / (true_negative + false_positive + eps)
    accuracy = (true_positive + true_negative + eps) / (
        true_positive + true_negative + false_positive + false_negative + eps
    )
    topology = _topology_scores(pred, tgt, roi_channels, eps=eps)

    dice_cases = dice.mean(axis=1)
    sensitivity_cases = sensitivity.mean(axis=1)
    specificity_cases = specificity.mean(axis=1)
    accuracy_cases = accuracy.mean(axis=1)
    topology_cases = topology.mean(axis=1)

    dice_mean = float(dice_cases.mean())
    sensitivity_mean = float(sensitivity_cases.mean())
    specificity_mean = float(specificity_cases.mean())
    accuracy_mean = float(accuracy_cases.mean())
    topology_mean = float(topology_cases.mean())
    classification_mean = float(np.mean([sensitivity_mean, specificity_mean, accuracy_mean]))
    selection_score = 0.4 * dice_mean + 0.3 * classification_mean + 0.3 * topology_mean

    return {
        "threshold": float(threshold),
        "dice_cases": [float(value) for value in dice_cases],
        "dice_channels": [float(value) for value in dice.mean(axis=0)],
        "dice_mean": dice_mean,
        "sensitivity_cases": [float(value) for value in sensitivity_cases],
        "sensitivity_channels": [float(value) for value in sensitivity.mean(axis=0)],
        "sensitivity_mean": sensitivity_mean,
        "specificity_cases": [float(value) for value in specificity_cases],
        "specificity_channels": [float(value) for value in specificity.mean(axis=0)],
        "specificity_mean": specificity_mean,
        "accuracy_cases": [float(value) for value in accuracy_cases],
        "accuracy_channels": [float(value) for value in accuracy.mean(axis=0)],
        "accuracy_mean": accuracy_mean,
        "classification_mean": classification_mean,
        "topology_cases": [float(value) for value in topology_cases],
        "topology_channels": [float(value) for value in topology.mean(axis=0)],
        "topology_mean": topology_mean,
        "selection_score": float(selection_score),
    }
