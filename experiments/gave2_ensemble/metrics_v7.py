from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from .metrics_v6 import challenge_selection_score


def _path_channel_score(prediction: np.ndarray, target: np.ndarray, wrong_target: np.ndarray) -> tuple[float, float]:
    structure = np.ones((3, 3), dtype=np.uint8)
    pred = np.asarray(prediction, dtype=bool)
    tgt = np.asarray(target, dtype=bool)
    if not tgt.any():
        return (1.0 if not pred.any() else 0.0), (1.0 if not pred.any() else 0.0)

    target_skeleton = skeletonize(tgt)
    predicted_labels, _ = ndimage.label(pred, structure=structure)
    target_labels, target_count = ndimage.label(target_skeleton, structure=structure)
    recalls: list[float] = []
    weights: list[float] = []
    for component in range(1, target_count + 1):
        component_mask = target_labels == component
        length = int(component_mask.sum())
        if length < 3:
            continue
        labels = predicted_labels[component_mask]
        labels = labels[labels > 0]
        largest_connected_overlap = int(np.bincount(labels).max()) if labels.size else 0
        recalls.append(largest_connected_overlap / float(length))
        weights.append(float(length))
    path_recall = float(np.average(recalls, weights=weights)) if recalls else 0.0

    predicted_skeleton = skeletonize(pred)
    if not predicted_skeleton.any():
        return path_recall, 0.0
    near_correct = ndimage.binary_dilation(tgt, structure=structure, iterations=1)
    near_wrong = ndimage.binary_dilation(np.asarray(wrong_target, dtype=bool), structure=structure, iterations=1)
    correct = predicted_skeleton & near_correct & ~near_wrong
    path_precision = float(correct.sum() / max(int(predicted_skeleton.sum()), 1))
    return path_recall, path_precision


def challenge_selection_score_v7(probabilities, targets, roi, threshold: float = 0.5) -> dict[str, object]:
    probability = np.asarray(probabilities, dtype=np.float32)
    target = np.asarray(targets, dtype=np.float32)
    roi_mask = np.asarray(roi, dtype=np.float32)
    base = challenge_selection_score(probability, target, roi_mask, threshold=threshold)
    binary = probability >= float(threshold)
    target_binary = target > 0.5
    roi_binary = roi_mask > 0.5

    recalls: list[float] = []
    precisions: list[float] = []
    for index in range(probability.shape[0]):
        case_roi = roi_binary[index, 0]
        for channel, wrong_channel in ((0, 2), (2, 0)):
            recall, precision = _path_channel_score(
                binary[index, channel] & case_roi,
                target_binary[index, channel] & case_roi,
                target_binary[index, wrong_channel] & case_roi,
            )
            recalls.append(recall)
            precisions.append(precision)

    path_recall = float(np.mean(recalls)) if recalls else 0.0
    path_precision = float(np.mean(precisions)) if precisions else 0.0
    path_score = 0.5 * path_recall + 0.5 * path_precision
    sensitivity = float(np.mean(np.asarray(base["sensitivity_channels"])[[0, 2]]))
    specificity = float(np.mean(np.asarray(base["specificity_channels"])[[0, 2]]))
    accuracy = float(np.mean(np.asarray(base["accuracy_channels"])[[0, 2]]))
    classification = 0.3 * sensitivity + 0.3 * specificity + 0.4 * accuracy
    selection = 0.4 * float(base["dice_mean"]) + 0.3 * classification + 0.3 * path_score
    return {
        **base,
        "av_sensitivity": sensitivity,
        "av_specificity": specificity,
        "av_accuracy": accuracy,
        "classification_mean": classification,
        "path_recall": path_recall,
        "path_precision": path_precision,
        "path_score": path_score,
        "selection_score": float(selection),
    }
