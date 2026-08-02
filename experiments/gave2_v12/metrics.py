from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from .constants import LIVE_CLASSIFICATION_WEIGHT, LIVE_DICE_WEIGHT, LIVE_TOPOLOGY_WEIGHT


def pixel_score(report: dict[str, object]) -> float:
    """Observed leaderboard contribution from reproducible pixel metrics only."""

    return 10.0 * (
        LIVE_CLASSIFICATION_WEIGHT * float(report["classification"])
        + LIVE_DICE_WEIGHT * float(report["dice"])
    )


def _case_metrics(probability: np.ndarray, target: np.ndarray, roi: np.ndarray, threshold: float) -> dict[str, float]:
    selected = (0, 2)
    pred = np.asarray(probability)[list(selected)] >= threshold
    truth = np.asarray(target)[list(selected)] > 0.5
    mask = np.asarray(roi, dtype=bool)
    values = {name: [] for name in ("dice", "sensitivity", "specificity", "accuracy", "topology")}
    for channel in range(2):
        prediction = pred[channel] & mask
        label = truth[channel] & mask
        tp = int((prediction & label).sum())
        fp = int((prediction & ~label & mask).sum())
        fn = int((~prediction & label).sum())
        tn = int((~prediction & ~label & mask).sum())
        values["dice"].append((2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6))
        values["sensitivity"].append((tp + 1e-6) / (tp + fn + 1e-6))
        values["specificity"].append((tn + 1e-6) / (tn + fp + 1e-6))
        values["accuracy"].append((tp + tn + 1e-6) / (tp + tn + fp + fn + 1e-6))
        target_skeleton = skeletonize(label)
        prediction_skeleton = skeletonize(prediction)
        tolerance_prediction = ndimage.binary_dilation(prediction, iterations=2)
        tolerance_target = ndimage.binary_dilation(label, iterations=2)
        recall = (target_skeleton & tolerance_prediction).sum() / max(int(target_skeleton.sum()), 1)
        precision = (prediction_skeleton & tolerance_target).sum() / max(int(prediction_skeleton.sum()), 1)
        values["topology"].append(2.0 * precision * recall / max(precision + recall, 1e-6))
    result = {name: float(np.mean(metric)) for name, metric in values.items()}
    result["classification"] = (
        0.3 * result["sensitivity"] + 0.3 * result["specificity"] + 0.4 * result["accuracy"]
    )
    result["score"] = 10.0 * (
        LIVE_CLASSIFICATION_WEIGHT * result["classification"]
        + LIVE_DICE_WEIGHT * result["dice"]
        + LIVE_TOPOLOGY_WEIGHT * result["topology"]
    )
    return result


def evaluate_cases(
    probabilities: list[np.ndarray],
    targets: list[np.ndarray],
    rois: list[np.ndarray],
    *,
    threshold: float = 0.5,
) -> dict[str, object]:
    if not (len(probabilities) == len(targets) == len(rois)) or not probabilities:
        raise ValueError("Metric inputs must contain the same nonzero number of cases")
    cases = [_case_metrics(probability, target, roi, threshold) for probability, target, roi in zip(probabilities, targets, rois)]
    names = tuple(cases[0])
    return {
        "threshold": float(threshold),
        "cases": len(cases),
        **{name: float(np.mean([case[name] for case in cases])) for name in names},
        "case_metrics": cases,
    }


def teacher_path_recall(candidate: np.ndarray, teacher: np.ndarray, threshold: float = 0.5) -> float:
    recalls = []
    for channel in (0, 2):
        path = skeletonize(np.asarray(teacher[channel]) >= threshold)
        if not path.any():
            continue
        recalls.append(float(((np.asarray(candidate[channel]) >= threshold) & path).sum() / path.sum()))
    return float(np.mean(recalls)) if recalls else 1.0
