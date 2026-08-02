from __future__ import annotations

import numpy as np


def positive_weights_from_counts(
    positive,
    total,
    minimum: float = 4.0,
    maximum: float = 20.0,
) -> np.ndarray:
    positive_array = np.asarray(positive, dtype=np.float64)
    total_array = np.asarray(total, dtype=np.float64)
    if positive_array.shape != total_array.shape:
        raise ValueError("positive and total counts must have the same shape")
    if np.any(positive_array <= 0) or np.any(total_array <= positive_array):
        raise ValueError("Every channel must have positive and negative pixels")
    weights = (total_array - positive_array) / positive_array
    return np.clip(weights, minimum, maximum).astype(np.float32)


def average_precision_from_histograms(positive_histogram, total_histogram) -> dict[str, object]:
    positive = np.asarray(positive_histogram, dtype=np.float64)
    total = np.asarray(total_histogram, dtype=np.float64)
    if positive.shape != total.shape or positive.ndim != 2:
        raise ValueError("Histograms must be two equally shaped channel-by-bin arrays")
    if np.any(positive < 0) or np.any(total < positive):
        raise ValueError("Histogram counts are invalid")
    positives = positive.sum(axis=1)
    pixels = total.sum(axis=1)
    if np.any(positives <= 0) or np.any(pixels <= positives):
        raise ValueError("Every channel needs positive and negative examples")

    cumulative_positive = np.cumsum(positive[:, ::-1], axis=1)
    cumulative_prediction = np.cumsum(total[:, ::-1], axis=1)
    precision = cumulative_positive / np.maximum(cumulative_prediction, 1e-12)
    recall = cumulative_positive / positives[:, None]
    recall_increment = np.diff(recall, axis=1, prepend=np.zeros((recall.shape[0], 1)))
    average_precision = np.sum(precision * recall_increment, axis=1)
    prevalence = positives / pixels
    lift = average_precision / np.maximum(prevalence, 1e-12)
    return {
        "average_precision": float(np.mean(average_precision)),
        "average_precision_channels": average_precision.tolist(),
        "prevalence": float(np.mean(prevalence)),
        "prevalence_channels": prevalence.tolist(),
        "average_precision_lift": float(np.mean(lift)),
        "average_precision_lift_channels": lift.tolist(),
    }


def assess_learning_gate(
    history: list[dict[str, object]],
    minimum_epochs: int = 15,
    minimum_average_precision: float = 0.10,
    minimum_average_precision_lift: float = 1.50,
) -> dict[str, object]:
    reasons: list[str] = []
    if not history:
        return {"ok": False, "reasons": ["training history is empty"]}
    completed_epochs = max(int(row.get("epoch", 0)) for row in history)
    if completed_epochs < minimum_epochs:
        reasons.append(f"only {completed_epochs} epochs completed; need {minimum_epochs}")
    rows_with_ap = [row for row in history if "average_precision" in row and "average_precision_lift" in row]
    if not rows_with_ap:
        reasons.append("average precision metrics are unavailable")
        return {"ok": False, "reasons": reasons, "completed_epochs": completed_epochs}

    best_ap_row = max(rows_with_ap, key=lambda row: float(row["average_precision"]))
    best_ap = float(best_ap_row["average_precision"])
    best_lift = max(float(row["average_precision_lift"]) for row in rows_with_ap)
    if best_ap < minimum_average_precision:
        reasons.append(f"best average precision {best_ap:.4f} is below {minimum_average_precision:.4f}")
    if best_lift < minimum_average_precision_lift:
        reasons.append(
            f"best average precision lift {best_lift:.3f} is below {minimum_average_precision_lift:.3f}"
        )

    channel_lifts = best_ap_row.get("average_precision_lift_channels")
    if channel_lifts is not None and min(float(value) for value in channel_lifts) < 1.20:
        reasons.append("at least one output channel has average precision lift below 1.20")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "completed_epochs": completed_epochs,
        "best_average_precision": best_ap,
        "best_average_precision_lift": best_lift,
    }


def recursive_stage_weights(count: int) -> list[float]:
    if count < 1:
        raise ValueError("At least one recursive prediction is required")
    if count == 3:
        return [0.2, 0.3, 0.5]
    raw = np.arange(1, count + 1, dtype=np.float64)
    raw /= raw.sum()
    return [float(value) for value in raw]
