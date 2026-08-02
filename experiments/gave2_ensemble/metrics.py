from __future__ import annotations

import torch


def dice_per_channel(probabilities: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5):
    pred = (probabilities >= threshold).float() * mask
    gt = target.float() * mask
    dims = (0, 2, 3)
    intersection = (pred * gt).sum(dims)
    denom = pred.sum(dims) + gt.sum(dims)
    return (2.0 * intersection + 1e-6) / (denom + 1e-6)


def mean_dice(probabilities: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5) -> float:
    return float(dice_per_channel(probabilities, target, mask, threshold=threshold).mean().detach().cpu())


def soft_dice(probabilities: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    prob = probabilities.float() * mask
    gt = target.float() * mask
    dims = (0, 2, 3)
    intersection = (prob * gt).sum(dims)
    denom = prob.sum(dims) + gt.sum(dims)
    dice = (2.0 * intersection + 1e-6) / (denom + 1e-6)
    return float(dice.mean().detach().cpu())


def best_threshold_dice(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    thresholds: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
) -> tuple[float, float]:
    best_score = -1.0
    best_threshold = thresholds[0]
    for threshold in thresholds:
        score = mean_dice(probabilities, target, mask, threshold=threshold)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_score, best_threshold


def roi_channel_means(probabilities: torch.Tensor, mask: torch.Tensor) -> list[float]:
    roi = mask.expand_as(probabilities)
    denom = roi.sum(dim=(0, 2, 3)).clamp_min(1.0)
    means = (probabilities.float() * roi).sum(dim=(0, 2, 3)) / denom
    return [float(value) for value in means.detach().cpu()]


def channel_confusion(probabilities: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5):
    pred = (probabilities >= threshold).bool()
    gt = target.bool()
    roi = mask.bool().expand_as(gt)
    tp = (pred & gt & roi).sum(dim=(0, 2, 3)).detach().cpu()
    fp = (pred & ~gt & roi).sum(dim=(0, 2, 3)).detach().cpu()
    fn = (~pred & gt & roi).sum(dim=(0, 2, 3)).detach().cpu()
    tn = (~pred & ~gt & roi).sum(dim=(0, 2, 3)).detach().cpu()
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
