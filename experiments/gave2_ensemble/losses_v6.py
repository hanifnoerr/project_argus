from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _last_prediction(predictions):
    if isinstance(predictions, (list, tuple)):
        return predictions[-1]
    return predictions


def _expanded_mask(mask: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(target[:, :1])
    if mask.shape != target[:, :1].shape:
        raise ValueError(f"Mask must have shape {target[:, :1].shape}, got {mask.shape}")
    return mask.to(device=target.device, dtype=target.dtype)


def soft_centerline_overlap(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    kernel_size: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    if probabilities.shape != target.shape:
        raise ValueError(f"Probability and target shapes must match, got {probabilities.shape} vs {target.shape}")
    if probabilities.ndim != 4:
        raise ValueError(f"Expected [N, C, H, W] tensors, got {probabilities.shape}")
    if kernel_size % 2 != 1:
        raise ValueError("kernel_size must be odd")
    mask1 = _expanded_mask(mask, target)
    maskc = mask1.expand_as(target)
    prob = probabilities.float() * maskc
    tgt = target.float() * maskc
    pad = kernel_size // 2
    pooled_target = F.max_pool2d(tgt, kernel_size=kernel_size, stride=1, padding=pad)
    pooled_probability = F.avg_pool2d(prob, kernel_size=kernel_size, stride=1, padding=pad)
    dims = (0, 2, 3)
    precision = (prob * pooled_target).sum(dims) / prob.sum(dims).clamp_min(eps)
    recall = (tgt * pooled_probability).sum(dims) / tgt.sum(dims).clamp_min(eps)
    return (2.0 * precision * recall) / (precision + recall).clamp_min(eps)


class VesselTopologyLoss(nn.Module):
    def __init__(self, from_logits: bool = False, kernel_size: int = 3, eps: float = 1e-6) -> None:
        super().__init__()
        self.from_logits = bool(from_logits)
        self.kernel_size = int(kernel_size)
        self.eps = float(eps)

    def forward(self, predictions, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        value = _last_prediction(predictions)
        probability = torch.sigmoid(value) if self.from_logits else value
        overlap = soft_centerline_overlap(
            probability,
            target,
            mask=mask,
            kernel_size=self.kernel_size,
            eps=self.eps,
        )
        return 1.0 - overlap.mean()
