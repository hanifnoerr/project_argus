from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _soft_erode(value: torch.Tensor) -> torch.Tensor:
    vertical = -F.max_pool2d(-value, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-value, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def _soft_dilate(value: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(value, 3, stride=1, padding=1)


def soft_skeleton(value: torch.Tensor, iterations: int = 8) -> torch.Tensor:
    opened = _soft_dilate(_soft_erode(value))
    skeleton = F.relu(value - opened)
    for _ in range(iterations):
        value = _soft_erode(value)
        opened = _soft_dilate(_soft_erode(value))
        delta = F.relu(value - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(probability: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probability = F.avg_pool2d(probability * mask, kernel_size=4, stride=4)
    target = F.max_pool2d(target * mask, kernel_size=4, stride=4)
    pred_skeleton = soft_skeleton(probability)
    target_skeleton = soft_skeleton(target)
    dims = (-2, -1)
    precision = (pred_skeleton * target).sum(dims) / (pred_skeleton.sum(dims) + 1e-6)
    sensitivity = (target_skeleton * probability).sum(dims) / (target_skeleton.sum(dims) + 1e-6)
    cldice = 2.0 * precision * sensitivity / (precision + sensitivity + 1e-6)
    return 1.0 - cldice.mean()


def weighted_binary_cross_entropy(
    probability: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    positive_weights: torch.Tensor,
) -> torch.Tensor:
    epsilon = 1e-5
    probability = probability.clamp(epsilon, 1.0 - epsilon)
    weight = 1.0 + target * (positive_weights[None, :, None, None] - 1.0)
    loss = -(target * torch.log(probability) + (1.0 - target) * torch.log1p(-probability)) * weight * mask
    return loss.sum() / ((weight * mask).sum() + epsilon)


def dice_loss(probability: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probability, target = probability * mask, target * mask
    dims = (-2, -1)
    score = (2.0 * (probability * target).sum(dims) + 1.0) / (
        probability.sum(dims) + target.sum(dims) + 1.0
    )
    return 1.0 - score.mean()


@dataclass(frozen=True)
class LossWeights:
    classification: float = 0.32
    dice: float = 0.18
    topology: float = 0.34
    crossing: float = 0.05
    teacher: float = 0.07
    residual: float = 0.04


class ResidualChallengeLoss(nn.Module):
    def __init__(self, positive_weights: torch.Tensor, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.register_buffer("positive_weights", positive_weights.float().clamp(1.0, 40.0))
        self.weights = weights or LossWeights()

    def forward(
        self,
        output: dict[str, torch.Tensor],
        target: torch.Tensor,
        teacher: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        probability = output["probability"]
        av_index = torch.tensor((0, 2), device=probability.device)
        av_probability = probability.index_select(1, av_index)
        av_target = target.index_select(1, av_index)
        av_weights = self.positive_weights.index_select(0, av_index)
        classification = weighted_binary_cross_entropy(av_probability, av_target, mask, av_weights)
        dice = dice_loss(av_probability, av_target, mask)
        topology = soft_cldice_loss(av_probability, av_target, mask)

        crossing_target = (target[:, 0:1] * target[:, 2:3]) * mask
        crossing_logit = output["crossing_logit"]
        crossing_weight = torch.where(crossing_target > 0.5, 30.0, 1.0)
        crossing_map = F.binary_cross_entropy_with_logits(crossing_logit, crossing_target, reduction="none")
        crossing = (crossing_map * crossing_weight * mask).mean()
        crossing = crossing + F.l1_loss(
            torch.minimum(probability[:, 0:1], probability[:, 2:3]) * mask,
            crossing_target,
        )

        confident = ((teacher >= 0.85) | (teacher <= 0.03)).to(probability.dtype) * mask
        teacher_loss = ((probability - teacher).abs() * confident).sum() / (confident.sum() + 1.0)
        residual = output["bounded_delta"].abs().mul(mask).sum() / (mask.sum() * 3.0 + 1.0)
        components = {
            "classification": classification,
            "dice": dice,
            "topology": topology,
            "crossing": crossing,
            "teacher": teacher_loss,
            "residual": residual,
        }
        total = sum(getattr(self.weights, name) * value for name, value in components.items())
        return total, components
