from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import focal_tversky_loss_with_logits, recursive_stage_weights, weighted_masked_bce_with_logits


def _last_prediction(predictions):
    return predictions[-1] if isinstance(predictions, (list, tuple)) else predictions


def conditional_probabilities_from_logits(predictions, temperature: float = 1.0) -> torch.Tensor:
    """Convert [A evidence, vessel, V evidence] logits to exclusive challenge probabilities.

    The minimum composition keeps the winning A/V class above 0.5 whenever the
    vessel and conditional class probabilities are above 0.5. Consequently A
    and V cannot both be positive at the challenge decision boundary.
    """

    logits = _last_prediction(predictions)
    if logits.ndim != 4 or logits.shape[1] != 3:
        raise ValueError(f"Expected [N,3,H,W] logits, got {tuple(logits.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    # Keep the final probability algebra in FP32 even under BF16 autocast.
    # BF16 rounds probabilities near 0.5 too coarsely to preserve exclusivity.
    working_logits = logits.float()
    vessel = torch.sigmoid(working_logits[:, 1:2])
    av_logits = torch.cat((working_logits[:, 0:1], working_logits[:, 2:3]), dim=1) / float(temperature)
    # Deterministically break exact equality so >=0.5 can never select both classes.
    tie_break = av_logits.new_tensor((1e-3, -1e-3)).view(1, 2, 1, 1)
    av = torch.softmax(av_logits + tie_break, dim=1)
    artery = torch.minimum(vessel, av[:, 0:1])
    vein = torch.minimum(vessel, av[:, 1:2])
    return torch.cat((artery, vessel, vein), dim=1)


def _masked_soft_dice(probability: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask3 = mask.expand_as(target)
    probability = probability * mask3
    target = target * mask3
    dims = (0, 2, 3)
    intersection = (probability * target).sum(dims)
    denominator = probability.sum(dims) + target.sum(dims)
    return 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def _balanced_av_cross_entropy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    artery = target[:, 0:1] > 0.5
    vein = target[:, 2:3] > 0.5
    exclusive = torch.logical_xor(artery, vein) & (mask > 0.5)
    selected = exclusive.squeeze(1)
    if not bool(selected.any()):
        return logits.sum() * 0.0
    labels = vein.long().squeeze(1)[selected]
    av_logits = torch.cat((logits[:, 0:1], logits[:, 2:3]), dim=1).permute(0, 2, 3, 1)[selected]
    counts = torch.bincount(labels, minlength=2).to(dtype=av_logits.dtype)
    weights = counts.sum() / counts.clamp_min(1.0)
    weights = (weights / weights.mean()).clamp(0.5, 2.0)
    return F.cross_entropy(av_logits, labels, weight=weights)


def _soft_skeleton(probability: torch.Tensor, iterations: int = 3) -> torch.Tensor:
    def soft_erode(value):
        horizontal = -F.max_pool2d(-value, (3, 1), stride=1, padding=(1, 0))
        vertical = -F.max_pool2d(-value, (1, 3), stride=1, padding=(0, 1))
        return torch.minimum(horizontal, vertical)

    def soft_open(value):
        return F.max_pool2d(soft_erode(value), 3, stride=1, padding=1)

    skeleton = F.relu(probability - soft_open(probability))
    value = probability
    for _ in range(iterations):
        value = soft_erode(value)
        delta = F.relu(value - soft_open(value))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def _multichannel_cldice(probability: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Half-resolution topology substantially reduces full-canvas training memory.
    probability = F.avg_pool2d(probability * mask, 2, stride=2)
    target = F.max_pool2d(target * mask, 2, stride=2)
    pred_skeleton = _soft_skeleton(probability)
    target_skeleton = _soft_skeleton(target)
    dims = (0, 2, 3)
    precision = (pred_skeleton * target).sum(dims) / pred_skeleton.sum(dims).clamp_min(1e-6)
    sensitivity = (target_skeleton * probability).sum(dims) / target_skeleton.sum(dims).clamp_min(1e-6)
    cldice = (2.0 * precision * sensitivity) / (precision + sensitivity).clamp_min(1e-6)
    return 1.0 - cldice.mean()


class ConditionalPathLossV7(nn.Module):
    """Vessel detection plus conditional A/V classification and centerline coverage."""

    def __init__(
        self,
        positive_weights,
        vessel_weight: float = 0.45,
        av_weight: float = 0.30,
        dice_weight: float = 0.20,
        topology_weight: float = 0.05,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        weights = torch.as_tensor(positive_weights, dtype=torch.float32)
        if weights.numel() != 3:
            raise ValueError("positive_weights must contain artery, vessel, and vein weights")
        self.register_buffer("vessel_positive_weight", weights[1:2])
        self.vessel_weight = float(vessel_weight)
        self.av_weight = float(av_weight)
        self.dice_weight = float(dice_weight)
        self.topology_weight = float(topology_weight)
        self.temperature = float(temperature)

    def _stage(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        vessel_logits = logits[:, 1:2]
        vessel_target = target[:, 1:2]
        vessel_bce = weighted_masked_bce_with_logits(
            vessel_logits, vessel_target, mask, self.vessel_positive_weight
        )
        vessel_tversky = focal_tversky_loss_with_logits(vessel_logits, vessel_target, mask)
        vessel_loss = 0.35 * vessel_bce + 0.65 * vessel_tversky
        av_loss = _balanced_av_cross_entropy(logits, target, mask)
        probability = conditional_probabilities_from_logits(logits, temperature=self.temperature)
        dice_loss = _masked_soft_dice(probability, target, mask)
        topology_loss = _multichannel_cldice(probability, target, mask)
        return (
            self.vessel_weight * vessel_loss
            + self.av_weight * av_loss
            + self.dice_weight * dice_loss
            + self.topology_weight * topology_loss
        )

    def forward(self, predictions, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        stages = list(predictions) if isinstance(predictions, (list, tuple)) else [predictions]
        weights = recursive_stage_weights(len(stages))
        return sum(weight * self._stage(logits, target, mask) for weight, logits in zip(weights, stages))
