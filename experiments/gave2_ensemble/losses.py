from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .training_utils_v2 import recursive_stage_weights


def _last_prediction(predictions):
    if isinstance(predictions, (list, tuple)):
        return predictions[-1]
    return predictions


def masked_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask3 = mask.expand_as(target)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    denom = mask3.sum().clamp_min(1.0)
    return (loss * mask3).sum() / denom


def soft_dice_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    prob = torch.sigmoid(logits) * mask
    target = target * mask
    dims = (0, 2, 3)
    intersection = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def focal_tversky_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    false_positive_weight: float = 0.30,
    false_negative_weight: float = 0.70,
    gamma: float = 0.75,
    eps: float = 1e-6,
) -> torch.Tensor:
    probability = torch.sigmoid(logits) * mask
    target = target * mask
    dims = (0, 2, 3)
    true_positive = (probability * target).sum(dims)
    false_positive = (probability * (1.0 - target) * mask).sum(dims)
    false_negative = ((1.0 - probability) * target).sum(dims)
    tversky = (true_positive + eps) / (
        true_positive
        + false_positive_weight * false_positive
        + false_negative_weight * false_negative
        + eps
    )
    return torch.pow(1.0 - tversky, gamma).mean()


def vessel_consistency_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    artery = prob[:, 0:1]
    vessel = prob[:, 1:2]
    vein = prob[:, 2:3]
    over_artery = F.relu(artery - vessel)
    over_vein = F.relu(vein - vessel)
    return ((over_artery + over_vein) * mask).mean()


def _masked_bce_flat(logits: torch.Tensor, target: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    selected = keep > 0.5
    if int(selected.sum().detach().cpu()) == 0:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[selected], target[selected])


class OfficialBCE3Loss(nn.Module):
    """CMRRWNet baseline BCE3 loss for AV3 labels.

    Channel order in this package is A/VT/V:
    0 = artery, 1 = vessel tree, 2 = vein.
    Ambiguous vessel-tree pixels are ignored for artery/vein supervision but
    still used for vessel-tree supervision, matching the official baseline.
    """

    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask1 = torch.round(mask[:, 0, :, :]).float()
        gt_a = target[:, 0, :, :]
        gt_vt = target[:, 1, :, :]
        gt_v = target[:, 2, :, :]
        pred_a = logits[:, 0, :, :]
        pred_vt = logits[:, 1, :, :]
        pred_v = logits[:, 2, :, :]

        uncertain = (gt_vt - gt_v - gt_a).clamp_min(0.0)
        known_av = (mask1 - uncertain).clamp_min(0.0)

        loss = _masked_bce_flat(pred_a, gt_a, known_av)
        loss = loss + _masked_bce_flat(pred_v, gt_v, known_av)
        loss = loss + _masked_bce_flat(pred_vt, gt_vt, mask1)
        return loss


class OfficialRRLoss(nn.Module):
    """Recursive refinement loss used by the official CMRRWNet baseline."""

    def __init__(self) -> None:
        super().__init__()
        self.base_criterion = OfficialBCE3Loss()

    def forward(self, predictions, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not isinstance(predictions, (list, tuple)):
            return self.base_criterion(predictions, target, mask)
        loss_1 = self.base_criterion(predictions[0], target, mask)
        if len(predictions) == 1:
            return loss_1

        loss_2 = self.base_criterion(predictions[1], target, mask)
        if len(predictions) == 2:
            return loss_1 + loss_2
        for i, logits in enumerate(predictions[2:], 2):
            loss_2 = loss_2 + i * self.base_criterion(logits, target, mask)

        k = len(predictions[1:])
        normalizer = 0.5 * k * (k + 1)
        return loss_1 + loss_2 / normalizer


class GAVE2SegmentationLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        consistency_weight: float = 0.15,
        refinement_decay: float = 0.75,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.consistency_weight = consistency_weight
        self.refinement_decay = refinement_decay

    def _single(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = self.bce_weight * masked_bce_with_logits(logits, target, mask)
        loss = loss + self.dice_weight * soft_dice_loss_with_logits(logits, target, mask)
        loss = loss + self.consistency_weight * vessel_consistency_loss(logits, mask)
        return loss

    def forward(self, predictions, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not isinstance(predictions, (list, tuple)):
            return self._single(predictions, target, mask)
        losses = []
        weights = []
        for i, logits in enumerate(predictions):
            weights.append(self.refinement_decay ** (len(predictions) - i - 1))
            losses.append(self._single(logits, target, mask))
        total_weight = sum(weights)
        return sum(weight * loss for weight, loss in zip(weights, losses)) / total_weight


def probabilities_from_logits(predictions: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    return torch.sigmoid(_last_prediction(predictions))


def weighted_masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    positive_weights: torch.Tensor,
) -> torch.Tensor:
    weights = positive_weights.to(device=logits.device, dtype=logits.dtype).view(1, -1, 1, 1)
    mask3 = mask.expand_as(target)
    positive = -weights * target * F.logsigmoid(logits)
    negative = -(1.0 - target) * F.logsigmoid(-logits)
    return ((positive + negative) * mask3).sum() / mask3.sum().clamp_min(1.0)


def exclusive_av_classification_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    artery = target[:, 0:1] > 0.5
    vein = target[:, 2:3] > 0.5
    exclusive = torch.logical_xor(artery, vein) & (mask > 0.5)
    if not bool(exclusive.any()):
        return logits.sum() * 0.0
    av_logits = torch.cat((logits[:, 0:1], logits[:, 2:3]), dim=1)
    labels = vein.long().squeeze(1)
    selected = exclusive.squeeze(1)
    return F.cross_entropy(av_logits.permute(0, 2, 3, 1)[selected], labels[selected])


def vessel_hierarchy_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    artery = probabilities[:, 0:1]
    vessel = probabilities[:, 1:2]
    vein = probabilities[:, 2:3]
    violations = F.relu(artery - vessel) + F.relu(vein - vessel)
    return (violations * mask).sum() / mask.sum().clamp_min(1.0)


def _soft_skeleton(probability: torch.Tensor, iterations: int = 3) -> torch.Tensor:
    def soft_erode(value):
        horizontal = -F.max_pool2d(-value, (3, 1), stride=1, padding=(1, 0))
        vertical = -F.max_pool2d(-value, (1, 3), stride=1, padding=(0, 1))
        return torch.minimum(horizontal, vertical)

    def soft_open(value):
        eroded = soft_erode(value)
        return F.max_pool2d(eroded, 3, stride=1, padding=1)

    skeleton = F.relu(probability - soft_open(probability))
    value = probability
    for _ in range(iterations):
        value = soft_erode(value)
        delta = F.relu(value - soft_open(value))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, iterations: int = 3) -> torch.Tensor:
    probability = torch.sigmoid(logits[:, 1:2]) * mask
    ground_truth = target[:, 1:2] * mask
    pred_skeleton = _soft_skeleton(probability, iterations=iterations)
    target_skeleton = _soft_skeleton(ground_truth, iterations=iterations)
    precision = (pred_skeleton * ground_truth).sum() / pred_skeleton.sum().clamp_min(1e-6)
    sensitivity = (target_skeleton * probability).sum() / target_skeleton.sum().clamp_min(1e-6)
    cldice = (2.0 * precision * sensitivity) / (precision + sensitivity).clamp_min(1e-6)
    return 1.0 - cldice


class BalancedRecursiveLoss(nn.Module):
    def __init__(
        self,
        positive_weights,
        bce_weight: float = 0.25,
        dice_weight: float = 0.15,
        tversky_weight: float = 0.60,
        classification_weight: float = 0.15,
        hierarchy_weight: float = 0.05,
        topology_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer("positive_weights", torch.as_tensor(positive_weights, dtype=torch.float32))
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.tversky_weight = float(tversky_weight)
        self.classification_weight = float(classification_weight)
        self.hierarchy_weight = float(hierarchy_weight)
        self.topology_weight = float(topology_weight)

    def _stage(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bce = weighted_masked_bce_with_logits(logits, target, mask, self.positive_weights)
        dice = soft_dice_loss_with_logits(logits, target, mask)
        tversky = focal_tversky_loss_with_logits(logits, target, mask)
        return self.bce_weight * bce + self.dice_weight * dice + self.tversky_weight * tversky

    def forward(self, predictions, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        stages = list(predictions) if isinstance(predictions, (list, tuple)) else [predictions]
        weights = recursive_stage_weights(len(stages))
        loss = sum(weight * self._stage(logits, target, mask) for weight, logits in zip(weights, stages))
        final = stages[-1]
        loss = loss + self.classification_weight * exclusive_av_classification_loss(final, target, mask)
        loss = loss + self.hierarchy_weight * vessel_hierarchy_loss(final, mask)
        if self.topology_weight > 0:
            loss = loss + self.topology_weight * soft_cldice_loss(final, target, mask)
        return loss
