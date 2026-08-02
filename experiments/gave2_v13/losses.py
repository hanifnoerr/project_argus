from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from experiments.gave2_v12.losses import dice_loss, soft_cldice_loss


def _masked_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / (weight.sum() + 1.0)


def probability_bce(
    probability: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    probability = probability.float().clamp(1e-5, 1.0 - 1e-5)
    target = target.float()
    loss = -(target * torch.log(probability) + (1.0 - target) * torch.log1p(-probability))
    return _masked_mean(loss, weight.float())


@dataclass(frozen=True)
class LossWeights:
    classification: float = 0.22
    state: float = 0.16
    dice: float = 0.18
    topology: float = 0.25
    centerline: float = 0.12
    teacher: float = 0.04
    residual: float = 0.03


class ChannelPathLoss(nn.Module):
    def __init__(
        self,
        positive_weights: torch.Tensor,
        state_weights: torch.Tensor,
        weights: LossWeights | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("positive_weights", positive_weights.float().clamp(1.0, 40.0))
        self.register_buffer("state_weights", state_weights.float().clamp(0.25, 60.0))
        self.weights = weights or LossWeights()

    def forward(
        self,
        output: dict[str, torch.Tensor],
        target: torch.Tensor,
        state_target: torch.Tensor,
        centerline: torch.Tensor,
        teacher: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        probability = output["probability"].float()
        av_index = torch.tensor((0, 2), device=probability.device)
        av_probability = probability.index_select(1, av_index)
        av_target = target.index_select(1, av_index).float()
        positive_weights = self.positive_weights.index_select(0, av_index)
        av_weights = 1.0 + av_target * (positive_weights[None, :, None, None] - 1.0)
        classification = probability_bce(av_probability, av_target, av_weights * mask)

        state_log_probability = torch.log_softmax(output["state_logits"].float(), dim=1)
        state_map = torch.nn.functional.nll_loss(
            state_log_probability,
            state_target.long(),
            weight=self.state_weights,
            reduction="none",
        )
        state = _masked_mean(state_map, mask[:, 0])
        dice = dice_loss(av_probability, av_target, mask.float())
        topology = soft_cldice_loss(av_probability, av_target, mask.float())

        centerline_weight = centerline.float() * mask
        centerline_positive = probability_bce(
            av_probability,
            torch.ones_like(av_probability),
            centerline_weight * 6.0,
        )
        wrong_class = torch.stack((av_probability[:, 1], av_probability[:, 0]), dim=1)
        exclusive = centerline * (1.0 - torch.stack((av_target[:, 1], av_target[:, 0]), dim=1))
        centerline_negative = probability_bce(
            wrong_class,
            torch.zeros_like(wrong_class),
            exclusive * mask * 2.0,
        )
        centerline_loss = centerline_positive + 0.5 * centerline_negative

        confident = ((teacher >= 0.92) | (teacher <= 0.01)).float() * mask
        teacher_loss = _masked_mean((probability - teacher.float()).abs(), confident)
        residual = _masked_mean(output["bounded_delta"].float().abs(), mask.expand_as(output["bounded_delta"]))
        components = {
            "classification": classification,
            "state": state,
            "dice": dice,
            "topology": topology,
            "centerline": centerline_loss,
            "teacher": teacher_loss,
            "residual": residual,
        }
        total = sum(getattr(self.weights, name) * value for name, value in components.items())
        return total, components
