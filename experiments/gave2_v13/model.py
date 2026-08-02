from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from experiments.gave2_v12.model import Down, ResidualBlock, Up


STATE_NAMES = ("background", "artery", "vein", "crossing", "uncertain_vessel")


def teacher_to_states(teacher: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """Decompose A/BV/V probabilities into five mutually exclusive states."""

    if teacher.ndim != 4 or teacher.shape[1] != 3:
        raise ValueError(f"Expected [N,3,H,W] teacher, got {tuple(teacher.shape)}")
    artery = teacher[:, 0:1].clamp(0.0, 1.0)
    vessel = teacher[:, 1:2].clamp(0.0, 1.0)
    vein = teacher[:, 2:3].clamp(0.0, 1.0)
    crossing = torch.minimum(artery, vein)
    artery_only = (artery - crossing).clamp_min(0.0)
    vein_only = (vein - crossing).clamp_min(0.0)
    uncertain = (vessel - torch.maximum(artery, vein)).clamp_min(0.0)
    background = (1.0 - vessel).clamp_min(0.0)
    states = torch.cat((background, artery_only, vein_only, crossing, uncertain), dim=1)
    return (states + epsilon) / (states.sum(dim=1, keepdim=True) + 5.0 * epsilon)


def states_to_channels(states: torch.Tensor) -> torch.Tensor:
    if states.ndim != 4 or states.shape[1] != len(STATE_NAMES):
        raise ValueError(f"Expected [N,5,H,W] states, got {tuple(states.shape)}")
    artery = states[:, 1:2] + states[:, 3:4]
    vein = states[:, 2:3] + states[:, 3:4]
    vessel = 1.0 - states[:, 0:1]
    vessel = torch.maximum(vessel, torch.maximum(artery, vein))
    return torch.cat((artery, vessel, vein), dim=1).clamp(0.0, 1.0)


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int
    base_channels: int = 20
    max_state_delta: float = 3.0
    support_threshold: float = 0.15
    support_radius: int = 2
    activation_checkpointing: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ChannelCoupledResidualUNet(nn.Module):
    """Five-state residual model that exactly reproduces R2-V2 at epoch zero."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        base = config.base_channels
        self.stem = ResidualBlock(config.input_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.down4 = Down(base * 8, base * 12)
        self.bottleneck = ResidualBlock(base * 12, base * 12)
        self.up3 = Up(base * 12, base * 8, base * 8)
        self.up2 = Up(base * 8, base * 4, base * 4)
        self.up1 = Up(base * 4, base * 2, base * 2)
        self.up0 = Up(base * 2, base, base)
        self.state_delta_head = nn.Conv2d(base, len(STATE_NAMES), 1)
        nn.init.zeros_(self.state_delta_head.weight)
        nn.init.zeros_(self.state_delta_head.bias)

    def _run(self, module: nn.Module, *values: torch.Tensor) -> torch.Tensor:
        if self.config.activation_checkpointing and self.training and any(value.requires_grad for value in values):
            from torch.utils.checkpoint import checkpoint

            return checkpoint(module, *values, use_reentrant=False)
        return module(*values)

    def forward(
        self,
        features: torch.Tensor,
        teacher: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x0 = self._run(self.stem, features)
        x1 = self._run(self.down1, x0)
        x2 = self._run(self.down2, x1)
        x3 = self._run(self.down3, x2)
        x4 = self._run(self.down4, x3)
        value = self._run(self.bottleneck, x4)
        value = self._run(self.up3, value, x3)
        value = self._run(self.up2, value, x2)
        value = self._run(self.up1, value, x1)
        value = self._run(self.up0, value, x0)

        teacher_states = teacher_to_states(teacher.float()).to(value.dtype)
        raw_delta = self.state_delta_head(value)
        bounded_delta = self.config.max_state_delta * torch.tanh(raw_delta)
        state_logits = torch.log(teacher_states.clamp_min(1e-5)) + bounded_delta
        state_probability = torch.softmax(state_logits.float(), dim=1).to(value.dtype)
        probability = states_to_channels(state_probability.float()).to(value.dtype)

        vessel_support = teacher[:, 1:2] >= self.config.support_threshold
        if self.config.support_radius > 0:
            radius = self.config.support_radius
            vessel_support = F.max_pool2d(
                vessel_support.to(value.dtype),
                kernel_size=2 * radius + 1,
                stride=1,
                padding=radius,
            ) > 0.5
        support = vessel_support.expand_as(probability)
        probability = torch.where(support, probability, teacher)
        probability = probability * mask
        state_probability = state_probability * mask
        return {
            "probability": probability,
            "state_probability": state_probability,
            "state_logits": state_logits,
            "raw_delta": raw_delta,
            "bounded_delta": bounded_delta,
            "support": support.to(value.dtype) * mask,
        }


def build_model(config: dict[str, object] | ModelConfig) -> ChannelCoupledResidualUNet:
    if not isinstance(config, ModelConfig):
        config = ModelConfig(**config)
    return ChannelCoupledResidualUNet(config)

