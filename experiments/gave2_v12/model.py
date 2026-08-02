from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(value) + self.projection(value))


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.block = ResidualBlock(out_channels, out_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(self.downsample(value))


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.block = ResidualBlock(out_channels + skip_channels, out_channels)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(self.reduce(value), size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((value, skip), dim=1))


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int
    base_channels: int = 20
    max_logit_delta: float = 2.5
    corridor_radius: int = 2
    correction_mode: str = "prune"
    activation_checkpointing: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def teacher_corridor(teacher: torch.Tensor, radius: int = 4, threshold: float = 0.5) -> torch.Tensor:
    if teacher.ndim != 4 or teacher.shape[1] != 3:
        raise ValueError(f"Expected [N,3,H,W] teacher, got {tuple(teacher.shape)}")
    binary = (teacher >= threshold).to(teacher.dtype)
    if radius <= 0:
        return binary
    return F.max_pool2d(binary, kernel_size=2 * radius + 1, stride=1, padding=radius)


class TopologyResidualUNet(nn.Module):
    """A bounded residual refiner whose zero state exactly reproduces V8."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.correction_mode not in {"prune", "vessel_support"}:
            raise ValueError(f"Unknown correction mode: {config.correction_mode}")
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
        self.residual_head = nn.Conv2d(base, 3, 1)
        self.crossing_head = nn.Conv2d(base, 1, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.crossing_head.weight)
        nn.init.constant_(self.crossing_head.bias, -6.0)

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
        corridor: torch.Tensor | None = None,
        *,
        protect_teacher: bool = True,
    ) -> dict[str, torch.Tensor]:
        epsilon = 1e-4
        teacher = teacher.clamp(epsilon, 1.0 - epsilon)
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
        raw_delta = self.residual_head(value)
        bounded_delta = self.config.max_logit_delta * torch.tanh(raw_delta)
        teacher_logits = torch.logit(teacher)
        probability = torch.sigmoid(teacher_logits + bounded_delta)
        if corridor is None:
            corridor = teacher_corridor(teacher, radius=self.config.corridor_radius)
        elif corridor.shape != teacher.shape:
            raise ValueError(f"Corridor shape {tuple(corridor.shape)} does not match teacher {tuple(teacher.shape)}")
        if protect_teacher:
            teacher_positive = teacher >= 0.5
            if self.config.correction_mode == "prune":
                allowed_positive = teacher_positive
            else:
                vessel_support = teacher[:, 1:2] >= 0.5
                allowed_positive = vessel_support.expand_as(teacher)
            probability = torch.where(
                allowed_positive,
                probability,
                torch.minimum(probability, teacher),
            )
            protected_path = (corridor > 0.5) & teacher_positive
            probability = torch.where(protected_path, torch.maximum(probability, teacher), probability)
        artery, vessel, vein = probability[:, 0:1], probability[:, 1:2], probability[:, 2:3]
        vessel = torch.maximum(vessel, torch.maximum(artery, vein))
        probability = torch.cat((artery, vessel, vein), dim=1) * mask
        crossing_logit = self.crossing_head(value)
        return {
            "probability": probability,
            "raw_delta": raw_delta,
            "bounded_delta": bounded_delta,
            "crossing_logit": crossing_logit,
            "crossing_probability": torch.sigmoid(crossing_logit) * mask,
            "corridor": corridor * mask,
        }


def build_model(config: dict[str, object] | ModelConfig) -> TopologyResidualUNet:
    if not isinstance(config, ModelConfig):
        config = ModelConfig(**config)
    return TopologyResidualUNet(config)
