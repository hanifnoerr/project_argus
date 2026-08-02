from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .cmrrwnet_v2 import challenge_order_from_internal, default_official_source, load_official_module

try:
    from torch.utils.checkpoint import checkpoint
except ImportError:  # pragma: no cover - torch packaging variant
    checkpoint = None


class CorrectedRecursiveCMRRWNetV6(nn.Module):
    """Full-resolution V6 CMRRWNet with task-specific 4/6-channel inputs."""

    def __init__(
        self,
        task: str,
        base_channels: int,
        num_refinements: int = 2,
        activation_checkpointing: bool = True,
        official_source: Path | str | None = None,
    ) -> None:
        super().__init__()
        task = task.lower()
        if task not in {"task1", "task2"}:
            raise ValueError(f"Unsupported task {task!r}")
        if base_channels < 4:
            raise ValueError("base_channels must be at least 4")
        if num_refinements < 0:
            raise ValueError("num_refinements must be non-negative")

        module = load_official_module(official_source)
        in_channels = 4 if task == "task1" else 6
        first_cls = module.UNetModule if task == "task1" else module.NewUNetModule
        self.first_u = first_cls(in_channels, 3, base_channels)
        self.refiner = module.UNetModule(3, 2, base_channels)
        self.task = task
        self.input_channels = in_channels
        self.base_channels = int(base_channels)
        self.num_refinements = int(num_refinements)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.official_source = str(Path(official_source) if official_source is not None else default_official_source())

    def _run_module(self, module: nn.Module, value: torch.Tensor) -> torch.Tensor:
        if self.training and self.activation_checkpointing and checkpoint is not None:
            return checkpoint(module, value, use_reentrant=False)
        return module(value)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != self.input_channels:
            raise ValueError(f"Expected BCHW input with {self.input_channels} channels, got {tuple(image.shape)}")
        first_internal = self._run_module(self.first_u, image)
        if first_internal.shape[1] != 3:
            raise RuntimeError(f"First stage returned {first_internal.shape[1]} channels instead of three")

        vessel_logits = first_internal[:, 2:3]
        predictions = [challenge_order_from_internal(first_internal)]
        recursive_input = torch.sigmoid(first_internal)
        for _ in range(self.num_refinements):
            artery_vein_logits = self._run_module(self.refiner, recursive_input)
            if artery_vein_logits.shape[1] != 2:
                raise RuntimeError(
                    f"Refinement stage returned {artery_vein_logits.shape[1]} channels instead of two"
                )
            internal = torch.cat((artery_vein_logits, vessel_logits), dim=1)
            predictions.append(challenge_order_from_internal(internal))
            recursive_input = torch.cat((torch.sigmoid(artery_vein_logits), torch.sigmoid(vessel_logits)), dim=1)
        return predictions

    def metadata(self) -> dict[str, object]:
        return {
            "model_class": type(self).__name__,
            "task": self.task,
            "input_channels": self.input_channels,
            "base_channels": self.base_channels,
            "num_refinements": self.num_refinements,
            "activation_checkpointing": self.activation_checkpointing,
            "official_source": self.official_source,
        }


def create_cmrrwnet_v6(
    task: str,
    base_channels: int,
    num_refinements: int = 2,
    activation_checkpointing: bool = True,
    official_source: Path | str | None = None,
) -> CorrectedRecursiveCMRRWNetV6:
    return CorrectedRecursiveCMRRWNetV6(
        task=task,
        base_channels=base_channels,
        num_refinements=num_refinements,
        activation_checkpointing=activation_checkpointing,
        official_source=official_source,
    )
