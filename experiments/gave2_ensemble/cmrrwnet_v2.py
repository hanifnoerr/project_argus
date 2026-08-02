from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def default_official_source() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "knowledge_base"
        / "sources"
        / "github"
        / "Peng2004_CMRRWNet"
        / "train"
        / "models.py"
    )


def load_official_module(source: Path | str | None = None):
    source_path = Path(source) if source is not None else default_official_source()
    if not source_path.is_file():
        raise FileNotFoundError(f"Official CMRRWNet source is required: {source_path}")
    spec = importlib.util.spec_from_file_location("gave2_cmrrwnet_v2_official", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create an import specification for {source_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RuntimeError(f"Failed to import official CMRRWNet source {source_path}: {exc}") from exc
    for name in ("UNetModule", "NewUNetModule"):
        if not hasattr(module, name):
            raise RuntimeError(f"Official CMRRWNet source {source_path} has no {name}")
    return module


def challenge_order_from_internal(internal_logits: torch.Tensor) -> torch.Tensor:
    """Convert official internal [artery, vein, vessel] to [artery, vessel, vein]."""

    if internal_logits.ndim != 4 or internal_logits.shape[1] != 3:
        raise ValueError(f"Expected BCHW logits with three channels, got {tuple(internal_logits.shape)}")
    return internal_logits[:, [0, 2, 1], :, :]


class CorrectedRecursiveCMRRWNet(nn.Module):
    """Official CMRRWNet modules with corrected channel semantics and bounded refinement.

    The copied baseline's recursive code treats its third internal channel as
    the fixed vessel tree, while the GAVE loader and submission use the green
    channel for the vessel tree. This wrapper keeps the recursive computation
    in its intended internal order and exposes every supervised prediction in
    challenge order.
    """

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
        in_channels = 3 if task == "task1" else 5
        first_cls = module.UNetModule if task == "task1" else module.NewUNetModule
        self.first_u = first_cls(in_channels, 3, base_channels)
        self.refiner = module.UNetModule(3, 2, base_channels)
        self.task = task
        self.base_channels = int(base_channels)
        self.num_refinements = int(num_refinements)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.official_source = str(Path(official_source) if official_source is not None else default_official_source())

    def _run_module(self, module: nn.Module, value: torch.Tensor) -> torch.Tensor:
        # Non-reentrant checkpointing computes parameter gradients even when
        # the image input itself does not require gradients. The first U-Net is
        # the largest activation consumer, so it must be checkpointed too.
        if self.training and self.activation_checkpointing:
            return checkpoint(module, value, use_reentrant=False)
        return module(value)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
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
            "base_channels": self.base_channels,
            "num_refinements": self.num_refinements,
            "activation_checkpointing": self.activation_checkpointing,
            "official_source": self.official_source,
        }


def create_cmrrwnet_v2(
    task: str,
    base_channels: int,
    num_refinements: int = 2,
    activation_checkpointing: bool = True,
    official_source: Path | str | None = None,
) -> CorrectedRecursiveCMRRWNet:
    return CorrectedRecursiveCMRRWNet(
        task=task,
        base_channels=base_channels,
        num_refinements=num_refinements,
        activation_checkpointing=activation_checkpointing,
        official_source=official_source,
    )
