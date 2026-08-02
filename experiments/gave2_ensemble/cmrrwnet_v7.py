from __future__ import annotations

from pathlib import Path

from .cmrrwnet_v6 import CorrectedRecursiveCMRRWNetV6


class ConditionalCMRRWNetV7(CorrectedRecursiveCMRRWNetV6):
    """Checkpoint-compatible CMRRWNet whose logits have conditional V7 semantics."""

    def metadata(self) -> dict[str, object]:
        metadata = super().metadata()
        metadata.update(
            {
                "model_class": type(self).__name__,
                "version": 7,
                "output_semantics": "vessel_sigmoid_plus_conditional_av_softmax",
            }
        )
        return metadata


def create_cmrrwnet_v7(
    task: str,
    base_channels: int,
    num_refinements: int = 2,
    activation_checkpointing: bool = True,
    official_source: Path | str | None = None,
) -> ConditionalCMRRWNetV7:
    return ConditionalCMRRWNetV7(
        task=task,
        base_channels=base_channels,
        num_refinements=num_refinements,
        activation_checkpointing=activation_checkpointing,
        official_source=official_source,
    )
