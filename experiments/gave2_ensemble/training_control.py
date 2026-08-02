from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    mode: str = "max"
    patience: int = 35
    min_delta: float = 1e-4
    best: float | None = None
    stale_epochs: int = 0
    best_epoch: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        if self.patience < 1:
            raise ValueError("patience must be at least 1")

    @property
    def should_stop(self) -> bool:
        return self.stale_epochs >= self.patience

    def update(self, value: float, epoch: int | None = None) -> bool:
        if self.best is None:
            self.best = float(value)
            self.best_epoch = epoch
            self.stale_epochs = 0
            return True

        value = float(value)
        if self.mode == "max":
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta

        if improved:
            self.best = value
            self.best_epoch = epoch
            self.stale_epochs = 0
            return True

        self.stale_epochs += 1
        return False
