from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import V12Dataset, collate
from .losses import ResidualChallengeLoss
from .model import ModelConfig, build_model
from .utils import case_ids


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The memory gate requires a CUDA GPU")
    if args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support BF16")
    dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    ids = case_ids(args.data_root, "training")[: args.batch_size]
    dataset = V12Dataset(
        args.data_root,
        "training",
        args.task,
        ids,
        args.teacher_store,
        args.prepared_root,
        augment=False,
        corridor_radius=args.corridor_radius,
    )
    batch = collate([dataset[index] for index in range(args.batch_size)])
    config = ModelConfig(
        input_channels=dataset.input_channels,
        base_channels=args.base_channels,
        corridor_radius=args.corridor_radius,
        correction_mode="prune",
        activation_checkpointing=args.activation_checkpointing,
    )
    device = "cuda"
    model = build_model(config).to(device).train()
    criterion = ResidualChallengeLoss(torch.tensor((20.0, 12.0, 20.0), device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        features = batch["features"].to(device)
        teacher = batch["teacher"].to(device)
        mask = batch["mask"].to(device)
        target = batch["target"].to(device)
        corridor = batch["corridor"].to(device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            output = model(features, teacher, mask, corridor)
            loss, _ = criterion(output, target, teacher, mask)
        loss.backward()
        optimizer.step()
    report = {
        "task": args.task,
        "base_channels": args.base_channels,
        "batch_size": args.batch_size,
        "activation_checkpointing": args.activation_checkpointing,
        "steps": args.steps,
        "loss": float(loss.detach()),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "device": torch.cuda.get_device_name(0),
    }
    print(json.dumps(report, indent=2))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-canvas V12 CUDA memory smoke test.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--teacher-store", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--task", choices=("task1", "task2"), required=True)
    parser.add_argument("--base-channels", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--corridor-radius", type=int, default=2)
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--steps", type=int, default=2)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
