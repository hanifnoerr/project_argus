from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.gave2_v12.utils import case_ids

from .data import V13Dataset, collate
from .losses import ChannelPathLoss
from .model import ModelConfig, build_model
from .train import _training_weights


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the V13 memory test")
    if args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported")
    dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    ids = case_ids(args.data_root, "training")[: args.batch_size]
    dataset = V13Dataset(
        args.data_root,
        "training",
        args.task,
        ids,
        args.teacher_store,
        args.prepared_root,
        augment=False,
        include_targets=True,
        preload_targets=True,
    )
    batch = collate([dataset[index] for index in range(len(ids))])
    device = "cuda"
    model = build_model(
        ModelConfig(
            input_channels=13 if args.task == "task2" else 8,
            base_channels=args.base_channels,
            activation_checkpointing=args.activation_checkpointing,
        )
    ).to(device).train()
    positive, states = _training_weights(args.data_root, case_ids(args.data_root, "training"))
    criterion = ChannelPathLoss(torch.from_numpy(positive).cuda(), torch.from_numpy(states).cuda()).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            output = model(batch["features"].cuda(), batch["teacher"].cuda(), batch["mask"].cuda())
            loss, _ = criterion(
                output,
                batch["target"].cuda(),
                batch["state_target"].cuda(),
                batch["centerline"].cuda(),
                batch["teacher"].cuda(),
                batch["mask"].cuda(),
            )
        loss.backward()
        optimizer.step()
    report = {
        "version": 13,
        "task": args.task,
        "base_channels": args.base_channels,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "loss": float(loss.detach()),
    }
    print(json.dumps(report, indent=2))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V13 full-canvas CUDA memory test.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--teacher-store", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--task", choices=("task1", "task2"), required=True)
    parser.add_argument("--base-channels", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--amp", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
