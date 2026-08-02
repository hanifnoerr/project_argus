from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from .data_v6 import GAVE2DatasetV6
from .losses_v7 import ConditionalPathLossV7
from .train_v6 import (
    BatchProfile,
    _import_torch,
    _move_batch,
    autocast_context,
    compute_fold_positive_weights_v6,
    compute_roi_channel_stats_v6,
    load_fold_manifest,
    torch_collate_v6,
)


def test_profile(args: argparse.Namespace, batch_size: int, checkpointing: bool) -> dict[str, object]:
    from .cmrrwnet_v7 import create_cmrrwnet_v7

    torch, DataLoader = _import_torch()
    manifest = load_fold_manifest(args.fold_manifest)
    train_ids = list(manifest["folds"][0]["training"])
    dataset = GAVE2DatasetV6(args.data_root, split="training", task=args.task, case_ids=train_ids[:batch_size])
    normalization = compute_roi_channel_stats_v6(args.data_root, args.task, train_ids)
    positive_weights = compute_fold_positive_weights_v6(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=torch_collate_v6, num_workers=0)
    model = create_cmrrwnet_v7(args.task, args.base_channels, args.num_refinements, checkpointing).cuda().train()
    criterion = ConditionalPathLossV7(positive_weights).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.steps):
        for batch in loader:
            batch = _move_batch(batch, "cuda:0", normalization)
            with autocast_context(torch, "cuda:0", "bf16"):
                loss = criterion(model(batch["images"]), batch["targets"], batch["masks"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    report = {
        "batch_size": batch_size,
        "activation_checkpointing": checkpointing,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }
    del model, criterion, optimizer, loader, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return report


def run_memory_test(args: argparse.Namespace) -> dict[str, object]:
    torch, _ = _import_torch()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("V7 production training requires CUDA BF16 support")
    failures = []
    for batch_size in (6, 4, 2):
        for checkpointing in (False, True):
            try:
                report = test_profile(args, batch_size, checkpointing)
                total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if report["peak_reserved_gib"] <= total_gib * 0.92:
                    return {"task": args.task, "gpu": torch.cuda.get_device_name(0), **report}
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" not in str(exc).lower() and not isinstance(exc, torch.OutOfMemoryError):
                    raise
                failures.append({"batch_size": batch_size, "checkpointing": checkpointing, "error": str(exc)})
                torch.cuda.empty_cache()
    raise RuntimeError(f"No V7 BF16 profile fits: {failures}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hardware-neutral BF16 memory gate for V7.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--task", choices=("task1", "task2"), required=True)
    parser.add_argument("--base-channels", type=int, required=True)
    parser.add_argument("--num-refinements", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    return parser.parse_args(argv)


def main() -> None:
    print(json.dumps(run_memory_test(parse_args()), indent=2))


if __name__ == "__main__":
    main()
