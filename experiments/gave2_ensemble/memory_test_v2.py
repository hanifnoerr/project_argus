from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import GAVE2Dataset, compute_roi_channel_stats, list_case_ids, torch_collate
from .integrity_v2 import load_fold_manifest, validate_fold_manifest
from .train_v2 import (
    _import_torch,
    _move_batch,
    autocast_context,
    compute_fold_positive_weights,
    set_seeds,
)


def run_memory_test(args: argparse.Namespace) -> dict[str, float | int | str]:
    from .cmrrwnet_v2 import create_cmrrwnet_v2
    from .losses import BalancedRecursiveLoss

    torch, DataLoader = _import_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("The production memory test requires a CUDA GPU")
    device = "cuda:0"
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    total_gib = total_bytes / 1024**3
    free_gib = free_bytes / 1024**3
    if total_gib < args.minimum_total_gib:
        raise RuntimeError(f"GPU has {total_gib:.2f} GiB total, need at least {args.minimum_total_gib:.2f} GiB")

    manifest = load_fold_manifest(args.fold_manifest)
    case_ids = list_case_ids(args.data_root, split="training")
    validate_fold_manifest(manifest, case_ids, n_folds=int(manifest["n_folds"]))
    train_ids = manifest["folds"][0]["training"]
    dataset = GAVE2Dataset(
        args.data_root,
        split="training",
        task=args.task,
        case_ids=train_ids[: max(2, args.steps)],
        preprocess="none",
    )
    normalization = compute_roi_channel_stats(args.data_root, args.task, train_ids, preprocess="none")
    positive_weights = compute_fold_positive_weights(dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=torch_collate)

    set_seeds(77)
    model = create_cmrrwnet_v2(
        task=args.task,
        base_channels=args.base_channels,
        num_refinements=args.num_refinements,
        activation_checkpointing=True,
    ).to(device)
    criterion = BalancedRecursiveLoss(positive_weights, topology_weight=args.topology_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    completed = 0
    model.train()
    while completed < args.steps:
        for batch in loader:
            batch = _move_batch(batch, device, normalization)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(torch, device, "bf16"):
                predictions = model(batch["images"])
                loss = criterion(predictions, batch["targets"], batch["masks"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            completed += 1
            if completed >= args.steps:
                break

    torch.cuda.synchronize()
    allocated_gib = torch.cuda.max_memory_allocated() / 1024**3
    reserved_gib = torch.cuda.max_memory_reserved() / 1024**3
    report: dict[str, float | int | str] = {
        "gpu": torch.cuda.get_device_name(0),
        "total_gib": round(total_gib, 3),
        "free_gib_before": round(free_gib, 3),
        "task": args.task,
        "base_channels": args.base_channels,
        "num_refinements": args.num_refinements,
        "steps": completed,
        "peak_allocated_gib": round(allocated_gib, 3),
        "peak_reserved_gib": round(reserved_gib, 3),
    }
    print(json.dumps(report, indent=2))
    if allocated_gib > args.max_allocated_gib or reserved_gib > args.max_reserved_gib:
        raise RuntimeError(
            f"Memory profile rejected: allocated={allocated_gib:.2f} GiB, reserved={reserved_gib:.2f} GiB"
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure a production-equivalent CMRRWNet v2 L4 training step.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--task", choices=("task1", "task2"), default="task2")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--num-refinements", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--topology-weight", type=float, default=0.0)
    parser.add_argument("--minimum-total-gib", type=float, default=20.0)
    parser.add_argument("--max-allocated-gib", type=float, default=18.5)
    parser.add_argument("--max-reserved-gib", type=float, default=20.5)
    return parser.parse_args()


def main() -> None:
    run_memory_test(parse_args())


if __name__ == "__main__":
    main()
