from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from .data_v6 import GAVE2DatasetV6
from .train_v6 import (
    _import_torch,
    _move_batch,
    accumulation_plan,
    autocast_context,
    compute_fold_positive_weights_v6,
    compute_roi_channel_stats_v6,
    load_fold_manifest,
    preferred_batch_profiles,
    select_batch_profile,
    set_seeds,
    torch_collate_v6,
    validate_fold_manifest,
)


def memory_profile_plan(total_batches: int, grad_accum: int, optimizer_steps: int) -> dict[str, object]:
    if optimizer_steps < 1:
        raise ValueError("optimizer_steps must be at least 1")
    epoch_plan = accumulation_plan(total_batches=total_batches, grad_accum=grad_accum)
    windows = []
    current_window = []
    current_size = None
    for item in epoch_plan:
        current_window.append(int(item["step_index"]))
        current_size = int(item["window_size"])
        if bool(item["should_step"]):
            windows.append({"window_size": current_size, "batches": current_window})
            current_window = []
            current_size = None
    selected = windows[:optimizer_steps]
    return {
        "window_sizes": [window["window_size"] for window in selected],
        "step_indices": [window["batches"][-1] for window in selected],
        "batches_per_step": [window["batches"] for window in selected],
    }


def _profile_single_batch(args: argparse.Namespace, profile, device: str):
    from .cmrrwnet_v6 import create_cmrrwnet_v6
    from .losses import BalancedRecursiveLoss

    torch, DataLoader = _import_torch()
    manifest = load_fold_manifest(args.fold_manifest)
    case_paths = sorted((Path(args.data_root) / "training" / "images").glob("*.png"))
    validate_fold_manifest(manifest, [path.stem for path in case_paths])
    train_ids = list(manifest["folds"][0]["training"])
    required_batches = max(1, (int(args.steps) - 1) * int(profile.grad_accum) + 1)
    required_cases = min(len(train_ids), max(int(profile.batch_size), int(profile.batch_size) * required_batches))
    dataset = GAVE2DatasetV6(
        args.data_root,
        split="training",
        task=args.task,
        case_ids=train_ids[:required_cases],
    )
    normalization = compute_roi_channel_stats_v6(args.data_root, args.task, train_ids)
    positive_weights = compute_fold_positive_weights_v6(dataset)
    loader = DataLoader(dataset, batch_size=profile.batch_size, shuffle=False, num_workers=0, collate_fn=torch_collate_v6)

    set_seeds(77)
    model = create_cmrrwnet_v6(
        task=args.task,
        base_channels=args.base_channels,
        num_refinements=args.num_refinements,
        activation_checkpointing=not args.no_activation_checkpointing,
        official_source=args.official_source,
    ).to(device)
    criterion = BalancedRecursiveLoss(positive_weights=positive_weights, topology_weight=0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    completed = 0
    total_batches = len(loader)
    if total_batches < 1:
        raise RuntimeError("Memory profile loader is empty")
    batch_plan = accumulation_plan(total_batches=total_batches, grad_accum=profile.grad_accum)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    while completed < args.steps:
        for batch_state, batch in zip(batch_plan, loader):
            batch = _move_batch(batch, device, normalization)
            divisor = int(batch_state["window_size"])
            with autocast_context(torch, device, "bf16"):
                predictions = model(batch["images"])
                loss = criterion(predictions, batch["targets"], batch["masks"]) / max(divisor, 1)
            loss.backward()
            if bool(batch_state["should_step"]):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                completed += 1
                if completed >= args.steps:
                    break

    torch.cuda.synchronize()
    report = {
        "task": args.task,
        "batch_size": profile.batch_size,
        "grad_accum": profile.grad_accum,
        "steps": completed,
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "gpu": torch.cuda.get_device_name(0),
    }
    del model, criterion, optimizer, loader, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return report


def run_memory_test(args: argparse.Namespace) -> dict[str, object]:
    torch, _ = _import_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("The production memory test requires a CUDA GPU")
    device = "cuda:0"
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    total_gib = total_bytes / 1024**3
    if total_gib < args.minimum_total_gib:
        raise RuntimeError(f"GPU has {total_gib:.2f} GiB total, need at least {args.minimum_total_gib:.2f} GiB")

    selected, profile_report = select_batch_profile(lambda profile: _profile_single_batch(args, profile, device), preferred_batch_profiles())
    report = {
        "task": args.task,
        "gpu_total_gib": round(total_gib, 3),
        "gpu_free_gib_before": round(free_bytes / 1024**3, 3),
        "selected_batch_size": selected.batch_size,
        "selected_grad_accum": selected.grad_accum,
        **profile_report,
    }
    print(json.dumps(report, indent=2))
    if report["peak_allocated_gib"] > args.max_allocated_gib or report["peak_reserved_gib"] > args.max_reserved_gib:
        raise RuntimeError(
            f"Memory profile rejected: allocated={report['peak_allocated_gib']:.2f} GiB, reserved={report['peak_reserved_gib']:.2f} GiB"
        )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile V6 full-resolution CMRRWNet training memory on CUDA.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--task", choices=("task1", "task2"), default="task2")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--num-refinements", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--minimum-total-gib", type=float, default=20.0)
    parser.add_argument("--max-allocated-gib", type=float, default=18.5)
    parser.add_argument("--max-reserved-gib", type=float, default=20.5)
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    parser.add_argument("--official-source", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    run_memory_test(parse_args())


if __name__ == "__main__":
    main()
