from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from experiments.gave2_v8.store import ProbabilityStore
from experiments.gave2_v12.folds import validate_manifest
from experiments.gave2_v12.utils import case_ids, sha256_file

from .data import V13Dataset, collate
from .model import build_model


def _load_checkpoint(path: Path, device: str):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().to(device), checkpoint


def _amp(device: str, name: str):
    import torch

    if not device.startswith("cuda") or name == "fp32":
        return None
    if name == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is not supported by this CUDA device")
        return torch.bfloat16
    return torch.float16


def _predict_tta(model, features, teacher, mask, amp_dtype, use_tta: bool):
    import torch

    transforms = ((False, False),) if not use_tta else ((False, False), (False, True), (True, False), (True, True))
    predictions = []
    for flip_y, flip_x in transforms:
        dims = tuple(dim for flag, dim in ((flip_y, -2), (flip_x, -1)) if flag)
        x = torch.flip(features, dims) if dims else features
        t = torch.flip(teacher, dims) if dims else teacher
        m = torch.flip(mask, dims) if dims else mask
        with torch.inference_mode(), torch.autocast(
            device_type="cuda" if features.device.type == "cuda" else "cpu",
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            probability = model(x, t, m)["probability"].float()
        predictions.append(torch.flip(probability, dims) if dims else probability)
    probability = torch.stack(predictions).mean(dim=0)
    artery, vessel, vein = probability[:, 0:1], probability[:, 1:2], probability[:, 2:3]
    return torch.cat((artery, torch.maximum(vessel, torch.maximum(artery, vein)), vein), dim=1) * mask


def run_prediction(args: argparse.Namespace) -> dict[str, object]:
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = _amp(device, args.amp)
    manifest = json.loads(args.fold_manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest, case_ids(args.data_root, "training"))
    ids = case_ids(args.data_root, args.split)
    teacher = ProbabilityStore(args.teacher_store, namespace="r2v2_direct", split=args.split)
    if teacher.list_cases() != ids:
        raise RuntimeError(f"Teacher store is incomplete for {args.split}")
    namespace = f"gave2_v13_raw_{args.task}"
    output = ProbabilityStore(args.output_store, namespace=namespace, split=args.split)
    checkpoint_paths = [args.run_dir / "models" / args.task / f"fold_{fold}" / "best.pt" for fold in range(3)]
    if not all(path.exists() for path in checkpoint_paths):
        raise FileNotFoundError(f"Missing V13 checkpoints: {[str(path) for path in checkpoint_paths if not path.exists()]}")
    checkpoints = [sha256_file(path) for path in checkpoint_paths]
    provenance = {
        case_id: {
            "version": 13,
            "task": args.task,
            "split": args.split,
            "tta": bool(args.tta),
            "checkpoint_sha256": checkpoints,
            "teacher_sha256": teacher.case_record(case_id)["sha256"],
        }
        for case_id in ids
    }
    pending = output.pending(ids, provenance)
    if not pending:
        return {"task": args.task, "split": args.split, "cases": len(ids), "new_cases": 0, "output": str(args.output_store)}

    loaded = [_load_checkpoint(path, device) for path in checkpoint_paths]
    fold_by_case = {
        case_id: int(fold["fold"])
        for fold in manifest["folds"]
        for case_id in fold["validation"]
    }
    dataset = V13Dataset(
        args.data_root,
        args.split,
        args.task,
        ids,
        args.teacher_store,
        args.prepared_root,
        augment=False,
        include_targets=False,
    )
    index_by_case = {case_id: index for index, case_id in enumerate(ids)}
    for sequence, case_id in enumerate(pending, 1):
        sample = collate([dataset[index_by_case[case_id]]])
        features = sample["features"].to(device)
        teacher_tensor = sample["teacher"].to(device)
        mask = sample["mask"].to(device)
        model_indices = [fold_by_case[case_id]] if args.split == "training" else [0, 1, 2]
        predictions = [
            _predict_tta(loaded[index][0], features, teacher_tensor, mask, amp_dtype, args.tta)
            for index in model_indices
        ]
        probability = torch.stack(predictions).mean(dim=0)[0].cpu().numpy().astype(np.float32)
        output.write_case(case_id, probability, provenance[case_id])
        print(f"[{sequence:02d}/{len(pending):02d}] {args.task} {args.split} {case_id}", flush=True)
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return {
        "version": 13,
        "task": args.task,
        "split": args.split,
        "cases": len(ids),
        "new_cases": len(pending),
        "output": str(args.output_store),
        "namespace": namespace,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict V13 OOF or validation probabilities.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--teacher-store", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--output-store", type=Path, required=True)
    parser.add_argument("--task", choices=("task1", "task2"), required=True)
    parser.add_argument("--split", choices=("training", "validation"), required=True)
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main() -> None:
    print(json.dumps(run_prediction(parse_args()), indent=2))


if __name__ == "__main__":
    main()
