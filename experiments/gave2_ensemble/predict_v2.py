from __future__ import annotations

import argparse
import gc
import json
import shutil
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .data import GAVE2Dataset, list_case_ids, torch_collate
from .integrity_v2 import certified_checkpoints, load_fold_manifest, validate_fold_manifest
from .submission import save_probability_png


def project_probabilities(probability_chw: np.ndarray, roi_hw: np.ndarray) -> np.ndarray:
    """Apply challenge hierarchy and ROI constraints without thresholding."""

    probability = np.asarray(probability_chw, dtype=np.float32)
    roi = np.asarray(roi_hw, dtype=np.float32)
    if probability.ndim != 3 or probability.shape[0] != 3:
        raise ValueError(f"Expected 3xHxW probabilities, got {probability.shape}")
    if roi.ndim == 3 and roi.shape[0] == 1:
        roi = roi[0]
    if roi.shape != probability.shape[1:]:
        raise ValueError(f"ROI shape {roi.shape} does not match probability shape {probability.shape[1:]}")
    if not np.isfinite(probability).all():
        raise ValueError("Probabilities contain non-finite values")

    output = np.clip(probability, 0.0, 1.0).copy()
    output[1] = np.maximum(output[1], np.maximum(output[0], output[2]))
    output *= (roi > 0.5)[None, :, :]
    return np.ascontiguousarray(output, dtype=np.float32)


class DiskProbabilityAccumulator:
    """Accumulate native-resolution fold probabilities without retaining them in RAM."""

    def __init__(self, root: Path | str, case_ids: list[str], shape: tuple[int, int, int]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shape = tuple(int(value) for value in shape)
        if len(self.shape) != 3 or self.shape[0] != 3:
            raise ValueError(f"Expected a 3xHxW accumulator shape, got {self.shape}")
        self.case_ids = list(case_ids)
        if len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("Accumulator case IDs must be unique")
        self.counts = {case_id: 0 for case_id in self.case_ids}
        for case_id in self.case_ids:
            path = self._path(case_id)
            array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=self.shape)
            array[:] = 0.0
            array.flush()
            del array

    def _path(self, case_id: str) -> Path:
        if case_id not in self.counts and hasattr(self, "counts"):
            raise KeyError(case_id)
        return self.root / f"{case_id}.npy"

    def add(self, case_id: str, probability: np.ndarray) -> None:
        if case_id not in self.counts:
            raise KeyError(case_id)
        value = np.asarray(probability, dtype=np.float32)
        if value.shape != self.shape:
            raise ValueError(f"{case_id}: expected {self.shape}, got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{case_id}: probabilities contain non-finite values")
        array = np.lib.format.open_memmap(self._path(case_id), mode="r+")
        array += value
        array.flush()
        del array
        self.counts[case_id] += 1

    def mean(self, case_id: str, expected_count: int) -> np.ndarray:
        if case_id not in self.counts:
            raise KeyError(case_id)
        actual = self.counts[case_id]
        if actual != expected_count:
            raise RuntimeError(f"{case_id}: expected {expected_count} predictions, accumulated {actual}")
        array = np.load(self._path(case_id), mmap_mode="r")
        result = np.asarray(array / float(expected_count), dtype=np.float32)
        return np.array(result, dtype=np.float32, copy=True)


def oof_case_ownership(manifest: dict[str, object]) -> dict[str, int]:
    ownership: dict[str, int] = {}
    folds = manifest.get("folds")
    if not isinstance(folds, list):
        raise ValueError("Fold manifest has no fold list")
    for fold in folds:
        fold_index = int(fold["fold"])
        for case_id in fold.get("validation", []):
            if case_id in ownership:
                raise ValueError(f"OOF case {case_id} belongs to more than one fold")
            ownership[str(case_id)] = fold_index
    expected = [str(case_id) for case_id in manifest.get("case_ids", [])]
    if sorted(ownership) != sorted(expected):
        raise ValueError("OOF ownership does not cover every manifest case exactly once")
    return ownership


def _import_torch():
    try:
        import torch
        from torch.utils.data import DataLoader

        return torch, DataLoader
    except ImportError as exc:  # pragma: no cover - Colab dependency
        raise RuntimeError("CMRRWNet v2 prediction requires PyTorch") from exc


def _load_checkpoint(torch, path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch
        return torch.load(path, map_location=device)


def _load_certified_model(checkpoint_path: Path, task: str, device: str):
    from .cmrrwnet_v2 import create_cmrrwnet_v2

    torch, _ = _import_torch()
    config_path = checkpoint_path.with_name("config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = _load_checkpoint(torch, checkpoint_path, "cpu")
    if checkpoint.get("config") != config:
        raise RuntimeError(f"Checkpoint/config mismatch: {checkpoint_path}")
    if config.get("task") != task or config.get("model_class") != "CorrectedRecursiveCMRRWNet":
        raise RuntimeError(f"Unexpected checkpoint configuration: {checkpoint_path}")
    model = create_cmrrwnet_v2(
        task=task,
        base_channels=int(config["base_channels"]),
        num_refinements=int(config["num_refinements"]),
        activation_checkpointing=False,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, config


def _autocast(torch, device: str):
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _predict_case(model, sample, normalization: dict[str, list[float]], device: str, tta: str) -> np.ndarray:
    from .losses import probabilities_from_logits

    torch, _ = _import_torch()
    batch = torch_collate([sample])
    image = batch["images"].to(device, non_blocking=True)
    mask = batch["masks"].to(device, non_blocking=True)
    mean = torch.as_tensor(normalization["mean"], dtype=image.dtype, device=device).view(1, -1, 1, 1)
    std = torch.as_tensor(normalization["std"], dtype=image.dtype, device=device).view(1, -1, 1, 1)
    image = ((image - mean) / std.clamp_min(1e-6)) * mask
    transforms = [(False, False)]
    if tta == "flips":
        transforms.extend([(True, False), (False, True), (True, True)])
    probabilities = []
    with torch.inference_mode():
        for flip_h, flip_w in transforms:
            value = image
            if flip_h:
                value = torch.flip(value, dims=(-2,))
            if flip_w:
                value = torch.flip(value, dims=(-1,))
            with _autocast(torch, device):
                prediction = probabilities_from_logits(model(value)).float()
            if flip_w:
                prediction = torch.flip(prediction, dims=(-1,))
            if flip_h:
                prediction = torch.flip(prediction, dims=(-2,))
            probabilities.append(prediction)
    result = torch.stack(probabilities).mean(dim=0)[0].cpu().numpy().astype(np.float32)
    return project_probabilities(result, sample.mask[0])


def _clean_task_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def _release_model(model, device: str) -> None:
    torch, _ = _import_torch()
    model.to("cpu")
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def run_prediction(args: argparse.Namespace) -> Path:
    torch, _ = _import_torch()
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoints = certified_checkpoints(args.run_dir, task=args.task, expected_folds=args.expected_folds)
    manifest = load_fold_manifest(args.fold_manifest)
    train_ids = list_case_ids(args.data_root, split="training")
    validate_fold_manifest(manifest, train_ids, n_folds=args.expected_folds)
    ownership = oof_case_ownership(manifest)
    calibration = None
    if args.calibration is not None:
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        if calibration.get("task") != args.task or len(calibration.get("thresholds", [])) != 3:
            raise ValueError(f"Invalid calibration for {args.task}: {args.calibration}")

    split = "training" if args.mode == "oof" else "validation"
    case_ids = train_ids if args.mode == "oof" else list_case_ids(args.data_root, split="validation")
    task_name = "Task1" if args.task == "task1" else "Task2"
    output_dir = Path(args.output_root) / "cmrrwnet_v2" / args.team_id / task_name
    _clean_task_directory(output_dir)

    if args.mode == "oof":
        for fold_index, checkpoint_path in enumerate(checkpoints):
            model, config = _load_certified_model(checkpoint_path, args.task, device)
            fold_ids = [case_id for case_id in case_ids if ownership[case_id] == fold_index]
            dataset = GAVE2Dataset(
                args.data_root,
                split=split,
                task=args.task,
                case_ids=fold_ids,
                require_target=False,
                preprocess=config["preprocess"],
            )
            for sample in dataset:
                probability = _predict_case(model, sample, config["normalization"], device, args.tta)
                if calibration is not None:
                    from .probability_calibration_v2 import apply_threshold_calibration

                    probability = project_probabilities(
                        apply_threshold_calibration(probability, calibration["thresholds"]), sample.mask[0]
                    )
                save_probability_png(probability, output_dir / f"{sample.case_id}.png")
                print(f"OOF fold {fold_index}: {sample.case_id}")
            _release_model(model, device)
    else:
        first_sample = GAVE2Dataset(
            args.data_root, split=split, task=args.task, case_ids=[case_ids[0]], require_target=False
        )[0]
        accumulator_dir = Path(args.accumulator_dir) / args.task
        if accumulator_dir.exists():
            shutil.rmtree(accumulator_dir)
        accumulator = DiskProbabilityAccumulator(accumulator_dir, case_ids, (3, *first_sample.original_size))
        for fold_index, checkpoint_path in enumerate(checkpoints):
            model, config = _load_certified_model(checkpoint_path, args.task, device)
            dataset = GAVE2Dataset(
                args.data_root,
                split=split,
                task=args.task,
                case_ids=case_ids,
                require_target=False,
                preprocess=config["preprocess"],
            )
            for sample in dataset:
                accumulator.add(
                    sample.case_id,
                    _predict_case(model, sample, config["normalization"], device, args.tta),
                )
                print(f"validation fold {fold_index}: {sample.case_id}")
            _release_model(model, device)
        roi_dataset = GAVE2Dataset(
            args.data_root, split=split, task=args.task, case_ids=case_ids, require_target=False
        )
        for sample in roi_dataset:
            probability = accumulator.mean(sample.case_id, len(checkpoints))
            if calibration is not None:
                from .probability_calibration_v2 import apply_threshold_calibration

                probability = apply_threshold_calibration(probability, calibration["thresholds"])
            probability = project_probabilities(probability, sample.mask[0])
            save_probability_png(probability, output_dir / f"{sample.case_id}.png")
        shutil.rmtree(accumulator_dir)

    actual = sorted(path.stem for path in output_dir.glob("*.png"))
    if actual != sorted(case_ids):
        raise RuntimeError(f"Prediction output is incomplete: expected {len(case_ids)}, found {len(actual)}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict CMRRWNet v2 fold-ensemble prediction.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--team-id", required=True)
    parser.add_argument("--task", choices=("task1", "task2"), required=True)
    parser.add_argument("--mode", choices=("oof", "validation"), required=True)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--tta", choices=("none", "flips"), default="flips")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--accumulator-dir", type=Path, default=Path(".prediction_accumulator"))
    parser.add_argument("--calibration", type=Path)
    return parser.parse_args()


def main() -> None:
    output = run_prediction(parse_args())
    print(f"Prediction complete: {output}")


if __name__ == "__main__":
    main()
