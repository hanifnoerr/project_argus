from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Callable

import numpy as np

from .data import list_case_ids
from .data_v6 import GAVE2DatasetV6
from .predict_v2 import project_probabilities
from .submission import save_probability_png
from .training_state_v6 import hash_file_bytes


FOUR_TTA_TRANSFORMS = [
    {"name": "identity"},
    {"name": "horizontal"},
    {"name": "vertical"},
    {"name": "both"},
]


def _import_torch():
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("CMRRWNet v6 prediction requires PyTorch") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")


def _atomic_write_json(path: Path, payload: object) -> Path:
    data = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    return _atomic_write_bytes(path, data)


def _atomic_write_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _atomic_save_npy(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _normalize_chw(array: np.ndarray, *, expected_channels: int | None = None) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.ndim != 3:
        raise ValueError(f"Expected CHW array, got {value.shape}")
    if expected_channels is not None and int(value.shape[0]) != int(expected_channels):
        raise ValueError(f"Expected {expected_channels} channels, got {value.shape[0]}")
    if not np.isfinite(value).all():
        raise ValueError("Array contains non-finite values")
    return np.ascontiguousarray(value)


def _normalize_probability(probability: np.ndarray) -> np.ndarray:
    return _normalize_chw(probability, expected_channels=3)


def _hash_probability_array(probability: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(probability).tobytes())


def _load_json(path: Path | str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply_tta_transform(value_chw: np.ndarray, transform: str) -> np.ndarray:
    value = _normalize_chw(value_chw)
    if transform == "identity":
        return value.copy()
    if transform == "horizontal":
        return np.ascontiguousarray(np.flip(value, axis=2))
    if transform == "vertical":
        return np.ascontiguousarray(np.flip(value, axis=1))
    if transform == "both":
        return np.ascontiguousarray(np.flip(np.flip(value, axis=1), axis=2))
    raise ValueError(f"Unsupported TTA transform {transform!r}")


def invert_tta_transform(value_chw: np.ndarray, transform: str) -> np.ndarray:
    return apply_tta_transform(value_chw, transform)


def average_tta_probabilities(probabilities: list[np.ndarray], expected_transforms: int = 4) -> np.ndarray:
    if len(probabilities) != int(expected_transforms):
        raise ValueError(f"Expected {expected_transforms} TTA probabilities, got {len(probabilities)}")
    stacked = np.stack([_normalize_probability(item) for item in probabilities], axis=0)
    return np.ascontiguousarray(stacked.mean(axis=0, dtype=np.float32), dtype=np.float32)


def average_fold_probabilities(probabilities: list[np.ndarray], expected_folds: int = 3) -> np.ndarray:
    if len(probabilities) != int(expected_folds):
        raise ValueError(f"Expected {expected_folds} fold probabilities, got {len(probabilities)}")
    stacked = np.stack([_normalize_probability(item) for item in probabilities], axis=0)
    return np.ascontiguousarray(stacked.mean(axis=0, dtype=np.float32), dtype=np.float32)


def validate_prediction_manifest(manifest: dict[str, object]) -> list[str]:
    if int(manifest.get("seed", -1)) != 77:
        raise ValueError("Prediction manifest must use seed 77")
    folds = manifest.get("folds")
    if not isinstance(folds, list) or len(folds) != 3:
        raise ValueError("Prediction manifest must contain exactly three folds")

    case_ids: set[str] = set()
    ownership: dict[str, int] = {}
    normalized_folds: list[dict[str, list[str]]] = []
    for expected_index, fold in enumerate(folds):
        fold_index = int(fold.get("fold_index", -1))
        if fold_index != expected_index:
            raise ValueError("Prediction manifest fold indices must be immutable and sequential")
        training = sorted(str(case_id) for case_id in fold.get("training", []))
        validation = sorted(str(case_id) for case_id in fold.get("validation", []))
        for case_id in validation:
            if case_id in ownership:
                raise ValueError("OOF case ownership must include each case exactly once")
            ownership[case_id] = fold_index
        if set(training) & set(validation):
            raise ValueError(f"Prediction manifest fold {fold_index} overlaps training and validation")
        case_ids.update(training)
        case_ids.update(validation)
        normalized_folds.append({"training": training, "validation": validation})

    all_case_ids = sorted(case_ids)
    if sorted(ownership) != all_case_ids:
        raise ValueError("OOF case ownership must include each case exactly once")

    for fold_index, fold in enumerate(normalized_folds):
        expected_training = sorted(case_id for case_id in all_case_ids if case_id not in set(fold["validation"]))
        if fold["training"] != expected_training:
            raise ValueError(f"Prediction manifest fold {fold_index} training membership must be the complement of validation")
    return all_case_ids


def oof_case_ownership(manifest: dict[str, object]) -> dict[str, int]:
    validate_prediction_manifest(manifest)
    ownership: dict[str, int] = {}
    for fold_index, fold in enumerate(manifest["folds"]):
        for case_id in fold["validation"]:
            ownership[str(case_id)] = fold_index
    return ownership


class FloatProbabilityStore:
    """Crash-safe float16 per-case probability cache with verified completion manifest."""

    def __init__(self, root: Path | str, *, task: str, split: str) -> None:
        self.root = Path(root)
        self.task = str(task)
        self.split = str(split)
        self.arrays_dir = self.root / "arrays"
        self.metadata_dir = self.root / "metadata"
        self.manifest_path = self.root / "completion_manifest.json"
        self.arrays_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            _atomic_write_json(self.manifest_path, self._empty_manifest())

    def _empty_manifest(self) -> dict[str, object]:
        return {
            "version": 6,
            "task": self.task,
            "split": self.split,
            "complete_cases": [],
            "cases": {},
        }

    def _load_manifest(self) -> dict[str, object]:
        if not self.manifest_path.exists():
            return self._empty_manifest()
        manifest = _load_json(self.manifest_path)
        if manifest.get("task") != self.task or manifest.get("split") != self.split:
            raise RuntimeError("Probability store manifest does not match the requested task/split")
        manifest.setdefault("complete_cases", [])
        manifest.setdefault("cases", {})
        return manifest

    def case_array_path(self, case_id: str) -> Path:
        return self.arrays_dir / f"{case_id}.npy"

    def case_metadata_path(self, case_id: str) -> Path:
        return self.metadata_dir / f"{case_id}.json"

    def write_case(self, case_id: str, probability: np.ndarray, provenance: dict[str, object] | None = None) -> dict[str, object]:
        value = _normalize_probability(probability).astype(np.float16)
        sha256 = _hash_probability_array(value)
        metadata = {
            "case_id": str(case_id),
            "task": self.task,
            "split": self.split,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": sha256,
            "provenance": provenance or {},
            "array_path": self.case_array_path(case_id).name,
        }
        _atomic_save_npy(self.case_array_path(case_id), value)
        _atomic_write_json(self.case_metadata_path(case_id), metadata)

        manifest = self._load_manifest()
        manifest["cases"][str(case_id)] = {
            "case_id": metadata["case_id"],
            "task": metadata["task"],
            "split": metadata["split"],
            "shape": metadata["shape"],
            "dtype": metadata["dtype"],
            "sha256": metadata["sha256"],
            "provenance": metadata["provenance"],
        }
        manifest["complete_cases"] = sorted(set(manifest["complete_cases"]) | {str(case_id)})
        _atomic_write_json(self.manifest_path, manifest)
        return metadata

    def _metadata_matches_expected(self, metadata: dict[str, object], expected: dict[str, object] | None) -> bool:
        if expected is None:
            return True
        for key in ("case_id", "task", "split"):
            if key in expected and metadata.get(key) != expected[key]:
                return False
        if "provenance" in expected and metadata.get("provenance") != expected["provenance"]:
            return False
        return True

    def is_case_complete(self, case_id: str, expected: dict[str, object] | None = None) -> bool:
        manifest = self._load_manifest()
        case_key = str(case_id)
        manifest_case = manifest.get("cases", {}).get(case_key)
        if manifest_case is None or case_key not in manifest.get("complete_cases", []):
            return False
        array_path = self.case_array_path(case_key)
        metadata_path = self.case_metadata_path(case_key)
        if not array_path.exists() or not metadata_path.exists():
            return False
        metadata = _load_json(metadata_path)
        if metadata.get("shape") != manifest_case.get("shape"):
            return False
        if metadata.get("dtype") != "float16":
            return False
        if not self._metadata_matches_expected(metadata, expected):
            return False
        if not self._metadata_matches_expected(manifest_case, expected):
            return False
        try:
            stored = np.load(array_path, allow_pickle=False)
        except Exception:
            return False
        if tuple(stored.shape) != tuple(metadata["shape"]):
            return False
        if str(stored.dtype) != metadata["dtype"]:
            return False
        return _hash_probability_array(stored) == metadata["sha256"] == manifest_case.get("sha256")

    def repair_case(self, case_id: str) -> None:
        case_key = str(case_id)
        _remove_if_exists(self.case_array_path(case_key))
        _remove_if_exists(self.case_metadata_path(case_key))
        manifest = self._load_manifest()
        manifest.get("cases", {}).pop(case_key, None)
        manifest["complete_cases"] = [value for value in manifest.get("complete_cases", []) if value != case_key]
        _atomic_write_json(self.manifest_path, manifest)

    def pending_case_ids(self, expected_case_ids_or_records) -> list[str]:
        if isinstance(expected_case_ids_or_records, dict):
            expected_records = {str(case_id): dict(record) for case_id, record in expected_case_ids_or_records.items()}
        else:
            expected_records = {
                str(case_id): {"case_id": str(case_id), "task": self.task, "split": self.split}
                for case_id in expected_case_ids_or_records
            }
        pending: list[str] = []
        for case_id, expected in expected_records.items():
            if self.is_case_complete(case_id, expected=expected):
                continue
            self.repair_case(case_id)
            pending.append(case_id)
        return pending

    def list_complete_cases(self) -> list[str]:
        manifest = self._load_manifest()
        return sorted(str(case_id) for case_id in manifest.get("complete_cases", []))

    def read_case(self, case_id: str) -> np.ndarray:
        if not self.is_case_complete(case_id):
            raise FileNotFoundError(f"Case {case_id} is not complete in the float probability store")
        stored = np.load(self.case_array_path(case_id), allow_pickle=False)
        return np.ascontiguousarray(stored.astype(np.float32))


def normalize_sample_image(sample, normalization: dict[str, list[float]]) -> np.ndarray:
    image = _normalize_chw(sample.image)
    mask = _normalize_chw(sample.mask, expected_channels=1)
    mean = np.asarray(normalization["mean"], dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(normalization["std"], dtype=np.float32).reshape(-1, 1, 1)
    if mean.shape[0] != image.shape[0] or std.shape[0] != image.shape[0]:
        raise ValueError("Normalization channel counts do not match the sample image")
    return np.ascontiguousarray(((image - mean) / np.maximum(std, 1e-6)) * mask, dtype=np.float32)


def _default_predictor(model, sample, normalized_image: np.ndarray, device: str, transform_name: str) -> np.ndarray:
    torch = _import_torch()
    from .losses import probabilities_from_logits

    del sample, transform_name
    batch = torch.from_numpy(np.ascontiguousarray(normalized_image)[None, ...]).to(device)
    with torch.inference_mode():
        probability = probabilities_from_logits(model(batch)).float().cpu().numpy()[0]
    return np.ascontiguousarray(probability, dtype=np.float32)


def infer_case_probability(
    sample,
    fold_records: list[dict[str, object]],
    loaded_models: dict[int, object],
    device: str,
    *,
    predictor: Callable[[object, object, np.ndarray, str, str], np.ndarray] | None = None,
) -> np.ndarray:
    predict_fn = _default_predictor if predictor is None else predictor
    fold_probabilities: list[np.ndarray] = []
    for fold_record in fold_records:
        model = loaded_models[int(fold_record["fold_index"])]
        normalized = normalize_sample_image(sample, fold_record["config"]["normalization"])
        tta_probabilities: list[np.ndarray] = []
        for transform in FOUR_TTA_TRANSFORMS:
            transformed_image = apply_tta_transform(normalized, transform["name"])
            predicted = _normalize_probability(predict_fn(model, sample, transformed_image, device, transform["name"]))
            restored = invert_tta_transform(predicted, transform["name"])
            tta_probabilities.append(restored)
        fold_probabilities.append(average_tta_probabilities(tta_probabilities, expected_transforms=len(FOUR_TTA_TRANSFORMS)))
    return average_fold_probabilities(fold_probabilities, expected_folds=len(fold_records))


def _load_certified_model(fold_record: dict[str, object], device: str):
    from .train_v6 import build_model_from_config

    torch = _import_torch()
    checkpoint_path = Path(fold_record["checkpoint_path"])
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        checkpoint = torch.load(checkpoint_path, map_location=device)
    config = fold_record["config"]
    checkpoint_config = checkpoint.get("config", {})
    if checkpoint_config.get("config_content_sha256") != config.get("config_content_sha256"):
        raise RuntimeError(f"Checkpoint/config mismatch: {checkpoint_path}")
    model = build_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def load_certified_folds(
    run_dir: Path | str,
    task: str,
    fold_manifest: Path | str,
    *,
    expected_folds: int = 3,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest = _load_json(fold_manifest)
    all_case_ids = validate_prediction_manifest(manifest)
    manifest_hash = hash_file_bytes(fold_manifest)
    task_root = Path(run_dir) / "cmrrwnet_v6" / str(task)
    fold_records: list[dict[str, object]] = []
    for fold_index in range(expected_folds):
        fold_dir = task_root / f"fold_{fold_index}"
        checkpoint_path = fold_dir / "best.pt"
        config_path = fold_dir / "config.json"
        certified_path = fold_dir / "best.certified.json"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        if not certified_path.is_file():
            raise FileNotFoundError(certified_path)
        config = _load_json(config_path)
        certified = _load_json(certified_path)
        fold = manifest["folds"][fold_index]
        if str(config.get("task")) != str(task):
            raise ValueError(f"Fold {fold_index} config task does not match {task}")
        if int(config.get("fold_index", -1)) != fold_index:
            raise ValueError(f"Fold {fold_index} config index mismatch")
        if config.get("fold_manifest_sha256") != manifest_hash:
            raise ValueError(f"Fold {fold_index} manifest hash mismatch")
        if sorted(str(case_id) for case_id in config.get("train_ids", [])) != sorted(str(case_id) for case_id in fold["training"]):
            raise ValueError(f"Fold {fold_index} train_ids do not match the manifest")
        if sorted(str(case_id) for case_id in config.get("val_ids", [])) != sorted(str(case_id) for case_id in fold["validation"]):
            raise ValueError(f"Fold {fold_index} val_ids do not match the manifest")
        if certified.get("config_content_sha256") != config.get("config_content_sha256"):
            raise ValueError(f"Fold {fold_index} certified config hash mismatch")
        if "normalization" not in config:
            raise ValueError(f"Fold {fold_index} is missing normalization")
        fold_records.append(
            {
                "fold_index": fold_index,
                "checkpoint_path": checkpoint_path,
                "config_path": config_path,
                "certified_path": certified_path,
                "config": config,
                "certified": certified,
            }
        )
    if sorted(oof_case_ownership(manifest)) != all_case_ids:
        raise ValueError("Manifest validation ownership does not cover every case")
    return manifest, fold_records


def _prediction_case_ids(args: argparse.Namespace, ownership: dict[str, int], dataset_split: str) -> list[str]:
    if args.case_ids:
        return list(args.case_ids)
    if args.mode == "oof":
        return sorted(ownership)
    return list_case_ids(args.data_root, split=dataset_split)


def _expected_case_record(
    case_id: str,
    *,
    task: str,
    split: str,
    fold_records: list[dict[str, object]],
    manifest_hash: str,
) -> dict[str, object]:
    return {
        "case_id": str(case_id),
        "task": str(task),
        "split": str(split),
        "provenance": {
            "fold_indices": [int(record["fold_index"]) for record in fold_records],
            "transform_names": [item["name"] for item in FOUR_TTA_TRANSFORMS],
            "manifest_sha256": str(manifest_hash),
            "checkpoint_config_sha256": [str(record["config"]["config_content_sha256"]) for record in fold_records],
        },
    }


def apply_calibration(probability: np.ndarray, calibration: dict[str, object], *, case_id: str | None = None) -> np.ndarray:
    from .calibration_v6 import apply_calibration as _apply_calibration

    return _apply_calibration(probability, calibration, case_id=case_id)


def prepare_probability_for_promotion(
    probability: np.ndarray,
    roi: np.ndarray,
    *,
    calibrator: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    value = _normalize_probability(probability)
    if calibrator is not None:
        value = _normalize_probability(calibrator(value))
    return np.ascontiguousarray(project_probabilities(value, roi), dtype=np.float32)


def promote_probability_to_png(
    probability: np.ndarray,
    roi: np.ndarray,
    output_path: Path | str,
    *,
    calibrator: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    prepared = prepare_probability_for_promotion(probability, roi, calibrator=calibrator)
    save_probability_png(prepared, output_path)
    return prepared


def _case_ids_from_text(value: str | None) -> list[str]:
    if value is None or not str(value).strip():
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V6 prediction inference, float cache, and promotion helper.")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--task", choices=("task1", "task2"), required=True)
    parser.add_argument("--mode", choices=("oof", "validation", "promote"), required=True)
    parser.add_argument("--fold-manifest", type=Path)
    parser.add_argument("--case-ids", type=str)
    parser.add_argument("--output-root", "--promote-root", dest="output_root", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--team-id", default="team")
    parser.add_argument("--expected-folds", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--source-split", choices=("oof", "validation"), default="validation")
    args = parser.parse_args(argv)
    args.case_ids = _case_ids_from_text(args.case_ids)
    return args


def _resolve_device(device: str) -> str:
    if str(device) != "auto":
        return str(device)
    torch = _import_torch()
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _run_inference_mode(
    args: argparse.Namespace,
    *,
    dataset_factory,
    model_loader,
    predictor,
) -> dict[str, object]:
    if args.data_root is None or args.run_dir is None or args.fold_manifest is None:
        raise ValueError("Inference mode requires --data-root, --run-dir, and --fold-manifest")
    injected_cpu_pipeline = model_loader is not _load_certified_model and predictor is not _default_predictor
    device = "cpu" if str(args.device) == "auto" and injected_cpu_pipeline else _resolve_device(args.device)
    manifest, fold_records = load_certified_folds(args.run_dir, args.task, args.fold_manifest, expected_folds=int(args.expected_folds))
    ownership = oof_case_ownership(manifest)
    manifest_hash = hash_file_bytes(args.fold_manifest)
    dataset_split = "training" if args.mode == "oof" else "validation"
    case_ids = _prediction_case_ids(args, ownership, dataset_split)
    dataset = dataset_factory(args.data_root, split=dataset_split, task=args.task, case_ids=case_ids, require_target=False)
    store = FloatProbabilityStore(args.store_root, task=args.task, split=args.mode)

    expected_records: dict[str, dict[str, object]] = {}
    used_fold_indices: set[int] = set()
    for case_id in case_ids:
        selected_folds = [fold_records[ownership[case_id]]] if args.mode == "oof" else list(fold_records)
        used_fold_indices.update(int(record["fold_index"]) for record in selected_folds)
        expected_records[case_id] = _expected_case_record(
            case_id,
            task=args.task,
            split=args.mode,
            fold_records=selected_folds,
            manifest_hash=manifest_hash,
        )

    pending = set(store.pending_case_ids(expected_records))
    loaded_models = {
        int(record["fold_index"]): model_loader(record, device)
        for record in fold_records
        if int(record["fold_index"]) in used_fold_indices
    }
    written_cases: list[str] = []
    for sample in dataset:
        if sample.case_id not in pending:
            continue
        selected_folds = [fold_records[ownership[sample.case_id]]] if args.mode == "oof" else list(fold_records)
        probability = infer_case_probability(sample, selected_folds, loaded_models, device, predictor=predictor)
        store.write_case(sample.case_id, probability, provenance=expected_records[sample.case_id]["provenance"])
        written_cases.append(sample.case_id)

    return {
        "task": args.task,
        "mode": args.mode,
        "store_root": str(args.store_root),
        "complete_cases": [case_id for case_id in case_ids if store.is_case_complete(case_id, expected_records[case_id])],
        "pending_cases": sorted(pending - set(written_cases)),
        "written_cases": sorted(written_cases),
    }


def _run_promotion_mode(args: argparse.Namespace, *, dataset_factory) -> dict[str, object]:
    if args.data_root is None or args.output_root is None or args.calibration is None:
        raise ValueError("Promotion mode requires --data-root, --output-root, and --calibration")
    calibration = _load_json(args.calibration)
    store = FloatProbabilityStore(args.store_root, task=args.task, split=args.source_split)
    case_ids = list(args.case_ids) if args.case_ids else store.list_complete_cases()
    dataset_split = "training" if args.source_split == "oof" else "validation"
    dataset = dataset_factory(args.data_root, split=dataset_split, task=args.task, case_ids=case_ids, require_target=False)
    task_name = "Task1" if args.task == "task1" else "Task2"
    output_dir = Path(args.output_root) / args.team_id / task_name
    output_dir.mkdir(parents=True, exist_ok=True)

    written_cases: list[str] = []
    for sample in dataset:
        probability = store.read_case(sample.case_id)
        promote_probability_to_png(
            probability,
            sample.mask[0],
            output_dir / f"{sample.case_id}.png",
            calibrator=lambda value, sample_case_id=sample.case_id: apply_calibration(value, calibration, case_id=sample_case_id),
        )
        written_cases.append(sample.case_id)
    return {
        "task": args.task,
        "mode": "promote",
        "output_dir": str(output_dir),
        "written_cases": sorted(written_cases),
    }


def run_prediction(
    args: argparse.Namespace,
    *,
    dataset_factory=GAVE2DatasetV6,
    model_loader: Callable[[dict[str, object], str], object] | None = None,
    predictor: Callable[[object, object, np.ndarray, str, str], np.ndarray] | None = None,
) -> dict[str, object]:
    if args.mode == "promote":
        return _run_promotion_mode(args, dataset_factory=dataset_factory)
    return _run_inference_mode(
        args,
        dataset_factory=dataset_factory,
        model_loader=_load_certified_model if model_loader is None else model_loader,
        predictor=_default_predictor if predictor is None else predictor,
    )


def main() -> None:
    print(json.dumps(run_prediction(parse_args()), indent=2))


if __name__ == "__main__":
    main()
