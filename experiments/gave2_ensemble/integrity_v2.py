from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


REQUIRED_MODALITIES = {
    "training": ("images", "masks", "FFA_A", "FFA_AV", "av", "biomarker"),
    "validation": ("images", "masks", "FFA_A", "FFA_AV"),
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_path_metadata(root: Path | str) -> str:
    root_path = Path(root)
    rows = []
    for path in sorted(
        candidate
        for candidate in root_path.rglob("*")
        if candidate.is_file()
        and "__pycache__" not in candidate.parts
        and candidate.suffix.lower() in {".py", ".json", ".txt", ".md", ".ipynb"}
    ):
        stat = path.stat()
        rows.append((path.relative_to(root_path).as_posix(), stat.st_size))
    return sha256_bytes(_canonical_json(rows).encode("utf-8"))


def _ids_for_directory(path: Path) -> list[str]:
    suffix = ".txt" if path.name == "biomarker" else ".png"
    return [item.stem for item in sorted(path.glob(f"*{suffix}"))]


def audit_dataset(
    data_root: Path | str,
    expected_training: int = 50,
    expected_validation: int = 50,
    expected_size: tuple[int, int] = (1024, 1536),
) -> dict[str, object]:
    root = Path(data_root)
    report: dict[str, object] = {"data_root": str(root.resolve()), "expected_size": list(expected_size)}
    split_ids: dict[str, list[str]] = {}
    for split, modalities in REQUIRED_MODALITIES.items():
        ids_by_modality: dict[str, list[str]] = {}
        for modality in modalities:
            directory = root / split / modality
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            ids_by_modality[modality] = _ids_for_directory(directory)
        reference = ids_by_modality["images"]
        mismatched = {name: ids for name, ids in ids_by_modality.items() if ids != reference}
        if mismatched:
            details = {name: len(ids) for name, ids in ids_by_modality.items()}
            raise ValueError(f"{split} modality IDs do not match: {details}")
        split_ids[split] = reference

        for modality in modalities:
            if modality == "biomarker":
                continue
            for case_id in reference:
                path = root / split / modality / f"{case_id}.png"
                with Image.open(path) as image:
                    actual = (image.height, image.width)
                if actual != expected_size:
                    raise ValueError(f"{path}: size {actual} != {expected_size}")

    if len(split_ids["training"]) != expected_training:
        raise ValueError(f"Expected {expected_training} training cases, found {len(split_ids['training'])}")
    if len(split_ids["validation"]) != expected_validation:
        raise ValueError(f"Expected {expected_validation} validation cases, found {len(split_ids['validation'])}")
    overlap = sorted(set(split_ids["training"]) & set(split_ids["validation"]))
    if overlap:
        raise ValueError(f"Training/validation ID overlap: {overlap}")

    report["training_ids"] = split_ids["training"]
    report["validation_ids"] = split_ids["validation"]
    report["metadata_sha256"] = hash_path_metadata(root)
    return report


def build_fold_manifest(case_ids: Iterable[str], n_folds: int = 5, seed: int = 77) -> dict[str, object]:
    ids = sorted(set(case_ids))
    if len(ids) < n_folds or n_folds < 2:
        raise ValueError("Need at least two folds and at least one case per fold")
    rng = np.random.default_rng(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    validation_chunks = [list(chunk) for chunk in np.array_split(shuffled, n_folds)]
    folds = []
    for index, validation in enumerate(validation_chunks):
        held_out = set(validation)
        training = [case_id for case_id in shuffled if case_id not in held_out]
        folds.append({"fold": index, "training": training, "validation": validation})
    manifest: dict[str, object] = {
        "version": 2,
        "seed": int(seed),
        "n_folds": int(n_folds),
        "case_ids": ids,
        "folds": folds,
    }
    manifest["sha256"] = fold_manifest_sha256(manifest)
    return manifest


def fold_manifest_sha256(manifest: dict[str, object]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "sha256"}
    return sha256_bytes(_canonical_json(payload).encode("utf-8"))


def validate_fold_manifest(
    manifest: dict[str, object],
    case_ids: Iterable[str],
    n_folds: int = 5,
) -> None:
    expected = sorted(set(case_ids))
    if manifest.get("n_folds") != n_folds:
        raise ValueError(f"Fold manifest has {manifest.get('n_folds')} folds, expected {n_folds}")
    if manifest.get("case_ids") != expected:
        raise ValueError("Fold manifest case IDs do not match the dataset")
    if manifest.get("sha256") != fold_manifest_sha256(manifest):
        raise ValueError("Fold manifest SHA-256 is invalid")
    folds = manifest.get("folds")
    if not isinstance(folds, list) or len(folds) != n_folds:
        raise ValueError("Fold manifest has an invalid fold list")
    held_out: list[str] = []
    for index, fold in enumerate(folds):
        if fold.get("fold") != index:
            raise ValueError(f"Expected fold index {index}, found {fold.get('fold')}")
        training = fold.get("training", [])
        validation = fold.get("validation", [])
        if set(training) & set(validation):
            raise ValueError(f"Fold {index} has training/validation leakage")
        if sorted(set(training) | set(validation)) != expected:
            raise ValueError(f"Fold {index} does not cover every case")
        held_out.extend(validation)
    if sorted(held_out) != expected or len(set(held_out)) != len(expected):
        raise ValueError("Fold validation sets do not partition every case exactly once")


def save_fold_manifest(path: Path | str, manifest: dict[str, object]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return destination


def load_fold_manifest(path: Path | str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_run_manifest(run_dir: Path | str, manifest: dict[str, object]) -> Path:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "run_manifest.json"
    normalized = json.loads(_canonical_json(manifest))
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise RuntimeError(f"Existing run manifest does not match requested run: {path}")
        return path
    path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def certified_checkpoints(
    run_dir: Path | str,
    task: str,
    expected_folds: int = 5,
    expected_model_class: str = "CorrectedRecursiveCMRRWNet",
) -> list[Path]:
    task = task.lower()
    if task not in {"task1", "task2"}:
        raise ValueError(f"Unsupported task {task!r}")
    task_root = Path(run_dir) / "cmrrwnet_v2" / task
    expected_names = [f"fold_{index}" for index in range(expected_folds)]
    actual_names = sorted(path.name for path in task_root.glob("fold_*") if path.is_dir())
    if actual_names != expected_names:
        raise FileNotFoundError(f"Expected fold directories {expected_names}, found {actual_names}")

    paths = []
    shared_fold_sha = None
    shared_model_signature = None
    hashes = set()
    for index, name in enumerate(expected_names):
        fold_dir = task_root / name
        checkpoint = fold_dir / "best.pt"
        config_path = fold_dir / "config.json"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Certified checkpoint missing: {checkpoint}; last.pt is not accepted")
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("task") != task or config.get("fold_index") != index:
            raise ValueError(f"Checkpoint config mismatch in {fold_dir}")
        if config.get("model_class") != expected_model_class:
            raise ValueError(f"Unexpected model class in {fold_dir}: {config.get('model_class')}")
        required = ("base_channels", "num_refinements", "preprocess", "normalization")
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(f"Missing certified configuration values in {config_path}: {missing}")
        signature = {
            "base_channels": config["base_channels"],
            "num_refinements": config["num_refinements"],
            "preprocess": config["preprocess"],
        }
        if shared_model_signature is None:
            shared_model_signature = signature
        elif signature != shared_model_signature:
            raise ValueError("Fold checkpoints do not share one model/preprocessing configuration")
        fold_sha = config.get("fold_manifest_sha256")
        if not fold_sha:
            raise ValueError(f"Missing fold manifest SHA-256 in {config_path}")
        if shared_fold_sha is None:
            shared_fold_sha = fold_sha
        elif fold_sha != shared_fold_sha:
            raise ValueError("Fold checkpoints do not share one fold manifest")
        checkpoint_sha = hash_file(checkpoint)
        if checkpoint_sha in hashes:
            raise ValueError("Two fold checkpoints have identical SHA-256 values")
        hashes.add(checkpoint_sha)
        paths.append(checkpoint)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the GAVE2 v2 dataset and build deterministic folds.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_dataset(args.data_root)
    manifest = build_fold_manifest(audit["training_ids"], n_folds=args.folds, seed=args.seed)
    validate_fold_manifest(manifest, audit["training_ids"], n_folds=args.folds)
    save_fold_manifest(args.run_dir / "fold_manifest.json", manifest)
    print(json.dumps({"audit": audit, "fold_manifest": manifest}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
