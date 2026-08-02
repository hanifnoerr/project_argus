from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.gave2_ensemble.data import derive_av3_target

from .utils import atomic_json, case_ids, sha256_file


def _record(data_root: Path, case_id: str) -> dict[str, object]:
    label_path = data_root / "training" / "av" / f"{case_id}.png"
    mask_path = data_root / "training" / "masks" / f"{case_id}.png"
    raw = np.asarray(Image.open(label_path).convert("RGB"), dtype=np.float32) / 255.0
    target = derive_av3_target(raw)
    roi = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 127
    denominator = max(int(roi.sum()), 1)
    prevalence = [float((target[channel] * roi).sum() / denominator) for channel in range(3)]
    return {
        "case_id": case_id,
        "prevalence": prevalence,
        "label_sha256": sha256_file(label_path),
        "mask_sha256": sha256_file(mask_path),
    }


def build_manifest(data_root: Path | str, seed: int = 77) -> dict[str, object]:
    root = Path(data_root)
    records = [_record(root, case_id) for case_id in case_ids(root, "training")]
    if len(records) != 50:
        raise ValueError(f"Expected 50 training cases, found {len(records)}")
    vectors = np.asarray([record["prevalence"] for record in records], dtype=np.float64)
    mean = vectors.mean(axis=0)
    scale = np.maximum(vectors.std(axis=0), 1e-8)
    rng = np.random.default_rng(seed)
    order = sorted(
        range(len(records)),
        key=lambda index: (
            -float(np.linalg.norm((vectors[index] - mean) / scale)),
            float(rng.random()),
            str(records[index]["case_id"]),
        ),
    )
    target_sizes = (17, 17, 16)
    assignments: list[list[int]] = [[], [], []]
    sums = [np.zeros(3, dtype=np.float64) for _ in range(3)]
    for index in order:
        choices = []
        for fold, target_size in enumerate(target_sizes):
            if len(assignments[fold]) >= target_size:
                continue
            proposed = sums[fold] + vectors[index]
            expected = mean * (len(assignments[fold]) + 1)
            imbalance = float(np.square((proposed - expected) / scale).sum())
            fill = (len(assignments[fold]) + 1) / target_size
            choices.append((imbalance + 0.05 * fill, fold))
        _, chosen = min(choices)
        assignments[chosen].append(index)
        sums[chosen] += vectors[index]

    all_ids = sorted(str(record["case_id"]) for record in records)
    folds = []
    for fold, indices in enumerate(assignments):
        validation = sorted(str(records[index]["case_id"]) for index in indices)
        validation_set = set(validation)
        folds.append(
            {
                "fold": fold,
                "training": [case_id for case_id in all_ids if case_id not in validation_set],
                "validation": validation,
            }
        )
    manifest = {
        "version": 12,
        "seed": int(seed),
        "n_folds": 3,
        "records": records,
        "folds": folds,
    }
    validate_manifest(manifest, all_ids)
    return manifest


def validate_manifest(manifest: dict[str, object], expected_ids: list[str]) -> None:
    folds = manifest.get("folds")
    if manifest.get("n_folds") != 3 or not isinstance(folds, list) or len(folds) != 3:
        raise ValueError("V12 requires exactly three folds")
    expected = set(expected_ids)
    seen: set[str] = set()
    sizes = []
    for index, fold in enumerate(folds):
        training = set(fold["training"])
        validation = set(fold["validation"])
        if training & validation or training | validation != expected:
            raise ValueError(f"Fold {index} leaks or omits cases")
        if seen & validation:
            raise ValueError(f"Case appears in multiple validation folds: {sorted(seen & validation)}")
        seen |= validation
        sizes.append(len(validation))
    if seen != expected or sorted(sizes) != [16, 17, 17]:
        raise ValueError("Three-fold validation partition is incomplete or unbalanced")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the immutable balanced GAVE2 V12 three-fold manifest.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=77)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.data_root, args.seed)
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous != manifest:
            raise RuntimeError(f"Existing fold manifest differs: {args.output}")
    else:
        atomic_json(args.output, manifest)
    print(json.dumps({"output": str(args.output), "fold_sizes": [len(f["validation"]) for f in manifest["folds"]]}, indent=2))


if __name__ == "__main__":
    main()

