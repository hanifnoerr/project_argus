from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .constants import BIOMARKER_KEYS
from .features import FEATURE_SCHEMA_VERSION, extract_task3_features, load_rgb_float, load_roi


def read_biomarker_txt(path: Path | str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"Invalid biomarker line in {path}: {line!r}")
        values[fields[0]] = float(fields[1])
    if set(values) != set(BIOMARKER_KEYS) or not np.isfinite(list(values.values())).all():
        raise ValueError(f"Invalid biomarker keys or values in {path}")
    return values


class V8ProbabilityReader:
    """Read and verify the stable float16 store produced by V8."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        manifest_path = self.root / "completion_manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("namespace") != "r2v2_direct" or self.manifest.get("split") != "training":
            raise RuntimeError(f"Unexpected V8 store identity at {self.root}")

    def list_cases(self) -> list[str]:
        return sorted(str(value) for value in self.manifest.get("complete_cases", []))

    def read_case(self, case_id: str) -> np.ndarray:
        record = self.manifest.get("cases", {}).get(case_id)
        if not isinstance(record, dict):
            raise FileNotFoundError(f"Missing {case_id} in {self.root}")
        path = self.root / "arrays" / f"{case_id}.npy"
        array = np.load(path, allow_pickle=False)
        digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        if (
            str(array.dtype) != record.get("dtype")
            or list(array.shape) != record.get("shape")
            or digest != record.get("sha256")
        ):
            raise RuntimeError(f"V8 store integrity failed for {case_id}")
        return np.ascontiguousarray(array.astype(np.float32))


def _case_ids(data_root: Path, split: str) -> list[str]:
    case_ids = [path.stem for path in sorted((data_root / split / "images").glob("*.png"))]
    if not case_ids:
        raise FileNotFoundError(f"No {split} images found under {data_root}")
    return case_ids


def _ground_truth_probability(path: Path) -> np.ndarray:
    rgba = load_rgb_float(path)
    artery = (rgba[..., 0] >= 0.5).astype(np.float32)
    vein = (rgba[..., 2] >= 0.5).astype(np.float32)
    vessel = np.max(rgba, axis=2).astype(np.float32)
    return np.stack((artery, vessel, vein), axis=0)


def extract_training_cache(
    *,
    data_root: Path,
    output: Path,
    source: str,
    prediction_store: Path | None = None,
) -> dict[str, object]:
    if source not in {"v8_direct", "ground_truth"}:
        raise ValueError(f"Unsupported feature source: {source}")
    case_ids = _case_ids(data_root, "training")
    store = None
    if source == "v8_direct":
        if prediction_store is None:
            raise ValueError("prediction_store is required for v8_direct")
        store = V8ProbabilityReader(prediction_store)
        if store.list_cases() != case_ids:
            raise RuntimeError("V8 training probability store is incomplete")

    vectors: list[np.ndarray] = []
    names: list[str] | None = None
    metadata: dict[str, object] = {}
    targets = {key: [] for key in BIOMARKER_KEYS}
    for index, case_id in enumerate(case_ids, start=1):
        probability = (
            store.read_case(case_id)
            if store is not None
            else _ground_truth_probability(data_root / "training" / "av" / f"{case_id}.png")
        )
        cfp = load_rgb_float(data_root / "training" / "images" / f"{case_id}.png")
        roi = load_roi(data_root / "training" / "masks" / f"{case_id}.png")
        vector, current_names, case_metadata = extract_task3_features(probability, cfp, roi)
        if names is None:
            names = current_names
        elif names != current_names:
            raise RuntimeError("Feature schema changed between cases")
        vectors.append(vector)
        metadata[case_id] = case_metadata
        label = read_biomarker_txt(data_root / "training" / "biomarker" / f"{case_id}.txt")
        for key in BIOMARKER_KEYS:
            targets[key].append(label[key])
        print(f"[{index:02d}/{len(case_ids):02d}] {source} {case_id}", flush=True)

    feature_matrix = np.stack(vectors)
    target_matrix = np.column_stack([targets[key] for key in BIOMARKER_KEYS])
    if not np.isfinite(feature_matrix).all() or not np.isfinite(target_matrix).all():
        raise RuntimeError("Extracted cache contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=feature_matrix,
        targets=target_matrix,
        case_ids=np.asarray(case_ids),
        feature_names=np.asarray(names or []),
        target_names=np.asarray(BIOMARKER_KEYS),
    )
    report = {
        "version": FEATURE_SCHEMA_VERSION,
        "source": source,
        "cases": len(case_ids),
        "features": feature_matrix.shape[1],
        "cache": str(output),
        "localization": metadata,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_training_cache(path: Path | str) -> tuple[np.ndarray, dict[str, np.ndarray], list[str], list[str]]:
    with np.load(path, allow_pickle=False) as payload:
        features = np.asarray(payload["features"], dtype=np.float64)
        target_matrix = np.asarray(payload["targets"], dtype=np.float64)
        feature_names = [str(value) for value in payload["feature_names"]]
        target_names = [str(value) for value in payload["target_names"]]
        case_ids = [str(value) for value in payload["case_ids"]]
    if target_matrix.shape != (features.shape[0], len(target_names)):
        raise ValueError("Invalid target matrix in feature cache")
    targets = {key: target_matrix[:, index] for index, key in enumerate(target_names)}
    return features, targets, feature_names, case_ids


def _find_zip_member(names: list[str], task: str, filename: str) -> str:
    suffix = f"{task}/{filename}"
    matches = [name for name in names if name.replace("\\", "/").endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one ZIP member ending in {suffix}, found {matches}")
    return matches[0]


def extract_validation_cache(
    *,
    data_root: Path,
    v8_zip: Path,
    output: Path,
) -> dict[str, object]:
    case_ids = _case_ids(data_root, "validation")
    vectors: list[np.ndarray] = []
    feature_names: list[str] | None = None
    metadata: dict[str, object] = {}
    member_map: dict[str, str] = {}
    with zipfile.ZipFile(v8_zip) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"V8 source ZIP failed CRC validation: {corrupt}")
        names = archive.namelist()
        for index, case_id in enumerate(case_ids, start=1):
            member = _find_zip_member(names, "Task2", f"{case_id}.png")
            with Image.open(io.BytesIO(archive.read(member))) as image:
                probability = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
            cfp = load_rgb_float(data_root / "validation" / "images" / f"{case_id}.png")
            roi = load_roi(data_root / "validation" / "masks" / f"{case_id}.png")
            vector, current_names, case_metadata = extract_task3_features(probability, cfp, roi)
            if feature_names is None:
                feature_names = current_names
            elif feature_names != current_names:
                raise RuntimeError("Validation feature schema changed between cases")
            vectors.append(vector)
            metadata[case_id] = case_metadata
            member_map[case_id] = member
            print(f"[{index:02d}/{len(case_ids):02d}] v8_zip {case_id}", flush=True)

    feature_matrix = np.stack(vectors)
    if not np.isfinite(feature_matrix).all():
        raise RuntimeError("Validation cache contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=feature_matrix,
        case_ids=np.asarray(case_ids),
        feature_names=np.asarray(feature_names or []),
    )
    report = {
        "version": FEATURE_SCHEMA_VERSION,
        "source": "v8_submission_zip",
        "v8_zip": str(v8_zip),
        "cases": len(case_ids),
        "features": feature_matrix.shape[1],
        "cache": str(output),
        "members": member_map,
        "localization": metadata,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_validation_cache(path: Path | str) -> tuple[np.ndarray, list[str], list[str]]:
    with np.load(path, allow_pickle=False) as payload:
        features = np.asarray(payload["features"], dtype=np.float64)
        feature_names = [str(value) for value in payload["feature_names"]]
        case_ids = [str(value) for value in payload["case_ids"]]
    if features.shape != (len(case_ids), len(feature_names)) or not np.isfinite(features).all():
        raise ValueError("Invalid validation feature cache")
    return features, feature_names, case_ids


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract deterministic GAVE2 V11 Task 3 feature caches.")
    parser.add_argument("--split", choices=("training", "validation"), default="training")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=("v8_direct", "ground_truth"))
    parser.add_argument("--prediction-store", type=Path)
    parser.add_argument("--v8-zip", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.split == "training":
        if args.source is None:
            raise ValueError("--source is required for training extraction")
        report = extract_training_cache(
            data_root=args.data_root,
            output=args.output,
            source=args.source,
            prediction_store=args.prediction_store,
        )
    else:
        if args.v8_zip is None:
            raise ValueError("--v8-zip is required for validation extraction")
        report = extract_validation_cache(data_root=args.data_root, v8_zip=args.v8_zip, output=args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
