from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        np.save(handle, array, allow_pickle=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


class ProbabilityStore:
    """Crash-safe, resumable float16 probability storage."""

    VERSION = 8

    def __init__(self, root: Path | str, *, namespace: str, split: str) -> None:
        self.root = Path(root)
        self.namespace = str(namespace)
        self.split = str(split)
        self.arrays = self.root / "arrays"
        self.metadata = self.root / "metadata"
        self.manifest_path = self.root / "completion_manifest.json"
        self.arrays.mkdir(parents=True, exist_ok=True)
        self.metadata.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            _atomic_json(self.manifest_path, self._empty_manifest())

    def _empty_manifest(self) -> dict[str, object]:
        return {
            "version": self.VERSION,
            "namespace": self.namespace,
            "split": self.split,
            "complete_cases": [],
            "cases": {},
        }

    def _manifest(self) -> dict[str, object]:
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("namespace") != self.namespace or payload.get("split") != self.split:
            raise RuntimeError(f"Store identity mismatch at {self.root}")
        return payload

    def array_path(self, case_id: str) -> Path:
        return self.arrays / f"{case_id}.npy"

    def metadata_path(self, case_id: str) -> Path:
        return self.metadata / f"{case_id}.json"

    def case_record(self, case_id: str) -> dict[str, object]:
        manifest = self._manifest()
        record = manifest.get("cases", {}).get(str(case_id))
        if not isinstance(record, dict):
            raise FileNotFoundError(f"Missing case {case_id} in {self.root}")
        return record

    def write_case(
        self,
        case_id: str,
        probability: np.ndarray,
        provenance: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        value = np.asarray(probability, dtype=np.float32)
        if value.ndim != 3 or value.shape[0] != 3:
            raise ValueError(f"Expected [3,H,W] probability, got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite probability for {case_id}")
        value = np.clip(value, 0.0, 1.0).astype(np.float16)
        record = {
            "case_id": str(case_id),
            "shape": list(value.shape),
            "dtype": "float16",
            "sha256": _sha256_array(value),
            "provenance": dict(provenance or {}),
        }
        _atomic_npy(self.array_path(case_id), value)
        _atomic_json(self.metadata_path(case_id), record)
        manifest = self._manifest()
        manifest.setdefault("cases", {})[str(case_id)] = record
        manifest["complete_cases"] = sorted(set(manifest.get("complete_cases", [])) | {str(case_id)})
        _atomic_json(self.manifest_path, manifest)
        return record

    def is_complete(self, case_id: str, provenance: Mapping[str, object] | None = None) -> bool:
        try:
            record = self.case_record(case_id)
        except (FileNotFoundError, json.JSONDecodeError, RuntimeError):
            return False
        if provenance is not None and record.get("provenance") != dict(provenance):
            return False
        array_path = self.array_path(case_id)
        metadata_path = self.metadata_path(case_id)
        if not array_path.exists() or not metadata_path.exists():
            return False
        try:
            array = np.load(array_path, allow_pickle=False)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return (
            list(array.shape) == record.get("shape") == metadata.get("shape")
            and str(array.dtype) == record.get("dtype") == metadata.get("dtype") == "float16"
            and _sha256_array(array) == record.get("sha256") == metadata.get("sha256")
        )

    def read_case(self, case_id: str) -> np.ndarray:
        if not self.is_complete(case_id):
            raise FileNotFoundError(f"Incomplete probability case {case_id} in {self.root}")
        return np.ascontiguousarray(np.load(self.array_path(case_id), allow_pickle=False).astype(np.float32))

    def list_cases(self) -> list[str]:
        return sorted(str(case_id) for case_id in self._manifest().get("complete_cases", []))

    def pending(
        self,
        case_ids: Iterable[str],
        provenance_by_case: Mapping[str, Mapping[str, object]],
    ) -> list[str]:
        return [
            str(case_id)
            for case_id in case_ids
            if not self.is_complete(str(case_id), provenance_by_case[str(case_id)])
        ]

