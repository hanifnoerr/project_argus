from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gave2_v13 import RUNTIME_BUILD_ID


def _member_sha256(archive: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(name) as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(path: Path, *, expect_code_only: bool) -> dict[str, object]:
    required = {
        "experiments/gave2_v13/train.py",
        "experiments/gave2_v13/predict.py",
        "experiments/gave2_v13/selection.py",
        "experiments/gave2_v13/task3.py",
        "experiments/gave2_v13/submission.py",
        "experiments/gave2_v13/release.py",
        "experiments/gave2_v13/compact.py",
        "submission/GAVE2_Channel_Path_FFA_V13_Colab.ipynb",
        "submission/GAVE2_R2V2_FFA_Residual_V12_Colab.ipynb",
        "tests/gave2_v13/test_notebook.py",
        "tests/gave2_v13/test_compact_submission.py",
        "tests/gave2_v13/test_losses.py",
    }
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"CRC failure: {corrupt}")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("Duplicate ZIP members")
        for name in names:
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts or "\\" in name:
                raise RuntimeError(f"Unsafe ZIP member: {name}")
        manifest = json.loads(archive.read("archive_manifest.json"))
        if manifest.get("runtime_build_id") != RUNTIME_BUILD_ID:
            raise RuntimeError("Runtime build ID mismatch")
        expected_kind = "source-only" if expect_code_only else "colab-runtime-with-dataset"
        if manifest.get("kind") != expected_kind:
            raise RuntimeError(f"Archive kind mismatch: {manifest.get('kind')} != {expected_kind}")
        if set(names) != set(manifest["members"]) | {"archive_manifest.json"}:
            raise RuntimeError("Manifest member list mismatch")
        missing = required - set(names)
        if missing:
            raise RuntimeError(f"Missing required members: {sorted(missing)}")
        forbidden = [
            name
            for name in names
            if name.lower().endswith((".pt", ".pth", ".ckpt", ".pyc", ".pyo"))
            or "__pycache__" in PurePosixPath(name).parts
            or "runs" in PurePosixPath(name).parts
        ]
        if forbidden:
            raise RuntimeError(f"Forbidden payload: {forbidden[:10]}")
        for name, expected in manifest["sha256"].items():
            if _member_sha256(archive, name) != expected:
                raise RuntimeError(f"SHA256 mismatch: {name}")

        dataset_members = [name for name in names if name.startswith("GAVE2_preliminary/")]
        if expect_code_only and dataset_members:
            raise RuntimeError("Source-only archive contains competition data")
        if not expect_code_only and len(dataset_members) != 502:
            raise RuntimeError(f"Expected 502 dataset members, found {len(dataset_members)}")
        notebook = json.loads(archive.read("submission/GAVE2_Channel_Path_FFA_V13_Colab.ipynb"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                if cell.get("execution_count") is not None or cell.get("outputs"):
                    raise RuntimeError("Notebook contains saved execution state")
                ast.parse("".join(cell["source"]))
    return {
        "archive": str(path),
        "runtime_build_id": RUNTIME_BUILD_ID,
        "kind": expected_kind,
        "members": len(names) - 1,
        "dataset_members": len(dataset_members),
        "crc": "passed",
        "member_sha256": "passed",
        "notebook": "clean_and_parsed",
        "forbidden_payload": "absent",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently audit a GAVE2 V13 archive.")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--expect-code-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(audit(args.archive, expect_code_only=args.expect_code_only), indent=2))
