from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from .submission_v2 import build_submission_zip, validate_submission_v2, validate_team_id


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def readback_zip(path: Path, expected_team_id: str) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"ZIP CRC failure: {corrupt}")
        names = sorted(archive.namelist())
    prefix = f"{validate_team_id(expected_team_id)}/"
    if not names or any(not name.startswith(prefix) for name in names):
        raise RuntimeError("ZIP root does not match the team ID")
    counts = {
        task: sum(name.startswith(f"{prefix}{task}/") for name in names)
        for task in ("Task1", "Task2", "Task3")
    }
    return {"sha256": sha256_file(path), "names": names, "counts": counts}


def certify_submission_atomic(
    team_root: Path,
    data_root: Path,
    output_zip: Path,
    report_path: Path,
    *,
    refuse_overwrite: bool = True,
) -> dict[str, object]:
    if refuse_overwrite and (output_zip.exists() or report_path.exists()):
        raise FileExistsError("Certified output already exists and will not be overwritten")
    report = validate_submission_v2(team_root, data_root)
    if not report.ok:
        raise RuntimeError("Submission validation failed:\n" + "\n".join(report.errors[:25]))
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_zip.parent) as tmp:
        staged = Path(tmp) / output_zip.name
        build_submission_zip(team_root, data_root, staged)
        readback = readback_zip(staged, team_root.name)
        os.replace(staged, output_zip)
    payload = {
        "version": 6,
        "team_id": team_root.name,
        "zip": str(output_zip),
        "validation": {"ok": report.ok, "counts": report.counts, "errors": report.errors},
        "readback": readback,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, report_path)
    return payload


def copy_base_as_refined(base_team_root: Path, refined_team_root: Path) -> Path:
    if refined_team_root.exists():
        raise FileExistsError("Refined tree already exists")
    refined_team_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base_team_root, refined_team_root)
    return refined_team_root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Certify a GAVE2 V6 submission without overwriting prior artifacts.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--team-root", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    print(json.dumps(certify_submission_atomic(args.team_root, args.data_root, args.output_zip, args.report), indent=2))


if __name__ == "__main__":
    main()
