from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .constants import BIOMARKER_KEYS, SCORED_TARGETS
from .dataset import _find_zip_member
from .workflow import EXPECTED_CASES, EXPECTED_SIZE, EXPECTED_V8_SHA256, _read_biomarker_bytes


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_release(
    *,
    v8_zip: Path,
    v11_zip: Path,
    team_id: str,
    certification_path: Path | None = None,
) -> dict[str, object]:
    errors: list[str] = []
    if not team_id or Path(team_id).name != team_id or any(value in team_id for value in ("/", "\\")):
        errors.append("team_id is not a path-safe outer filename label")
    v8_hash = _sha256_file(v8_zip)
    if v8_hash.lower() != EXPECTED_V8_SHA256:
        errors.append(f"wrong V8 source hash: {v8_hash}")
    expected_members = {
        f"{task}/{case_id}{suffix}"
        for task, suffix in (("Task1", ".png"), ("Task2", ".png"), ("Task3", ".txt"))
        for case_id in EXPECTED_CASES
    }
    changed_task3_counts = {target: 0 for target in SCORED_TARGETS}
    with zipfile.ZipFile(v8_zip) as source, zipfile.ZipFile(v11_zip) as candidate:
        source_corrupt = source.testzip()
        candidate_corrupt = candidate.testzip()
        if source_corrupt is not None:
            errors.append(f"V8 CRC failure: {source_corrupt}")
        if candidate_corrupt is not None:
            errors.append(f"V11 CRC failure: {candidate_corrupt}")
        actual_members = set(candidate.namelist())
        if actual_members != expected_members:
            errors.append(
                f"V11 member mismatch: missing={sorted(expected_members - actual_members)[:5]}, "
                f"extra={sorted(actual_members - expected_members)[:5]}"
            )
        source_names = source.namelist()
        for case_id in EXPECTED_CASES:
            for task in ("Task1", "Task2"):
                source_member = _find_zip_member(source_names, task, f"{case_id}.png")
                candidate_member = f"{task}/{case_id}.png"
                if candidate_member not in actual_members:
                    continue
                source_bytes = source.read(source_member)
                candidate_bytes = candidate.read(candidate_member)
                if candidate_bytes != source_bytes:
                    errors.append(f"{task} bytes changed for {case_id}")
                with Image.open(io.BytesIO(candidate_bytes)) as image:
                    image.load()
                    if image.mode != "RGB" or image.size != EXPECTED_SIZE:
                        errors.append(f"invalid {task} PNG for {case_id}: {image.mode}, {image.size}")

            baseline_member = _find_zip_member(source_names, "Task3", f"{case_id}.txt")
            candidate_member = f"Task3/{case_id}.txt"
            if candidate_member not in actual_members:
                continue
            baseline = _read_biomarker_bytes(source.read(baseline_member))
            values = _read_biomarker_bytes(candidate.read(candidate_member))
            if set(values) != set(BIOMARKER_KEYS):
                errors.append(f"invalid Task3 keys for {case_id}")
            if not all(math.isfinite(value) and value > 0.0 for value in values.values()):
                errors.append(f"invalid Task3 value for {case_id}")
            if values["AVR"] != baseline["AVR"]:
                errors.append(f"frozen AVR changed for {case_id}")
            if values["CRAE"] != baseline["CRAE"] or values["CRVE"] != baseline["CRVE"]:
                errors.append(f"unscored CRAE/CRVE changed for {case_id}")
            if not math.isclose(values["AVR"], values["CRAE"] / values["CRVE"], rel_tol=2e-6):
                errors.append(f"AVR consistency failed for {case_id}")
            for target in SCORED_TARGETS:
                changed_task3_counts[target] += int(values[target] != baseline[target])

    expected_changes = {
        "AVR": 0,
        "artery_density": 50,
        "vein_density": 50,
        "artery_fractal_dimension": 50,
        "vein_fractal_dimension": 50,
    }
    if changed_task3_counts != expected_changes:
        errors.append(f"unexpected Task3 change counts: {changed_task3_counts}")
    certification_check = None
    if certification_path is not None:
        certification = json.loads(certification_path.read_text(encoding="utf-8"))
        certification_check = {
            "zip_sha256_matches": certification.get("zip_sha256") == _sha256_file(v11_zip),
            "v8_sha256_matches": certification.get("source_v8_sha256", "").lower() == v8_hash.lower(),
            "domain_audit_passed": certification.get("domain_audit", {}).get("passed") is True,
            "nested_audit_passed": certification.get("nested_audit", {}).get("passed") is True,
        }
        if not all(certification_check.values()):
            errors.append(f"certification mismatch: {certification_check}")
    return {
        "passed": not errors,
        "errors": errors,
        "v8_sha256": v8_hash,
        "v11_sha256": _sha256_file(v11_zip),
        "team_id": team_id,
        "member_count": len(expected_members),
        "layout": "Task1|Task2|Task3 directly at ZIP root",
        "task3_change_counts": changed_task3_counts,
        "certification": certification_check,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently audit a built GAVE2 V11 submission.")
    parser.add_argument("--v8-zip", type=Path, required=True)
    parser.add_argument("--v11-zip", type=Path, required=True)
    parser.add_argument("--team-id", required=True)
    parser.add_argument("--certification", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    report = audit_release(
        v8_zip=args.v8_zip,
        v11_zip=args.v11_zip,
        team_id=args.team_id,
        certification_path=args.certification,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
