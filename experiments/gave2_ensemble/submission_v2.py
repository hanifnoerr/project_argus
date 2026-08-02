from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .biomarkers import BIOMARKER_KEYS
from .biomarkers_v2 import read_biomarker_txt
from .data import list_case_ids, read_png_float


@dataclass(frozen=True)
class ScientificSubmissionReport:
    ok: bool
    errors: list[str]
    counts: dict[str, int]


def validate_team_id(team_id: str) -> str:
    value = team_id.strip()
    normalized = re.sub(r"[^a-z0-9]", "", value.lower())
    if not value or normalized in {"teamid", "yourteamid", "placeholder"}:
        raise ValueError("Replace the placeholder with the exact Baidu team name/ID")
    if "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError("Team ID must be one path component")
    return value


def _expected_files(case_ids: list[str], suffix: str) -> set[str]:
    return {f"{case_id}{suffix}" for case_id in case_ids}


def validate_submission_v2(
    team_root: Path | str,
    data_root: Path | str,
    expected_case_ids: list[str] | None = None,
    expected_size: tuple[int, int] = (1024, 1536),
) -> ScientificSubmissionReport:
    root = Path(team_root)
    data = Path(data_root)
    errors: list[str] = []
    counts: dict[str, int] = {}
    try:
        validate_team_id(root.name)
    except ValueError as exc:
        errors.append(str(exc))
    case_ids = expected_case_ids or list_case_ids(data, split="validation")
    if len(case_ids) != len(set(case_ids)):
        errors.append("Expected case IDs contain duplicates")
    expected_h, expected_w = expected_size

    expected_directories = {"Task1", "Task2", "Task3"}
    actual_directories = {path.name for path in root.iterdir() if path.is_dir()} if root.is_dir() else set()
    if actual_directories != expected_directories:
        errors.append(f"Expected only {sorted(expected_directories)} directories, found {sorted(actual_directories)}")
    root_files = [path.name for path in root.iterdir() if path.is_file()] if root.is_dir() else []
    if root_files:
        errors.append(f"Unexpected files at team root: {root_files[:5]}")

    for task in ("Task1", "Task2"):
        task_dir = root / task
        files = sorted(task_dir.glob("*.png")) if task_dir.is_dir() else []
        counts[task] = len(files)
        actual = {path.name for path in files}
        expected = _expected_files(case_ids, ".png")
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            errors.append(f"{task}: missing={missing[:5]}, extra={extra[:5]}")
        unexpected = [path.name for path in task_dir.iterdir() if not path.is_file() or path.suffix.lower() != ".png"] if task_dir.is_dir() else []
        if unexpected:
            errors.append(f"{task}: unexpected payload {unexpected[:5]}")
        for path in files:
            try:
                with Image.open(path) as image:
                    mode = image.mode
                    size = image.size
                    array = np.asarray(image)
                if mode != "RGB":
                    errors.append(f"{task}/{path.name}: mode {mode}, expected RGB")
                    continue
                if size != (expected_w, expected_h):
                    errors.append(f"{task}/{path.name}: size {size}, expected {(expected_w, expected_h)}")
                    continue
                roi_path = data / "validation" / "masks" / path.name
                roi = read_png_float(roi_path, channels=1)[..., 0] > 0.5
                if roi.shape != array.shape[:2]:
                    errors.append(f"{task}/{path.name}: ROI shape mismatch")
                    continue
                outside = array[~roi]
                if outside.size and int(outside.max()) != 0:
                    errors.append(f"{task}/{path.name}: nonzero probability outside ROI")
                inside = array[roi]
                if inside.size == 0 or any(int(np.ptp(inside[:, channel])) == 0 for channel in range(3)):
                    errors.append(f"{task}/{path.name}: one or more probability channels are constant inside ROI")
                if inside.size and np.any(inside[:, 1].astype(np.int16) + 1 < inside[:, 0].astype(np.int16)):
                    errors.append(f"{task}/{path.name}: vessel probability is below artery probability")
                if inside.size and np.any(inside[:, 1].astype(np.int16) + 1 < inside[:, 2].astype(np.int16)):
                    errors.append(f"{task}/{path.name}: vessel probability is below vein probability")
            except Exception as exc:
                errors.append(f"{task}/{path.name}: unreadable PNG ({exc})")

    task3_dir = root / "Task3"
    files = sorted(task3_dir.glob("*.txt")) if task3_dir.is_dir() else []
    counts["Task3"] = len(files)
    actual = {path.name for path in files}
    expected = _expected_files(case_ids, ".txt")
    if actual != expected:
        errors.append(f"Task3: missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}")
    unexpected = [path.name for path in task3_dir.iterdir() if not path.is_file() or path.suffix.lower() != ".txt"] if task3_dir.is_dir() else []
    if unexpected:
        errors.append(f"Task3: unexpected payload {unexpected[:5]}")
    for path in files:
        try:
            values = read_biomarker_txt(path)
            if set(values) != set(BIOMARKER_KEYS) or any(value <= 0 for value in values.values()):
                errors.append(f"Task3/{path.name}: expected exactly seven positive finite biomarkers")
            ratio = values["CRAE"] / values["CRVE"]
            if not np.isclose(values["AVR"], ratio, rtol=2e-5, atol=2e-5):
                errors.append(f"Task3/{path.name}: AVR is inconsistent with CRAE/CRVE")
        except Exception as exc:
            errors.append(f"Task3/{path.name}: invalid biomarker file ({exc})")
    return ScientificSubmissionReport(ok=not errors, errors=errors, counts=counts)


def build_submission_zip(
    team_root: Path | str,
    data_root: Path | str,
    output_zip: Path | str,
    expected_case_ids: list[str] | None = None,
    expected_size: tuple[int, int] = (1024, 1536),
) -> Path:
    root = Path(team_root)
    case_ids = expected_case_ids or list_case_ids(data_root, split="validation")
    report = validate_submission_v2(root, data_root, case_ids, expected_size)
    if not report.ok:
        raise RuntimeError("Submission validation failed:\n" + "\n".join(report.errors[:25]))
    destination = Path(output_zip)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    payload = []
    for task, suffix in (("Task1", ".png"), ("Task2", ".png"), ("Task3", ".txt")):
        for case_id in sorted(case_ids):
            path = root / task / f"{case_id}{suffix}"
            archive_name = f"{root.name}/{task}/{path.name}"
            payload.append((path, archive_name))
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, archive_name in payload:
            archive.write(path, archive_name)
    expected_names = sorted(name for _, name in payload)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Submission ZIP failed CRC validation")
        if sorted(archive.namelist()) != expected_names:
            raise RuntimeError("Submission ZIP payload does not match the certified tree")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Certify and package a GAVE2 v2 submission.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--team-root", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_ids = list_case_ids(args.data_root, split="validation")
    report = validate_submission_v2(args.team_root, args.data_root, case_ids)
    print(json.dumps(asdict(report), indent=2, ensure_ascii=False))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    if not report.ok:
        raise SystemExit(1)
    output = build_submission_zip(args.team_root, args.data_root, args.output_zip, case_ids)
    print(f"Certified submission ZIP: {output}")


if __name__ == "__main__":
    main()
