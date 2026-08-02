from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.gave2_ensemble.submission_v2 import validate_submission_v2

from .store import ProbabilityStore


def _read_roi(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    roi = np.asarray(Image.open(path).convert("L")) > 127
    if roi.shape != expected_shape:
        raise ValueError(f"ROI shape mismatch at {path}: {roi.shape} != {expected_shape}")
    return roi


def _save_probability_png(probability: np.ndarray, roi: np.ndarray, output: Path) -> None:
    value = np.asarray(probability, dtype=np.float32)
    if value.ndim != 3 or value.shape[0] != 3 or value.shape[1:] != roi.shape:
        raise ValueError(f"Probability shape {value.shape} is incompatible with ROI {roi.shape}")
    value = np.clip(value, 0.0, 1.0) * roi[None, ...]
    image = np.round(value.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(output)


def _validate_task3_source(task3_source: Path, case_ids: list[str]) -> None:
    expected = {f"{case_id}.txt" for case_id in case_ids}
    actual = {path.name for path in task3_source.glob("*.txt")}
    if expected != actual:
        raise RuntimeError(
            f"Task 3 source mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    required_keys = {
        "CRAE",
        "CRVE",
        "AVR",
        "artery_density",
        "vein_density",
        "artery_fractal_dimension",
        "vein_fractal_dimension",
    }
    for case_id in case_ids:
        rows = [line.split() for line in (task3_source / f"{case_id}.txt").read_text(encoding="utf-8").splitlines()]
        keys = {row[0] for row in rows if len(row) == 2}
        if keys != required_keys:
            raise RuntimeError(f"Task 3 keys mismatch for {case_id}: {sorted(keys)}")


def build_team_directory(
    *,
    data_root: Path,
    task1_store: ProbabilityStore,
    task2_store: ProbabilityStore,
    task3_source: Path,
    team_root: Path,
) -> Path:
    case_ids = [path.stem for path in sorted((data_root / "validation" / "images").glob("*.png"))]
    if len(case_ids) != 50:
        raise RuntimeError(f"Expected 50 validation cases, found {len(case_ids)}")
    if task1_store.list_cases() != case_ids or task2_store.list_cases() != case_ids:
        raise RuntimeError("Task 1/2 probability stores are incomplete or out of order")
    _validate_task3_source(task3_source, case_ids)
    if team_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing team directory: {team_root}")
    for task in ("Task1", "Task2", "Task3"):
        (team_root / task).mkdir(parents=True, exist_ok=False)
    for case_id in case_ids:
        probability1 = task1_store.read_case(case_id)
        probability2 = task2_store.read_case(case_id)
        expected_shape = tuple(int(value) for value in probability1.shape[1:])
        if probability2.shape[1:] != expected_shape:
            raise RuntimeError(f"Task shape mismatch for {case_id}")
        roi = _read_roi(data_root / "validation" / "masks" / f"{case_id}.png", expected_shape)
        _save_probability_png(probability1, roi, team_root / "Task1" / f"{case_id}.png")
        _save_probability_png(probability2, roi, team_root / "Task2" / f"{case_id}.png")
        shutil.copy2(task3_source / f"{case_id}.txt", team_root / "Task3" / f"{case_id}.txt")
    return team_root


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def certify_root_task_submission_atomic(
    team_root: Path,
    data_root: Path,
    output_zip: Path,
    report_path: Path,
    *,
    expected_size: tuple[int, int] = (1024, 1536),
) -> dict[str, object]:
    """Certify a portal ZIP whose root contains Task1, Task2, and Task3."""

    if output_zip.exists() or report_path.exists():
        raise FileExistsError("Certified output already exists and will not be overwritten")
    validation = validate_submission_v2(team_root, data_root, expected_size=expected_size)
    if not validation.ok:
        raise RuntimeError("Submission validation failed:\n" + "\n".join(validation.errors[:25]))

    case_ids = [path.stem for path in sorted((data_root / "validation" / "images").glob("*.png"))]
    payload = [
        (team_root / task / f"{case_id}{suffix}", f"{task}/{case_id}{suffix}")
        for task, suffix in (("Task1", ".png"), ("Task2", ".png"), ("Task3", ".txt"))
        for case_id in case_ids
    ]
    expected_names = sorted(archive_name for _, archive_name in payload)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_zip.parent) as temporary:
        staged = Path(temporary) / output_zip.name
        with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path, archive_name in payload:
                archive.write(path, archive_name)
        with zipfile.ZipFile(staged) as archive:
            corrupt = archive.testzip()
            names = sorted(archive.namelist())
        if corrupt is not None:
            raise RuntimeError(f"Submission ZIP failed CRC validation: {corrupt}")
        if names != expected_names:
            raise RuntimeError("Submission ZIP payload does not match the certified task tree")
        readback = {
            "sha256": _sha256_file(staged),
            "names": names,
            "counts": {
                task: sum(name.startswith(f"{task}/") for name in names)
                for task in ("Task1", "Task2", "Task3")
            },
            "layout": "tasks_at_zip_root",
        }
        os.replace(staged, output_zip)

    certification = {
        "version": 8,
        "team_id": team_root.name,
        "zip": str(output_zip),
        "validation": {
            "ok": validation.ok,
            "counts": validation.counts,
            "errors": validation.errors,
        },
        "readback": readback,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(json.dumps(certification, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary_report, report_path)
    return certification


def build_candidate(
    *,
    data_root: Path,
    task1_store: ProbabilityStore,
    task2_store: ProbabilityStore,
    task3_source: Path,
    output_root: Path,
    team_id: str,
    submission_id: str,
    version: str,
) -> dict[str, object]:
    candidate_root = output_root / f"{submission_id}__{version}"
    team_root = candidate_root / team_id
    build_team_directory(
        data_root=data_root,
        task1_store=task1_store,
        task2_store=task2_store,
        task3_source=task3_source,
        team_root=team_root,
    )
    zip_path = output_root / f"{submission_id}__{version}__{team_id}.zip"
    report_path = output_root / f"{submission_id}__{version}__certification.json"
    certification = certify_root_task_submission_atomic(team_root, data_root, zip_path, report_path)
    return {
        "submission_id": submission_id,
        "version": version,
        "team_root": str(team_root),
        "zip": str(zip_path),
        "certification": certification,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and certify direct or graph-selected R2-V2 submissions.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task1-store", type=Path, required=True)
    parser.add_argument("--task2-store", type=Path, required=True)
    parser.add_argument("--namespace", choices=("r2v2_direct", "r2v2_graph"), required=True)
    parser.add_argument("--task3-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--team-id", required=True)
    parser.add_argument("--submission-id", required=True)
    parser.add_argument("--version", required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    task1 = ProbabilityStore(args.task1_store, namespace=args.namespace, split="validation")
    task2 = ProbabilityStore(args.task2_store, namespace=args.namespace, split="validation")
    report = build_candidate(
        data_root=args.data_root,
        task1_store=task1,
        task2_store=task2,
        task3_source=args.task3_source,
        output_root=args.output_root,
        team_id=args.team_id,
        submission_id=args.submission_id,
        version=args.version,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
