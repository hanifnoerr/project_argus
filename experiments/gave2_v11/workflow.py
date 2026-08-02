from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .calibrator import predict_targets
from .constants import BIOMARKER_KEYS
from .dataset import _find_zip_member, load_training_cache, load_validation_cache, read_biomarker_txt


EXPECTED_SIZE = (1536, 1024)
EXPECTED_CASES = tuple(f"g_{index:03d}" for index in range(51, 101))
EXPECTED_V8_SHA256 = "88267cc219240d17186ab45199185834c7433a83a2202e919ebde00687d732d7"
DOMAIN_MAX_MEAN_SHIFT = 1.0
DOMAIN_MAX_ABS_Z = 6.0
FIXED_ZIP_TIME = (2026, 7, 17, 0, 0, 0)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _biomarker_text(values: dict[str, float]) -> bytes:
    if set(values) != set(BIOMARKER_KEYS):
        raise ValueError(f"Task 3 keys are invalid: {sorted(values)}")
    if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in values.values()):
        raise ValueError("Task 3 values must be finite and positive")
    return ("\n".join(f"{key} {float(values[key]):.6f}" for key in BIOMARKER_KEYS) + "\n").encode("utf-8")


def _read_biomarker_bytes(value: bytes) -> dict[str, float]:
    rows = [line.split() for line in value.decode("utf-8").splitlines()]
    parsed = {row[0]: float(row[1]) for row in rows if len(row) == 2}
    if set(parsed) != set(BIOMARKER_KEYS) or not np.isfinite(list(parsed.values())).all():
        raise ValueError("Invalid baseline Task 3 text")
    return parsed


def _audit_hash(report_path: Path) -> str:
    text = report_path.read_text(encoding="utf-8").rstrip("\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def domain_audit(
    training_cache: Path,
    validation_cache: Path,
    calibrator: dict[str, object],
) -> tuple[dict[str, object], dict[str, dict[str, float]]]:
    training, targets, training_names, _ = load_training_cache(training_cache)
    validation, validation_names, case_ids = load_validation_cache(validation_cache)
    if training_names != validation_names or training_names != calibrator["feature_names"]:
        raise RuntimeError("Training, validation, and calibrator feature schemas differ")
    if tuple(case_ids) != EXPECTED_CASES:
        raise RuntimeError(f"Unexpected validation cases: {case_ids}")

    report_targets: dict[str, object] = {}
    predictions: dict[str, dict[str, float]] = {case_id: {} for case_id in case_ids}
    passed = True
    for target_name in calibrator["accepted_targets"]:
        model = calibrator["models"][target_name]
        indices = np.asarray(model["feature_indices"], dtype=int)
        train_selected = training[:, indices]
        validation_selected = validation[:, indices]
        training_std = np.maximum(train_selected.std(axis=0), 1e-8)
        mean_shift = np.abs((validation_selected.mean(axis=0) - train_selected.mean(axis=0)) / training_std)
        z = np.abs(
            (validation_selected - np.asarray(model["mean"], dtype=np.float64))
            / np.asarray(model["std"], dtype=np.float64)
        )
        target_predictions = []
        ood_cases = []
        for case_id, vector in zip(case_ids, validation, strict=True):
            values, diagnostics = predict_targets(calibrator, vector)
            if target_name not in values:
                ood_cases.append(case_id)
                continue
            predictions[case_id][target_name] = values[target_name]
            target_predictions.append(values[target_name])
            if diagnostics[target_name]["ood"]:
                ood_cases.append(case_id)
        target_passed = (
            not ood_cases
            and float(mean_shift.max()) <= DOMAIN_MAX_MEAN_SHIFT
            and float(z.max()) <= DOMAIN_MAX_ABS_Z
            and len(target_predictions) == len(case_ids)
        )
        passed &= target_passed
        report_targets[target_name] = {
            "passed": bool(target_passed),
            "feature_count": len(indices),
            "max_mean_shift_sd": float(mean_shift.max()),
            "mean_mean_shift_sd": float(mean_shift.mean()),
            "max_abs_z": float(z.max()),
            "ood_cases": ood_cases,
            "prediction_minimum": float(np.min(target_predictions)),
            "prediction_median": float(np.median(target_predictions)),
            "prediction_maximum": float(np.max(target_predictions)),
            "training_target_minimum": float(np.min(targets[target_name])),
            "training_target_median": float(np.median(targets[target_name])),
            "training_target_maximum": float(np.max(targets[target_name])),
        }
    report = {
        "passed": bool(passed),
        "limits": {
            "max_mean_shift_sd": DOMAIN_MAX_MEAN_SHIFT,
            "max_abs_z": DOMAIN_MAX_ABS_Z,
        },
        "targets": report_targets,
    }
    return report, predictions


def _validate_png(value: bytes, case_id: str, task: str) -> None:
    with Image.open(io.BytesIO(value)) as image:
        image.load()
        if image.mode != "RGB" or image.size != EXPECTED_SIZE:
            raise RuntimeError(f"Invalid {task} PNG for {case_id}: mode={image.mode}, size={image.size}")


def build_v11_submission(
    *,
    data_root: Path,
    v8_zip: Path,
    training_cache: Path,
    validation_cache: Path,
    calibrator_path: Path,
    nested_audit_path: Path,
    output_dir: Path,
    team_id: str,
    submission_id: str = "GAVE2-S010",
    version: str = "v11-task3-audited",
) -> dict[str, object]:
    if not team_id or Path(team_id).name != team_id or any(value in team_id for value in ("/", "\\")):
        raise ValueError("team_id must be one path-safe filename label")
    source_v8_sha256 = _sha256_file(v8_zip)
    if source_v8_sha256.lower() != EXPECTED_V8_SHA256:
        raise RuntimeError(
            f"V8 source SHA256 mismatch: {source_v8_sha256}; expected {EXPECTED_V8_SHA256}"
        )
    calibrator = json.loads(calibrator_path.read_text(encoding="utf-8"))
    nested = json.loads(nested_audit_path.read_text(encoding="utf-8"))
    if not calibrator.get("audit_passed") or not nested.get("passed"):
        raise RuntimeError("V11 nested audit did not pass")
    if calibrator.get("audit_sha256") != _audit_hash(nested_audit_path):
        raise RuntimeError("V11 calibrator is not bound to this nested audit")
    if calibrator.get("accepted_targets") != nested.get("accepted_targets"):
        raise RuntimeError("Calibrator targets differ from the nested audit")
    domain_report, predictions = domain_audit(training_cache, validation_cache, calibrator)
    if not domain_report["passed"]:
        raise RuntimeError("V11 validation-domain audit failed")

    expected_cases = [path.stem for path in sorted((data_root / "validation" / "images").glob("*.png"))]
    if tuple(expected_cases) != EXPECTED_CASES:
        raise RuntimeError("Dataset validation cases do not match the competition contract")
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_root = output_dir / "competition_format"
    if not submission_root.resolve().is_relative_to(output_dir.resolve()):
        raise RuntimeError("Resolved competition output escapes output_dir")
    submission_root.mkdir(parents=True, exist_ok=True)

    # Remove only V11-generated payload directories, including the obsolete team wrapper.
    generated_paths = [submission_root / task for task in ("Task1", "Task2", "Task3")]
    generated_paths.append(submission_root / team_id)
    for generated_path in generated_paths:
        resolved = generated_path.resolve()
        if not resolved.is_relative_to(submission_root.resolve()):
            raise RuntimeError(f"Refusing to clean path outside competition output: {resolved}")
        if generated_path.exists():
            shutil.rmtree(generated_path)
    for task in ("Task1", "Task2", "Task3"):
        (submission_root / task).mkdir(parents=True, exist_ok=False)

    source_hashes: dict[str, dict[str, str]] = {"Task1": {}, "Task2": {}, "Task3": {}}
    output_hashes: dict[str, dict[str, str]] = {"Task1": {}, "Task2": {}, "Task3": {}}
    baseline_task3: dict[str, dict[str, float]] = {}
    with zipfile.ZipFile(v8_zip) as source:
        corrupt = source.testzip()
        if corrupt is not None:
            raise RuntimeError(f"V8 source ZIP failed CRC validation: {corrupt}")
        names = source.namelist()
        for case_id in expected_cases:
            for task in ("Task1", "Task2"):
                member = _find_zip_member(names, task, f"{case_id}.png")
                value = source.read(member)
                _validate_png(value, case_id, task)
                destination = submission_root / task / f"{case_id}.png"
                destination.write_bytes(value)
                digest = _sha256_bytes(value)
                source_hashes[task][case_id] = digest
                output_hashes[task][case_id] = _sha256_file(destination)
                if output_hashes[task][case_id] != digest:
                    raise RuntimeError(f"{task} byte preservation failed for {case_id}")

            task3_member = _find_zip_member(names, "Task3", f"{case_id}.txt")
            baseline_bytes = source.read(task3_member)
            baseline = _read_biomarker_bytes(baseline_bytes)
            baseline_task3[case_id] = baseline
            values = dict(baseline)
            values.update(predictions[case_id])
            if "AVR" in predictions[case_id]:
                geometric = math.sqrt(max(values["CRAE"] * values["CRVE"], 1e-8))
                values["CRAE"] = geometric * math.sqrt(values["AVR"])
                values["CRVE"] = geometric / math.sqrt(values["AVR"])
            output_bytes = _biomarker_text(values)
            destination = submission_root / "Task3" / f"{case_id}.txt"
            destination.write_bytes(output_bytes)
            source_hashes["Task3"][case_id] = _sha256_bytes(baseline_bytes)
            output_hashes["Task3"][case_id] = _sha256_bytes(output_bytes)

    unchanged_task3_targets = sorted(set(("AVR",)) - set(calibrator["accepted_targets"]))
    for case_id in expected_cases:
        if "AVR" in unchanged_task3_targets:
            output_values = read_biomarker_txt(submission_root / "Task3" / f"{case_id}.txt")
            if output_values["AVR"] != baseline_task3[case_id]["AVR"]:
                raise RuntimeError(f"Rejected AVR changed for {case_id}")

    zip_path = output_dir / f"{team_id}.zip"
    if zip_path.exists():
        zip_path.unlink()
    payload = [
        (submission_root / task / f"{case_id}{suffix}", f"{task}/{case_id}{suffix}")
        for task, suffix in (("Task1", ".png"), ("Task2", ".png"), ("Task3", ".txt"))
        for case_id in expected_cases
    ]
    with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
        staged = Path(temporary) / zip_path.name
        with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path, member in payload:
                info = zipfile.ZipInfo(member, FIXED_ZIP_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(
                    info,
                    path.read_bytes(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=6,
                )
        with zipfile.ZipFile(staged) as archive:
            corrupt = archive.testzip()
            names = archive.namelist()
            if corrupt is not None:
                raise RuntimeError(f"V11 ZIP failed CRC validation: {corrupt}")
            expected_names = [member for _, member in payload]
            if names != expected_names:
                raise RuntimeError("V11 ZIP layout or ordering is incorrect")
            for task in ("Task1", "Task2"):
                for case_id in expected_cases:
                    value = archive.read(f"{task}/{case_id}.png")
                    if _sha256_bytes(value) != source_hashes[task][case_id]:
                        raise RuntimeError(f"Final ZIP changed V8 {task} bytes for {case_id}")
        shutil.copyfile(staged, zip_path)

    certification = {
        "version": 11,
        "submission_id": submission_id,
        "submission_version": version,
        "team_id": team_id,
        "source_v8_zip": str(v8_zip),
        "source_v8_sha256": source_v8_sha256,
        "training_cache_sha256": _sha256_file(training_cache),
        "validation_cache_sha256": _sha256_file(validation_cache),
        "calibrator_sha256": _sha256_file(calibrator_path),
        "nested_audit_sha256": _sha256_file(nested_audit_path),
        "accepted_task3_targets": calibrator["accepted_targets"],
        "frozen_task3_targets": unchanged_task3_targets,
        "nested_audit": nested,
        "domain_audit": domain_report,
        "task1_task2_byte_identical_to_v8": True,
        "zip": str(zip_path),
        "zip_sha256": _sha256_file(zip_path),
        "zip_layout": "Task1|Task2|Task3 directly at ZIP root",
        "counts": {"Task1": 50, "Task2": 50, "Task3": 50},
        "source_hashes": source_hashes,
        "output_hashes": output_hashes,
    }
    certification_path = output_dir / f"{submission_id}__{version}__certification.json"
    certification_path.write_text(json.dumps(certification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return certification


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and certify the GAVE2 V11 submission.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--v8-zip", type=Path, required=True)
    parser.add_argument("--training-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--calibrator", type=Path, required=True)
    parser.add_argument("--nested-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--team-id", required=True)
    parser.add_argument("--submission-id", default="GAVE2-S010")
    parser.add_argument("--version", default="v11-task3-audited")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    report = build_v11_submission(
        data_root=args.data_root,
        v8_zip=args.v8_zip,
        training_cache=args.training_cache,
        validation_cache=args.validation_cache,
        calibrator_path=args.calibrator,
        nested_audit_path=args.nested_audit,
        output_dir=args.output_dir,
        team_id=args.team_id,
        submission_id=args.submission_id,
        version=args.version,
    )
    print(json.dumps({key: report[key] for key in ("zip", "zip_sha256", "counts", "domain_audit")}, indent=2))


if __name__ == "__main__":
    main()
