from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


EXPECTED_CASES = tuple(f"g_{index:03d}" for index in range(51, 101))
EXPECTED_SIZE = (1536, 1024)
FIXED_ZIP_TIME = (2026, 7, 18, 0, 0, 0)
PORTAL_MAXIMUM_BYTES = 100_000_000
DEFAULT_BIT_CANDIDATES = (7, 6, 5, 4)


class SubmissionSizeError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantize_threshold_safe(array: np.ndarray, bits: int) -> np.ndarray:
    if not 1 <= bits <= 8:
        raise ValueError("bits must be in [1, 8]")
    source = np.asarray(array, dtype=np.uint8)
    step = 1 << (8 - bits)
    value = ((source.astype(np.uint16) + step // 2) // step) * step
    value = np.clip(value, 0, 255)
    value[(source < 128) & (value >= 128)] = 128 - step
    value[(source >= 128) & (value < 128)] = 128
    return value.astype(np.uint8)


def _png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(
        buffer,
        format="PNG",
        optimize=True,
        compress_level=9,
    )
    return buffer.getvalue()


def _expected_names() -> list[str]:
    return [
        f"{task}/{case_id}{suffix}"
        for task, suffix in (("Task1", ".png"), ("Task2", ".png"), ("Task3", ".txt"))
        for case_id in EXPECTED_CASES
    ]


def compact_submission_zip(
    source: Path,
    output: Path,
    report_path: Path,
    *,
    bits: int,
    maximum_bytes: int = PORTAL_MAXIMUM_BYTES,
) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    report_path = report_path.resolve()
    if source == output:
        raise ValueError("The compact ZIP must not overwrite its full-precision source")
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() or report_path.exists():
        raise FileExistsError("Compact output already exists; use a new immutable candidate directory")

    expected = _expected_names()
    expected_set = set(expected)
    encoded: dict[str, bytes] = {}
    task3_source_hashes: dict[str, str] = {}
    threshold_mismatches = 0
    maximum_error = 0
    absolute_error = 0
    value_count = 0

    with zipfile.ZipFile(source) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"Full-precision ZIP CRC failure: {corrupt}")
        source_files = {name for name in archive.namelist() if not name.endswith("/")}
        if source_files != expected_set:
            raise RuntimeError(
                f"Full-precision layout mismatch: missing={sorted(expected_set - source_files)}, "
                f"extra={sorted(source_files - expected_set)}"
            )
        for index, name in enumerate(expected, 1):
            payload = archive.read(name)
            if name.endswith(".txt"):
                encoded[name] = payload
                task3_source_hashes[name] = hashlib.sha256(payload).hexdigest()
                continue
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                if image.mode != "RGB" or image.size != EXPECTED_SIZE:
                    raise RuntimeError(f"Invalid full-precision PNG {name}: {image.mode}, {image.size}")
                source_pixels = np.asarray(image, dtype=np.uint8)
            compact_pixels = quantize_threshold_safe(source_pixels, bits)
            threshold_mismatches += int(
                np.count_nonzero((source_pixels >= 128) != (compact_pixels >= 128))
            )
            difference = np.abs(source_pixels.astype(np.int16) - compact_pixels.astype(np.int16))
            maximum_error = max(maximum_error, int(difference.max()))
            absolute_error += int(difference.sum())
            value_count += int(difference.size)
            encoded[name] = _png_bytes(compact_pixels)
            if index % 10 == 0:
                print(f"[{index:03d}/{len(expected):03d}] compacted {name}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as handle:
        staged = Path(handle.name)
    try:
        with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_STORED) as archive:
            for name in expected:
                info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, encoded[name])
        if staged.stat().st_size >= maximum_bytes:
            raise SubmissionSizeError(
                f"{bits}-bit candidate is {staged.stat().st_size} bytes; limit is {maximum_bytes}"
            )
        with zipfile.ZipFile(staged) as archive:
            if archive.testzip() is not None or archive.namelist() != expected:
                raise RuntimeError("Compact ZIP failed CRC, ordering, or root-layout readback")
            for name in expected:
                payload = archive.read(name)
                if name.endswith(".txt"):
                    if hashlib.sha256(payload).hexdigest() != task3_source_hashes[name]:
                        raise RuntimeError(f"Task 3 changed during compaction: {name}")
                    continue
                with Image.open(io.BytesIO(payload)) as image:
                    image.load()
                    if image.mode != "RGB" or image.size != EXPECTED_SIZE:
                        raise RuntimeError(f"Compact PNG readback failed: {name}")
        if threshold_mismatches:
            raise RuntimeError(f"Compaction changed {threshold_mismatches} threshold decisions")
        os.replace(staged, output)
    finally:
        if staged.exists():
            staged.unlink()

    report = {
        "submission_version": 13,
        "strategy": "adaptive threshold-safe probability quantization",
        "source_sha256": _sha256_file(source),
        "source_bytes": source.stat().st_size,
        "output_sha256": _sha256_file(output),
        "output_bytes": output.stat().st_size,
        "maximum_bytes": int(maximum_bytes),
        "headroom_bytes": int(maximum_bytes - output.stat().st_size),
        "probability_bits": int(bits),
        "maximum_intensity_error": maximum_error,
        "mean_absolute_intensity_error": absolute_error / max(value_count, 1),
        "threshold_mismatch_pixels": threshold_mismatches,
        "threshold_masks_equivalent": threshold_mismatches == 0,
        "task3_byte_identical": True,
        "layout": "tasks_at_zip_root",
        "counts": {"Task1": 50, "Task2": 50, "Task3": 50},
        "crc": "passed",
        "png_contract": {"mode": "RGB", "size": list(EXPECTED_SIZE)},
    }
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_report, report_path)
    return report


def compact_to_portal_limit(
    source: Path,
    output: Path,
    report_path: Path,
    *,
    bit_candidates: tuple[int, ...] = DEFAULT_BIT_CANDIDATES,
    maximum_bytes: int = PORTAL_MAXIMUM_BYTES,
) -> dict[str, object]:
    failures: list[str] = []
    for bits in bit_candidates:
        try:
            report = compact_submission_zip(
                source,
                output,
                report_path,
                bits=bits,
                maximum_bytes=maximum_bytes,
            )
            report["attempted_probability_bits"] = list(bit_candidates[: len(failures) + 1])
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            return report
        except SubmissionSizeError as exc:
            failures.append(str(exc))
            print(f"Size gate: {exc}; trying lower precision", flush=True)
    raise SubmissionSizeError("No threshold-safe candidate fits the portal limit: " + "; ".join(failures))
