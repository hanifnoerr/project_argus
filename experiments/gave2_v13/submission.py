from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from experiments.gave2_v8.submission import build_team_directory, certify_root_task_submission_atomic
from experiments.gave2_v12.utils import atomic_json, sha256_file

from .compact import PORTAL_MAXIMUM_BYTES, compact_to_portal_limit
from .selection import _open_store


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for member in sorted(path.glob("*.txt")):
        digest.update(member.name.encode("utf-8"))
        digest.update(member.read_bytes())
    return digest.hexdigest()


def build(args: argparse.Namespace) -> dict[str, object]:
    task1_selection = json.loads(args.task1_selection.read_text(encoding="utf-8"))
    task2_selection = json.loads(args.task2_selection.read_text(encoding="utf-8"))
    if not task1_selection.get("accepted") or not task2_selection.get("accepted"):
        raise RuntimeError("V13 submission requires accepted Task 1 and Task 2 OOF selections")
    task3_audit = json.loads(args.task3_audit.read_text(encoding="utf-8"))
    if not task3_audit.get("accepted_targets"):
        raise RuntimeError("V13 submission requires at least one accepted registered-FFA Task 3 target")

    root = args.output_root / "v13_candidate"
    payload = root / args.team_id
    zip_path = root / f"{args.team_id}.zip"
    certification_path = root / "certification.json"
    compact_report_path = root / "compact_certification.json"
    manifest_path = root / "manifest.json"
    input_hashes = {
        "task1_store_manifest": sha256_file(args.task1_store / "completion_manifest.json"),
        "task2_store_manifest": sha256_file(args.task2_store / "completion_manifest.json"),
        "task3_payload": _directory_sha256(args.task3_source),
        "task1_selection": sha256_file(args.task1_selection),
        "task2_selection": sha256_file(args.task2_selection),
        "task3_audit": sha256_file(args.task3_audit),
    }
    candidate_paths = (payload, zip_path, certification_path, compact_report_path, manifest_path)
    if any(path.exists() for path in candidate_paths):
        if not all(path.exists() for path in candidate_paths):
            raise RuntimeError(f"Partial V13 submission exists; use a new output root: {root}")
        certification = json.loads(certification_path.read_text(encoding="utf-8"))
        compact = json.loads(compact_report_path.read_text(encoding="utf-8"))
        if certification.get("zip_sha256") != sha256_file(zip_path):
            raise RuntimeError("Existing V13 ZIP no longer matches its certification")
        if (
            compact.get("output_sha256") != certification.get("zip_sha256")
            or compact.get("output_bytes") != zip_path.stat().st_size
            or zip_path.stat().st_size >= PORTAL_MAXIMUM_BYTES
            or compact.get("threshold_mismatch_pixels") != 0
            or not compact.get("task3_byte_identical")
        ):
            raise RuntimeError("Existing V13 ZIP failed its compact-output certification")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("input_hashes") != input_hashes:
            raise RuntimeError("Existing V13 ZIP was built from stale inputs; use a new output root")
        return existing

    root.mkdir(parents=True, exist_ok=True)
    build_team_directory(
        data_root=args.data_root,
        task1_store=_open_store(args.task1_store),
        task2_store=_open_store(args.task2_store),
        task3_source=args.task3_source,
        team_root=payload,
    )
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        full_precision_zip = Path(temporary) / f"{args.team_id}.full_precision.zip"
        full_precision_report = Path(temporary) / "full_precision_certification.json"
        raw_certification = certify_root_task_submission_atomic(
            payload,
            args.data_root,
            full_precision_zip,
            full_precision_report,
        )
        compact = compact_to_portal_limit(
            full_precision_zip,
            zip_path,
            compact_report_path,
            maximum_bytes=PORTAL_MAXIMUM_BYTES,
        )
    certification = {
        "version": 13,
        "team_id": args.team_id,
        "zip": str(zip_path),
        "zip_sha256": compact["output_sha256"],
        "zip_bytes": compact["output_bytes"],
        "maximum_submission_bytes": PORTAL_MAXIMUM_BYTES,
        "headroom_bytes": compact["headroom_bytes"],
        "validation": raw_certification["validation"],
        "layout": compact["layout"],
        "counts": compact["counts"],
        "compaction": compact,
    }
    atomic_json(certification_path, certification)
    report = {
        "version": 13,
        "submission_name": "v13-channel-path-ffa",
        "team_id": args.team_id,
        "zip": str(zip_path),
        "zip_sha256": certification["zip_sha256"],
        "zip_bytes": certification["zip_bytes"],
        "maximum_submission_bytes": certification["maximum_submission_bytes"],
        "headroom_bytes": certification["headroom_bytes"],
        "layout": certification["layout"],
        "counts": certification["counts"],
        "compact_certification": str(compact_report_path),
        "compact_certification_sha256": sha256_file(compact_report_path),
        "probability_bits": compact["probability_bits"],
        "threshold_mismatch_pixels": compact["threshold_mismatch_pixels"],
        "task3_byte_identical": compact["task3_byte_identical"],
        "task1_selection": str(args.task1_selection),
        "task2_selection": str(args.task2_selection),
        "task3_audit": str(args.task3_audit),
        "accepted_task3_targets": task3_audit["accepted_targets"],
        "input_hashes": input_hashes,
        "warning": "Do not submit unless release_decision.json says READY_FOR_ONE_CAUTIOUS_SUBMISSION.",
    }
    atomic_json(manifest_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the certified root-task V13 competition ZIP.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--team-id", required=True)
    parser.add_argument("--task1-store", type=Path, required=True)
    parser.add_argument("--task2-store", type=Path, required=True)
    parser.add_argument("--task3-source", type=Path, required=True)
    parser.add_argument("--task1-selection", type=Path, required=True)
    parser.add_argument("--task2-selection", type=Path, required=True)
    parser.add_argument("--task3-audit", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
