from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from experiments.gave2_v8.store import ProbabilityStore
from experiments.gave2_v8.submission import build_team_directory, certify_root_task_submission_atomic

from .utils import atomic_json, sha256_file


def _open_store(path: Path) -> ProbabilityStore:
    manifest = json.loads((path / "completion_manifest.json").read_text(encoding="utf-8"))
    return ProbabilityStore(path, namespace=str(manifest["namespace"]), split=str(manifest["split"]))


def _build_variant(
    *,
    name: str,
    data_root: Path,
    output_root: Path,
    team_id: str,
    task1: Path,
    task2: Path,
    task3_source: Path,
) -> dict[str, object]:
    variant_root = output_root / name
    payload_root = variant_root / team_id
    zip_path = variant_root / f"{team_id}.zip"
    report_path = variant_root / "certification.json"
    if zip_path.exists() and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected_hash = report.get("readback", {}).get("sha256")
        if expected_hash != sha256_file(zip_path):
            raise RuntimeError(f"Existing {name} ZIP does not match its certification")
        if report.get("readback", {}).get("layout") != "tasks_at_zip_root":
            raise RuntimeError(f"Existing {name} ZIP has the wrong layout certification")
        return {"variant": name, "zip": str(zip_path), "reused": True, "certification": report}
    if payload_root.exists() or zip_path.exists() or report_path.exists():
        raise RuntimeError(f"Partial output exists for {name}; remove only that variant directory before rerunning")
    build_team_directory(
        data_root=data_root,
        task1_store=_open_store(task1),
        task2_store=_open_store(task2),
        task3_source=task3_source,
        team_root=payload_root,
    )
    report = certify_root_task_submission_atomic(payload_root, data_root, zip_path, report_path)
    return {"variant": name, "zip": str(zip_path), "reused": False, "certification": report}


def build_submissions(args: argparse.Namespace) -> dict[str, object]:
    task1_gate = json.loads(args.task1_gate.read_text(encoding="utf-8"))
    task2_gate = json.loads(args.task2_gate.read_text(encoding="utf-8"))
    safe_task1 = args.selected_task1 if task1_gate.get("accepted") else args.teacher_task1
    safe_task2 = args.selected_task2 if task2_gate.get("accepted") else args.teacher_task2
    if args.force:
        for name in ("v12_safe", "v12_task2_probe", "v12_full"):
            path = args.output_root / name
            if path.exists():
                shutil.rmtree(path)
        manifest_path = args.output_root / "submission_manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()
    variants = [
        _build_variant(
            name="v12_safe",
            data_root=args.data_root,
            output_root=args.output_root,
            team_id=args.team_id,
            task1=safe_task1,
            task2=safe_task2,
            task3_source=args.task3_source,
        ),
    ]
    report = {
        "version": 12,
        "team_id": args.team_id,
        "task1_gate_accepted": bool(task1_gate.get("accepted")),
        "task2_gate_accepted": bool(task2_gate.get("accepted")),
        "safe_sources": {"task1": str(safe_task1), "task2": str(safe_task2), "task3": str(args.task3_source)},
        "variants": variants,
        "submission_order": ["v12_safe"],
        "warning": "V12 intentionally builds one unambiguous candidate. Submit it only when release_decision.json says READY_FOR_ONE_CAUTIOUS_SUBMISSION.",
    }
    atomic_json(args.output_root / "submission_manifest.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build certified V12 root-task submissions.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--team-id", required=True)
    parser.add_argument("--teacher-task1", type=Path, required=True)
    parser.add_argument("--teacher-task2", type=Path, required=True)
    parser.add_argument("--selected-task1", type=Path, required=True)
    parser.add_argument("--selected-task2", type=Path, required=True)
    parser.add_argument("--task1-gate", type=Path, required=True)
    parser.add_argument("--task2-gate", type=Path, required=True)
    parser.add_argument("--task3-source", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    print(json.dumps(build_submissions(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
