from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from .submission_v6 import certify_submission_atomic, readback_zip


def assemble_hybrid(base_team_root: Path, refined_team_root: Path, output_team_root: Path) -> Path:
    if output_team_root.exists():
        raise FileExistsError(output_team_root)
    output_team_root.mkdir(parents=True)
    for task in ("Task1", "Task2"):
        shutil.copytree(base_team_root / task, output_team_root / task)
    shutil.copytree(refined_team_root / "Task3", output_team_root / "Task3")
    return output_team_root


def assemble_v7(v7_team_root: Path, refined_team_root: Path, output_team_root: Path) -> Path:
    if output_team_root.exists():
        raise FileExistsError(output_team_root)
    output_team_root.mkdir(parents=True)
    for task in ("Task1", "Task2"):
        shutil.copytree(v7_team_root / task, output_team_root / task)
    shutil.copytree(refined_team_root / "Task3", output_team_root / "Task3")
    return output_team_root


def certify_v7(team_root: Path, data_root: Path, output_zip: Path, report: Path) -> dict[str, object]:
    payload = certify_submission_atomic(team_root, data_root, output_zip, report)
    payload["version"] = 7
    payload["task3_source"] = "v6_refined_conservative"
    temporary = report.with_suffix(report.suffix + ".tmp-v7")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, report)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble and certify conservative GAVE2 V7 submissions.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--task12-team-root", type=Path, required=True)
    assemble.add_argument("--refined-team-root", type=Path, required=True)
    assemble.add_argument("--output-team-root", type=Path, required=True)
    certify = subparsers.add_parser("certify")
    certify.add_argument("--team-root", type=Path, required=True)
    certify.add_argument("--data-root", type=Path, required=True)
    certify.add_argument("--output-zip", type=Path, required=True)
    certify.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.command == "assemble":
        result = assemble_v7(args.task12_team_root, args.refined_team_root, args.output_team_root)
        print(result)
    else:
        print(json.dumps(certify_v7(args.team_root, args.data_root, args.output_zip, args.report), indent=2))


if __name__ == "__main__":
    main()
