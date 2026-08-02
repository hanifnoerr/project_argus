from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .constants import BIOMARKER_KEYS
from .utils import atomic_json, case_ids


def read_biomarker(path: Path) -> dict[str, float]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"Invalid biomarker line at {path}: {line!r}")
        values[fields[0]] = float(fields[1])
    if set(values) != set(BIOMARKER_KEYS) or not all(math.isfinite(value) and value > 0 for value in values.values()):
        raise ValueError(f"Invalid Task 3 values at {path}")
    if not math.isclose(values["AVR"], values["CRAE"] / values["CRVE"], rel_tol=2e-5):
        raise ValueError(f"AVR is inconsistent at {path}")
    return values


def audit_frozen_task3(data_root: Path, source: Path, output: Path | None = None) -> dict[str, object]:
    ids = case_ids(data_root, "validation")
    expected = {f"{case_id}.txt" for case_id in ids}
    actual = {path.name for path in source.glob("*.txt")}
    errors = []
    if actual != expected:
        errors.append(f"member mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    for case_id in ids:
        try:
            read_biomarker(source / f"{case_id}.txt")
        except (FileNotFoundError, ValueError) as error:
            errors.append(str(error))
    report = {
        "version": 12,
        "strategy": "freeze proven V8 Task 3 after V11 failed the public leaderboard",
        "source": str(source),
        "cases": len(ids),
        "passed": not errors,
        "errors": errors,
    }
    if output is not None:
        atomic_json(output, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the deliberately frozen V8 Task 3 payload.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    report = audit_frozen_task3(args.data_root, args.source, args.output)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
