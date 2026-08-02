from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import list_case_ids
from .submission import validate_submission_tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate all GAVE2 submission folders.")
    parser.add_argument("--data-root", type=Path, default=Path("GAVE2_preliminary"))
    parser.add_argument("--submission-root", type=Path, default=Path("submissions"))
    parser.add_argument("--team-id", type=str, default="team_id")
    parser.add_argument("--split", choices=("training", "validation"), default="validation")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1536)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_ids = list_case_ids(args.data_root, split=args.split)
    all_ok = True
    summary = {}
    for branch in ("cmrrwnet", "sam3", "yolo_native", "ensemble"):
        root = args.submission_root / branch / args.team_id
        report = validate_submission_tree(root, case_ids, expected_size=(args.height, args.width))
        all_ok = all_ok and report.ok
        summary[branch] = {"ok": report.ok, "counts": report.counts, "errors": report.errors}
    print(json.dumps(summary, indent=2))
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
