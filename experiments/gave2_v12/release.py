from __future__ import annotations

import argparse
import json
from pathlib import Path

from .utils import atomic_json, sha256_file


V8_OFFICIAL_OVERALL = 7.49452
TASK_WEIGHTS = {"task1": 0.20, "task2": 0.40}
RECORDED_TOP3_TARGET = 7.80106
RECORDED_FIRST_TARGET = 8.04644


def _conservative_gain(gate: dict[str, object]) -> float:
    if not gate.get("accepted"):
        return 0.0
    return max(
        0.0,
        min(
            float(gate["pixel_score_gain"]),
            float(gate["minimum_fold_pixel_score_gain"]),
        ),
    )


def decide_release(
    task1_gate: dict[str, object],
    task2_gate: dict[str, object],
    *,
    output_root: Path,
    team_id: str,
    release_target: float = 7.95,
) -> dict[str, object]:
    gains = {
        "task1": _conservative_gain(task1_gate),
        "task2": _conservative_gain(task2_gate),
    }
    accepted = {
        "task1": bool(task1_gate.get("accepted")),
        "task2": bool(task2_gate.get("accepted")),
    }

    candidates: list[dict[str, object]] = []
    if accepted["task1"] or accepted["task2"]:
        changed = [task for task in ("task1", "task2") if accepted[task]]
        candidates.append(
            {
                "variant": "v12_safe",
                "changed_tasks": changed,
                "conservative_projected_score": V8_OFFICIAL_OVERALL
                + sum(TASK_WEIGHTS[task] * gains[task] for task in changed),
            }
        )
    for candidate in candidates:
        path = output_root / str(candidate["variant"]) / f"{team_id}.zip"
        candidate["zip"] = str(path)
        candidate["zip_exists"] = path.is_file()
        candidate["zip_sha256"] = sha256_file(path) if path.is_file() else None

    passing = [
        candidate
        for candidate in candidates
        if bool(candidate["zip_exists"])
        and float(candidate["conservative_projected_score"]) >= release_target
    ]
    if passing:
        recommended = max(passing, key=lambda candidate: float(candidate["conservative_projected_score"]))
        status = "READY_FOR_ONE_CAUTIOUS_SUBMISSION"
        reason = "The conservative worst-fold projection clears the release target. Submit only the recommended ZIP."
    else:
        recommended = None
        status = "DO_NOT_SUBMIT"
        best_projection = max(
            (float(candidate["conservative_projected_score"]) for candidate in candidates),
            default=V8_OFFICIAL_OVERALL,
        )
        reason = (
            f"No structurally accepted candidate reaches {release_target:.5f}; "
            f"best conservative projection is {best_projection:.5f}."
        )

    return {
        "version": 12,
        "status": status,
        "reason": reason,
        "v8_official_control": V8_OFFICIAL_OVERALL,
        "release_target": float(release_target),
        "recorded_top3_target": RECORDED_TOP3_TARGET,
        "recorded_first_target": RECORDED_FIRST_TARGET,
        "leaderboard_checked_at": "2026-07-17",
        "stretch_target": 8.0,
        "accepted_tasks": accepted,
        "conservative_task_gains": gains,
        "candidates": candidates,
        "recommended": recommended,
        "warning": (
            "Projection assumes support constraints keep official topology near V8. "
            "It is a risk filter, not a leaderboard guarantee."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make a single conservative V12 release decision.")
    parser.add_argument("--task1-gate", type=Path, required=True)
    parser.add_argument("--task2-gate", type=Path, required=True)
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--team-id", required=True)
    parser.add_argument("--release-target", type=float, default=7.95)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    report = decide_release(
        json.loads(args.task1_gate.read_text(encoding="utf-8")),
        json.loads(args.task2_gate.read_text(encoding="utf-8")),
        output_root=args.submission_root,
        team_id=args.team_id,
        release_target=args.release_target,
    )
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
