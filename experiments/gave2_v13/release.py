from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from experiments.gave2_v12.utils import atomic_json, sha256_file

from .compact import EXPECTED_CASES, PORTAL_MAXIMUM_BYTES


V12_OFFICIAL_OVERALL = 7.5341
V12_TASK3_SCORE = 7.2118
TASK_WEIGHTS = {"task1": 0.20, "task2": 0.40, "task3": 0.40}
TASK3_SCORED_TARGETS = 5
SEGMENTATION_TRANSFER_FACTORS = {
    "task1": 0.2651888535486985,
    "task2": 0.30933186253928613,
}


def _zip_layout_valid(path: Path) -> bool:
    expected = {
        f"{task}/{case_id}{suffix}"
        for task, suffix in (("Task1", ".png"), ("Task2", ".png"), ("Task3", ".txt"))
        for case_id in EXPECTED_CASES
    }
    try:
        with zipfile.ZipFile(path) as archive:
            names = [member.filename for member in archive.infolist() if not member.is_dir()]
            return archive.testzip() is None and len(names) == len(set(names)) and set(names) == expected
    except (OSError, zipfile.BadZipFile):
        return False


def _segmentation_gain(
    task: str,
    selection: dict[str, object],
    transfer_scale: float,
) -> tuple[float, float]:
    if not selection.get("accepted"):
        return 0.0, 0.0
    selected = selection["selected"]
    stable_candidates = [
        float(selected["score_gain"]),
        float(selected["minimum_fold_score_gain"]),
    ]
    for key in ("fast_score_gain", "fast_minimum_fold_score_gain"):
        if key in selected:
            stable_candidates.append(float(selected[key]))
    local_gain = max(0.0, min(stable_candidates))
    official_gain = (
        local_gain
        * SEGMENTATION_TRANSFER_FACTORS[task]
        * max(0.0, transfer_scale)
    )
    return official_gain, local_gain


def _task3_gain(audit: dict[str, object]) -> float:
    accepted = list(audit.get("accepted_targets", []))
    if not accepted:
        return 0.0
    gains = [
        max(0.0, float(audit["nested_audit"]["targets"][target]["nested_relative_gain"]))
        for target in accepted
    ]
    fraction = len(accepted) / TASK3_SCORED_TARGETS
    return (10.0 - V12_TASK3_SCORE) * fraction * min(gains)


def decide(args: argparse.Namespace) -> dict[str, object]:
    selections = {
        "task1": json.loads(args.task1_selection.read_text(encoding="utf-8")),
        "task2": json.loads(args.task2_selection.read_text(encoding="utf-8")),
    }
    task3 = json.loads(args.task3_audit.read_text(encoding="utf-8"))
    submission = json.loads(args.submission_manifest.read_text(encoding="utf-8"))
    zip_path = Path(submission["zip"])
    compact_path = Path(submission.get("compact_certification", ""))
    compact = (
        json.loads(compact_path.read_text(encoding="utf-8"))
        if compact_path.is_file()
        else {}
    )
    actual_zip_bytes = zip_path.stat().st_size if zip_path.is_file() else -1
    compact_valid = (
        compact_path.is_file()
        and submission.get("compact_certification_sha256") == sha256_file(compact_path)
        and compact.get("output_sha256") == submission.get("zip_sha256")
        and compact.get("output_bytes") == actual_zip_bytes
        and compact.get("maximum_bytes") == PORTAL_MAXIMUM_BYTES
        and compact.get("threshold_mismatch_pixels") == 0
        and compact.get("threshold_masks_equivalent") is True
        and compact.get("task3_byte_identical") is True
        and compact.get("layout") == "tasks_at_zip_root"
        and compact.get("counts") == {"Task1": 50, "Task2": 50, "Task3": 50}
    )
    zip_valid = (
        zip_path.is_file()
        and _zip_layout_valid(zip_path)
        and submission.get("layout") == "tasks_at_zip_root"
        and submission.get("counts") == {"Task1": 50, "Task2": 50, "Task3": 50}
        and sha256_file(zip_path) == submission.get("zip_sha256")
        and submission.get("zip_bytes") == actual_zip_bytes
        and submission.get("maximum_submission_bytes") == PORTAL_MAXIMUM_BYTES
        and 0 < actual_zip_bytes < PORTAL_MAXIMUM_BYTES
        and compact_valid
    )
    segmentation_results = {
        task: _segmentation_gain(task, report, args.segmentation_transfer_scale)
        for task, report in selections.items()
    }
    segmentation_gains = {task: result[0] for task, result in segmentation_results.items()}
    segmentation_local_gains = {task: result[1] for task, result in segmentation_results.items()}
    task3_gain = _task3_gain(task3)
    projected = V12_OFFICIAL_OVERALL + sum(
        TASK_WEIGHTS[task] * gain for task, gain in segmentation_gains.items()
    ) + TASK_WEIGHTS["task3"] * task3_gain

    reasons: list[str] = []
    for task, report in selections.items():
        if not report.get("accepted"):
            reasons.append(f"{task} failed OOF selection")
        elif float(report["selected"]["score"]) < args.minimum_local_task_score:
            reasons.append(
                f"{task} local score {float(report['selected']['score']):.4f} "
                f"< {args.minimum_local_task_score:.4f}"
            )
    accepted_task3 = set(task3.get("accepted_targets", []))
    if not set(args.required_task3_targets).issubset(accepted_task3):
        reasons.append(
            "Task 3 did not accept all required targets: "
            + ", ".join(sorted(set(args.required_task3_targets) - accepted_task3))
        )
    if projected < args.release_target:
        reasons.append(f"conservative projection {projected:.5f} < {args.release_target:.5f}")
    if not zip_valid:
        reasons.append("submission ZIP layout, count, SHA256, compact-equivalence, or 100 MB gate failed")
    status = "READY_FOR_ONE_CAUTIOUS_SUBMISSION" if not reasons else "DO_NOT_SUBMIT"
    report = {
        "version": 13,
        "status": status,
        "reasons": reasons,
        "release_target": args.release_target,
        "minimum_local_task_score": args.minimum_local_task_score,
        "v12_official_control": V12_OFFICIAL_OVERALL,
        "segmentation_stable_local_gains": segmentation_local_gains,
        "segmentation_conservative_gains": segmentation_gains,
        "segmentation_transfer_factors": SEGMENTATION_TRANSFER_FACTORS,
        "segmentation_transfer_scale": args.segmentation_transfer_scale,
        "task3_conservative_gain": task3_gain,
        "conservative_projected_score": projected,
        "accepted_task3_targets": sorted(accepted_task3),
        "zip": str(zip_path),
        "zip_valid": zip_valid,
        "compact_valid": compact_valid,
        "zip_bytes": actual_zip_bytes,
        "maximum_submission_bytes": PORTAL_MAXIMUM_BYTES,
        "headroom_bytes": PORTAL_MAXIMUM_BYTES - actual_zip_bytes if actual_zip_bytes >= 0 else None,
        "zip_sha256": submission.get("zip_sha256"),
        "warning": (
            "This is an evidence gate, not a leaderboard guarantee. The official COR/INF implementation "
            "is unavailable locally. Segmentation gains are discounted using the observed V8-to-V12 "
            "local-to-official transfer ratio for each task."
        ),
    }
    atomic_json(args.output, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make the audited V13 submission decision.")
    parser.add_argument("--task1-selection", type=Path, required=True)
    parser.add_argument("--task2-selection", type=Path, required=True)
    parser.add_argument("--task3-audit", type=Path, required=True)
    parser.add_argument("--submission-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-target", type=float, default=7.7)
    parser.add_argument("--minimum-local-task-score", type=float, default=8.0)
    parser.add_argument("--segmentation-transfer-scale", type=float, default=1.0)
    parser.add_argument("--required-task3-targets", nargs="+", default=("vein_density",))
    return parser.parse_args(argv)


def main() -> None:
    report = decide(parse_args())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "READY_FOR_ONE_CAUTIOUS_SUBMISSION":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
