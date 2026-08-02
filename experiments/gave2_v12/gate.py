from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

from experiments.gave2_ensemble.data import derive_av3_target
from experiments.gave2_v8.metrics import evaluate_store
from experiments.gave2_v8.store import ProbabilityStore

from .folds import validate_manifest
from .metrics import pixel_score
from .utils import atomic_json, case_ids


def _pixel_report(data_root: Path, store: ProbabilityStore, ids: list[str]) -> dict[str, float]:
    channels = {channel: {key: 0 for key in ("tp", "fp", "fn", "tn")} for channel in (0, 2)}
    for case_id in ids:
        probability = store.read_case(case_id)
        raw = np.asarray(
            Image.open(data_root / "training" / "av" / f"{case_id}.png").convert("RGB"),
            dtype=np.float32,
        ) / 255.0
        target = derive_av3_target(raw) > 0.5
        roi = np.asarray(Image.open(data_root / "training" / "masks" / f"{case_id}.png").convert("L")) > 127
        for channel in (0, 2):
            prediction = (probability[channel] >= 0.5) & roi
            truth = target[channel] & roi
            values = channels[channel]
            values["tp"] += int((prediction & truth).sum())
            values["fp"] += int((prediction & ~truth & roi).sum())
            values["fn"] += int((~prediction & truth).sum())
            values["tn"] += int((~prediction & ~truth & roi).sum())

    reports = []
    for channel in (0, 2):
        values = channels[channel]
        tp, fp, fn, tn = (values[key] for key in ("tp", "fp", "fn", "tn"))
        reports.append(
            {
                "dice": 2.0 * tp / max(2 * tp + fp + fn, 1),
                "sensitivity": tp / max(tp + fn, 1),
                "specificity": tn / max(tn + fp, 1),
                "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
            }
        )
    report = {
        metric: float(np.mean([channel[metric] for channel in reports]))
        for metric in ("dice", "sensitivity", "specificity", "accuracy")
    }
    report["classification"] = (
        0.3 * report["sensitivity"] + 0.3 * report["specificity"] + 0.4 * report["accuracy"]
    )
    report["pixel_score"] = pixel_score(report)
    return report


def structural_counts(
    candidate: np.ndarray,
    teacher: np.ndarray,
    correction_mode: str,
) -> dict[str, int]:
    candidate_binary = np.asarray(candidate) >= 0.5
    teacher_binary = np.asarray(teacher) >= 0.5
    if candidate_binary.shape != teacher_binary.shape or candidate_binary.shape[0] != 3:
        raise ValueError("Candidate and teacher must be matching three-channel arrays")
    if correction_mode == "prune":
        allowed = teacher_binary
    elif correction_mode == "vessel_support":
        allowed = np.broadcast_to(teacher_binary[1:2], teacher_binary.shape)
    else:
        raise ValueError(f"Unknown correction mode: {correction_mode}")

    missing_skeleton = 0
    for channel in (0, 2):
        path = skeletonize(teacher_binary[channel])
        missing_skeleton += int((path & ~candidate_binary[channel]).sum())
    class_additions = int(
        (candidate_binary[[0, 2]] & ~teacher_binary[[0, 2]]).sum()
    )
    return {
        "off_support_additions": int((candidate_binary & ~allowed).sum()),
        "protected_skeleton_missing": missing_skeleton,
        "class_reassignment_additions": class_additions,
        "teacher_class_positive": int(teacher_binary[[0, 2]].sum()),
    }


def _fold_reports(
    data_root: Path,
    teacher: ProbabilityStore,
    candidate: ProbabilityStore,
    manifest: dict[str, object],
) -> list[dict[str, object]]:
    reports = []
    for fold in manifest["folds"]:
        ids = list(fold["validation"])
        teacher_report = _pixel_report(data_root, teacher, ids)
        candidate_report = _pixel_report(data_root, candidate, ids)
        reports.append(
            {
                "fold": int(fold["fold"]),
                "cases": len(ids),
                "pixel_score_gain": float(candidate_report["pixel_score"] - teacher_report["pixel_score"]),
                "dice_gain": float(candidate_report["dice"] - teacher_report["dice"]),
                "sensitivity_drop": float(teacher_report["sensitivity"] - candidate_report["sensitivity"]),
            }
        )
    return reports


def run_gate(args: argparse.Namespace) -> dict[str, object]:
    ids = case_ids(args.data_root, "training")
    teacher = ProbabilityStore(args.teacher_store, namespace="r2v2_direct", split="training")
    candidate = ProbabilityStore(
        args.candidate_store,
        namespace=f"gave2_v12_selected_{args.task}",
        split="training",
    )
    if teacher.list_cases() != ids or candidate.list_cases() != ids:
        raise RuntimeError("Gate requires complete teacher and selected OOF stores")

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("task") != args.task:
        raise RuntimeError("Selection report task does not match the gate task")
    selected = selection["selected"]
    correction_mode = str(selected["correction_mode"])
    manifest = json.loads(args.fold_manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest, ids)
    expected_settings = {
        key: selected[key]
        for key in ("alpha", "decision_threshold", "temperature", "corridor_radius", "correction_mode")
    }
    stale_provenance = [
        case_id
        for case_id in ids
        if candidate.case_record(case_id).get("provenance", {}).get("settings") != expected_settings
    ]

    totals = {
        "off_support_additions": 0,
        "protected_skeleton_missing": 0,
        "class_reassignment_additions": 0,
        "teacher_class_positive": 0,
    }
    for case_id in ids:
        counts = structural_counts(candidate.read_case(case_id), teacher.read_case(case_id), correction_mode)
        for key, value in counts.items():
            totals[key] += value
    reassignment_fraction = totals["class_reassignment_additions"] / max(totals["teacher_class_positive"], 1)

    teacher_report = _pixel_report(args.data_root, teacher, ids)
    candidate_report = _pixel_report(args.data_root, candidate, ids)
    score_gain = float(candidate_report["pixel_score"] - teacher_report["pixel_score"])
    dice_gain = float(candidate_report["dice"] - teacher_report["dice"])
    sensitivity_drop = float(teacher_report["sensitivity"] - candidate_report["sensitivity"])
    fold_reports = _fold_reports(args.data_root, teacher, candidate, manifest)
    fold_gains = [float(fold["pixel_score_gain"]) for fold in fold_reports]
    minimum_fold_gain = min(fold_gains) if fold_gains else 0.0
    maximum_fold_sensitivity_drop = (
        max(float(fold["sensitivity_drop"]) for fold in fold_reports) if fold_reports else 0.0
    )

    reasons = []
    if not selection.get("accepted") or not selected.get("accepted"):
        reasons.append("OOF selection did not accept a learned correction")
    if not fold_reports:
        reasons.append("selected correction has no per-fold OOF reports")
    if stale_provenance:
        reasons.append(f"candidate provenance disagrees with selection for {len(stale_provenance)} cases")
    if score_gain < args.minimum_pixel_score_gain:
        reasons.append(f"pixel score gain {score_gain:.4f} < {args.minimum_pixel_score_gain:.4f}")
    if minimum_fold_gain < args.minimum_fold_pixel_score_gain:
        reasons.append(
            f"minimum fold pixel gain {minimum_fold_gain:.4f} < {args.minimum_fold_pixel_score_gain:.4f}"
        )
    if dice_gain < args.minimum_dice_gain:
        reasons.append(f"Dice gain {dice_gain:.4f} < {args.minimum_dice_gain:.4f}")
    if sensitivity_drop > args.maximum_sensitivity_drop:
        reasons.append(f"sensitivity drop {sensitivity_drop:.4f} > {args.maximum_sensitivity_drop:.4f}")
    if maximum_fold_sensitivity_drop > args.maximum_fold_sensitivity_drop:
        reasons.append(
            "maximum fold sensitivity drop "
            f"{maximum_fold_sensitivity_drop:.4f} > {args.maximum_fold_sensitivity_drop:.4f}"
        )
    if totals["off_support_additions"]:
        reasons.append(f"found {totals['off_support_additions']} additions outside allowed teacher support")
    if totals["protected_skeleton_missing"]:
        reasons.append(f"lost {totals['protected_skeleton_missing']} protected teacher skeleton pixels")
    if correction_mode == "vessel_support" and reassignment_fraction > args.maximum_reassignment_fraction:
        reasons.append(
            f"class reassignment fraction {reassignment_fraction:.4f} > {args.maximum_reassignment_fraction:.4f}"
        )

    path_diagnostics = []
    if args.diagnostic_paths > 0:
        for seed in args.seeds:
            teacher_path = evaluate_store(
                args.data_root,
                teacher,
                threshold=0.5,
                paths_per_case=args.diagnostic_paths,
                seed=seed,
            )
            candidate_path = evaluate_store(
                args.data_root,
                candidate,
                threshold=0.5,
                paths_per_case=args.diagnostic_paths,
                seed=seed,
            )
            path_diagnostics.append(
                {
                    "seed": seed,
                    "teacher_score": teacher_path["score_observed"],
                    "candidate_score": candidate_path["score_observed"],
                    "warning": "diagnostic only; this proxy failed to predict V9/V10 official topology",
                }
            )

    report = {
        "version": 12,
        "task": args.task,
        "accepted": not reasons,
        "reasons": reasons,
        "acceptance_basis": "OOF pixel gain, fold stability, and deterministic support/skeleton invariants",
        "path_proxy_used_for_acceptance": False,
        "fold_reports_recomputed_from_candidate_store": True,
        "correction_mode": correction_mode,
        "teacher": teacher_report,
        "candidate": candidate_report,
        "pixel_score_gain": score_gain,
        "minimum_fold_pixel_score_gain": minimum_fold_gain,
        "dice_gain": dice_gain,
        "sensitivity_drop": sensitivity_drop,
        "maximum_fold_sensitivity_drop": maximum_fold_sensitivity_drop,
        "structural": {**totals, "class_reassignment_fraction": reassignment_fraction},
        "path_diagnostics": path_diagnostics,
        "thresholds": {
            "minimum_pixel_score_gain": args.minimum_pixel_score_gain,
            "minimum_fold_pixel_score_gain": args.minimum_fold_pixel_score_gain,
            "minimum_dice_gain": args.minimum_dice_gain,
            "maximum_sensitivity_drop": args.maximum_sensitivity_drop,
            "maximum_fold_sensitivity_drop": args.maximum_fold_sensitivity_drop,
            "maximum_reassignment_fraction": args.maximum_reassignment_fraction,
        },
    }
    atomic_json(args.output, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a support-constrained V12 OOF candidate.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--teacher-store", type=Path, required=True)
    parser.add_argument("--candidate-store", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--task", choices=("task1", "task2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-pixel-score-gain", type=float, default=0.10)
    parser.add_argument("--minimum-fold-pixel-score-gain", type=float, default=0.0)
    parser.add_argument("--minimum-dice-gain", type=float, default=0.01)
    parser.add_argument("--maximum-sensitivity-drop", type=float, default=0.025)
    parser.add_argument("--maximum-fold-sensitivity-drop", type=float, default=0.05)
    parser.add_argument("--maximum-reassignment-fraction", type=float, default=0.08)
    parser.add_argument("--diagnostic-paths", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=(77, 137))
    return parser.parse_args(argv)


def main() -> None:
    report = run_gate(parse_args())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
