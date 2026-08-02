from __future__ import annotations

import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .registration import ffa_feature_stack, fit_registration, load_matches, warp_to_reference
from .utils import atomic_json, case_ids, sha256_file


def _read_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _atomic_npz(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        np.savez_compressed(handle, **values)
        temporary = Path(handle.name)
    os.replace(temporary, path)


class PreparedFFACache:
    VERSION = 12

    def __init__(self, root: Path | str, split: str) -> None:
        self.root = Path(root) / split
        self.split = split

    def array_path(self, case_id: str) -> Path:
        return self.root / "arrays" / f"{case_id}.npz"

    def metadata_path(self, case_id: str) -> Path:
        return self.root / "metadata" / f"{case_id}.json"

    def read_case(self, case_id: str) -> np.ndarray:
        with np.load(self.array_path(case_id), allow_pickle=False) as payload:
            features = np.asarray(payload["features"], dtype=np.float32)
        if features.ndim != 3 or features.shape[0] != 5 or not np.isfinite(features).all():
            raise RuntimeError(f"Invalid prepared FFA features for {case_id}: {features.shape}")
        return features

    def metadata(self, case_id: str) -> dict[str, object]:
        return json.loads(self.metadata_path(case_id).read_text(encoding="utf-8"))

    def is_complete(self, case_id: str, provenance: dict[str, object] | None = None) -> bool:
        try:
            metadata = self.metadata(case_id)
            features = self.read_case(case_id)
        except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError, RuntimeError):
            return False
        return (
            metadata.get("version") == self.VERSION
            and list(features.shape) == metadata.get("shape")
            and (provenance is None or metadata.get("provenance") == provenance)
        )


def prepare_split(args: argparse.Namespace) -> dict[str, object]:
    cache = PreparedFFACache(args.output_root, args.split)
    reports: dict[str, object] = {}
    accepted = {"FFA_A": 0, "FFA_AV": 0}
    identities = {"FFA_A": 0, "FFA_AV": 0}
    ids = case_ids(args.data_root, args.split)
    if args.limit_cases is not None:
        ids = ids[: args.limit_cases]
    for index, case_id in enumerate(ids, 1):
        split_root = args.data_root / args.split
        cfp_path = split_root / "images" / f"{case_id}.png"
        roi_path = split_root / "masks" / f"{case_id}.png"
        early_path = split_root / "FFA_A" / f"{case_id}.png"
        late_path = split_root / "FFA_AV" / f"{case_id}.png"
        match_paths = {
            "FFA_A": args.matches_root / args.split / "FFA_A" / f"{case_id}.npz",
            "FFA_AV": args.matches_root / args.split / "FFA_AV" / f"{case_id}.npz",
        }
        provenance = {
            "cfp_sha256": sha256_file(cfp_path),
            "roi_sha256": sha256_file(roi_path),
            "early_sha256": sha256_file(early_path),
            "late_sha256": sha256_file(late_path),
            "early_matches_sha256": sha256_file(match_paths["FFA_A"]) if match_paths["FFA_A"].exists() else None,
            "late_matches_sha256": sha256_file(match_paths["FFA_AV"]) if match_paths["FFA_AV"].exists() else None,
            "fallback": args.fallback,
        }
        if cache.is_complete(case_id, provenance):
            report = cache.metadata(case_id)
            reports[case_id] = report
            for phase in ("FFA_A", "FFA_AV"):
                phase_report = report["registration"][phase]
                accepted[phase] += int(bool(phase_report["accepted"]))
                identities[phase] += int(not bool(phase_report["accepted"]))
            continue
        print(f"[{index:02d}/{len(ids):02d}] prepare registered FFA {args.split} {case_id}", flush=True)
        roi = _read_gray(roi_path) > 0.5
        shape = roi.shape
        registrations: dict[str, tuple[np.ndarray, object]] = {}
        for phase, match_path in match_paths.items():
            if match_path.exists():
                try:
                    moving, fixed, confidence = load_matches(match_path)
                    registrations[phase] = fit_registration(moving, fixed, confidence, shape, seed=args.seed)
                except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile) as error:
                    if args.fallback != "identity":
                        raise
                    from .registration import _failed_qa

                    registrations[phase] = (
                        np.eye(3),
                        _failed_qa(0, f"invalid MINIMA {phase} matches; identity fallback: {error}"),
                    )
            elif args.fallback == "identity":
                from .registration import _failed_qa

                registrations[phase] = (
                    np.eye(3),
                    _failed_qa(0, f"MINIMA {phase} matches absent; identity fallback"),
                )
            else:
                raise FileNotFoundError(f"MINIMA {phase} matches absent for {case_id}: {match_path}")
        early_matrix, early_qa = registrations["FFA_A"]
        late_matrix, late_qa = registrations["FFA_AV"]
        early = warp_to_reference(_read_gray(early_path), early_matrix, shape)
        late = warp_to_reference(_read_gray(late_path), late_matrix, shape)
        features = ffa_feature_stack(early, late, roi)
        _atomic_npz(
            cache.array_path(case_id),
            features=features.astype(np.float16),
            early_matrix=early_matrix.astype(np.float64),
            late_matrix=late_matrix.astype(np.float64),
        )
        report = {
            "version": cache.VERSION,
            "case_id": case_id,
            "shape": list(features.shape),
            "registration": {"FFA_A": early_qa.to_dict(), "FFA_AV": late_qa.to_dict()},
            "matrix_moving_to_fixed": {"FFA_A": early_matrix.tolist(), "FFA_AV": late_matrix.tolist()},
            "provenance": provenance,
        }
        atomic_json(cache.metadata_path(case_id), report)
        reports[case_id] = report
        for phase, qa in (("FFA_A", early_qa), ("FFA_AV", late_qa)):
            accepted[phase] += int(qa.accepted)
            identities[phase] += int(not qa.accepted)
    summary = {
        "version": 12,
        "split": args.split,
        "cases": len(ids),
        "accepted_registrations": accepted,
        "identity_fallbacks": identities,
        "acceptance_fraction": {
            phase: accepted[phase] / max(len(ids), 1) for phase in ("FFA_A", "FFA_AV")
        },
        "output_root": str(cache.root),
        "case_reports": reports,
    }
    atomic_json(cache.root / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare full-canvas registered FFA features for V12.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--matches-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("training", "validation"), required=True)
    parser.add_argument("--fallback", choices=("identity", "error"), default="identity")
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--limit-cases", type=int)
    return parser.parse_args(argv)


def main() -> None:
    summary = prepare_split(parse_args())
    print(json.dumps({key: value for key, value in summary.items() if key != "case_reports"}, indent=2))


if __name__ == "__main__":
    main()
