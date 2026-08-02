from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .store import ProbabilityStore


def fuse_direct_probabilities(
    av_probability: np.ndarray,
    bv_probability: np.ndarray,
    *,
    bv_class_weight: float = 0.0,
    vessel_mode: str = "max",
) -> np.ndarray:
    av = np.asarray(av_probability, dtype=np.float32)
    bv = np.asarray(bv_probability, dtype=np.float32)
    if av.shape != bv.shape or av.ndim != 3 or av.shape[0] != 3:
        raise ValueError(f"Incompatible probabilities: {av.shape}, {bv.shape}")
    if not 0.0 <= bv_class_weight <= 1.0:
        raise ValueError("bv_class_weight must be in [0,1]")
    artery = (1.0 - bv_class_weight) * av[0] + bv_class_weight * bv[0]
    vein = (1.0 - bv_class_weight) * av[2] + bv_class_weight * bv[2]
    if vessel_mode == "max":
        vessel = np.maximum.reduce((av[1], bv[1], artery, vein))
    elif vessel_mode == "mean":
        vessel = np.maximum.reduce((0.5 * (av[1] + bv[1]), artery, vein))
    else:
        raise ValueError(f"Unknown vessel mode {vessel_mode!r}")
    return np.clip(np.stack((artery, vessel, vein), axis=0), 0.0, 1.0).astype(np.float32)


def run_fusion(args: argparse.Namespace) -> dict[str, object]:
    av_store = ProbabilityStore(args.av_store, namespace="r2v2_av", split=args.split)
    bv_store = ProbabilityStore(args.bv_store, namespace="r2v2_bv", split=args.split)
    av_cases = av_store.list_cases()
    bv_cases = bv_store.list_cases()
    if av_cases != bv_cases:
        raise RuntimeError("R2-V2 av and bv stores do not contain identical cases")
    output = ProbabilityStore(args.output_store, namespace="r2v2_direct", split=args.split)
    settings = {
        "revision": "r2v2_direct_fusion_v1",
        "bv_class_weight": float(args.bv_class_weight),
        "vessel_mode": args.vessel_mode,
    }
    new_cases = 0
    for case_id in av_cases:
        provenance = {
            **settings,
            "av_sha256": av_store.case_record(case_id)["sha256"],
            "bv_sha256": bv_store.case_record(case_id)["sha256"],
        }
        if output.is_complete(case_id, provenance):
            continue
        fused = fuse_direct_probabilities(
            av_store.read_case(case_id),
            bv_store.read_case(case_id),
            bv_class_weight=args.bv_class_weight,
            vessel_mode=args.vessel_mode,
        )
        output.write_case(case_id, fused, provenance)
        new_cases += 1
    return {"cases": len(av_cases), "new_cases": new_cases, "output_store": str(args.output_store), **settings}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse R2-V2 av and bv predictions without suppressing crossings.")
    parser.add_argument("--av-store", type=Path, required=True)
    parser.add_argument("--bv-store", type=Path, required=True)
    parser.add_argument("--output-store", type=Path, required=True)
    parser.add_argument("--split", choices=("training", "validation"), required=True)
    parser.add_argument("--bv-class-weight", type=float, default=0.0)
    parser.add_argument("--vessel-mode", choices=("max", "mean"), default="max")
    return parser.parse_args(argv)


def main() -> None:
    print(json.dumps(run_fusion(parse_args()), indent=2))


if __name__ == "__main__":
    main()

