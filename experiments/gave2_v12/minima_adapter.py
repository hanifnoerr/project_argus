from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

from .utils import atomic_json, case_ids, sha256_file


def _enable_pinned_minima_numpy_compatibility() -> None:
    """Provide the one removed NumPy alias used by pinned MINIMA."""

    if "float" not in np.__dict__:
        setattr(np, "float", np.float64)


def _enable_pinned_minima_kornia_compatibility() -> None:
    """Expose Kornia's current public grid function at LoFTR's old import path."""

    try:
        importlib.import_module("kornia.utils.grid")
        return
    except ModuleNotFoundError as error:
        if error.name != "kornia.utils.grid":
            raise

    utils = importlib.import_module("kornia.utils")
    create_meshgrid = getattr(utils, "create_meshgrid", None)
    if not callable(create_meshgrid):
        raise RuntimeError("Installed Kornia does not expose kornia.utils.create_meshgrid")
    compatibility = ModuleType("kornia.utils.grid")
    compatibility.create_meshgrid = create_meshgrid
    sys.modules[compatibility.__name__] = compatibility
    setattr(utils, "grid", compatibility)


def _atomic_matches(path: Path, moving: np.ndarray, fixed: np.ndarray, confidence: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            np.savez_compressed(
                handle,
                moving_xy=moving.astype(np.float32),
                fixed_xy=fixed.astype(np.float32),
                confidence=confidence.astype(np.float32),
            )
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _is_complete(path: Path, metadata: Path, provenance: dict[str, object]) -> bool:
    if not path.exists() or not metadata.exists():
        return False
    try:
        previous = json.loads(metadata.read_text(encoding="utf-8"))
        with np.load(path, allow_pickle=False) as payload:
            moving = np.asarray(payload["moving_xy"])
            fixed = np.asarray(payload["fixed_xy"])
            confidence = np.asarray(payload["confidence"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return (
        previous.get("provenance") == provenance
        and moving.ndim == fixed.ndim == 2
        and moving.shape == fixed.shape
        and moving.shape[1:] == (2,)
        and confidence.shape == (len(moving),)
        and np.isfinite(moving).all()
        and np.isfinite(fixed).all()
        and np.isfinite(confidence).all()
    )


def _discard_incomplete(path: Path, metadata: Path) -> None:
    for candidate in (path, metadata):
        if candidate.exists():
            candidate.unlink()


def _normalise_result(result: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(result, tuple) and len(result) >= 2:
        first, second = result[:2]
        confidence = result[2] if len(result) > 2 else np.ones(len(first), dtype=np.float32)
    elif isinstance(result, dict):
        first = result.get("mkpts0", result.get("keypoints0"))
        second = result.get("mkpts1", result.get("keypoints1"))
        confidence = result.get("mconf", result.get("confidence"))
        if first is None or second is None:
            raise RuntimeError(f"Unknown MINIMA result keys: {sorted(result)}")
        if confidence is None:
            confidence = np.ones(len(first), dtype=np.float32)
    else:
        raise RuntimeError(f"Unknown MINIMA result type: {type(result)!r}")

    def numpy(value: object) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    return numpy(first).reshape(-1, 2), numpy(second).reshape(-1, 2), numpy(confidence).reshape(-1)


def load_minima_matcher(source_dir: Path, checkpoint: Path, threshold: float):
    if not (source_dir / "load_model.py").is_file():
        raise FileNotFoundError(f"MINIMA load_model.py not found at {source_dir}")
    source_text = str(source_dir)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    _enable_pinned_minima_numpy_compatibility()
    _enable_pinned_minima_kornia_compatibility()
    from load_model import load_model

    arguments = SimpleNamespace(ckpt=str(checkpoint), thr=float(threshold))
    return load_model("loftr", arguments)


def run(args: argparse.Namespace) -> dict[str, object]:
    _enable_pinned_minima_numpy_compatibility()
    phase = str(args.phase)
    output_root = args.output_root / args.split / phase
    output_root.mkdir(parents=True, exist_ok=True)
    ids = case_ids(args.data_root, args.split)
    if args.limit_cases is not None:
        ids = ids[: args.limit_cases]
    reports: dict[str, object] = {}
    failure_policy = getattr(args, "failure_policy", "error")
    load_error = None
    try:
        matcher = load_minima_matcher(args.source_dir.resolve(), args.checkpoint.resolve(), args.threshold)
    except Exception as error:
        if failure_policy == "error":
            raise
        traceback.print_exc()
        matcher = None
        load_error = f"{type(error).__name__}: {error}"
        print(f"WARNING: MINIMA {phase} unavailable; prepare.py will use recorded identity fallbacks", flush=True)
    if matcher is None:
        for case_id in ids:
            output = output_root / f"{case_id}.npz"
            _discard_incomplete(output, output.with_suffix(".json"))
        summary = {
            "split": args.split,
            "phase": phase,
            "cases": len(ids),
            "successful_cases": 0,
            "failed_cases": len(ids),
            "load_error": load_error,
            "failure_policy": failure_policy,
            "output_root": str(output_root),
            "cases_report": {},
        }
        atomic_json(output_root / "summary.json", summary)
        return summary

    checkpoint_sha256 = sha256_file(args.checkpoint)
    successful = 0
    failed = 0
    for index, case_id in enumerate(ids, 1):
        output = output_root / f"{case_id}.npz"
        ffa = args.data_root / args.split / phase / f"{case_id}.png"
        cfp = args.data_root / args.split / "images" / f"{case_id}.png"
        provenance = {
            "checkpoint_sha256": checkpoint_sha256,
            "moving_sha256": sha256_file(ffa),
            "fixed_sha256": sha256_file(cfp),
            "threshold": float(args.threshold),
            "phase": phase,
        }
        metadata = output.with_suffix(".json")
        if _is_complete(output, metadata, provenance):
            reports[case_id] = json.loads(metadata.read_text(encoding="utf-8"))
            successful += 1
            continue
        print(f"[{index:02d}/{len(ids):02d}] MINIMA {phase} -> CFP {case_id}", flush=True)
        try:
            result = matcher(str(ffa), str(cfp))
            moving, fixed, confidence = _normalise_result(result)
        except Exception as error:
            if failure_policy == "error":
                raise
            traceback.print_exc()
            _discard_incomplete(output, metadata)
            reports[case_id] = {
                "case_id": case_id,
                "status": "identity_fallback",
                "error": f"{type(error).__name__}: {error}",
                "provenance": provenance,
            }
            failed += 1
            continue
        _atomic_matches(output, moving, fixed, confidence)
        report = {"case_id": case_id, "matches": len(moving), "provenance": provenance}
        atomic_json(metadata, report)
        reports[case_id] = report
        successful += 1
        gc.collect()
    summary = {
        "split": args.split,
        "phase": phase,
        "cases": len(ids),
        "successful_cases": successful,
        "failed_cases": failed,
        "load_error": load_error,
        "failure_policy": failure_policy,
        "output_root": str(output_root),
        "cases_report": reports,
    }
    atomic_json(output_root / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract pinned MINIMA LoFTR correspondences for GAVE2.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("training", "validation"), required=True)
    parser.add_argument("--phase", choices=("FFA_A", "FFA_AV"), required=True)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--failure-policy", choices=("error", "identity"), default="error")
    parser.add_argument("--limit-cases", type=int)
    return parser.parse_args(argv)


def main() -> None:
    summary = run(parse_args())
    print(json.dumps({key: value for key, value in summary.items() if key != "cases_report"}, indent=2))


if __name__ == "__main__":
    main()
