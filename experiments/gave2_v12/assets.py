from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.request
from pathlib import Path

from . import MINIMA_COMMIT
from .constants import MINIMA_LOFTR_SHA256, MINIMA_LOFTR_URL, MINIMA_SOURCE_URL
from .utils import atomic_json, sha256_file


def _git(source: Path, *arguments: str) -> str:
    command = ["git", "-c", f"safe.directory={source.resolve().as_posix()}", "-C", str(source), *arguments]
    return subprocess.check_output(command, text=True).strip()


def _ensure_checkpoint(checkpoint: Path) -> str:
    if checkpoint.exists() and sha256_file(checkpoint) == MINIMA_LOFTR_SHA256:
        return MINIMA_LOFTR_SHA256

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    partial = checkpoint.with_suffix(checkpoint.suffix + ".partial")
    partial.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(MINIMA_LOFTR_URL, timeout=60) as response, partial.open("wb") as handle:
            while block := response.read(8 * 1024 * 1024):
                handle.write(block)
        actual = sha256_file(partial)
        if actual != MINIMA_LOFTR_SHA256:
            raise RuntimeError(
                f"MINIMA checkpoint SHA-256 mismatch: expected {MINIMA_LOFTR_SHA256}, got {actual}"
            )
        os.replace(partial, checkpoint)
    finally:
        partial.unlink(missing_ok=True)
    return MINIMA_LOFTR_SHA256


def ensure_minima(source_dir: Path, checkpoint: Path) -> dict[str, object]:
    if not source_dir.exists():
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--recursive", MINIMA_SOURCE_URL, str(source_dir)], check=True)
    if not (source_dir / ".git").exists():
        raise RuntimeError(f"Existing MINIMA path is not a git checkout: {source_dir}")
    try:
        _git(source_dir, "cat-file", "-e", f"{MINIMA_COMMIT}^{{commit}}")
    except subprocess.CalledProcessError:
        _git(source_dir, "fetch", "--depth", "1", "origin", MINIMA_COMMIT)
    _git(source_dir, "checkout", "--detach", MINIMA_COMMIT)
    _git(source_dir, "submodule", "update", "--init", "--recursive")
    actual = _git(source_dir, "rev-parse", "HEAD")
    if actual != MINIMA_COMMIT:
        raise RuntimeError(f"MINIMA source mismatch: {actual}")
    checkpoint_sha256 = _ensure_checkpoint(checkpoint)
    report = {
        "source_url": MINIMA_SOURCE_URL,
        "source_commit": actual,
        "checkpoint_url": MINIMA_LOFTR_URL,
        "checkpoint": str(checkpoint),
        "checkpoint_size": checkpoint.stat().st_size,
        "checkpoint_sha256": checkpoint_sha256,
    }
    atomic_json(checkpoint.parent / "minima_manifest.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acquire pinned MINIMA source and LoFTR weights.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    print(json.dumps(ensure_minima(args.source_dir, args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
