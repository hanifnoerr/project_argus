from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from . import R2V2_SOURCE_COMMIT


SOURCE_URL = "https://github.com/j-morano/R2-V2.git"
RELEASE_TAG = "v1"
ASSETS = {
    "av.pth": {
        "url": "https://github.com/j-morano/R2-V2/releases/download/v1/av.pth",
        "sha256": "74d425afb714384cb3f4d5db9cc852c1ea6d7552e46c866e29a3777db12b9d80",
        "size": 248305890,
    },
    "av_config.json": {
        "url": "https://github.com/j-morano/R2-V2/releases/download/v1/av_config.json",
        "sha256": "8c4bb170f0f4df5cc21ce6929ac1e6e738c82404fe420310181974f572beff54",
        "size": 626,
    },
    "bv.pth": {
        "url": "https://github.com/j-morano/R2-V2/releases/download/v1/bv.pth",
        "sha256": "db816a3867e8bc235661e76def115ef9a0a865fb34fd4f4c8259586b7f096a61",
        "size": 248311778,
    },
    "bv_config.json": {
        "url": "https://github.com/j-morano/R2-V2/releases/download/v1/bv_config.json",
        "sha256": "06dcf9b6598bfe9cfbbf8f42d3706a4cfa9e33ac1810e27b2b97db7c56f6cca7",
        "size": 485,
    },
}


def sha256_file(path: Path | str, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def verify_asset(path: Path | str, name: str) -> None:
    path = Path(path)
    expected = ASSETS[name]
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(expected["size"]):
        raise RuntimeError(f"Size mismatch for {path}: {path.stat().st_size} != {expected['size']}")
    actual = sha256_file(path)
    if actual.lower() != str(expected["sha256"]).lower():
        raise RuntimeError(f"SHA256 mismatch for {path}: {actual}")


def download_asset(name: str, output_dir: Path | str) -> Path:
    if name not in ASSETS:
        raise KeyError(name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / name
    if destination.exists():
        verify_asset(destination, name)
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(
        str(ASSETS[name]["url"]),
        headers={"User-Agent": "MICCAI2026-GAVE2-V8", **({"Range": f"bytes={existing}-"} if existing else {})},
    )
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as error:
        if error.code == 416 and partial.exists():
            os.replace(partial, destination)
            verify_asset(destination, name)
            return destination
        raise
    append = existing > 0 and getattr(response, "status", None) == 206
    mode = "ab" if append else "wb"
    written = existing if append else 0
    expected_size = int(ASSETS[name]["size"])
    with response, partial.open(mode) as handle:
        while block := response.read(8 * 1024 * 1024):
            handle.write(block)
            written += len(block)
            print(f"\r{name}: {written / expected_size:6.1%}", end="", flush=True)
    print()
    os.replace(partial, destination)
    verify_asset(destination, name)
    return destination


def _git(source_dir: Path, *arguments: str) -> str:
    command = ["git", "-c", f"safe.directory={source_dir.resolve().as_posix()}", "-C", str(source_dir), *arguments]
    return subprocess.check_output(command, text=True).strip()


def ensure_source(source_dir: Path | str) -> Path:
    source_dir = Path(source_dir)
    if not source_dir.exists():
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", SOURCE_URL, str(source_dir)], check=True)
    if not (source_dir / ".git").exists():
        raise RuntimeError(f"Existing R2-V2 source is not a git checkout: {source_dir}")
    try:
        _git(source_dir, "cat-file", "-e", f"{R2V2_SOURCE_COMMIT}^{{commit}}")
    except subprocess.CalledProcessError:
        _git(source_dir, "fetch", "--depth", "1", "origin", R2V2_SOURCE_COMMIT)
    _git(source_dir, "checkout", "--detach", R2V2_SOURCE_COMMIT)
    actual = _git(source_dir, "rev-parse", "HEAD")
    if actual != R2V2_SOURCE_COMMIT:
        raise RuntimeError(f"R2-V2 source mismatch: {actual}")
    return source_dir


def prepare_r2v2(source_dir: Path | str, weights_dir: Path | str) -> dict[str, object]:
    source = ensure_source(source_dir)
    weights = Path(weights_dir)
    for name in ASSETS:
        download_asset(name, weights)
    manifest = {
        "source_url": SOURCE_URL,
        "source_commit": R2V2_SOURCE_COMMIT,
        "release_tag": RELEASE_TAG,
        "assets": ASSETS,
    }
    manifest_path = weights / "r2v2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acquire and verify pinned R2-V2 source and weights.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    print(json.dumps(prepare_r2v2(args.source_dir, args.weights_dir), indent=2))


if __name__ == "__main__":
    main()

