from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gave2_v13 import RUNTIME_BUILD_ID


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _included_files(code_only: bool) -> list[Path]:
    roots = [
        ROOT / "experiments/gave2_v13",
        ROOT / "experiments/gave2_v12",
        ROOT / "experiments/gave2_v11",
        ROOT / "experiments/gave2_v8",
        ROOT / "experiments/gave2_ensemble",
        ROOT / "tests/gave2_v13",
        ROOT / "tests/gave2_v12",
        ROOT / "tests/gave2_v8",
    ]
    if not code_only:
        roots.insert(0, ROOT / "GAVE2_preliminary")
    individual = [
        ROOT / "experiments/__init__.py",
        ROOT / "submission/GAVE2_Channel_Path_FFA_V13_Colab.ipynb",
        ROOT / "submission/GAVE2_R2V2_FFA_Residual_V12_Colab.ipynb",
        ROOT / "scripts/build_miccai_v13_archive.py",
        ROOT / "scripts/audit_miccai_v13_archive.py",
    ]
    files: list[Path] = []
    for root in roots:
        files.extend(path for path in root.rglob("*") if path.is_file())
    files.extend(path for path in individual if path.is_file())
    forbidden_suffixes = {".pyc", ".pyo", ".pt", ".pth", ".ckpt"}
    return sorted(
        {
            path.resolve()
            for path in files
            if "__pycache__" not in path.parts
            and ".pytest_cache" not in path.parts
            and path.suffix.lower() not in forbidden_suffixes
        },
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _validate_local_import_closure(files: list[Path]) -> None:
    available = {
        _module_name(path): path.resolve()
        for path in ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    selected = {path.resolve() for path in files}
    required: set[str] = set()

    def require(module: str) -> None:
        parts = module.split(".")
        for index in range(1, len(parts) + 1):
            candidate = ".".join(parts[:index])
            if candidate in available:
                required.add(candidate)

    for path in files:
        if path.suffix != ".py":
            continue
        module = _module_name(path)
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    require(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = package.split(".") if package else []
                    parent_count = node.level - 1
                    if parent_count > len(parts):
                        continue
                    base = ".".join(parts[: len(parts) - parent_count])
                    if node.module:
                        base = ".".join(value for value in (base, node.module) if value)
                else:
                    base = node.module or ""
                if base:
                    require(base)
                    for alias in node.names:
                        require(f"{base}.{alias.name}")
    missing = sorted(
        available[module].relative_to(ROOT).as_posix()
        for module in required
        if available[module] not in selected
    )
    if missing:
        raise RuntimeError("Archive omits local import dependencies: " + ", ".join(missing))


def build_archive(output: Path, *, code_only: bool, force: bool = False) -> dict[str, object]:
    output = output.resolve()
    if output.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --force after reviewing the target")
    files = _included_files(code_only)
    if not files:
        raise RuntimeError("Archive payload is empty")
    _validate_local_import_closure(files)
    hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in files}
    manifest = {
        "version": 13,
        "runtime_build_id": RUNTIME_BUILD_ID,
        "kind": "source-only" if code_only else "colab-runtime-with-dataset",
        "members": sorted(hashes),
        "sha256": hashes,
        "forbidden": ["runs/", "checkpoints", ".pt", ".pth", ".ckpt"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files:
                archive.write(path, path.relative_to(ROOT).as_posix())
            archive.writestr("archive_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        with zipfile.ZipFile(temporary) as archive:
            corrupt = archive.testzip()
            names = set(archive.namelist())
            if corrupt is not None:
                raise RuntimeError(f"Archive CRC failure: {corrupt}")
            if names != set(manifest["members"]) | {"archive_manifest.json"}:
                raise RuntimeError("Archive members differ from manifest")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "output": str(output),
        "runtime_build_id": RUNTIME_BUILD_ID,
        "kind": manifest["kind"],
        "members": len(files),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the self-auditing GAVE2 V13 archive.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(build_archive(args.output, code_only=args.code_only, force=args.force), indent=2))
