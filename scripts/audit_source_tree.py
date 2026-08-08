from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "archive_manifest.json"
RUNTIME_BUILD_ID = "gave2-v13-r6-r51-fine-calibration"
SUBMISSION_ID = "GAVE2-S013"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def local_import_closure(files: list[Path]) -> list[str]:
    available = {
        module_name(path): path
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
        module = module_name(path)
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

    return sorted(
        available[module].relative_to(ROOT).as_posix()
        for module in required
        if available[module].resolve() not in selected
    )


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["runtime_build_id"] != RUNTIME_BUILD_ID:
        raise RuntimeError("Runtime build ID mismatch")
    if manifest["submission_id"] != SUBMISSION_ID:
        raise RuntimeError("Submission ID mismatch")

    forbidden_roots = {"GAVE2_preliminary", "runs", "checkpoints", "external", "assets"}
    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".pyc", ".pyo", ".zip"}
    actual = []
    forbidden = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ROOT)
        if relative == Path("archive_manifest.json"):
            continue
        actual.append(relative.as_posix())
        if relative.parts[0] in forbidden_roots or path.suffix.lower() in forbidden_suffixes:
            forbidden.append(relative.as_posix())
    if forbidden:
        raise RuntimeError(f"Forbidden release payload: {forbidden}")
    if sorted(actual) != manifest["members"]:
        raise RuntimeError("Source tree differs from archive_manifest.json")

    mismatches = [
        name for name in actual
        if sha256(ROOT / name) != manifest["sha256"][name]
    ]
    if mismatches:
        raise RuntimeError(f"Source hash mismatch: {mismatches}")

    notebooks = [
        ROOT / "submission/GAVE2_R2V2_FFA_Residual_V12_Colab.ipynb",
        ROOT / "submission/GAVE2_Channel_Path_FFA_V13_Colab.ipynb",
    ]
    for notebook in notebooks:
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        for cell in payload["cells"]:
            if cell["cell_type"] != "code":
                continue
            if cell.get("execution_count") is not None or cell.get("outputs"):
                raise RuntimeError(f"Notebook contains execution state: {notebook.name}")
            ast.parse("".join(cell.get("source", [])))

    source_files = [ROOT / name for name in actual]
    missing_imports = local_import_closure(source_files)
    if missing_imports:
        raise RuntimeError(f"Missing local import dependencies: {missing_imports}")

    print(json.dumps({
        "submission_id": SUBMISSION_ID,
        "runtime_build_id": RUNTIME_BUILD_ID,
        "members": len(actual),
        "manifest_hashes": "passed",
        "notebooks": "clean_and_parsed",
        "local_import_closure": "passed",
        "forbidden_payload": "absent",
    }, indent=2))


if __name__ == "__main__":
    main()
