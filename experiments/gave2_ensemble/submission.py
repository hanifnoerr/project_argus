from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SubmissionValidationReport:
    ok: bool
    errors: list[str]
    counts: dict[str, int]


def _as_numpy(array) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array)


def save_probability_png(probability, path: Path | str) -> None:
    arr = _as_numpy(probability).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected CHW or HWC probability array, got {arr.shape}")
    if arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    if arr.shape[2] != 3:
        raise ValueError(f"Expected 3 output channels, got {arr.shape}")
    out = np.clip(arr, 0.0, 1.0)
    out = np.rint(out * 255.0).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode="RGB").save(path)


def load_probability_png(path: Path | str) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def create_submission_tree(root: Path | str, team_id: str = "team_id") -> Path:
    base = Path(root) / team_id
    for task in ("Task1", "Task2", "Task3"):
        (base / task).mkdir(parents=True, exist_ok=True)
    return base


def validate_submission_tree(
    submission_root: Path | str,
    expected_cases: Iterable[str],
    expected_size: tuple[int, int] = (1024, 1536),
) -> SubmissionValidationReport:
    root = Path(submission_root)
    errors: list[str] = []
    counts: dict[str, int] = {}
    expected = list(expected_cases)
    expected_pngs = {f"{case_id}.png" for case_id in expected}
    expected_txts = {f"{case_id}.txt" for case_id in expected}
    expected_h, expected_w = expected_size

    for task in ("Task1", "Task2"):
        task_dir = root / task
        files = sorted(path.name for path in task_dir.glob("*.png")) if task_dir.exists() else []
        counts[task] = len(files)
        missing = sorted(expected_pngs - set(files))
        extra = sorted(set(files) - expected_pngs)
        if missing:
            errors.append(f"{task}: missing PNGs {missing[:5]}")
        if extra:
            errors.append(f"{task}: unexpected PNGs {extra[:5]}")
        for name in files:
            path = task_dir / name
            try:
                with Image.open(path) as image:
                    mode = image.mode
                    size = image.size
                if mode != "RGB":
                    errors.append(f"{task}/{name}: expected RGB mode, got {mode}")
                if size != (expected_w, expected_h):
                    errors.append(
                        f"{task}/{name}: expected {expected_w}x{expected_h}, got {size[0]}x{size[1]}"
                    )
            except Exception as exc:  # pragma: no cover - defensive validation path
                errors.append(f"{task}/{name}: could not read PNG: {exc}")

    task3_dir = root / "Task3"
    txt_files = sorted(path.name for path in task3_dir.glob("*.txt")) if task3_dir.exists() else []
    counts["Task3"] = len(txt_files)
    missing_txt = sorted(expected_txts - set(txt_files))
    extra_txt = sorted(set(txt_files) - expected_txts)
    if missing_txt:
        errors.append(f"Task3: missing TXTs {missing_txt[:5]}")
    if extra_txt:
        errors.append(f"Task3: unexpected TXTs {extra_txt[:5]}")
    for name in txt_files:
        text = (task3_dir / name).read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            errors.append(f"Task3/{name}: empty biomarker file")

    return SubmissionValidationReport(ok=not errors, errors=errors, counts=counts)
