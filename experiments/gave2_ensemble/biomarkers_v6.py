from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .biomarkers import BIOMARKER_KEYS, write_biomarker_txt
from .biomarkers_v2 import extract_biomarker_features, read_biomarker_txt
from .data import list_case_ids, read_png_float
from .predict_v6 import FloatProbabilityStore


SCORED_TARGETS = (
    "AVR",
    "artery_density",
    "vein_density",
    "artery_fractal_dimension",
    "vein_fractal_dimension",
)


def _ridge_fit(features: np.ndarray, target: np.ndarray, alpha: float) -> dict[str, object]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-8] = 1.0
    x = (features - mean) / std
    y = np.log(np.maximum(target, 1e-8))
    intercept = float(y.mean())
    beta = np.linalg.solve(x.T @ x + np.eye(x.shape[1]) * alpha, x.T @ (y - intercept))
    return {"mean": mean, "std": std, "intercept": intercept, "beta": beta}


def _ridge_predict(model: dict[str, object], features: np.ndarray) -> np.ndarray:
    value = model["intercept"] + ((features - model["mean"]) / model["std"]) @ model["beta"]
    return np.exp(value)


def _errors(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    nmae = float(np.mean(np.abs(target - prediction)) / max(float(np.median(target)), 1e-8))
    smape = float(np.mean(2.0 * np.abs(target - prediction) / np.maximum(target + prediction, 1e-8)))
    return nmae, smape


def fit_target_specific_models(
    features: np.ndarray,
    targets: dict[str, np.ndarray],
    feature_names: list[str],
    *,
    folds: int = 3,
    repeats: int = 10,
    seed: int = 77,
) -> dict[str, object]:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < folds or x.shape[1] != len(feature_names):
        raise ValueError("Invalid biomarker feature matrix")
    rng = np.random.default_rng(seed)
    splits = [np.array_split(rng.permutation(x.shape[0]), folds) for _ in range(repeats)]
    output: dict[str, object] = {}
    for key in BIOMARKER_KEYS:
        y = np.asarray(targets[key], dtype=np.float64)
        if y.shape != (x.shape[0],) or np.any(y <= 0) or not np.isfinite(y).all():
            raise ValueError(f"Invalid target {key}")
        best = None
        for alpha in (0.1, 1.0, 10.0, 100.0):
            prediction_sum = np.zeros_like(y)
            baseline_sum = np.zeros_like(y)
            counts = np.zeros_like(y)
            for repeat in splits:
                for validation in repeat:
                    training = np.setdiff1d(np.arange(x.shape[0]), validation)
                    model = _ridge_fit(x[training], y[training], alpha)
                    prediction_sum[validation] += _ridge_predict(model, x[validation])
                    baseline_sum[validation] += np.median(y[training])
                    counts[validation] += 1
            predicted = prediction_sum / counts
            baseline = baseline_sum / counts
            for shrinkage in (0.25, 0.5, 0.75, 1.0):
                mixed = np.exp(shrinkage * np.log(np.maximum(predicted, 1e-8)) + (1 - shrinkage) * np.log(baseline))
                metrics = _errors(y, mixed)
                baseline_metrics = _errors(y, baseline)
                candidate = (sum(metrics), alpha, shrinkage, metrics, baseline_metrics)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        assert best is not None
        _, alpha, shrinkage, metrics, baseline_metrics = best
        accepted = metrics[0] < baseline_metrics[0] and metrics[1] < baseline_metrics[1]
        fitted = _ridge_fit(x, y, alpha)
        span = float(y.max() - y.min())
        output[key] = {
            "accepted": bool(accepted),
            "alpha": float(alpha),
            "shrinkage": float(shrinkage if accepted else 0.0),
            "median": float(np.median(y)),
            "minimum": float(max(1e-8, y.min() - 0.05 * span)),
            "maximum": float(y.max() + 0.05 * span),
            "mean": fitted["mean"].tolist(),
            "std": fitted["std"].tolist(),
            "intercept": float(fitted["intercept"]),
            "beta": fitted["beta"].tolist(),
            "cv_nmae": metrics[0],
            "cv_smape": metrics[1],
            "baseline_nmae": baseline_metrics[0],
            "baseline_smape": baseline_metrics[1],
            "ood_z_limit": 8.0,
        }
    return {"version": 6, "feature_names": feature_names, "targets": output, "scored_targets": list(SCORED_TARGETS)}


def predict_target_specific(model: dict[str, object], feature_vector: np.ndarray) -> dict[str, float]:
    vector = np.asarray(feature_vector, dtype=np.float64)
    values: dict[str, float] = {}
    for key in BIOMARKER_KEYS:
        target = model["targets"][key]
        mean = np.asarray(target["mean"], dtype=np.float64)
        std = np.asarray(target["std"], dtype=np.float64)
        z = (vector - mean) / np.maximum(std, 1e-8)
        value = float(target["median"])
        if bool(target["accepted"]) and np.isfinite(z).all() and np.max(np.abs(z)) <= float(target["ood_z_limit"]):
            beta = np.asarray(target["beta"], dtype=np.float64)
            learned = math.exp(float(target["intercept"]) + float(z @ beta))
            shrinkage = float(target["shrinkage"])
            value = math.exp(shrinkage * math.log(max(learned, 1e-8)) + (1 - shrinkage) * math.log(value))
        values[key] = float(np.clip(value, target["minimum"], target["maximum"]))

    # Preserve the directly predicted AVR while making CRAE and CRVE consistent with it.
    avr = max(values["AVR"], 1e-8)
    geometric = math.sqrt(max(values["CRAE"] * values["CRVE"], 1e-8))
    values["CRAE"] = geometric * math.sqrt(avr)
    values["CRVE"] = geometric / math.sqrt(avr)
    values["AVR"] = values["CRAE"] / values["CRVE"]
    return values


def _case_features(data_root: Path, split: str, store: FloatProbabilityStore, case_id: str):
    probability = store.read_case(case_id)
    cfp = read_png_float(data_root / split / "images" / f"{case_id}.png", channels=3)
    roi = read_png_float(data_root / split / "masks" / f"{case_id}.png", channels=1)[..., 0]
    return extract_biomarker_features(probability, cfp, roi)


def fit_from_oof_store(data_root: Path, prediction_root: Path, output: Path) -> Path:
    case_ids = list_case_ids(data_root, split="training")
    store = FloatProbabilityStore(prediction_root, task="task2", split="oof")
    vectors: list[np.ndarray] = []
    names: list[str] | None = None
    targets = {key: [] for key in BIOMARKER_KEYS}
    localization: dict[str, object] = {}
    for case_id in case_ids:
        vector, current_names, metadata = _case_features(data_root, "training", store, case_id)
        names = current_names if names is None else names
        if current_names != names:
            raise RuntimeError("Biomarker feature schema changed between cases")
        vectors.append(vector)
        localization[case_id] = metadata
        values = read_biomarker_txt(data_root / "training" / "biomarker" / f"{case_id}.txt")
        for key in BIOMARKER_KEYS:
            targets[key].append(values[key])
    model = fit_target_specific_models(
        np.stack(vectors),
        {key: np.asarray(value) for key, value in targets.items()},
        names or [],
    )
    model["training_case_ids"] = case_ids
    model["localization_qa"] = localization
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2), encoding="utf-8")
    return output


def generate_task3_from_store(
    data_root: Path,
    prediction_root: Path,
    model_path: Path,
    output_dir: Path,
    *,
    split: str = "validation",
) -> Path:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    store_split = "oof" if split == "training" else "validation"
    store = FloatProbabilityStore(prediction_root, task="task2", split=store_split)
    output_dir.mkdir(parents=True, exist_ok=True)
    for case_id in list_case_ids(data_root, split=split):
        try:
            vector, names, metadata = _case_features(data_root, split, store, case_id)
            if names != model["feature_names"] or float(metadata.get("od_confidence", 0.0)) < 0.01:
                raise ValueError("Optic-disc localization QA failed")
            values = predict_target_specific(model, vector)
        except (ValueError, RuntimeError, FileNotFoundError):
            values = {key: float(model["targets"][key]["median"]) for key in BIOMARKER_KEYS}
            avr = max(values["AVR"], 1e-8)
            geometric = math.sqrt(values["CRAE"] * values["CRVE"])
            values["CRAE"] = geometric * math.sqrt(avr)
            values["CRVE"] = geometric / math.sqrt(avr)
        write_biomarker_txt(values, output_dir / f"{case_id}.txt")
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GAVE2 V6 target-specific Task 3 modeling.")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--data-root", type=Path, required=True)
    fit.add_argument("--prediction-root", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    predict = sub.add_parser("predict")
    predict.add_argument("--data-root", type=Path, required=True)
    predict.add_argument("--prediction-root", type=Path, required=True)
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--split", choices=("training", "validation"), default="validation")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.command == "fit":
        result = fit_from_oof_store(args.data_root, args.prediction_root, args.output)
    else:
        result = generate_task3_from_store(args.data_root, args.prediction_root, args.model, args.output_dir, split=args.split)
    print(result)


if __name__ == "__main__":
    main()
