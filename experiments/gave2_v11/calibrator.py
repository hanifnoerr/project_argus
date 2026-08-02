from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .constants import SCORED_TARGETS
from .dataset import load_training_cache


INNER_FOLDS = 5
INNER_REPEATS = 5
OUTER_FOLDS = 5
OUTER_REPEATS = 10
INNER_MIN_GAIN = 0.01
TARGET_MIN_GAIN = 0.03
TARGET_MIN_BOOTSTRAP_PROBABILITY = 0.80
TARGET_MIN_SELECTION_RATE = 0.50
OVERALL_MIN_GAIN = 0.03
MINIMUM_ACCEPTED_TARGETS = 3


@dataclass(frozen=True)
class Candidate:
    group: str
    feature_indices: tuple[int, ...]
    alpha: float
    shrinkage: float


def _splits(size: int, folds: int, repeats: int, seed: int):
    rng = np.random.default_rng(seed)
    for repeat in range(repeats):
        permutation = rng.permutation(size)
        for fold, validation in enumerate(np.array_split(permutation, folds)):
            training = np.setdiff1d(np.arange(size), validation, assume_unique=True)
            yield repeat, fold, training, np.asarray(validation, dtype=int)


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    absolute = np.abs(target - prediction)
    nmae = float(absolute.mean() / max(float(np.median(target)), 1e-8))
    smape = float(np.mean(2.0 * absolute / np.maximum(np.abs(target) + np.abs(prediction), 1e-8)))
    return {"nmae": nmae, "smape": smape, "composite": 0.5 * (nmae + smape)}


def _fit_ridge(features: np.ndarray, target: np.ndarray, candidate: Candidate) -> dict[str, object]:
    selected = features[:, candidate.feature_indices]
    mean = selected.mean(axis=0)
    std = selected.std(axis=0)
    keep = std >= 1e-8
    if not keep.any():
        raise ValueError("Candidate feature group is constant")
    mean = mean[keep]
    std = std[keep]
    x = (selected[:, keep] - mean) / std
    y = np.log(np.maximum(target, 1e-8))
    intercept = float(y.mean())
    beta = np.linalg.solve(
        x.T @ x + np.eye(x.shape[1], dtype=np.float64) * candidate.alpha,
        x.T @ (y - intercept),
    )
    span = float(np.ptp(target))
    return {
        "group": candidate.group,
        "feature_indices": [candidate.feature_indices[index] for index in np.flatnonzero(keep)],
        "alpha": candidate.alpha,
        "shrinkage": candidate.shrinkage,
        "mean": mean,
        "std": std,
        "intercept": intercept,
        "beta": beta,
        "median": float(np.median(target)),
        "minimum": float(max(1e-8, np.min(target) - 0.05 * span)),
        "maximum": float(np.max(target) + 0.05 * span),
    }


def _predict_ridge(model: dict[str, object], features: np.ndarray) -> np.ndarray:
    indices = np.asarray(model["feature_indices"], dtype=int)
    mean = np.asarray(model["mean"], dtype=np.float64)
    std = np.asarray(model["std"], dtype=np.float64)
    beta = np.asarray(model["beta"], dtype=np.float64)
    z = (features[:, indices] - mean) / std
    learned = np.exp(float(model["intercept"]) + z @ beta)
    median = float(model["median"])
    shrinkage = float(model["shrinkage"])
    prediction = np.exp(
        shrinkage * np.log(np.maximum(learned, 1e-8))
        + (1.0 - shrinkage) * math.log(max(median, 1e-8))
    )
    return np.clip(prediction, float(model["minimum"]), float(model["maximum"]))


def _indices(feature_names: list[str], predicate) -> tuple[int, ...]:
    return tuple(index for index, name in enumerate(feature_names) if predicate(name))


def feature_groups(feature_names: list[str], target: str) -> dict[str, tuple[int, ...]]:
    """Define label-independent, target-specific physical feature groups."""

    if target == "AVR":
        primary = _indices(feature_names, lambda name: name.startswith("log_knudtson_ratio_t"))
        density = _indices(feature_names, lambda name: name.startswith("log_zone_density_ratio_t"))
        combined = tuple(sorted(set(primary + density)))
    elif target in {"artery_density", "vein_density"}:
        prefix = target.removesuffix("_density")
        other = "vein" if prefix == "artery" else "artery"
        primary = _indices(
            feature_names,
            lambda name: name.startswith(f"{prefix}_zone_density_t")
            or name in {f"{prefix}_zone_soft", f"{prefix}_zone_soft2"},
        )
        global_group = _indices(
            feature_names,
            lambda name: name.startswith(f"{prefix}_roi_density_t")
            or name in {f"{prefix}_roi_soft", f"{prefix}_roi_soft2"},
        )
        cross = _indices(
            feature_names,
            lambda name: (
                name.startswith(f"{prefix}_roi_density_t")
                or name.startswith(f"{other}_roi_density_t")
                or name in {
                    f"{prefix}_roi_soft",
                    f"{prefix}_roi_soft2",
                    f"{other}_roi_soft",
                    f"{other}_roi_soft2",
                }
            ),
        )
        density = global_group
        combined = cross
    elif target in {"artery_fractal_dimension", "vein_fractal_dimension"}:
        prefix = target.removesuffix("_fractal_dimension")
        primary = _indices(feature_names, lambda name: name.startswith(f"{prefix}_roi_fractal_t"))
        density = _indices(
            feature_names,
            lambda name: name.startswith(f"{prefix}_roi_density_t")
            or name in {f"{prefix}_roi_soft", f"{prefix}_roi_soft2"},
        )
        combined = tuple(sorted(set(primary + density)))
    else:
        raise ValueError(f"Unsupported scored target: {target}")

    groups: dict[str, tuple[int, ...]] = {
        "physical_primary": primary,
        "physical_secondary": density,
        "physical_combined": combined,
    }
    for index in primary:
        groups[f"single:{feature_names[index]}"] = (index,)
    if any(not group for group in groups.values()):
        raise RuntimeError(f"Empty physical feature group for {target}")
    return groups


def candidates_for_target(feature_names: list[str], target: str) -> list[Candidate]:
    output = []
    for group_name, feature_indices in feature_groups(feature_names, target).items():
        for alpha in (1.0, 10.0, 100.0):
            for shrinkage in (0.25, 0.50, 0.75, 1.0):
                output.append(Candidate(group_name, feature_indices, alpha, shrinkage))
    return output


def _cross_validated_candidate(
    features: np.ndarray,
    target: np.ndarray,
    candidate: Candidate,
    *,
    seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    prediction_sum = np.zeros_like(target, dtype=np.float64)
    baseline_sum = np.zeros_like(target, dtype=np.float64)
    counts = np.zeros_like(target, dtype=np.float64)
    for _, _, training, validation in _splits(
        len(target), INNER_FOLDS, INNER_REPEATS, seed
    ):
        model = _fit_ridge(features[training], target[training], candidate)
        prediction_sum[validation] += _predict_ridge(model, features[validation])
        baseline_sum[validation] += float(np.median(target[training]))
        counts[validation] += 1.0
    return _metrics(target, prediction_sum / counts), _metrics(target, baseline_sum / counts)


def select_model(
    features: np.ndarray,
    target: np.ndarray,
    feature_names: list[str],
    target_name: str,
    *,
    seed: int,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    best: tuple[float, Candidate, dict[str, float], dict[str, float]] | None = None
    for candidate in candidates_for_target(feature_names, target_name):
        metrics, baseline = _cross_validated_candidate(features, target, candidate, seed=seed)
        result = (metrics["composite"], candidate, metrics, baseline)
        if best is None or result[0] < best[0]:
            best = result
    assert best is not None
    _, candidate, metrics, baseline = best
    gain = 1.0 - metrics["composite"] / max(baseline["composite"], 1e-8)
    accepted = (
        gain >= INNER_MIN_GAIN
        and metrics["nmae"] < baseline["nmae"]
        and metrics["smape"] < baseline["smape"]
    )
    selection = {
        "accepted": bool(accepted),
        "candidate": asdict(candidate),
        "metrics": metrics,
        "baseline": baseline,
        "relative_gain": float(gain),
    }
    return (_fit_ridge(features, target, candidate) if accepted else None), selection


def _bootstrap_difference(
    target: np.ndarray,
    prediction: np.ndarray,
    baseline: np.ndarray,
    *,
    seed: int,
    samples: int = 5000,
) -> dict[str, float]:
    scale = max(float(np.median(target)), 1e-8)

    def contribution(value: np.ndarray) -> np.ndarray:
        absolute = np.abs(target - value)
        return 0.5 * (
            absolute / scale
            + 2.0 * absolute / np.maximum(np.abs(target) + np.abs(value), 1e-8)
        )

    difference = contribution(prediction) - contribution(baseline)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(target), size=(samples, len(target)))
    bootstrapped = difference[indices].mean(axis=1)
    return {
        "mean_difference": float(difference.mean()),
        "ci80_low": float(np.quantile(bootstrapped, 0.10)),
        "ci80_high": float(np.quantile(bootstrapped, 0.90)),
        "ci90_low": float(np.quantile(bootstrapped, 0.05)),
        "ci90_high": float(np.quantile(bootstrapped, 0.95)),
        "probability_improvement": float(np.mean(bootstrapped < 0.0)),
    }


def nested_audit(
    features: np.ndarray,
    targets: dict[str, np.ndarray],
    feature_names: list[str],
    *,
    seed: int = 77,
) -> tuple[dict[str, object], dict[str, object]]:
    report_targets: dict[str, object] = {}
    fitted_targets: dict[str, object] = {}
    accepted_targets: list[str] = []
    overall_model_errors = []
    overall_baseline_errors = []

    for target_index, target_name in enumerate(SCORED_TARGETS):
        target = np.asarray(targets[target_name], dtype=np.float64)
        prediction_sum = np.zeros_like(target)
        baseline_sum = np.zeros_like(target)
        counts = np.zeros_like(target)
        learned_selections = 0
        selection_summaries: list[dict[str, object]] = []
        for repeat, fold, training, validation in _splits(
            len(target), OUTER_FOLDS, OUTER_REPEATS, seed + 1000 * target_index
        ):
            model, selection = select_model(
                features[training],
                target[training],
                feature_names,
                target_name,
                seed=seed + 100000 * target_index + 1000 * repeat + fold,
            )
            baseline_sum[validation] += float(np.median(target[training]))
            if model is None:
                prediction_sum[validation] += float(np.median(target[training]))
            else:
                prediction_sum[validation] += _predict_ridge(model, features[validation])
                learned_selections += 1
            counts[validation] += 1.0
            selection_summaries.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "accepted": selection["accepted"],
                    "group": selection["candidate"]["group"],
                    "alpha": selection["candidate"]["alpha"],
                    "shrinkage": selection["candidate"]["shrinkage"],
                    "relative_gain": selection["relative_gain"],
                }
            )
        prediction = prediction_sum / counts
        baseline_prediction = baseline_sum / counts
        metrics = _metrics(target, prediction)
        baseline_metrics = _metrics(target, baseline_prediction)
        gain = 1.0 - metrics["composite"] / max(baseline_metrics["composite"], 1e-8)
        selection_rate = learned_selections / (OUTER_FOLDS * OUTER_REPEATS)
        bootstrap = _bootstrap_difference(
            target,
            prediction,
            baseline_prediction,
            seed=seed + 7000 + target_index,
        )
        final_model, final_selection = select_model(
            features,
            target,
            feature_names,
            target_name,
            seed=seed + 9000 + target_index,
        )
        accepted = (
            final_model is not None
            and gain >= TARGET_MIN_GAIN
            and metrics["nmae"] < baseline_metrics["nmae"]
            and metrics["smape"] < baseline_metrics["smape"]
            and bootstrap["probability_improvement"] >= TARGET_MIN_BOOTSTRAP_PROBABILITY
            and selection_rate >= TARGET_MIN_SELECTION_RATE
        )
        if accepted:
            accepted_targets.append(target_name)
            fitted_targets[target_name] = _serializable_model(final_model)
            overall_model_errors.append(metrics["composite"])
        else:
            overall_model_errors.append(baseline_metrics["composite"])
        overall_baseline_errors.append(baseline_metrics["composite"])
        report_targets[target_name] = {
            "accepted": bool(accepted),
            "nested_metrics": metrics,
            "nested_baseline": baseline_metrics,
            "nested_relative_gain": float(gain),
            "selection_rate": float(selection_rate),
            "bootstrap": bootstrap,
            "final_selection": final_selection,
            "outer_selections": selection_summaries,
        }

    overall_gain = 1.0 - float(np.mean(overall_model_errors)) / max(
        float(np.mean(overall_baseline_errors)), 1e-8
    )
    passed = len(accepted_targets) >= MINIMUM_ACCEPTED_TARGETS and overall_gain >= OVERALL_MIN_GAIN
    report = {
        "version": 11,
        "protocol": {
            "outer_folds": OUTER_FOLDS,
            "outer_repeats": OUTER_REPEATS,
            "inner_folds": INNER_FOLDS,
            "inner_repeats": INNER_REPEATS,
            "seed": seed,
            "target_min_gain": TARGET_MIN_GAIN,
            "target_min_bootstrap_probability": TARGET_MIN_BOOTSTRAP_PROBABILITY,
            "target_min_selection_rate": TARGET_MIN_SELECTION_RATE,
            "overall_min_gain": OVERALL_MIN_GAIN,
            "minimum_accepted_targets": MINIMUM_ACCEPTED_TARGETS,
        },
        "passed": bool(passed),
        "accepted_targets": accepted_targets,
        "overall_nested_relative_gain": float(overall_gain),
        "targets": report_targets,
    }
    calibrator = {
        "version": 11,
        "feature_names": feature_names,
        "accepted_targets": accepted_targets,
        "models": fitted_targets,
        "audit_passed": bool(passed),
        "audit_sha256": "",
    }
    return report, calibrator


def _serializable_model(model: dict[str, object]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in model.items():
        if isinstance(value, np.ndarray):
            output[key] = value.tolist()
        elif isinstance(value, np.generic):
            output[key] = value.item()
        else:
            output[key] = value
    return output


def predict_targets(calibrator: dict[str, object], vector: np.ndarray) -> tuple[dict[str, float], dict[str, object]]:
    feature = np.asarray(vector, dtype=np.float64)[None, :]
    values: dict[str, float] = {}
    diagnostics: dict[str, object] = {}
    for target_name in calibrator["accepted_targets"]:
        model = calibrator["models"][target_name]
        indices = np.asarray(model["feature_indices"], dtype=int)
        z = (feature[0, indices] - np.asarray(model["mean"])) / np.asarray(model["std"])
        ood = not np.isfinite(z).all() or bool(np.max(np.abs(z)) > 8.0)
        diagnostics[target_name] = {"max_abs_z": float(np.max(np.abs(z))), "ood": bool(ood)}
        if not ood:
            values[target_name] = float(_predict_ridge(model, feature)[0])
    return values, diagnostics


def run_audit(cache: Path, output_dir: Path, seed: int = 77) -> dict[str, object]:
    features, targets, feature_names, case_ids = load_training_cache(cache)
    report, calibrator = nested_audit(features, targets, feature_names, seed=seed)
    report["cache"] = str(cache)
    report["training_cases"] = case_ids
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "nested_cv_audit.json"
    report_text = json.dumps(report, indent=2, sort_keys=True)
    report_path.write_text(report_text + "\n", encoding="utf-8")
    calibrator["audit_sha256"] = hashlib.sha256(report_text.encode("utf-8")).hexdigest()
    calibrator_path = output_dir / "task3_calibrator.json"
    calibrator_path.write_text(json.dumps(calibrator, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"report": str(report_path), "calibrator": str(calibrator_path), **report}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the leakage-controlled GAVE2 V11 Task 3 audit.")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=77)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    print(json.dumps(run_audit(args.cache, args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
