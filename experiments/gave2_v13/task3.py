from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from experiments.gave2_v11.constants import BIOMARKER_KEYS
from experiments.gave2_v11.dataset import read_biomarker_txt
from experiments.gave2_v11.features import extract_task3_features, load_rgb_float, load_roi, zone_c_mask
from experiments.gave2_v12.prepare import PreparedFFACache
from experiments.gave2_v12.utils import atomic_json, case_ids

from .selection import _open_store


PATCHABLE_TARGETS = ("AVR", "vein_density")


@dataclass(frozen=True)
class Candidate:
    group: str
    indices: tuple[int, ...]
    alpha: float
    shrinkage: float


def _append(values: list[float], names: list[str], name: str, value: float) -> None:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite Task 3 feature {name}")
    values.append(value)
    names.append(name)


def _weighted_mean(value: np.ndarray, weight: np.ndarray, roi: np.ndarray) -> float:
    selected_weight = np.clip(np.asarray(weight, dtype=np.float64), 0.0, 1.0) * roi
    return float((np.asarray(value, dtype=np.float64) * selected_weight).sum() / max(selected_weight.sum(), 1e-8))


def extract_features(
    probability: np.ndarray,
    cfp: np.ndarray,
    roi: np.ndarray,
    ffa: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    base, base_names, metadata = extract_task3_features(probability, cfp, roi)
    values = [float(value) for value in base]
    names = list(base_names)
    zone = zone_c_mask(roi.shape, (metadata["od_x"], metadata["od_y"]), metadata["od_diameter"], roi)
    channel_names = ("early", "late", "difference", "artery_cue", "vein_cue")
    for channel, channel_name in zip(ffa, channel_names, strict=True):
        for region, region_name in ((roi, "roi"), (zone, "zone")):
            selected = np.asarray(channel, dtype=np.float64)[region]
            _append(values, names, f"ffa_{channel_name}_{region_name}_mean", selected.mean())
            _append(values, names, f"ffa_{channel_name}_{region_name}_std", selected.std())
            _append(values, names, f"ffa_{channel_name}_{region_name}_p90", np.percentile(selected, 90.0))
        _append(values, names, f"ffa_{channel_name}_artery_weighted", _weighted_mean(channel, probability[0], roi))
        _append(values, names, f"ffa_{channel_name}_vein_weighted", _weighted_mean(channel, probability[2], roi))

    early_a = values[names.index("ffa_early_artery_weighted")]
    late_v = values[names.index("ffa_late_vein_weighted")]
    artery_cue = values[names.index("ffa_artery_cue_artery_weighted")]
    vein_cue = values[names.index("ffa_vein_cue_vein_weighted")]
    _append(values, names, "ffa_log_early_artery_late_vein_ratio", math.log((early_a + 1e-4) / (late_v + 1e-4)))
    _append(values, names, "ffa_artery_minus_vein_cue", artery_cue - vein_cue)
    vector = np.asarray(values, dtype=np.float64)
    if len(set(names)) != len(names) or not np.isfinite(vector).all():
        raise RuntimeError("Invalid V13 Task 3 feature schema")
    return vector, names


def _feature_groups(names: list[str], target: str) -> dict[str, tuple[int, ...]]:
    def indices(predicate) -> tuple[int, ...]:
        return tuple(index for index, name in enumerate(names) if predicate(name))

    if target == "AVR":
        base = indices(lambda name: name.startswith("log_knudtson_ratio_t") or name.startswith("log_zone_density_ratio_t"))
        ffa = indices(
            lambda name: name in {"ffa_log_early_artery_late_vein_ratio", "ffa_artery_minus_vein_cue"}
            or name.endswith("_artery_weighted")
            or name.endswith("_vein_weighted")
        )
    elif target == "vein_density":
        base = indices(
            lambda name: name.startswith("vein_zone_density_t")
            or name.startswith("vein_roi_density_t")
            or name in {"vein_zone_soft", "vein_zone_soft2", "vein_roi_soft", "vein_roi_soft2"}
        )
        ffa = indices(
            lambda name: name.startswith("ffa_late_")
            or name.startswith("ffa_vein_cue_")
            or name == "ffa_artery_minus_vein_cue"
        )
    else:
        raise ValueError(target)
    groups = {
        "ffa_only": ffa,
        "hybrid": tuple(sorted(set(base + ffa))),
    }
    if any(not value for value in groups.values()):
        raise RuntimeError(f"Empty V13 Task 3 group for {target}")
    preferred_singles = {
        "AVR": {
            "ffa_log_early_artery_late_vein_ratio",
            "ffa_artery_minus_vein_cue",
            "ffa_early_artery_weighted",
            "ffa_late_vein_weighted",
        },
        "vein_density": {
            "ffa_late_zone_mean",
            "ffa_late_zone_p90",
            "ffa_late_vein_weighted",
            "ffa_vein_cue_vein_weighted",
        },
    }[target]
    for index in ffa:
        if names[index] not in preferred_singles:
            continue
        groups[f"ffa_single:{names[index]}"] = (index,)
    return groups


def _candidates(names: list[str], target: str) -> list[Candidate]:
    return [
        Candidate(group, indices, alpha, shrinkage)
        for group, indices in _feature_groups(names, target).items()
        for alpha in (1.0, 10.0, 100.0)
        for shrinkage in (0.25, 0.5, 0.75, 1.0)
    ]


def _splits(size: int, folds: int, repeats: int, seed: int):
    rng = np.random.default_rng(seed)
    all_indices = np.arange(size)
    for repeat in range(repeats):
        permutation = rng.permutation(size)
        for fold, validation in enumerate(np.array_split(permutation, folds)):
            training = np.setdiff1d(all_indices, validation, assume_unique=True)
            yield repeat, fold, training, np.asarray(validation, dtype=int)


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.abs(np.asarray(target) - np.asarray(prediction))
    nmae = float(error.mean() / max(float(np.median(target)), 1e-8))
    smape = float(np.mean(2.0 * error / np.maximum(np.abs(target) + np.abs(prediction), 1e-8)))
    return {"nmae": nmae, "smape": smape, "composite": 0.5 * (nmae + smape)}


def _fit(features: np.ndarray, target: np.ndarray, candidate: Candidate) -> dict[str, object]:
    selected = features[:, candidate.indices]
    mean, std = selected.mean(axis=0), selected.std(axis=0)
    keep = std >= 1e-8
    if not keep.any():
        raise ValueError("Constant Task 3 candidate")
    used_indices = np.asarray(candidate.indices)[keep]
    mean, std = mean[keep], std[keep]
    x = (features[:, used_indices] - mean) / std
    y = np.log(np.maximum(target, 1e-8))
    intercept = float(y.mean())
    beta = np.linalg.solve(
        x.T @ x + candidate.alpha * np.eye(x.shape[1], dtype=np.float64),
        x.T @ (y - intercept),
    )
    span = float(np.ptp(target))
    return {
        "group": candidate.group,
        "indices": used_indices.tolist(),
        "alpha": candidate.alpha,
        "shrinkage": candidate.shrinkage,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "intercept": intercept,
        "beta": beta.tolist(),
        "median": float(np.median(target)),
        "minimum": float(max(np.min(target) - 0.05 * span, 1e-8)),
        "maximum": float(np.max(target) + 0.05 * span),
    }


def _predict(model: dict[str, object], features: np.ndarray) -> np.ndarray:
    indices = np.asarray(model["indices"], dtype=int)
    x = (features[:, indices] - np.asarray(model["mean"])) / np.asarray(model["std"])
    learned = np.exp(float(model["intercept"]) + x @ np.asarray(model["beta"]))
    median = float(model["median"])
    shrinkage = float(model["shrinkage"])
    value = np.exp(
        shrinkage * np.log(np.maximum(learned, 1e-8))
        + (1.0 - shrinkage) * math.log(max(median, 1e-8))
    )
    return np.clip(value, float(model["minimum"]), float(model["maximum"]))


def _select(
    features: np.ndarray,
    target: np.ndarray,
    names: list[str],
    target_name: str,
    seed: int,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    best: tuple[float, Candidate, dict[str, float], dict[str, float]] | None = None
    for candidate in _candidates(names, target_name):
        prediction = np.zeros_like(target, dtype=np.float64)
        baseline = np.zeros_like(target, dtype=np.float64)
        counts = np.zeros_like(target, dtype=np.float64)
        try:
            for _, _, training, validation in _splits(len(target), 5, 3, seed):
                model = _fit(features[training], target[training], candidate)
                prediction[validation] += _predict(model, features[validation])
                baseline[validation] += float(np.median(target[training]))
                counts[validation] += 1.0
        except (ValueError, np.linalg.LinAlgError):
            continue
        trial_metrics = _metrics(target, prediction / counts)
        baseline_metrics = _metrics(target, baseline / counts)
        value = (trial_metrics["composite"], candidate, trial_metrics, baseline_metrics)
        if best is None or value[0] < best[0]:
            best = value
    if best is None:
        raise RuntimeError(f"No valid Task 3 candidates for {target_name}")
    _, candidate, trial_metrics, baseline_metrics = best
    gain = 1.0 - trial_metrics["composite"] / max(baseline_metrics["composite"], 1e-8)
    accepted = (
        gain >= 0.02
        and trial_metrics["nmae"] < baseline_metrics["nmae"]
        and trial_metrics["smape"] < baseline_metrics["smape"]
    )
    report = {
        "accepted": bool(accepted),
        "candidate": asdict(candidate),
        "metrics": trial_metrics,
        "baseline": baseline_metrics,
        "relative_gain": float(gain),
    }
    return (_fit(features, target, candidate) if accepted else None), report


def _bootstrap(target: np.ndarray, prediction: np.ndarray, baseline: np.ndarray, seed: int) -> dict[str, float]:
    scale = max(float(np.median(target)), 1e-8)
    error = lambda value: 0.5 * (
        np.abs(target - value) / scale
        + 2.0 * np.abs(target - value) / np.maximum(np.abs(target) + np.abs(value), 1e-8)
    )
    difference = error(prediction) - error(baseline)
    rng = np.random.default_rng(seed)
    draws = difference[rng.integers(0, len(target), size=(4000, len(target)))].mean(axis=1)
    return {
        "probability_improvement": float(np.mean(draws < 0.0)),
        "ci90_low": float(np.quantile(draws, 0.05)),
        "ci90_high": float(np.quantile(draws, 0.95)),
    }


def nested_audit(
    features: np.ndarray,
    targets: dict[str, np.ndarray],
    names: list[str],
    *,
    seed: int,
    outer_repeats: int,
    minimum_gain: float,
    minimum_selection_rate: float,
    minimum_bootstrap_probability: float,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    reports: dict[str, object] = {}
    models: dict[str, dict[str, object]] = {}
    accepted_targets: list[str] = []
    for target_index, target_name in enumerate(PATCHABLE_TARGETS):
        target = np.asarray(targets[target_name], dtype=np.float64)
        prediction = np.zeros_like(target)
        baseline = np.zeros_like(target)
        counts = np.zeros_like(target)
        selections = 0
        for repeat, fold, training, validation in _splits(len(target), 5, outer_repeats, seed + target_index * 10000):
            model, _ = _select(
                features[training],
                target[training],
                names,
                target_name,
                seed + target_index * 100000 + repeat * 100 + fold,
            )
            baseline[validation] += float(np.median(target[training]))
            if model is None:
                prediction[validation] += float(np.median(target[training]))
            else:
                prediction[validation] += _predict(model, features[validation])
                selections += 1
            counts[validation] += 1.0
        prediction /= counts
        baseline /= counts
        metrics = _metrics(target, prediction)
        baseline_metrics = _metrics(target, baseline)
        gain = 1.0 - metrics["composite"] / max(baseline_metrics["composite"], 1e-8)
        selection_rate = selections / (5 * outer_repeats)
        bootstrap = _bootstrap(target, prediction, baseline, seed + target_index * 1000)
        final_model, final_selection = _select(features, target, names, target_name, seed + 9000 + target_index)
        ffa_coefficient_fraction = 0.0
        if final_model is not None:
            used_names = [names[int(index)] for index in final_model["indices"]]
            magnitudes = np.abs(np.asarray(final_model["beta"], dtype=np.float64))
            ffa_coefficient_fraction = float(
                magnitudes[[name.startswith("ffa_") for name in used_names]].sum()
                / max(magnitudes.sum(), 1e-8)
            )
        accepted = (
            final_model is not None
            and str(final_model["group"]).startswith(("ffa", "hybrid"))
            and ffa_coefficient_fraction >= 0.35
            and gain >= minimum_gain
            and selection_rate >= minimum_selection_rate
            and bootstrap["probability_improvement"] >= minimum_bootstrap_probability
            and metrics["nmae"] < baseline_metrics["nmae"]
            and metrics["smape"] < baseline_metrics["smape"]
        )
        if accepted:
            accepted_targets.append(target_name)
            models[target_name] = final_model
        reports[target_name] = {
            "accepted_before_domain_audit": bool(accepted),
            "nested_metrics": metrics,
            "nested_baseline": baseline_metrics,
            "nested_relative_gain": float(gain),
            "selection_rate": float(selection_rate),
            "ffa_coefficient_fraction": ffa_coefficient_fraction,
            "bootstrap": bootstrap,
            "final_selection": final_selection,
        }
    return {
        "protocol": {
            "outer_folds": 5,
            "outer_repeats": outer_repeats,
            "inner_folds": 5,
            "inner_repeats": 3,
            "minimum_gain": minimum_gain,
            "minimum_selection_rate": minimum_selection_rate,
            "minimum_bootstrap_probability": minimum_bootstrap_probability,
        },
        "accepted_before_domain_audit": accepted_targets,
        "targets": reports,
    }, models


def _extract_split(
    data_root: Path,
    split: str,
    store_root: Path,
    prepared_root: Path,
) -> tuple[np.ndarray, list[str], list[str]]:
    ids = case_ids(data_root, split)
    store = _open_store(store_root)
    if store.split != split or store.list_cases() != ids:
        raise RuntimeError(f"Incomplete or mismatched {split} Task 2 store")
    cache = PreparedFFACache(prepared_root, split)
    vectors: list[np.ndarray] = []
    names: list[str] | None = None
    for index, case_id in enumerate(ids, 1):
        vector, current_names = extract_features(
            store.read_case(case_id),
            load_rgb_float(data_root / split / "images" / f"{case_id}.png"),
            load_roi(data_root / split / "masks" / f"{case_id}.png"),
            cache.read_case(case_id),
        )
        if names is None:
            names = current_names
        elif names != current_names:
            raise RuntimeError("V13 Task 3 feature schema changed between cases")
        vectors.append(vector)
        print(f"[{index:02d}/{len(ids):02d}] task3 features {split} {case_id}", flush=True)
    return np.stack(vectors), list(names or []), ids


def _training_targets(data_root: Path, ids: list[str]) -> dict[str, np.ndarray]:
    records = [read_biomarker_txt(data_root / "training" / "biomarker" / f"{case_id}.txt") for case_id in ids]
    return {key: np.asarray([record[key] for record in records], dtype=np.float64) for key in BIOMARKER_KEYS}


def _domain_audit(
    training: np.ndarray,
    validation: np.ndarray,
    models: dict[str, dict[str, object]],
    *,
    max_mean_shift: float,
    max_abs_z: float,
) -> tuple[dict[str, object], list[str]]:
    reports: dict[str, object] = {}
    accepted: list[str] = []
    for target, model in models.items():
        indices = np.asarray(model["indices"], dtype=int)
        train = training[:, indices]
        valid = validation[:, indices]
        empirical_std = np.maximum(train.std(axis=0), 1e-8)
        mean_shift = np.abs((valid.mean(axis=0) - train.mean(axis=0)) / empirical_std)
        z = np.abs((valid - np.asarray(model["mean"])) / np.asarray(model["std"]))
        passed = float(mean_shift.max()) <= max_mean_shift and float(z.max()) <= max_abs_z
        if passed:
            accepted.append(target)
        reports[target] = {
            "passed": bool(passed),
            "max_mean_shift_sd": float(mean_shift.max()),
            "max_abs_z": float(z.max()),
        }
    return {
        "limits": {"max_mean_shift_sd": max_mean_shift, "max_abs_z": max_abs_z},
        "targets": reports,
    }, accepted


def _write_task3(
    source: Path,
    output: Path,
    ids: list[str],
    predictions: dict[str, np.ndarray],
    accepted: list[str],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for row, case_id in enumerate(ids):
        values = read_biomarker_txt(source / f"{case_id}.txt")
        for target in accepted:
            values[target] = float(predictions[target][row])
        if "AVR" in accepted:
            geometric = math.sqrt(max(values["CRAE"] * values["CRVE"], 1e-8))
            values["CRAE"] = geometric * math.sqrt(values["AVR"])
            values["CRVE"] = geometric / math.sqrt(values["AVR"])
        text = "\n".join(f"{key} {float(values[key]):.6f}" for key in BIOMARKER_KEYS) + "\n"
        (output / f"{case_id}.txt").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.run_dir / "task3"
    report_path = output_dir / "audit.json"
    existing_output = output_dir / "validation"
    if report_path.exists() and existing_output.is_dir():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected = {f"{case_id}.txt" for case_id in case_ids(args.data_root, "validation")}
        if {path.name for path in existing_output.glob("*.txt")} != expected:
            raise RuntimeError("Existing V13 Task 3 output is incomplete")
        print(json.dumps({"reused_task3": str(report_path), "accepted_targets": report["accepted_targets"]}, indent=2))
        return report
    output_dir.mkdir(parents=True, exist_ok=True)
    training, names, training_ids = _extract_split(
        args.data_root, "training", args.training_store, args.prepared_root
    )
    validation, validation_names, validation_ids = _extract_split(
        args.data_root, "validation", args.validation_store, args.prepared_root
    )
    if names != validation_names:
        raise RuntimeError("Training and validation Task 3 schemas differ")
    np.savez_compressed(
        output_dir / "feature_cache.npz",
        training=training,
        validation=validation,
        feature_names=np.asarray(names),
        training_ids=np.asarray(training_ids),
        validation_ids=np.asarray(validation_ids),
    )
    audit, models = nested_audit(
        training,
        _training_targets(args.data_root, training_ids),
        names,
        seed=args.seed,
        outer_repeats=args.outer_repeats,
        minimum_gain=args.minimum_gain,
        minimum_selection_rate=args.minimum_selection_rate,
        minimum_bootstrap_probability=args.minimum_bootstrap_probability,
    )
    domain, accepted_domain = _domain_audit(
        training,
        validation,
        models,
        max_mean_shift=args.maximum_mean_shift,
        max_abs_z=args.maximum_abs_z,
    )
    accepted = [target for target in audit["accepted_before_domain_audit"] if target in accepted_domain]
    predictions = {target: _predict(models[target], validation) for target in accepted}
    output_task3 = output_dir / "validation"
    if output_task3.exists():
        raise FileExistsError(f"Refusing to overwrite V13 Task 3 output: {output_task3}")
    _write_task3(args.v8_task3_source, output_task3, validation_ids, predictions, accepted)
    report = {
        "version": 13,
        "strategy": "strict registered-FFA correction for V8 AVR and vein density only",
        "accepted_targets": accepted,
        "frozen_targets": sorted(set(PATCHABLE_TARGETS) - set(accepted)),
        "always_frozen_targets": ["artery_density", "artery_fractal_dimension", "vein_fractal_dimension"],
        "nested_audit": audit,
        "domain_audit": domain,
        "models": models,
        "feature_count": len(names),
        "output": str(output_task3),
    }
    atomic_json(report_path, report)
    print(json.dumps({key: report[key] for key in ("accepted_targets", "frozen_targets", "domain_audit", "output")}, indent=2))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and build V13 registered-FFA Task 3 corrections.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--training-store", type=Path, required=True)
    parser.add_argument("--validation-store", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--v8-task3-source", type=Path, required=True)
    parser.add_argument("--outer-repeats", type=int, default=10)
    parser.add_argument("--minimum-gain", type=float, default=0.12)
    parser.add_argument("--minimum-selection-rate", type=float, default=0.80)
    parser.add_argument("--minimum-bootstrap-probability", type=float, default=0.95)
    parser.add_argument("--maximum-mean-shift", type=float, default=1.0)
    parser.add_argument("--maximum-abs-z", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=77)
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
