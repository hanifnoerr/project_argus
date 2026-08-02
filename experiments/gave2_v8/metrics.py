from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize

from experiments.gave2_ensemble.data import derive_av3_target

from .store import ProbabilityStore


NEIGHBORS = ((1, 0, 1.0), (0, 1, 1.0), (1, 1, 2.0**0.5), (1, -1, 2.0**0.5))


@dataclass(frozen=True)
class PathCounts:
    correct: int = 0
    infeasible: int = 0
    incorrect: int = 0

    @property
    def total(self) -> int:
        return self.correct + self.infeasible + self.incorrect

    def __add__(self, other: "PathCounts") -> "PathCounts":
        return PathCounts(
            self.correct + other.correct,
            self.infeasible + other.infeasible,
            self.incorrect + other.incorrect,
        )


def _pixel_graph(skeleton: np.ndarray) -> tuple[np.ndarray, np.ndarray, sparse.csr_matrix]:
    binary = np.asarray(skeleton, dtype=bool)
    coordinates = np.argwhere(binary)
    index_map = np.full(binary.shape, -1, dtype=np.int32)
    if coordinates.size == 0:
        return coordinates, index_map, sparse.csr_matrix((0, 0), dtype=np.float32)
    index_map[coordinates[:, 0], coordinates[:, 1]] = np.arange(len(coordinates), dtype=np.int32)
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    height, width = binary.shape
    for delta_y, delta_x, weight in NEIGHBORS:
        source_y0 = max(0, -delta_y)
        source_y1 = min(height, height - delta_y)
        source_x0 = max(0, -delta_x)
        source_x1 = min(width, width - delta_x)
        source = binary[source_y0:source_y1, source_x0:source_x1]
        target = binary[
            source_y0 + delta_y : source_y1 + delta_y,
            source_x0 + delta_x : source_x1 + delta_x,
        ]
        ys, xs = np.nonzero(source & target)
        if ys.size == 0:
            continue
        source_indices = index_map[ys + source_y0, xs + source_x0]
        target_indices = index_map[ys + source_y0 + delta_y, xs + source_x0 + delta_x]
        rows.extend((source_indices, target_indices))
        columns.extend((target_indices, source_indices))
        edge_values = np.full(source_indices.shape, weight, dtype=np.float32)
        values.extend((edge_values, edge_values))
    if not rows:
        return coordinates, index_map, sparse.csr_matrix((len(coordinates), len(coordinates)), dtype=np.float32)
    graph = sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(len(coordinates), len(coordinates)),
        dtype=np.float32,
    ).tocsr()
    return coordinates, index_map, graph


def _stable_rng(seed: int, case_id: str, channel: int):
    material = f"{seed}:{case_id}:{channel}".encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.random.default_rng(derived)


def _sample_pairs(
    target_skeleton: np.ndarray,
    count: int,
    rng: np.random.Generator,
    minimum_component_size: int = 8,
) -> np.ndarray:
    labels, number = ndimage.label(target_skeleton, structure=np.ones((3, 3), dtype=np.uint8))
    components = []
    weights = []
    for component in range(1, number + 1):
        coordinates = np.argwhere(labels == component)
        if len(coordinates) >= minimum_component_size:
            components.append(coordinates)
            weights.append(float(len(coordinates)))
    if not components:
        return np.empty((0, 2, 2), dtype=np.int32)
    probabilities = np.asarray(weights, dtype=np.float64)
    probabilities /= probabilities.sum()
    pairs = []
    attempts = 0
    while len(pairs) < count and attempts < count * 20:
        attempts += 1
        coordinates = components[int(rng.choice(len(components), p=probabilities))]
        selected = rng.choice(len(coordinates), size=2, replace=False)
        pair = coordinates[selected]
        if float(np.linalg.norm(pair[0] - pair[1])) < 5.0:
            continue
        pairs.append(pair)
    return np.asarray(pairs, dtype=np.int32)


def _pair_distances(graph: sparse.csr_matrix, sources: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if len(sources) == 0:
        return np.empty(0, dtype=np.float32)
    unique_sources, inverse = np.unique(sources, return_inverse=True)
    distances = dijkstra(graph, directed=False, indices=unique_sources, return_predecessors=False)
    if distances.ndim == 1:
        distances = distances[None, :]
    return np.asarray(distances[inverse, targets], dtype=np.float32)


def path_counts(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    paths: int = 100,
    tolerance: float = 0.10,
    seed: int = 77,
    case_id: str = "case",
    channel: int = 0,
) -> PathCounts:
    """Approximate the organizer's centerline nearest-point path protocol."""

    target_skeleton = skeletonize(np.asarray(target, dtype=bool))
    sampled = _sample_pairs(target_skeleton, paths, _stable_rng(seed, case_id, channel))
    if len(sampled) == 0:
        return PathCounts()
    target_coordinates, target_index, target_graph = _pixel_graph(target_skeleton)
    del target_coordinates
    target_sources = target_index[sampled[:, 0, 0], sampled[:, 0, 1]]
    target_targets = target_index[sampled[:, 1, 0], sampled[:, 1, 1]]
    target_lengths = _pair_distances(target_graph, target_sources, target_targets)

    prediction_mask = np.asarray(prediction, dtype=bool)
    if not prediction_mask.any():
        return PathCounts(infeasible=len(sampled))
    # The protocol maps centerline points to the nearest predicted mask point,
    # then measures the shortest route through the complete predicted mask.
    _, prediction_index, prediction_graph = _pixel_graph(prediction_mask)
    _, nearest = ndimage.distance_transform_edt(~prediction_mask, return_indices=True)
    first_y = nearest[0, sampled[:, 0, 0], sampled[:, 0, 1]]
    first_x = nearest[1, sampled[:, 0, 0], sampled[:, 0, 1]]
    second_y = nearest[0, sampled[:, 1, 0], sampled[:, 1, 1]]
    second_x = nearest[1, sampled[:, 1, 0], sampled[:, 1, 1]]
    prediction_sources = prediction_index[first_y, first_x]
    prediction_targets = prediction_index[second_y, second_x]

    component_labels, _ = ndimage.label(prediction_mask, structure=np.ones((3, 3), dtype=np.uint8))
    feasible = component_labels[first_y, first_x] == component_labels[second_y, second_x]
    prediction_lengths = np.full(len(sampled), np.inf, dtype=np.float32)
    if feasible.any():
        prediction_lengths[feasible] = _pair_distances(
            prediction_graph,
            prediction_sources[feasible],
            prediction_targets[feasible],
        )
    valid_target = np.isfinite(target_lengths) & (target_lengths > 0)
    feasible &= np.isfinite(prediction_lengths) & valid_target
    relative_error = np.full(len(sampled), np.inf, dtype=np.float32)
    relative_error[feasible] = np.abs(prediction_lengths[feasible] - target_lengths[feasible]) / target_lengths[feasible]
    correct = feasible & (relative_error <= tolerance)
    infeasible = ~np.isfinite(prediction_lengths)
    return PathCounts(
        correct=int(correct.sum()),
        infeasible=int(infeasible.sum()),
        incorrect=int(len(sampled) - correct.sum() - infeasible.sum()),
    )


def _score_components(
    dice: float,
    sensitivity: float,
    specificity: float,
    accuracy: float,
    correct: float,
    infeasible: float,
) -> dict[str, float]:
    classification = 0.3 * sensitivity + 0.3 * specificity + 0.4 * accuracy
    topology = 0.5 * correct + 0.5 * (1.0 - infeasible)
    return {
        "classification": classification,
        "topology": topology,
        "score_published": 10.0 * (0.3 * classification + 0.4 * dice + 0.3 * topology),
        "score_observed": 10.0 * (0.4 * classification + 0.2 * dice + 0.4 * topology),
        "score_gave2025_paper": 10.0 * (0.3 * classification + 0.3 * dice + 0.4 * topology),
    }


def evaluate_store(
    data_root: Path,
    store: ProbabilityStore,
    *,
    threshold: float = 0.5,
    paths_per_case: int = 100,
    seed: int = 77,
) -> dict[str, object]:
    totals = {
        0: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "paths": PathCounts()},
        2: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "paths": PathCounts()},
    }
    case_reports = []
    for case_id in store.list_cases():
        probability = store.read_case(case_id)
        raw_target = np.asarray(Image.open(data_root / "training" / "av" / f"{case_id}.png").convert("RGB"), dtype=np.float32) / 255.0
        target = derive_av3_target(raw_target) > 0.5
        roi = np.asarray(Image.open(data_root / "training" / "masks" / f"{case_id}.png").convert("L")) > 127
        case_report = {"case_id": case_id, "channels": {}}
        for channel, name in ((0, "artery"), (2, "vein")):
            prediction = (probability[channel] >= threshold) & roi
            truth = target[channel] & roi
            tp = int((prediction & truth).sum())
            fp = int((prediction & ~truth & roi).sum())
            fn = int((~prediction & truth).sum())
            tn = int((~prediction & ~truth & roi).sum())
            counts = path_counts(
                prediction,
                truth,
                paths=paths_per_case,
                seed=seed,
                case_id=case_id,
                channel=channel,
            )
            for key, value in (("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn)):
                totals[channel][key] += value
            totals[channel]["paths"] = totals[channel]["paths"] + counts
            case_report["channels"][name] = {
                "dice": 2.0 * tp / max(2 * tp + fp + fn, 1),
                "correct_paths": counts.correct,
                "infeasible_paths": counts.infeasible,
                "incorrect_paths": counts.incorrect,
            }
        case_reports.append(case_report)

    channel_reports = {}
    for channel, name in ((0, "artery"), (2, "vein")):
        values = totals[channel]
        tp, fp, fn, tn = (int(values[key]) for key in ("tp", "fp", "fn", "tn"))
        counts = values["paths"]
        channel_reports[name] = {
            "dice": 2.0 * tp / max(2 * tp + fp + fn, 1),
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
            "cor": counts.correct / max(counts.total, 1),
            "inf": counts.infeasible / max(counts.total, 1),
            "paths": counts.total,
        }
    means = {
        metric: float(np.mean([channel_reports[name][metric] for name in ("artery", "vein")]))
        for metric in ("dice", "sensitivity", "specificity", "accuracy", "cor", "inf")
    }
    scores = _score_components(
        means["dice"],
        means["sensitivity"],
        means["specificity"],
        means["accuracy"],
        means["cor"],
        means["inf"],
    )
    return {
        "threshold": float(threshold),
        "paths_per_case": int(paths_per_case),
        "seed": int(seed),
        "cases": len(case_reports),
        "channels": channel_reports,
        "mean": means,
        **scores,
        "case_reports": case_reports,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GAVE2 probabilities with sampled shortest-path metrics.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--namespace", default="r2v2_direct")
    parser.add_argument("--split", default="training")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--paths-per-case", type=int, default=100)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    store = ProbabilityStore(args.store, namespace=args.namespace, split=args.split)
    report = evaluate_store(
        args.data_root,
        store,
        threshold=args.threshold,
        paths_per_case=args.paths_per_case,
        seed=args.seed,
    )
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
