from __future__ import annotations

import argparse
import itertools
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import ndimage
from scipy.special import expit, logit
from skimage.morphology import skeletonize

from .metrics import evaluate_store
from .store import ProbabilityStore


@dataclass(frozen=True)
class GraphParameters:
    seed_threshold: float = 0.50
    grow_threshold: float = 0.20
    min_component_size: int = 16
    junction_dilation: int = 1
    unary_scale: float = 1.0
    bifurcation_weight: float = 4.0
    straight_weight: float = 10.0
    tree_strength: float = 0.85
    selected_floor: float = 0.60
    unselected_ceiling: float = 0.40
    crossing_floor: float = 0.60
    vessel_floor: float = 0.55

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "GraphParameters":
        known = {field for field in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in payload.items() if key in known})


def high_recall_support(probability: np.ndarray, parameters: GraphParameters) -> np.ndarray:
    vessel = np.asarray(probability, dtype=np.float32)[1]
    seeds = vessel >= parameters.seed_threshold
    weak = vessel >= parameters.grow_threshold
    support = ndimage.binary_propagation(seeds, structure=np.ones((3, 3), dtype=bool), mask=weak)
    labels, count = ndimage.label(support, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return support
    sizes = np.bincount(labels.ravel())
    keep = sizes >= parameters.min_component_size
    keep[0] = False
    return keep[labels]


def _segment_skeleton(support: np.ndarray, parameters: GraphParameters):
    skeleton = skeletonize(support)
    degree = ndimage.convolve(skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant")
    degree = degree - skeleton.astype(np.uint8)
    junction = skeleton & (degree >= 3)
    junction_zone = ndimage.binary_dilation(junction, iterations=parameters.junction_dilation)
    segments, count = ndimage.label(skeleton & ~junction_zone, structure=np.ones((3, 3), dtype=np.uint8))
    return skeleton, junction_zone, segments, count


def _segment_direction(segment_mask: np.ndarray, center: np.ndarray, radius: float = 18.0) -> np.ndarray | None:
    coordinates = np.argwhere(segment_mask)
    if len(coordinates) == 0:
        return None
    distances = np.linalg.norm(coordinates - center[None, :], axis=1)
    local = coordinates[distances <= radius]
    if len(local) < 2:
        local = coordinates[np.argsort(distances)[: min(12, len(coordinates))]]
    vector = local.mean(axis=0) - center
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-6 else None


def _junction_relations(
    segment_labels: np.ndarray,
    junction_zone: np.ndarray,
    parameters: GraphParameters,
) -> tuple[dict[tuple[int, int], float], np.ndarray]:
    relations: dict[tuple[int, int], float] = {}
    crossing_zone = np.zeros_like(junction_zone, dtype=bool)
    junction_labels, junction_count = ndimage.label(junction_zone, structure=np.ones((3, 3), dtype=np.uint8))
    for junction_index in range(1, junction_count + 1):
        region = junction_labels == junction_index
        adjacent_zone = ndimage.binary_dilation(region, iterations=1)
        adjacent = sorted(int(value) for value in np.unique(segment_labels[adjacent_zone]) if value > 0)
        if len(adjacent) < 2:
            continue
        center = np.argwhere(region).mean(axis=0)
        directions = {
            segment: _segment_direction(segment_labels == segment, center)
            for segment in adjacent
        }

        def add_relation(first: int, second: int, weight: float) -> None:
            key = tuple(sorted((first, second)))
            relations[key] = max(relations.get(key, 0.0), float(weight))

        if len(adjacent) <= 3:
            for first, second in itertools.combinations(adjacent, 2):
                add_relation(first, second, parameters.bifurcation_weight)
            continue

        candidates = []
        for first, second in itertools.combinations(adjacent, 2):
            first_direction = directions[first]
            second_direction = directions[second]
            if first_direction is None or second_direction is None:
                cosine = 1.0
            else:
                cosine = float(np.dot(first_direction, second_direction))
            candidates.append((cosine, first, second))
        used: set[int] = set()
        pairs = []
        for _, first, second in sorted(candidates):
            if first in used or second in used:
                continue
            used.update((first, second))
            pairs.append((first, second))
            add_relation(first, second, parameters.straight_weight)
        if len(pairs) >= 2:
            crossing_zone |= region
    return relations, crossing_zone


def _segment_unaries(
    probability: np.ndarray,
    segment_labels: np.ndarray,
    segment_count: int,
    unary_scale: float,
) -> dict[int, tuple[float, float]]:
    artery = np.clip(probability[0], 1e-4, 1.0 - 1e-4)
    vein = np.clip(probability[2], 1e-4, 1.0 - 1e-4)
    evidence = logit(artery) - logit(vein)
    unaries = {}
    for segment in range(1, segment_count + 1):
        values = evidence[segment_labels == segment]
        if len(values) == 0:
            continue
        pooled = float(np.median(values))
        artery_probability = float(expit(np.clip(pooled, -8.0, 8.0)))
        effective_length = float(np.sqrt(min(len(values), 256)))
        scale = unary_scale * max(effective_length, 1.0)
        artery_cost = -np.log(max(artery_probability, 1e-6)) * scale
        vein_cost = -np.log(max(1.0 - artery_probability, 1e-6)) * scale
        unaries[segment] = (float(artery_cost), float(vein_cost))
    return unaries


def _solve_labels(
    unaries: dict[int, tuple[float, float]],
    relations: dict[tuple[int, int], float],
) -> dict[int, int]:
    source = "artery_source"
    sink = "vein_sink"
    graph = nx.DiGraph()
    for segment, (artery_cost, vein_cost) in unaries.items():
        graph.add_edge(source, segment, capacity=vein_cost)
        graph.add_edge(segment, sink, capacity=artery_cost)
    for (first, second), weight in relations.items():
        if first not in unaries or second not in unaries:
            continue
        graph.add_edge(first, second, capacity=weight)
        graph.add_edge(second, first, capacity=weight)
    _, partition = nx.minimum_cut(graph, source, sink, capacity="capacity")
    artery_partition, _ = partition
    return {segment: (0 if segment in artery_partition else 2) for segment in unaries}


def graph_refine_probability(
    probability: np.ndarray,
    parameters: GraphParameters,
) -> tuple[np.ndarray, dict[str, object]]:
    raw = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    if raw.ndim != 3 or raw.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W], got {raw.shape}")
    support = high_recall_support(raw, parameters)
    skeleton, junction_zone, segments, segment_count = _segment_skeleton(support, parameters)
    if segment_count == 0:
        return raw.copy(), {"segments": 0, "support_pixels": int(support.sum()), "crossing_pixels": 0}
    relations, possible_crossings = _junction_relations(segments, junction_zone, parameters)
    unaries = _segment_unaries(raw, segments, segment_count, parameters.unary_scale)
    labels = _solve_labels(unaries, relations)

    _, nearest = ndimage.distance_transform_edt(segments == 0, return_indices=True)
    nearest_segment = segments[nearest[0], nearest[1]]
    assignment = np.full(raw.shape[1:], -1, dtype=np.int8)
    for segment, channel in labels.items():
        assignment[(nearest_segment == segment) & support] = channel
    artery_assigned = assignment == 0
    vein_assigned = assignment == 2

    crossing = np.zeros_like(support, dtype=bool)
    possible_labels, number = ndimage.label(possible_crossings, structure=np.ones((3, 3), dtype=np.uint8))
    for component in range(1, number + 1):
        region = possible_labels == component
        neighborhood = ndimage.binary_dilation(region, iterations=2)
        if artery_assigned[neighborhood].any() and vein_assigned[neighborhood].any():
            crossing |= region

    artery_target = np.where(artery_assigned, np.maximum(raw[0], parameters.selected_floor), np.minimum(raw[0], parameters.unselected_ceiling))
    vein_target = np.where(vein_assigned, np.maximum(raw[2], parameters.selected_floor), np.minimum(raw[2], parameters.unselected_ceiling))
    artery = raw[0].copy()
    vein = raw[2].copy()
    artery[support] = (
        (1.0 - parameters.tree_strength) * raw[0][support]
        + parameters.tree_strength * artery_target[support]
    )
    vein[support] = (
        (1.0 - parameters.tree_strength) * raw[2][support]
        + parameters.tree_strength * vein_target[support]
    )
    artery[artery_assigned] = np.maximum(artery[artery_assigned], 0.5005)
    vein[artery_assigned] = np.minimum(vein[artery_assigned], 0.4995)
    vein[vein_assigned] = np.maximum(vein[vein_assigned], 0.5005)
    artery[vein_assigned] = np.minimum(artery[vein_assigned], 0.4995)
    artery[crossing] = np.maximum(artery[crossing], parameters.crossing_floor)
    vein[crossing] = np.maximum(vein[crossing], parameters.crossing_floor)
    vessel = np.maximum.reduce((raw[1], artery, vein, support.astype(np.float32) * parameters.vessel_floor))
    refined = np.clip(np.stack((artery, vessel, vein), axis=0), 0.0, 1.0).astype(np.float32)
    diagnostics = {
        "segments": int(segment_count),
        "relations": len(relations),
        "support_pixels": int(support.sum()),
        "skeleton_pixels": int(skeleton.sum()),
        "artery_pixels": int(artery_assigned.sum()),
        "vein_pixels": int(vein_assigned.sum()),
        "crossing_pixels": int(crossing.sum()),
    }
    return refined, diagnostics


def apply_graph_store(
    input_store: ProbabilityStore,
    output_store: ProbabilityStore,
    parameters: GraphParameters,
    *,
    revision: str,
) -> dict[str, object]:
    new_cases = 0
    diagnostics = {}
    settings = {"revision": revision, "parameters": asdict(parameters)}
    for case_id in input_store.list_cases():
        provenance = {
            **settings,
            "input_sha256": input_store.case_record(case_id)["sha256"],
        }
        if output_store.is_complete(case_id, provenance):
            continue
        refined, case_diagnostics = graph_refine_probability(input_store.read_case(case_id), parameters)
        output_store.write_case(case_id, refined, provenance)
        diagnostics[case_id] = case_diagnostics
        new_cases += 1
    return {"cases": len(input_store.list_cases()), "new_cases": new_cases, "diagnostics": diagnostics}


def _candidate_parameters() -> list[GraphParameters]:
    candidates = []
    for grow, strength, unary in itertools.product((0.15, 0.20), (0.75, 0.90), (0.35, 0.75)):
        candidates.append(
            GraphParameters(
                grow_threshold=grow,
                tree_strength=strength,
                unary_scale=unary,
                bifurcation_weight=4.0,
                straight_weight=10.0,
            )
        )
    return candidates


def search_graph(
    data_root: Path,
    input_store: ProbabilityStore,
    *,
    work_dir: Path,
    paths_per_case: int,
    seed: int,
    minimum_gain: float,
    maximum_dice_drop: float,
    maximum_sensitivity_drop: float,
) -> dict[str, object]:
    work_dir.mkdir(parents=True, exist_ok=True)
    direct_report = evaluate_store(
        data_root,
        input_store,
        paths_per_case=paths_per_case,
        seed=seed,
    )
    candidates = []
    with tempfile.TemporaryDirectory(prefix="graph-search-", dir=work_dir) as temporary_root:
        for index, parameters in enumerate(_candidate_parameters()):
            candidate_store = ProbabilityStore(
                Path(temporary_root) / f"candidate_{index:02d}",
                namespace="r2v2_graph_candidate",
                split="training",
            )
            apply_graph_store(input_store, candidate_store, parameters, revision="r2v2_graph_search_v1")
            report = evaluate_store(
                data_root,
                candidate_store,
                paths_per_case=paths_per_case,
                seed=seed,
            )
            candidates.append({"index": index, "parameters": asdict(parameters), "report": report})
    best = max(candidates, key=lambda item: float(item["report"]["score_observed"]))
    gain = float(best["report"]["score_observed"]) - float(direct_report["score_observed"])
    dice_drop = float(direct_report["mean"]["dice"]) - float(best["report"]["mean"]["dice"])
    sensitivity_drop = float(direct_report["mean"]["sensitivity"]) - float(best["report"]["mean"]["sensitivity"])
    accepted = (
        gain >= minimum_gain
        and dice_drop <= maximum_dice_drop
        and sensitivity_drop <= maximum_sensitivity_drop
    )
    return {
        "revision": "r2v2_graph_selection_v1",
        "accepted": bool(accepted),
        "direct_report": direct_report,
        "selected": best,
        "gain": gain,
        "dice_drop": dice_drop,
        "sensitivity_drop": sensitivity_drop,
        "gates": {
            "minimum_gain": minimum_gain,
            "maximum_dice_drop": maximum_dice_drop,
            "maximum_sensitivity_drop": maximum_sensitivity_drop,
        },
        "candidate_summaries": [
            {
                "index": item["index"],
                "parameters": item["parameters"],
                "score_observed": item["report"]["score_observed"],
                "dice": item["report"]["mean"]["dice"],
                "sensitivity": item["report"]["mean"]["sensitivity"],
                "cor": item["report"]["mean"]["cor"],
                "inf": item["report"]["mean"]["inf"],
            }
            for item in candidates
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crossing-aware graph projection with a direct-output safety gate.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    search = subparsers.add_parser("search")
    search.add_argument("--data-root", type=Path, required=True)
    search.add_argument("--input-store", type=Path, required=True)
    search.add_argument("--work-dir", type=Path, required=True)
    search.add_argument("--output", type=Path, required=True)
    search.add_argument("--paths-per-case", type=int, default=100)
    search.add_argument("--seed", type=int, default=77)
    search.add_argument("--minimum-gain", type=float, default=0.10)
    search.add_argument("--maximum-dice-drop", type=float, default=0.03)
    search.add_argument("--maximum-sensitivity-drop", type=float, default=0.02)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--input-store", type=Path, required=True)
    apply_parser.add_argument("--output-store", type=Path, required=True)
    apply_parser.add_argument("--selection", type=Path, required=True)
    apply_parser.add_argument("--split", choices=("training", "validation"), required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.command == "search":
        input_store = ProbabilityStore(args.input_store, namespace="r2v2_direct", split="training")
        report = search_graph(
            args.data_root,
            input_store,
            work_dir=args.work_dir,
            paths_per_case=args.paths_per_case,
            seed=args.seed,
            minimum_gain=args.minimum_gain,
            maximum_dice_drop=args.maximum_dice_drop,
            maximum_sensitivity_drop=args.maximum_sensitivity_drop,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "direct_report"}, indent=2))
        return

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    input_store = ProbabilityStore(args.input_store, namespace="r2v2_direct", split=args.split)
    output_store = ProbabilityStore(args.output_store, namespace="r2v2_graph", split=args.split)
    if selection.get("accepted"):
        parameters = GraphParameters.from_dict(selection["selected"]["parameters"])
        result = apply_graph_store(input_store, output_store, parameters, revision="r2v2_graph_selected_v1")
        result["graph_accepted"] = True
    else:
        new_cases = 0
        for case_id in input_store.list_cases():
            provenance = {
                "revision": "r2v2_graph_rejected_direct_copy_v1",
                "selection_revision": str(selection.get("revision", "unknown")),
                "input_sha256": input_store.case_record(case_id)["sha256"],
            }
            if output_store.is_complete(case_id, provenance):
                continue
            output_store.write_case(case_id, input_store.read_case(case_id), provenance)
            new_cases += 1
        result = {
            "cases": len(input_store.list_cases()),
            "new_cases": new_cases,
            "graph_accepted": False,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
