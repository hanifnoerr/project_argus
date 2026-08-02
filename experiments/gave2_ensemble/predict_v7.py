from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .data import list_case_ids
from .data_v6 import GAVE2DatasetV6
from .predict_v6 import (
    FloatProbabilityStore,
    _expected_case_record,
    _import_torch,
    infer_case_probability,
    oof_case_ownership,
    validate_prediction_manifest,
)
from .training_state_v6 import hash_file_bytes


V7_PREDICTION_REVISION = "conditional_av_ensemble_repair_v2"


def repair_ensemble_av_overlap(probability: np.ndarray, decision_threshold: float = 0.5) -> np.ndarray:
    """Restore A/V exclusivity after averaging independently exclusive predictions.

    Fold or TTA disagreement can make both averaged classes cross 0.5. Only
    conflicting pixels are changed, and the weaker class is capped far enough
    below 0.5 to remain below it after the float16 probability-store round trip.
    """

    value = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0).copy()
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W] probability, got {value.shape}")
    threshold = float(decision_threshold)
    overlap = (value[0] >= threshold) & (value[2] >= threshold)
    if not bool(overlap.any()):
        return np.ascontiguousarray(value)
    artery_wins = value[0] >= value[2]
    safe_loser = float(np.float16(threshold - 1e-3))
    suppress_vein = overlap & artery_wins
    suppress_artery = overlap & ~artery_wins
    value[2, suppress_vein] = np.minimum(value[2, suppress_vein], safe_loser)
    value[0, suppress_artery] = np.minimum(value[0, suppress_artery], safe_loser)
    value[0] = np.minimum(value[0], value[1])
    value[2] = np.minimum(value[2], value[1])
    return np.ascontiguousarray(value, dtype=np.float32)


def load_certified_folds_v7(run_dir: Path, task: str, fold_manifest: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(Path(fold_manifest).read_text(encoding="utf-8"))
    validate_prediction_manifest(manifest)
    manifest_hash = hash_file_bytes(fold_manifest)
    records = []
    for fold_index in range(3):
        fold_dir = Path(run_dir) / "cmrrwnet_v7" / task / f"fold_{fold_index}"
        paths = {
            "checkpoint_path": fold_dir / "best.pt",
            "config_path": fold_dir / "config.json",
            "certified_path": fold_dir / "best.certified.json",
        }
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        config = json.loads(paths["config_path"].read_text(encoding="utf-8"))
        certified = json.loads(paths["certified_path"].read_text(encoding="utf-8"))
        fold = manifest["folds"][fold_index]
        if int(config.get("version", -1)) != 7 or config.get("output_semantics") != "vessel_sigmoid_plus_conditional_av_softmax":
            raise ValueError(f"Fold {fold_index} is not a V7 checkpoint")
        if config.get("fold_manifest_sha256") != manifest_hash:
            raise ValueError(f"Fold {fold_index} manifest hash mismatch")
        if sorted(config.get("train_ids", [])) != sorted(fold["training"]):
            raise ValueError(f"Fold {fold_index} training IDs mismatch")
        if sorted(config.get("val_ids", [])) != sorted(fold["validation"]):
            raise ValueError(f"Fold {fold_index} validation IDs mismatch")
        if certified.get("config_content_sha256") != config.get("config_content_sha256"):
            raise ValueError(f"Fold {fold_index} certification mismatch")
        records.append({"fold_index": fold_index, "config": config, "certified": certified, **paths})
    return manifest, records


def _load_model(record: dict, device: str):
    from .train_v7 import build_model_from_config

    torch = _import_torch()
    try:
        checkpoint = torch.load(record["checkpoint_path"], map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(record["checkpoint_path"], map_location=device)
    if checkpoint["config"]["config_content_sha256"] != record["config"]["config_content_sha256"]:
        raise RuntimeError("V7 checkpoint/config mismatch")
    model = build_model_from_config(record["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def _predictor(model, sample, normalized_image: np.ndarray, device: str, transform_name: str) -> np.ndarray:
    from .losses_v7 import conditional_probabilities_from_logits

    del sample, transform_name
    torch = _import_torch()
    batch = torch.from_numpy(np.ascontiguousarray(normalized_image)[None]).to(device)
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else nullcontext()
    with torch.inference_mode(), context:
        probability = conditional_probabilities_from_logits(model(batch)).float().cpu().numpy()[0]
    return np.ascontiguousarray(probability, dtype=np.float32)


def run_prediction(args: argparse.Namespace) -> dict[str, object]:
    torch = _import_torch()
    device = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and not torch.cuda.is_available():
        device = "cpu"
    manifest, records = load_certified_folds_v7(args.run_dir, args.task, args.fold_manifest)
    ownership = oof_case_ownership(manifest)
    manifest_hash = hash_file_bytes(args.fold_manifest)
    split = "training" if args.mode == "oof" else "validation"
    case_ids = sorted(ownership) if args.mode == "oof" else list_case_ids(args.data_root, split=split)
    if args.limit_cases:
        case_ids = case_ids[: args.limit_cases]
    store = FloatProbabilityStore(args.store_root, task=args.task, split=args.mode)
    expected = {}
    for case_id in case_ids:
        selected = [records[ownership[case_id]]] if args.mode == "oof" else records
        expected[case_id] = _expected_case_record(
            case_id,
            task=args.task,
            split=args.mode,
            fold_records=selected,
            manifest_hash=manifest_hash,
        )
        expected[case_id]["provenance"]["version"] = 7
        expected[case_id]["provenance"]["output_semantics"] = "conditional_av"
        # Cache revisions are explicit so a resumed run cannot silently reuse
        # probabilities written before an inference/postprocessing fix.
        expected[case_id]["provenance"]["prediction_revision"] = V7_PREDICTION_REVISION
    pending = set(store.pending_case_ids(expected))
    needed_folds = {ownership[case_id] for case_id in pending} if args.mode == "oof" else set(range(3))
    models = {index: _load_model(records[index], device) for index in sorted(needed_folds)}
    dataset = GAVE2DatasetV6(args.data_root, split=split, task=args.task, case_ids=case_ids, require_target=False)
    written = []
    for sample in dataset:
        if sample.case_id not in pending:
            continue
        selected = [records[ownership[sample.case_id]]] if args.mode == "oof" else records
        probability = infer_case_probability(sample, selected, models, device, predictor=_predictor)
        probability = repair_ensemble_av_overlap(probability)
        overlap = (probability[0] >= 0.5) & (probability[2] >= 0.5)
        if bool(overlap.any()):
            raise RuntimeError(f"Conditional V7 overlap invariant failed for {sample.case_id}")
        store.write_case(sample.case_id, probability, provenance=expected[sample.case_id]["provenance"])
        written.append(sample.case_id)
    return {
        "version": 7,
        "task": args.task,
        "mode": args.mode,
        "written_cases": written,
        "complete_cases": [case_id for case_id in case_ids if store.is_case_complete(case_id, expected[case_id])],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-fold full-resolution V7 inference.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--task", choices=("task1", "task2"), required=True)
    parser.add_argument("--mode", choices=("oof", "validation"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-cases", type=int)
    return parser.parse_args(argv)


def main() -> None:
    print(json.dumps(run_prediction(parse_args()), indent=2))


if __name__ == "__main__":
    main()
