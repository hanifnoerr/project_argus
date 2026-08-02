from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def validate_crossfit_provenance(
    case_ids: list[str],
    ownership: dict[str, int],
    apply_fold: int,
    provenance: dict[str, dict[str, object]],
) -> None:
    for case_id in case_ids:
        if int(ownership[case_id]) == int(apply_fold):
            raise ValueError(f"Training case {case_id} belongs to refiner apply fold {apply_fold}")
        record = provenance.get(case_id, {})
        trained_on = {str(value) for value in record.get("training_case_ids", [])}
        if case_id in trained_on:
            raise ValueError(f"Base prediction for {case_id} is in-sample")


def paired_bootstrap_interval(
    differences: np.ndarray,
    *,
    repeats: int = 5000,
    seed: int = 77,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Paired differences must be a finite vector with at least two cases")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(int(repeats), values.size))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def evaluate_refiner_gate(
    base: dict[str, object],
    refined: dict[str, object],
    *,
    repeats: int = 5000,
    seed: int = 77,
) -> dict[str, object]:
    base_scores = np.asarray(base["selection_score_cases"], dtype=np.float64)
    refined_scores = np.asarray(refined["selection_score_cases"], dtype=np.float64)
    if base_scores.shape != refined_scores.shape:
        raise ValueError("Base and refined case metrics must have identical ownership")
    lower, upper = paired_bootstrap_interval(refined_scores - base_scores, repeats=repeats, seed=seed)
    dice_gain = float(refined["dice_mean"]) - float(base["dice_mean"])
    artery_delta = float(refined["dice_channels"][0]) - float(base["dice_channels"][0])
    vein_delta = float(refined["dice_channels"][2]) - float(base["dice_channels"][2])
    topology_delta = float(refined["topology_mean"]) - float(base["topology_mean"])
    selection_gain = float(refined["selection_score"]) - float(base["selection_score"])
    checks = {
        "dice_gain_at_least_0_01": dice_gain >= 0.01,
        "artery_drop_within_0_005": artery_delta >= -0.005,
        "vein_drop_within_0_005": vein_delta >= -0.005,
        "topology_not_worse": topology_delta >= 0.0,
        "selection_score_improves": selection_gain > 0.0,
        "bootstrap_lower_not_negative": lower >= 0.0,
    }
    return {
        "accepted": bool(all(checks.values())),
        "checks": checks,
        "dice_gain": dice_gain,
        "artery_dice_delta": artery_delta,
        "vein_dice_delta": vein_delta,
        "topology_delta": topology_delta,
        "selection_score_gain": selection_gain,
        "bootstrap_95_ci": [lower, upper],
    }


def _torch_modules():
    try:
        import torch
        from torch import nn

        return torch, nn
    except ImportError as exc:  # pragma: no cover - Colab dependency
        raise RuntimeError("The residual refiner requires PyTorch") from exc


def create_residual_refiner(image_channels: int, base_channels: int = 8, residual_bound: float = 2.0):
    torch, nn = _torch_modules()

    class Block(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.GroupNorm(max(1, min(8, out_channels)), out_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.GroupNorm(max(1, min(8, out_channels)), out_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, value):
            return self.layers(value)

    class ResidualRefiner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            in_channels = int(image_channels) + 3
            self.enc1 = Block(in_channels, base_channels)
            self.enc2 = Block(base_channels, base_channels * 2)
            self.bottleneck = Block(base_channels * 2, base_channels * 4)
            self.dec2 = Block(base_channels * 6, base_channels * 2)
            self.dec1 = Block(base_channels * 3, base_channels)
            self.out = nn.Conv2d(base_channels, 3, 1)
            self.bound = float(residual_bound)

        def forward(self, image, base_probability):
            import torch.nn.functional as functional

            x1 = self.enc1(torch.cat((image, base_probability), dim=1))
            x2 = self.enc2(functional.max_pool2d(x1, 2))
            x3 = self.bottleneck(functional.max_pool2d(x2, 2))
            y2 = functional.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
            y2 = self.dec2(torch.cat((y2, x2), dim=1))
            y1 = functional.interpolate(y2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
            y1 = self.dec1(torch.cat((y1, x1), dim=1))
            residual = torch.tanh(self.out(y1)) * self.bound
            base_logits = torch.logit(base_probability.clamp(1e-5, 1.0 - 1e-5))
            return base_logits + residual

    return ResidualRefiner()


def _load_manifest(path: Path) -> dict[str, object]:
    from .predict_v6 import validate_prediction_manifest

    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_prediction_manifest(manifest)
    return manifest


def _sample_map(data_root: Path, split: str, task: str, case_ids: list[str]):
    from .data_v6 import GAVE2DatasetV6

    dataset = GAVE2DatasetV6(data_root, split=split, task=task, case_ids=case_ids, require_target=split == "training")
    return {sample.case_id: sample for sample in dataset}


def _refiner_loss(torch, logits, target, roi):
    import torch.nn.functional as functional

    mask = roi.expand_as(target)
    bce = functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * mask).sum() / mask.sum().clamp_min(1.0)
    probability = torch.sigmoid(logits) * mask
    dims = (0, 2, 3)
    dice = (2.0 * (probability * target).sum(dims) + 1e-6) / (
        probability.sum(dims) + target.sum(dims) + 1e-6
    )
    return bce + (1.0 - dice.mean())


def _tensor_case(torch, sample, base_probability: np.ndarray, device: str):
    image = torch.from_numpy(np.ascontiguousarray(sample.image)[None]).to(device=device, dtype=torch.float32)
    base = torch.from_numpy(np.ascontiguousarray(base_probability)[None]).to(device=device, dtype=torch.float32)
    roi = torch.from_numpy(np.ascontiguousarray(sample.mask)[None]).to(device=device, dtype=torch.float32)
    target = None
    if sample.target is not None:
        target = torch.from_numpy(np.ascontiguousarray(sample.target)[None]).to(device=device, dtype=torch.float32)
    return image, base, roi, target


def _evaluate_refiner(torch, model, samples, store, case_ids: list[str], device: str) -> float:
    from .metrics_v6 import challenge_selection_score

    probabilities = []
    targets = []
    masks = []
    model.eval()
    with torch.inference_mode():
        for case_id in case_ids:
            sample = samples[case_id]
            image, base, roi, target = _tensor_case(torch, sample, store.read_case(case_id), device)
            probability = torch.sigmoid(model(image, base)) * roi
            probabilities.append(probability.cpu().numpy()[0])
            targets.append(target.cpu().numpy()[0])
            masks.append(roi.cpu().numpy()[0])
    metrics = challenge_selection_score(
        np.stack(probabilities), np.stack(targets), np.stack(masks), threshold=0.5
    )
    return float(metrics["selection_score"])


def train_crossfit_refiners(
    data_root: Path,
    base_store_root: Path,
    fold_manifest: Path,
    output_dir: Path,
    *,
    task: str,
    epochs: int = 20,
    base_channels: int = 8,
    learning_rate: float = 2e-4,
    patience: int = 5,
    seed: int = 77,
    device: str = "auto",
) -> dict[str, object]:
    from .predict_v6 import FloatProbabilityStore, oof_case_ownership
    from .training_state_v6 import atomic_torch_save, atomic_write_json, capture_rng_state, restore_rng_state

    completed_report = output_dir / "training_report.json"
    if completed_report.is_file() and all((output_dir / f"fold_{fold}" / "best.pt").is_file() for fold in range(3)):
        return json.loads(completed_report.read_text(encoding="utf-8"))

    torch, _ = _torch_modules()
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    manifest = _load_manifest(fold_manifest)
    ownership = oof_case_ownership(manifest)
    all_case_ids = sorted(ownership)
    samples = _sample_map(data_root, "training", task, all_case_ids)
    base_store = FloatProbabilityStore(base_store_root, task=task, split="oof")
    image_channels = 4 if task == "task1" else 6
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"version": 6, "task": task, "folds": []}

    for fold_index, fold in enumerate(manifest["folds"]):
        training_ids = [str(case_id) for case_id in fold["training"]]
        validation_ids = [str(case_id) for case_id in fold["validation"]]
        for case_id in training_ids + validation_ids:
            metadata = json.loads(base_store.case_metadata_path(case_id).read_text(encoding="utf-8"))
            predicted_by = [int(value) for value in metadata.get("provenance", {}).get("fold_indices", [])]
            if predicted_by != [ownership[case_id]] or case_id in manifest["folds"][ownership[case_id]]["training"]:
                raise RuntimeError(f"Base OOF provenance is not honest for {case_id}")

        fold_dir = output_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "version": 6,
            "task": task,
            "fold_index": fold_index,
            "training_ids": training_ids,
            "validation_ids": validation_ids,
            "image_channels": image_channels,
            "base_channels": base_channels,
            "residual_bound": 2.0,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "seed": seed,
        }
        config_path = fold_dir / "config.json"
        if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise RuntimeError(f"Refiner configuration mismatch: {fold_dir}")
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        completed_path = fold_dir / "completed.json"
        if completed_path.is_file() and (fold_dir / "best.pt").is_file():
            report["folds"].append(json.loads(completed_path.read_text(encoding="utf-8")))
            continue

        random.seed(seed + fold_index)
        np.random.seed(seed + fold_index)
        torch.manual_seed(seed + fold_index)
        model = create_residual_refiner(image_channels, base_channels=base_channels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        best_score = -float("inf")
        stale = 0
        history = []
        start_epoch = 1
        last_path = fold_dir / "last.pt"
        if last_path.is_file():
            checkpoint = torch.load(last_path, map_location=device, weights_only=False)
            if checkpoint.get("config") != config:
                raise RuntimeError(f"Refiner resume configuration mismatch: {fold_dir}")
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            best_score = float(checkpoint["best_score"])
            stale = int(checkpoint["stale"])
            history = list(checkpoint["history"])
            restore_rng_state(checkpoint["rng_state"])
            start_epoch = int(checkpoint["epoch"]) + 1
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            order = list(training_ids)
            random.shuffle(order)
            losses = []
            for case_id in order:
                sample = samples[case_id]
                image, base, roi, target = _tensor_case(torch, sample, base_store.read_case(case_id), device)
                optimizer.zero_grad(set_to_none=True)
                context = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else None
                if context is None:
                    logits = model(image, base)
                    loss = _refiner_loss(torch, logits, target, roi)
                else:
                    with context:
                        logits = model(image, base)
                        loss = _refiner_loss(torch, logits, target, roi)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            score = _evaluate_refiner(torch, model, samples, base_store, validation_ids, device)
            row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "selection_score": score}
            history.append(row)
            print(json.dumps({"fold": fold_index, **row}), flush=True)
            if score > best_score + 1e-4:
                best_score = score
                stale = 0
                atomic_torch_save(
                    fold_dir / "best.pt",
                    {"state_dict": model.state_dict(), "config": config, "best_score": best_score, "epoch": epoch},
                )
            else:
                stale += 1
            atomic_write_json(fold_dir / "history.json", history)
            atomic_torch_save(
                last_path,
                {
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "best_score": best_score,
                    "stale": stale,
                    "history": history,
                    "epoch": epoch,
                    "rng_state": capture_rng_state(),
                },
                previous_path=fold_dir / "last.prev.pt",
            )
            if epoch >= 8 and stale >= patience:
                break
        completed = {"fold_index": fold_index, "best_score": best_score, "epochs": len(history)}
        atomic_write_json(completed_path, completed)
        report["folds"].append(completed)
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _load_refiner_models(refiner_dir: Path, task: str, device: str):
    torch, _ = _torch_modules()
    models = {}
    for fold_index in range(3):
        checkpoint_path = refiner_dir / f"fold_{fold_index}" / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        if config["task"] != task or int(config["fold_index"]) != fold_index:
            raise RuntimeError(f"Invalid refiner checkpoint {checkpoint_path}")
        model = create_residual_refiner(
            int(config["image_channels"]), base_channels=int(config["base_channels"]), residual_bound=float(config["residual_bound"])
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        models[fold_index] = model
    return models


def predict_refined_store(
    data_root: Path,
    base_store_root: Path,
    fold_manifest: Path,
    refiner_dir: Path,
    output_store_root: Path,
    *,
    task: str,
    mode: str,
    device: str = "auto",
) -> Path:
    from .data import list_case_ids
    from .predict_v6 import (
        FOUR_TTA_TRANSFORMS,
        FloatProbabilityStore,
        apply_tta_transform,
        average_fold_probabilities,
        average_tta_probabilities,
        invert_tta_transform,
        oof_case_ownership,
    )

    torch, _ = _torch_modules()
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    manifest = _load_manifest(fold_manifest)
    ownership = oof_case_ownership(manifest)
    split = "training" if mode == "oof" else "validation"
    case_ids = sorted(ownership) if mode == "oof" else list_case_ids(data_root, split="validation")
    samples = _sample_map(data_root, split, task, case_ids)
    base_store = FloatProbabilityStore(base_store_root, task=task, split=mode)
    output_store = FloatProbabilityStore(output_store_root, task=task, split=mode)
    models = _load_refiner_models(refiner_dir, task, device)

    from .training_state_v6 import hash_file_bytes
    checkpoint_hashes = {
        str(fold_index): hash_file_bytes(refiner_dir / f"fold_{fold_index}" / "best.pt")
        for fold_index in range(3)
    }
    base_manifest_hash = hash_file_bytes(base_store.manifest_path)
    for case_id in case_ids:
        sample = samples[case_id]
        selected = [ownership[case_id]] if mode == "oof" else [0, 1, 2]
        provenance = {
            "refiner_fold_indices": selected,
            "refiner_checkpoint_sha256": {str(index): checkpoint_hashes[str(index)] for index in selected},
            "base_store": str(base_store_root),
            "base_manifest_sha256": base_manifest_hash,
            "cross_fitted": mode == "oof",
        }
        expected = {"case_id": case_id, "task": task, "split": mode, "provenance": provenance}
        if output_store.is_case_complete(case_id, expected=expected):
            continue
        fold_probabilities = []
        for fold_index in selected:
            tta_probabilities = []
            for transform in FOUR_TTA_TRANSFORMS:
                name = transform["name"]
                image = apply_tta_transform(sample.image, name)
                base = apply_tta_transform(base_store.read_case(case_id), name)
                image_tensor = torch.from_numpy(image[None]).to(device=device, dtype=torch.float32)
                base_tensor = torch.from_numpy(base[None]).to(device=device, dtype=torch.float32)
                with torch.inference_mode():
                    probability = torch.sigmoid(models[fold_index](image_tensor, base_tensor)).float().cpu().numpy()[0]
                tta_probabilities.append(invert_tta_transform(probability, name))
            fold_probabilities.append(average_tta_probabilities(tta_probabilities))
        probability = (
            fold_probabilities[0]
            if len(fold_probabilities) == 1
            else average_fold_probabilities(fold_probabilities, expected_folds=3)
        )
        output_store.write_case(
            case_id,
            probability * sample.mask,
            provenance=provenance,
        )
    return output_store_root


def compare_prediction_stores(
    data_root: Path,
    base_store_root: Path,
    refined_store_root: Path,
    base_calibration_path: Path,
    refined_calibration_path: Path,
    *,
    task: str,
) -> dict[str, object]:
    from .calibration_v6 import apply_calibration
    from .data import derive_av3_target, list_case_ids, read_png_float
    from .metrics_v6 import challenge_selection_score
    from .predict_v6 import FloatProbabilityStore

    case_ids = list_case_ids(data_root, split="training")
    base_store = FloatProbabilityStore(base_store_root, task=task, split="oof")
    refined_store = FloatProbabilityStore(refined_store_root, task=task, split="oof")
    base_calibration = json.loads(base_calibration_path.read_text(encoding="utf-8"))
    refined_calibration = json.loads(refined_calibration_path.read_text(encoding="utf-8"))
    targets = []
    masks = []
    base_probabilities = []
    refined_probabilities = []
    for case_id in case_ids:
        target = derive_av3_target(read_png_float(data_root / "training/av" / f"{case_id}.png", channels=3))
        roi = read_png_float(data_root / "training/masks" / f"{case_id}.png", channels=1).transpose(2, 0, 1)
        targets.append(target)
        masks.append(roi)
        base_probabilities.append(apply_calibration(base_store.read_case(case_id), base_calibration, case_id=case_id))
        refined_probabilities.append(apply_calibration(refined_store.read_case(case_id), refined_calibration, case_id=case_id))

    def metrics(probabilities):
        report = challenge_selection_score(np.stack(probabilities), np.stack(targets), np.stack(masks))
        case_scores = []
        for index in range(len(case_ids)):
            classification = np.mean(
                [report["sensitivity_cases"][index], report["specificity_cases"][index], report["accuracy_cases"][index]]
            )
            case_scores.append(
                0.4 * report["dice_cases"][index]
                + 0.3 * classification
                + 0.3 * report["topology_cases"][index]
            )
        report["selection_score_cases"] = case_scores
        report["case_ids"] = case_ids
        return report

    base_metrics = metrics(base_probabilities)
    refined_metrics = metrics(refined_probabilities)
    gate = evaluate_refiner_gate(base_metrics, refined_metrics)
    return {"task": task, "base": base_metrics, "refined": refined_metrics, "gate": gate}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or gate the optional GAVE2 V6 residual refiner.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--data-root", type=Path, required=True)
    train.add_argument("--base-store-root", type=Path, required=True)
    train.add_argument("--fold-manifest", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--task", choices=("task1", "task2"), required=True)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--base-channels", type=int, default=8)
    train.add_argument("--device", default="auto")
    predict = subparsers.add_parser("predict")
    predict.add_argument("--data-root", type=Path, required=True)
    predict.add_argument("--base-store-root", type=Path, required=True)
    predict.add_argument("--fold-manifest", type=Path, required=True)
    predict.add_argument("--refiner-dir", type=Path, required=True)
    predict.add_argument("--output-store-root", type=Path, required=True)
    predict.add_argument("--task", choices=("task1", "task2"), required=True)
    predict.add_argument("--mode", choices=("oof", "validation"), required=True)
    predict.add_argument("--device", default="auto")
    compare = subparsers.add_parser("compare")
    compare.add_argument("--data-root", type=Path, required=True)
    compare.add_argument("--base-store-root", type=Path, required=True)
    compare.add_argument("--refined-store-root", type=Path, required=True)
    compare.add_argument("--base-calibration", type=Path, required=True)
    compare.add_argument("--refined-calibration", type=Path, required=True)
    compare.add_argument("--task", choices=("task1", "task2"), required=True)
    compare.add_argument("--output", type=Path, required=True)
    gate = subparsers.add_parser("gate")
    gate.add_argument("--base-metrics", type=Path, required=True)
    gate.add_argument("--refined-metrics", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.command == "train":
        report = train_crossfit_refiners(
            args.data_root,
            args.base_store_root,
            args.fold_manifest,
            args.output_dir,
            task=args.task,
            epochs=args.epochs,
            base_channels=args.base_channels,
            device=args.device,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "predict":
        result = predict_refined_store(
            args.data_root,
            args.base_store_root,
            args.fold_manifest,
            args.refiner_dir,
            args.output_store_root,
            task=args.task,
            mode=args.mode,
            device=args.device,
        )
        print(result)
    elif args.command == "compare":
        report = compare_prediction_stores(
            args.data_root,
            args.base_store_root,
            args.refined_store_root,
            args.base_calibration,
            args.refined_calibration,
            task=args.task,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report["gate"], indent=2))
    else:
        base = json.loads(args.base_metrics.read_text(encoding="utf-8"))
        refined = json.loads(args.refined_metrics.read_text(encoding="utf-8"))
        report = evaluate_refiner_gate(base, refined)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
