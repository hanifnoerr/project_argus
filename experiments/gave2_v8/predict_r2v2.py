from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from skimage import io
from skimage.transform import resize

from . import R2V2_SOURCE_COMMIT
from .assets import ASSETS, sha256_file, verify_asset
from .store import ProbabilityStore


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _read_zero_one(path: Path, resize_width: int = 1408) -> tuple[np.ndarray, tuple[int, int] | None]:
    image = io.imread(path).astype(np.float32)
    if image.max(initial=0) > 255:
        image /= 65535.0
    if image.max(initial=0) > 1:
        image /= 255.0
    image = np.clip(image, 0.0, 1.0)
    if image.ndim == 3:
        image = image[..., :3]
    height, width = image.shape[:2]
    original_shape = (height, width) if width != resize_width else None
    if original_shape is not None:
        new_height = int(height * (resize_width / width))
        image = resize(
            image,
            (new_height, resize_width),
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32)
    return image.astype(np.float32, copy=False), original_shape


def _padding(shape: tuple[int, ...], divisor: int = 32):
    height_pad = divisor - shape[0] % divisor
    width_pad = divisor - shape[1] % divisor
    spatial = (
        (height_pad // 2, height_pad - height_pad // 2),
        (width_pad // 2, width_pad - width_pad // 2),
    )
    return spatial + (((0, 0),) if len(shape) == 3 else ())


def _pad(array: np.ndarray):
    padding = _padding(array.shape)
    return np.pad(array, padding), padding


def _to_tensor(array: np.ndarray, device: str):
    import torch

    if array.ndim == 2:
        array = array[..., None]
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1), dtype=np.float32))[None].to(device)


def _transform(tensor, rotation: int, flip: bool):
    import torch

    value = torch.rot90(tensor, k=rotation, dims=(2, 3))
    return torch.flip(value, dims=(3,)) if flip else value


def _invert(tensor, rotation: int, flip: bool):
    import torch

    value = torch.flip(tensor, dims=(3,)) if flip else tensor
    return torch.rot90(value, k=-rotation, dims=(2, 3))


def _amp_settings(device: str, requested: str):
    import torch

    if not device.startswith("cuda"):
        return "fp32", None
    if requested == "auto":
        requested = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested but is not supported by this CUDA device")
        return requested, torch.bfloat16
    if requested == "fp16":
        return requested, torch.float16
    if requested == "fp32":
        return requested, None
    raise ValueError(requested)


def _predict_tta(model, image, mask, model_type: str, amp_dtype, use_tta: bool):
    import torch

    transforms = [(0, False)] if not use_tta else [(rotation, flip) for rotation in range(4) for flip in (False, True)]
    predictions = []
    for rotation, flip in transforms:
        transformed_image = _transform(image, rotation, flip)
        transformed_mask = _transform(mask, rotation, flip)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda" if image.device.type == "cuda" else "cpu",
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            output = model(transformed_image)
            logits = output[-1] if isinstance(output, (list, tuple)) else output
            probability = torch.sigmoid(logits.float()) * transformed_mask
        predictions.append(_invert(probability, rotation, flip))

    if len(predictions) == 1:
        return predictions[0] * mask
    artery = torch.stack([value[:, 0] for value in predictions], dim=1)
    vein = torch.stack([value[:, 1] for value in predictions], dim=1)
    vessel = torch.stack([value[:, 2] for value in predictions], dim=1)
    artery_mean = artery.mean(dim=1, keepdim=True)
    vein_mean = vein.mean(dim=1, keepdim=True)
    if model_type == "bv":
        av_union = (artery + vein).clamp(0.0, 1.0)
        vessel_views = torch.cat((vessel, av_union), dim=1)
        vessel_mean = vessel_views.mean(dim=1, keepdim=True)
        artery_mean = torch.where(artery_mean > 0.5, artery.max(dim=1, keepdim=True).values, artery_mean)
        vein_mean = torch.where(vein_mean > 0.5, vein.max(dim=1, keepdim=True).values, vein_mean)
        vessel_mean = torch.where(vessel_mean > 0.5, vessel_views.max(dim=1, keepdim=True).values, vessel_mean)
    else:
        vessel_mean = vessel.mean(dim=1, keepdim=True)
    return torch.cat((artery_mean, vein_mean, vessel_mean), dim=1) * mask


def load_model(source_dir: Path, weights_dir: Path, model_type: str, device: str):
    import torch

    model_module = _load_module("gave2_v8_r2v2_model", source_dir / "model.py")
    config = json.loads((weights_dir / f"{model_type}_config.json").read_text(encoding="utf-8"))
    try:
        checkpoint = torch.load(weights_dir / f"{model_type}.pth", map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(weights_dir / f"{model_type}.pth", map_location="cpu")
    model = model_module.RRWNet(
        config["in_channels"],
        config["out_channels"],
        config["base_channels"],
        config["num_iterations"],
    )
    model.load_state_dict(checkpoint)
    return model.eval().to(device), config


def predict_case(
    cfp_path: Path,
    mask_path: Path,
    *,
    source_dir: Path,
    model,
    model_type: str,
    device: str,
    amp_dtype,
    use_tta: bool,
    resize_width: int,
) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    preprocessing = _load_module("gave2_v8_r2v2_preprocessing", source_dir / "preprocessing.py")
    cfp, original_shape = _read_zero_one(cfp_path, resize_width=resize_width)
    mask, _ = _read_zero_one(mask_path, resize_width=resize_width)
    if mask.ndim == 3:
        mask = mask[..., 0]
    enhanced, enhanced_mask = preprocessing.preprocess_img(cfp, mask)
    image = np.concatenate((enhanced, cfp), axis=-1) if model_type == "bv" else enhanced
    image, padding = _pad(np.asarray(image, dtype=np.float32))
    padded_mask, _ = _pad(np.asarray(enhanced_mask, dtype=np.float32))
    padded_mask = (padded_mask > 0.5).astype(np.float32)[..., None]
    image_tensor = _to_tensor(image, device)
    mask_tensor = _to_tensor(padded_mask, device)
    prediction = _predict_tta(model, image_tensor, mask_tensor, model_type, amp_dtype, use_tta)
    top, bottom = padding[0]
    left, right = padding[1]
    prediction = prediction[..., top : prediction.shape[-2] - bottom, left : prediction.shape[-1] - right]
    if original_shape is not None:
        prediction = functional.interpolate(prediction, size=original_shape, mode="bilinear", align_corners=False)
    # Upstream order is artery, vein, all vessels. GAVE2 order is artery, all vessels, vein.
    gave = torch.stack((prediction[:, 0], prediction[:, 2], prediction[:, 1]), dim=1)
    return np.clip(gave[0].float().cpu().numpy(), 0.0, 1.0).astype(np.float32)


def _case_ids(data_root: Path, split: str) -> list[str]:
    return [path.stem for path in sorted((data_root / split / "images").glob("*.png"))]


def run_prediction(args: argparse.Namespace) -> dict[str, object]:
    import torch

    source_dir = args.source_dir.resolve()
    weights_dir = args.weights_dir.resolve()
    if not (source_dir / "model.py").exists() or not (source_dir / "preprocessing.py").exists():
        raise FileNotFoundError(f"Incomplete R2-V2 source at {source_dir}")
    for name in (f"{args.model_type}.pth", f"{args.model_type}_config.json"):
        verify_asset(weights_dir / name, name)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_name, amp_dtype = _amp_settings(device, args.amp)
    model, config = load_model(source_dir, weights_dir, args.model_type, device)
    store = ProbabilityStore(args.output_store, namespace=f"r2v2_{args.model_type}", split=args.split)
    case_ids = _case_ids(args.data_root, args.split)
    if args.limit_cases is not None:
        case_ids = case_ids[: args.limit_cases]
    common = {
        "revision": "r2v2_v8_rectangular_tta_v1",
        "source_commit": R2V2_SOURCE_COMMIT,
        "model_type": args.model_type,
        "weight_sha256": ASSETS[f"{args.model_type}.pth"]["sha256"],
        "config_sha256": ASSETS[f"{args.model_type}_config.json"]["sha256"],
        "tta": bool(args.tta),
        "amp": amp_name,
        "resize_width": int(args.resize_width),
    }
    provenance = {}
    for case_id in case_ids:
        cfp = args.data_root / args.split / "images" / f"{case_id}.png"
        mask = args.data_root / args.split / "masks" / f"{case_id}.png"
        provenance[case_id] = {
            **common,
            "cfp_sha256": sha256_file(cfp),
            "mask_sha256": sha256_file(mask),
        }
    pending = store.pending(case_ids, provenance)
    for index, case_id in enumerate(pending, 1):
        print(f"[{index}/{len(pending)}] {args.model_type} {args.split} {case_id}")
        probability = predict_case(
            args.data_root / args.split / "images" / f"{case_id}.png",
            args.data_root / args.split / "masks" / f"{case_id}.png",
            source_dir=source_dir,
            model=model,
            model_type=args.model_type,
            device=device,
            amp_dtype=amp_dtype,
            use_tta=args.tta,
            resize_width=args.resize_width,
        )
        store.write_case(case_id, probability, provenance[case_id])
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return {
        "model_type": args.model_type,
        "split": args.split,
        "cases": len(case_ids),
        "new_cases": len(pending),
        "output_store": str(args.output_store),
        "amp": amp_name,
        "device": device,
        "config": config,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable inference with released R2-V2 weights.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output-store", type=Path, required=True)
    parser.add_argument("--model-type", choices=("av", "bv"), required=True)
    parser.add_argument("--split", choices=("training", "validation"), default="validation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--resize-width", type=int, default=1408)
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-cases", type=int)
    return parser.parse_args(argv)


def main() -> None:
    print(json.dumps(run_prediction(parse_args()), indent=2))


if __name__ == "__main__":
    main()

