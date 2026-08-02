from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from .preprocessing import enhance_channel, green_biased_grayscale


def _clip01(array: np.ndarray) -> np.ndarray:
    return np.clip(array.astype(np.float32, copy=False), 0.0, 1.0)


def _as_uint8(channel: np.ndarray) -> np.ndarray:
    return np.round(_clip01(channel) * 255.0).astype(np.uint8)


def _unsharp_mask(channel: np.ndarray, radius: float = 2.0, amount: float = 1.25) -> np.ndarray:
    image = Image.fromarray(_as_uint8(channel), mode="L")
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
    blurred_array = np.asarray(blurred, dtype=np.float32) / 255.0
    return _clip01(channel + amount * (channel - blurred_array))


def _normalize_channel(channel: np.ndarray) -> np.ndarray:
    if channel.ndim == 2:
        return _clip01(channel)[..., None]
    if channel.ndim == 3 and channel.shape[2] == 1:
        return _clip01(channel)
    raise ValueError(f"Expected a single-channel image, got shape {channel.shape}")


def _zero_outside_roi(image: np.ndarray, roi: np.ndarray) -> np.ndarray:
    roi_channel = _normalize_channel(roi)
    if roi_channel.shape[:2] != image.shape[:2]:
        raise ValueError(f"ROI shape {roi_channel.shape[:2]} does not match image shape {image.shape[:2]}")
    return image * (roi_channel > 0.5).astype(np.float32)


def build_vessel_enhanced_channel(cfp_rgb: np.ndarray) -> np.ndarray:
    cfp_rgb = _clip01(cfp_rgb)
    if cfp_rgb.ndim != 3 or cfp_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB CFP image, got shape {cfp_rgb.shape}")
    gray = green_biased_grayscale(cfp_rgb)
    enhanced = enhance_channel(gray)
    return enhance_channel(_unsharp_mask(enhanced))


def preprocess_v6_modalities(
    cfp_rgb: np.ndarray,
    roi: np.ndarray,
    ffa_early: np.ndarray | None = None,
    ffa_late: np.ndarray | None = None,
) -> np.ndarray:
    """Build the V6 native-resolution input tensor.

    Task 1 returns RGB plus one enhanced grayscale vessel channel.
    Task 2 appends registered FFA-A and FFA-AV to those four channels.
    """

    cfp_rgb = _clip01(cfp_rgb)
    has_ffa_early = ffa_early is not None
    has_ffa_late = ffa_late is not None
    if has_ffa_early != has_ffa_late:
        raise ValueError("FFA inputs must be both present or both absent")
    vessel_channel = build_vessel_enhanced_channel(cfp_rgb)[..., None]
    channels = [cfp_rgb, vessel_channel]
    for channel in (ffa_early, ffa_late):
        if channel is not None:
            channels.append(_normalize_channel(channel))
    out = np.concatenate(channels, axis=2).astype(np.float32, copy=False)
    expected_channels = 6 if has_ffa_early else 4
    if out.shape[2] != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels, got {out.shape[2]}")
    return _zero_outside_roi(out, roi)
