from __future__ import annotations

from PIL import Image, ImageFilter, ImageOps
import numpy as np


PREPROCESS_MODES = ("none", "clahe", "gray_clahe", "vessel_enhance")


def _clip01(array: np.ndarray) -> np.ndarray:
    return np.clip(array.astype(np.float32, copy=False), 0.0, 1.0)


def _as_uint8(channel: np.ndarray) -> np.ndarray:
    return np.round(_clip01(channel) * 255.0).astype(np.uint8)


def _pil_equalize(channel: np.ndarray) -> np.ndarray:
    image = Image.fromarray(_as_uint8(channel), mode="L")
    image = ImageOps.equalize(ImageOps.autocontrast(image))
    return np.asarray(image).astype(np.float32) / 255.0


def enhance_channel(channel: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Enhance one [0, 1] channel with CLAHE if available.

    Colab usually ships with OpenCV, which gives true CLAHE. The PIL fallback
    keeps the same API for local smoke tests and still improves global contrast.
    """

    channel = _clip01(channel)
    try:
        import cv2  # type: ignore

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(_as_uint8(channel)).astype(np.float32) / 255.0
    except Exception:
        return _pil_equalize(channel)


def green_biased_grayscale(cfp_rgb: np.ndarray) -> np.ndarray:
    cfp_rgb = _clip01(cfp_rgb)
    if cfp_rgb.ndim != 3 or cfp_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB CFP image, got shape {cfp_rgb.shape}")
    red = cfp_rgb[..., 0]
    green = cfp_rgb[..., 1]
    blue = cfp_rgb[..., 2]
    return _clip01(0.20 * red + 0.70 * green + 0.10 * blue)


def _unsharp(channel: np.ndarray, radius: float = 2.0, amount: float = 1.25) -> np.ndarray:
    channel = _clip01(channel)
    image = Image.fromarray(_as_uint8(channel), mode="L")
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
    blurred_array = np.asarray(blurred).astype(np.float32) / 255.0
    return _clip01(channel + amount * (channel - blurred_array))


def _zero_outside_roi(image: np.ndarray, roi: np.ndarray) -> np.ndarray:
    if roi.ndim == 2:
        roi = roi[..., None]
    if roi.shape[:2] != image.shape[:2]:
        raise ValueError(f"ROI shape {roi.shape[:2]} does not match image shape {image.shape[:2]}")
    return image * (roi[..., :1] > 0.5).astype(np.float32)


def preprocess_modalities(
    cfp_rgb: np.ndarray,
    roi: np.ndarray,
    ffa_early: np.ndarray | None = None,
    ffa_late: np.ndarray | None = None,
    mode: str = "none",
) -> np.ndarray:
    """Build a native-size model input with optional vessel-contrast enhancement.

    Returns HWC float32 input channels. Task 1 returns 3 channels; Task 2 returns
    5 channels. Enhanced modes keep the same canvas and softly improve vessel
    contrast without thresholding vessel pixels.
    """

    mode = mode.lower()
    if mode not in PREPROCESS_MODES:
        raise ValueError(f"Unsupported preprocess mode {mode!r}; expected one of {PREPROCESS_MODES}")

    cfp_rgb = _clip01(cfp_rgb)
    extra_channels = [channel for channel in (ffa_early, ffa_late) if channel is not None]
    extra_channels = [_clip01(channel[..., :1] if channel.ndim == 3 else channel[..., None]) for channel in extra_channels]

    if mode == "none":
        channels = [cfp_rgb] + extra_channels
        out = np.concatenate(channels, axis=2).astype(np.float32, copy=False)
        return _zero_outside_roi(out, roi)

    if mode == "clahe":
        cfp_out = np.stack([enhance_channel(cfp_rgb[..., i]) for i in range(3)], axis=2)
    else:
        gray = enhance_channel(green_biased_grayscale(cfp_rgb))
        if mode == "vessel_enhance":
            gray = enhance_channel(_unsharp(gray))
        cfp_out = np.repeat(gray[..., None], 3, axis=2)

    enhanced_extra = [enhance_channel(channel[..., 0])[..., None] for channel in extra_channels]
    channels = [cfp_out] + enhanced_extra
    out = np.concatenate(channels, axis=2).astype(np.float32, copy=False)
    return _zero_outside_roi(out, roi)
