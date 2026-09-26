"""Evidence rendering: masks and class maps -> PNG data URIs for the UI and reports."""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from .tools.landcover import CLASS_COLORS, CLASSES

MAX_SIDE = 768

COLORS = {
    "water": (30, 120, 255),
    "vegetation": (40, 190, 80),
    "built_up": (240, 70, 70),
    "bare_soil": (220, 170, 90),
    "change": (255, 60, 60),
    "gained": (255, 60, 60),
    "lost": (255, 200, 0),
    "review": (255, 200, 0),
    "fused": (0, 220, 255),
}


def _to_uint8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _encode(arr: np.ndarray) -> str:
    img = Image.fromarray(arr)
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def image(rgb: np.ndarray) -> str:
    return _encode(_to_uint8(rgb))


def overlay(rgb: np.ndarray, layers: list[tuple[np.ndarray, tuple[int, int, int]]], alpha: float = 0.55,
            outline: bool = True) -> str:
    """Tint each mask with its colour over a slightly darkened base image."""
    base = _to_uint8(rgb).astype(np.float32) * 0.8
    for mask, color in layers:
        if mask is None or not mask.any():
            continue
        c = np.array(color, np.float32)
        base[mask] = base[mask] * (1 - alpha) + c * alpha
        if outline:
            from scipy.ndimage import binary_erosion

            edge = mask & ~binary_erosion(mask, iterations=1)
            base[edge] = c
    return _encode(base.clip(0, 255).astype(np.uint8))


def class_overlay(rgb: np.ndarray, class_map: np.ndarray, alpha: float = 0.55) -> str:
    base = _to_uint8(rgb).astype(np.float32) * 0.8
    for i, c in enumerate(CLASSES):
        r, g, b, a = CLASS_COLORS[c]
        if a == 0:
            continue
        m = class_map == i
        base[m] = base[m] * (1 - alpha) + np.array([r, g, b], np.float32) * alpha
    return _encode(base.clip(0, 255).astype(np.uint8))


def heatmap(values: np.ndarray) -> str:
    v = values.astype(np.float32)
    lo, hi = np.percentile(v, [2, 99.5])
    t = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
    # simple dark-blue -> yellow ramp (perceptually ordered)
    rgb = np.dstack([t ** 0.8, t ** 1.2 * 0.9 + 0.05, 0.35 * (1 - t) + 0.05])
    return image(rgb)
