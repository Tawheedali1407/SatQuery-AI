"""Shared helpers for the specialist tools: thresholds, clean-up and region extraction."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu


def otsu(values: np.ndarray, fallback: float) -> float:
    v = values[np.isfinite(values)].ravel()
    if v.size < 16 or np.ptp(v) < 1e-6:
        return fallback
    return float(threshold_otsu(v))


def clean_mask(mask: np.ndarray, min_px: int = 16, close: int = 2, open_: bool = True) -> np.ndarray:
    """Morphological open/close then drop specks smaller than `min_px`."""
    m = ndi.binary_opening(mask, iterations=1) if open_ else mask
    if close:
        m = ndi.binary_closing(m, iterations=close)
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_px
    return keep[lab]


def area(px: int | float, gsd_m: float | None) -> dict:
    out = {"pixels": int(px)}
    if gsd_m:
        m2 = float(px) * gsd_m * gsd_m
        out["m2"] = round(m2, 1)
        out["km2"] = round(m2 / 1e6, 4)
        out["hectares"] = round(m2 / 1e4, 3)
    return out


def regions(mask: np.ndarray, gsd_m: float | None, label: str, top_k: int = 12, score_map: np.ndarray | None = None) -> list[dict]:
    """Connected components -> bounding boxes, sorted by size (largest first).
    bbox is [x0, y0, x1, y1] in pixel coords of the analysed array; also returned
    normalised (0..1) so the UI can draw it at any display size."""
    lab, n = ndi.label(mask)
    if n == 0:
        return []
    h, w = mask.shape
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    order = np.argsort(sizes)[::-1][:top_k]
    slices = ndi.find_objects(lab)
    out = []
    for idx in order:
        sl = slices[idx]
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        reg = {
            "label": label,
            "bbox": [int(x0), int(y0), int(x1), int(y1)],
            "bbox_norm": [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)],
            "centroid_norm": [round((x0 + x1) / 2 / w, 4), round((y0 + y1) / 2 / h, 4)],
            "area": area(sizes[idx], gsd_m),
        }
        if score_map is not None:
            reg["confidence"] = round(float(np.clip(score_map[lab == idx + 1].mean(), 0, 1)), 3)
        out.append(reg)
    return out


def margin_confidence(index: np.ndarray, thr: float, mask: np.ndarray, spread: float | None = None) -> float:
    """Mean distance of classified pixels from the decision threshold, squashed to 0..1.
    Pixels far from the threshold are confidently classified; a mask made of
    near-threshold pixels gets a low score. This is a heuristic, not a calibrated probability."""
    if not mask.any():
        return 0.0
    spread = spread or (np.std(index) + 1e-6)
    d = np.abs(index[mask] - thr) / spread
    return round(float(np.clip(np.tanh(d.mean()), 0, 1)), 3)
