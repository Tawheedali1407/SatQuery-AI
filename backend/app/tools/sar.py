"""SAR processing: Lee speckle filter and low-backscatter water / flood mapping.

Calm open water is a specular reflector, so it returns very little energy to
the sensor and appears dark (typically below about -15 to -20 dB in VV). We
threshold filtered backscatter with Otsu, bounded to that physical range. Rough
water, wet snow and radar shadow are known confusers; the fusion tool uses the
optical scene to resolve them where it is available.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from ..raster import Scene
from .common import area, clean_mask, margin_confidence, otsu, regions


def lee_filter(img_db: np.ndarray, size: int = 5) -> np.ndarray:
    """Classic Lee filter, applied in linear power and returned in dB."""
    lin = 10 ** (img_db / 10.0)
    mean = uniform_filter(lin, size)
    sq_mean = uniform_filter(lin * lin, size)
    var = np.maximum(sq_mean - mean * mean, 0)
    noise_var = np.mean(var)
    weight = var / (var + noise_var + 1e-12)
    out = mean + weight * (lin - mean)
    return (10 * np.log10(np.clip(out, 1e-6, None))).astype(np.float32)


def sar_water(scene: Scene, filter_size: int = 5) -> dict:
    if scene.modality != "sar" or scene.sar_db is None:
        raise ValueError("SAR water mapping needs a SAR scene.")
    db = lee_filter(scene.sar_db, filter_size)
    thr = float(np.clip(otsu(db, -18.0), -24.0, -12.0))
    mask = clean_mask(db < thr, min_px=24)
    return {
        "mask": mask,
        "filtered_db": db,
        "threshold_db": round(thr, 2),
        "count": len(regions(mask, scene.gsd_m, "water", top_k=10_000)),
        "fraction": round(float(mask.mean()), 4),
        "area": area(int(mask.sum()), scene.gsd_m),
        "regions": regions(mask, scene.gsd_m, "water body"),
        "confidence": margin_confidence(db, thr, mask, spread=3.0),
        "method": f"Lee filter ({filter_size}x{filter_size}) + Otsu threshold on VV backscatter",
    }
