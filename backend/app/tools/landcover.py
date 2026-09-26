"""Land-cover and water mapping from spectral indices.

With a NIR band we use the standard NDVI / McFeeters NDWI. For RGB-only
imagery we fall back to visible-band rules (Excess Green with a green-dominance
check, a blue-over-red water index, and a very-dark-and-bluish rule for deep
water). Thresholds come from Otsu on each index, clamped to physically sensible
ranges so a scene with no water does not get "half water" by construction.
Clouds and no-data are masked out before any fraction is reported.

This is a transparent, training-free baseline. RGB-only mapping has known
limits (dark seagrass shallows vs dark forest, blue roofs vs water); with NIR
(Sentinel-2, LISS-IV) it is much more reliable. A fine-tuned segmentation
model can replace `classify()` without touching the API.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ..raster import Scene
from .common import area, clean_mask, margin_confidence, otsu, regions

CLASSES = ["water", "vegetation", "built_up", "bare_soil", "cloud", "other", "nodata"]
REPORTED = ["water", "vegetation", "built_up", "bare_soil", "cloud", "other"]
CLASS_COLORS = {  # RGBA used by overlays and mirrored in the UI legend
    "water": (30, 120, 255, 170),
    "vegetation": (40, 190, 80, 150),
    "built_up": (240, 70, 70, 150),
    "bare_soil": (220, 170, 90, 150),
    "cloud": (235, 235, 245, 150),
    "other": (0, 0, 0, 0),
    "nodata": (0, 0, 0, 0),
}


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / (a + b + 1e-6)


def indices(scene: Scene) -> dict[str, np.ndarray]:
    r, g, b = scene.band("R"), scene.band("G"), scene.band("B")
    nir = scene.band("NIR")
    s = r + g + b + 1e-6
    brightness = (r + g + b) / 3
    mx, mn = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    out = {"brightness": brightness, "saturation": (mx - mn) / (mx + 1e-6)}
    if nir is not None:
        out["veg"] = _nd(nir, r)  # NDVI
        out["water"] = _nd(g, nir)  # NDWI (McFeeters)
        out["_mode"] = "NIR"
    else:
        out["veg"] = (2 * g - r - b) / s  # Excess Green (chromatic coordinates)
        out["water"] = _nd(b, r)  # water absorbs red far more than blue
        out["_mode"] = "RGB"
    return out


def classify(scene: Scene) -> dict:
    if scene.modality != "optical":
        raise ValueError("Land-cover classification needs an optical scene.")
    ix = indices(scene)
    nir_mode = ix["_mode"] == "NIR"
    r, g, b = scene.band("R"), scene.band("G"), scene.band("B")
    valid = scene.valid if scene.valid is not None else np.ones(scene.shape, bool)
    bright, sat = ix["brightness"], ix["saturation"]

    cloud = clean_mask((bright > 0.75) & (sat < 0.2), min_px=32, close=1)
    cloud = valid & ndi.binary_dilation(cloud, iterations=2)  # include thin haze halos at cloud edges
    usable = valid & ~cloud

    w_thr = otsu(ix["water"][usable], 0.0 if nir_mode else 0.3)
    w_thr = float(np.clip(w_thr, -0.05, 0.4) if nir_mode else np.clip(w_thr, 0.15, 0.6))
    water = usable & (ix["water"] > w_thr)
    if not nir_mode:
        water &= (bright < 0.55) & (b >= g * 0.8)  # bright blue roofs / tarps are not water
        water |= usable & (bright < 0.07) & (b >= r)  # deep, very dark water
    water = clean_mask(water, min_px=24)

    v_thr = otsu(ix["veg"][usable & ~water], 0.2 if nir_mode else 0.06)
    v_thr = float(np.clip(v_thr, 0.15, 0.5) if nir_mode else np.clip(v_thr, 0.03, 0.15))
    veg = usable & ~water & (ix["veg"] > v_thr)
    if not nir_mode:
        veg &= (g > b) & (g >= r * 0.95)  # green must dominate; turquoise water fails this

    rest = usable & ~(veg | water)
    lit = bright > np.percentile(bright[usable], 45) if usable.any() else bright > 0.5
    built = rest & lit & (sat < 0.35)
    bare = rest & ~built & (r > b) & (r >= g * 0.95) & (sat >= 0.12)

    cmap = np.full(scene.shape, CLASSES.index("other"), np.uint8)
    for name, m in (("bare_soil", bare), ("built_up", built), ("vegetation", veg), ("water", water),
                    ("cloud", cloud)):
        cmap[m] = CLASSES.index(name)
    cmap[~valid] = CLASSES.index("nodata")

    total = max(int(valid.sum()), 1)  # fractions are of valid (non no-data) pixels
    counts = {c: int((cmap == CLASSES.index(c)).sum()) for c in REPORTED}
    return {
        "class_map": cmap,
        "masks": {c: cmap == CLASSES.index(c) for c in REPORTED},
        "fractions": {c: round(n / total, 4) for c, n in counts.items()},
        "areas": {c: area(n, scene.gsd_m) for c, n in counts.items()},
        "valid_fraction": round(float(valid.mean()), 4),
        "thresholds": {"vegetation": round(v_thr, 4), "water": round(w_thr, 4)},
        "confidence": {
            "water": margin_confidence(ix["water"], w_thr, water),
            "vegetation": margin_confidence(ix["veg"], v_thr, veg),
        },
        "index_mode": ix["_mode"],
        "method": ("NDVI / NDWI with Otsu thresholds, cloud + no-data masked" if nir_mode
                   else "visible-band indices (ExG, blue-red water index) with Otsu thresholds, cloud + no-data masked"),
    }


def water_bodies(scene: Scene, lc: dict | None = None, min_px: int = 24) -> dict:
    """Water mask + individual water bodies (connected components)."""
    if scene.modality == "sar":
        from .sar import sar_water

        return sar_water(scene)
    lc = lc or classify(scene)
    mask = lc["masks"]["water"]
    regs = regions(mask, scene.gsd_m, "water body", top_k=10_000)
    return {
        "mask": mask,
        "count": sum(r["area"]["pixels"] >= min_px for r in regs),
        "fraction": lc["fractions"]["water"],
        "area": lc["areas"]["water"],
        "regions": regs[:12],
        "confidence": lc["confidence"]["water"],
        "method": "optical " + lc["method"],
    }
