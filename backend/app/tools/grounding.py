"""Region grounding: find the areas a query refers to and return boxes."""
from __future__ import annotations

import numpy as np

from ..raster import Scene
from .common import regions
from .landcover import classify

# Named image sectors -> (x0, y0, x1, y1) in normalised coordinates.
SECTORS = {
    "north": (0, 0, 1, 0.5), "top": (0, 0, 1, 0.5),
    "south": (0, 0.5, 1, 1), "bottom": (0, 0.5, 1, 1),
    "west": (0, 0, 0.5, 1), "left": (0, 0, 0.5, 1),
    "east": (0.5, 0, 1, 1), "right": (0.5, 0, 1, 1),
    "north-west": (0, 0, 0.5, 0.5), "top-left": (0, 0, 0.5, 0.5),
    "north-east": (0.5, 0, 1, 0.5), "top-right": (0.5, 0, 1, 0.5),
    "south-west": (0, 0.5, 0.5, 1), "bottom-left": (0, 0.5, 0.5, 1),
    "south-east": (0.5, 0.5, 1, 1), "bottom-right": (0.5, 0.5, 1, 1),
    "centre": (0.25, 0.25, 0.75, 0.75), "center": (0.25, 0.25, 0.75, 0.75),
}


def find_sector(text: str) -> str | None:
    t = text.lower().replace("northwest", "north-west").replace("northeast", "north-east")
    t = t.replace("southwest", "south-west").replace("southeast", "south-east")
    t = t.replace("top left", "top-left").replace("top right", "top-right")
    t = t.replace("bottom left", "bottom-left").replace("bottom right", "bottom-right")
    for key in sorted(SECTORS, key=len, reverse=True):  # longest match first
        if key in t:
            return key
    return None


def ground(scene: Scene, target: str, sector: str | None = None, top_k: int = 8, lc: dict | None = None) -> dict:
    lc = lc or classify(scene)
    if target not in lc["masks"]:
        raise ValueError(f"Unknown target '{target}'.")
    mask = lc["masks"][target].copy()
    if sector:
        x0, y0, x1, y1 = SECTORS[sector]
        h, w = mask.shape
        keep = np.zeros_like(mask)
        keep[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = True
        mask &= keep
    regs = regions(mask, scene.gsd_m, target.replace("_", " "), top_k=top_k)
    return {
        "mask": mask,
        "target": target,
        "sector": sector,
        "regions": regs,
        "fraction": round(float(mask.mean()), 4),
        "method": "class mask -> connected components -> ranked bounding boxes",
    }
