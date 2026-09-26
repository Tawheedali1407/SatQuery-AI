"""Optical-SAR decision-level fusion for water / flood mapping.

Each sensor has blind spots: optical cannot see through cloud; SAR confuses
calm water with smooth tarmac and radar shadow. We map water independently in
both, then combine the maps:

* both agree            -> high-confidence water
* SAR only, under cloud -> accepted (optical is blind there)
* SAR only, clear sky   -> flagged for review (possible smooth surface/shadow)
* optical only          -> accepted with medium confidence (e.g. windy water rough on SAR)

The output is one water map plus an agreement report, which is the
"single interpretation from multi-sensor data" asked for in PS 26167.
"""
from __future__ import annotations

import numpy as np
from skimage.transform import resize

from ..raster import Scene
from .common import area, clean_mask, regions
from .landcover import classify
from .sar import sar_water


def fuse(optical: Scene, sar: Scene) -> dict:
    if optical.modality != "optical" or sar.modality != "sar":
        raise ValueError("Fusion needs one optical and one SAR scene of the same area.")
    warnings = []
    lc = classify(optical)
    opt_w = lc["masks"]["water"]
    sw = sar_water(sar)
    sar_w = sw["mask"]
    if sar_w.shape != opt_w.shape:
        warnings.append("SAR grid resampled to the optical grid; scenes are assumed co-registered.")
        sar_w = resize(sar_w.astype(float), opt_w.shape, order=0) > 0.5
    clouds = lc["masks"]["cloud"]

    both = opt_w & sar_w
    sar_only = sar_w & ~opt_w
    opt_only = opt_w & ~sar_w
    sar_cloud = sar_only & clouds
    review = sar_only & ~clouds

    fused = clean_mask(both | sar_cloud | opt_only, min_px=24)
    conf_map = np.zeros(opt_w.shape, np.float32)
    conf_map[opt_only] = 0.6
    conf_map[sar_cloud] = 0.7
    conf_map[both] = 0.95

    union = (opt_w | sar_w).sum()
    iou = float(both.sum() / union) if union else 1.0
    g = optical.gsd_m or sar.gsd_m
    sar_disp = sar.data[..., 0]
    if sar_disp.shape != opt_w.shape:
        sar_disp = resize(sar_disp, opt_w.shape, order=1)
    composite = np.dstack([sar_disp, optical.band("G"), optical.band("B")])  # false colour: SAR in red
    return {
        "mask": fused,
        "both_mask": both,
        "review_mask": review,
        "composite": composite,
        "fraction": round(float(fused.mean()), 4),
        "area": area(int(fused.sum()), g),
        "agreement_iou": round(iou, 3),
        "cloud_fraction": round(float(clouds.mean()), 4),
        "breakdown": {
            "both_sensors": area(int(both.sum()), g),
            "sar_only_under_cloud": area(int(sar_cloud.sum()), g),
            "optical_only": area(int(opt_only.sum()), g),
            "sar_only_needs_review": area(int(review.sum()), g),
        },
        "regions": regions(fused, g, "water (fused)", score_map=conf_map),
        "confidence": round(float(conf_map[fused].mean()) if fused.any() else 0.0, 3),
        "sar_threshold_db": sw["threshold_db"],
        "optical_landcover": lc["fractions"],
        "method": "independent optical + SAR water maps, rule-based decision fusion with cloud screening",
        "warnings": warnings,
    }
