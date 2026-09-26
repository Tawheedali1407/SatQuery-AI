"""Bi-temporal change detection.

Optical: histogram-match "after" to "before" (removes illumination / season
offsets), smooth, compute the Change Vector Analysis (CVA) magnitude across
bands, and threshold it with Otsu. SAR: log-ratio of Lee-filtered backscatter.
A class-focused mode ("how much new water?") classifies both dates and
reports gains and losses for that class, which is what a flood analyst asks for.

Both scenes are assumed to be co-registered (same footprint and grid), as they
are when exported from Bhoonidhi / Copernicus for the same AOI. If shapes
differ, the after-scene is resampled onto the before-grid and a warning is returned.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.exposure import match_histograms
from skimage.metrics import structural_similarity
from skimage.transform import resize

from ..config import settings
from ..raster import Scene
from .common import area, clean_mask, margin_confidence, otsu, regions
from .landcover import classify
from .sar import lee_filter, sar_water


def _align(before: Scene, after: Scene) -> tuple[np.ndarray, np.ndarray, list[str]]:
    warnings = []
    a, b = before.data, after.data
    if a.shape[:2] != b.shape[:2]:
        warnings.append(
            f"Scenes differ in size ({a.shape[1]}x{a.shape[0]} vs {b.shape[1]}x{b.shape[0]}); "
            "the after-scene was resampled onto the before-grid. Co-register them for best results."
        )
        b = resize(b, a.shape[:2] + (b.shape[2],), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
    if before.modality != after.modality:
        raise ValueError("Change detection needs two scenes of the same modality (optical/optical or SAR/SAR). "
                         "Use fusion to combine optical with SAR.")
    return a, b, warnings


def pick_backend(before: Scene, requested: str | None = None) -> str:
    """'changeformer' for high-resolution optical pairs when the weights are installed, else 'classical'."""
    from ..ml import changeformer

    requested = requested or settings.change_backend
    if requested == "classical" or before.modality != "optical":
        return "classical"
    if changeformer.available() and (requested == "changeformer" or changeformer.suitable(before.gsd_m)):
        return "changeformer"
    return "classical"


def detect_change(before: Scene, after: Scene, sigma: float = 1.5, min_px: int | None = None,
                  backend: str | None = None) -> dict:
    a, b, warnings = _align(before, after)
    h, w = a.shape[:2]
    min_px = min_px or max(16, int(h * w * 0.0004))
    backend = pick_backend(before, backend)
    spread = None

    if backend == "changeformer":
        from ..ml import changeformer

        # Prefer the unstretched 8-bit pixels the network was trained on; fall back to stretched RGB
        # for 16-bit / reflectance products.
        ra = before.rgb8 if before.rgb8 is not None else before.rgb()
        rb = after.rgb8 if after.rgb8 is not None else after.rgb()
        if rb.shape != ra.shape:
            rb = resize(rb, ra.shape, order=1, preserve_range=True).astype(np.float32)
        mag = changeformer.predict(ra, rb)  # P(change)
        thr = 0.5
        increase = (rb.mean(-1) - ra.mean(-1)) > 0
        min_px = 8  # the network's output is already spatially coherent
        method = "ChangeFormerV6 (transformer Siamese network, pretrained on LEVIR-CD), tiled 256 px inference"
        if before.gsd_m and before.gsd_m > settings.change_max_gsd_m:
            warnings.append(f"ChangeFormer was trained on 0.5 m imagery; this scene is {before.gsd_m:.1f} m/px.")
    elif before.modality == "sar":
        da = lee_filter(before.sar_db)
        db = lee_filter(after.sar_db if after.sar_db.shape == da.shape
                        else resize(after.sar_db, da.shape, order=1, preserve_range=True).astype(np.float32))
        mag = np.abs(db - da)
        thr = max(otsu(mag, 3.0), 3.0)  # at least 3 dB change
        increase = (db - da) > 0
        method = "log-ratio of Lee-filtered backscatter + Otsu (min 3 dB)"
        spread = 3.0
    else:
        nb = min(a.shape[2], b.shape[2])
        a, b = a[..., :nb], b[..., :nb]
        b = match_histograms(b, a, channel_axis=-1).astype(np.float32)
        a_s = ndi.gaussian_filter(a, sigma=(sigma, sigma, 0))
        b_s = ndi.gaussian_filter(b, sigma=(sigma, sigma, 0))
        diff = b_s - a_s
        cva = np.sqrt((diff ** 2).sum(-1)) / np.sqrt(nb)
        # Structural dissimilarity catches new buildings/roads whose colour is
        # close to what was there before but whose texture and edges are not.
        _, ssim_map = structural_similarity(a, b, channel_axis=-1, data_range=1.0, full=True,
                                            win_size=min(11, (min(h, w) // 2) * 2 - 1))
        dssim = 1.0 - ssim_map.mean(-1)
        mag = cva / (cva.std() + 1e-6) + dssim / (dssim.std() + 1e-6)
        thr = otsu(mag, 3.0)
        # Guard: when nothing changed, Otsu still splits noise in two. Require the
        # CVA component to show a real radiometric difference as well.
        if np.percentile(cva, 99) < 0.05:
            thr = float(mag.max()) + 1
        increase = diff.mean(-1) > 0  # brighter after (e.g. new roofs, cleared land)
        method = "histogram matching + CVA + structural dissimilarity (SSIM), Otsu threshold"

    deep = backend == "changeformer"
    mask = clean_mask(mag > thr, min_px=min_px, close=0 if deep else 2, open_=not deep)
    if backend == "changeformer":
        score = mag
        confidence = round(float(mag[mask].mean()), 3) if mask.any() else round(float(1 - mag.mean()), 3)
    else:
        score = np.clip((mag - thr) / (thr + 1e-6), 0, 1)
        confidence = margin_confidence(mag, thr, mask, spread=spread)
    return {
        "mask": mask,
        "magnitude": mag,
        "increase_mask": mask & increase,
        "decrease_mask": mask & ~increase,
        "threshold": round(float(thr), 4),
        "fraction": round(float(mask.mean()), 4),
        "area": area(int(mask.sum()), before.gsd_m),
        "regions": regions(mask, before.gsd_m, "building change" if deep else "change", score_map=score),
        "confidence": confidence,
        "method": method,
        "backend": backend,
        "warnings": warnings,
    }


def class_change(before: Scene, after: Scene, target: str) -> dict:
    """From-to change for one class (e.g. water gained = likely flooding)."""
    _, _, warnings = _align(before, after)

    def mask_for(scene: Scene) -> np.ndarray:
        if scene.modality == "sar":
            if target != "water":
                raise ValueError("SAR class change currently supports water only.")
            return sar_water(scene)["mask"]
        return classify(scene)["masks"][target]

    ma = mask_for(before)
    mb = mask_for(after)
    if mb.shape != ma.shape:
        mb = resize(mb.astype(float), ma.shape, order=0) > 0.5
    gained = clean_mask(mb & ~ma, min_px=24)
    lost = clean_mask(ma & ~mb, min_px=24)
    g = before.gsd_m
    return {
        "target": target,
        "mask": gained | lost,
        "gained_mask": gained,
        "lost_mask": lost,
        "before_fraction": round(float(ma.mean()), 4),
        "after_fraction": round(float(mb.mean()), 4),
        "gained": area(int(gained.sum()), g),
        "lost": area(int(lost.sum()), g),
        "net_change_pct_points": round(float(mb.mean() - ma.mean()) * 100, 2),
        "regions": regions(gained, g, f"{target} gained") + regions(lost, g, f"{target} lost", top_k=5),
        "method": f"per-date {target} mapping, then from-to comparison",
        "warnings": warnings,
    }
