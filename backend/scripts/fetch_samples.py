#!/usr/bin/env python3
"""Download real demo scenes into ../samples and write samples/catalog.json.

Sources (all public, no account needed):
  * Landsat 7 RGB scene      - rasterio test data (GitHub)
  * LEVIR-CD bi-temporal     - 0.5 m Google Earth pairs with change labels, via the BIT_CD repo (GitHub).
                               Academic-use dataset: Chen & Shi, Remote Sensing 2020.
  * Sen1Floods11 (India)     - co-registered Sentinel-1 (SAR) + Sentinel-2 (optical) 10 m flood chips with
                               hand-labelled water masks, public Google Cloud bucket. Bonafilia et al., CVPR-W 2020.

Usage:
    python scripts/fetch_samples.py              # everything
    python scripts/fetch_samples.py --skip-sen1floods11
    python scripts/fetch_samples.py --india-chips 6
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "samples"
RAW = "https://raw.githubusercontent.com"
LEVIR = f"{RAW}/justchenhao/BIT_CD/master/samples"
LEVIR_IDS = ["test_102_0512_0000", "test_121_0768_0256", "test_2_0000_0000", "test_55_0256_0000", "test_77_0512_0256"]
GCS = "https://storage.googleapis.com"
S1F_PREFIX = "v1.1/data/flood_events/HandLabeled"


def fetch(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as r, dest.open("wb") as f:
            f.write(r.read())
        print(f"  ✓ {dest.relative_to(ROOT)}")
        return True
    except Exception as e:  # keep going; one missing source should not block the rest
        print(f"  ✗ {url}: {e}", file=sys.stderr)
        dest.unlink(missing_ok=True)
        return False


def landsat() -> list[dict]:
    print("Landsat 7 RGB (rasterio test data)")
    if not fetch(f"{RAW}/rasterio/rasterio/main/tests/data/RGB.byte.tif", SAMPLES / "landsat" / "landsat7_rgb.tif"):
        return []
    return [{
        "id": "landsat-coast", "title": "Landsat 7 — coastal scene",
        "location": "Florida Keys / Bahamas coast (UTM 18N)", "source": "USGS Landsat 7 via rasterio test data",
        "description": "Georeferenced 30 m RGB scene with open sea, reef shallows and islands. Try water queries.",
        "suggested_queries": ["How much of the scene is water?", "Locate water bodies", "What is the land cover?"],
        "files": [{"file": "landsat/landsat7_rgb.tif", "role": "primary", "modality": "optical",
                   "name": "landsat7_rgb.tif"}],
    }]


def levir() -> list[dict]:
    print("LEVIR-CD change pairs (BIT_CD samples)")
    items = []
    for i, n in enumerate(LEVIR_IDS):
        ok = all(fetch(f"{LEVIR}/{d}/{n}.png", SAMPLES / "levir" / d / f"{n}.png") for d in ("A", "B", "label"))
        if not ok:
            continue
        items.append({
            "id": f"levir-{i + 1}", "title": f"LEVIR-CD urban growth pair {i + 1}",
            "location": "Texas, USA (0.5 m aerial)", "source": "LEVIR-CD (Chen & Shi 2020), academic use",
            "description": "Before/after imagery (2002–2018) with new buildings. Ground-truth change mask in samples/levir/label.",
            "suggested_queries": ["What changed between the two dates?", "How much new built-up area appeared?",
                                  "Detect change"],
            "files": [
                {"file": f"levir/A/{n}.png", "role": "primary", "modality": "optical", "gsd_m": 0.5,
                 "name": f"{n}_before.png", "date": "before"},
                {"file": f"levir/B/{n}.png", "role": "comparison", "modality": "optical", "gsd_m": 0.5,
                 "name": f"{n}_after.png", "date": "after"},
            ],
            "labels": f"levir/label/{n}.png",
        })
    return items


def sen1floods11(n_chips: int) -> list[dict]:
    print("Sen1Floods11 — India flood chips (Sentinel-1 + Sentinel-2)")
    try:
        import numpy as np
        import rasterio
    except ImportError:
        print("  ✗ needs numpy + rasterio (pip install -r requirements.txt)", file=sys.stderr)
        return []
    list_url = f"{GCS}/storage/v1/b/sen1floods11/o?prefix={S1F_PREFIX}/S2Hand/India&fields=items(name)"
    try:
        with urllib.request.urlopen(list_url, timeout=60) as r:
            names = [o["name"] for o in json.load(r).get("items", [])]
    except Exception as e:
        print(f"  ✗ could not list bucket: {e}", file=sys.stderr)
        return []
    chips = sorted({Path(nm).name.replace("_S2Hand.tif", "") for nm in names})[:n_chips]
    items = []
    out = SAMPLES / "sen1floods11"
    for chip in chips:
        raw = {k: out / "raw" / f"{chip}_{k}.tif" for k in ("S1Hand", "S2Hand", "LabelHand")}
        if not all(fetch(f"{GCS}/sen1floods11/{S1F_PREFIX}/{k}/{chip}_{k}.tif", p) for k, p in raw.items()):
            continue
        opt, sar = out / f"{chip}_optical.tif", out / f"{chip}_sar_vv.tif"
        # S2Hand has 13 bands (B1..B12, incl. B8A). Keep R,G,B,NIR = B4,B3,B2,B8.
        with rasterio.open(raw["S2Hand"]) as src:
            prof = src.profile | {"count": 4, "dtype": "float32", "compress": "deflate"}
            data = src.read([4, 3, 2, 8]).astype("float32")
            with rasterio.open(opt, "w", **prof) as dst:
                dst.write(data)
                for i, d in enumerate(["R", "G", "B", "NIR"], 1):
                    dst.set_band_description(i, d)
            bounds = src.bounds
            crs = src.crs
        # S1Hand has 2 bands (VV, VH) in dB. Keep VV.
        with rasterio.open(raw["S1Hand"]) as src:
            prof = src.profile | {"count": 1, "dtype": "float32", "compress": "deflate"}
            vv = src.read(1).astype("float32")
            vv = np.where(np.isfinite(vv), vv, -30.0)
            with rasterio.open(sar, "w", **prof) as dst:
                dst.write(vv, 1)
                dst.set_band_description(1, "VV")
        from rasterio.warp import transform_bounds

        wgs = [round(v, 5) for v in transform_bounds(crs, "EPSG:4326", *bounds)]
        items.append({
            "id": f"s1f-{chip.lower()}", "title": f"India flood — {chip}",
            "location": f"India ({wgs[1]:.2f}°N, {wgs[0]:.2f}°E)",
            "source": "Sen1Floods11 (Bonafilia et al. 2020), CC BY 4.0",
            "description": "Co-registered Sentinel-2 optical and Sentinel-1 VV SAR at 10 m during a flood event, "
                           "with hand-labelled water in samples/sen1floods11/raw.",
            "suggested_queries": ["Fuse optical and SAR to map flood water", "How much of the scene is water?",
                                  "Locate water bodies in the north"],
            "files": [
                {"file": f"sen1floods11/{opt.name}", "role": "primary", "modality": "optical", "name": opt.name},
                {"file": f"sen1floods11/{sar.name}", "role": "comparison", "modality": "sar", "name": sar.name},
            ],
            "labels": f"sen1floods11/raw/{raw['LabelHand'].name}",
        })
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-sen1floods11", action="store_true")
    ap.add_argument("--india-chips", type=int, default=4)
    args = ap.parse_args()
    SAMPLES.mkdir(exist_ok=True)
    catalog = landsat() + levir()
    if not args.skip_sen1floods11:
        catalog += sen1floods11(args.india_chips)
    (SAMPLES / "catalog.json").write_text(json.dumps(catalog, indent=2))
    print(f"\nWrote {len(catalog)} demo scenes to samples/catalog.json")


if __name__ == "__main__":
    main()
