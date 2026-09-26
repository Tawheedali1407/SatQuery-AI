"""Test fixtures.

Unit tests use small *synthetic* scenes with known geometry (a lake, a field,
a new building) so assertions are exact and CI needs no downloads. Real-data
accuracy is measured separately by scripts/evaluate.py against labelled samples.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.raster import load_scene  # noqa: E402

H = W = 128


def _base() -> np.ndarray:
    rng = np.random.default_rng(0)
    img = np.zeros((H, W, 3), np.float32)
    img[:] = (0.55, 0.45, 0.35)  # bare soil background
    img[:, 64:] = (0.25, 0.50, 0.20)  # vegetation in the east half
    yy, xx = np.mgrid[:H, :W]
    lake = (yy - 32) ** 2 + (xx - 32) ** 2 < 18 ** 2  # lake in the north-west
    img[lake] = (0.05, 0.15, 0.35)
    return np.clip(img + rng.normal(0, 0.015, img.shape), 0, 1)


def _save_png(arr: np.ndarray, path: Path) -> Path:
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)
    return path


@pytest.fixture
def optical_before(tmp_path):
    return load_scene(_save_png(_base(), tmp_path / "before.png"), gsd_m=10.0)


@pytest.fixture
def optical_after(tmp_path):
    img = _base()
    img[90:110, 20:50] = (0.85, 0.82, 0.80)  # new bright building on bare soil (south-west)
    yy, xx = np.mgrid[:H, :W]
    img[((yy - 100) ** 2 + (xx - 100) ** 2) < 12 ** 2] = (0.05, 0.15, 0.35)  # new flood patch in the field
    return load_scene(_save_png(img, tmp_path / "after.png"), gsd_m=10.0)


@pytest.fixture
def sar_scene(tmp_path):
    """Single-band SAR in dB (GeoTIFF, float32): lake is dark (-24 dB), land -8 dB, with speckle."""
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(1)
    db = np.full((H, W), -8.0, np.float32)
    yy, xx = np.mgrid[:H, :W]
    db[((yy - 32) ** 2 + (xx - 32) ** 2) < 18 ** 2] = -24.0
    db[((yy - 100) ** 2 + (xx - 100) ** 2) < 12 ** 2] = -23.0  # flood also visible to SAR
    # multiplicative speckle (gamma, 4 looks) applied in linear power
    lin = 10 ** (db / 10) * rng.gamma(4, 1 / 4, db.shape)
    db = (10 * np.log10(lin)).astype(np.float32)
    path = tmp_path / "scene_sar_vv.tif"
    with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32",
                       crs="EPSG:32644", transform=from_origin(500000, 1900000, 10, 10)) as dst:
        dst.write(db, 1)
    return load_scene(path)
