"""Scene loading: GeoTIFF / PNG / JPG -> normalised float32 array + metadata.

Large rasters are never read at full resolution. With rasterio we ask GDAL for a
decimated read (it uses internal overviews when present), so a 10k x 10k scene
is analysed at `max_analysis_px` on the longest side and the ground sampling
distance (GSD) is scaled accordingly.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

try:  # rasterio is optional so the core still runs where GDAL wheels are unavailable
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds

    HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    HAS_RASTERIO = False

GEOTIFF_EXT = {".tif", ".tiff"}
IMAGE_EXT = {".png", ".jpg", ".jpeg"}
SUPPORTED_EXT = GEOTIFF_EXT | IMAGE_EXT


@dataclass
class Scene:
    id: str
    name: str
    path: Path
    modality: str  # "optical" | "sar"
    data: np.ndarray  # (H, W, C) float32 in [0, 1]; SAR is stored as dB rescaled
    bands: list[str]  # e.g. ["R", "G", "B"] or ["R", "G", "B", "NIR"] or ["VV"]
    gsd_m: float | None  # ground sampling distance of `data`, metres per pixel
    crs: str | None = None
    bounds_wgs84: tuple[float, float, float, float] | None = None  # (W, S, E, N)
    original_shape: tuple[int, int] = (0, 0)
    size_bytes: int = 0
    sar_db: np.ndarray | None = None  # raw backscatter in dB for SAR scenes
    valid: np.ndarray | None = None  # False for no-data pixels (scene edges, masked areas)
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[:2]

    def band(self, name: str) -> np.ndarray | None:
        return self.data[..., self.bands.index(name)] if name in self.bands else None

    def rgb(self) -> np.ndarray:
        """Display-ready RGB (H, W, 3) float in [0, 1]."""
        if self.modality == "sar":
            g = self.data[..., 0]
            return np.dstack([g, g, g])
        return np.dstack([self.band("R"), self.band("G"), self.band("B")])

    def summary(self) -> dict:
        h, w = self.shape
        return {
            "id": self.id,
            "name": self.name,
            "modality": self.modality,
            "bands": self.bands,
            "width": w,
            "height": h,
            "original_width": self.original_shape[1],
            "original_height": self.original_shape[0],
            "gsd_m": self.gsd_m,
            "crs": self.crs,
            "bounds_wgs84": self.bounds_wgs84,
            "size_bytes": self.size_bytes,
            **{k: v for k, v in self.meta.items() if k in ("location", "date", "source", "sample")},
        }


def _stretch(arr: np.ndarray, mask: np.ndarray | None = None, lo_pct: float = 2, hi_pct: float = 98) -> np.ndarray:
    """Per-band percentile stretch to [0, 1] (robust to 8/16-bit and reflectance data).
    Percentiles are taken over valid pixels only so black no-data borders do not skew contrast."""
    out = np.empty(arr.shape, dtype=np.float32)
    for i in range(arr.shape[-1]):
        b = arr[..., i].astype(np.float32)
        valid = b[np.isfinite(b) & (mask if mask is not None else True)]
        lo, hi = (np.percentile(valid, [lo_pct, hi_pct]) if valid.size else (0.0, 1.0))
        out[..., i] = np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.nan_to_num(out)


def _guess_bands(count: int, modality: str) -> list[str]:
    if modality == "sar":
        return ["VV", "VH"][:count] if count <= 2 else [f"SAR{i}" for i in range(count)]
    if count == 1:
        return ["PAN"]
    if count >= 4:
        # Convention used by the bundled fetch script: R, G, B, NIR (+ extras)
        return ["R", "G", "B", "NIR"] + [f"B{i}" for i in range(5, count + 1)]
    return ["R", "G", "B"]


def load_scene(
    path: str | Path,
    *,
    name: str | None = None,
    modality: str | None = None,
    gsd_m: float | None = None,
    bands: list[str] | None = None,
    max_px: int = 2048,
    meta: dict | None = None,
) -> Scene:
    """Load a raster from disk. `modality` is auto-detected when omitted
    (single-band float/dB rasters or names containing 'sar'/'vv' are treated as SAR)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"Unsupported file type '{ext}'. Use GeoTIFF, PNG or JPG.")

    crs = bounds = None
    if ext in GEOTIFF_EXT and HAS_RASTERIO:
        with rasterio.open(path) as ds:
            h0, w0 = ds.height, ds.width
            scale = max(1.0, max(h0, w0) / max_px)
            out_h, out_w = int(round(h0 / scale)), int(round(w0 / scale))
            raw = ds.read(out_shape=(ds.count, out_h, out_w), resampling=Resampling.average, masked=True)
            raw = np.moveaxis(raw.astype(np.float32).filled(np.nan), 0, -1)
            if ds.crs:
                crs = ds.crs.to_string()
                try:
                    bounds = tuple(round(v, 6) for v in transform_bounds(ds.crs, "EPSG:4326", *ds.bounds))
                except Exception:
                    bounds = None
                if gsd_m is None and ds.crs.is_projected:
                    gsd_m = float(abs(ds.transform.a))
            if gsd_m is not None:
                gsd_m = gsd_m * (w0 / out_w)
            band_desc = [d for d in ds.descriptions if d]
            if bands is None and len(band_desc) == ds.count:
                bands = [d.upper() for d in band_desc]
            dtype_float = np.issubdtype(np.dtype(ds.dtypes[0]), np.floating)
    else:
        img = Image.open(path)
        h0, w0 = img.height, img.width
        scale = max(1.0, max(h0, w0) / max_px)
        if scale > 1:
            img = img.resize((int(w0 / scale), int(h0 / scale)), Image.Resampling.BOX)
            if gsd_m is not None:
                gsd_m *= scale
        if img.mode not in ("L", "RGB", "RGBA", "I;16", "F"):
            img = img.convert("RGB")
        raw = np.asarray(img).astype(np.float32)
        if raw.ndim == 2:
            raw = raw[..., None]
        if raw.shape[-1] == 4:  # drop alpha
            raw = raw[..., :3]
        dtype_float = False

    valid = np.isfinite(raw).all(-1) & (np.nan_to_num(raw) != 0).any(-1)
    if valid.mean() < 0.01:  # an all-zero band layout is data, not no-data
        valid = np.ones(raw.shape[:2], bool)

    lowered = path.stem.lower()
    if modality is None:
        looks_sar = any(t in lowered for t in ("sar", "_vv", "_vh", "s1", "risat", "eos04"))
        modality = "sar" if (looks_sar or (raw.shape[-1] <= 2 and dtype_float)) else "optical"

    sar_db = None
    if modality == "sar":
        band0 = raw[..., 0]
        finite = band0[np.isfinite(band0)]
        # Linear backscatter (sigma0 ~ 0..1) -> dB. Data already in dB has negative values.
        if dtype_float and finite.size and np.nanmin(finite) >= 0 and np.nanpercentile(finite, 99) < 5:
            band0 = 10 * np.log10(np.clip(band0, 1e-5, None))
        elif not dtype_float:  # 8-bit SAR quicklook: map DN 0..255 to roughly -30..0 dB
            band0 = band0 / 255.0 * 30.0 - 30.0
        sar_db = np.nan_to_num(band0, nan=-30.0).astype(np.float32)
        data = np.clip((sar_db + 30.0) / 30.0, 0, 1)[..., None].astype(np.float32)
        bands = bands[:1] if bands else ["VV"]
    else:
        data = _stretch(raw, valid)
        bands = bands or _guess_bands(data.shape[-1], modality)
        if data.shape[-1] == 1:  # panchromatic -> fake RGB for display, keep single band semantics
            data = np.repeat(data, 3, axis=-1)
            bands = ["R", "G", "B"]

    return Scene(
        id=uuid.uuid4().hex[:10],
        name=name or path.name,
        path=path,
        modality=modality,
        data=data,
        bands=list(bands),
        gsd_m=gsd_m,
        crs=crs,
        bounds_wgs84=bounds,
        original_shape=(h0, w0),
        size_bytes=path.stat().st_size,
        sar_db=sar_db,
        valid=valid,
        meta=meta or {},
    )
