"""FastAPI service for SatQuery AI."""
from __future__ import annotations

import io
import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from . import __version__
from .agent import run_query
from .config import settings
from .raster import HAS_RASTERIO, SUPPORTED_EXT, load_scene
from .store import load_sample, sample_catalog, store
from .tools.vqa import blip_available

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("satquery")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    shutil.rmtree(settings.data_dir, ignore_errors=True)  # uploaded scenes are session-local


app = FastAPI(title="SatQuery AI", version=__version__, lifespan=lifespan,
              description="Agentic vision-language assistant for optical / SAR remote sensing analysis.")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])


class QueryIn(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    primary_id: str
    comparison_id: str | None = None


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": __version__,
        "mode": "live",  # every answer is computed from pixels; there is no mock path
        "geotiff_support": HAS_RASTERIO,
        "vqa_backend": "rules + blip" if blip_available() else "rules",
        "scenes": len(store.list()),
    }


@app.get("/api/scenes")
def list_scenes():
    return [s.summary() | {"role": s.meta.get("role")} for s in store.list()]


@app.post("/api/scenes")
async def upload_scene(file: UploadFile = File(...), modality: str | None = Form(None),
                       gsd_m: float | None = Form(None)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Upload GeoTIFF, PNG or JPG.")
    if modality not in (None, "", "optical", "sar"):
        raise HTTPException(422, "modality must be 'optical' or 'sar'")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.data_dir / f"{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"
    size = 0
    with dest.open("wb") as out:  # stream to disk; never hold the whole upload in memory
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > settings.max_upload_mb * 1024 * 1024:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB.")
            out.write(chunk)
    try:
        scene = load_scene(dest, name=file.filename, modality=modality or None, gsd_m=gsd_m,
                           max_px=settings.max_analysis_px)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, f"Could not read raster: {e}") from e
    store.add(scene)
    log.info("registered %s (%s, %sx%s)", scene.name, scene.modality, *scene.shape[::-1])
    return scene.summary()


@app.delete("/api/scenes/{scene_id}")
def delete_scene(scene_id: str):
    if not store.remove(scene_id):
        raise HTTPException(404, "Scene not found")
    return {"deleted": scene_id}


@app.get("/api/scenes/{scene_id}/preview")
def preview(scene_id: str, size: int = 512):
    s = store.get(scene_id)
    if s is None:
        raise HTTPException(404, "Scene not found")
    img = Image.fromarray((np.clip(s.rgb(), 0, 1) * 255).astype(np.uint8))
    img.thumbnail((min(size, 1024),) * 2)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/samples")
def samples():
    return sample_catalog()


@app.post("/api/samples/{sample_id}/load")
def load_sample_route(sample_id: str):
    try:
        scenes = load_sample(sample_id)
    except KeyError:
        raise HTTPException(404, "Unknown sample") from None
    except FileNotFoundError:
        raise HTTPException(409, "Sample files missing — run `python scripts/fetch_samples.py`.") from None
    return [s.summary() | {"role": s.meta.get("role")} for s in scenes]


@app.post("/api/query")
def query(q: QueryIn):
    primary = store.get(q.primary_id)
    if primary is None:
        raise HTTPException(404, "Primary scene not found — register a scene first.")
    comparison = store.get(q.comparison_id) if q.comparison_id else None
    if q.comparison_id and comparison is None:
        raise HTTPException(404, "Comparison scene not found.")
    try:
        return run_query(q.query, primary, comparison)
    except ValueError as e:  # tool-level validation errors are user-facing
        raise HTTPException(422, str(e)) from e


# Serve the built React app when present (single-container deployment).
_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _dist.exists():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = _dist / path
        return FileResponse(f if path and f.is_file() else _dist / "index.html")

