"""In-memory scene registry for the session (scenes live on disk under data_dir)."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from .config import settings
from .raster import Scene, load_scene


class SceneStore:
    def __init__(self):
        self._scenes: dict[str, Scene] = {}
        self._lock = threading.Lock()

    def add(self, scene: Scene) -> Scene:
        with self._lock:
            self._scenes[scene.id] = scene
        return scene

    def get(self, scene_id: str) -> Scene | None:
        return self._scenes.get(scene_id)

    def remove(self, scene_id: str) -> bool:
        with self._lock:
            return self._scenes.pop(scene_id, None) is not None

    def list(self) -> list[Scene]:
        return list(self._scenes.values())

    def find_by_path(self, path: Path) -> Scene | None:
        return next((s for s in self._scenes.values() if s.path == path), None)


store = SceneStore()


def sample_catalog() -> list[dict]:
    """Demo scenes described in samples/catalog.json (created by scripts/fetch_samples.py)."""
    cat = settings.samples_dir / "catalog.json"
    if not cat.exists():
        return []
    items = json.loads(cat.read_text())
    for it in items:
        it["available"] = all((settings.samples_dir / f["file"]).exists() for f in it["files"])
    return items


def load_sample(sample_id: str) -> list[Scene]:
    item = next((i for i in sample_catalog() if i["id"] == sample_id), None)
    if item is None:
        raise KeyError(sample_id)
    scenes = []
    for f in item["files"]:
        path = settings.samples_dir / f["file"]
        existing = store.find_by_path(path)
        if existing:
            scenes.append(existing)
            continue
        sc = load_scene(path, name=f.get("name"), modality=f.get("modality"), gsd_m=f.get("gsd_m"),
                        max_px=settings.max_analysis_px,
                        meta={"location": item.get("location"), "date": f.get("date"), "source": item.get("source"),
                              "sample": sample_id, "role": f.get("role")})
        if f.get("bounds_wgs84") and sc.bounds_wgs84 is None:
            sc.bounds_wgs84 = tuple(f["bounds_wgs84"])
        scenes.append(store.add(sc))
    return scenes
