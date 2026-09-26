import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def _png(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray((arr * 255).astype(np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["mode"] == "live"


def test_upload_query_delete_roundtrip():
    img = np.full((64, 64, 3), (0.25, 0.5, 0.2), np.float32)
    img[10:30, 10:30] = (0.05, 0.15, 0.35)
    up = client.post("/api/scenes", files={"file": ("tile.png", _png(img), "image/png")}, data={"gsd_m": "10"})
    assert up.status_code == 200, up.text
    sid = up.json()["id"]
    assert client.get(f"/api/scenes/{sid}/preview").headers["content-type"] == "image/png"

    q = client.post("/api/query", json={"query": "Is there water?", "primary_id": sid})
    assert q.status_code == 200
    body = q.json()
    assert body["answer"].startswith("Yes") and body["evidence"]["images"]

    assert client.delete(f"/api/scenes/{sid}").status_code == 200
    assert client.post("/api/query", json={"query": "water?", "primary_id": sid}).status_code == 404


def test_rejects_bad_file_type():
    r = client.post("/api/scenes", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415
