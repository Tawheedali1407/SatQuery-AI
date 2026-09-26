"""Tests for the optional ML backends. Deep-model tests skip automatically when
torch or the ChangeFormer weights are not installed (e.g. in CI)."""
import io
import json

import numpy as np
import pytest

from app.config import settings
from app.ml import changeformer, gemini
from app.tools import vqa
from app.tools.change import detect_change, pick_backend
from app.tools.landcover import classify

needs_cf = pytest.mark.skipif(not changeformer.available(), reason="ChangeFormer weights not installed")


def test_backend_selection_respects_resolution(optical_before, monkeypatch):
    monkeypatch.setattr(changeformer, "available", lambda: True)
    optical_before.gsd_m = 10.0  # Sentinel-2 class: out of the 0.5 m training domain
    assert pick_backend(optical_before) == "classical"
    optical_before.gsd_m = 0.5
    assert pick_backend(optical_before) == "changeformer"
    assert pick_backend(optical_before, "classical") == "classical"


def test_backend_falls_back_when_not_installed(optical_before, monkeypatch):
    monkeypatch.setattr(changeformer, "available", lambda: False)
    optical_before.gsd_m = 0.5
    assert pick_backend(optical_before) == "classical"


@needs_cf
def test_changeformer_tiled_inference_any_size():
    rng = np.random.default_rng(0)
    a = rng.random((300, 410, 3)).astype(np.float32)  # not a multiple of 256 -> exercises overlap + edges
    p = changeformer.predict(a, a.copy())
    assert p.shape == (300, 410)
    assert 0 <= p.min() and p.max() <= 1
    assert p.mean() < 0.2  # identical images -> mostly "no change"


@needs_cf
def test_changeformer_route(optical_before, optical_after):
    optical_before.gsd_m = optical_after.gsd_m = 0.5
    ch = detect_change(optical_before, optical_after)
    assert ch["backend"] == "changeformer" and ch["mask"].shape == optical_before.shape


class _FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_gemini_request_is_grounded(optical_before, monkeypatch):
    sent = {}

    def fake_urlopen(req, timeout=0):
        sent["url"], sent["headers"], sent["body"] = req.full_url, dict(req.headers), json.loads(req.data)
        return _FakeResp(json.dumps({"candidates": [{"content": {"parts": [{"text": "A lake beside fields."}]}}]}).encode())

    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "vqa_backend", "auto")
    monkeypatch.setattr(gemini.urllib.request, "urlopen", fake_urlopen)
    lc = classify(optical_before)
    text, backend = vqa.open_answer(optical_before, "Describe this place", lc)
    assert backend == "gemini" and text.startswith("A lake beside fields.")
    assert settings.gemini_model in sent["url"]
    assert sent["headers"]["X-goog-api-key"] == "test-key"
    parts = sent["body"]["contents"][0]["parts"]
    assert "water" in parts[0]["text"] and "%" in parts[0]["text"]  # measured facts included
    assert parts[1]["inline_data"]["mime_type"] == "image/png"


def test_gemini_failure_falls_back(optical_before, monkeypatch):
    def boom(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "vqa_backend", "gemini")  # also disables BLIP
    monkeypatch.setattr(gemini.urllib.request, "urlopen", boom)
    assert vqa.open_answer(optical_before, "Describe this place", classify(optical_before)) is None


def test_no_key_means_no_gemini(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert not gemini.available()
