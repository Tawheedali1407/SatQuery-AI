"""Optional Google Gemini backend for open-ended questions.

Free tier: create a key at https://aistudio.google.com/apikey and set
SATQUERY_GEMINI_API_KEY (or GEMINI_API_KEY) in backend/.env. No extra Python
packages are needed; this uses the public REST endpoint.

Privacy: the scene preview and the question are sent to Google. Leave the key
unset for fully offline / on-premises use (ground-station mode).

To limit hallucination, Gemini is given the pixel measurements SatQuery already
computed and told to use them for any numbers, so it describes the scene but
does not invent percentages or areas.
"""
from __future__ import annotations

import base64
import io
import json
import os
import urllib.request

import numpy as np
from PIL import Image

from ..config import settings

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

PROMPT = """You are a remote-sensing image analyst. Answer the analyst's question about the attached
{modality} satellite/aerial image in at most 3 sentences.

Measured facts from pixel analysis (use these for any number you give; do not invent other numbers):
{facts}

Rules: describe only what is visible; if the image cannot answer the question, say so plainly.

Question: {question}"""


def _key() -> str:
    return settings.gemini_api_key or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")


def available() -> bool:
    return settings.vqa_backend in ("auto", "gemini") and bool(_key())


def _facts(scene, lc: dict | None) -> str:
    lines = [f"- image size {scene.shape[1]}x{scene.shape[0]} px"
             + (f", {scene.gsd_m:.1f} m per pixel" if scene.gsd_m else ", ground resolution unknown")]
    if scene.meta.get("location"):
        lines.append(f"- location: {scene.meta['location']}")
    if lc:
        fr = ", ".join(f"{k.replace('_', ' ')} {v * 100:.1f}%" for k, v in lc["fractions"].items() if v >= 0.005)
        lines.append(f"- land-cover fractions ({lc['index_mode']} indices): {fr}")
    return "\n".join(lines)


def answer(scene, question: str, lc: dict | None = None, timeout: float = 30) -> str:
    img = Image.fromarray((np.clip(scene.rgb(), 0, 1) * 255).astype(np.uint8))
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    body = {
        "contents": [{"parts": [
            {"text": PROMPT.format(modality="SAR (grey = backscatter)" if scene.modality == "sar" else "optical",
                                   facts=_facts(scene, lc), question=question)},
            {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(buf.getvalue()).decode()}},
        ]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 256},
    }
    req = urllib.request.Request(
        ENDPOINT.format(model=settings.gemini_model),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": _key()},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = " ".join(p.get("text", "") for p in parts).strip()
    if not text:
        raise RuntimeError(f"empty response: {str(data)[:200]}")
    return text + f" (Open-ended answer from Google {settings.gemini_model}, grounded on SatQuery's measurements.)"
