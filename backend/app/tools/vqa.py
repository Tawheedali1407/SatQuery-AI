"""Visual question answering.

Two backends:

* ``rules`` (default, always available): parses the question type (presence,
  proportion, count, dominant class, comparison) and answers from the measured
  land-cover statistics. Every answer is traceable to a pixel mask.
* ``blip`` (optional): a general VLM (BLIP-VQA) for open-ended questions the
  rules cannot parse. Enabled automatically when torch + transformers are
  installed (``pip install -r requirements-ml.txt``). It runs on CPU; a GPU
  just makes it faster.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

import numpy as np

from ..config import settings
from ..raster import Scene

log = logging.getLogger(__name__)

PRETTY = {"water": "water", "vegetation": "vegetation", "built_up": "built-up area",
          "bare_soil": "bare soil", "cloud": "cloud", "other": "unclassified surface"}


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def fmt_area(a: dict) -> str:
    if "km2" in a:
        return f"{a['km2']:.3f} km²" if a["km2"] >= 0.01 else f"{a['m2']:.0f} m²"
    return f"{a['pixels']:,} px"


def rule_answer(question: str, target: str | None, lc: dict, water: dict | None) -> dict | None:
    q = question.lower()
    fr = lc["fractions"]
    classes = [c for c in fr if c != "other"]

    two = [c for c in classes if c in _mentioned(q)]
    if len(two) >= 2 and re.search(r"\bmore\b|\bless\b|\bcompare|\bor\b|\bthan\b", q):
        a, b = two[:2]
        big = a if fr[a] >= fr[b] else b
        return {"answer": f"There is more {PRETTY[big]} ({pct(fr[big])}) than "
                          f"{PRETTY[b if big == a else a]} ({pct(fr[b if big == a else a])}).",
                "type": "comparison"}

    if re.search(r"how many|number of|count", q):
        if target in (None, "water") and water is not None:
            return {"answer": f"I count {water['count']} distinct water bod{'y' if water['count'] == 1 else 'ies'} "
                              f"covering {fmt_area(water['area'])} ({pct(water['fraction'])} of the scene).",
                    "type": "count"}
        if target:
            from scipy import ndimage as ndi

            n = ndi.label(lc["masks"][target])[1]
            return {"answer": f"I find {n} separate {PRETTY[target]} patches.", "type": "count"}

    if re.search(r"how much|what (percent|percentage|fraction|proportion)|percent|coverage|area of|how large", q) and target:
        return {"answer": f"{PRETTY[target].capitalize()} covers {pct(fr[target])} of the scene "
                          f"({fmt_area(lc['areas'][target])}).", "type": "proportion"}

    if re.search(r"^(is|are|does|do|can|any)\b|is there|are there", q) and target:
        present = fr[target] >= 0.005
        return {"answer": (f"Yes — {PRETTY[target]} is present, covering {pct(fr[target])} of the scene."
                           if present else f"No significant {PRETTY[target]} detected (under 0.5% of the scene)."),
                "type": "presence"}

    if re.search(r"dominant|main|mostly|primary|majority|what (kind|type)|land ?(cover|use)|what is (in|this)", q):
        ranked = sorted(classes, key=lambda c: fr[c], reverse=True)
        top = ranked[0]
        rest = ", ".join(f"{PRETTY[c]} {pct(fr[c])}" for c in ranked[1:] if fr[c] > 0.005)
        land = "rural / natural" if fr["vegetation"] + fr["bare_soil"] > fr["built_up"] * 2 else "urbanised"
        return {"answer": f"The scene is dominated by {PRETTY[top]} ({pct(fr[top])}); also {rest or 'little else'}. "
                          f"Overall it looks {land}.", "type": "scene"}
    return None


def _mentioned(q: str) -> set[str]:
    from ..agent import find_targets

    return set(find_targets(q))


@lru_cache(maxsize=1)
def _blip():
    from transformers import BlipForQuestionAnswering, BlipProcessor

    proc = BlipProcessor.from_pretrained(settings.blip_model)
    model = BlipForQuestionAnswering.from_pretrained(settings.blip_model)
    model.eval()
    return proc, model


def blip_available() -> bool:
    if settings.vqa_backend not in ("auto", "blip"):
        return False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def blip_answer(scene: Scene, question: str) -> str:
    import torch
    from PIL import Image

    proc, model = _blip()
    img = Image.fromarray((scene.rgb() * 255).astype(np.uint8))
    inputs = proc(img, question, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20)
    return proc.decode(out[0], skip_special_tokens=True)


def vqa_backends() -> str:
    from ..ml import gemini

    chain = [n for n, ok in (("gemini", gemini.available()), ("blip", blip_available())) if ok]
    return " + ".join(["rules", *chain])


def open_answer(scene: Scene, question: str, lc: dict) -> tuple[str, str] | None:
    """Open-ended question the rules could not parse: try Gemini, then BLIP. Returns (text, backend)."""
    from ..ml import gemini

    if gemini.available():
        try:
            return gemini.answer(scene, question, lc), "gemini"
        except Exception as e:  # network down, quota, bad key -> fall through to local model
            log.warning("Gemini failed (%s); falling back", e)
    if blip_available():
        text = blip_answer(scene, question)
        return f"{text.capitalize()}. (Short answer from BLIP-VQA, a general vision-language model — verify visually.)", "blip"
    return None
