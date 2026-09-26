"""The orchestrator: classify the query's intent, route it to a specialist
tool, and assemble an evidence-backed answer with a step-by-step trace.

The v1 intent classifier is rule-based (keyword/pattern scoring plus scene
context). It is deterministic, fast and explainable, and each decision is
written to the trace so an analyst can see *why* a tool was chosen. The
classifier sits behind `classify_intent()`, so a small local LLM doing function
calling can replace it later without touching the tools.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from . import evidence as ev
from .raster import Scene
from .tools import change as change_t
from .tools import fusion as fusion_t
from .tools import grounding as ground_t
from .tools import landcover as lc_t
from .tools import vqa as vqa_t
from .tools.common import area as common_area
from .tools.common import regions as regions_of
from .tools.vqa import PRETTY, fmt_area, pct

SYNONYMS = {
    "water": ["water", "river", "lake", "pond", "flood", "inundat", "reservoir", "sea", "canal", "wetland", "tank"],
    "vegetation": ["vegetation", "tree", "forest", "green", "crop", "farm", "agricultur", "field", "grass", "plant"],
    "built_up": ["building", "urban", "built", "settlement", "road", "construction", "house", "city", "town",
                 "infrastructure", "roof", "village"],
    "bare_soil": ["bare", "soil", "barren", "sand", "desert", "exposed"],
}

INTENT_PATTERNS = {
    "fusion": [r"\bfus(e|ion)\b", r"\bcombine\b", r"optical\s*(\+|and|&|with)\s*sar", r"sar\s*(\+|and|&|with)\s*optical",
               r"\bcloud", r"multi-?sensor", r"both sensors"],
    "change": [r"\bchang", r"\bbefore\b", r"\bafter\b", r"\bnew\b", r"\bincreas", r"\bdecreas", r"\bgrow", r"\bgrew\b",
               r"\bexpan", r"\bdifferen", r"\bcompare\b", r"\bloss\b", r"\blost\b", r"\bgained?\b", r"\bdamage",
               r"\bover time\b", r"\bsince\b"],
    "grounding": [r"\bwhere\b", r"\blocat", r"\bfind\b", r"\bshow\b", r"\bmark\b", r"\bhighlight", r"\bpoint",
                  r"\bwhich part", r"\blargest\b", r"\bbiggest\b", r"\bbounding"],
    "landcover": [r"land ?(cover|use)", r"\bclassif", r"\bcomposition", r"\btypes?\b", r"\bsegment",
                  r"\bbreakdown", r"\bmap (the|all)"],
    "vqa": [r"^(is|are|does|do|how|what|which|can|any)\b", r"\?$", r"how many", r"how much", r"percent"],
}


def find_targets(text: str) -> list[str]:
    t = text.lower()
    hits = []
    for cls, words in SYNONYMS.items():
        pos = [t.find(w) for w in words if w in t]
        if pos:
            hits.append((min(pos), cls))
    return [c for _, c in sorted(hits)]


@dataclass
class Trace:
    steps: list[dict] = field(default_factory=list)
    _t: float = field(default_factory=time.perf_counter)

    def add(self, step: str, detail: str, tool: str | None = None):
        now = time.perf_counter()
        self.steps.append({"step": step, "tool": tool, "detail": detail, "ms": round((now - self._t) * 1000, 1)})
        self._t = now


def classify_intent(query: str, primary: Scene, comparison: Scene | None) -> dict:
    q = query.lower().strip()
    scores = {k: sum(bool(re.search(p, q)) for p in pats) for k, pats in INTENT_PATTERNS.items()}
    mixed = comparison is not None and {primary.modality, comparison.modality} == {"optical", "sar"}
    # Scene context shifts the decision, e.g. two dates -> change is plausible.
    if mixed:
        scores["fusion"] += 2
    if comparison is not None and not mixed and scores["change"]:
        scores["change"] += 2  # two dates + any temporal cue -> change wins
    if comparison is None:
        scores["change"] = 0
        scores["fusion"] = 0

    order = ["fusion", "change", "grounding", "landcover", "vqa"]  # tie-break priority
    intent = max(order, key=lambda k: (scores[k], -order.index(k)))
    if scores[intent] == 0:
        intent = "vqa"
    if intent == "change" and comparison is None:
        intent = "vqa"
    total = sum(scores.values()) or 1
    return {
        "intent": intent,
        "scores": scores,
        "confidence": round(scores[intent] / total, 2) if scores[intent] else 0.3,
        "targets": find_targets(q),
        "sector": ground_t.find_sector(q),
        "needs_comparison": comparison is None and bool(re.search(r"chang|before|after|over time|since|grew|grow", q)),
    }


def run_query(query: str, primary: Scene, comparison: Scene | None = None) -> dict:
    trace = Trace()
    trace.add("query", f"received “{query[:120]}” with primary={primary.name} ({primary.modality})"
              + (f", comparison={comparison.name} ({comparison.modality})" if comparison else ""))
    it = classify_intent(query, primary, comparison)
    trace.add("intent", f"intent={it['intent']} (scores {it['scores']}); targets={it['targets'] or '—'}"
              + (f"; sector={it['sector']}" if it["sector"] else ""), tool="intent-classifier@rules-v1")

    warnings: list[str] = []
    if it["needs_comparison"]:
        warnings.append("This looks like a change question, but no comparison scene is registered. "
                        "Add an 'after' scene to run change detection; answering from the primary scene instead.")

    handler = {"fusion": _fusion, "change": _change, "grounding": _grounding, "landcover": _landcover}.get(
        it["intent"], _vqa)
    # Optical-only tools cannot run on a single SAR scene; route to SAR water mapping.
    if primary.modality == "sar" and it["intent"] in ("landcover", "grounding", "vqa") and comparison is None:
        handler = _sar_single
    trace.add("route", f"routed to {handler.__name__.strip('_')} module")
    result = handler(query, primary, comparison, it, trace)
    warnings += result.pop("warnings", [])
    trace.add("evidence", f"assembled {len(result['evidence']['images'])} image(s), "
              f"{len(result['evidence']['regions'])} region(s)")

    return {
        "query": query,
        "intent": it["intent"],
        "intent_confidence": it["confidence"],
        **result,
        "warnings": warnings,
        "trace": trace.steps,
        "total_ms": round(sum(s["ms"] for s in trace.steps), 1),
    }


# ----------------------------------------------------------------- handlers
def _pack(answer, confidence, images, regions=None, stats=None, method="", warnings=None, extra=None):
    return {
        "answer": answer,
        "confidence": confidence,
        "method": method,
        "evidence": {"images": images, "regions": regions or [], "stats": stats or {}},
        "warnings": warnings or [],
        **(extra or {}),
    }


def _landcover(query, scene, _cmp, it, trace):
    lc = lc_t.classify(scene)
    trace.add("tool", f"land cover via {lc['method']}", tool="landcover")
    fr = lc["fractions"]
    ranked = sorted([c for c in fr if c != "other"], key=lambda c: fr[c], reverse=True)
    parts = [f"{PRETTY[c]} {pct(fr[c])}" for c in ranked if fr[c] >= 0.005]
    answer = "Land-cover breakdown: " + ", ".join(parts) + "."
    if fr["other"] > 0.2:
        answer += f" {pct(fr['other'])} could not be confidently assigned (shadow, mixed pixels)."
    return _pack(answer, round(sum(lc["confidence"].values()) / 2, 3),
                 [{"label": "Scene", "src": ev.image(scene.rgb())},
                  {"label": "Land-cover map", "src": ev.class_overlay(scene.rgb(), lc["class_map"]), "legend": "landcover"}],
                 stats={"fractions": fr, "areas": lc["areas"], "thresholds": lc["thresholds"],
                        "index_mode": lc["index_mode"]},
                 method=lc["method"])


def _grounding(query, scene, _cmp, it, trace):
    target = (it["targets"] or ["water"])[0]
    g = ground_t.ground(scene, target, it["sector"])
    trace.add("tool", f"grounded '{target}'" + (f" in {it['sector']}" if it["sector"] else "")
              + f": {len(g['regions'])} region(s)", tool="grounding")
    regs = g["regions"]
    if regs:
        big = regs[0]
        cx, cy = big["centroid_norm"]
        where = ("upper" if cy < 0.4 else "lower" if cy > 0.6 else "middle") + "-" + \
                ("left" if cx < 0.4 else "right" if cx > 0.6 else "centre")
        answer = (f"Found {len(regs)} {PRETTY[target]} region(s)"
                  + (f" in the {it['sector']} of the scene" if it["sector"] else "")
                  + f". The largest ({fmt_area(big['area'])}) is in the {where} part — boxes are drawn on the map.")
    else:
        answer = f"No {PRETTY[target]} found" + (f" in the {it['sector']}" if it["sector"] else "") + "."
    color = ev.COLORS.get(target, (255, 255, 0))
    return _pack(answer, 0.8 if regs else 0.5,
                 [{"label": f"{PRETTY[target].capitalize()} located", "src": ev.overlay(scene.rgb(), [(g["mask"], color)]),
                   "boxes": True}],
                 regions=regs, stats={"fraction": g["fraction"], "target": target, "sector": g["sector"]},
                 method=g["method"])


def _change(query, before, after, it, trace):
    target = next((t for t in it["targets"] if t in ("water", "vegetation")), None)
    images = [{"label": "Before", "src": ev.image(before.rgb())}, {"label": "After", "src": ev.image(after.rgb())}]
    wants_built = "built_up" in it["targets"]
    if target and not (target != "water" and before.modality == "sar"):
        cc = change_t.class_change(before, after, target)
        trace.add("tool", f"from-to change for '{target}': +{cc['gained']['pixels']} / −{cc['lost']['pixels']} px",
                  tool="change-detection/class")
        name = PRETTY[target]
        answer = (f"{name.capitalize()} went from {pct(cc['before_fraction'])} to {pct(cc['after_fraction'])} of the scene "
                  f"({cc['net_change_pct_points']:+.2f} percentage points). Gained {fmt_area(cc['gained'])}, "
                  f"lost {fmt_area(cc['lost'])}.")
        if target == "water" and cc["net_change_pct_points"] > 1:
            answer += " The gain pattern is consistent with inundation — check the red areas first."
        images.append({"label": f"{name.capitalize()} gained (red) / lost (yellow)",
                       "src": ev.overlay(after.rgb(), [(cc["gained_mask"], ev.COLORS["gained"]),
                                                       (cc["lost_mask"], ev.COLORS["lost"])]), "boxes": True})
        return _pack(answer, 0.75, images, regions=cc["regions"],
                     stats={k: cc[k] for k in ("before_fraction", "after_fraction", "gained", "lost",
                                               "net_change_pct_points", "target")},
                     method=cc["method"], warnings=cc["warnings"])

    ch = change_t.detect_change(before, after)
    trace.add("tool", f"{ch['method']}; threshold={ch['threshold']}", tool="change-detection")
    n = len(ch["regions"])
    if wants_built and before.modality == "optical":
        # New construction = changed pixels that look built-up (or bare, i.e. cleared) in the after image.
        after_lc = lc_t.classify(after)
        new_built = ch["mask"] & (after_lc["masks"]["built_up"] | after_lc["masks"]["bare_soil"])
        nb_area = common_area(int(new_built.sum()), before.gsd_m)
        trace.add("tool", "intersected change with after-date built-up/bare mask", tool="landcover")
        answer = (f"About {fmt_area(nb_area)} ({pct(new_built.mean())} of the scene) is new built-up or cleared land: "
                  f"it changed between the dates and looks built-up/bare afterwards. "
                  f"Total change is {pct(ch['fraction'])} ({fmt_area(ch['area'])}).")
        images += [{"label": "Change magnitude", "src": ev.heatmap(ch["magnitude"])},
                   {"label": "New built-up / cleared (red)", "src": ev.overlay(after.rgb(), [(new_built, ev.COLORS["change"])]),
                    "boxes": True}]
        regs = regions_of(new_built, before.gsd_m, "new built-up")
        return _pack(answer, ch["confidence"], images, regions=regs,
                     stats={"new_built_up": nb_area, "fraction": ch["fraction"], "area": ch["area"],
                            "threshold": ch["threshold"]},
                     method=ch["method"] + " ∩ after-date land cover", warnings=ch["warnings"])
    answer = (f"{pct(ch['fraction'])} of the scene changed ({fmt_area(ch['area'])}) across {n} main region(s)."
              if ch["fraction"] > 0 else "No significant change detected between the two dates.")
    if n:
        inc = ch["increase_mask"].sum() / max(ch["mask"].sum(), 1)
        answer += (f" {pct(inc)} of changed pixels got brighter (typical of new construction or cleared land), "
                   f"the rest darker (new vegetation, water or shadow).")
    images += [{"label": "Change magnitude", "src": ev.heatmap(ch["magnitude"])},
               {"label": "Detected change", "src": ev.overlay(after.rgb(), [(ch["mask"], ev.COLORS["change"])]),
                "boxes": True}]
    return _pack(answer, ch["confidence"], images, regions=ch["regions"],
                 stats={"fraction": ch["fraction"], "area": ch["area"], "threshold": ch["threshold"]},
                 method=ch["method"], warnings=ch["warnings"])


def _fusion(query, a, b, it, trace):
    optical, sar = (a, b) if a.modality == "optical" else (b, a)
    fu = fusion_t.fuse(optical, sar)
    trace.add("tool", f"optical water + SAR water (threshold {fu['sar_threshold_db']} dB), "
              f"agreement IoU={fu['agreement_iou']}", tool="optical-sar-fusion")
    bd = fu["breakdown"]
    answer = (f"Fused water extent: {fmt_area(fu['area'])} ({pct(fu['fraction'])} of the scene). "
              f"Both sensors agree on {fmt_area(bd['both_sensors'])}; SAR added {fmt_area(bd['sar_only_under_cloud'])} "
              f"under cloud ({pct(fu['cloud_fraction'])} cloud cover). ")
    if bd["sar_only_needs_review"]["pixels"]:
        answer += f"{fmt_area(bd['sar_only_needs_review'])} is SAR-only in clear sky and is flagged for review."
    images = [{"label": "Optical", "src": ev.image(optical.rgb())},
              {"label": "SAR (VV)", "src": ev.image(sar.rgb())},
              {"label": "False colour: SAR→R, optical G/B", "src": ev.image(fu["composite"])},
              {"label": "Fused water (cyan) / review (yellow)",
               "src": ev.overlay(optical.rgb(), [(fu["mask"], ev.COLORS["fused"]), (fu["review_mask"], ev.COLORS["review"])]),
               "boxes": True}]
    return _pack(answer, fu["confidence"], images, regions=fu["regions"],
                 stats={k: fu[k] for k in ("fraction", "area", "agreement_iou", "cloud_fraction", "breakdown",
                                           "optical_landcover")},
                 method=fu["method"], warnings=fu["warnings"])


def _sar_single(query, scene, _cmp, it, trace):
    from .tools.sar import sar_water

    sw = sar_water(scene)
    trace.add("tool", f"{sw['method']}; threshold={sw['threshold_db']} dB", tool="sar-water")
    answer = (f"SAR water mapping: {fmt_area(sw['area'])} ({pct(sw['fraction'])} of the scene) has low backscatter "
              f"(< {sw['threshold_db']} dB) consistent with open water, in {sw['count']} bod"
              f"{'y' if sw['count'] == 1 else 'ies'}.")
    extra_warn = []
    if it["targets"] and it["targets"][0] != "water":
        extra_warn.append("Single-band SAR supports water mapping only; add an optical scene for land-cover classes.")
    return _pack(answer, sw["confidence"],
                 [{"label": "SAR (Lee filtered)", "src": ev.image(((sw["filtered_db"] + 30) / 30)[..., None].repeat(3, -1))},
                  {"label": "Water (low backscatter)", "src": ev.overlay(scene.rgb(), [(sw["mask"], ev.COLORS["water"])]),
                   "boxes": True}],
                 regions=sw["regions"], stats={"fraction": sw["fraction"], "area": sw["area"],
                                               "threshold_db": sw["threshold_db"]},
                 method=sw["method"], warnings=extra_warn)


def _vqa(query, scene, _cmp, it, trace):
    lc = lc_t.classify(scene)
    target = it["targets"][0] if it["targets"] else None
    water = lc_t.water_bodies(scene, lc) if target in (None, "water") else None
    trace.add("tool", f"measured scene ({lc['method']})", tool="landcover")
    ans = vqa_t.rule_answer(query, target, lc, water)
    backend = "rules"
    if ans is None and vqa_t.blip_available():
        text = vqa_t.blip_answer(scene, query)
        trace.add("tool", f"open-ended question → {vqa_t.settings.blip_model}", tool="vqa/blip")
        ans = {"answer": f"{text.capitalize()}. (Open-ended answer from a general vision-language model — verify visually.)",
               "type": "open"}
        backend = "blip"
    if ans is None:
        fr = lc["fractions"]
        ranked = sorted([c for c in fr if c != "other"], key=lambda c: fr[c], reverse=True)
        ans = {"answer": "I can answer questions about presence, extent, counts and comparisons of water, vegetation, "
                         "built-up area and bare soil, plus change and fusion. Scene summary: "
                         + ", ".join(f"{PRETTY[c]} {pct(fr[c])}" for c in ranked) + ".",
               "type": "fallback"}
        backend = "rules"
    trace.add("answer", f"question type={ans['type']}", tool=f"vqa/{backend}")
    layers = [(lc["masks"][target], ev.COLORS.get(target, (255, 255, 0)))] if target else []
    images = [{"label": "Scene", "src": ev.image(scene.rgb())}]
    images.append({"label": f"Evidence: {PRETTY[target]}", "src": ev.overlay(scene.rgb(), layers), "boxes": True}
                  if target else {"label": "Land-cover map", "src": ev.class_overlay(scene.rgb(), lc["class_map"]),
                                  "legend": "landcover"})
    regs = (water["regions"] if water and target == "water" else
            ground_t.ground(scene, target, None, lc=lc)["regions"] if target else [])
    conf = {"fallback": 0.3, "open": 0.4}.get(ans["type"], lc["confidence"].get(target or "water", 0.6) or 0.6)
    return _pack(ans["answer"], conf, images, regions=regs,
                 stats={"fractions": lc["fractions"], "areas": lc["areas"], "question_type": ans["type"]},
                 method=lc["method"], extra={"vqa_backend": backend})
