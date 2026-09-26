import numpy as np

from app.agent import classify_intent, find_targets, run_query
from app.tools.change import class_change, detect_change
from app.tools.fusion import fuse
from app.tools.grounding import ground
from app.tools.landcover import classify
from app.tools.sar import lee_filter, sar_water


# ------------------------------------------------------------------ tools
def test_landcover_finds_lake_and_field(optical_before):
    lc = classify(optical_before)
    fr = lc["fractions"]
    lake_frac = np.pi * 18 ** 2 / 128 ** 2
    assert abs(fr["water"] - lake_frac) < 0.02
    assert 0.4 < fr["vegetation"] < 0.6
    assert lc["areas"]["water"]["km2"] > 0  # GSD propagates to areas


def test_grounding_respects_sector(optical_before):
    nw = ground(optical_before, "water", "north-west")
    se = ground(optical_before, "water", "south-east")
    assert nw["regions"] and not se["regions"]
    x0, y0, x1, y1 = nw["regions"][0]["bbox_norm"]
    assert x1 <= 0.5 and y1 <= 0.5


def test_change_detects_new_building(optical_before, optical_after):
    ch = detect_change(optical_before, optical_after)
    m = ch["mask"]
    assert m[95:105, 25:45].mean() > 0.8  # the new building is found
    assert m[:20, 80:].mean() < 0.05  # untouched field is quiet


def test_no_change_on_identical_scenes(optical_before):
    ch = detect_change(optical_before, optical_before)
    assert ch["fraction"] == 0


def test_water_class_change_reports_gain(optical_before, optical_after):
    cc = class_change(optical_before, optical_after, "water")
    assert cc["gained"]["pixels"] > 300
    assert cc["net_change_pct_points"] > 1


def test_sar_water_and_lee_filter(sar_scene):
    filt = lee_filter(sar_scene.sar_db)
    assert filt.std() < sar_scene.sar_db.std()  # speckle reduced
    sw = sar_water(sar_scene)
    assert sw["mask"][28:36, 28:36].all()  # lake core is water
    assert sw["mask"][5:15, 70:120].mean() < 0.05  # land is not
    assert -24 <= sw["threshold_db"] <= -12


def test_fusion_agreement(optical_after, sar_scene):
    fu = fuse(optical_after, sar_scene)
    assert fu["agreement_iou"] > 0.6
    assert fu["breakdown"]["both_sensors"]["pixels"] > 1000


# ------------------------------------------------------------------ router
def test_targets_and_order():
    assert find_targets("is the river near the forest?") == ["water", "vegetation"]


def test_intent_routing(optical_before, optical_after, sar_scene):
    assert classify_intent("where is the water?", optical_before, None)["intent"] == "grounding"
    assert classify_intent("what changed?", optical_before, optical_after)["intent"] == "change"
    assert classify_intent("classify the land cover", optical_before, None)["intent"] == "landcover"
    assert classify_intent("how much water is there?", optical_before, None)["intent"] == "vqa"
    assert classify_intent("map flood water", optical_before, sar_scene)["intent"] == "fusion"
    # change without a second date falls back and warns
    it = classify_intent("what changed since 2019?", optical_before, None)
    assert it["intent"] != "change" and it["needs_comparison"]


def test_run_query_returns_evidence(optical_before):
    r = run_query("How much of the scene is water?", optical_before)
    assert "water" in r["answer"].lower() and "%" in r["answer"]
    assert r["evidence"]["images"] and r["trace"][0]["step"] == "query"
    assert 0 <= r["confidence"] <= 1


def test_single_sar_scene_routes_to_sar_water(sar_scene):
    r = run_query("Is there water?", sar_scene)
    assert "backscatter" in r["answer"]
