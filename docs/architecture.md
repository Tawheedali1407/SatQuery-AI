# Architecture

## Request flow

1. **Scene register.** `POST /api/scenes` streams the upload to disk and `raster.load_scene()` reads it:
   - GeoTIFF is read through rasterio at up to `max_analysis_px`, using GDAL overviews, so memory stays flat for large scenes. CRS, WGS-84 footprint and GSD are extracted.
   - No-data pixels (all-zero or NaN) are masked, and the contrast stretch uses valid pixels only.
   - SAR is converted to dB. Linear power, dB floats and 8-bit quicklooks are all handled.
2. **Query.** `POST /api/query` calls `agent.run_query()`, which does the following:
   - `classify_intent()` scores five intents with regex patterns and adds scene context: two optical dates boost *change*, and an optical + SAR pair boosts *fusion*. It also extracts target classes (for example "river" → water) and image sectors ("north-east").
   - It routes to a handler, which calls one or more tools from `app/tools/`.
   - The handler turns tool output (numpy masks + measurements) into an answer sentence, evidence images (`evidence.py`), ranked regions, stats and warnings.
   - Every step appends to a `Trace` with timings. The UI renders this as the "Routed tool trace".
3. **Report.** The UI builds a print-ready HTML report from the same JSON, which the browser saves as a PDF.

## Tool contract

Every tool takes `Scene` objects and returns a dict with at least a boolean `mask`, measurements (`fraction`, `area` in pixels, m², km² and ha when GSD is known), `regions` (bounding boxes, normalised and in pixels), `confidence` and `method`. Replacing a tool with a deep model means returning the same dict. For example, a ChangeFormer wrapper only needs to produce `mask` and `magnitude`.

## Why rule-based routing first?

- Judges and analysts can see *why* a tool was picked (the scores are in the trace).
- It is deterministic, which makes testing possible (`tests/test_tools.py::test_intent_routing`).
- It has zero latency and zero GPU cost, and it works offline in a ground station.
- An LLM router can be added behind `classify_intent()`, with the rules kept as a fallback.

## Configuration

All settings are `SATQUERY_*` environment variables (see `.env.example` and `app/config.py`).
