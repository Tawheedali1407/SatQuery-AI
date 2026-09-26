# SatQuery AI

**Agentic vision-language assistant for optical and SAR remote-sensing analysis.**
Ask a question about a satellite scene in plain English. An orchestrator routes it to the right specialist tool and returns an answer backed by evidence: masks, bounding boxes, areas in km², a confidence score and the full tool trace.

> Smart India Hackathon 2026 · Problem Statement **26167** (ISRO / SAC) · Space Technology · **Team Hydra**

![ChangeFormer finding a new warehouse between two dates](docs/screenshots/03_change.png)

**No paid APIs and no API keys are required.** Every model is open (weights from GitHub) and runs locally on CPU. Google Gemini is an *optional* free add-on for open-ended questions.

---

## Why it's different

| | Typical "chat with an image" demo | SatQuery AI |
|---|---|---|
| Answers | Free text from one big VLM | Numbers measured from pixels (%, km², counts) |
| Evidence | None | Mask overlay, ranked boxes, change heat-map, trace |
| Change over time | Not supported | Bi-temporal change + per-class gain / loss |
| SAR | Not supported | Lee speckle filter, backscatter water mapping |
| Optical + SAR | — | Decision-level fusion with cloud screening |
| Hardware | GPU required | Runs on any laptop CPU; uses a GPU automatically if present |
| Accuracy | Unknown | Benchmarked against ground truth: **change-detection F1 0.90** (`docs/benchmarks.md`) |

## What it can do

| Module | Example query | How it works |
|---|---|---|
| **RS-VQA** | *"How much of the scene is water?"*, *"How many water bodies?"*, *"Is there more vegetation or built-up area?"* | Question-type parser answers from measured land-cover statistics. Open-ended questions go to Gemini (optional free key), then BLIP-VQA (local), grounded on the measured numbers |
| **Land cover** | *"Classify the land cover"* | NDVI / NDWI (with NIR) or visible-band indices, Otsu thresholds, cloud + no-data masking |
| **Region grounding** | *"Locate water bodies in the north-east"* | Class mask → connected components → ranked boxes, sector filter |
| **Change detection** | *"What changed between the two dates?"*, *"How much new built-up area appeared?"* | **ChangeFormerV6** transformer (pretrained on LEVIR-CD, open GitHub weights) with tiled inference for high-resolution imagery. Classical CVA + SSIM for 10 m data or when weights aren't installed. SAR uses log-ratio |
| **Flood / class change** | *"Did water increase?"* | Per-date class maps → gained / lost area |
| **Optical–SAR fusion** | *"Fuse optical and SAR to map flood water"* | Independent water maps, rule-based fusion: SAR fills in under cloud, disagreements flagged for review |

## Architecture

```mermaid
flowchart LR
    Q[Natural-language query] --> I[Intent classifier<br/>rules-v1 + scene context]
    S[(Scene register<br/>GeoTIFF / PNG / SAR)] --> I
    I -->|vqa| V[RS-VQA]
    I -->|landcover| L[Land cover]
    I -->|grounding| G[Region grounding]
    I -->|change| C[Change detection]
    I -->|fusion| F[Optical–SAR fusion]
    V & L & G & C & F --> E[Evidence builder<br/>overlays · boxes · km² · confidence]
    E --> A[Answer + tool trace + PDF report]
```

Each specialist is a plain Python function with the same contract (scenes in, masks + measurements out), so any one of them can be swapped for a deep model without touching the router, API or UI. See [docs/architecture.md](docs/architecture.md).

## Results (measured, not claimed)

Reproduce with `python backend/scripts/evaluate.py`. These are CPU numbers on the LEVIR-CD **test** pairs in `samples/` (never seen in training):

| Task | Method | Precision | Recall | F1 | IoU | CPU time / 256² pair |
|---|---|---|---|---|---|---|
| Change detection | Classical CVA + SSIM (no training) | 0.243 | 0.564 | 0.340 | 0.205 | 41 ms |
| Change detection | **ChangeFormerV6 (pretrained, open weights)** | **0.906** | **0.892** | **0.899** | **0.816** | ≈1.7 s (2-core VM) |
| Water: optical / SAR / fused | Sen1Floods11 India chips | — | — | — | — | run `evaluate.py` after downloading the chips |

The pretrained model is **2.6× more accurate** than the classical baseline, which stays as the zero-dependency fallback. Five pairs is a small sanity check. The ChangeFormer paper reports F1 0.904 on the full 2,048-tile test set, which matches what we measure here.

## Quick start

**Requirements:** Python 3.10+ and Node 18+. No GPU needed.

```bash
git clone https://github.com/Tawheedali1407/SatQuery-AI.git
cd SatQuery-AI

# 1. Backend
cd backend
python -m venv .venv
.venv\Scripts\activate           # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python scripts/fetch_samples.py  # downloads real demo scenes (~5 MB, more with Sen1Floods11)
uvicorn app.main:app --reload    # http://localhost:8000/docs for the API

# 2. Frontend (new terminal)
cd frontend
npm install
npm run dev                      # http://localhost:5173
```

**One command with Docker:**

```bash
docker compose up --build        # console + API + ChangeFormer on http://localhost:8000
# smaller CPU-only image without the deep model:  docker build --build-arg WITH_ML=0 -t satquery .
```

**Accuracy upgrade: pretrained change model** (recommended; open weights, no keys):

```bash
pip install -r requirements-ml.txt   # torch, timm, einops, transformers (CPU is fine)
python scripts/fetch_models.py       # ChangeFormer code + LEVIR-CD weights from GitHub (~1 GB download, 165 MB kept)
```

The sidebar then shows `change: changeformer`. On an NVIDIA GPU it is used automatically.

**Optional: Google Gemini for open-ended questions** (free tier, no credit card):

1. Get a key at <https://aistudio.google.com/apikey>.
2. Copy `.env.example` to `backend/.env` and set `SATQUERY_GEMINI_API_KEY=...`.

Gemini receives the scene preview and SatQuery's own measurements, and is told to use those numbers, which limits hallucination. Without a key, BLIP (local) or the rule-based answers are used instead. Leave it unset for fully offline, ground-station use: then no data leaves the machine.

## Demo script (3 minutes)

1. **Load demo → Landsat 7 coastal scene.** Ask *"Locate water bodies in the north-east"*. You get boxes, km² per region, and the trace showing `grounding` was chosen.
2. Ask *"What is the land cover?"*. You get the class map with a legend; clouds and the no-data border are excluded from the percentages.
3. **Load demo → LEVIR-CD pair.** Ask *"How much new built-up area appeared?"*. This produces a change heat-map and new-construction mask in m². Mention the measured F1 honestly.
4. **Load demo → India flood chip** (Sen1Floods11). Ask *"Fuse optical and SAR to map flood water"*. This shows water that SAR sees under cloud, sensor agreement IoU, and areas flagged for review.
5. Click **Export report (PDF)** to get a decision-ready handout for a disaster-management officer.

## Using your own data

- **GeoTIFF** (recommended): CRS, footprint and pixel size are read automatically, so areas come out in km² and the scene appears on the globe. Multiband order is assumed to be R, G, B, NIR unless band descriptions say otherwise.
- **SAR**: a single-band VV raster in dB or linear power. Filenames containing `sar`, `vv`, `s1`, `risat` or `eos04` are auto-detected, or pick "SAR" in the upload box.
- **PNG / JPG**: enter the GSD (m/pixel) in the upload box to get real-world areas.
- **Large scenes** are read at up to 2048 px on the longest side through GDAL overviews (`SATQUERY_MAX_ANALYSIS_PX`), so a full scene never has to fit in memory.
- ISRO sources that work directly: Bhoonidhi (Resourcesat LISS-IV / AWiFS, Cartosat, EOS-04 SAR) GeoTIFF exports.

## Project structure

```
satquery-ai/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI routes
│   │   ├── agent.py         # intent classifier, router, answer + trace assembly
│   │   ├── raster.py        # GeoTIFF/PNG loading, overview reads, no-data, SAR dB
│   │   ├── evidence.py      # overlays, heat-maps → PNG
│   │   ├── store.py         # session scene registry, demo catalogue
│   │   ├── tools/           # landcover · sar · change · fusion · grounding · vqa
│   │   └── ml/              # optional deep backends: changeformer (tiled inference) · gemini
│   ├── scripts/
│   │   ├── fetch_samples.py # downloads real demo data
│   │   ├── fetch_models.py  # downloads open pretrained weights from GitHub
│   │   └── evaluate.py      # benchmarks vs ground truth → docs/benchmarks.md
│   └── tests/               # pytest: tools, router, API
├── frontend/                # React + Vite console (Console · Library · CesiumJS globe)
├── docs/                    # architecture, benchmarks, screenshots
├── Dockerfile · docker-compose.yml
└── .github/workflows/ci.yml # lint + tests + benchmark + frontend build
```

## Honest limitations

- The intent classifier is rule-based (v1). It is deterministic and explainable, but it only understands the query patterns it was written for, and unrecognised questions get a scene summary plus a list of supported question types.
- RGB-only land cover confuses dark seagrass shallows with dark forest, and blue roofs with water. NIR imagery (Sentinel-2, LISS-IV) fixes most of this.
- ChangeFormer was trained on 0.5 m imagery for *building* change. It is used only when GSD ≤ 2 m (or unknown), and 10 m Sentinel-2 pairs use the classical detector, which over-reports seasonal and shadow change.
- Confidence values are heuristics from classification margins, not calibrated probabilities. The UI says so.
- Scenes passed to change detection or fusion are assumed to be co-registered. Mismatched sizes are resampled, and a warning is returned.

## Roadmap

1. Fine-tune ChangeFormer on Indian imagery (Cartosat / LISS-IV pairs), and train a segmentation model on Sen1Floods11 / BigEarthNet-MM for land cover and water; GeoChat for grounded open-ended VQA.
2. A small local LLM (via Ollama) for function-calling intent routing, keeping the rules as a fallback.
3. Automatic co-registration and cloud masking with Sentinel-2 SCL / Fmask.
4. Direct Bhoonidhi search and ingest; tiled full-resolution inference for complete scenes.

## References

- Lobry et al., *RSVQA: Visual Question Answering for Remote Sensing Data*, IEEE TGRS 2020 — [arXiv:2003.07333](https://arxiv.org/abs/2003.07333)
- Kuckreja et al., *GeoChat: Grounded Large Vision-Language Model for Remote Sensing*, CVPR 2024 — [arXiv:2311.15826](https://arxiv.org/abs/2311.15826)
- Chen & Shi, *A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection* (LEVIR-CD), Remote Sensing 2020
- Chen et al., *Remote Sensing Image Change Detection with Transformers* (BIT), IEEE TGRS 2021
- Bandara & Patel, *A Transformer-Based Siamese Network for Change Detection* (ChangeFormer), IGARSS 2022
- Bonafilia et al., *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1*, CVPR Workshops 2020
- Sumbul et al., *BigEarthNet-MM*, 2021 — [arXiv:2105.07921](https://arxiv.org/abs/2105.07921)
- Lee, *Digital image enhancement and noise filtering by use of local statistics*, IEEE TPAMI 1980 (Lee filter)

## Team Hydra

| Name | Role |
|---|---|
| Ali | Lead · ML / backend |
| _add teammate_ | _role_ |
| _add teammate_ | _role_ |

Licensed under the [MIT License](LICENSE). Demo datasets keep their own licences, listed in [samples/README.md](samples/README.md).
