# Demo scenes

This folder is filled by `python backend/scripts/fetch_samples.py` and is git-ignored.
Nothing here is committed: every scene is downloaded from its public source so the
dataset licences stay with their owners.

| Sample | Source | Licence |
|---|---|---|
| `landsat/` | USGS Landsat 7, via rasterio test data | Public domain (USGS) |
| `levir/` | LEVIR-CD, Chen & Shi, *Remote Sensing* 2020 (samples from the BIT_CD repo) | Academic research use |
| `sen1floods11/` | Sen1Floods11, Bonafilia et al., CVPR-W 2020 — India flood chips | CC BY 4.0 |

`catalog.json` describes each demo (files, roles, suggested queries) and is read by the API.

## Models

`backend/scripts/fetch_models.py` downloads ChangeFormerV6 code (MIT, pinned commit) and its LEVIR-CD
weights from the authors' GitHub release (github.com/wgcban/ChangeFormer) into `backend/third_party/`
and `backend/weights/`, both git-ignored.
