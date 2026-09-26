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
