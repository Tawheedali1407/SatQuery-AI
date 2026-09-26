"""ChangeFormerV6 (LEVIR-CD) wrapper with tiled inference.

The network was trained on 256x256 tiles of 0.5 m Google Earth imagery to find
*building* change. We therefore:

* run it only on high-resolution optical pairs (GSD <= `change_max_gsd_m`, or
  unknown GSD such as plain PNG/JPG) and fall back to the classical detector
  for 10 m Sentinel-2-class data, where it would be out of domain;
* tile the scene into 256 px windows with overlap and average the softmax
  probabilities, so any scene size works without resizing the pixels.

Code and weights are fetched by `scripts/fetch_models.py` (MIT, github.com/wgcban/ChangeFormer).
"""
from __future__ import annotations

import logging
import sys
import threading
from functools import lru_cache

import numpy as np

from ..config import settings

log = logging.getLogger(__name__)
TILE = 256
WEIGHTS = "changeformer_v6_levir.pt"
_lock = threading.Lock()


def available() -> bool:
    if settings.change_backend == "classical":
        return False
    if not (settings.weights_dir / WEIGHTS).exists():
        return False
    if not (settings.third_party_dir / "changeformer" / "models" / "ChangeFormer.py").exists():
        return False
    try:
        import einops  # noqa: F401
        import timm  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def suitable(gsd_m: float | None) -> bool:
    """Is the imagery close enough to the training domain (0.5 m aerial)?"""
    if settings.change_backend == "changeformer":
        return True
    return gsd_m is None or gsd_m <= settings.change_max_gsd_m


@lru_cache(maxsize=1)
def _model():
    import torch

    root = str(settings.third_party_dir / "changeformer")
    if root not in sys.path:
        sys.path.insert(0, root)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # timm deprecation noise from third-party imports
        from models.ChangeFormer import ChangeFormerV6

    if settings.torch_threads:
        torch.set_num_threads(settings.torch_threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ChangeFormerV6(embed_dim=256)
    ckpt = torch.load(settings.weights_dir / WEIGHTS, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["state_dict"])
    net.eval().to(device)
    log.info("ChangeFormerV6 loaded on %s", device)
    return net, device


def warmup() -> None:
    try:
        _model()
    except Exception as e:  # never block startup on an optional model
        log.warning("ChangeFormer warm-up failed: %s", e)


def _windows(n: int, tile: int, stride: int) -> list[int]:
    if n <= tile:
        return [0]
    starts = list(range(0, n - tile, stride))
    starts.append(n - tile)
    return starts


def predict(before_rgb: np.ndarray, after_rgb: np.ndarray, overlap: int = 32, batch: int = 4) -> np.ndarray:
    """Return P(change) per pixel, shape (H, W), for two co-registered RGB arrays in [0, 1]."""
    import torch

    net, device = _model()
    h, w = before_rgb.shape[:2]
    # Pad small scenes up to one tile (reflect keeps statistics natural).
    ph, pw = max(0, TILE - h), max(0, TILE - w)
    if ph or pw:
        pad = ((0, ph), (0, pw), (0, 0))
        before_rgb, after_rgb = np.pad(before_rgb, pad, mode="reflect"), np.pad(after_rgb, pad, mode="reflect")
    H, W = before_rgb.shape[:2]

    def to_t(x):  # the model was trained with Normalize(mean=0.5, std=0.5)
        return torch.from_numpy(((x.astype(np.float32) - 0.5) / 0.5).transpose(2, 0, 1).copy())

    prob = np.zeros((H, W), np.float32)
    weight = np.zeros((H, W), np.float32)
    # Down-weight tile borders so overlapping predictions blend smoothly.
    ramp = np.minimum(np.arange(TILE) + 1, np.arange(TILE)[::-1] + 1).astype(np.float32)
    win = np.minimum.outer(ramp, ramp)
    win = np.clip(win / max(overlap, 1), 0.05, 1.0)

    coords = [(y, x) for y in _windows(H, TILE, TILE - overlap) for x in _windows(W, TILE, TILE - overlap)]
    with _lock, torch.no_grad():  # one inference at a time keeps memory predictable on laptops
        for i in range(0, len(coords), batch):
            chunk = coords[i:i + batch]
            a = torch.stack([to_t(before_rgb[y:y + TILE, x:x + TILE]) for y, x in chunk]).to(device)
            b = torch.stack([to_t(after_rgb[y:y + TILE, x:x + TILE]) for y, x in chunk]).to(device)
            p = torch.softmax(net(a, b)[-1], dim=1)[:, 1].cpu().numpy()
            for (y, x), pt in zip(chunk, p, strict=True):
                prob[y:y + TILE, x:x + TILE] += pt * win
                weight[y:y + TILE, x:x + TILE] += win
    return (prob / np.maximum(weight, 1e-6))[:h, :w]
