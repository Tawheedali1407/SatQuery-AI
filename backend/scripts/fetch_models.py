#!/usr/bin/env python3
"""Download open pretrained models used by SatQuery AI (no API keys, no accounts).

  * ChangeFormerV6 trained on LEVIR-CD (building change, 0.5 m imagery)
      code    : github.com/wgcban/ChangeFormer (MIT), pinned commit, 4 files from models/
      weights : GitHub release v0.1.0 of the same repo (~985 MB zip -> ~165 MB model-only file)
      paper   : Bandara & Patel, "A Transformer-Based Siamese Network for Change Detection", IGARSS 2022

Needs: pip install -r requirements-ml.txt  (torch, timm, einops)

Usage:
    python scripts/fetch_models.py
    python scripts/fetch_models.py --keep-zip     # keep the downloaded release zip
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from app.config import settings  # noqa: E402

CF_REPO = "wgcban/ChangeFormer"
CF_COMMIT = "afd1b7ed640aa265a2c730de958416ae7356a2f9"
CF_FILES = ["models/ChangeFormer.py", "models/ChangeFormerBaseNetworks.py", "models/help_funcs.py",
            "models/pixel_shuffel_up.py", "LICENSE"]
CF_WEIGHTS_URL = (f"https://github.com/{CF_REPO}/releases/download/v0.1.0/"
                  "CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_"
                  "multi_infer_False_shuffle_AB_False_embed_dim_256.zip")
CF_WEIGHTS_OUT = "changeformer_v6_levir.pt"


def download(url: str, dest: Path, label: str) -> None:
    with urllib.request.urlopen(url, timeout=120) as r, dest.open("wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total > 5e6:
                print(f"\r  {label}: {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print(f"\r  ✓ {label}" + " " * 30)


def fetch_changeformer(keep_zip: bool) -> None:
    code_dir = settings.third_party_dir / "changeformer"
    (code_dir / "models").mkdir(parents=True, exist_ok=True)
    print(f"ChangeFormer code @ {CF_COMMIT[:7]}")
    for f in CF_FILES:
        dest = code_dir / f
        if not dest.exists():
            download(f"https://raw.githubusercontent.com/{CF_REPO}/{CF_COMMIT}/{f}", dest, f)
    (code_dir / "models" / "__init__.py").touch()

    out = settings.weights_dir / CF_WEIGHTS_OUT
    if out.exists():
        print(f"  weights already present: {out.relative_to(BACKEND)}")
        return
    settings.weights_dir.mkdir(parents=True, exist_ok=True)
    import torch

    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "cf_levir.zip"
        print("ChangeFormer LEVIR-CD weights (GitHub release, ~985 MB download)")
        download(CF_WEIGHTS_URL, zpath, "weights")
        with zipfile.ZipFile(zpath) as z:
            name = next(n for n in z.namelist() if n.endswith("best_ckpt.pt"))
            z.extract(name, tmp)
        ckpt = torch.load(Path(tmp) / name, map_location="cpu", weights_only=False)
        # Keep only the network weights (drops optimiser state: 490 MB -> ~165 MB).
        torch.save({"state_dict": ckpt["model_G_state_dict"], "source": CF_WEIGHTS_URL,
                    "best_val_acc": float(ckpt.get("best_val_acc", 0))}, out)
        if keep_zip:
            shutil.copy(zpath, settings.weights_dir / zpath.name)
    print(f"  ✓ {out.relative_to(BACKEND)} ({out.stat().st_size / 1e6:.0f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()
    try:
        import timm  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        sys.exit("Install the ML extras first:  pip install -r requirements-ml.txt")
    fetch_changeformer(args.keep_zip)
    print("\nDone. Restart the API; /api/health should report change backend 'changeformer'.")


if __name__ == "__main__":
    main()
