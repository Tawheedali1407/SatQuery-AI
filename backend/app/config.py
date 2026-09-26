"""Runtime settings, overridable with SATQUERY_* environment variables (or backend/.env)."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SATQUERY_", env_file=str(BACKEND / ".env"), extra="ignore")

    # Where uploaded scenes are kept for the session.
    data_dir: Path = BACKEND / "data"
    # Where bundled / downloaded demo scenes live.
    samples_dir: Path = BACKEND.parent / "samples"
    # Pretrained weights and third-party model code (both fetched by scripts/fetch_models.py, git-ignored).
    weights_dir: Path = BACKEND / "weights"
    third_party_dir: Path = BACKEND / "third_party"
    # Scenes larger than this (in pixels, longest side) are read at a reduced
    # resolution through raster overviews so a laptop never loads a full scene.
    max_analysis_px: int = 2048
    max_upload_mb: int = 512

    # Change detection: "auto" uses ChangeFormer when its weights are installed and the
    # imagery is high-resolution (GSD <= change_max_gsd_m, or unknown), else the classical method.
    change_backend: str = "auto"  # auto | classical | changeformer
    change_max_gsd_m: float = 2.0
    torch_threads: int = 0  # 0 = let torch decide

    # Open-ended VQA fallback chain: gemini (if key) -> blip (if installed) -> rules.
    vqa_backend: str = "auto"  # auto | rules | blip | gemini
    blip_model: str = "Salesforce/blip-vqa-base"
    # Free key from https://aistudio.google.com/apikey — the image and question are sent to Google.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    cors_origins: list[str] = ["*"]


settings = Settings()
