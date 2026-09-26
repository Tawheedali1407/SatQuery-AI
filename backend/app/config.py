"""Runtime settings, overridable with SATQUERY_* environment variables."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SATQUERY_", env_file=".env", extra="ignore")

    # Where uploaded scenes are kept for the session.
    data_dir: Path = Path(__file__).resolve().parents[1] / "data"
    # Where bundled / downloaded demo scenes live.
    samples_dir: Path = Path(__file__).resolve().parents[2] / "samples"
    # Scenes larger than this (in pixels, longest side) are read at a reduced
    # resolution through raster overviews so a laptop never loads a full scene.
    max_analysis_px: int = 2048
    max_upload_mb: int = 512
    # Optional deep models. "auto" uses them only if torch/transformers are installed.
    vqa_backend: str = "auto"  # auto | rules | blip
    blip_model: str = "Salesforce/blip-vqa-base"
    cors_origins: list[str] = ["*"]


settings = Settings()
