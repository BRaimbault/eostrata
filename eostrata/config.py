"""Central configuration — all settings via environment variables or .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="EOSTRATA_",
        extra="ignore",
    )

    # ── Storage ───────────────────────────────────────────────────────────────
    zarr_root: Path = Path("data/zarr")
    raw_dir: Path = Path("data/raw")
    catalog_path: Path = Path("data/catalog.json")

    # ── Area of interest (bbox) ───────────────────────────────────────────────
    # Format: west, south, east, north (EPSG:4326)
    bbox_west: float = -180.0
    bbox_south: float = -90.0
    bbox_east: float = 180.0
    bbox_north: float = 90.0

    @model_validator(mode="after")
    def validate_bbox(self) -> Settings:
        if self.bbox_west >= self.bbox_east:
            raise ValueError("bbox_west must be less than bbox_east")
        if self.bbox_south >= self.bbox_north:
            raise ValueError("bbox_south must be less than bbox_north")
        return self

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Return bbox as (west, south, east, north)."""
        return (self.bbox_west, self.bbox_south, self.bbox_east, self.bbox_north)

    # ── Cache / eviction ──────────────────────────────────────────────────────
    # Maximum size of the Zarr store in megabytes.  0 means unlimited.
    store_quota_mb: float = 0.0

    # ── WorldPop ──────────────────────────────────────────────────────────────
    worldpop_base_url: str = "https://data.worldpop.org/GIS/Population/Global_2000_2020"
    worldpop_resolution: str = "1km"


settings = Settings()
