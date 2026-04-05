"""Central configuration — all settings via environment variables or .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator, model_validator
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
        if not (-180.0 <= self.bbox_west <= 180.0):
            raise ValueError("bbox_west must be in [-180, 180]")
        if not (-180.0 <= self.bbox_east <= 180.0):
            raise ValueError("bbox_east must be in [-180, 180]")
        if not (-90.0 <= self.bbox_south <= 90.0):
            raise ValueError("bbox_south must be in [-90, 90]")
        if not (-90.0 <= self.bbox_north <= 90.0):
            raise ValueError("bbox_north must be in [-90, 90]")
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

    # Optional headroom to keep free inside the quota before a new download.
    # Recommended: ~10% of quota.  Ignored if it equals or exceeds the quota.
    # Example: quota=10000, buffer=1000 → evict until store ≤ 9000 MB.
    store_eviction_buffer_mb: float = 0.0

    # Whether to update per-timestamp access sentinels on tile/zonal-stats requests.
    # When False, last_access reflects the ingestion time only.
    track_access: bool = True

    # ── Memory tuning ─────────────────────────────────────────────────────────
    # Spatial chunk tile size for Zarr writes (applied to both x and y dimensions).
    # Each in-memory tile during reads or writes is chunk_size² × 4 bytes (float32).
    # Halving this value quarters the per-operation memory footprint at the cost of
    # more I/O operations.
    #
    # Recommended values by available RAM:
    #   ≤ 512 MB  → 128 or 256
    #     1–2 GB  → 256 or 512  (default)
    #       ≥ 4GB → 512 or 1024
    #
    # NOTE: changing this only affects newly written Zarr groups; existing groups
    # retain their original chunk layout.
    zarr_chunk_size: int = 512

    @field_validator("zarr_chunk_size")
    @classmethod
    def validate_zarr_chunk_size(cls, v: int) -> int:
        if v < 64 or v > 4096:
            raise ValueError("zarr_chunk_size must be between 64 and 4096")
        return v

    # Threshold for switching to batched temporal aggregation.
    # When a request spans more timesteps than this limit, the reduction is split
    # into sequential batches of aggregation_batch_size timesteps each.
    # Set to 0 to process all timesteps in a single pass (lowest latency, highest RAM).
    #
    # Recommended values by available RAM:
    #   ≤ 512 MB → 6   (triggers batching for any range > 6 months)
    #     1–2 GB → 60  (five years of monthly data)
    #      ≥ 4GB → 0   (unlimited)
    max_aggregation_timesteps: int = 0

    @field_validator("max_aggregation_timesteps")
    @classmethod
    def validate_max_aggregation_timesteps(cls, v: int) -> int:
        if v < 0:
            raise ValueError(
                "max_aggregation_timesteps must be 0 (unlimited) or a positive integer"
            )
        return v

    # Number of timesteps loaded per batch when batched aggregation is active.
    # Lower values reduce peak RAM at the cost of more sequential zarr reads.
    # On a 512 MB instance with global CHIRPS (≈103 MB/month): set to 1.
    # Ignored when max_aggregation_timesteps=0 (no batching).
    aggregation_batch_size: int = 1

    @field_validator("aggregation_batch_size")
    @classmethod
    def validate_aggregation_batch_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError("aggregation_batch_size must be a positive integer")
        return v

    # Maximum number of concurrent temporal aggregations (tile renders + zonalstats)
    # that may run at the same time.  Each aggregation loads one or more full spatial
    # slices into memory; on memory-constrained instances (≤ 512 MB) set this to 1
    # so that operations are serialised and never overlap.
    # 0 means unlimited (rely on the OS / worker count).
    max_concurrent_aggregations: int = 1

    @field_validator("max_concurrent_aggregations")
    @classmethod
    def validate_max_concurrent_aggregations(cls, v: int) -> int:
        if v < 0:
            raise ValueError(
                "max_concurrent_aggregations must be 0 (unlimited) or a positive integer"
            )
        return v

    # ── Ingestion ─────────────────────────────────────────────────────────────
    # Maximum number of concurrent ingestion jobs.  Increase for deployments
    # that serve many users submitting ingest requests simultaneously.
    ingest_max_workers: int = 3

    # Maximum number of queued + running ingestion jobs.  Requests submitted
    # when this limit is reached receive a 429 Too Many Requests response.
    ingest_max_queued: int = 20

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Allowed origins for CORS.  Use ["*"] for public servers.  Restrict to
    # specific origins (e.g. ["https://app.example.com"]) for internal
    # deployments to prevent cross-site requests from untrusted origins.
    cors_origins: list[str] = ["*"]

    # ── Logging ───────────────────────────────────────────────────────────────
    # Log file path. Rotates daily; 30 days of history kept automatically.
    # Set to empty string to disable file logging.
    log_file: str = "data/eostrata.log"

    # ── WorldPop ──────────────────────────────────────────────────────────────
    worldpop_base_url: str = "https://data.worldpop.org/GIS/Population/Global_2000_2020"
    worldpop_resolution: str = "1km"

    # ── Sentinel NDVI (CGLS) ──────────────────────────────────────────────────
    # Optional Copernicus Land Service API token.  Leave empty for public access.
    # Register at https://land.copernicus.eu/global/ to obtain a token.
    cgls_api_key: str = ""

    # ── CDS / ERA5 (Climate Data Store) ──────────────────────────────────────
    # CDS API URL.  Defaults to the public CDS endpoint.
    # Override only if using a custom/mirror CDS instance.
    cds_url: str = "https://cds.climate.copernicus.eu/api"

    # CDS API key in "<uid>:<api-key>" format.  Leave empty to use ~/.cdsapirc.
    # Register and obtain your key at https://cds.climate.copernicus.eu/how-to-api
    cds_key: str = ""

    # ── CAMS (Atmosphere Data Store) ──────────────────────────────────────────
    # ADS API URL.  Defaults to the public ADS endpoint.
    # Override only if using a custom/mirror ADS instance.
    ads_url: str = "https://ads.atmosphere.copernicus.eu/api"

    # ADS API key in "<uid>:<api-key>" format.  Leave empty to use ~/.adsapirc.
    # Register and obtain your key at https://ads.atmosphere.copernicus.eu/how-to-api
    ads_key: str = ""

    # ── TROPOMI / Copernicus Data Space (CDSE) ────────────────────────────────
    # CDSE username (email address used to register).
    # Register for free at https://dataspace.copernicus.eu
    cdse_user: str = ""

    # CDSE password.
    cdse_password: str = ""


settings = Settings()
