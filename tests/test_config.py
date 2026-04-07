"""Tests for config.py — Settings validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eostrata.config import Settings


class TestSettings:
    def test_default_bbox_is_global(self, monkeypatch):
        # Clear any env-file overrides so we get the class defaults
        for key in (
            "EOSTRATA_BBOX_WEST",
            "EOSTRATA_BBOX_SOUTH",
            "EOSTRATA_BBOX_EAST",
            "EOSTRATA_BBOX_NORTH",
        ):
            monkeypatch.delenv(key, raising=False)
        s = Settings(_env_file=None)
        assert s.bbox == (-180.0, -90.0, 180.0, 90.0)

    def test_bbox_property(self):
        s = Settings(bbox_west=2.0, bbox_south=4.0, bbox_east=15.0, bbox_north=14.0)
        assert s.bbox == (2.0, 4.0, 15.0, 14.0)

    def test_invalid_west_ge_east_raises(self):
        with pytest.raises((ValidationError, ValueError)):
            Settings(bbox_west=10.0, bbox_east=5.0)

    def test_invalid_south_ge_north_raises(self):
        with pytest.raises((ValidationError, ValueError)):
            Settings(bbox_south=10.0, bbox_north=5.0)

    def test_default_quota_unlimited(self, monkeypatch):
        monkeypatch.delenv("EOSTRATA_STORE_QUOTA_MB", raising=False)
        s = Settings(_env_file=None)
        assert s.store_quota_mb == 0.0

    def test_quota_setting(self):
        s = Settings(store_quota_mb=5000.0)
        assert s.store_quota_mb == 5000.0

    def test_default_paths(self):
        s = Settings()
        assert "zarr" in str(s.zarr_root)
        assert "raw" in str(s.raw_dir)
        assert "catalog" in str(s.catalog_path)

    # ── Bbox coordinate range validation ──────────────────────────────────────

    def test_bbox_west_out_of_range_raises(self):
        with pytest.raises((ValidationError, ValueError)):
            Settings(bbox_west=-181.0, bbox_east=10.0)

    def test_bbox_east_out_of_range_raises(self):
        with pytest.raises((ValidationError, ValueError)):
            Settings(bbox_west=0.0, bbox_east=181.0)

    def test_bbox_south_out_of_range_raises(self):
        with pytest.raises((ValidationError, ValueError)):
            Settings(bbox_south=-91.0, bbox_north=10.0)

    def test_bbox_north_out_of_range_raises(self):
        with pytest.raises((ValidationError, ValueError)):
            Settings(bbox_south=0.0, bbox_north=91.0)

    def test_bbox_at_poles_is_valid(self):
        """Bbox touching exactly ±90° latitude or ±180° longitude is valid."""
        s = Settings(bbox_west=-180.0, bbox_south=-90.0, bbox_east=180.0, bbox_north=90.0)
        assert s.bbox == (-180.0, -90.0, 180.0, 90.0)

    def test_bbox_near_north_pole(self):
        s = Settings(bbox_west=-10.0, bbox_south=80.0, bbox_east=10.0, bbox_north=90.0)
        assert s.bbox_north == 90.0

    def test_bbox_near_south_pole(self):
        s = Settings(bbox_west=-10.0, bbox_south=-90.0, bbox_east=10.0, bbox_north=-80.0)
        assert s.bbox_south == -90.0

    def test_bbox_west_at_antimeridian(self):
        """Bbox where west touches −180° is valid."""
        s = Settings(bbox_west=-180.0, bbox_south=-10.0, bbox_east=-170.0, bbox_north=10.0)
        assert s.bbox_west == -180.0

    def test_bbox_east_at_antimeridian(self):
        """Bbox where east touches +180° is valid."""
        s = Settings(bbox_west=170.0, bbox_south=-10.0, bbox_east=180.0, bbox_north=10.0)
        assert s.bbox_east == 180.0

    # ── New configurable settings ─────────────────────────────────────────────

    def test_default_ingest_max_workers(self, monkeypatch):
        monkeypatch.delenv("EOSTRATA_INGEST_MAX_WORKERS", raising=False)
        s = Settings(_env_file=None)
        assert s.ingest_max_workers == 3

    def test_default_ingest_max_queued(self, monkeypatch):
        monkeypatch.delenv("EOSTRATA_INGEST_MAX_QUEUED", raising=False)
        s = Settings(_env_file=None)
        assert s.ingest_max_queued == 20

    def test_default_cors_origins(self, monkeypatch):
        monkeypatch.delenv("EOSTRATA_CORS_ORIGINS", raising=False)
        s = Settings(_env_file=None)
        assert s.cors_origins == ["*"]

    def test_custom_cors_origins(self):
        s = Settings(cors_origins=["https://app.example.com", "https://staging.example.com"])
        assert "https://app.example.com" in s.cors_origins

    def test_zarr_chunk_size_too_small_raises(self):
        with pytest.raises(Exception, match="zarr_chunk_size"):
            Settings(zarr_chunk_size=32)

    def test_zarr_chunk_size_too_large_raises(self):
        with pytest.raises(Exception, match="zarr_chunk_size"):
            Settings(zarr_chunk_size=8192)

    def test_max_aggregation_timesteps_negative_raises(self):
        with pytest.raises(Exception, match="max_aggregation_timesteps"):
            Settings(max_aggregation_timesteps=-1)

    def test_agg_cache_max_entries_negative_raises(self):
        with pytest.raises(ValidationError):
            Settings(agg_cache_max_entries=-1)

    def test_agg_cache_ttl_seconds_zero_raises(self):
        with pytest.raises(ValidationError):
            Settings(agg_cache_ttl_seconds=0)
