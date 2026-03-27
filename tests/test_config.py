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
