"""Tests for the cache eviction module."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import xarray as xr

from eostrata.cache import (
    check_and_evict,
    evict_group,
    list_groups,
    store_size_mb,
)


def _write_fake_group(zarr_root: Path, group: str, size_kb: int = 10) -> None:
    """Write a dummy Zarr group with roughly *size_kb* of data."""
    data = np.random.rand(size_kb, 8).astype("float32")  # ~size_kb * 32 bytes
    ds = xr.Dataset({"data": (("y", "x"), data)})
    ds.to_zarr(str(zarr_root), group=group, mode="w")


class TestStoreSizeMb:
    def test_nonexistent_root(self, tmp_path):
        assert store_size_mb(tmp_path / "nonexistent") == 0.0

    def test_empty_root(self, tmp_path):
        assert store_size_mb(tmp_path) == 0.0

    def test_nonempty_store(self, tmp_path):
        _write_fake_group(tmp_path, "worldpop/nga")
        size = store_size_mb(tmp_path)
        assert size > 0


class TestListGroups:
    def test_empty(self, tmp_path):
        assert list_groups(tmp_path) == []

    def test_nonexistent(self, tmp_path):
        assert list_groups(tmp_path / "nope") == []

    def test_returns_two_level_groups(self, tmp_path):
        _write_fake_group(tmp_path, "worldpop/nga")
        _write_fake_group(tmp_path, "chirps/global")
        groups = list_groups(tmp_path)
        paths = [g[0] for g in groups]
        assert "worldpop/nga" in paths
        assert "chirps/global" in paths

    def test_sorted_oldest_first(self, tmp_path):
        _write_fake_group(tmp_path, "worldpop/nga")
        time.sleep(0.05)  # ensure different mtime
        _write_fake_group(tmp_path, "chirps/global")
        groups = list_groups(tmp_path)
        paths = [g[0] for g in groups]
        assert paths[0] == "worldpop/nga"
        assert paths[1] == "chirps/global"


class TestEvictGroup:
    def test_removes_directory(self, tmp_path):
        _write_fake_group(tmp_path, "worldpop/nga")
        assert (tmp_path / "worldpop" / "nga").exists()
        freed = evict_group(tmp_path, "worldpop/nga")
        assert freed > 0
        assert not (tmp_path / "worldpop" / "nga").exists()

    def test_nonexistent_group_returns_zero(self, tmp_path):
        freed = evict_group(tmp_path, "worldpop/nonexistent")
        assert freed == 0.0


class TestCheckAndEvict:
    def test_unlimited_quota_is_noop(self, tmp_path):
        _write_fake_group(tmp_path, "worldpop/nga")
        # Should not raise or evict anything
        check_and_evict(tmp_path, quota_mb=0.0)
        assert (tmp_path / "worldpop" / "nga").exists()

    def test_within_quota_no_eviction(self, tmp_path):
        _write_fake_group(tmp_path, "worldpop/nga", size_kb=1)
        check_and_evict(tmp_path, quota_mb=1000.0)
        assert (tmp_path / "worldpop" / "nga").exists()

    def test_exceeds_quota_evicts_oldest(self, tmp_path):
        _write_fake_group(tmp_path, "worldpop/nga", size_kb=50)
        time.sleep(0.05)
        _write_fake_group(tmp_path, "chirps/global", size_kb=50)

        size_before = store_size_mb(tmp_path)
        # Set quota just below current size — should evict oldest (worldpop/nga)
        quota = size_before * 0.6
        check_and_evict(tmp_path, quota_mb=quota)

        assert not (tmp_path / "worldpop" / "nga").exists()
        # chirps/global may or may not survive depending on sizes, but store is smaller
        assert store_size_mb(tmp_path) <= quota
