"""Tests for the cache eviction module."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from eostrata.cache import (
    _ACCESSED_SENTINEL,
    _DEBOUNCE_S,
    check_and_evict,
    evict_group,
    list_groups,
    record_access,
    store_size_mb,
)


def _write_fake_group(zarr_root: Path, group: str, size_kb: int = 10) -> None:
    """Write a dummy Zarr group with roughly *size_kb* of data."""
    data = np.random.rand(size_kb, 8).astype("float32")  # ~size_kb * 32 bytes
    ds = xr.Dataset({"data": (("y", "x"), data)})
    ds.to_zarr(str(zarr_root), group=group, mode="w")


class TestRecordAccess:
    def test_creates_sentinel_file(self, tmp_path):
        _write_fake_group(tmp_path, "worldpop/nga")
        record_access(tmp_path, "worldpop/nga")
        assert (tmp_path / "worldpop" / "nga" / _ACCESSED_SENTINEL).exists()

    def test_updates_sentinel_mtime(self, tmp_path):
        _write_fake_group(tmp_path, "worldpop/nga")
        sentinel = tmp_path / "worldpop" / "nga" / _ACCESSED_SENTINEL
        sentinel.touch()
        mtime_before = sentinel.stat().st_mtime
        time.sleep(0.05)
        record_access(tmp_path, "worldpop/nga")
        assert sentinel.stat().st_mtime >= mtime_before

    def test_debounce_skips_touch_within_window(self, tmp_path):
        """Second call within _DEBOUNCE_S seconds does not update the sentinel."""
        _write_fake_group(tmp_path, "worldpop/nga")
        record_access(tmp_path, "worldpop/nga")
        sentinel = tmp_path / "worldpop" / "nga" / _ACCESSED_SENTINEL
        mtime_first = sentinel.stat().st_mtime
        # Simulate time just inside the debounce window
        with patch("eostrata.cache.time.time", return_value=mtime_first + _DEBOUNCE_S - 1):
            record_access(tmp_path, "worldpop/nga")
        assert sentinel.stat().st_mtime == mtime_first

    def test_debounce_allows_touch_after_window(self, tmp_path):
        """Call after _DEBOUNCE_S seconds does update the sentinel."""
        _write_fake_group(tmp_path, "worldpop/nga")
        record_access(tmp_path, "worldpop/nga")
        sentinel = tmp_path / "worldpop" / "nga" / _ACCESSED_SENTINEL
        mtime_first = sentinel.stat().st_mtime
        # Simulate time just past the debounce window
        with patch("eostrata.cache.time.time", return_value=mtime_first + _DEBOUNCE_S + 1):
            record_access(tmp_path, "worldpop/nga")
        assert sentinel.stat().st_mtime >= mtime_first

    def test_oserror_is_silenced(self, tmp_path):
        """record_access must not raise even if the sentinel cannot be written."""
        with patch("eostrata.cache.Path.touch", side_effect=OSError("read-only")):
            record_access(tmp_path, "worldpop/nga")  # should not raise

    def test_sentinel_affects_lru_order(self, tmp_path):
        """Touching the sentinel of the older group makes it appear newer."""
        _write_fake_group(tmp_path, "worldpop/nga")
        time.sleep(0.05)
        _write_fake_group(tmp_path, "chirps/global")

        # worldpop/nga is the oldest; touch its sentinel to make it the newest
        time.sleep(0.05)
        record_access(tmp_path, "worldpop/nga")

        groups = list_groups(tmp_path)
        # chirps/global should now be first (oldest)
        assert groups[0][0] == "chirps/global"
        assert groups[1][0] == "worldpop/nga"


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

    def test_exceeds_quota_no_groups_raises(self, tmp_path):
        """RuntimeError when store exceeds quota but no two-level groups found (line 151)."""
        with (
            patch("eostrata.cache.store_size_mb", return_value=100.0),
            patch("eostrata.cache.list_groups", return_value=[]),
            pytest.raises(RuntimeError, match="no groups found"),
        ):
            check_and_evict(tmp_path, quota_mb=10.0)

    def test_eviction_break_when_first_eviction_sufficient(self, tmp_path):
        """Eviction loop breaks early once quota is satisfied (line 159)."""
        with (
            patch("eostrata.cache.store_size_mb", return_value=10.0),
            patch(
                "eostrata.cache.list_groups",
                return_value=[
                    ("worldpop/nga", 7.0, 1.0),
                    ("chirps/global", 3.0, 2.0),
                ],
            ),
            patch("eostrata.cache.evict_group") as mock_evict,
        ):
            check_and_evict(tmp_path, quota_mb=5.0)

        mock_evict.assert_called_once_with(tmp_path, "worldpop/nga")

    def test_still_over_quota_after_all_evictions_raises(self, tmp_path):
        """RuntimeError when freed space (per list_groups) doesn't cover the quota (line 172)."""
        with (
            patch("eostrata.cache.store_size_mb", return_value=100.0),
            patch("eostrata.cache.list_groups", return_value=[("worldpop/nga", 1.0, 1.0)]),
            patch("eostrata.cache.evict_group"),
            pytest.raises(RuntimeError, match="Could not reduce store"),
        ):
            check_and_evict(tmp_path, quota_mb=50.0)
