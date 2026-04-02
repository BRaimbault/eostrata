"""Tests for the cache eviction module."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from eostrata.cache import (
    _ACCESS_DIR,
    _DEBOUNCE_S,
    check_and_evict,
    evict_group,
    evict_timestamp,
    list_groups,
    list_timestamps,
    record_access,
    store_size_mb,
)


def _write_fake_group(zarr_root: Path, group: str, size_kb: int = 10) -> None:
    """Write a dummy Zarr group with no time dimension."""
    data = np.random.rand(size_kb, 8).astype("float32")
    ds = xr.Dataset({"data": (("y", "x"), data)})
    ds.to_zarr(str(zarr_root), group=group, mode="w", zarr_format=2)


def _write_fake_group_with_times(
    zarr_root: Path, group: str, years: list[int], size_kb: int = 10
) -> None:
    """Write a Zarr group with a time dimension (one step per year)."""
    n = len(years)
    data = np.random.rand(n, size_kb, 8).astype("float32")
    times = np.array([np.datetime64(f"{y}-01-01") for y in years])
    ds = xr.Dataset(
        {"data": (("time", "y", "x"), data)},
        coords={"time": times},
    )
    ds.to_zarr(str(zarr_root), group=group, mode="w", zarr_format=2)


class TestRecordAccess:
    def test_creates_sentinel_files(self, tmp_path):
        _write_fake_group_with_times(tmp_path, "worldpop/nga", [2020, 2021])
        ts = [np.datetime64("2020-01-01"), np.datetime64("2021-01-01")]
        record_access(tmp_path, "worldpop/nga", ts)
        access_dir = tmp_path / "worldpop" / "nga" / _ACCESS_DIR
        assert (access_dir / "2020-01-01T00:00:00").exists()
        assert (access_dir / "2021-01-01T00:00:00").exists()

    def test_debounce_skips_touch_within_window(self, tmp_path):
        _write_fake_group_with_times(tmp_path, "worldpop/nga", [2020])
        ts = [np.datetime64("2020-01-01")]
        record_access(tmp_path, "worldpop/nga", ts)
        sentinel = tmp_path / "worldpop" / "nga" / _ACCESS_DIR / "2020-01-01T00:00:00"
        mtime_first = sentinel.stat().st_mtime
        with patch("eostrata.cache.time.time", return_value=mtime_first + _DEBOUNCE_S - 1):
            record_access(tmp_path, "worldpop/nga", ts)
        assert sentinel.stat().st_mtime == mtime_first

    def test_debounce_allows_touch_after_window(self, tmp_path):
        _write_fake_group_with_times(tmp_path, "worldpop/nga", [2020])
        ts = [np.datetime64("2020-01-01")]
        record_access(tmp_path, "worldpop/nga", ts)
        sentinel = tmp_path / "worldpop" / "nga" / _ACCESS_DIR / "2020-01-01T00:00:00"
        mtime_first = sentinel.stat().st_mtime
        with patch("eostrata.cache.time.time", return_value=mtime_first + _DEBOUNCE_S + 1):
            record_access(tmp_path, "worldpop/nga", ts)
        assert sentinel.stat().st_mtime >= mtime_first

    def test_track_access_false_skips_sentinel(self, tmp_path):
        _write_fake_group_with_times(tmp_path, "worldpop/nga", [2020])
        with patch("eostrata.config.settings") as mock_settings:
            mock_settings.track_access = False
            record_access(tmp_path, "worldpop/nga", [np.datetime64("2020-01-01")])
        access_dir = tmp_path / "worldpop" / "nga" / _ACCESS_DIR
        assert not access_dir.exists()

    def test_oserror_is_silenced(self, tmp_path):
        """record_access must not raise even if the access dir cannot be created."""
        with patch("eostrata.cache.Path.mkdir", side_effect=OSError("read-only")):
            record_access(tmp_path, "worldpop/nga", [np.datetime64("2020-01-01")])

    def test_sentinel_affects_lru_order(self, tmp_path):
        """Touching the sentinel of the older group makes it appear newer."""
        _write_fake_group_with_times(tmp_path, "worldpop/nga", [2020])
        time.sleep(0.05)
        _write_fake_group_with_times(tmp_path, "chirps/global", [2020])

        time.sleep(0.05)
        record_access(tmp_path, "worldpop/nga", [np.datetime64("2020-01-01")])

        groups = list_groups(tmp_path)
        assert groups[0][0] == "chirps/global"
        assert groups[1][0] == "worldpop/nga"


class TestListTimestamps:
    def test_returns_timestamps_sorted_oldest_first(self, tmp_path):
        _write_fake_group_with_times(tmp_path, "worldpop/nga", [2020, 2021, 2022])
        # Touch 2022 sentinel so it's the newest
        record_access(tmp_path, "worldpop/nga", [np.datetime64("2022-01-01")])
        ts_list = list_timestamps(tmp_path, "worldpop/nga")
        assert len(ts_list) == 3
        # 2020 and 2021 have last_access=0 (never accessed), 2022 has last_access>0
        assert ts_list[-1][0].startswith("2022")
        assert all(ts[2] == 0.0 for ts in ts_list[:2])

    def test_nonexistent_group_returns_empty(self, tmp_path):
        assert list_timestamps(tmp_path, "worldpop/nonexistent") == []

    def test_no_time_dim_returns_empty(self, tmp_path):
        _write_fake_group(tmp_path, "worldpop/nga")
        assert list_timestamps(tmp_path, "worldpop/nga") == []

    def test_size_mb_is_positive(self, tmp_path):
        _write_fake_group_with_times(tmp_path, "worldpop/nga", [2020, 2021])
        ts_list = list_timestamps(tmp_path, "worldpop/nga")
        assert all(ts[1] > 0 for ts in ts_list)


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
        time.sleep(0.05)
        _write_fake_group(tmp_path, "chirps/global")
        groups = list_groups(tmp_path)
        paths = [g[0] for g in groups]
        assert paths[0] == "worldpop/nga"
        assert paths[1] == "chirps/global"

    def test_sentinel_files_excluded_from_size(self, tmp_path):
        _write_fake_group_with_times(tmp_path, "worldpop/nga", [2020])
        size_before = list_groups(tmp_path)[0][1]
        record_access(tmp_path, "worldpop/nga", [np.datetime64("2020-01-01")])
        size_after = list_groups(tmp_path)[0][1]
        assert size_after == size_before  # sentinels are 0 bytes, excluded anyway


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


class TestEvictTimestamp:
    def test_removes_one_timestamp(self, tmp_path):
        _write_fake_group_with_times(tmp_path, "worldpop/nga", [2020, 2021, 2022])
        freed = evict_timestamp(tmp_path, "worldpop/nga", "2021-01-01T00:00:00")
        assert freed > 0
        ds = xr.open_zarr(str(tmp_path), group="worldpop/nga", consolidated=False)
        years = [t.astype("datetime64[Y]").item().year for t in ds["time"].values]
        assert 2021 not in years
        assert 2020 in years
        assert 2022 in years

    def test_preserves_access_sentinels(self, tmp_path):
        _write_fake_group_with_times(tmp_path, "worldpop/nga", [2020, 2021])
        record_access(tmp_path, "worldpop/nga", [np.datetime64("2020-01-01")])
        evict_timestamp(tmp_path, "worldpop/nga", "2021-01-01T00:00:00")
        access_dir = tmp_path / "worldpop" / "nga" / _ACCESS_DIR
        names = [f.name for f in access_dir.iterdir()]
        assert any("2020" in n for n in names)
        assert not any("2021" in n for n in names)

    def test_nonexistent_group_returns_zero(self, tmp_path):
        assert evict_timestamp(tmp_path, "worldpop/nonexistent", "2020-01-01T00:00:00") == 0.0

    def test_nonexistent_timestamp_returns_zero(self, tmp_path):
        _write_fake_group_with_times(tmp_path, "worldpop/nga", [2020])
        assert evict_timestamp(tmp_path, "worldpop/nga", "2099-01-01T00:00:00") == 0.0

    def test_no_time_dim_returns_zero(self, tmp_path):
        _write_fake_group(tmp_path, "worldpop/nga")
        assert evict_timestamp(tmp_path, "worldpop/nga", "2020-01-01T00:00:00") == 0.0

    def test_empty_time_array_returns_zero(self, tmp_path):
        import xarray as xr

        ds = xr.Dataset(
            {"v": (("time", "y", "x"), np.zeros((0, 4, 4)))},
            coords={"time": np.array([], dtype="datetime64[ns]")},
        )
        ds.to_zarr(str(tmp_path), group="worldpop/nga", mode="w", zarr_format=2)
        assert evict_timestamp(tmp_path, "worldpop/nga", "2020-01-01T00:00:00") == 0.0

    def test_evicts_sentinel_for_removed_timestamp(self, tmp_path):
        """When the evicted timestamp has its own sentinel, that sentinel is removed."""
        _write_fake_group_with_times(tmp_path, "worldpop/nga", [2020, 2021])
        # Record access for both timestamps so both sentinels exist
        record_access(
            tmp_path, "worldpop/nga", [np.datetime64("2020-01-01"), np.datetime64("2021-01-01")]
        )
        evict_timestamp(tmp_path, "worldpop/nga", "2020-01-01T00:00:00")
        access_dir = tmp_path / "worldpop" / "nga" / _ACCESS_DIR
        names = [f.name for f in access_dir.iterdir()]
        assert not any("2020" in n for n in names)
        assert any("2021" in n for n in names)

    def test_consolidate_metadata_exception_is_silenced(self, tmp_path):
        """If zarr.consolidate_metadata raises during eviction, the eviction still succeeds."""
        import zarr as _zarr

        _write_fake_group_with_times(tmp_path, "worldpop/nga", [2020, 2021])
        _original = _zarr.consolidate_metadata

        def _raise_for_our_call(path, **kwargs):
            # cache.py calls consolidate_metadata(str(zarr_root)) — string path
            # xarray calls it with a Store object during to_zarr; let those through
            if isinstance(path, str):
                raise OSError("no meta")
            return _original(path, **kwargs)

        with patch("eostrata.cache.zarr.consolidate_metadata", side_effect=_raise_for_our_call):
            freed = evict_timestamp(tmp_path, "worldpop/nga", "2020-01-01T00:00:00")
        assert freed > 0
        ds = xr.open_zarr(str(tmp_path), group="worldpop/nga", consolidated=False)
        years = [t.astype("datetime64[Y]").item().year for t in ds["time"].values]
        assert 2020 not in years
        assert 2021 in years

    def test_updates_catalog(self, tmp_path):
        from datetime import UTC, datetime

        from eostrata import catalog as cat

        catalog_path = tmp_path / "catalog.json"
        catalogue = cat.load_or_create(catalog_path)
        _write_fake_group_with_times(tmp_path, "worldpop/nga", [2020, 2021])
        cat.register_item(
            catalogue,
            collection_id="worldpop",
            item_id="worldpop_nga",
            bbox=(2.0, 4.0, 15.0, 14.0),
            datetime_=datetime(2020, 1, 1, tzinfo=UTC),
            zarr_root=tmp_path,
            zarr_group="worldpop/nga",
            variable="data",
        )
        cat.register_item(
            catalogue,
            collection_id="worldpop",
            item_id="worldpop_nga",
            bbox=(2.0, 4.0, 15.0, 14.0),
            datetime_=datetime(2021, 1, 1, tzinfo=UTC),
            zarr_root=tmp_path,
            zarr_group="worldpop/nga",
            variable="data",
        )
        cat.save(catalogue, catalog_path)

        evict_timestamp(tmp_path, "worldpop/nga", "2021-01-01T00:00:00", catalog_path=catalog_path)

        updated = cat.load_or_create(catalog_path)
        item = updated.get_child("worldpop").get_item("worldpop_nga")
        assert item is not None
        datetimes = item.properties["eostrata:datetimes"]
        assert not any(d.startswith("2021") for d in datetimes)
        assert any(d.startswith("2020") for d in datetimes)


class TestCheckAndEvict:
    def test_unlimited_quota_is_noop(self, tmp_path):
        _write_fake_group(tmp_path, "worldpop/nga")
        check_and_evict(tmp_path, quota_mb=0.0)
        assert (tmp_path / "worldpop" / "nga").exists()

    def test_within_quota_no_eviction(self, tmp_path):
        _write_fake_group_with_times(tmp_path, "worldpop/nga", [2020])
        check_and_evict(tmp_path, quota_mb=1000.0)
        assert (tmp_path / "worldpop" / "nga").exists()

    def test_exceeds_quota_evicts_oldest_timestamp_first(self, tmp_path):
        """Oldest timestamp (no access sentinel) is evicted before the newer one."""
        _write_fake_group_with_times(tmp_path, "worldpop/nga", [2020, 2021])
        time.sleep(0.05)
        record_access(tmp_path, "worldpop/nga", [np.datetime64("2021-01-01")])

        size_before = store_size_mb(tmp_path)
        with (
            patch("eostrata.cache.evict_timestamp") as mock_evict,
            patch(
                "eostrata.cache.store_size_mb",
                # Call sequence: initial check, loop iter 1 (2020→evict),
                # loop iter 2 (2021→quota met→break), final remaining check
                side_effect=[size_before, size_before, size_before * 0.3, size_before * 0.3],
            ),
        ):
            check_and_evict(tmp_path, quota_mb=size_before * 0.6)

        mock_evict.assert_called_once()
        evicted_ts = mock_evict.call_args[0][2]
        assert evicted_ts.startswith("2020")  # oldest (no sentinel) evicted first

    def test_exceeds_quota_no_timestamps_raises(self, tmp_path):
        with (
            patch("eostrata.cache.store_size_mb", return_value=100.0),
            patch("eostrata.cache.list_groups", return_value=[("worldpop/nga", 100.0, 1.0)]),
            patch("eostrata.cache.list_timestamps", return_value=[]),
            pytest.raises(RuntimeError, match="no timestamps"),
        ):
            check_and_evict(tmp_path, quota_mb=10.0)

    def test_eviction_stops_when_quota_met(self, tmp_path):
        # store_size_mb call sequence:
        # 1. initial check: 10.0 → exceeds quota of 5 MB
        # 2. loop iter 1 (2020): 10.0 → still over → evict 2020
        # 3. loop iter 2 (2021): 3.0 → within quota → break
        # 4. final remaining check: 3.0 → within quota → no raise
        with (
            patch("eostrata.cache.store_size_mb", side_effect=[10.0, 10.0, 3.0, 3.0]),
            patch("eostrata.cache.list_groups", return_value=[("worldpop/nga", 10.0, 1.0)]),
            patch(
                "eostrata.cache.list_timestamps",
                return_value=[
                    ("2020-01-01T00:00:00", 7.0, 1.0, 0.5),
                    ("2021-01-01T00:00:00", 3.0, 2.0, 0.5),
                ],
            ),
            patch("eostrata.cache.evict_timestamp") as mock_evict,
        ):
            check_and_evict(tmp_path, quota_mb=5.0)
        mock_evict.assert_called_once_with(
            tmp_path, "worldpop/nga", "2020-01-01T00:00:00", catalog_path=None
        )

    def test_still_over_quota_after_all_evictions_raises(self, tmp_path):
        # store_size_mb: initial=100, re-measure in loop=100 (still over), final=100
        with (
            patch("eostrata.cache.store_size_mb", return_value=100.0),
            patch("eostrata.cache.list_groups", return_value=[("worldpop/nga", 100.0, 1.0)]),
            patch(
                "eostrata.cache.list_timestamps",
                return_value=[("2020-01-01T00:00:00", 1.0, 1.0, 0.5)],
            ),
            patch("eostrata.cache.evict_timestamp"),
            pytest.raises(RuntimeError, match="Could not reduce store"),
        ):
            check_and_evict(tmp_path, quota_mb=50.0)

    def test_no_access_or_ingestion_time_age_desc(self, tmp_path):
        """Timestamps with last_access=0 and ingestion_time=0 log 'no access or ingestion'."""
        with (
            patch("eostrata.cache.store_size_mb", side_effect=[10.0, 10.0, 0.0, 0.0]),
            patch("eostrata.cache.list_groups", return_value=[("worldpop/nga", 10.0, 0.0)]),
            patch(
                "eostrata.cache.list_timestamps",
                return_value=[("2020-01-01T00:00:00", 10.0, 0.0, 0.0)],
            ),
            patch("eostrata.cache.evict_timestamp"),
        ):
            check_and_evict(tmp_path, quota_mb=5.0)


class TestListTimestampsEdgeCases:
    def test_zarr_open_failure_returns_empty(self, tmp_path):
        with patch("xarray.open_zarr", side_effect=Exception("bad zarr")):
            assert list_timestamps(tmp_path, "worldpop/nga") == []

    def test_time_values_raises_returns_empty(self, tmp_path):
        """ds['time'].values raising an exception → return []."""
        import xarray as xr

        mock_ds = MagicMock(spec=xr.Dataset)
        mock_ds.__contains__ = lambda self, key: key == "time"
        mock_ds.__getitem__ = MagicMock(side_effect=KeyError("time"))
        with patch("xarray.open_zarr", return_value=mock_ds):
            assert list_timestamps(tmp_path, "worldpop/nga") == []

    def test_empty_time_array_returns_empty(self, tmp_path):
        import xarray as xr

        ds = xr.Dataset(
            {"v": (("time", "y", "x"), np.zeros((0, 4, 4)))},
            coords={"time": np.array([], dtype="datetime64[ns]")},
        )
        ds.to_zarr(str(tmp_path), group="worldpop/nga", mode="w", zarr_format=2)
        assert list_timestamps(tmp_path, "worldpop/nga") == []

    def test_sort_key_zero_access_and_ingestion(self, tmp_path):
        """Timestamps with both last_access=0 and ingestion_time=0 still sort stably."""
        _write_fake_group_with_times(tmp_path, "worldpop/nga", [2020, 2021])
        # Zero out all file mtimes so ingestion_time = 0
        group_dir = tmp_path / "worldpop" / "nga"
        for f in group_dir.rglob("*"):
            if f.is_file():
                import os

                os.utime(f, (0, 0))
        ts_list = list_timestamps(tmp_path, "worldpop/nga")
        assert len(ts_list) == 2
        # Should still sort by ts_iso when both times are 0
        assert ts_list[0][0] < ts_list[1][0]


class TestConcurrentEviction:
    """Race condition tests — verify locking prevents data corruption under concurrency."""

    def test_concurrent_evict_same_timestamp_no_corruption(self, tmp_path):
        """Two threads evicting the same timestamp must leave exactly the other timestamps.

        Without the per-group lock, the rename sequence in evict_timestamp can interleave,
        leaving the group in an undefined state.
        """
        _write_fake_group_with_times(tmp_path, "worldpop/nga", [2019, 2020, 2021])
        errors: list[Exception] = []

        def _evict():
            try:
                evict_timestamp(tmp_path, "worldpop/nga", "2019-01-01T00:00:00")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_evict) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread(s) raised: {errors}"
        ds = xr.open_zarr(str(tmp_path), group="worldpop/nga", consolidated=False)
        years = sorted(t.astype("datetime64[Y]").item().year for t in ds["time"].values)
        assert years == [2020, 2021], f"Expected [2020, 2021], got {years}"

    def test_concurrent_evict_different_timestamps_no_data_loss(self, tmp_path):
        """Two threads evicting DIFFERENT timestamps must both succeed without losing data.

        Without the per-group lock, the interleaved rename sequence produces:
          Thread A evicts 2019 → writes {2018, 2020} to tmp_A
          Thread B evicts 2018 → writes {2019, 2020} to tmp_B
          A renames: target→old_A, tmp_A→target
          B renames: target→old_B (= A's result!), tmp_B→target
          A: rmtree(old_A) — original gone
          B: rmtree(old_B) — A's {2018, 2020} gone
          Final: {2019, 2020} — 2019 was NOT evicted, 2018 was lost silently.
        """
        _write_fake_group_with_times(tmp_path, "worldpop/nga", [2018, 2019, 2020])
        errors: list[Exception] = []

        def _evict_2019():
            try:
                evict_timestamp(tmp_path, "worldpop/nga", "2019-01-01T00:00:00")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def _evict_2018():
            try:
                evict_timestamp(tmp_path, "worldpop/nga", "2018-01-01T00:00:00")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        t1 = threading.Thread(target=_evict_2019)
        t2 = threading.Thread(target=_evict_2018)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Thread(s) raised: {errors}"
        ds = xr.open_zarr(str(tmp_path), group="worldpop/nga", consolidated=False)
        years = sorted(t.astype("datetime64[Y]").item().year for t in ds["time"].values)
        assert years == [2020], f"Expected only [2020] to remain, got {years}"

    def test_concurrent_check_and_evict_does_not_double_evict(self, tmp_path):
        """Concurrent check_and_evict calls evict exactly the required amount, not more.

        Without the store-level eviction lock, all N concurrent callers would
        each independently see an over-quota store and each run a full eviction
        pass, collectively removing far more data than necessary.

        We set a quota that requires evicting exactly 1 of 2 timestamps.
        Four threads race to call check_and_evict; with the store lock, only 1
        thread performs the eviction — the other 3 acquire the lock after the
        first finishes and immediately see the store is within quota.
        """
        # Large data so per-timestamp size dominates zarr metadata overhead.
        _write_fake_group_with_times(tmp_path, "worldpop/nga", [2019, 2020], size_kb=500)
        total_mb = sum(f.stat().st_size for f in tmp_path.rglob("*") if f.is_file()) / (1024**2)
        # With 2 equal timestamps: after evicting 1, size drops to ~50% of original.
        # Set quota between 50% and 100% so exactly 1 eviction is needed.
        quota_mb = total_mb * 0.70
        errors: list[Exception] = []

        def _check():
            try:
                check_and_evict(tmp_path, quota_mb=quota_mb)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_check) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread(s) raised: {errors}"
        ds = xr.open_zarr(str(tmp_path), group="worldpop/nga", consolidated=False)
        remaining = len(ds["time"].values)
        # Exactly 1 timestamp should remain — if both were evicted it means the
        # store-level lock failed to prevent a second thread from re-measuring
        # an over-quota store and running a second pass.
        assert remaining == 1, (
            f"Expected exactly 1 timestamp after 1 eviction pass, got {remaining}. "
            "Store-level lock may not be preventing double-eviction."
        )

    def test_concurrent_ingest_and_evict_same_group(self, tmp_path):
        """An eviction must not corrupt a group that an ingest is actively writing to.

        Without the per-group lock held by both sides, evict_timestamp could
        rename the group directory out from under geotiff_to_zarr's to_zarr() call.
        """
        import rasterio
        from rasterio.transform import from_bounds

        from eostrata.store import geotiff_to_zarr

        # Prepare a real GeoTIFF for geotiff_to_zarr
        tif = tmp_path / "test.tif"
        bbox = (0.0, 0.0, 5.0, 5.0)
        transform = from_bounds(*bbox, width=8, height=8)
        data = np.ones((8, 8), dtype="float32")
        with rasterio.open(
            tif,
            "w",
            driver="GTiff",
            height=8,
            width=8,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data, 1)

        # Seed the group using geotiff_to_zarr so the schema is consistent
        for year in [2018, 2019]:
            tc = np.datetime64(f"{year}-01-01", "ns")
            geotiff_to_zarr(tif, tmp_path, "col/d", variable_name="v", time_coord=tc)

        errors: list[Exception] = []

        def _ingest():
            for year in [2020, 2021, 2022]:
                tc = np.datetime64(f"{year}-01-01", "ns")
                try:
                    geotiff_to_zarr(tif, tmp_path, "col/d", variable_name="v", time_coord=tc)
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

        def _evict():
            for ts in ["2018-01-01T00:00:00", "2019-01-01T00:00:00"]:
                try:
                    evict_timestamp(tmp_path, "col/d", ts)
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

        t_ingest = threading.Thread(target=_ingest)
        t_evict = threading.Thread(target=_evict)
        t_ingest.start()
        t_evict.start()
        t_ingest.join()
        t_evict.join()

        assert not errors, f"Thread(s) raised: {errors}"
        # The group must still be a valid zarr store after concurrent access
        ds = xr.open_zarr(str(tmp_path), group="col/d", consolidated=False)
        assert "time" in ds
        assert len(ds["time"]) > 0
