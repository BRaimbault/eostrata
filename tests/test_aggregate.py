"""Tests for temporal aggregation logic."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from eostrata.aggregate import (
    _AGG_CACHE,
    _AGG_CACHE_LOCK,
    _agg_cache_key,
    _chunked_aggregate,
    _chunked_mean,
    _get_agg_cache,
    _parse_datetime_interval,
    _put_agg_cache,
    _strip_tz,
    apply_temporal_aggregation,
    invalidate_agg_cache_for_group,
    resolve_accessed_times,
)


def _make_da(years: list[int]) -> xr.DataArray:
    """Create a (time, y, x) DataArray with constant values equal to the year."""
    times = np.array([np.datetime64(f"{y}-01-01") for y in years])
    data = np.stack([np.full((4, 4), float(y), dtype="float32") for y in years])
    return xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={"time": times, "y": np.arange(4), "x": np.arange(4)},
    )


class TestStripTz:
    def test_utc_offset_stripped(self):
        assert _strip_tz("2021-01-01T00:00:00+00:00") == "2021-01-01T00:00:00"

    def test_z_suffix_stripped(self):
        assert _strip_tz("2021-01-01T00:00:00Z") == "2021-01-01T00:00:00"

    def test_positive_offset_stripped(self):
        assert _strip_tz("2021-06-15T12:00:00+05:30") == "2021-06-15T12:00:00"

    def test_date_only_unchanged(self):
        assert _strip_tz("2021-01-01") == "2021-01-01"

    def test_datetime_no_tz_unchanged(self):
        assert _strip_tz("2021-01-01T00:00:00") == "2021-01-01T00:00:00"


class TestParseDatetimeInterval:
    def test_none_returns_none_none(self):
        assert _parse_datetime_interval(None) == (None, None)

    def test_empty_string(self):
        assert _parse_datetime_interval("") == (None, None)

    def test_single_date(self):
        assert _parse_datetime_interval("2021-01-01") == ("2021-01-01", "2021-01-01")

    def test_interval(self):
        assert _parse_datetime_interval("2021-01-01/2022-12-31") == (
            "2021-01-01",
            "2022-12-31",
        )

    def test_open_start(self):
        start, end = _parse_datetime_interval("/2022-12-31")
        assert start is None
        assert end == "2022-12-31"

    def test_open_end(self):
        start, end = _parse_datetime_interval("2021-01-01/")
        assert start == "2021-01-01"
        assert end is None


class TestApplyTemporalAggregation:
    def test_no_time_dimension_passthrough(self):
        da = xr.DataArray(np.ones((4, 4)), dims=("y", "x"))
        result = apply_temporal_aggregation(da)
        assert result.dims == ("y", "x")
        assert float(result.mean()) == pytest.approx(1.0)

    def test_irrecoverably_non_monotonic_time_raises(self):
        """sortby that still produces non-monotonic time should raise ValueError."""
        from unittest.mock import patch

        da = _make_da([2022, 2021, 2020])  # reversed = non-monotonic
        # Patch sortby to return the same non-sorted DataArray so the second
        # monotonicity check also fails, reaching the raise on line 110.
        with (
            patch.object(type(da), "sortby", return_value=da),
            pytest.raises(ValueError, match="not monotonic"),
        ):
            apply_temporal_aggregation(da)

    def test_no_args_returns_last_timestep(self):
        da = _make_da([2020, 2021, 2022])
        result = apply_temporal_aggregation(da)
        assert result.dims == ("y", "x")
        assert float(result.mean()) == pytest.approx(2022.0)

    def test_datetime_slice_single(self):
        da = _make_da([2020, 2021, 2022])
        result = apply_temporal_aggregation(da, datetime_str="2021-01-01")
        assert float(result.mean()) == pytest.approx(2021.0)

    def test_datetime_slice_interval(self):
        da = _make_da([2020, 2021, 2022])
        result = apply_temporal_aggregation(da, datetime_str="2020-01-01/2021-12-31", agg="mean")
        assert float(result.mean()) == pytest.approx(2020.5)

    def test_agg_mean(self):
        da = _make_da([2020, 2021, 2022])
        result = apply_temporal_aggregation(da, agg="mean")
        assert float(result.mean()) == pytest.approx(2021.0)

    def test_agg_sum(self):
        da = _make_da([1, 2, 3])
        result = apply_temporal_aggregation(da, agg="sum")
        assert float(result.mean()) == pytest.approx(6.0)

    def test_agg_min(self):
        da = _make_da([2020, 2021, 2022])
        result = apply_temporal_aggregation(da, agg="min")
        assert float(result.mean()) == pytest.approx(2020.0)

    def test_agg_max(self):
        da = _make_da([2020, 2021, 2022])
        result = apply_temporal_aggregation(da, agg="max")
        assert float(result.mean()) == pytest.approx(2022.0)

    def test_agg_anomaly(self):
        da = _make_da([2020, 2021, 2022])
        # baseline mean = 2020.5, overall mean = 2021 → anomaly = 0.5
        result = apply_temporal_aggregation(
            da,
            datetime_str="2020-01-01/2022-12-31",
            agg="anomaly",
            baseline="2020-01-01/2021-12-31",
        )
        assert float(result.mean()) == pytest.approx(0.5)

    def test_agg_anomaly_requires_baseline(self):
        da = _make_da([2020, 2021])
        with pytest.raises(ValueError, match="baseline"):
            apply_temporal_aggregation(da, agg="anomaly")

    def test_agg_anomaly_empty_baseline(self):
        da = _make_da([2020, 2021])
        with pytest.raises(ValueError, match="No data found for baseline"):
            apply_temporal_aggregation(
                da,
                agg="anomaly",
                baseline="1990-01-01/1995-01-01",
            )

    def test_single_date_out_of_range_returns_nearest(self):
        # Single dates use nearest-neighbour — no error, returns closest timestep.
        da = _make_da([2020, 2021])
        result = apply_temporal_aggregation(da, datetime_str="1990-01-01")
        assert float(result.mean()) == pytest.approx(2020.0)

    def test_interval_with_no_data_raises(self):
        # A *range* that yields zero timesteps still raises.
        da = _make_da([2020, 2021])
        with pytest.raises(ValueError, match="No data found"):
            apply_temporal_aggregation(da, datetime_str="1990-01-01/1995-12-31", agg="mean")

    def test_unknown_agg_raises(self):
        da = _make_da([2020])
        with pytest.raises(ValueError, match="Unknown agg method"):
            apply_temporal_aggregation(da, agg="median")  # type: ignore[arg-type]

    def test_tz_aware_datetime_string_accepted(self):
        """UTC-offset datetime strings (from the STAC catalog) must not raise."""
        da = _make_da([2020, 2021])
        result = apply_temporal_aggregation(da, datetime_str="2020-01-01T00:00:00+00:00")
        assert float(result.mean()) == pytest.approx(2020.0)

    def test_tz_aware_interval_accepted(self):
        da = _make_da([2020, 2021, 2022])
        result = apply_temporal_aggregation(
            da,
            datetime_str="2020-01-01T00:00:00+00:00/2021-01-01T00:00:00+00:00",
            agg="mean",
        )
        assert float(result.mean()) == pytest.approx(2020.5)

    def test_non_monotonic_time_is_sorted(self):
        """Non-monotonic time axis is sorted before slicing (line 107)."""
        times = np.array(
            [np.datetime64("2022-01-01"), np.datetime64("2020-01-01"), np.datetime64("2021-01-01")]
        )
        data = np.stack([np.full((4, 4), float(y), dtype="float32") for y in [2022, 2020, 2021]])
        da = xr.DataArray(
            data,
            dims=("time", "y", "x"),
            coords={"time": times, "y": np.arange(4), "x": np.arange(4)},
        )
        result = apply_temporal_aggregation(da, datetime_str="2020-01-01/2021-01-01", agg="mean")
        assert float(result.mean()) == pytest.approx(2020.5)

    def test_open_end_interval_uses_nearest(self):
        """Open-end interval '2021-01-01/' → t0 set, t1=None → nearest match (line 124)."""
        da = _make_da([2020, 2021, 2022])
        result = apply_temporal_aggregation(da, datetime_str="2021-01-01/")
        assert float(result.mean()) == pytest.approx(2021.0)

    def test_corrupt_time_axis_still_non_monotonic_after_sort_raises(self, monkeypatch):
        """DataArray whose time axis stays non-monotonic after sortby raises ValueError."""
        da = _make_da([2022, 2020, 2021])
        monkeypatch.setattr(xr.DataArray, "sortby", lambda self, *a, **kw: self)
        with pytest.raises(ValueError, match="corrupt"):
            apply_temporal_aggregation(da, agg="mean")

    def test_duplicate_timestamps_deduplicated(self):
        """Duplicate timestamps are deduplicated before aggregation to avoid InvalidIndexError."""
        times = np.array(
            [np.datetime64("2021-01-01"), np.datetime64("2021-01-01"), np.datetime64("2022-01-01")]
        )
        data = np.stack(
            [np.full((4, 4), 2021.0), np.full((4, 4), 2021.0), np.full((4, 4), 2022.0)]
        ).astype("float32")
        da = xr.DataArray(
            data,
            dims=("time", "y", "x"),
            coords={"time": times, "y": np.arange(4), "x": np.arange(4)},
        )
        # Should not raise InvalidIndexError despite duplicate timestamps
        result = apply_temporal_aggregation(da, agg="mean")
        assert result.dims == ("y", "x")
        assert float(result.mean()) == pytest.approx((2021.0 + 2022.0) / 2)


    def test_agg_anomaly_baseline_outside_datetime_range(self):
        """Baseline period that does not overlap with datetime_str must still work.

        Regression test: da was sliced to datetime_str BEFORE baseline selection,
        making non-overlapping baselines always raise ValueError.
        """
        da = _make_da([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])
        # datetime_str is 2021-2022, baseline is 2015-2019 — no overlap.
        result = apply_temporal_aggregation(
            da,
            datetime_str="2021-01-01/2022-12-31",
            agg="anomaly",
            baseline="2015-01-01/2019-12-31",
        )
        # mean(2021,2022) = 2021.5; baseline mean(2015..2019) = 2017.0 → anomaly = 4.5
        assert float(result.mean()) == pytest.approx(4.5)


class TestChunkedAggregation:
    """Batched helpers produce the same result as single-pass xarray reductions."""

    def test_chunked_mean_exact(self):
        da = _make_da([1, 2, 3, 4, 5])
        result = _chunked_mean(da, batch_size=2)
        expected = da.mean("time").values
        np.testing.assert_allclose(result.values, expected, rtol=1e-5)

    def test_chunked_mean_single_batch(self):
        da = _make_da([10, 20])
        result = _chunked_mean(da, batch_size=10)
        np.testing.assert_allclose(result.values, da.mean("time").values, rtol=1e-5)

    def test_chunked_mean_via_aggregate(self):
        da = _make_da([1, 2, 3, 4])
        result = _chunked_aggregate(da, "mean", batch_size=2)
        np.testing.assert_allclose(result.values, da.mean("time").values, rtol=1e-5)

    def test_chunked_sum(self):
        da = _make_da([1, 2, 3, 4])
        result = _chunked_aggregate(da, "sum", batch_size=2)
        np.testing.assert_allclose(result.values, da.sum("time").values, rtol=1e-5)

    def test_chunked_aggregate_unknown_agg_raises(self):
        da = _make_da([2020, 2021])
        with pytest.raises(ValueError, match="Unknown agg method"):
            _chunked_aggregate(da, "median", batch_size=2)  # type: ignore[arg-type]

    def test_chunked_reduce_empty_raises(self):
        from eostrata.aggregate import _chunked_reduce

        empty = xr.DataArray(
            np.empty((0, 4, 4), dtype="float32"),
            dims=("time", "y", "x"),
            coords={
                "time": np.array([], dtype="datetime64[ns]"),
                "y": np.arange(4),
                "x": np.arange(4),
            },
        )
        with pytest.raises(ValueError, match="no time steps"):
            _chunked_reduce(empty, lambda b: b.sum("time"), lambda a, b: a + b, batch_size=2)

    def test_chunked_min(self):
        da = _make_da([5, 2, 8, 1, 3])
        result = _chunked_aggregate(da, "min", batch_size=2)
        np.testing.assert_allclose(result.values, da.min("time").values, rtol=1e-5)

    def test_chunked_max(self):
        da = _make_da([5, 2, 8, 1, 3])
        result = _chunked_aggregate(da, "max", batch_size=2)
        np.testing.assert_allclose(result.values, da.max("time").values, rtol=1e-5)

    def test_chunked_mean_nan_handling(self):
        """_chunked_mean must produce the same result as da.mean("time") for NaN data.

        Regression test: the original implementation used sum/n_total_timesteps,
        which gave a result 3x too small for a pixel with 2 NaN and 1 valid value.
        """
        times = np.array([np.datetime64(f"{y}-01-01") for y in [2020, 2021, 2022]])
        data = np.array(
            [
                [[1.0, np.nan], [np.nan, 4.0]],  # 2020
                [[np.nan, np.nan], [np.nan, np.nan]],  # 2021 — all NaN
                [[3.0, np.nan], [np.nan, 8.0]],  # 2022
            ],
            dtype="float32",
        )
        da = xr.DataArray(data, dims=("time", "y", "x"), coords={"time": times})
        result = _chunked_mean(da, batch_size=2)
        expected = da.mean("time").values
        np.testing.assert_allclose(result.values, expected, rtol=1e-5, equal_nan=True)

    def test_apply_uses_batched_path_when_limit_set(self, monkeypatch):
        """apply_temporal_aggregation routes to batched path when limit is exceeded."""
        import eostrata.aggregate as agg_mod
        import eostrata.config as cfg

        da = _make_da([2020, 2021, 2022, 2023, 2024])  # 5 timesteps
        monkeypatch.setattr(cfg.settings, "max_aggregation_timesteps", 2)
        # Force re-read of the setting inside the module
        monkeypatch.setattr(agg_mod._eostrata_config, "settings", cfg.settings)

        result = apply_temporal_aggregation(da, agg="mean")
        expected = da.mean("time").values
        np.testing.assert_allclose(result.values, expected, rtol=1e-5)

    def test_apply_batched_anomaly(self, monkeypatch):
        import eostrata.aggregate as agg_mod
        import eostrata.config as cfg

        da = _make_da([2018, 2019, 2020, 2021, 2022])
        monkeypatch.setattr(cfg.settings, "max_aggregation_timesteps", 2)
        monkeypatch.setattr(agg_mod._eostrata_config, "settings", cfg.settings)

        result = apply_temporal_aggregation(
            da,
            agg="anomaly",
            baseline="2018-01-01/2019-12-31",
        )
        expected = apply_temporal_aggregation(da, agg="anomaly", baseline="2018-01-01/2019-12-31")
        np.testing.assert_allclose(result.values, expected.values, rtol=1e-5)


class TestResolveAccessedTimes:
    def _make_ds(self, years: list[int]) -> xr.Dataset:
        times = np.array([np.datetime64(f"{y}-01-01") for y in years])
        data = np.ones((len(years), 4, 4), dtype="float32")
        return xr.Dataset(
            {"v": (("time", "y", "x"), data)},
            coords={"time": times},
        )

    def test_no_time_dim_returns_empty(self):
        ds = xr.Dataset({"v": (("y", "x"), np.ones((4, 4)))})
        assert resolve_accessed_times(ds, None) == []

    def test_time_values_access_error_returns_empty(self):
        """Dataset with time coord but values raises AttributeError → []."""
        import unittest.mock as mock

        ds = self._make_ds([2020])
        with mock.patch.object(
            type(ds["time"]), "values", new_callable=mock.PropertyMock, side_effect=AttributeError
        ):
            result = resolve_accessed_times(ds, None)
        assert result == []

    def test_empty_time_returns_empty(self):
        ds = xr.Dataset(
            {"v": (("time", "y", "x"), np.ones((0, 4, 4), dtype="float32"))},
            coords={"time": np.array([], dtype="datetime64[ns]")},
        )
        assert resolve_accessed_times(ds, None) == []

    def test_none_datetime_returns_last(self):
        ds = self._make_ds([2020, 2021, 2022])
        result = resolve_accessed_times(ds, None)
        assert len(result) == 1
        assert result[0].astype("datetime64[Y]").item().year == 2022

    def test_open_start_interval_returns_last(self):
        """Open-start interval (t0=None from _parse_datetime_interval) → last timestamp."""
        ds = self._make_ds([2020, 2021])
        result = resolve_accessed_times(ds, "/2021-12-31")
        assert len(result) == 1

    def test_single_point_returns_nearest(self):
        ds = self._make_ds([2020, 2021, 2022])
        result = resolve_accessed_times(ds, "2021-01-01")
        assert len(result) == 1
        assert result[0].astype("datetime64[Y]").item().year == 2021

    def test_interval_returns_range(self):
        ds = self._make_ds([2020, 2021, 2022])
        result = resolve_accessed_times(ds, "2020-01-01/2021-12-31")
        years = [r.astype("datetime64[Y]").item().year for r in result]
        assert sorted(years) == [2020, 2021]

    def test_anomaly_includes_baseline_timestamps(self):
        ds = self._make_ds([2018, 2019, 2020, 2021])
        result = resolve_accessed_times(ds, "2021-01-01", "anomaly", "2018-01-01/2019-12-31")
        years = sorted(r.astype("datetime64[Y]").item().year for r in result)
        assert years == [2018, 2019, 2021]

    def test_anomaly_no_duplicate_when_overlap(self):
        """A timestamp in both period and baseline should appear only once."""
        ds = self._make_ds([2020, 2021])
        result = resolve_accessed_times(
            ds, "2020-01-01/2021-12-31", "anomaly", "2020-01-01/2021-12-31"
        )
        assert len(result) == 2


class TestAggSemaphore:
    def test_unlimited_returns_none(self, monkeypatch):
        import eostrata.aggregate as agg_mod
        from eostrata.config import settings

        monkeypatch.setattr(settings, "max_concurrent_aggregations", 0)
        agg_mod._agg_semaphore = None
        assert agg_mod._get_agg_semaphore() is None

    def test_nullctx_is_noop(self):
        from eostrata.aggregate import _nullctx

        with _nullctx() as ctx:
            assert ctx is not None  # just exercises enter/exit

    def test_semaphore_created_with_limit(self, monkeypatch):
        import eostrata.aggregate as agg_mod
        from eostrata.config import settings

        monkeypatch.setattr(settings, "max_concurrent_aggregations", 2)
        agg_mod._agg_semaphore = None
        sem = agg_mod._get_agg_semaphore()
        assert sem is not None


class TestMaxConcurrentAggregationsValidator:
    def test_negative_raises(self):
        import pytest
        from pydantic import ValidationError

        from eostrata.config import Settings

        with pytest.raises(ValidationError):
            Settings(max_concurrent_aggregations=-1)

    def test_zero_is_valid(self):
        from eostrata.config import Settings

        s = Settings(max_concurrent_aggregations=0)
        assert s.max_concurrent_aggregations == 0


class TestAggCache:
    """Tests for the in-memory aggregation result cache."""

    def setup_method(self):
        """Clear the cache before each test to avoid inter-test pollution."""
        with _AGG_CACHE_LOCK:
            _AGG_CACHE.clear()

    def _make_da_2d(self) -> xr.DataArray:
        return xr.DataArray(
            np.ones((4, 4), dtype="float32"),
            dims=("y", "x"),
            coords={"y": np.arange(4.0), "x": np.arange(4.0)},
        )

    def test_cache_miss_returns_none(self):
        key = _agg_cache_key("path", "group", "var", "2021", "mean", None)
        assert _get_agg_cache(key) is None

    def test_put_then_get_returns_entry(self, monkeypatch):
        import eostrata.config as cfg

        monkeypatch.setattr(cfg.settings, "agg_cache_maxsize", 4)
        monkeypatch.setattr(cfg.settings, "agg_cache_ttl_seconds", 300)

        da = self._make_da_2d()
        key = _agg_cache_key("path", "group", "var", "2021", "mean", None)
        _put_agg_cache(key, da, [])
        result = _get_agg_cache(key)
        assert result is not None
        cached_da, accessed = result
        np.testing.assert_array_equal(cached_da.values, da.values)
        assert accessed == []

    def test_cache_ttl_expiry(self, monkeypatch):
        import eostrata.aggregate as agg_mod
        import eostrata.config as cfg

        monkeypatch.setattr(cfg.settings, "agg_cache_maxsize", 4)
        monkeypatch.setattr(cfg.settings, "agg_cache_ttl_seconds", 10)

        da = self._make_da_2d()
        key = _agg_cache_key("path", "group", "var", "2021", "mean", None)
        # Put entry with current time
        _put_agg_cache(key, da, [])
        assert _get_agg_cache(key) is not None

        # Advance monotonic clock past TTL
        monkeypatch.setattr(agg_mod._time, "monotonic", lambda: 1e12)
        assert _get_agg_cache(key) is None  # expired

    def test_cache_maxsize_evicts_oldest(self, monkeypatch):
        import eostrata.config as cfg

        monkeypatch.setattr(cfg.settings, "agg_cache_maxsize", 2)
        monkeypatch.setattr(cfg.settings, "agg_cache_ttl_seconds", 300)

        da = self._make_da_2d()
        key_a = _agg_cache_key("p", "g", "v", "2020", "mean", None)
        key_b = _agg_cache_key("p", "g", "v", "2021", "mean", None)
        key_c = _agg_cache_key("p", "g", "v", "2022", "mean", None)
        _put_agg_cache(key_a, da, [])
        _put_agg_cache(key_b, da, [])
        _put_agg_cache(key_c, da, [])  # should evict key_a (oldest)

        assert _get_agg_cache(key_a) is None
        assert _get_agg_cache(key_b) is not None
        assert _get_agg_cache(key_c) is not None

    def test_put_noop_when_maxsize_zero(self, monkeypatch):
        import eostrata.config as cfg

        monkeypatch.setattr(cfg.settings, "agg_cache_maxsize", 0)
        monkeypatch.setattr(cfg.settings, "agg_cache_ttl_seconds", 300)

        da = self._make_da_2d()
        key = _agg_cache_key("path", "group", "var", "2021", "mean", None)
        _put_agg_cache(key, da, [])
        assert _get_agg_cache(key) is None  # nothing stored

    def test_invalidate_removes_only_target_group(self, monkeypatch):
        import eostrata.config as cfg

        monkeypatch.setattr(cfg.settings, "agg_cache_maxsize", 8)
        monkeypatch.setattr(cfg.settings, "agg_cache_ttl_seconds", 300)

        da = self._make_da_2d()
        key_nga = _agg_cache_key("p", "worldpop/nga", "pop", "2021", "mean", None)
        key_tza = _agg_cache_key("p", "worldpop/tza", "pop", "2021", "mean", None)
        _put_agg_cache(key_nga, da, [])
        _put_agg_cache(key_tza, da, [])

        invalidate_agg_cache_for_group("worldpop/nga")

        assert _get_agg_cache(key_nga) is None
        assert _get_agg_cache(key_tza) is not None
