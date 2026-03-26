"""Tests for temporal aggregation logic."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from eostrata.aggregate import _parse_datetime_interval, apply_temporal_aggregation


def _make_da(years: list[int]) -> xr.DataArray:
    """Create a (time, y, x) DataArray with constant values equal to the year."""
    times = np.array([np.datetime64(f"{y}-01-01") for y in years])
    data = np.stack([np.full((4, 4), float(y), dtype="float32") for y in years])
    return xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={"time": times, "y": np.arange(4), "x": np.arange(4)},
    )


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

    def test_empty_time_slice_raises(self):
        da = _make_da([2020, 2021])
        with pytest.raises(ValueError, match="No data found"):
            apply_temporal_aggregation(da, datetime_str="1990-01-01")

    def test_unknown_agg_raises(self):
        da = _make_da([2020])
        with pytest.raises(ValueError, match="Unknown agg method"):
            apply_temporal_aggregation(da, agg="median")  # type: ignore[arg-type]
