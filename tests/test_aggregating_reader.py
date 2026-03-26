"""Tests for AggregatingReader — titiler.xarray Reader subclass."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from eostrata.aggregate import AggregatingReader


def _make_zarr_with_time(tmp_path: Path) -> Path:
    """Write a tiny Zarr with a time dimension and return the root."""
    zarr_root = tmp_path / "zarr"
    times = np.array(
        [np.datetime64("2020-01-01"), np.datetime64("2021-01-01")], dtype="datetime64[ns]"
    )
    data = np.stack([np.full((8, 8), float(y), dtype="float32") for y in [2020, 2021]])
    ds = xr.Dataset(
        {"population": (("time", "y", "x"), data)},
        coords={
            "time": times,
            "y": np.linspace(14.0, 4.0, 8),
            "x": np.linspace(2.0, 15.0, 8),
        },
    )
    ds.to_zarr(str(zarr_root), group="worldpop/nga", mode="w", consolidated=True)
    return zarr_root


class TestAggregatingReader:
    def test_get_variable_no_agg_returns_2d(self, tmp_path):
        zarr_root = _make_zarr_with_time(tmp_path)
        reader = AggregatingReader(
            str(zarr_root),
            variable="population",
            group="worldpop/nga",
        )
        ds = xr.open_zarr(str(zarr_root), group="worldpop/nga", consolidated=True)
        da = reader.get_variable(ds, "population")
        assert "time" not in da.dims
        # No agg → last timestep → 2021
        assert float(da.mean()) == pytest.approx(2021.0)

    def test_get_variable_with_agg_mean(self, tmp_path):
        zarr_root = _make_zarr_with_time(tmp_path)
        reader = AggregatingReader(
            str(zarr_root),
            variable="population",
            group="worldpop/nga",
        )
        reader._agg_method = "mean"
        reader._agg_datetime = "2020-01-01/2021-12-31"
        reader._agg_baseline = None

        ds = xr.open_zarr(str(zarr_root), group="worldpop/nga", consolidated=True)
        da = reader.get_variable(ds, "population")
        assert "time" not in da.dims
        assert float(da.mean()) == pytest.approx(2020.5)

    def test_get_variable_no_time_passthrough(self, tmp_path):
        zarr_root = tmp_path / "zarr"
        ds_2d = xr.Dataset(
            {"v": (("y", "x"), np.ones((4, 4), dtype="float32"))},
            coords={"y": np.arange(4.0), "x": np.arange(4.0)},
        )
        ds_2d.to_zarr(str(zarr_root), group="test/v", mode="w", consolidated=True)

        reader = AggregatingReader(str(zarr_root), variable="v", group="test/v")
        reader._agg_method = "mean"
        ds = xr.open_zarr(str(zarr_root), group="test/v", consolidated=True)
        da = reader.get_variable(ds, "v")
        # 2D — aggregation should be skipped
        assert da.dims == ("y", "x")
