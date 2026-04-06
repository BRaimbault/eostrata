"""Tests for AggregatingReader — titiler.xarray Reader subclass."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from eostrata.aggregate import (
    _AGG_CACHE,
    _AGG_CACHE_LOCK,
    _CTX_AGG_BASELINE,
    _CTX_AGG_DATETIME,
    _CTX_AGG_METHOD,
    AggregatingReader,
)


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

    def test_context_var_agg_mean(self, tmp_path):
        """Context vars are stored and aggregation is deferred to tile().

        reader.input is now the last-timestep placeholder; aggregation happens
        in tile() after spatial pre-clip.  Verify the context is stored and that
        applying temporal aggregation to _unagg_input yields the correct result.
        """
        zarr_root = _make_zarr_with_time(tmp_path)
        _CTX_AGG_DATETIME.set("2020-01-01/2021-12-31")
        _CTX_AGG_METHOD.set("mean")
        _CTX_AGG_BASELINE.set(None)
        try:
            reader = AggregatingReader(
                str(zarr_root),
                variable="population",
                group="worldpop/nga",
            )
        finally:
            _CTX_AGG_DATETIME.set(None)
            _CTX_AGG_METHOD.set(None)
            _CTX_AGG_BASELINE.set(None)
        # reader.input is the last-timestep placeholder (2D, no time dim)
        assert "time" not in reader.input.dims
        assert reader._tile_method == "mean"
        assert reader._tile_datetime == "2020-01-01/2021-12-31"
        # The unaggregated array is stored; aggregating it gives the correct result.
        from eostrata.aggregate import apply_temporal_aggregation

        result = apply_temporal_aggregation(
            reader._unagg_input, datetime_str="2020-01-01/2021-12-31", agg="mean"
        )
        assert float(result.mean()) == pytest.approx(2020.5)

    def test_context_var_single_datetime(self, tmp_path):
        """Single datetime: context stored, unagg_input sliced correctly."""
        zarr_root = _make_zarr_with_time(tmp_path)
        _CTX_AGG_DATETIME.set("2020-01-01")
        _CTX_AGG_METHOD.set(None)
        _CTX_AGG_BASELINE.set(None)
        try:
            reader = AggregatingReader(
                str(zarr_root),
                variable="population",
                group="worldpop/nga",
            )
        finally:
            _CTX_AGG_DATETIME.set(None)
            _CTX_AGG_METHOD.set(None)
            _CTX_AGG_BASELINE.set(None)
        assert "time" not in reader.input.dims
        assert reader._tile_datetime == "2020-01-01"
        assert reader._tile_method is None
        from eostrata.aggregate import apply_temporal_aggregation

        result = apply_temporal_aggregation(reader._unagg_input, datetime_str="2020-01-01")
        assert float(result.mean()) == pytest.approx(2020.0)

    def test_context_var_anomaly(self, tmp_path):
        """Anomaly agg context stored; applying it to _unagg_input yields correct deviation."""
        zarr_root = _make_zarr_with_time(tmp_path)
        _CTX_AGG_DATETIME.set("2020-01-01/2021-12-31")
        _CTX_AGG_METHOD.set("anomaly")
        _CTX_AGG_BASELINE.set("2020-01-01/2020-12-31")
        try:
            reader = AggregatingReader(
                str(zarr_root),
                variable="population",
                group="worldpop/nga",
            )
        finally:
            _CTX_AGG_DATETIME.set(None)
            _CTX_AGG_METHOD.set(None)
            _CTX_AGG_BASELINE.set(None)
        assert "time" not in reader.input.dims
        assert reader._tile_method == "anomaly"
        assert reader._tile_baseline == "2020-01-01/2020-12-31"
        # mean(2020,2021)=2020.5, baseline mean(2020)=2020 → anomaly=0.5
        from eostrata.aggregate import apply_temporal_aggregation

        result = apply_temporal_aggregation(
            reader._unagg_input,
            datetime_str="2020-01-01/2021-12-31",
            agg="anomaly",
            baseline="2020-01-01/2020-12-31",
        )
        assert float(result.mean()) == pytest.approx(0.5)

    def test_tile_clips_spatially_before_aggregation(self, tmp_path):
        """tile() renders the tile after spatial pre-clip + temporal aggregation."""
        import morecantile

        zarr_root = _make_zarr_with_time(tmp_path)
        _CTX_AGG_DATETIME.set("2020-01-01/2021-12-31")
        _CTX_AGG_METHOD.set("mean")
        _CTX_AGG_BASELINE.set(None)
        try:
            reader = AggregatingReader(
                str(zarr_root),
                variable="population",
                group="worldpop/nga",
            )
        finally:
            _CTX_AGG_DATETIME.set(None)
            _CTX_AGG_METHOD.set(None)
            _CTX_AGG_BASELINE.set(None)

        # Find a zoom-3 tile that covers the test data (lon 2–15°E, lat 4–14°N)
        tms = morecantile.tms.get("WebMercatorQuad")
        tile = next(tms.tiles(2.0, 4.0, 15.0, 14.0, zooms=3))
        img = reader.tile(tile.x, tile.y, tile.z)
        # tile() should return an ImageData object; pixel values for valid data
        # pixels should be near the expected mean (2020.5 ≈ 2020.5).
        data = img.data[np.isfinite(img.data)]  # exclude NaN out-of-extent pixels
        assert data.size > 0
        # Values are float32 filled with years; mean of 2020 and 2021 is 2020.5
        assert float(data.mean()) == pytest.approx(2020.5, abs=1.0)

    def test_valid_time_coord_renamed_to_time(self, tmp_path):
        """AggregatingReader renames valid_time → time in __attrs_post_init__ (line 190)."""
        zarr_root = tmp_path / "zarr"
        times = np.array([np.datetime64("2020-01-01")], dtype="datetime64[ns]")
        data = np.ones((1, 8, 8), dtype="float32") * 42.0
        ds = xr.Dataset(
            {"t2m": (("valid_time", "y", "x"), data)},
            coords={
                "valid_time": times,
                "y": np.linspace(14.0, 4.0, 8),
                "x": np.linspace(2.0, 15.0, 8),
            },
        )
        ds.to_zarr(str(zarr_root), group="era5/t2m", mode="w", consolidated=True)

        reader = AggregatingReader(str(zarr_root), variable="t2m", group="era5/t2m")
        # After renaming valid_time → time and aggregating to last timestep,
        # the result should be 2D with no time dimension
        assert "time" not in reader.input.dims
        assert float(reader.input.mean()) == pytest.approx(42.0)

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


class TestAggregatingReaderCache:
    """Integration tests: AggregatingReader.tile() uses the aggregation cache."""

    def setup_method(self):
        """Clear the aggregation cache before each test."""
        with _AGG_CACHE_LOCK:
            _AGG_CACHE.clear()

    def test_tile_populates_cache_and_second_call_hits_it(self, tmp_path, monkeypatch):
        """First tile call populates the cache; second call must not recompute."""
        import eostrata.aggregate as agg_mod
        import eostrata.config as cfg

        monkeypatch.setattr(cfg.settings, "agg_cache_maxsize", 4)
        monkeypatch.setattr(cfg.settings, "agg_cache_ttl_seconds", 300)

        zarr_root = tmp_path / "zarr"
        times = np.array(
            [np.datetime64("2020-01-01"), np.datetime64("2021-01-01")], dtype="datetime64[ns]"
        )
        data = np.stack([np.full((32, 32), float(y), dtype="float32") for y in [2020, 2021]])
        ds = xr.Dataset(
            {"population": (("time", "y", "x"), data)},
            coords={
                "time": times,
                "y": np.linspace(14.0, 4.0, 32),
                "x": np.linspace(2.0, 15.0, 32),
            },
        )
        ds.to_zarr(str(zarr_root), group="worldpop/nga", mode="w", consolidated=True)

        call_count = {"n": 0}
        original_apply = agg_mod.apply_temporal_aggregation

        def counting_apply(da, **kwargs):
            call_count["n"] += 1
            return original_apply(da, **kwargs)

        monkeypatch.setattr(agg_mod, "apply_temporal_aggregation", counting_apply)

        _CTX_AGG_DATETIME.set("2020-01-01/2021-12-31")
        _CTX_AGG_METHOD.set("mean")
        _CTX_AGG_BASELINE.set(None)
        try:
            reader1 = AggregatingReader(str(zarr_root), variable="population", group="worldpop/nga")
            reader1.tile(0, 0, 0)  # cache miss — apply_temporal_aggregation called

            reader2 = AggregatingReader(str(zarr_root), variable="population", group="worldpop/nga")
            reader2.tile(0, 0, 0)  # cache hit — apply_temporal_aggregation NOT called again
        finally:
            _CTX_AGG_DATETIME.set(None)
            _CTX_AGG_METHOD.set(None)
            _CTX_AGG_BASELINE.set(None)

        assert call_count["n"] == 1, (
            f"apply_temporal_aggregation called {call_count['n']} times; expected 1 (cache hit on second tile)"
        )

    def test_tile_falls_back_to_clip_first_when_cache_disabled(self, tmp_path, monkeypatch):
        """When agg_cache_maxsize=0, clip-first path is used (apply called per tile)."""
        import eostrata.aggregate as agg_mod
        import eostrata.config as cfg

        monkeypatch.setattr(cfg.settings, "agg_cache_maxsize", 0)

        zarr_root = tmp_path / "zarr"
        times = np.array(
            [np.datetime64("2020-01-01"), np.datetime64("2021-01-01")], dtype="datetime64[ns]"
        )
        data = np.stack([np.full((32, 32), float(y), dtype="float32") for y in [2020, 2021]])
        ds = xr.Dataset(
            {"population": (("time", "y", "x"), data)},
            coords={
                "time": times,
                "y": np.linspace(14.0, 4.0, 32),
                "x": np.linspace(2.0, 15.0, 32),
            },
        )
        ds.to_zarr(str(zarr_root), group="worldpop/nga", mode="w", consolidated=True)

        call_count = {"n": 0}
        original_apply = agg_mod.apply_temporal_aggregation

        def counting_apply(da, **kwargs):
            call_count["n"] += 1
            return original_apply(da, **kwargs)

        monkeypatch.setattr(agg_mod, "apply_temporal_aggregation", counting_apply)

        _CTX_AGG_DATETIME.set("2020-01-01/2021-12-31")
        _CTX_AGG_METHOD.set("mean")
        _CTX_AGG_BASELINE.set(None)
        try:
            for _ in range(2):
                reader = AggregatingReader(
                    str(zarr_root), variable="population", group="worldpop/nga"
                )
                reader.tile(0, 0, 0)
        finally:
            _CTX_AGG_DATETIME.set(None)
            _CTX_AGG_METHOD.set(None)
            _CTX_AGG_BASELINE.set(None)

        # Cache disabled → apply called once per tile
        assert call_count["n"] == 2
