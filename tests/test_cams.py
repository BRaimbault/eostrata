"""Tests for the CAMS EAC4 air quality source."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from eostrata.sources.cams import (
    CAMSSource,
    _MULTI_LEVEL_VARS,
    _VARIABLE_MAP,
    _cams_netcdf_to_zarr,
)


class TestCAMSVariableMap:
    def test_all_variables_mapped(self):
        expected = {"pm2p5", "pm10", "no2", "co", "o3", "so2", "aod550"}
        assert set(_VARIABLE_MAP) == expected

    def test_multi_level_vars_are_subset(self):
        assert _MULTI_LEVEL_VARS < set(_VARIABLE_MAP)

    def test_aod550_is_single_level(self):
        assert "aod550" not in _MULTI_LEVEL_VARS


class TestCAMSSource:
    def setup_method(self):
        self.source = CAMSSource()

    def test_metadata(self):
        assert self.source.id == "cams"
        assert self.source.collection_id == "cams"
        assert self.source.temporal_resolution == "monthly"
        assert self.source.VARIABLE == "pm2p5"

    def test_zarr_group_default(self):
        assert self.source.zarr_group() == "cams/pm2p5"

    def test_zarr_group_custom_variable(self):
        assert self.source.zarr_group(variable="no2") == "cams/no2"
        assert self.source.zarr_group(variable="aod550") == "cams/aod550"

    def test_stac_item_id_default(self):
        assert self.source.stac_item_id() == "cams_pm2p5"

    def test_stac_item_id_custom_variable(self):
        assert self.source.stac_item_id(variable="o3") == "cams_o3"

    def test_stac_properties_multi_level(self):
        props = self.source.stac_properties(variable="no2", year=2022)
        assert props["eostrata:variable"] == "no2"
        assert props["eostrata:resolution"] == "0.75deg"
        assert props["eostrata:dataset"] == "cams-global-reanalysis-eac4-monthly"
        assert "1000hPa" in props["eostrata:pressure_level"]

    def test_stac_properties_single_level(self):
        props = self.source.stac_properties(variable="aod550", year=2022)
        assert props["eostrata:pressure_level"] == "single"

    def test_latest_available_is_in_past(self):
        latest = self.source.latest_available()
        assert latest < datetime.now(tz=UTC)

    def test_latest_available_has_timezone(self):
        latest = self.source.latest_available()
        assert isinstance(latest, datetime)
        assert latest.tzinfo is not None

    def test_latest_available_lag_about_4_months(self):
        latest = self.source.latest_available()
        now = datetime.now(tz=UTC)
        delta_days = (now - latest).days
        # Should be roughly 90–150 days in the past
        assert 60 <= delta_days <= 180

    def test_latest_available_wraps_year(self):
        """Month calculation should not produce month <= 0."""
        with patch("eostrata.sources.cams.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 3, 15, tzinfo=UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = CAMSSource().latest_available()
        # March - 4 months = November of previous year
        assert result.year == 2023
        assert result.month == 11

    def test_catalog_meta(self):
        meta = CAMSSource.catalog_meta("pm2p5")
        assert meta["item_id"] == "cams_pm2p5"
        assert meta["variable"] == "pm2p5"

    def test_ui_fields(self):
        assert "variable" in self.source.ui_fields
        assert "years" in self.source.ui_fields
        assert "months" in self.source.ui_fields


class TestCAMSIterPeriods:
    def test_yields_one_entry_per_year(self):
        periods = list(CAMSSource.iter_periods(
            variable="no2", years=[2021, 2022], months=[1, 2, 3]
        ))
        assert len(periods) == 2

    def test_label_format(self):
        periods = list(CAMSSource.iter_periods(
            variable="pm2p5", years=[2020], months=[6, 7]
        ))
        label, kwargs = periods[0]
        assert label == "pm2p5/2020"
        assert kwargs["variable"] == "pm2p5"
        assert kwargs["year"] == 2020
        assert kwargs["months"] == [6, 7]

    def test_kwargs_contain_months_list(self):
        months = [1, 6, 12]
        periods = list(CAMSSource.iter_periods(variable="co", years=[2021], months=months))
        _, kwargs = periods[0]
        assert kwargs["months"] == months


class TestCAMSStacRegistrations:
    def setup_method(self):
        self.source = CAMSSource()

    def test_returns_one_item_per_month(self):
        ds = MagicMock()
        period_kwargs = {"variable": "no2", "year": 2022, "months": [1, 2, 3]}
        items = self.source.stac_registrations(ds, period_kwargs)
        assert len(items) == 3

    def test_item_structure(self):
        ds = MagicMock()
        period_kwargs = {"variable": "pm2p5", "year": 2021, "months": [6]}
        items = self.source.stac_registrations(ds, period_kwargs)
        item = items[0]
        assert item["item_id"] == "cams_pm2p5"
        assert item["datetime_"] == datetime(2021, 6, 1, tzinfo=UTC)
        assert item["variable"] == "pm2p5"
        assert "eostrata:variable" in item["extra_properties"]

    def test_datetimes_are_month_starts(self):
        ds = MagicMock()
        period_kwargs = {"variable": "o3", "year": 2020, "months": [3, 4, 5]}
        items = self.source.stac_registrations(ds, period_kwargs)
        for i, item in enumerate(items, start=3):
            assert item["datetime_"].month == i
            assert item["datetime_"].day == 1


class TestCAMSNetcdfToZarr:
    """Integration-style test for the NetCDF→Zarr conversion function."""

    def _make_cams_netcdf(self, tmp_path, variable="pm2p5", has_pressure_level=True):
        """Create a minimal CAMS-like NetCDF file for testing."""
        import xarray as xr

        lats = np.array([0.0, 0.75, 1.5], dtype="float64")
        lons = np.array([0.0, 0.75, 1.5], dtype="float64")
        times = np.array(["2022-01-01", "2022-02-01"], dtype="datetime64[ns]")

        if has_pressure_level:
            data = np.random.rand(2, 1, 3, 3).astype("float32")
            da = xr.DataArray(
                data,
                dims=("valid_time", "pressure_level", "latitude", "longitude"),
                coords={
                    "valid_time": times,
                    "pressure_level": np.array([1000.0]),
                    "latitude": lats,
                    "longitude": lons,
                },
                name=variable,
            )
        else:
            data = np.random.rand(2, 3, 3).astype("float32")
            da = xr.DataArray(
                data,
                dims=("valid_time", "latitude", "longitude"),
                coords={
                    "valid_time": times,
                    "latitude": lats,
                    "longitude": lons,
                },
                name=variable,
            )

        ds = da.to_dataset()
        nc_path = tmp_path / f"cams_{variable}.nc"
        ds.to_netcdf(nc_path)
        return nc_path

    def test_multi_level_variable_to_zarr(self, tmp_path):
        nc_path = self._make_cams_netcdf(tmp_path, "pm2p5", has_pressure_level=True)
        zarr_root = tmp_path / "zarr"
        bbox = (0.0, 0.0, 2.0, 2.0)

        ds = _cams_netcdf_to_zarr(nc_path, zarr_root, "cams/pm2p5", variable="pm2p5", bbox=bbox)

        assert "pm2p5" in ds
        assert "time" in ds.dims
        assert (zarr_root / "cams" / "pm2p5").exists()

    def test_single_level_variable_to_zarr(self, tmp_path):
        nc_path = self._make_cams_netcdf(tmp_path, "aod550", has_pressure_level=False)
        zarr_root = tmp_path / "zarr"
        bbox = (0.0, 0.0, 2.0, 2.0)

        ds = _cams_netcdf_to_zarr(nc_path, zarr_root, "cams/aod550", variable="aod550", bbox=bbox)

        assert "aod550" in ds
        assert "time" in ds.dims

    def test_to_zarr_via_source(self, tmp_path):
        nc_path = self._make_cams_netcdf(tmp_path, "no2", has_pressure_level=True)
        zarr_root = tmp_path / "zarr"
        bbox = (0.0, 0.0, 2.0, 2.0)

        source = CAMSSource()
        ds = source.to_zarr(nc_path, zarr_root, bbox, variable="no2", year=2022)

        assert "no2" in ds
        assert (zarr_root / "cams" / "no2").exists()
