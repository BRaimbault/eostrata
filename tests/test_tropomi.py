"""Tests for the Sentinel-5P TROPOMI air quality source."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from eostrata.sources.tropomi import (
    TROPOMISource,
    _VARIABLE_MAP,
    _build_bbox_wkt,
    _grid_swath_data,
)


class TestTROPOMIVariableMap:
    def test_all_expected_variables_present(self):
        expected = {"no2", "co", "o3", "so2", "ch4", "hcho", "aer_ai"}
        assert set(_VARIABLE_MAP) == expected

    def test_each_entry_has_product_type_and_path(self):
        for var, (product_type, var_path) in _VARIABLE_MAP.items():
            assert product_type.startswith("L2__"), f"{var}: bad product type {product_type!r}"
            assert var_path.startswith("PRODUCT/"), f"{var}: bad var path {var_path!r}"

    def test_no2_product_type(self):
        pt, _ = _VARIABLE_MAP["no2"]
        assert pt == "L2__NO2___"

    def test_co_product_type(self):
        pt, _ = _VARIABLE_MAP["co"]
        assert pt == "L2__CO____"


class TestBuildBboxWkt:
    def test_returns_polygon_wkt(self):
        wkt = _build_bbox_wkt((10.0, 20.0, 30.0, 40.0))
        assert wkt.startswith("POLYGON(")
        assert "10.0" in wkt
        assert "40.0" in wkt

    def test_closes_polygon(self):
        """First and last coordinate pair must be the same."""
        wkt = _build_bbox_wkt((0.0, 0.0, 1.0, 1.0))
        # The polygon string ends with the closing coord before ")"
        coords_str = wkt[len("POLYGON(("):-2]  # strip 'POLYGON((' and '))'
        pairs = [p.strip() for p in coords_str.split(",")]
        assert pairs[0] == pairs[-1]


class TestGridSwathData:
    def _make_uniform_swath(self, n=100, bbox=(0.0, 0.0, 10.0, 10.0)):
        """Return lat/lon/val arrays uniformly covering *bbox*."""
        west, south, east, north = bbox
        rng = np.random.default_rng(42)
        lat = rng.uniform(south, north, n).astype("float32")
        lon = rng.uniform(west, east, n).astype("float32")
        val = rng.random(n).astype("float64")
        return lat, lon, val

    def test_output_shape(self):
        lat, lon, val = self._make_uniform_swath(500)
        bbox = (0.0, 0.0, 10.0, 10.0)
        grid, lats, lons = _grid_swath_data(lat, lon, val, bbox, resolution=1.0)
        assert grid.shape == (len(lats), len(lons))

    def test_grid_contains_data(self):
        lat, lon, val = self._make_uniform_swath(1000)
        bbox = (0.0, 0.0, 5.0, 5.0)
        grid, lats, lons = _grid_swath_data(lat, lon, val, bbox, resolution=1.0)
        assert np.any(np.isfinite(grid))

    def test_out_of_bbox_pixels_ignored(self):
        """Pixels entirely outside the bbox should not appear in the grid."""
        lat = np.array([50.0, 51.0], dtype="float32")
        lon = np.array([50.0, 51.0], dtype="float32")
        val = np.array([1.0, 2.0], dtype="float64")
        bbox = (0.0, 0.0, 5.0, 5.0)
        grid, _, _ = _grid_swath_data(lat, lon, val, bbox, resolution=1.0)
        assert np.all(np.isnan(grid))

    def test_empty_input_produces_nan_grid(self):
        lat = np.array([], dtype="float32")
        lon = np.array([], dtype="float32")
        val = np.array([], dtype="float64")
        bbox = (0.0, 0.0, 5.0, 5.0)
        grid, lats, lons = _grid_swath_data(lat, lon, val, bbox, resolution=1.0)
        assert np.all(np.isnan(grid))

    def test_values_are_averaged_not_summed(self):
        """Two pixels in the same cell should be averaged."""
        # Place two pixels in the same 1° cell centred on (0.5, 0.5)
        lat = np.array([0.3, 0.7], dtype="float32")
        lon = np.array([0.3, 0.7], dtype="float32")
        val = np.array([0.0, 1.0], dtype="float64")
        bbox = (0.0, 0.0, 1.0, 1.0)
        grid, _, _ = _grid_swath_data(lat, lon, val, bbox, resolution=1.0)
        finite = grid[np.isfinite(grid)]
        assert len(finite) == 1
        assert finite[0] == pytest.approx(0.5, abs=1e-6)


class TestTROPOMISource:
    def setup_method(self):
        self.source = TROPOMISource()

    def test_metadata(self):
        assert self.source.id == "tropomi"
        assert self.source.collection_id == "tropomi"
        assert self.source.temporal_resolution == "daily"
        assert self.source.VARIABLE == "no2"
        assert self.source.skip_404 is True

    def test_zarr_group_default(self):
        assert self.source.zarr_group() == "tropomi/no2"

    def test_zarr_group_custom_variable(self):
        assert self.source.zarr_group(variable="co") == "tropomi/co"
        assert self.source.zarr_group(variable="aer_ai") == "tropomi/aer_ai"

    def test_stac_item_id_default(self):
        assert self.source.stac_item_id() == "tropomi_no2"

    def test_stac_item_id_custom_variable(self):
        assert self.source.stac_item_id(variable="ch4") == "tropomi_ch4"

    def test_stac_properties(self):
        props = self.source.stac_properties(variable="no2", year=2023, month=6, day=15)
        assert props["eostrata:variable"] == "no2"
        assert props["eostrata:tropomi_product"] == "L2__NO2___"
        assert "0.1deg" in props["eostrata:resolution"]
        assert props["eostrata:qa_threshold"] == 0.75

    def test_latest_available_is_in_past(self):
        latest = self.source.latest_available()
        assert latest < datetime.now(tz=UTC)

    def test_latest_available_lag_is_3_days(self):
        latest = self.source.latest_available()
        now = datetime.now(tz=UTC)
        delta = now - latest
        assert 2 <= delta.days <= 4  # ~3 days lag

    def test_latest_available_has_timezone(self):
        latest = self.source.latest_available()
        assert latest.tzinfo is not None

    def test_catalog_meta(self):
        meta = TROPOMISource.catalog_meta("no2")
        assert meta["item_id"] == "tropomi_no2"
        assert meta["variable"] == "no2"

    def test_ui_fields_include_days(self):
        assert "days" in self.source.ui_fields
        assert "variable" in self.source.ui_fields


class TestTROPOMIIterPeriods:
    def test_yields_one_entry_per_day(self):
        periods = list(TROPOMISource.iter_periods(
            variable="no2", years=[2023], months=[1], days=[1, 2, 3]
        ))
        assert len(periods) == 3

    def test_multiple_years_months_days(self):
        periods = list(TROPOMISource.iter_periods(
            variable="co", years=[2022, 2023], months=[6, 7], days=[1, 15]
        ))
        assert len(periods) == 2 * 2 * 2  # 8 total

    def test_label_format(self):
        periods = list(TROPOMISource.iter_periods(
            variable="no2", years=[2023], months=[3], days=[5]
        ))
        label, kwargs = periods[0]
        assert label == "no2/2023-03-05"

    def test_kwargs_structure(self):
        periods = list(TROPOMISource.iter_periods(
            variable="so2", years=[2022], months=[12], days=[31]
        ))
        _, kwargs = periods[0]
        assert kwargs == {"variable": "so2", "year": 2022, "month": 12, "day": 31}


class TestTROPOMIStacRegistrations:
    def setup_method(self):
        self.source = TROPOMISource()

    def test_returns_one_item(self):
        ds = MagicMock()
        period_kwargs = {"variable": "no2", "year": 2023, "month": 6, "day": 15}
        items = self.source.stac_registrations(ds, period_kwargs)
        assert len(items) == 1

    def test_item_structure(self):
        ds = MagicMock()
        period_kwargs = {"variable": "co", "year": 2022, "month": 8, "day": 10}
        items = self.source.stac_registrations(ds, period_kwargs)
        item = items[0]
        assert item["item_id"] == "tropomi_co"
        assert item["datetime_"] == datetime(2022, 8, 10, tzinfo=UTC)
        assert item["variable"] == "co"
        assert "eostrata:variable" in item["extra_properties"]


class TestTROPOMIDownloadRequiresCredentials:
    def test_raises_without_credentials(self, tmp_path):
        source = TROPOMISource()
        with patch("eostrata.config.settings") as mock_settings:
            mock_settings.cdse_user = ""
            mock_settings.cdse_password = ""
            with pytest.raises(RuntimeError, match="CDSE credentials"):
                source.download(
                    tmp_path,
                    (0.0, 0.0, 10.0, 10.0),
                    variable="no2",
                    year=2023,
                    month=6,
                    day=1,
                )

    def test_raises_for_unknown_variable(self, tmp_path):
        source = TROPOMISource()
        with pytest.raises(ValueError, match="Unknown TROPOMI variable"):
            source.download(
                tmp_path,
                (0.0, 0.0, 10.0, 10.0),
                variable="unknown_var",
                year=2023,
                month=6,
                day=1,
            )


class TestTROPOMIToZarrWritesDailyGrid:
    """Integration-style tests for the swath→Zarr pipeline."""

    def _make_fake_swath(self, tmp_path, var_path="PRODUCT/nitrogendioxide_tropospheric_column"):
        """Create a minimal TROPOMI-like HDF5 file with a handful of valid pixels."""
        import h5py

        nc_path = tmp_path / "fake_swath.nc"
        nlines, npixels = 5, 10

        with h5py.File(nc_path, "w") as f:
            grp = f.require_group("PRODUCT")
            # Coordinates — shape (1, nlines, npixels)
            lat = np.linspace(0.5, 4.5, nlines)
            lon = np.linspace(0.5, 9.5, npixels)
            lat_arr = np.tile(lat[:, None], (1, npixels))[None, ...]
            lon_arr = np.tile(lon[None, :], (nlines, 1))[None, ...]

            grp.create_dataset("latitude", data=lat_arr.astype("float32"))
            grp.create_dataset("longitude", data=lon_arr.astype("float32"))
            grp.create_dataset("qa_value", data=np.ones((1, nlines, npixels), dtype="float32"))

            # Data variable under the nested path
            parts = var_path.split("/")
            parent = "/".join(parts[:-1])
            var_name = parts[-1]
            grp_parent = f.require_group(parent.replace("PRODUCT/", "") if "/" in parent else "PRODUCT")
            data = np.full((1, nlines, npixels), 1e-5, dtype="float64")
            ds = grp_parent.create_dataset(var_name, data=data)
            ds.attrs["_FillValue"] = np.float64(9.96921e36)

        return nc_path

    def test_to_zarr_produces_group(self, tmp_path):
        _, var_path = _VARIABLE_MAP["no2"]
        swath_nc = self._make_fake_swath(tmp_path, var_path)
        zarr_root = tmp_path / "zarr"
        bbox = (0.0, 0.0, 10.0, 5.0)

        source = TROPOMISource()
        ds = source.to_zarr(swath_nc, zarr_root, bbox, variable="no2", year=2023, month=6, day=1)

        assert "no2" in ds
        assert "time" in ds.dims
        assert (zarr_root / "tropomi" / "no2").exists()

    def test_to_zarr_time_coordinate(self, tmp_path):
        _, var_path = _VARIABLE_MAP["no2"]
        swath_nc = self._make_fake_swath(tmp_path, var_path)
        zarr_root = tmp_path / "zarr"
        bbox = (0.0, 0.0, 10.0, 5.0)

        source = TROPOMISource()
        ds = source.to_zarr(swath_nc, zarr_root, bbox, variable="no2", year=2023, month=6, day=15)

        expected_time = np.datetime64("2023-06-15", "ns")
        assert expected_time in ds["time"].values
