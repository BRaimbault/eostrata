"""Tests for the Sentinel-5P TROPOMI air quality source."""

from __future__ import annotations

import io
import zipfile
from datetime import UTC, date, datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from eostrata.constants import PROP_RESOLUTION, PROP_VARIABLE
from eostrata.sources.tropomi import (
    _VARIABLE_MAP,
    TROPOMISource,
    _build_bbox_wkt,
    _download_product,
    _get_cdse_token,
    _grid_swath_data,
    _read_swath,
    _search_products,
)


class TestGetCdseToken:
    def test_returns_access_token(self, mocker):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"access_token": "tok123"}
        mock_post = mocker.patch("eostrata.sources.tropomi.httpx.post", return_value=mock_resp)
        token = _get_cdse_token("user@example.com", "secret")
        assert token == "tok123"
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]["data"]
        assert call_kwargs["username"] == "user@example.com"
        assert call_kwargs["client_id"] == "cdse-public"

    def test_raises_on_http_error(self, mocker):
        import httpx

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=MagicMock()
        )
        mocker.patch("eostrata.sources.tropomi.httpx.post", return_value=mock_resp)
        with pytest.raises(httpx.HTTPStatusError):
            _get_cdse_token("bad", "creds")


class TestSearchProducts:
    def _make_search_response(self, products, next_link=None):
        mock_resp = MagicMock()
        body = {"value": products}
        if next_link:
            body["@odata.nextLink"] = next_link
        mock_resp.json.return_value = body
        return mock_resp

    def test_returns_products_list(self, mocker):
        products = [{"Id": "abc", "Name": "S5P_OFFL_L2__NO2"}]
        mock_resp = self._make_search_response(products)
        mocker.patch("eostrata.sources.tropomi.httpx.get", return_value=mock_resp)
        result = _search_products("L2__NO2___", date(2023, 6, 1), (0, 0, 10, 10))
        assert result == products

    def test_follows_pagination(self, mocker):
        page1 = [{"Id": "p1", "Name": "orbit1"}]
        page2 = [{"Id": "p2", "Name": "orbit2"}]
        resp1 = self._make_search_response(page1, next_link="https://next.page")
        resp2 = self._make_search_response(page2)
        mocker.patch("eostrata.sources.tropomi.httpx.get", side_effect=[resp1, resp2])
        result = _search_products("L2__NO2___", date(2023, 6, 1), (0, 0, 10, 10))
        assert len(result) == 2
        assert result[0]["Id"] == "p1"
        assert result[1]["Id"] == "p2"

    def test_returns_empty_list_when_no_products(self, mocker):
        mock_resp = self._make_search_response([])
        mocker.patch("eostrata.sources.tropomi.httpx.get", return_value=mock_resp)
        result = _search_products("L2__CO____", date(2023, 1, 1), (0, 0, 5, 5))
        assert result == []


class TestDownloadProduct:
    def _make_stream_context(self, content: bytes):
        """Build a mock httpx stream context that yields *content* as a single chunk."""
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_bytes.return_value = iter([content])
        return mock_resp

    def test_skips_if_dest_exists(self, tmp_path):
        dest = tmp_path / "product.nc"
        dest.write_bytes(b"already here")
        result = _download_product("uuid-123", dest, "token")
        assert result == dest

    def test_downloads_raw_hdf5_file(self, tmp_path, mocker):
        # HDF5 magic bytes: \x89HDF\r\n\x1a\n
        hdf5_content = b"\x89HDF\r\n\x1a\n" + b"\x00" * 100
        mock_stream = self._make_stream_context(hdf5_content)
        dest = tmp_path / "product.nc"
        mocker.patch("eostrata.sources.tropomi.httpx.stream", return_value=mock_stream)
        result = _download_product("uuid-hdf5", dest, "mytoken")
        assert result == dest
        assert dest.read_bytes() == hdf5_content

    def test_downloads_and_extracts_zip(self, tmp_path, mocker):
        # Create a real in-memory ZIP containing a .nc file
        nc_content = b"\x89HDF fake nc content"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("orbit_data.nc", nc_content)
        zip_bytes = buf.getvalue()

        mock_stream = self._make_stream_context(zip_bytes)
        dest = tmp_path / "product.nc"
        mocker.patch("eostrata.sources.tropomi.httpx.stream", return_value=mock_stream)
        result = _download_product("uuid-zip", dest, "mytoken")
        assert result == dest
        assert dest.read_bytes() == nc_content

    def test_zip_with_no_nc_raises_runtime_error(self, tmp_path, mocker):
        """A ZIP that contains no .nc members must raise RuntimeError."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("readme.txt", "no nc here")
        zip_bytes = buf.getvalue()

        mock_stream = self._make_stream_context(zip_bytes)
        dest = tmp_path / "product.nc"
        mocker.patch("eostrata.sources.tropomi.httpx.stream", return_value=mock_stream)
        with pytest.raises(RuntimeError, match="No .nc file found"):
            _download_product("uuid-bad-zip", dest, "tok")

    def test_cleans_up_tmp_on_exception(self, tmp_path, mocker):
        mock_stream = mocker.MagicMock()
        mock_stream.__enter__ = lambda s: mock_stream
        mock_stream.__exit__ = mocker.MagicMock(return_value=False)
        mock_stream.raise_for_status.side_effect = RuntimeError("network error")
        dest = tmp_path / "product.nc"
        mocker.patch("eostrata.sources.tropomi.httpx.stream", return_value=mock_stream)
        with pytest.raises(RuntimeError):
            _download_product("uuid-fail", dest, "tok")
        tmp = dest.with_suffix(".tmp")
        assert not tmp.exists()


class TestReadSwath:
    def _make_hdf5_swath(
        self, tmp_path, var_name="nitrogendioxide_tropospheric_column", add_fill=True
    ):
        import h5py

        nc_path = tmp_path / "swath.nc"
        nlines, npixels = 3, 4
        with h5py.File(nc_path, "w") as f:
            grp = f.require_group("PRODUCT")
            lat = np.linspace(1.0, 3.0, nlines)
            lon = np.linspace(1.0, 4.0, npixels)
            lat_arr = np.tile(lat[:, None], (1, npixels))[None, ...]
            lon_arr = np.tile(lon[None, :], (nlines, 1))[None, ...]
            grp.create_dataset("latitude", data=lat_arr.astype("float32"))
            grp.create_dataset("longitude", data=lon_arr.astype("float32"))
            grp.create_dataset("qa_value", data=np.ones((1, nlines, npixels), dtype="float32"))
            data = np.full((1, nlines, npixels), 1e-5, dtype="float64")
            ds = grp.create_dataset(var_name, data=data)
            if add_fill:
                ds.attrs["_FillValue"] = np.float64(9.96921e36)
        return nc_path

    def test_returns_flat_arrays(self, tmp_path):
        nc_path = self._make_hdf5_swath(tmp_path)
        lat, lon, val = _read_swath(nc_path, "PRODUCT/nitrogendioxide_tropospheric_column")
        assert lat.ndim == 1
        assert lon.ndim == 1
        assert val.ndim == 1
        assert len(lat) == len(lon) == len(val)

    def test_raises_keyerror_for_missing_variable(self, tmp_path):
        nc_path = self._make_hdf5_swath(tmp_path)
        with pytest.raises(KeyError, match="not found"):
            _read_swath(nc_path, "PRODUCT/nonexistent_variable")

    def test_fill_values_become_nan_and_filtered(self, tmp_path):
        import h5py

        nc_path = tmp_path / "swath_fill.nc"
        nlines, npixels = 2, 2
        fill = 9.96921e36
        with h5py.File(nc_path, "w") as f:
            grp = f.require_group("PRODUCT")
            lat = np.array([[1.0, 2.0], [3.0, 4.0]])[None, ...].astype("float32")
            lon = np.array([[1.0, 2.0], [3.0, 4.0]])[None, ...].astype("float32")
            # Mix of real values and fill values
            data = np.array([[[1e-5, fill], [2e-5, fill]]])[None, ...].astype("float64")
            # Reshape to (1, nlines, npixels)
            data = data.reshape(1, nlines, npixels)
            grp.create_dataset("latitude", data=lat)
            grp.create_dataset("longitude", data=lon)
            grp.create_dataset("qa_value", data=np.ones((1, nlines, npixels), dtype="float32"))
            ds = grp.create_dataset("testvar", data=data)
            ds.attrs["_FillValue"] = np.float64(fill)
        # Only non-fill pixels should be returned
        lat_r, lon_r, val_r = _read_swath(nc_path, "PRODUCT/testvar")
        assert np.all(np.isfinite(val_r))


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
        coords_str = wkt[len("POLYGON((") : -2]  # strip 'POLYGON((' and '))'
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
        assert self.source.stac_item_id() == "no2"

    def test_stac_item_id_custom_variable(self):
        assert self.source.stac_item_id(variable="ch4") == "ch4"

    def test_stac_properties(self):
        props = self.source.stac_properties(variable="no2", year=2023, month=6, day=15)
        assert props[PROP_VARIABLE] == "no2"
        assert props["eostrata:tropomi_product"] == "L2__NO2___"
        assert "0.1deg" in props[PROP_RESOLUTION]
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
        assert meta["item_id"] == "no2"
        assert meta["variable"] == "no2"

    def test_ui_fields_include_days(self):
        assert "days" in self.source.ui_fields
        assert "variable" in self.source.ui_fields


class TestTROPOMIIterPeriods:
    def test_yields_one_entry_per_day(self):
        periods = list(
            TROPOMISource.iter_periods(variable="no2", years=[2023], months=[1], days=[1, 2, 3])
        )
        assert len(periods) == 3

    def test_multiple_years_months_days(self):
        periods = list(
            TROPOMISource.iter_periods(
                variable="co", years=[2022, 2023], months=[6, 7], days=[1, 15]
            )
        )
        assert len(periods) == 2 * 2 * 2  # 8 total

    def test_label_format(self):
        periods = list(
            TROPOMISource.iter_periods(variable="no2", years=[2023], months=[3], days=[5])
        )
        label, kwargs = periods[0]
        assert label == "no2/2023-03-05"

    def test_kwargs_structure(self):
        periods = list(
            TROPOMISource.iter_periods(variable="so2", years=[2022], months=[12], days=[31])
        )
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
        assert item["item_id"] == "co"
        assert item["datetime_"] == datetime(2022, 8, 10, tzinfo=UTC)
        assert item["variable"] == "co"
        assert PROP_VARIABLE in item["extra_properties"]


class TestGridSwathDataZeroSizeGrid:
    def test_zero_size_bbox_returns_single_nan_cell(self):
        """When bbox produces 0-length lat/lon axes, return a (1,1) NaN grid."""
        # Use a bbox where arange produces zero points at given resolution
        lat = np.array([5.0], dtype="float32")
        lon = np.array([5.0], dtype="float32")
        val = np.array([1.0], dtype="float64")
        # north == south after rounding makes ny=0
        bbox = (5.0, 5.0, 5.0, 5.0)  # zero-width: arange will produce empty arrays
        grid, lats, lons = _grid_swath_data(lat, lon, val, bbox, resolution=1.0)
        assert grid.shape == (1, 1)
        assert np.isnan(grid).all()


class TestTROPOMIDownloadRequiresCredentials:
    def test_raises_without_credentials(self, tmp_path, mocker):
        source = TROPOMISource()
        mock_settings = mocker.patch("eostrata.config.settings")
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

    def test_download_returns_empty_when_no_products_found(self, tmp_path, mocker):
        source = TROPOMISource()
        mock_settings = mocker.patch("eostrata.config.settings")
        mock_settings.cdse_user = "user@example.com"
        mock_settings.cdse_password = "secret"
        mocker.patch("eostrata.sources.tropomi._search_products", return_value=[])
        result = source.download(
            tmp_path, (0, 0, 10, 10), variable="no2", year=2023, month=6, day=1
        )
        assert result == []

    def test_download_returns_paths_on_success(self, tmp_path, mocker):
        source = TROPOMISource()
        fake_nc = tmp_path / "tropomi" / "2023" / "06" / "01" / "orbit1.nc"
        products = [{"Id": "uuid-1", "Name": "orbit1"}]
        mock_settings = mocker.patch("eostrata.config.settings")
        mock_settings.cdse_user = "user@example.com"
        mock_settings.cdse_password = "secret"
        mocker.patch("eostrata.sources.tropomi._search_products", return_value=products)
        mocker.patch("eostrata.sources.tropomi._get_cdse_token", return_value="tok")
        mocker.patch("eostrata.sources.tropomi._download_product", return_value=fake_nc)
        result = source.download(
            tmp_path, (0, 0, 10, 10), variable="no2", year=2023, month=6, day=1
        )

        assert result == [fake_nc]

    def test_download_skips_failed_products(self, tmp_path, mocker):
        """If one product download raises, it is logged and skipped — others succeed."""
        source = TROPOMISource()
        fake_nc = tmp_path / "orbit2.nc"
        products = [{"Id": "uuid-fail", "Name": "fail"}, {"Id": "uuid-ok", "Name": "ok"}]

        def side_effect(pid, dest, token):
            if pid == "uuid-fail":
                raise RuntimeError("network error")
            return fake_nc

        mock_settings = mocker.patch("eostrata.config.settings")
        mock_settings.cdse_user = "user@example.com"
        mock_settings.cdse_password = "secret"
        mocker.patch("eostrata.sources.tropomi._search_products", return_value=products)
        mocker.patch("eostrata.sources.tropomi._get_cdse_token", return_value="tok")
        mocker.patch("eostrata.sources.tropomi._download_product", side_effect=side_effect)
        result = source.download(
            tmp_path, (0, 0, 10, 10), variable="no2", year=2023, month=6, day=1
        )
        assert result == [fake_nc]


class TestTROPOMIExtractItemBbox:
    def test_returns_x_y_extent(self):
        import xarray as xr

        source = TROPOMISource()
        lons = np.array([1.0, 5.0, 9.0], dtype="float64")
        lats = np.array([2.0, 5.0, 8.0], dtype="float64")
        data = np.zeros((3, 3), dtype="float32")
        da = xr.DataArray(data, dims=("y", "x"), coords={"y": lats, "x": lons})
        ds = da.to_dataset(name="no2")
        bbox = source.extract_item_bbox(ds)
        assert bbox == (1.0, 2.0, 9.0, 8.0)


class TestTROPOMIToZarrMissingBranches:
    def _make_fake_swath(self, tmp_path, var_name="nitrogendioxide_tropospheric_column"):
        import h5py

        nc_path = tmp_path / "swath.nc"
        nlines, npixels = 3, 5
        with h5py.File(nc_path, "w") as f:
            grp = f.require_group("PRODUCT")
            lat = np.linspace(1.0, 3.0, nlines)
            lon = np.linspace(1.0, 5.0, npixels)
            lat_arr = np.tile(lat[:, None], (1, npixels))[None, ...]
            lon_arr = np.tile(lon[None, :], (nlines, 1))[None, ...]
            grp.create_dataset("latitude", data=lat_arr.astype("float32"))
            grp.create_dataset("longitude", data=lon_arr.astype("float32"))
            grp.create_dataset("qa_value", data=np.ones((1, nlines, npixels), dtype="float32"))
            data = np.full((1, nlines, npixels), 2e-5, dtype="float64")
            ds = grp.create_dataset(var_name, data=data)
            ds.attrs["_FillValue"] = np.float64(9.96921e36)
        return nc_path

    def test_to_zarr_raises_for_unknown_variable(self, tmp_path):
        source = TROPOMISource()
        fake_nc = tmp_path / "orbit.nc"
        fake_nc.write_bytes(b"")
        with pytest.raises(ValueError, match="Unknown TROPOMI variable"):
            source.to_zarr(
                fake_nc,
                tmp_path / "zarr",
                (0, 0, 10, 5),
                variable="bad_var",
                year=2023,
                month=6,
                day=1,
            )

    def test_to_zarr_skips_unreadable_swath_files(self, tmp_path):
        """If a sibling swath file is corrupt/unreadable, it is warned and skipped."""
        _, var_path = _VARIABLE_MAP["no2"]
        good_swath = self._make_fake_swath(tmp_path)
        # Create a corrupt sibling
        (tmp_path / "corrupt.nc").write_bytes(b"not an hdf5 file")

        zarr_root = tmp_path / "zarr"
        source = TROPOMISource()
        # Should not raise — corrupt file is skipped
        ds = source.to_zarr(
            good_swath, zarr_root, (0, 0, 6, 4), variable="no2", year=2023, month=6, day=1
        )
        assert "no2" in ds

    def test_to_zarr_handles_no_valid_pixels(self, tmp_path):
        """When the directory has no readable swath files at all, writes an empty grid."""
        # Put a bad file in the directory so glob finds it but _read_swath fails
        bad_nc = tmp_path / "bad.nc"
        bad_nc.write_bytes(b"garbage")

        zarr_root = tmp_path / "zarr"
        source = TROPOMISource()
        ds = source.to_zarr(
            bad_nc, zarr_root, (0, 0, 10, 5), variable="no2", year=2023, month=6, day=2
        )
        assert "no2" in ds

    def test_write_daily_grid_falls_back_when_existing_read_fails(self, tmp_path, mocker):
        """If open_zarr raises while checking for duplicate timestamps, append proceeds."""
        _, var_path = _VARIABLE_MAP["no2"]
        (tmp_path / "swaths").mkdir()
        swath = self._make_fake_swath(tmp_path / "swaths")
        zarr_root = tmp_path / "zarr"
        bbox = (0.0, 0.0, 6.0, 4.0)
        source = TROPOMISource()

        # First write to create the group
        source.to_zarr(swath, zarr_root, bbox, variable="no2", year=2023, month=7, day=1)

        mocker.patch("xarray.open_zarr", side_effect=OSError("broken"))
        # Should not raise — fallback appends
        source.to_zarr(swath, zarr_root, bbox, variable="no2", year=2023, month=7, day=2)

    def test_to_zarr_append_skips_duplicate_day(self, tmp_path):
        """Writing the same day twice must not create duplicate time entries."""
        _, var_path = _VARIABLE_MAP["no2"]
        (tmp_path / "swaths").mkdir()
        swath = self._make_fake_swath(tmp_path / "swaths")
        zarr_root = tmp_path / "zarr"
        bbox = (0.0, 0.0, 6.0, 4.0)
        source = TROPOMISource()

        # First write
        source.to_zarr(swath, zarr_root, bbox, variable="no2", year=2023, month=6, day=5)
        # Second write — same day, should be skipped by duplicate guard
        source.to_zarr(swath, zarr_root, bbox, variable="no2", year=2023, month=6, day=5)

        import xarray as xr

        stored = xr.open_zarr(str(zarr_root), group="tropomi/no2", consolidated=False)
        assert len(stored["time"]) == 1


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
            grp_parent = f.require_group(
                parent.replace("PRODUCT/", "") if "/" in parent else "PRODUCT"
            )
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
