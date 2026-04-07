"""Tests for the CAMS EAC4 air quality source."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from eostrata.constants import PROP_RESOLUTION, PROP_VARIABLE
from eostrata.sources.cams import (
    _MULTI_LEVEL_VARS,
    _VARIABLE_MAP,
    CAMSSource,
    _cams_netcdf_to_zarr,
    _download_cams,
    _get_cdsapi,
)


class TestCAMSVariableMap:
    def test_all_variables_mapped(self):
        expected = {"pm2p5", "pm10", "no2", "co", "o3", "so2", "aod550"}
        assert set(_VARIABLE_MAP) == expected

    def test_multi_level_vars_are_subset(self):
        assert set(_VARIABLE_MAP) > _MULTI_LEVEL_VARS

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
        assert self.source.stac_item_id() == "pm2p5"

    def test_stac_item_id_custom_variable(self):
        assert self.source.stac_item_id(variable="o3") == "o3"

    def test_stac_properties_multi_level(self):
        props = self.source.stac_properties(variable="no2", year=2022)
        assert props[PROP_VARIABLE] == "no2"
        assert props[PROP_RESOLUTION] == "0.75deg"
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

    def test_latest_available_wraps_year(self, mocker):
        """Month calculation should not produce month <= 0."""
        mock_dt = mocker.patch("eostrata.sources.cams.datetime")
        mock_dt.now.return_value = datetime(2024, 3, 15, tzinfo=UTC)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = CAMSSource().latest_available()
        # March - 4 months = November of previous year
        assert result.year == 2023
        assert result.month == 11

    def test_catalog_meta(self):
        meta = CAMSSource.catalog_meta("pm2p5")
        assert meta["item_id"] == "pm2p5"
        assert meta["variable"] == "pm2p5"

    def test_ui_fields(self):
        assert "variable" in self.source.ui_fields
        assert "years" in self.source.ui_fields
        assert "months" in self.source.ui_fields


class TestCAMSIterPeriods:
    def test_yields_one_entry_per_year(self):
        periods = list(
            CAMSSource.iter_periods(variable="no2", years=[2021, 2022], months=[1, 2, 3])
        )
        assert len(periods) == 2

    def test_label_format(self):
        periods = list(CAMSSource.iter_periods(variable="pm2p5", years=[2020], months=[6, 7]))
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

    def test_returns_one_item_per_month(self, mocker):
        ds = mocker.MagicMock()
        period_kwargs = {"variable": "no2", "year": 2022, "months": [1, 2, 3]}
        items = self.source.stac_registrations(ds, period_kwargs)
        assert len(items) == 3

    def test_item_structure(self, mocker):
        ds = mocker.MagicMock()
        period_kwargs = {"variable": "pm2p5", "year": 2021, "months": [6]}
        items = self.source.stac_registrations(ds, period_kwargs)
        item = items[0]
        assert item["item_id"] == "pm2p5"
        assert item["datetime_"] == datetime(2021, 6, 1, tzinfo=UTC)
        assert item["variable"] == "pm2p5"
        assert PROP_VARIABLE in item["extra_properties"]

    def test_datetimes_are_month_starts(self, mocker):
        ds = mocker.MagicMock()
        period_kwargs = {"variable": "o3", "year": 2020, "months": [3, 4, 5]}
        items = self.source.stac_registrations(ds, period_kwargs)
        for i, item in enumerate(items, start=3):
            assert item["datetime_"].month == i
            assert item["datetime_"].day == 1


class TestGetCdsapi:
    def test_returns_cdsapi_when_available(self):
        result = _get_cdsapi()
        import cdsapi

        assert result is cdsapi

    def test_raises_import_error_when_missing(self, mocker):
        import sys

        mocker.patch.dict(sys.modules, {"cdsapi": None})
        mocker.patch("builtins.__import__", side_effect=ImportError("no module"))
        with pytest.raises((ImportError, AttributeError)):
            _get_cdsapi()


class TestDownloadCams:
    def test_skips_if_dest_exists(self, tmp_path):
        """_download_cams must skip the network call when the file is already on disk."""
        dest = tmp_path / "cams" / "cams_no2_2022_01.nc"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"fake nc")

        result = _download_cams(dest, variable="no2", year=2022, months=[1], bbox=(0, 0, 10, 10))

        assert result == dest

    def test_download_multi_level_var_with_key(self, tmp_path, mocker):
        """Multi-level variable request includes pressure_level; key triggers keyed client."""
        dest = tmp_path / "cams" / "cams_no2_2022_01.nc"
        mock_client = mocker.MagicMock()
        mock_cdsapi = mocker.MagicMock()
        mock_cdsapi.Client.return_value = mock_client

        def _fake_retrieve(_ds, _req, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"\x89HDF\r\n\x1a\n")  # HDF5 magic — not a ZIP

        mock_client.retrieve.side_effect = _fake_retrieve

        mocker.patch("eostrata.sources.cams._get_cdsapi", return_value=mock_cdsapi)
        mock_settings = mocker.patch("eostrata.config.settings")
        mock_settings.ads_url = "https://ads.example.com/api"
        mock_settings.ads_key = "uid:secret"
        _download_cams(dest, variable="no2", year=2022, months=[1, 2], bbox=(0, 0, 10, 10))

        mock_cdsapi.Client.assert_called_once_with(
            url="https://ads.example.com/api", key="uid:secret", quiet=True
        )
        retrieve_call = mock_client.retrieve.call_args
        assert retrieve_call[0][0] == "cams-global-reanalysis-eac4-monthly"
        request = retrieve_call[0][1]
        assert "pressure_level" in request
        assert request["pressure_level"] == ["1000"]

    def test_download_single_level_var_without_key(self, tmp_path, mocker):
        """Single-level variable (aod550) omits pressure_level; no key uses keyless client."""
        dest = tmp_path / "cams" / "cams_aod550_2022_06.nc"
        mock_client = mocker.MagicMock()
        mock_cdsapi = mocker.MagicMock()
        mock_cdsapi.Client.return_value = mock_client

        def _fake_retrieve(_ds, _req, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"\x89HDF\r\n\x1a\n")

        mock_client.retrieve.side_effect = _fake_retrieve

        mocker.patch("eostrata.sources.cams._get_cdsapi", return_value=mock_cdsapi)
        mock_settings = mocker.patch("eostrata.config.settings")
        mock_settings.ads_url = "https://ads.example.com/api"
        mock_settings.ads_key = ""
        _download_cams(dest, variable="aod550", year=2022, months=[6], bbox=(0, 0, 10, 10))

        # No key → client created without key argument
        mock_cdsapi.Client.assert_called_once_with(url="https://ads.example.com/api", quiet=True)
        request = mock_client.retrieve.call_args[0][1]
        assert "pressure_level" not in request

    def test_download_uses_default_ads_url_when_settings_url_empty(self, tmp_path, mocker):
        """Falls back to _DEFAULT_ADS_URL when ads_url is empty string."""
        from eostrata.sources.cams import _DEFAULT_ADS_URL

        dest = tmp_path / "cams" / "cams_co_2022_01.nc"
        mock_client = mocker.MagicMock()
        mock_cdsapi = mocker.MagicMock()
        mock_cdsapi.Client.return_value = mock_client

        def _fake_retrieve(_ds, _req, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"\x89HDF\r\n\x1a\n")

        mock_client.retrieve.side_effect = _fake_retrieve

        mocker.patch("eostrata.sources.cams._get_cdsapi", return_value=mock_cdsapi)
        mock_settings = mocker.patch("eostrata.config.settings")
        mock_settings.ads_url = ""  # falsy → should fall back
        mock_settings.ads_key = ""
        _download_cams(dest, variable="co", year=2022, months=[1], bbox=(0, 0, 10, 10))

        mock_cdsapi.Client.assert_called_once_with(url=_DEFAULT_ADS_URL, quiet=True)

    def test_download_extracts_nc_from_zip(self, tmp_path, mocker):
        """If ADS returns a ZIP archive, the .nc member is extracted in-place."""
        import io
        import zipfile as zf

        dest = tmp_path / "cams" / "cams_pm2p5_2023_06.nc"
        nc_content = b"\x89HDF\r\n\x1a\n" + b"\x00" * 100  # fake HDF5 bytes

        # Build an in-memory ZIP containing one .nc file
        buf = io.BytesIO()
        with zf.ZipFile(buf, "w") as z:
            z.writestr("data.nc", nc_content)
        zip_bytes = buf.getvalue()

        mock_client = mocker.MagicMock()
        mock_cdsapi = mocker.MagicMock()
        mock_cdsapi.Client.return_value = mock_client

        def _fake_retrieve(_ds, _req, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(zip_bytes)

        mock_client.retrieve.side_effect = _fake_retrieve
        mocker.patch("eostrata.sources.cams._get_cdsapi", return_value=mock_cdsapi)
        mock_settings = mocker.patch("eostrata.config.settings")
        mock_settings.ads_url = "https://ads.example.com/api"
        mock_settings.ads_key = "k"

        result = _download_cams(dest, variable="pm2p5", year=2023, months=[6], bbox=(0, 0, 1, 1))

        assert result == dest
        assert dest.read_bytes() == nc_content  # ZIP was unpacked
        assert not dest.with_suffix(".zip").exists()  # temp ZIP cleaned up


class TestCAMSSourceDownload:
    def test_download_calls_download_cams(self, tmp_path, mocker):
        """CAMSSource.download() constructs the right filename and delegates."""
        source = CAMSSource()
        fake_path = tmp_path / "cams" / "cams_pm2p5_2021_01-02-03.nc"

        mock_dl = mocker.patch("eostrata.sources.cams._download_cams", return_value=fake_path)
        result = source.download(
            tmp_path, (0, 0, 10, 10), variable="pm2p5", year=2021, months=[1, 2, 3]
        )

        assert result == [fake_path]
        mock_dl.assert_called_once()
        _, kwargs = mock_dl.call_args
        assert kwargs["variable"] == "pm2p5"
        assert kwargs["year"] == 2021
        assert kwargs["months"] == [1, 2, 3]

    def test_download_defaults_to_all_12_months(self, tmp_path, mocker):
        source = CAMSSource()
        mock_dl = mocker.patch(
            "eostrata.sources.cams._download_cams", return_value=tmp_path / "x.nc"
        )
        source.download(tmp_path, (0, 0, 10, 10), variable="no2", year=2021)

        _, kwargs = mock_dl.call_args
        assert kwargs["months"] == list(range(1, 13))

    def test_download_raises_for_unknown_variable(self, tmp_path):
        source = CAMSSource()
        with pytest.raises(ValueError, match="Unknown CAMS variable 'bad_var'"):
            source.download(tmp_path, (0, 0, 10, 10), variable="bad_var", year=2021)


class TestCAMSExtractItemBbox:
    def _make_ds(self, coords, values):
        """Build a tiny real xarray Dataset with the given coord names."""
        import xarray as xr

        data = np.zeros((len(values["y"]), len(values["x"])), dtype="float32")
        coord_keys = list(coords.keys())
        da = xr.DataArray(
            data,
            dims=(coord_keys[0], coord_keys[1]),
            coords=dict(zip(coord_keys, [values["y"], values["x"]], strict=True)),
        )
        return da.to_dataset(name="test")

    def test_uses_x_y_coords(self, tmp_path):

        source = CAMSSource()
        nc = self._make_cams_netcdf_for_bbox(tmp_path, "x", "y")
        bbox = source.extract_item_bbox(nc)
        assert bbox[0] < bbox[2]  # west < east
        assert bbox[1] < bbox[3]  # south < north

    def _make_cams_netcdf_for_bbox(self, tmp_path, x_name, y_name):
        """Return a tiny xr.Dataset with the requested coordinate names."""
        import xarray as xr

        lons = np.array([0.0, 1.0, 2.0], dtype="float64")
        lats = np.array([0.0, 1.0, 2.0], dtype="float64")
        data = np.zeros((3, 3), dtype="float32")
        da = xr.DataArray(data, dims=(y_name, x_name), coords={y_name: lats, x_name: lons})
        return da.to_dataset(name="var")

    def test_falls_back_to_longitude_latitude(self, tmp_path):
        source = CAMSSource()
        ds = self._make_cams_netcdf_for_bbox(tmp_path, "longitude", "latitude")
        bbox = source.extract_item_bbox(ds)
        assert bbox == (0.0, 0.0, 2.0, 2.0)


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

    def _make_cams_netcdf_lon_lat(self, tmp_path, variable="pm2p5"):
        """NetCDF using 'lon'/'lat' short dim names instead of 'longitude'/'latitude'."""
        lats = np.array([0.0, 0.75, 1.5], dtype="float64")
        lons = np.array([0.0, 0.75, 1.5], dtype="float64")
        times = np.array(["2022-01-01"], dtype="datetime64[ns]")
        data = np.random.rand(1, 3, 3).astype("float32")
        da = xr.DataArray(
            data,
            dims=("time", "lat", "lon"),
            coords={"time": times, "lat": lats, "lon": lons},
            name=variable,
        )
        nc_path = tmp_path / f"cams_lonlat_{variable}.nc"
        da.to_dataset().to_netcdf(nc_path)
        return nc_path

    def test_lon_lat_dim_names_are_renamed(self, tmp_path):
        """Files with 'lon'/'lat' coords should be renamed to x/y."""
        nc_path = self._make_cams_netcdf_lon_lat(tmp_path, "pm2p5")
        zarr_root = tmp_path / "zarr"
        bbox = (0.0, 0.0, 2.0, 2.0)

        ds = _cams_netcdf_to_zarr(nc_path, zarr_root, "cams/pm2p5", variable="pm2p5", bbox=bbox)

        assert "x" in ds.coords
        assert "y" in ds.coords

    def _make_cams_netcdf_with_expver(self, tmp_path, variable="o3"):
        """NetCDF that includes an 'expver' auxiliary variable (ERA5/CAMS metadata)."""
        lats = np.array([0.0, 0.75, 1.5], dtype="float64")
        lons = np.array([0.0, 0.75, 1.5], dtype="float64")
        times = np.array(["2022-03-01"], dtype="datetime64[ns]")
        data = np.random.rand(1, 3, 3).astype("float32")
        da = xr.DataArray(
            data,
            dims=("valid_time", "latitude", "longitude"),
            coords={"valid_time": times, "latitude": lats, "longitude": lons},
            name=variable,
        )
        ds = da.to_dataset()
        ds["expver"] = xr.DataArray(["0001"], dims=("valid_time",))
        nc_path = tmp_path / f"cams_expver_{variable}.nc"
        ds.to_netcdf(nc_path)
        return nc_path

    def test_expver_auxiliary_variable_is_dropped(self, tmp_path):
        """'expver' metadata variable must not appear in the Zarr output."""
        nc_path = self._make_cams_netcdf_with_expver(tmp_path, "o3")
        zarr_root = tmp_path / "zarr"
        bbox = (0.0, 0.0, 2.0, 2.0)

        ds = _cams_netcdf_to_zarr(nc_path, zarr_root, "cams/o3", variable="o3", bbox=bbox)

        assert "expver" not in ds

    def test_zarr_append_falls_back_when_existing_read_fails(self, tmp_path, mocker):
        """If open_zarr raises while checking existing timestamps, append proceeds anyway."""
        nc_path = self._make_cams_netcdf(tmp_path, "pm2p5", has_pressure_level=False)
        zarr_root = tmp_path / "zarr"
        bbox = (0.0, 0.0, 2.0, 2.0)

        # Write once to create the group
        _cams_netcdf_to_zarr(nc_path, zarr_root, "cams/pm2p5", variable="pm2p5", bbox=bbox)

        # On the second call, patch open_zarr to raise inside the duplicate check
        def _raise_once(store, **kwargs):
            raise OSError("simulated read failure")

        mocker.patch("xarray.open_zarr", side_effect=_raise_once)
        # Should not raise — falls back to appending
        _cams_netcdf_to_zarr(nc_path, zarr_root, "cams/pm2p5", variable="pm2p5", bbox=bbox)

    def test_zarr_append_skips_duplicate_timestamps(self, tmp_path):
        """Writing the same period twice must not create duplicate time entries."""
        nc_path = self._make_cams_netcdf(tmp_path, "so2", has_pressure_level=False)
        zarr_root = tmp_path / "zarr"
        bbox = (0.0, 0.0, 2.0, 2.0)

        # First write
        _cams_netcdf_to_zarr(nc_path, zarr_root, "cams/so2", variable="so2", bbox=bbox)
        # Second write — same timestamps, should be skipped
        _cams_netcdf_to_zarr(nc_path, zarr_root, "cams/so2", variable="so2", bbox=bbox)

        stored = xr.open_zarr(str(zarr_root), group="cams/so2", consolidated=False)
        # Timestamps should not be duplicated
        assert len(stored["time"]) == len(
            np.array(["2022-01-01", "2022-02-01"], dtype="datetime64[ns]")
        )

    def test_zarr_append_partial_overlap_skips_existing_keeps_new(self, tmp_path):
        """When a second write contains both existing and new timestamps,
        existing ones are dropped and only the new one is appended."""
        bbox = (0.0, 0.0, 2.0, 2.0)
        zarr_root = tmp_path / "zarr"
        lats = np.array([0.0, 0.75, 1.5], dtype="float64")
        lons = np.array([0.0, 0.75, 1.5], dtype="float64")

        # First write: Jan 2022
        times1 = np.array(["2022-01-01"], dtype="datetime64[ns]")
        da1 = xr.DataArray(
            np.random.rand(1, 3, 3).astype("float32"),
            dims=("valid_time", "latitude", "longitude"),
            coords={"valid_time": times1, "latitude": lats, "longitude": lons},
            name="pm10",
        )
        nc1 = tmp_path / "nc_jan.nc"
        da1.to_dataset().to_netcdf(nc1)
        _cams_netcdf_to_zarr(nc1, zarr_root, "cams/pm10", variable="pm10", bbox=bbox)

        # Second write: Jan (already present) + Feb (new)
        times2 = np.array(["2022-01-01", "2022-02-01"], dtype="datetime64[ns]")
        da2 = xr.DataArray(
            np.random.rand(2, 3, 3).astype("float32"),
            dims=("valid_time", "latitude", "longitude"),
            coords={"valid_time": times2, "latitude": lats, "longitude": lons},
            name="pm10",
        )
        nc2 = tmp_path / "nc_jan_feb.nc"
        da2.to_dataset().to_netcdf(nc2)
        _cams_netcdf_to_zarr(nc2, zarr_root, "cams/pm10", variable="pm10", bbox=bbox)

        stored = xr.open_zarr(str(zarr_root), group="cams/pm10", consolidated=False)
        assert len(stored["time"]) == 2  # Jan + Feb, no duplicate

    def test_zarr_append_adds_new_timestamps(self, tmp_path):
        """A second write with NEW timestamps must append to the existing group."""
        bbox = (0.0, 0.0, 2.0, 2.0)
        zarr_root = tmp_path / "zarr"

        # First NetCDF: Jan + Feb 2022
        (tmp_path / "nc1").mkdir()
        nc1 = self._make_cams_netcdf(tmp_path / "nc1", "co", has_pressure_level=False)
        _cams_netcdf_to_zarr(nc1, zarr_root, "cams/co", variable="co", bbox=bbox)

        # Second NetCDF: Mar 2022 only
        lats = np.array([0.0, 0.75, 1.5], dtype="float64")
        lons = np.array([0.0, 0.75, 1.5], dtype="float64")
        times_mar = np.array(["2022-03-01"], dtype="datetime64[ns]")
        data = np.random.rand(1, 3, 3).astype("float32")
        da2 = xr.DataArray(
            data,
            dims=("valid_time", "latitude", "longitude"),
            coords={"valid_time": times_mar, "latitude": lats, "longitude": lons},
            name="co",
        )
        nc2 = tmp_path / "nc2" / "cams_co_mar.nc"
        nc2.parent.mkdir(parents=True)
        da2.to_dataset().to_netcdf(nc2)
        _cams_netcdf_to_zarr(nc2, zarr_root, "cams/co", variable="co", bbox=bbox)

        stored = xr.open_zarr(str(zarr_root), group="cams/co", consolidated=False)
        assert len(stored["time"]) == 3  # Jan + Feb + Mar

    def test_level_dim_alternative_name(self, tmp_path):
        """Handles NetCDF files that use 'level' instead of 'pressure_level'."""
        lats = np.array([0.0, 0.75, 1.5], dtype="float64")
        lons = np.array([0.0, 0.75, 1.5], dtype="float64")
        times = np.array(["2022-01-01"], dtype="datetime64[ns]")
        data = np.random.rand(1, 1, 3, 3).astype("float32")
        da = xr.DataArray(
            data,
            dims=("valid_time", "level", "latitude", "longitude"),
            coords={
                "valid_time": times,
                "level": np.array([1000.0]),
                "latitude": lats,
                "longitude": lons,
            },
            name="no2",
        )
        nc_path = tmp_path / "cams_level_dim.nc"
        da.to_dataset().to_netcdf(nc_path)

        zarr_root = tmp_path / "zarr"
        ds = _cams_netcdf_to_zarr(nc_path, zarr_root, "cams/no2", variable="no2", bbox=(0, 0, 2, 2))

        assert "level" not in ds.dims
        assert "no2" in ds

    def test_variable_rename_when_named_by_long_name(self, tmp_path):
        """If the file stores the variable under its long name, it's renamed to the short name."""
        lats = np.array([0.0, 0.75, 1.5], dtype="float64")
        lons = np.array([0.0, 0.75, 1.5], dtype="float64")
        times = np.array(["2022-01-01"], dtype="datetime64[ns]")
        data = np.random.rand(1, 3, 3).astype("float32")
        # Store under the long name 'nitrogen_dioxide' rather than 'no2'
        da = xr.DataArray(
            data,
            dims=("valid_time", "latitude", "longitude"),
            coords={"valid_time": times, "latitude": lats, "longitude": lons},
            name="nitrogen_dioxide",
        )
        nc_path = tmp_path / "cams_long_name.nc"
        da.to_dataset().to_netcdf(nc_path)

        zarr_root = tmp_path / "zarr"
        ds = _cams_netcdf_to_zarr(nc_path, zarr_root, "cams/no2", variable="no2", bbox=(0, 0, 2, 2))

        assert "no2" in ds


class TestIsConfigured:
    def test_true_when_ads_key_set(self, monkeypatch):
        from eostrata.config import settings
        from eostrata.sources.cams import CAMSSource

        monkeypatch.setattr(settings, "ads_key", "uid:apikey")
        ok, msg = CAMSSource.is_configured()
        assert ok is True and msg == ""

    def test_true_when_adsapirc_exists(self, tmp_path, monkeypatch):
        from eostrata.config import settings
        from eostrata.sources.cams import CAMSSource

        (tmp_path / ".adsapirc").write_text(
            "url: https://ads.atmosphere.copernicus.eu/api\nkey: fake\n"
        )
        monkeypatch.setattr(settings, "ads_key", "")
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        ok, msg = CAMSSource.is_configured()
        assert ok is True and msg == ""

    def test_false_when_not_configured(self, monkeypatch):
        from eostrata.config import settings
        from eostrata.sources.cams import CAMSSource

        monkeypatch.setattr(settings, "ads_key", "")
        monkeypatch.setattr("pathlib.Path.home", lambda: __import__("pathlib").Path("/nonexistent"))
        ok, msg = CAMSSource.is_configured()
        assert ok is False and "ADS" in msg


class TestDownloadCamsMarsNoData:
    """Cover lines 136-144 (MarsNoDataError) and line 157 (ZIP with no .nc member)."""

    def test_non_mars_exception_is_reraised(self, tmp_path, mocker):
        """Non-MARS exceptions from retrieve() are re-raised as-is (line 144)."""
        dest = tmp_path / "cams" / "cams_no2_2022_01.nc"
        mock_client = mocker.MagicMock()
        mock_cdsapi = mocker.MagicMock()
        mock_cdsapi.Client.return_value = mock_client

        def _fake_retrieve(_ds, _req, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            raise ConnectionError("network error")

        mock_client.retrieve.side_effect = _fake_retrieve
        mocker.patch("eostrata.sources.cams._get_cdsapi", return_value=mock_cdsapi)
        mock_settings = mocker.patch("eostrata.config.settings")
        mock_settings.ads_url = "https://ads.example.com/api"
        mock_settings.ads_key = "k"

        with pytest.raises(ConnectionError, match="network error"):
            _download_cams(dest, variable="no2", year=2022, months=[1], bbox=(0, 0, 10, 10))

    def test_mars_no_data_raises_runtime_error(self, tmp_path, mocker):
        """If ADS raises with MarsNoDataError in the message, a RuntimeError is raised."""
        dest = tmp_path / "cams" / "cams_no2_2022_01.nc"
        mock_client = mocker.MagicMock()
        mock_cdsapi = mocker.MagicMock()
        mock_cdsapi.Client.return_value = mock_client

        def _fake_retrieve(_ds, _req, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            raise Exception("MarsNoDataError: dataset not available")

        mock_client.retrieve.side_effect = _fake_retrieve
        mocker.patch("eostrata.sources.cams._get_cdsapi", return_value=mock_cdsapi)
        mock_settings = mocker.patch("eostrata.config.settings")
        mock_settings.ads_url = "https://ads.example.com/api"
        mock_settings.ads_key = "k"

        with pytest.raises(RuntimeError, match="production lag"):
            _download_cams(dest, variable="no2", year=2022, months=[1], bbox=(0, 0, 10, 10))

    def test_zip_without_nc_raises_runtime_error(self, tmp_path, mocker):
        """If ADS returns a ZIP containing no .nc file, a RuntimeError is raised."""
        import io
        import zipfile as zf

        dest = tmp_path / "cams" / "cams_pm2p5_2023_06.nc"

        # Build a ZIP with no .nc member
        buf = io.BytesIO()
        with zf.ZipFile(buf, "w") as z:
            z.writestr("data.txt", b"not netcdf")
        zip_bytes = buf.getvalue()

        mock_client = mocker.MagicMock()
        mock_cdsapi = mocker.MagicMock()
        mock_cdsapi.Client.return_value = mock_client

        def _fake_retrieve(_ds, _req, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(zip_bytes)

        mock_client.retrieve.side_effect = _fake_retrieve
        mocker.patch("eostrata.sources.cams._get_cdsapi", return_value=mock_cdsapi)
        mock_settings = mocker.patch("eostrata.config.settings")
        mock_settings.ads_url = "https://ads.example.com/api"
        mock_settings.ads_key = "k"

        with pytest.raises(RuntimeError, match="No .nc file"):
            _download_cams(dest, variable="pm2p5", year=2023, months=[6], bbox=(0, 0, 1, 1))
