"""Tests for CDS source NetCDF → Zarr conversion path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr


def _write_era5_nc(path: Path, variable: str = "t2m", use_lat_lon: bool = False) -> Path:
    """Write a minimal ERA5-style NetCDF file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    times = [np.datetime64(f"2023-{m:02d}-01") for m in range(1, 3)]
    if use_lat_lon:
        y_dim, x_dim = "latitude", "longitude"
    else:
        y_dim, x_dim = "y", "x"
    y = np.linspace(14.0, 4.0, 5)
    x = np.linspace(2.0, 15.0, 5)
    data = np.full((2, 5, 5), 280.0, dtype="float32")
    ds = xr.Dataset(
        {variable: (("time", y_dim, x_dim), data)},
        coords={"time": times, y_dim: y, x_dim: x},
    )
    ds.to_netcdf(str(path))
    return path


class TestNetcdfToZarr:
    def test_basic_write(self, tmp_path):
        from eostrata.sources.cds import _netcdf_to_zarr

        nc = tmp_path / "era5_t2m_2023.nc"
        _write_era5_nc(nc, variable="t2m")
        zarr_root = tmp_path / "zarr"
        bbox = (2.0, 4.0, 15.0, 14.0)

        _netcdf_to_zarr(nc, zarr_root, "era5/t2m", variable="2m_temperature", bbox=bbox)
        assert (zarr_root / "era5" / "t2m").exists()

    def test_lat_lon_renamed_to_y_x(self, tmp_path):
        from eostrata.sources.cds import _netcdf_to_zarr

        nc = tmp_path / "era5_t2m_latlon.nc"
        _write_era5_nc(nc, variable="t2m", use_lat_lon=True)
        zarr_root = tmp_path / "zarr"
        bbox = (2.0, 4.0, 15.0, 14.0)

        _netcdf_to_zarr(nc, zarr_root, "era5/t2m", variable="2m_temperature", bbox=bbox)
        # Should write without error — lat/lon renamed to y/x
        assert (zarr_root / "era5" / "t2m").exists()

    def test_lon_lat_renamed_to_y_x(self, tmp_path):
        """_netcdf_to_zarr handles 'lon'/'lat' coord names (not just longitude/latitude)."""
        from eostrata.sources.cds import _netcdf_to_zarr

        nc = tmp_path / "era5_lonlat.nc"
        times = [np.datetime64("2023-01-01")]
        lat = np.linspace(14.0, 4.0, 5)
        lon = np.linspace(2.0, 15.0, 5)
        data = np.full((1, 5, 5), 280.0, dtype="float32")
        ds = xr.Dataset(
            {"t2m": (("time", "lat", "lon"), data)},
            coords={"time": times, "lat": lat, "lon": lon},
        )
        ds.to_netcdf(str(nc))

        zarr_root = tmp_path / "zarr"
        _netcdf_to_zarr(nc, zarr_root, "era5/t2m", variable="2m_temperature", bbox=(2.0, 4.0, 15.0, 14.0))
        assert (zarr_root / "era5" / "t2m").exists()

    def test_variable_renamed_to_short_name(self, tmp_path):
        """_netcdf_to_zarr renames the CDS long variable name to the short zarr group name."""
        from eostrata.sources.cds import _netcdf_to_zarr

        nc = tmp_path / "era5_cds_name.nc"
        times = [np.datetime64("2023-01-01")]
        y = np.linspace(14.0, 4.0, 5)
        x = np.linspace(2.0, 15.0, 5)
        data = np.full((1, 5, 5), 280.0, dtype="float32")
        # Dataset uses CDS long name "2m_temperature", not short name "t2m"
        ds = xr.Dataset(
            {"2m_temperature": (("time", "y", "x"), data)},
            coords={"time": times, "y": y, "x": x},
        )
        ds.to_netcdf(str(nc))

        zarr_root = tmp_path / "zarr"
        _netcdf_to_zarr(nc, zarr_root, "era5/t2m", variable="2m_temperature", bbox=(2.0, 4.0, 15.0, 14.0))
        assert (zarr_root / "era5" / "t2m").exists()

    def test_append_second_call(self, tmp_path):
        from eostrata.sources.cds import _netcdf_to_zarr

        nc = tmp_path / "era5_t2m.nc"
        _write_era5_nc(nc, variable="t2m")
        zarr_root = tmp_path / "zarr"
        bbox = (2.0, 4.0, 15.0, 14.0)

        _netcdf_to_zarr(nc, zarr_root, "era5/t2m", variable="2m_temperature", bbox=bbox)
        _netcdf_to_zarr(nc, zarr_root, "era5/t2m", variable="2m_temperature", bbox=bbox)
        # Second call should append — group still exists
        assert (zarr_root / "era5" / "t2m").exists()


class TestCDSSourceToZarr:
    def test_to_zarr_default_variable(self, tmp_path):
        from eostrata.sources.cds import CDSSource

        nc = tmp_path / "era5_t2m.nc"
        _write_era5_nc(nc, variable="t2m")
        zarr_root = tmp_path / "zarr"
        source = CDSSource()
        source.to_zarr(nc, zarr_root, (2.0, 4.0, 15.0, 14.0), year=2023)
        assert (zarr_root / "era5" / "t2m").exists()

    def test_to_zarr_custom_variable(self, tmp_path):
        from eostrata.sources.cds import CDSSource

        nc = tmp_path / "era5_tp.nc"
        _write_era5_nc(nc, variable="tp")
        zarr_root = tmp_path / "zarr"
        source = CDSSource()
        source.to_zarr(nc, zarr_root, (2.0, 4.0, 15.0, 14.0), variable="tp", year=2023)
        assert (zarr_root / "era5" / "tp").exists()

    def test_cdsapi_unavailable_raises_on_download(self, tmp_path):
        """CDSSource.download raises a helpful ImportError if cdsapi is missing."""
        import sys
        from unittest.mock import patch

        from eostrata.sources.cds import CDSSource

        source = CDSSource()
        with patch.dict(sys.modules, {"cdsapi": None}), pytest.raises(ImportError, match="cdsapi"):
            source.download(tmp_path, (0, 0, 10, 10), year=2023)
