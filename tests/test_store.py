"""Tests for store.py — GeoTIFF to Zarr conversion."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_bounds

from eostrata.store import geotiff_to_zarr, netcdf_to_zarr


def _write_tif(path, bbox, width=20, height=20, nodata=-9999.0):
    transform = from_bounds(*bbox, width=width, height=height)
    data = np.arange(width * height, dtype="float32").reshape(height, width)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return path


class TestGeotiffToZarr:
    def test_basic_write(self, tmp_path):
        tif = tmp_path / "test.tif"
        bbox = (0.0, 0.0, 10.0, 10.0)
        _write_tif(tif, bbox)
        zarr_root = tmp_path / "zarr"
        ds = geotiff_to_zarr(tif, zarr_root, "test/data", bbox=bbox)
        assert "test" in ds.data_vars or "data" in ds.data_vars
        assert (zarr_root / "test" / "data").exists()

    def test_variable_name_from_arg(self, tmp_path):
        tif = tmp_path / "test.tif"
        _write_tif(tif, (0.0, 0.0, 5.0, 5.0))
        zarr_root = tmp_path / "zarr"
        ds = geotiff_to_zarr(tif, zarr_root, "mygroup/sub", variable_name="myvar")
        assert "myvar" in ds.data_vars

    def test_with_time_coord(self, tmp_path):
        tif = tmp_path / "test.tif"
        _write_tif(tif, (0.0, 0.0, 5.0, 5.0))
        zarr_root = tmp_path / "zarr"
        tc = np.datetime64("2021-01-01", "ns")
        ds = geotiff_to_zarr(tif, zarr_root, "col/d", variable_name="val", time_coord=tc)
        assert "time" in ds.dims
        assert ds.sizes["time"] == 1

    def test_no_time_coord_no_time_dim(self, tmp_path):
        tif = tmp_path / "test.tif"
        _write_tif(tif, (0.0, 0.0, 5.0, 5.0))
        zarr_root = tmp_path / "zarr"
        ds = geotiff_to_zarr(tif, zarr_root, "col/d", variable_name="val")
        assert "time" not in ds.dims

    def test_nodata_replaced_with_nan(self, tmp_path):
        tif = tmp_path / "test.tif"
        bbox = (0.0, 0.0, 5.0, 5.0)
        transform = from_bounds(*bbox, width=4, height=4)
        data = np.full((4, 4), -9999.0, dtype="float32")
        with rasterio.open(
            tif,
            "w",
            driver="GTiff",
            height=4,
            width=4,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)
        zarr_root = tmp_path / "zarr"
        ds = geotiff_to_zarr(tif, zarr_root, "col/nd", variable_name="v")
        arr = ds["v"].values
        assert np.all(np.isnan(arr))

    def test_append_second_timestep(self, tmp_path):
        tif = tmp_path / "test.tif"
        bbox = (0.0, 0.0, 5.0, 5.0)
        _write_tif(tif, bbox)
        zarr_root = tmp_path / "zarr"
        tc1 = np.datetime64("2020-01-01", "ns")
        tc2 = np.datetime64("2021-01-01", "ns")
        geotiff_to_zarr(tif, zarr_root, "col/d", variable_name="v", time_coord=tc1)
        geotiff_to_zarr(tif, zarr_root, "col/d", variable_name="v", time_coord=tc2)
        ds = xr.open_zarr(str(zarr_root), group="col/d")
        assert ds.sizes["time"] == 2

    def test_cf_attributes(self, tmp_path):
        tif = tmp_path / "test.tif"
        _write_tif(tif, (0.0, 0.0, 5.0, 5.0))
        zarr_root = tmp_path / "zarr"
        ds = geotiff_to_zarr(tif, zarr_root, "col/d", variable_name="v")
        assert ds.attrs.get("Conventions") == "CF-1.8"
        assert "crs" in ds.data_vars

    def test_without_bbox_uses_full_extent(self, tmp_path):
        tif = tmp_path / "test.tif"
        _write_tif(tif, (0.0, 0.0, 10.0, 10.0), width=8, height=8)
        zarr_root = tmp_path / "zarr"
        ds = geotiff_to_zarr(tif, zarr_root, "col/d", variable_name="v", bbox=None)
        # All 8×8 pixels should be present
        assert ds.sizes["x"] == 8
        assert ds.sizes["y"] == 8

    def test_custom_chunks(self, tmp_path):
        tif = tmp_path / "test.tif"
        _write_tif(tif, (0.0, 0.0, 5.0, 5.0))
        zarr_root = tmp_path / "zarr"
        # Should not raise — chunks are advisory for Zarr
        ds = geotiff_to_zarr(
            tif,
            zarr_root,
            "col/d",
            variable_name="v",
            chunks={"y": 64, "x": 64},
        )
        assert "v" in ds.data_vars

    def test_duplicate_timestamp_skipped(self, tmp_path):
        """Writing the same time_coord twice should not append a duplicate."""
        tif = tmp_path / "test.tif"
        _write_tif(tif, (0.0, 0.0, 5.0, 5.0))
        zarr_root = tmp_path / "zarr"
        t = np.datetime64("2020-01-01", "ns")
        geotiff_to_zarr(tif, zarr_root, "col/d", variable_name="v", time_coord=t)
        geotiff_to_zarr(tif, zarr_root, "col/d", variable_name="v", time_coord=t)
        ds = xr.open_zarr(str(zarr_root), group="col/d", consolidated=False)
        assert len(ds["time"]) == 1

    def test_nodata_override_replaces_value_when_file_has_no_nodata(self, tmp_path):
        """nodata_override masks sentinel values even when file metadata omits nodata."""
        tif = tmp_path / "test.tif"
        bbox = (0.0, 0.0, 5.0, 5.0)
        # Write a file with -9999 values but NO nodata tag in metadata
        transform = from_bounds(*bbox, width=4, height=4)
        data = np.full((4, 4), -9999.0, dtype="float32")
        with rasterio.open(
            tif,
            "w",
            driver="GTiff",
            height=4,
            width=4,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            # nodata intentionally omitted
        ) as dst:
            dst.write(data, 1)
        zarr_root = tmp_path / "zarr"
        ds = geotiff_to_zarr(tif, zarr_root, "col/nd", variable_name="v", nodata_override=-9999.0)
        assert np.all(np.isnan(ds["v"].values))

    def test_appends_when_existing_zarr_unreadable(self, tmp_path, mocker):
        """If the existing group can't be opened, proceed with append (don't crash)."""
        tif = tmp_path / "test.tif"
        _write_tif(tif, (0.0, 0.0, 5.0, 5.0))
        zarr_root = tmp_path / "zarr"
        t1 = np.datetime64("2020-01-01", "ns")
        geotiff_to_zarr(tif, zarr_root, "col/d", variable_name="v", time_coord=t1)
        t2 = np.datetime64("2021-01-01", "ns")
        from unittest.mock import patch as _patch

        with _patch("eostrata.store.xr.open_zarr", side_effect=OSError("corrupted")):
            geotiff_to_zarr(tif, zarr_root, "col/d", variable_name="v", time_coord=t2)
        ds = xr.open_zarr(str(zarr_root), group="col/d", consolidated=False)
        assert len(ds["time"]) >= 1

    def test_concurrent_writes_no_duplicate_timestamps(self, tmp_path):
        """Concurrent writes of the same timestamp must not produce duplicates.

        Launches N threads that all try to write the same time_coord simultaneously.
        The file-based lock in geotiff_to_zarr ensures only one write succeeds and
        subsequent calls detect the existing timestamp and skip.
        """
        tif = tmp_path / "test.tif"
        _write_tif(tif, (0.0, 0.0, 5.0, 5.0))
        zarr_root = tmp_path / "zarr"
        t = np.datetime64("2020-06-01", "ns")
        n_threads = 5
        errors: list[Exception] = []

        def _write():
            try:
                geotiff_to_zarr(tif, zarr_root, "col/d", variable_name="v", time_coord=t)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_write) for _ in range(n_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert not errors, f"Thread(s) raised errors: {errors}"
        ds = xr.open_zarr(str(zarr_root), group="col/d", consolidated=False)
        assert len(ds["time"]) == 1, (
            f"Expected exactly 1 timestamp, got {len(ds['time'])}. "
            "Concurrent writes produced duplicates."
        )

    def test_concurrent_writes_different_timestamps(self, tmp_path):
        """Concurrent writes of N different timestamps must all be appended exactly once."""
        tif = tmp_path / "test.tif"
        _write_tif(tif, (0.0, 0.0, 5.0, 5.0))
        zarr_root = tmp_path / "zarr"
        timestamps = [np.datetime64(f"202{i}-01-01", "ns") for i in range(5)]
        errors: list[Exception] = []

        def _write(tc):
            try:
                geotiff_to_zarr(tif, zarr_root, "col/d", variable_name="v", time_coord=tc)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_write, args=(tc,)) for tc in timestamps]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert not errors, f"Thread(s) raised errors: {errors}"
        ds = xr.open_zarr(str(zarr_root), group="col/d", consolidated=False)
        assert len(ds["time"]) == len(timestamps), (
            f"Expected {len(timestamps)} timestamps, got {len(ds['time'])}."
        )


def _write_nc(
    tmp_path: Path,
    variable: str = "ndvi",
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    ascending: bool = False,
) -> Path:
    """Write a minimal NetCDF file for netcdf_to_zarr testing."""
    lats = np.linspace(0.0, 5.0, 10) if ascending else np.linspace(5.0, 0.0, 10)
    lons = np.linspace(0.0, 5.0, 10)
    data = np.ones((10, 10), dtype="float32") * 42.0
    da = xr.DataArray(
        data,
        dims=(lat_dim, lon_dim),
        coords={lat_dim: lats, lon_dim: lons},
        name=variable,
    )
    nc_path = tmp_path / f"{variable}.nc"
    da.to_dataset().to_netcdf(str(nc_path))
    return nc_path


class TestNetcdfToZarr:
    """Tests for netcdf_to_zarr (lines 62-161 in store.py)."""

    def test_basic_write_no_time(self, tmp_path):
        """netcdf_to_zarr writes a 2D array without a time dimension."""
        nc = _write_nc(tmp_path, "ndvi")
        zarr_root = tmp_path / "zarr"
        ds = netcdf_to_zarr(nc, zarr_root, "cgls/global", variable="ndvi")
        assert "ndvi" in ds.data_vars
        assert "time" not in ds.dims
        assert (zarr_root / "cgls" / "global").exists()

    def test_write_with_time_coord(self, tmp_path):
        """netcdf_to_zarr adds a time dimension when time_coord is given."""
        nc = _write_nc(tmp_path, "ndvi")
        zarr_root = tmp_path / "zarr"
        tc = np.datetime64("2021-06-01", "ns")
        ds = netcdf_to_zarr(nc, zarr_root, "cgls/global", variable="ndvi", time_coord=tc)
        assert "time" in ds.dims
        assert ds.sizes["time"] == 1

    def test_cf_conventions(self, tmp_path):
        """Output dataset has CF-1.8 conventions and a crs variable."""
        nc = _write_nc(tmp_path, "ndvi")
        zarr_root = tmp_path / "zarr"
        ds = netcdf_to_zarr(nc, zarr_root, "cgls/global", variable="ndvi")
        assert ds.attrs.get("Conventions") == "CF-1.8"
        assert "crs" in ds.data_vars

    def test_bbox_clip_descending_lat(self, tmp_path):
        """Bbox clip works when latitude is descending (north→south)."""
        nc = _write_nc(tmp_path, "ndvi", ascending=False)
        zarr_root = tmp_path / "zarr"
        ds = netcdf_to_zarr(
            nc, zarr_root, "cgls/global", variable="ndvi", bbox=(1.0, 1.0, 4.0, 4.0)
        )
        assert "ndvi" in ds.data_vars
        # Clipped extent should be smaller than full extent
        assert ds.sizes["x"] < 10

    def test_bbox_clip_ascending_lat(self, tmp_path):
        """Bbox clip works when latitude is ascending (south→north)."""
        nc = _write_nc(tmp_path, "ndvi", ascending=True)
        zarr_root = tmp_path / "zarr"
        ds = netcdf_to_zarr(
            nc, zarr_root, "cgls/global", variable="ndvi", bbox=(1.0, 1.0, 4.0, 4.0)
        )
        assert "ndvi" in ds.data_vars
        assert ds.sizes["x"] < 10

    def test_append_new_timestamp(self, tmp_path):
        """Second call with a different time_coord appends to the existing group."""
        nc = _write_nc(tmp_path, "ndvi")
        zarr_root = tmp_path / "zarr"
        tc1 = np.datetime64("2021-01-01", "ns")
        tc2 = np.datetime64("2021-02-01", "ns")
        netcdf_to_zarr(nc, zarr_root, "cgls/global", variable="ndvi", time_coord=tc1)
        netcdf_to_zarr(nc, zarr_root, "cgls/global", variable="ndvi", time_coord=tc2)
        stored = xr.open_zarr(str(zarr_root), group="cgls/global", consolidated=False)
        assert len(stored["time"]) == 2

    def test_duplicate_timestamp_skipped(self, tmp_path):
        """Writing the same time_coord twice does not create a duplicate entry."""
        nc = _write_nc(tmp_path, "ndvi")
        zarr_root = tmp_path / "zarr"
        tc = np.datetime64("2021-06-01", "ns")
        netcdf_to_zarr(nc, zarr_root, "cgls/global", variable="ndvi", time_coord=tc)
        netcdf_to_zarr(nc, zarr_root, "cgls/global", variable="ndvi", time_coord=tc)
        stored = xr.open_zarr(str(zarr_root), group="cgls/global", consolidated=False)
        assert len(stored["time"]) == 1

    def test_append_falls_back_when_existing_read_fails(self, tmp_path, mocker):
        """If open_zarr raises during duplicate check, proceed with append."""
        nc = _write_nc(tmp_path, "ndvi")
        zarr_root = tmp_path / "zarr"
        tc1 = np.datetime64("2021-01-01", "ns")
        netcdf_to_zarr(nc, zarr_root, "cgls/global", variable="ndvi", time_coord=tc1)

        mocker.patch("eostrata.store.xr.open_zarr", side_effect=OSError("corrupted"))
        tc2 = np.datetime64("2021-02-01", "ns")
        # Should not raise
        netcdf_to_zarr(nc, zarr_root, "cgls/global", variable="ndvi", time_coord=tc2)

    def test_no_bbox_uses_full_extent(self, tmp_path):
        """Without bbox, the full extent of the NetCDF is written."""
        nc = _write_nc(tmp_path, "ndvi")
        zarr_root = tmp_path / "zarr"
        ds = netcdf_to_zarr(nc, zarr_root, "cgls/global", variable="ndvi", bbox=None)
        assert ds.sizes["x"] == 10
        assert ds.sizes["y"] == 10

    def test_custom_chunks(self, tmp_path):
        """Custom chunk sizes are accepted without raising."""
        nc = _write_nc(tmp_path, "ndvi")
        zarr_root = tmp_path / "zarr"
        ds = netcdf_to_zarr(
            nc, zarr_root, "cgls/global", variable="ndvi", chunks={"y": 64, "x": 64}
        )
        assert "ndvi" in ds.data_vars
