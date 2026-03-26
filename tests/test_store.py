"""Tests for store.py — GeoTIFF to Zarr conversion."""

from __future__ import annotations

import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_bounds

from eostrata.store import geotiff_to_zarr


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
