"""Zarr store helpers — clip, convert and write raster data."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
import rasterio.mask
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import array_bounds
from shapely.geometry import box

logger = logging.getLogger(__name__)


def geotiff_to_zarr(
    tif_path: Path,
    zarr_root: Path,
    dataset_name: str,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    time_coord: np.datetime64 | None = None,
    chunks: dict[str, int] | None = None,
    variable_name: str | None = None,
) -> xr.Dataset:
    """
    Clip a GeoTIFF to bbox, convert to a CF-compliant xarray Dataset
    and write it to the Zarr store.

    Parameters
    ----------
    tif_path:
        Path to the source GeoTIFF.
    zarr_root:
        Root directory of the Zarr store.
    dataset_name:
        Group name inside the store, e.g. ``worldpop/nga``.
    bbox:
        Optional (west, south, east, north) clip extent in EPSG:4326.
    time_coord:
        Optional datetime64 value. When provided, a leading ``time``
        dimension is added so multiple files can be appended.
    chunks:
        Zarr chunk sizes. Defaults to 512x512.
    variable_name:
        Name of the data variable in the dataset. Defaults to the last
        segment of dataset_name if not provided.

    Returns
    -------
    xr.Dataset
        The dataset that was written.
    """
    chunk_sizes = chunks or {"y": 512, "x": 512}
    zarr_root = Path(zarr_root)
    zarr_root.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tif_path) as src:
        if bbox is not None:
            west, south, east, north = bbox
            clip_geom = [box(west, south, east, north).__geo_interface__]
            data, transform = rasterio.mask.mask(
                src, clip_geom, crop=True, nodata=src.nodata
            )
            data = data[0].astype("float32")
            crs: CRS = src.crs or CRS.from_epsg(4326)
            nodata = src.nodata
        else:
            data = src.read(1).astype("float32")
            transform = src.transform
            crs = src.crs or CRS.from_epsg(4326)
            nodata = src.nodata

    # Replace nodata with NaN
    if nodata is not None:
        data[data == nodata] = float("nan")

    height, width = data.shape
    west_out, south_out, east_out, north_out = array_bounds(height, width, transform)
    lons = np.linspace(west_out, east_out, width, dtype="float64")
    lats = np.linspace(north_out, south_out, height, dtype="float64")

    coords: dict = {"y": lats, "x": lons}
    dims: tuple = ("y", "x")
    arr = data

    if time_coord is not None:
        coords["time"] = np.array([time_coord], dtype="datetime64[ns]")
        dims = ("time", "y", "x")
        arr = data[np.newaxis, ...]

    var_name = variable_name or dataset_name.split("/")[-1]
    da = xr.DataArray(arr, dims=dims, coords=coords, name=var_name)
    da.attrs.update(
        grid_mapping="crs",
        long_name=dataset_name,
    )

    ds = da.to_dataset()
    ds["crs"] = xr.DataArray(
        np.int32(0),
        attrs={
            "grid_mapping_name": "latitude_longitude",
            "crs_wkt": crs.to_wkt(),
            "spatial_ref": crs.to_wkt(),
        },
    )
    ds.attrs["Conventions"] = "CF-1.8"
    ds.attrs["source"] = str(tif_path.name)

    # Build per-variable chunk encoding — no dask required
    cy = chunk_sizes.get("y", 512)
    cx = chunk_sizes.get("x", 512)
    encoding: dict = {
        var_name: {
            "chunks": (1, cy, cx) if time_coord is not None else (cy, cx),
            "_FillValue": float("nan"),
        },
    }

    store_path = str(zarr_root)
    group_exists = (zarr_root / dataset_name).exists()

    if group_exists and time_coord is not None:
        # Append new timestep along the time dimension
        logger.info("Appending to existing Zarr dataset '%s'", dataset_name)
        ds.to_zarr(store_path, group=dataset_name, mode="a",
                   append_dim="time", consolidated=True)
    else:
        logger.info("Writing new Zarr dataset '%s'", dataset_name)
        ds.to_zarr(store_path, group=dataset_name, mode="w",
                   encoding=encoding, consolidated=True)

    logger.info("Done: %s", dataset_name)
    return ds
