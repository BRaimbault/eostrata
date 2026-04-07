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

# _group_lock is defined in cache.py (single source of truth for the lock path).
# Imported here so tropomi.py and cds.py can also import it from store.py without
# creating a circular dependency (cache → store would be circular; store → cache is fine).
import eostrata.config as _eostrata_config
from eostrata.cache import _group_lock  # noqa: F401 — re-exported

logger = logging.getLogger(__name__)


def netcdf_to_zarr(
    nc_path: Path,
    zarr_root: Path,
    zarr_group: str,
    *,
    variable: str,
    bbox: tuple[float, float, float, float] | None = None,
    time_coord: np.datetime64 | None = None,
    chunks: dict[str, int] | None = None,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> xr.Dataset:
    """
    Clip a NetCDF file to bbox, rename lat/lon → y/x, and write to the Zarr store.

    Parameters
    ----------
    nc_path:
        Path to the source NetCDF file.
    zarr_root:
        Root directory of the Zarr store.
    zarr_group:
        Group name inside the store, e.g. ``cgls/global``.
    variable:
        Name of the data variable to extract (e.g. ``"NDVI"``).
    bbox:
        Optional (west, south, east, north) clip extent in EPSG:4326.
    time_coord:
        Optional datetime64 value. When provided, a leading ``time``
        dimension is added so multiple files can be appended.
    chunks:
        Zarr chunk sizes. Defaults to 512x512.
    lat_dim:
        Name of the latitude dimension in the NetCDF file.
    lon_dim:
        Name of the longitude dimension in the NetCDF file.
    """
    default_tile = _eostrata_config.settings.zarr_chunk_size
    chunk_sizes = chunks or {"y": default_tile, "x": default_tile}
    zarr_root = Path(zarr_root)
    zarr_root.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(nc_path, engine="h5netcdf", mask_and_scale=True) as nc:
        da = nc[variable].squeeze()

        if bbox is not None:
            west, south, east, north = bbox
            lat = da[lat_dim]
            # lat may be descending (north→south) or ascending
            if float(lat[0]) > float(lat[-1]):
                da = da.sel({lat_dim: slice(north, south), lon_dim: slice(west, east)})
            else:
                da = da.sel({lat_dim: slice(south, north), lon_dim: slice(west, east)})

        lats = da[lat_dim].values.astype("float64")
        lons = da[lon_dim].values.astype("float64")
        data = da.values.astype("float32")

    var_name = variable.lower()

    coords: dict = {"y": lats, "x": lons}
    dims: tuple = ("y", "x")
    arr = data

    if time_coord is not None:
        coords["time"] = np.array([time_coord], dtype="datetime64[ns]")
        dims = ("time", "y", "x")
        arr = data[np.newaxis, ...]

    da_out = xr.DataArray(arr, dims=dims, coords=coords, name=var_name)
    da_out.attrs.update(grid_mapping="crs", long_name=zarr_group)

    ds = da_out.to_dataset()

    crs_wkt = CRS.from_epsg(4326).to_wkt(version="WKT2_2019")
    ds["crs"] = xr.DataArray(
        np.int32(0),
        attrs={
            "grid_mapping_name": "latitude_longitude",
            "crs_wkt": crs_wkt,
            "spatial_ref": crs_wkt,
        },
    )
    ds.attrs["Conventions"] = "CF-1.8"
    ds.attrs["source"] = str(nc_path.name)

    cy = chunk_sizes.get("y", 512)
    cx = chunk_sizes.get("x", 512)
    encoding: dict = {
        var_name: {
            "chunks": (1, cy, cx) if time_coord is not None else (cy, cx),
        },
    }

    store_path = str(zarr_root)

    with _group_lock(zarr_root, zarr_group):
        group_exists = (zarr_root / zarr_group).exists()

        if group_exists and time_coord is not None:
            try:
                existing = xr.open_zarr(store_path, group=zarr_group, consolidated=False)
                try:
                    already_present = "time" in existing and time_coord in existing["time"].values
                finally:
                    existing.close()
                if already_present:
                    logger.info(
                        "Timestamp %s already exists in '%s' — skipping duplicate write",
                        time_coord,
                        zarr_group,
                    )
                    return ds
            except (OSError, KeyError, ValueError):
                logger.debug(
                    "Could not read existing Zarr group '%s', proceeding with append", zarr_group
                )
            logger.info("Appending to existing Zarr dataset '%s'", zarr_group)
            ds.to_zarr(
                store_path,
                group=zarr_group,
                mode="a",
                append_dim="time",
                consolidated=True,
            )
        else:
            logger.info("Writing new Zarr dataset '%s'", zarr_group)
            ds.to_zarr(
                store_path,
                group=zarr_group,
                mode="w",
                encoding=encoding,
                consolidated=True,
            )

    logger.info("Done: %s", zarr_group)
    return ds


def geotiff_to_zarr(
    tif_path: Path,
    zarr_root: Path,
    zarr_group: str,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    time_coord: np.datetime64 | None = None,
    chunks: dict[str, int] | None = None,
    variable_name: str | None = None,
    nodata_override: float | None = None,
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
    zarr_group:
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
        segment of zarr_group if not provided.

    Returns
    -------
    xr.Dataset
        The dataset that was written.
    """
    default_tile = _eostrata_config.settings.zarr_chunk_size
    chunk_sizes = chunks or {"y": default_tile, "x": default_tile}
    zarr_root = Path(zarr_root)
    zarr_root.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tif_path) as src:
        if bbox is not None:
            west, south, east, north = bbox
            clip_geom = [box(west, south, east, north).__geo_interface__]
            data, transform = rasterio.mask.mask(src, clip_geom, crop=True, nodata=src.nodata)
            data = data[0].astype("float32")
            crs: CRS = src.crs or CRS.from_epsg(4326)
            nodata = src.nodata
        else:
            data = src.read(1).astype("float32")
            transform = src.transform
            crs = src.crs or CRS.from_epsg(4326)
            nodata = src.nodata

    effective_nodata = nodata_override if nodata_override is not None else nodata
    if effective_nodata is not None:
        data[data == effective_nodata] = float("nan")

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

    var_name = variable_name or zarr_group.split("/")[-1]
    da = xr.DataArray(arr, dims=dims, coords=coords, name=var_name)
    da.attrs.update(
        # CF-1.8 / GeoZarr: link data variable to its grid mapping (CRS) variable
        grid_mapping="crs",
        long_name=zarr_group,
    )

    ds = da.to_dataset()

    # GeoZarr CRS variable: scalar holding the coordinate reference system.
    # Stores WKT2 in 'crs_wkt' (CF-1.8) and 'spatial_ref' (GDAL/PROJ compatibility).
    # zarr 3 writes this as dimension-less array metadata in zarr.json.
    crs_wkt = crs.to_wkt(version="WKT2_2019")
    ds["crs"] = xr.DataArray(
        np.int32(0),
        attrs={
            "grid_mapping_name": "latitude_longitude",
            "crs_wkt": crs_wkt,
            "spatial_ref": crs_wkt,
        },
    )

    # CF-1.8 Conventions; zarr 3 stores these in zarr.json group attributes,
    # making the store self-describing for GeoZarr-compatible readers.
    ds.attrs["Conventions"] = "CF-1.8"
    ds.attrs["source"] = str(tif_path.name)

    # Chunk encoding: zarr 3 uses zstd by default (superior to zarr 2's lz4/blosc).
    # Chunk shape follows the (time, y, x) or (y, x) layout with 512-tile spatial tiles —
    # a good default for typical earth-observation resolutions (~0.01°).
    cy = chunk_sizes.get("y", 512)
    cx = chunk_sizes.get("x", 512)
    encoding: dict = {
        var_name: {
            "chunks": (1, cy, cx) if time_coord is not None else (cy, cx),
        },
    }

    store_path = str(zarr_root)

    with _group_lock(zarr_root, zarr_group):
        group_exists = (zarr_root / zarr_group).exists()

        if group_exists and time_coord is not None:
            # Check whether this timestamp is already in the store — skip if so
            try:
                existing = xr.open_zarr(store_path, group=zarr_group, consolidated=False)
                try:
                    already_present = "time" in existing and time_coord in existing["time"].values
                finally:
                    existing.close()
                if already_present:
                    logger.info(
                        "Timestamp %s already exists in '%s' — skipping duplicate write",
                        time_coord,
                        zarr_group,
                    )
                    return ds
            except (OSError, KeyError, ValueError):
                logger.debug(
                    "Could not read existing Zarr group '%s', proceeding with append", zarr_group
                )
            # Append new timestep along the time dimension
            logger.info("Appending to existing Zarr dataset '%s'", zarr_group)
            ds.to_zarr(
                store_path,
                group=zarr_group,
                mode="a",
                append_dim="time",
                consolidated=True,
            )
        else:
            logger.info("Writing new Zarr dataset '%s'", zarr_group)
            ds.to_zarr(
                store_path,
                group=zarr_group,
                mode="w",
                encoding=encoding,
                consolidated=True,
            )

    logger.info("Done: %s", zarr_group)
    return ds
