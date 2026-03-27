"""Ingestion service — download + zarr write + STAC registration.

These are plain synchronous functions, extracted from the CLI commands so they
can be called from both the CLI and the HTTP ingest API without duplication.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def run_worldpop_ingest(
    *,
    iso3: str,
    years: list[int],
    zarr_root: Path,
    raw_dir: Path,
    catalog_path: Path,
    bbox: tuple[float, float, float, float],
    quota_mb: float = 0.0,
) -> None:
    """Download WorldPop rasters, write to Zarr, and register STAC items."""
    from eostrata import catalog as cat
    from eostrata.cache import check_and_evict
    from eostrata.sources import WorldPopSource

    check_and_evict(zarr_root, quota_mb=quota_mb)

    source = WorldPopSource()
    zarr_group = source.zarr_group(iso3=iso3)
    catalogue = cat.load_or_create(catalog_path)

    for year in years:
        logger.info("WorldPop: ingesting iso3=%s year=%d", iso3.upper(), year)
        paths = source.download(raw_dir, bbox, iso3=iso3, year=year)
        ds = source.to_zarr(paths[0], zarr_root, bbox, iso3=iso3, year=year)
        paths[0].unlink(missing_ok=True)
        logger.debug("WorldPop: removed raw file %s", paths[0])
        item_bbox = (float(ds.x.min()), float(ds.y.min()), float(ds.x.max()), float(ds.y.max()))
        cat.register_item(
            catalogue,
            collection_id=source.collection_id,
            item_id=source.stac_item_id(iso3=iso3),
            bbox=item_bbox,
            datetime_=datetime(year, 1, 1, tzinfo=UTC),
            zarr_root=zarr_root,
            zarr_group=zarr_group,
            variable=source.VARIABLE,
            extra_properties=source.stac_properties(iso3=iso3, year=year),
        )

    cat.save(catalogue, catalog_path)
    logger.info("WorldPop: STAC item saved to %s", catalog_path)


def run_chirps_ingest(
    *,
    years: list[int],
    months: list[int],
    zarr_root: Path,
    raw_dir: Path,
    catalog_path: Path,
    bbox: tuple[float, float, float, float],
    quota_mb: float = 0.0,
) -> None:
    """Download CHIRPS rasters, write to Zarr, and register STAC items."""
    from eostrata import catalog as cat
    from eostrata.cache import check_and_evict
    from eostrata.sources.chirps import CHIRPSSource

    check_and_evict(zarr_root, quota_mb=quota_mb)

    source = CHIRPSSource()
    zarr_group = source.zarr_group()
    catalogue = cat.load_or_create(catalog_path)

    success = False
    for year in years:
        for month in months:
            logger.info("CHIRPS: ingesting year=%d month=%02d", year, month)
            paths = source.download(raw_dir, bbox, year=year, month=month)
            ds = source.to_zarr(paths[0], zarr_root, bbox, year=year, month=month)
            paths[0].unlink(missing_ok=True)
            logger.debug("CHIRPS: removed raw file %s", paths[0])
            item_bbox = (
                float(ds.x.min()),
                float(ds.y.min()),
                float(ds.x.max()),
                float(ds.y.max()),
            )
            cat.register_item(
                catalogue,
                collection_id=source.collection_id,
                item_id=source.stac_item_id(),
                bbox=item_bbox,
                datetime_=datetime(year, month, 1, tzinfo=UTC),
                zarr_root=zarr_root,
                zarr_group=zarr_group,
                variable=source.VARIABLE,
                extra_properties=source.stac_properties(year=year, month=month),
            )
            success = True

    if success:
        cat.save(catalogue, catalog_path)
        logger.info("CHIRPS: STAC item saved to %s", catalog_path)


def run_cds_ingest(
    *,
    variable: str,
    years: list[int],
    months: list[int],
    zarr_root: Path,
    raw_dir: Path,
    catalog_path: Path,
    bbox: tuple[float, float, float, float],
    quota_mb: float = 0.0,
) -> None:
    """Download ERA5 NetCDF files, write to Zarr, and register STAC items."""
    from eostrata import catalog as cat
    from eostrata.cache import check_and_evict
    from eostrata.sources.cds import CDSSource

    check_and_evict(zarr_root, quota_mb=quota_mb)

    source = CDSSource()
    zarr_group = source.zarr_group(variable=variable)
    catalogue = cat.load_or_create(catalog_path)

    success = False
    for year in years:
        logger.info("CDS: ingesting variable=%s year=%d", variable, year)
        paths = source.download(raw_dir, bbox, variable=variable, year=year, months=months)
        ds = source.to_zarr(paths[0], zarr_root, bbox, variable=variable, year=year)
        paths[0].unlink(missing_ok=True)
        logger.debug("CDS: removed raw file %s", paths[0])
        x_dim = "x" if "x" in ds.coords else "longitude"
        y_dim = "y" if "y" in ds.coords else "latitude"
        item_bbox = (
            float(ds[x_dim].min()),
            float(ds[y_dim].min()),
            float(ds[x_dim].max()),
            float(ds[y_dim].max()),
        )
        for month in months:
            cat.register_item(
                catalogue,
                collection_id=source.collection_id,
                item_id=source.stac_item_id(variable=variable),
                bbox=item_bbox,
                datetime_=datetime(year, month, 1, tzinfo=UTC),
                zarr_root=zarr_root,
                zarr_group=zarr_group,
                variable=variable,
                extra_properties=source.stac_properties(variable=variable, year=year),
            )
        success = True

    if success:
        cat.save(catalogue, catalog_path)
        logger.info("CDS: STAC item saved to %s", catalog_path)


def rebuild_catalog_from_zarr(
    *,
    zarr_root: Path,
    catalog_path: Path,
) -> dict[str, int]:
    """
    Rebuild the STAC catalogue from scratch by scanning all Zarr groups.

    Opens each Zarr group to read time coordinates and spatial extent directly
    from the data.  Infers collection, item ID, and variable from the group path
    convention (``worldpop/<iso3>``, ``chirps/global``, ``era5/<variable>``).

    Returns
    -------
    dict mapping ``group_path -> number of timestamps registered``.
    """
    import pandas as pd
    import xarray as xr

    from eostrata import catalog as cat
    from eostrata.cache import list_groups

    zarr_root = Path(zarr_root)
    catalog_path = Path(catalog_path)

    if not zarr_root.exists():
        logger.warning("Zarr root does not exist: %s", zarr_root)
        return {}

    catalogue = cat.create_empty()
    groups = list_groups(zarr_root)
    results: dict[str, int] = {}

    for group_path, _size_mb, _last_access in groups:
        parts = group_path.split("/")
        if len(parts) != 2:
            logger.warning("Skipping unexpected group path: %s", group_path)
            continue

        source_type, dataset_name = parts

        try:
            ds = xr.open_zarr(str(zarr_root), group=group_path, consolidated=True)
        except Exception as exc:
            logger.warning("Cannot open Zarr group '%s': %s", group_path, exc)
            continue

        try:
            times = ds.time.values
            bbox = (
                float(ds.x.min()),
                float(ds.y.min()),
                float(ds.x.max()),
                float(ds.y.max()),
            )
        except Exception as exc:
            logger.warning("Cannot read coordinates from '%s': %s", group_path, exc)
            ds.close()
            continue

        if source_type == "worldpop":
            collection_id = "worldpop"
            item_id = f"worldpop_{dataset_name}"
            variable = "population"
            extra: dict = {"eostrata:iso3": dataset_name.upper(), "eostrata:variable": variable}
        elif source_type == "chirps":
            collection_id = "chirps"
            item_id = "chirps_global"
            variable = "precipitation"
            extra = {"eostrata:variable": variable}
        elif source_type == "era5":
            collection_id = "cds"
            item_id = f"era5_{dataset_name}"
            variable = dataset_name
            extra = {"eostrata:variable": variable}
        else:
            logger.warning("Unknown source type '%s' in '%s' — skipping", source_type, group_path)
            ds.close()
            continue

        for ts in times:
            dt = pd.Timestamp(ts).to_pydatetime().replace(tzinfo=UTC)
            cat.register_item(
                catalogue,
                collection_id=collection_id,
                item_id=item_id,
                bbox=bbox,
                datetime_=dt,
                zarr_root=zarr_root,
                zarr_group=group_path,
                variable=variable,
                extra_properties=extra,
            )

        results[group_path] = len(times)
        logger.info("Rebuilt %d timestamps for '%s'", len(times), group_path)
        ds.close()

    cat.save(catalogue, catalog_path)
    logger.info(
        "Catalogue rebuilt: %d groups, %d timestamps total",
        len(results),
        sum(results.values()),
    )
    return results
