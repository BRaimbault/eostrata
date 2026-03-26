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
) -> None:
    """Download WorldPop rasters, write to Zarr, and register STAC items."""
    from eostrata import catalog as cat
    from eostrata.sources import WorldPopSource

    source = WorldPopSource()
    zarr_group = source.zarr_group(iso3=iso3)
    catalogue = cat.load_or_create(catalog_path)

    for year in years:
        logger.info("WorldPop: ingesting iso3=%s year=%d", iso3.upper(), year)
        paths = source.download(raw_dir, bbox, iso3=iso3, year=year)
        ds = source.to_zarr(paths[0], zarr_root, bbox, iso3=iso3, year=year)
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
) -> None:
    """Download CHIRPS rasters, write to Zarr, and register STAC items."""
    from eostrata import catalog as cat
    from eostrata.sources.chirps import CHIRPSSource

    source = CHIRPSSource()
    zarr_group = source.zarr_group()
    catalogue = cat.load_or_create(catalog_path)

    success = False
    for year in years:
        for month in months:
            logger.info("CHIRPS: ingesting year=%d month=%02d", year, month)
            paths = source.download(raw_dir, bbox, year=year, month=month)
            ds = source.to_zarr(paths[0], zarr_root, bbox, year=year, month=month)
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
) -> None:
    """Download ERA5 NetCDF files, write to Zarr, and register STAC items."""
    from eostrata import catalog as cat
    from eostrata.sources.cds import CDSSource

    source = CDSSource()
    zarr_group = source.zarr_group(variable=variable)
    catalogue = cat.load_or_create(catalog_path)

    success = False
    for year in years:
        logger.info("CDS: ingesting variable=%s year=%d", variable, year)
        paths = source.download(raw_dir, bbox, variable=variable, year=year, months=months)
        ds = source.to_zarr(paths[0], zarr_root, bbox, variable=variable, year=year)
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
