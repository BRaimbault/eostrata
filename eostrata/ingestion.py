"""Ingestion service — download + zarr write + STAC registration.

These are plain synchronous functions, extracted from the CLI commands so they
can be called from both the CLI and the HTTP ingest API without duplication.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


def run_ingest(
    source_id: str,
    *,
    zarr_root: Path,
    raw_dir: Path,
    catalog_path: Path,
    bbox: tuple[float, float, float, float],
    quota_mb: float = 0.0,
    eviction_buffer_mb: float = 0.0,
    **source_params,
) -> tuple[list[str], bool]:
    """Generic ingestion: download + zarr write + STAC registration for any registered source.

    Returns ``(failed, saved)`` where *failed* is a list of period labels that
    encountered errors and *saved* is True if at least one period was written.
    """
    import httpx

    from eostrata import catalog as cat
    from eostrata.cache import check_and_evict
    from eostrata.sources.base import get_source

    check_and_evict(
        zarr_root, quota_mb=quota_mb, required_mb=eviction_buffer_mb, catalog_path=catalog_path
    )

    source_cls = get_source(source_id)
    source = source_cls()
    zarr_group = source.zarr_group(**source_params)
    catalogue = cat.load_or_create(catalog_path)
    failed: list[str] = []
    saved = False

    for label, period_kwargs in source_cls.iter_periods(**source_params):
        logger.info("%s: ingesting %s", source_id, label)
        try:
            paths = source.download(raw_dir, bbox, **period_kwargs)
            ds = source.to_zarr(paths[0], zarr_root, bbox, **period_kwargs)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404 and source_cls.skip_404:
                logger.warning("%s: %s not available (404), skipping", source_id, label)
                continue
            logger.error("%s: HTTP error for %s: %s", source_id, label, exc)
            failed.append(label)
            continue
        except Exception as exc:
            logger.error("%s: failed to ingest %s: %s", source_id, label, exc)
            failed.append(label)
            continue
        paths[0].unlink(missing_ok=True)
        item_bbox = source.extract_item_bbox(ds)
        for reg in source.stac_registrations(ds, period_kwargs):
            cat.register_item(
                catalogue,
                collection_id=source.collection_id,
                item_id=reg["item_id"],
                bbox=item_bbox,
                datetime_=reg["datetime_"],
                zarr_root=zarr_root,
                zarr_group=zarr_group,
                variable=reg["variable"],
                extra_properties=reg["extra_properties"],
            )
        saved = True

    if saved:
        cat.save(catalogue, catalog_path)
        logger.info("%s: STAC items saved to %s", source_id, catalog_path)

    return failed, saved


def run_worldpop_ingest(
    *,
    iso3: str,
    years: list[int],
    zarr_root: Path,
    raw_dir: Path,
    catalog_path: Path,
    bbox: tuple[float, float, float, float],
    quota_mb: float = 0.0,
    eviction_buffer_mb: float = 0.0,
) -> tuple[list[str], bool]:
    """Download WorldPop rasters, write to Zarr, and register STAC items.

    Returns ``(failed, saved)`` where *failed* is a list of period labels that
    encountered errors and *saved* is True if at least one period was written.
    """
    from eostrata import catalog as cat
    from eostrata.cache import check_and_evict
    from eostrata.sources import WorldPopSource

    check_and_evict(
        zarr_root, quota_mb=quota_mb, required_mb=eviction_buffer_mb, catalog_path=catalog_path
    )

    source = WorldPopSource()
    zarr_group = source.zarr_group(iso3=iso3)
    catalogue = cat.load_or_create(catalog_path)
    failed: list[str] = []
    saved = False

    for year in years:
        label = f"{iso3.upper()}/{year}"
        logger.info("WorldPop: ingesting iso3=%s year=%d", iso3.upper(), year)
        try:
            paths = source.download(raw_dir, bbox, iso3=iso3, year=year)
            ds = source.to_zarr(paths[0], zarr_root, bbox, iso3=iso3, year=year)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.warning("WorldPop: %s not available yet (404), skipping", label)
                continue
            logger.error("WorldPop: HTTP error for %s: %s", label, exc)
            failed.append(label)
            continue
        except Exception as exc:
            logger.error("WorldPop: failed to ingest %s: %s", label, exc)
            failed.append(label)
            continue
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
        saved = True

    if saved:
        cat.save(catalogue, catalog_path)
        logger.info("WorldPop: STAC item saved to %s", catalog_path)

    return failed, saved


def run_chirps_ingest(
    *,
    years: list[int],
    months: list[int],
    zarr_root: Path,
    raw_dir: Path,
    catalog_path: Path,
    bbox: tuple[float, float, float, float],
    quota_mb: float = 0.0,
    eviction_buffer_mb: float = 0.0,
) -> tuple[list[str], bool]:
    """Download CHIRPS rasters, write to Zarr, and register STAC items.

    Returns ``(failed, saved)`` where *failed* is a list of period labels that
    encountered errors and *saved* is True if at least one period was written.
    """
    from eostrata import catalog as cat
    from eostrata.cache import check_and_evict
    from eostrata.sources.chirps import CHIRPSSource

    check_and_evict(
        zarr_root, quota_mb=quota_mb, required_mb=eviction_buffer_mb, catalog_path=catalog_path
    )

    source = CHIRPSSource()
    zarr_group = source.zarr_group()
    catalogue = cat.load_or_create(catalog_path)
    failed: list[str] = []
    saved = False

    for year in years:
        for month in months:
            label = f"{year}-{month:02d}"
            logger.info("CHIRPS: ingesting year=%d month=%02d", year, month)
            try:
                paths = source.download(raw_dir, bbox, year=year, month=month)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    logger.warning("CHIRPS: %d-%02d not available yet (404), skipping", year, month)
                    continue
                logger.error("CHIRPS: HTTP error for %s: %s", label, exc)
                failed.append(label)
                continue
            except Exception as exc:
                logger.error("CHIRPS: failed to download %s: %s", label, exc)
                failed.append(label)
                continue
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
            saved = True

    if saved:
        cat.save(catalogue, catalog_path)
        logger.info("CHIRPS: STAC item saved to %s", catalog_path)

    return failed, saved


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
    eviction_buffer_mb: float = 0.0,
) -> tuple[list[str], bool]:
    """Download ERA5 NetCDF files, write to Zarr, and register STAC items.

    Returns ``(failed, saved)`` where *failed* is a list of period labels that
    encountered errors and *saved* is True if at least one period was written.
    """
    from eostrata import catalog as cat
    from eostrata.cache import check_and_evict
    from eostrata.sources.cds import CDSSource

    check_and_evict(
        zarr_root, quota_mb=quota_mb, required_mb=eviction_buffer_mb, catalog_path=catalog_path
    )

    source = CDSSource()
    zarr_group = source.zarr_group(variable=variable)
    catalogue = cat.load_or_create(catalog_path)
    failed: list[str] = []
    saved = False

    for year in years:
        label = f"{variable}/{year}"
        logger.info("CDS: ingesting variable=%s year=%d", variable, year)
        try:
            paths = source.download(raw_dir, bbox, variable=variable, year=year, months=months)
            ds = source.to_zarr(paths[0], zarr_root, bbox, variable=variable, year=year)
        except Exception as exc:
            logger.error("CDS: failed to ingest %s: %s", label, exc)
            failed.append(label)
            continue
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
        saved = True

    if saved:
        cat.save(catalogue, catalog_path)
        logger.info("CDS: STAC item saved to %s", catalog_path)

    return failed, saved


def run_sentinel_ndvi_ingest(
    *,
    years: list[int],
    months: list[int],
    dekads: list[int],
    zarr_root: Path,
    raw_dir: Path,
    catalog_path: Path,
    bbox: tuple[float, float, float, float],
    quota_mb: float = 0.0,
    eviction_buffer_mb: float = 0.0,
) -> tuple[list[str], bool]:
    """Download Sentinel NDVI dekadal rasters, write to Zarr, and register STAC items.

    Returns ``(failed, saved)`` where *failed* is a list of period labels that
    encountered errors and *saved* is True if at least one period was written.
    """
    from eostrata import catalog as cat
    from eostrata.cache import check_and_evict
    from eostrata.sources.sentinel_ndvi import SentinelNDVISource

    check_and_evict(
        zarr_root, quota_mb=quota_mb, required_mb=eviction_buffer_mb, catalog_path=catalog_path
    )

    source = SentinelNDVISource()
    zarr_group = source.zarr_group()
    catalogue = cat.load_or_create(catalog_path)
    failed: list[str] = []
    saved = False

    for year in years:
        for month in months:
            for dekad in dekads:
                label = f"{year}-{month:02d}-d{dekad}"
                logger.info(
                    "Sentinel NDVI: ingesting year=%d month=%02d dekad=%d", year, month, dekad
                )
                try:
                    paths = source.download(raw_dir, bbox, year=year, month=month, dekad=dekad)
                except Exception as exc:
                    logger.error("Sentinel NDVI: failed to download %s: %s", label, exc)
                    failed.append(label)
                    continue
                try:
                    ds = source.to_zarr(
                        paths[0], zarr_root, bbox, year=year, month=month, dekad=dekad
                    )
                except Exception as exc:
                    logger.error("Sentinel NDVI: failed to write Zarr for %s: %s", label, exc)
                    failed.append(label)
                    continue
                paths[0].unlink(missing_ok=True)
                logger.debug("Sentinel NDVI: removed raw file %s", paths[0])
                item_bbox = (
                    float(ds.x.min()),
                    float(ds.y.min()),
                    float(ds.x.max()),
                    float(ds.y.max()),
                )
                start_day = {1: 1, 2: 11, 3: 21}[dekad]
                cat.register_item(
                    catalogue,
                    collection_id=source.collection_id,
                    item_id=source.stac_item_id(),
                    bbox=item_bbox,
                    datetime_=datetime(year, month, start_day, tzinfo=UTC),
                    zarr_root=zarr_root,
                    zarr_group=zarr_group,
                    variable=source.VARIABLE,
                    extra_properties=source.stac_properties(
                        year=year, month=month, dekad=dekad
                    ),
                )
                saved = True

    if saved:
        cat.save(catalogue, catalog_path)
        logger.info("Sentinel NDVI: STAC item saved to %s", catalog_path)

    return failed, saved


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

        from eostrata.sources.base import _REGISTRY as _src_registry

        # Look up the source class by its zarr_prefix
        source_cls = next(
            (cls for cls in _src_registry.values() if cls.zarr_prefix == source_type),
            None,
        )
        if source_cls is None:
            logger.warning("Unknown source type '%s' in '%s' — skipping", source_type, group_path)
            ds.close()
            continue

        meta = source_cls.catalog_meta(dataset_name)
        collection_id: str = source_cls.collection_id
        item_id: str = meta["item_id"]
        variable: str = meta["variable"]
        extra: dict = meta["extra"]

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
