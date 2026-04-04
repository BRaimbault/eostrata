"""Ingestion service — download + zarr write + STAC registration.

These are plain synchronous functions, extracted from the CLI commands so they
can be called from both the CLI and the HTTP ingest API without duplication.
"""

from __future__ import annotations

import logging
from datetime import UTC
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
    from eostrata import catalog as cat
    from eostrata.cache import check_and_evict
    from eostrata.sources.base import get_source

    check_and_evict(
        zarr_root, quota_mb=quota_mb, required_mb=eviction_buffer_mb, catalog_path=catalog_path
    )

    source_cls = get_source(source_id)
    source = source_cls()
    catalogue = cat.load_or_create(catalog_path)
    failed: list[str] = []
    saved = False

    for label, period_kwargs in source_cls.iter_periods(**source_params):
        logger.info("%s: ingesting %s", source_id, label)
        try:
            paths = source.download(raw_dir, bbox, **period_kwargs)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404 and source_cls.skip_404:
                logger.debug("%s: %s not available (404) — skipping as expected", source_id, label)
                continue
            logger.error("%s: HTTP error for %s: %s", source_id, label, exc)
            failed.append(label)
            continue
        except Exception as exc:
            logger.error("%s: failed to download %s: %s", source_id, label, exc)
            failed.append(label)
            continue
        try:
            ds = source.to_zarr(paths[0], zarr_root, bbox, **period_kwargs)
        except Exception as exc:
            logger.error("%s: failed to write Zarr for %s: %s", source_id, label, exc)
            paths[0].unlink(missing_ok=True)
            failed.append(label)
            continue
        paths[0].unlink(missing_ok=True)
        zarr_group = source.zarr_group(**period_kwargs)
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

        # Sort all timestamps up-front and register the group in a single pass:
        # register_item() once for the earliest datetime (creates the item), then
        # patch PROP_DATETIMES directly for remaining timestamps.  This avoids
        # calling register_item() N times, which would do a pystac get+remove+add
        # cycle for each timestamp (O(N²) in pystac list operations per group).
        from eostrata.constants import PROP_DATETIMES

        dts = sorted(pd.Timestamp(ts).to_pydatetime().replace(tzinfo=UTC) for ts in times)
        if not dts:
            continue

        cat.register_item(
            catalogue,
            collection_id=collection_id,
            item_id=item_id,
            bbox=bbox,
            datetime_=dts[0],
            zarr_root=zarr_root,
            zarr_group=group_path,
            variable=variable,
            extra_properties=extra,
        )

        if len(dts) > 1:
            collection_obj = catalogue.get_child(collection_id)
            registered_item = collection_obj.get_item(item_id)
            all_iso = [dt.isoformat() for dt in dts]
            registered_item.properties[PROP_DATETIMES] = all_iso
            registered_item.properties["start_datetime"] = dts[0].isoformat()
            registered_item.properties["end_datetime"] = dts[-1].isoformat()
            registered_item.common_metadata.start_datetime = dts[0]
            registered_item.common_metadata.end_datetime = dts[-1]

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
