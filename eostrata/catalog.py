"""STAC catalogue management — create, load, register items, serve via stac-fastapi."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import attr
import pystac
from stac_fastapi.types.core import BaseCoreClient
from stac_fastapi.types.errors import NotFoundError
from stac_fastapi.types.stac import Collection, Collections, Item, ItemCollection

logger = logging.getLogger(__name__)

CATALOG_ID = "eostrata"
CATALOG_DESCRIPTION = "eostrata earth observation data catalogue"

# STAC collection IDs
COLLECTION_WORLDPOP = "worldpop"
COLLECTION_CDS = "cds"
COLLECTION_CHIRPS = "chirps"
COLLECTION_SENTINEL_NDVI = "sentinel_ndvi"

_COLLECTIONS = {
    COLLECTION_WORLDPOP: {
        "title": "WorldPop population",
        "description": "Global population rasters from WorldPop (worldpop.org)",
    },
    COLLECTION_CDS: {
        "title": "CDS / ERA5 climate reanalysis",
        "description": "Climate reanalysis data from the Copernicus Climate Data Store",
    },
    COLLECTION_CHIRPS: {
        "title": "CHIRPS precipitation",
        "description": "Climate Hazards Group InfraRed Precipitation with Station data",
    },
    COLLECTION_SENTINEL_NDVI: {
        "title": "Sentinel NDVI (CGLS)",
        "description": "Sentinel-3 NDVI 300m dekadal composites from the Copernicus Global Land Service",
    },
}


def _make_catalog() -> pystac.Catalog:
    """Create a fresh catalog with the three default collections."""
    catalog = pystac.Catalog(
        id=CATALOG_ID,
        description=CATALOG_DESCRIPTION,
        catalog_type=pystac.CatalogType.SELF_CONTAINED,
    )
    for coll_id, meta in _COLLECTIONS.items():
        collection = pystac.Collection(
            id=coll_id,
            title=meta["title"],
            description=meta["description"],
            extent=pystac.Extent(
                spatial=pystac.SpatialExtent(bboxes=[[-180, -90, 180, 90]]),
                temporal=pystac.TemporalExtent(intervals=[[None, None]]),
            ),
        )
        catalog.add_child(collection)
    return catalog


def create_empty() -> pystac.Catalog:
    """Create a fresh catalog with the three default collections and no items."""
    return _make_catalog()


def load_or_create(catalog_path: Path) -> pystac.Catalog:
    """Load an existing catalog.json or create a new one."""
    catalog_path = Path(catalog_path)
    if catalog_path.exists():
        logger.info("Loading existing catalog from %s", catalog_path)
        return pystac.Catalog.from_file(str(catalog_path))
    logger.info("No catalog found at %s — creating new one", catalog_path)
    return _make_catalog()


def save(catalog: pystac.Catalog, catalog_path: Path) -> None:
    """Normalise links and write catalog.json to disk."""
    catalog_path = Path(catalog_path)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.normalize_hrefs(str(catalog_path.parent))
    catalog.save(dest_href=str(catalog_path.parent))
    logger.info("Catalog saved to %s", catalog_path)


def register_item(
    catalog: pystac.Catalog,
    *,
    collection_id: str,
    item_id: str,
    bbox: tuple[float, float, float, float],
    datetime_: datetime,
    zarr_root: Path,
    zarr_group: str,
    variable: str,
    extra_properties: dict | None = None,
) -> pystac.Item:
    """
    Create or update a STAC item in the given collection.

    When an item with *item_id* already exists, its ``datetime`` interval is
    extended to include *datetime_* and the item is replaced in place.
    This allows multiple years of the same country to accumulate under one item.

    Parameters
    ----------
    catalog:
        The catalog to register into.
    collection_id:
        One of ``worldpop``, ``cds``, ``chirps``.
    item_id:
        Unique item identifier, e.g. ``worldpop_nga``.
    bbox:
        Spatial extent (west, south, east, north).
    datetime_:
        Reference datetime for the new data being added.
    zarr_root:
        Path to the Zarr store root.
    zarr_group:
        Group path within the store, e.g. ``worldpop/nga``.
    variable:
        Name of the primary data variable inside the Zarr group.
    extra_properties:
        Any additional item properties.

    Returns
    -------
    pystac.Item
    """
    collection = catalog.get_child(collection_id)
    if collection is None:
        raise ValueError(
            f"Collection '{collection_id}' not found in catalog. "
            f"Available: {[c.id for c in catalog.get_children()]}"
        )

    # Extend datetime interval and accumulate ingested timestamps if item already exists
    existing = collection.get_item(item_id)
    if existing is not None:
        # Read interval from properties since we use start/end not datetime
        existing_start = existing.common_metadata.start_datetime or existing.datetime
        existing_end = existing.common_metadata.end_datetime or existing.datetime
        if existing_start and existing_end:
            interval_start = min(
                existing_start.replace(tzinfo=UTC)
                if existing_start.tzinfo is None
                else existing_start,
                datetime_,
            )
            interval_end = max(
                existing_end.replace(tzinfo=UTC) if existing_end.tzinfo is None else existing_end,
                datetime_,
            )
        else:
            interval_start = datetime_
            interval_end = datetime_
        # Accumulate the exact list of ingested timestamps (preserves gap information)
        existing_timestamps: list[str] = existing.properties.get("eostrata:datetimes", [])
        new_ts = datetime_.isoformat()
        ingested_datetimes = sorted(set(existing_timestamps) | {new_ts})
        collection.remove_item(item_id)
        logger.info("Extending datetime interval for STAC item '%s'", item_id)
    else:
        interval_start = datetime_
        interval_end = datetime_
        ingested_datetimes = [datetime_.isoformat()]

    west, south, east, north = bbox
    geometry = {
        "type": "Polygon",
        "coordinates": [
            [
                [west, south],
                [east, south],
                [east, north],
                [west, north],
                [west, south],
            ]
        ],
    }

    properties = {
        "eostrata:source": collection_id,
        "eostrata:variable": variable,
        "eostrata:zarr_group": zarr_group,
        "eostrata:zarr_root": str(zarr_root),
        "eostrata:datetimes": ingested_datetimes,
        "datetime": None,
        "start_datetime": interval_start.isoformat(),
        "end_datetime": interval_end.isoformat(),
        **(extra_properties or {}),
    }

    item = pystac.Item(
        id=item_id,
        geometry=geometry,
        bbox=list(bbox),
        datetime=None,
        properties=properties,
    )
    item.common_metadata.start_datetime = interval_start
    item.common_metadata.end_datetime = interval_end

    item.add_asset(
        "zarr",
        pystac.Asset(
            href=str(Path(zarr_root) / zarr_group),
            media_type="application/vnd+zarr",
            roles=["data"],
            extra_fields={
                "xarray:open_kwargs": {
                    "engine": "zarr",
                    "group": zarr_group,
                    "consolidated": True,
                },
                "xarray:variable": variable,
            },
        ),
    )

    collection.add_item(item)
    logger.info("Registered STAC item '%s' in collection '%s'", item_id, collection_id)
    return item


def remove_timestamp(
    catalog: pystac.Catalog,
    group_path: str,
    timestamp_iso: str,
) -> bool:
    """Remove one timestamp from the STAC item that references *group_path*.

    Matches by the date portion of *timestamp_iso* (first 10 chars) against
    ``eostrata:datetimes`` values.  If no timestamps remain, the item is
    removed from its collection entirely.

    Returns True if the catalog was modified.
    """
    ts_date = timestamp_iso[:10]
    for collection in catalog.get_children():
        if not isinstance(collection, pystac.Collection):
            continue
        for item in list(collection.get_items()):
            if item.properties.get("eostrata:zarr_group") != group_path:
                continue
            datetimes: list[str] = item.properties.get("eostrata:datetimes", [])
            remaining = [dt for dt in datetimes if not dt.startswith(ts_date)]
            if len(remaining) == len(datetimes):
                continue
            if not remaining:
                collection.remove_item(item.id)
                logger.info("Removed STAC item '%s' (no timestamps remain)", item.id)
            else:
                dts = [datetime.fromisoformat(dt) for dt in remaining]
                start = min(dts).replace(tzinfo=UTC) if min(dts).tzinfo is None else min(dts)
                end = max(dts).replace(tzinfo=UTC) if max(dts).tzinfo is None else max(dts)
                item.properties["eostrata:datetimes"] = sorted(remaining)
                item.properties["start_datetime"] = start.isoformat()
                item.properties["end_datetime"] = end.isoformat()
                item.common_metadata.start_datetime = start
                item.common_metadata.end_datetime = end
                logger.info("Removed timestamp '%s' from STAC item '%s'", timestamp_iso, item.id)
            return True
    logger.debug("Timestamp '%s' not found in catalog for group '%s'", timestamp_iso, group_path)
    return False


def resolve_item(
    catalog_path: Path,
    collection_id: str,
    item_id: str,
) -> dict:
    """
    Look up a STAC item and return the zarr_root, zarr_group and variable
    needed to open the dataset.

    Returns
    -------
    dict with keys: zarr_root, zarr_group, variable
    """
    catalog = load_or_create(catalog_path)
    collection = catalog.get_child(collection_id)
    if collection is None:
        raise ValueError(f"Collection '{collection_id}' not found.")
    item = collection.get_item(item_id)
    if item is None:
        raise ValueError(f"Item '{item_id}' not found in collection '{collection_id}'.")

    zarr_asset = item.assets.get("zarr")
    if zarr_asset is None:
        raise ValueError(f"Item '{item_id}' has no zarr asset.")

    return {
        "zarr_root": item.properties.get("eostrata:zarr_root", "data/zarr"),
        "zarr_group": item.properties["eostrata:zarr_group"],
        "variable": item.properties["eostrata:variable"],
        "start_datetime": item.properties.get("start_datetime"),
        "end_datetime": item.properties.get("end_datetime"),
    }


# ── stac-fastapi client ───────────────────────────────────────────────────────


def _collection_to_dict(coll: pystac.Collection) -> dict:
    d = coll.to_dict()
    d.setdefault("links", [])
    return d


@attr.s
class PystacClient(BaseCoreClient):
    """
    Read-only stac-fastapi client backed by the eostrata pystac catalog.json.
    Reloads from disk on every request — suitable for development scale.
    """

    catalog_path: str = attr.ib(default=None)

    def _catalog(self) -> pystac.Catalog:
        from eostrata.config import settings

        path = self.catalog_path or str(settings.catalog_path)
        return load_or_create(path)

    def all_collections(self, **kwargs: Any) -> Collections:
        cat = self._catalog()
        colls = [
            _collection_to_dict(c) for c in cat.get_children() if isinstance(c, pystac.Collection)
        ]
        return Collections(collections=colls, links=[])

    def get_collection(self, collection_id: str, **kwargs: Any) -> Collection:
        cat = self._catalog()
        coll = cat.get_child(collection_id)
        if coll is None or not isinstance(coll, pystac.Collection):
            raise NotFoundError(f"Collection '{collection_id}' not found.")
        return _collection_to_dict(coll)

    def item_collection(
        self, collection_id: str, bbox=None, datetime=None, limit=10, token=None, **kwargs: Any
    ) -> ItemCollection:
        cat = self._catalog()
        coll = cat.get_child(collection_id)
        if coll is None:
            raise NotFoundError(f"Collection '{collection_id}' not found.")
        items = [i.to_dict() for i in coll.get_items()]
        return ItemCollection(type="FeatureCollection", features=items, links=[])

    def get_item(self, item_id: str, collection_id: str, **kwargs: Any) -> Item:
        cat = self._catalog()
        coll = cat.get_child(collection_id)
        if coll is None:
            raise NotFoundError(f"Collection '{collection_id}' not found.")
        item = coll.get_item(item_id)
        if item is None:
            raise NotFoundError(f"Item '{item_id}' not found in '{collection_id}'.")
        return item.to_dict()

    def get_search(self, **kwargs: Any) -> ItemCollection:
        cat = self._catalog()
        items = []
        for coll in cat.get_children():
            if isinstance(coll, pystac.Collection):
                items.extend(i.to_dict() for i in coll.get_items())
        return ItemCollection(type="FeatureCollection", features=items, links=[])

    def post_search(self, search_request: Any, **kwargs: Any) -> ItemCollection:
        return self.get_search(**kwargs)
