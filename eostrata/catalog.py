"""STAC catalogue management — create, load, register items."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pystac
import xarray as xr

logger = logging.getLogger(__name__)

CATALOG_ID = "eostrata"
CATALOG_DESCRIPTION = "eostrata earth observation data catalogue"

# STAC collection IDs
COLLECTION_WORLDPOP = "worldpop"
COLLECTION_CDS = "cds"
COLLECTION_CHIRPS = "chirps"

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
    Create and register a STAC item in the given collection.

    Parameters
    ----------
    catalog:
        The catalog to register into.
    collection_id:
        One of ``worldpop``, ``cds``, ``chirps``.
    item_id:
        Unique item identifier, e.g. ``worldpop_nga_2020_1km``.
    bbox:
        Spatial extent (west, south, east, north).
    datetime_:
        Reference datetime for the item.
    zarr_root:
        Path to the Zarr store root.
    zarr_group:
        Group path within the store, e.g. ``worldpop/nga_2020_1km``.
    variable:
        Name of the primary data variable inside the Zarr group.
    extra_properties:
        Any additional item properties.

    Returns
    -------
    pystac.Item
    """
    west, south, east, north = bbox
    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [west, south], [east, south],
            [east, north], [west, north],
            [west, south],
        ]],
    }

    properties = {
        "eostrata:source": collection_id,
        "eostrata:variable": variable,
        "eostrata:zarr_group": zarr_group,
        **(extra_properties or {}),
    }

    item = pystac.Item(
        id=item_id,
        geometry=geometry,
        bbox=list(bbox),
        datetime=datetime_,
        properties=properties,
    )

    item.add_asset(
        "zarr",
        pystac.Asset(
            href=str(Path(zarr_root) / zarr_group.replace("/", "_")),
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

    collection = catalog.get_child(collection_id)
    if collection is None:
        raise ValueError(
            f"Collection '{collection_id}' not found in catalog. "
            f"Available: {[c.id for c in catalog.get_children()]}"
        )

    # Replace existing item with the same id rather than duplicating
    existing = collection.get_item(item_id)
    if existing is not None:
        collection.remove_item(item_id)
        logger.info("Replaced existing STAC item '%s'", item_id)

    collection.add_item(item)
    logger.info("Registered STAC item '%s' in collection '%s'", item_id, collection_id)
    return item
