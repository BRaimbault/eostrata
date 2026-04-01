"""OGC API - Tiles router.

Exposes datasets via OGC-compatible URLs:
    GET /collections/{collection_id}/tiles/{tileMatrixSetId}/map.html
    GET /collections/{collection_id}/tiles/{tileMatrixSetId}/{z}/{x}/{y}
    GET /collections/{collection_id}/tiles/{tileMatrixSetId}/tilejson.json
    GET /collections/{collection_id}/info

Query parameters:
    item        - item id within the collection, e.g. nga
    datetime    - ISO 8601 datetime or interval, e.g. 2021-01-01/2022-12-31
    agg         - aggregation method: mean|sum|min|max|anomaly
    baseline    - ISO 8601 interval for anomaly baseline
    colormap_name, rescale, ... passed through to TiTiler
"""

from __future__ import annotations

import logging
from urllib.parse import urlencode

from fastapi import APIRouter, FastAPI, HTTPException, Path, Query, Request
from fastapi.responses import HTMLResponse, Response
from httpx import ASGITransport, AsyncClient
from titiler.xarray.extensions import VariablesExtension
from titiler.xarray.factory import TilerFactory

from eostrata.aggregate import (
    _CTX_AGG_BASELINE,
    _CTX_AGG_DATETIME,
    _CTX_AGG_METHOD,
    AggregatingReader,
)
from eostrata.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Tiles"])

# Internal TiTiler factory and app — built once at import time, reused per request
_tiler = TilerFactory(
    reader=AggregatingReader,
    router_prefix="/internal",
    extensions=[VariablesExtension()],
)
_internal_app = FastAPI()
_internal_app.include_router(_tiler.router, prefix="/internal")


def _resolve(collection_id: str, item_id: str | None) -> dict:
    """Resolve collection + optional item to zarr_root, zarr_group, variable."""
    from eostrata.catalog import load_or_create

    catalog = load_or_create(settings.catalog_path)
    collection = catalog.get_child(collection_id)
    if collection is None:
        raise HTTPException(404, detail=f"Collection '{collection_id}' not found.")

    if item_id is None:
        items = list(collection.get_items())
        if not items:
            raise HTTPException(404, detail=f"Collection '{collection_id}' has no items.")
        item = items[0]
    else:
        item = collection.get_item(item_id)
        if item is None:
            raise HTTPException(404, detail=f"Item '{item_id}' not found in '{collection_id}'.")

    if "zarr" not in item.assets:
        raise HTTPException(422, detail=f"Item '{item.id}' has no zarr asset.")

    return {
        "zarr_root": item.properties.get("eostrata:zarr_root", str(settings.zarr_root)),
        "zarr_group": item.properties["eostrata:zarr_group"],
        "variable": item.properties["eostrata:variable"],
    }


async def _delegate(path: str, params: dict) -> Response:
    """Delegate a request to the internal TiTiler app and return its response."""
    async with AsyncClient(
        transport=ASGITransport(app=_internal_app), base_url="http://test"
    ) as client:
        resp = await client.get(f"/internal/{path}", params=params)

    return Response(
        content=resp.content,
        media_type=resp.headers.get("content-type", "application/json"),
        status_code=resp.status_code,
    )


@router.get(
    "/collections/{collection_id}/tiles/{tileMatrixSetId}/map.html",
    response_class=HTMLResponse,
    summary="Interactive map viewer for a collection",
)
async def collection_map(
    request: Request,
    collection_id: str = Path(..., description="Collection ID"),
    tileMatrixSetId: str = Path(..., description="Tile matrix set"),
    item: str | None = Query(None, description="Item ID within the collection"),
    datetime: str | None = Query(
        None,
        description="ISO 8601 datetime or interval (`2023-06` or `2023-01/2023-12`).",
    ),
    agg: str | None = Query(
        None, description="Temporal aggregation method: mean|sum|min|max|anomaly"
    ),
    baseline: str | None = Query(
        None,
        description="ISO 8601 interval for anomaly baseline (e.g. `2015-01/2020-12`). Required when `agg=anomaly`.",
    ),
    colormap_name: str | None = Query(
        None, description="Colormap name (e.g. viridis, plasma, inferno)"
    ),
    rescale: str | None = Query(None, description="Colormap range as min,max"),
) -> HTMLResponse:
    _resolve(collection_id, item)  # validate collection/item exist (raises 404 if not)
    qs_params: dict = {"collection": collection_id}
    if item:
        qs_params["item"] = item
    if datetime:
        qs_params["datetime"] = datetime
    if agg:
        qs_params["agg"] = agg
    if baseline:
        qs_params["baseline"] = baseline
    if colormap_name:
        qs_params["colormap_name"] = colormap_name
    if rescale:
        qs_params["rescale"] = rescale
    base = str(request.base_url).rstrip("/")
    url = f"{base}/map?{urlencode(qs_params)}"
    return HTMLResponse(
        content=f'<meta http-equiv="refresh" content="0;url={url}">',
    )


@router.get(
    "/collections/{collection_id}/tiles/{tileMatrixSetId}/tilejson.json",
    summary="TileJSON for a collection item",
)
async def collection_tilejson(
    request: Request,
    collection_id: str = Path(..., description="Collection ID"),
    tileMatrixSetId: str = Path(..., description="Tile matrix set"),
    item: str | None = Query(None, description="Item ID within the collection"),
    datetime: str | None = Query(
        None,
        description="ISO 8601 datetime or interval (`2023-06` or `2023-01/2023-12`).",
    ),
    agg: str | None = Query(
        None, description="Temporal aggregation method: mean|sum|min|max|anomaly"
    ),
    baseline: str | None = Query(
        None,
        description="ISO 8601 interval for anomaly baseline (e.g. `2015-01/2020-12`). Required when `agg=anomaly`.",
    ),
    colormap_name: str | None = Query(None, description="Colormap name (e.g. viridis, plasma)"),
    rescale: str | None = Query(None, description="Colormap range as min,max"),
) -> dict:
    _resolve(collection_id, item)  # validate early
    base = str(request.base_url).rstrip("/")
    tile_base = f"{base}/collections/{collection_id}/tiles/{tileMatrixSetId}/{{z}}/{{x}}/{{y}}"
    qs: dict = {}
    if item:
        qs["item"] = item
    if datetime:
        qs["datetime"] = datetime
    if agg:
        qs["agg"] = agg
    if baseline:
        qs["baseline"] = baseline
    if colormap_name:
        qs["colormap_name"] = colormap_name
    if rescale:
        qs["rescale"] = rescale
    tile_url = f"{tile_base}?{urlencode(qs)}" if qs else tile_base
    return {
        "tilejson": "2.2.0",
        "name": f"{collection_id}/{item or ''}",
        "tiles": [tile_url],
        "minzoom": 0,
        "maxzoom": 18,
    }


@router.get(
    "/collections/{collection_id}/tiles/{tileMatrixSetId}/{z}/{x}/{y}",
    responses={200: {"content": {"image/png": {}}}},
    summary="Map tile for a collection item",
)
async def collection_tile(
    collection_id: str = Path(..., description="Collection ID"),
    tileMatrixSetId: str = Path(..., description="Tile matrix set"),
    z: int = Path(..., description="Zoom level"),
    x: int = Path(..., description="Tile column"),
    y: int = Path(..., description="Tile row"),
    item: str | None = Query(None, description="Item ID within the collection"),
    datetime: str | None = Query(
        None,
        description=(
            "ISO 8601 datetime or interval. "
            "Single value: `2023-06` or `2023-06-01`. "
            "Interval: `2023-01/2023-12` (enables temporal aggregation via `agg`)."
        ),
    ),
    agg: str | None = Query(
        None,
        description=(
            "Temporal aggregation method applied over a `datetime` interval: "
            "`mean`, `sum`, `min`, `max`, or `anomaly` (deviation from `baseline`). "
            "Ignored when `datetime` is a single value."
        ),
    ),
    baseline: str | None = Query(
        None,
        description=(
            "ISO 8601 interval defining the reference period for anomaly computation "
            "(e.g. `2015-01/2020-12`). Required when `agg=anomaly`, ignored otherwise."
        ),
    ),
    colormap_name: str | None = Query(None, description="Colormap name (e.g. viridis, plasma)"),
    rescale: str | None = Query(None, description="Colormap range as min,max"),
) -> Response:
    resolved = _resolve(collection_id, item)

    # Propagate aggregation parameters to AggregatingReader via context vars.
    # ASGITransport runs the inner ASGI app in the same coroutine (no new Task),
    # so ContextVar values set here are visible inside __attrs_post_init__.
    tok_dt = _CTX_AGG_DATETIME.set(datetime)
    tok_method = _CTX_AGG_METHOD.set(agg)
    tok_baseline = _CTX_AGG_BASELINE.set(baseline)

    params: dict = {
        "url": resolved["zarr_root"],
        "group": resolved["zarr_group"],
        "variable": resolved["variable"],
    }
    if colormap_name:
        params["colormap_name"] = colormap_name
    if rescale:
        params["rescale"] = rescale

    try:
        return await _delegate(f"tiles/{tileMatrixSetId}/{z}/{x}/{y}", params)
    finally:
        _CTX_AGG_DATETIME.reset(tok_dt)
        _CTX_AGG_METHOD.reset(tok_method)
        _CTX_AGG_BASELINE.reset(tok_baseline)


@router.get(
    "/collections/{collection_id}/info",
    summary="Dataset info for a collection item",
)
async def collection_info(
    collection_id: str = Path(..., description="Collection ID"),
    item: str | None = Query(None, description="Item ID within the collection"),
) -> Response:
    resolved = _resolve(collection_id, item)
    params = {
        "url": resolved["zarr_root"],
        "group": resolved["zarr_group"],
        "variable": resolved["variable"],
    }
    return await _delegate("info", params)
