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

from fastapi import APIRouter, HTTPException, Path, Query, Request
from fastapi.responses import HTMLResponse, Response
from titiler.xarray.extensions import VariablesExtension
from titiler.xarray.factory import TilerFactory

from eostrata.aggregate import AggregatingReader
from eostrata.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["OGC Tiles"])

# Internal TiTiler factory — delegated to for actual tile rendering
_tiler = TilerFactory(
    reader=AggregatingReader,
    router_prefix="/internal",
    extensions=[VariablesExtension()],
)


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
    from fastapi import FastAPI as _FastAPI
    from httpx import ASGITransport, AsyncClient

    _app = _FastAPI()
    _app.include_router(_tiler.router, prefix="/internal")

    async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as client:
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
    datetime: str | None = Query(None, description="ISO 8601 datetime for time selection"),
    agg: str | None = Query(
        None, description="Temporal aggregation method: mean|sum|min|max|anomaly"
    ),
    colormap_name: str | None = Query(
        None, description="Colormap name (e.g. viridis, plasma, inferno, RdYlGn)"
    ),
    rescale: str | None = Query(None, description="Colormap range as min,max"),
) -> HTMLResponse:
    resolved = _resolve(collection_id, item)
    params: dict = {
        "url": resolved["zarr_root"],
        "group": resolved["zarr_group"],
        "variable": resolved["variable"],
    }
    if datetime:
        params["sel"] = f"time={datetime}"
    if agg:
        params["agg"] = agg
    if colormap_name:
        params["colormap_name"] = colormap_name
    if rescale:
        params["rescale"] = rescale

    base = str(request.base_url).rstrip("/")
    from urllib.parse import quote

    parts = []
    for k, v in params.items():
        parts.append(f"{k}={quote(str(v), safe='/,:')}")
    url = f"{base}/tiles/{tileMatrixSetId}/map.html?{'&'.join(parts)}"
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
    datetime: str | None = Query(None, description="ISO 8601 datetime for time selection"),
    agg: str | None = Query(
        None, description="Temporal aggregation method: mean|sum|min|max|anomaly"
    ),
    colormap_name: str | None = Query(
        None, description="Colormap name (e.g. viridis, plasma, RdYlGn)"
    ),
    rescale: str | None = Query(None, description="Colormap range as min,max"),
) -> dict:
    _resolve(collection_id, item)  # validate early
    base = str(request.base_url).rstrip("/")
    tile_url = (
        f"{base}/collections/{collection_id}"
        f"/tiles/{tileMatrixSetId}/{{z}}/{{x}}/{{y}}"
        f"?item={item or ''}"
    )
    if datetime:
        tile_url += f"&datetime={datetime}"
    if agg:
        tile_url += f"&agg={agg}"
    if colormap_name:
        tile_url += f"&colormap_name={colormap_name}"
    if rescale:
        tile_url += f"&rescale={rescale}"
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
    datetime: str | None = Query(None, description="ISO 8601 datetime for time selection"),
    agg: str | None = Query(
        None, description="Temporal aggregation method: mean|sum|min|max|anomaly"
    ),
    baseline: str | None = Query(
        None, description="ISO 8601 interval used as baseline for anomaly aggregation"
    ),
    colormap_name: str | None = Query(
        None, description="Colormap name (e.g. viridis, plasma, RdYlGn)"
    ),
    rescale: str | None = Query(None, description="Colormap range as min,max"),
) -> Response:
    resolved = _resolve(collection_id, item)
    params: dict = {
        "url": resolved["zarr_root"],
        "group": resolved["zarr_group"],
        "variable": resolved["variable"],
    }
    if datetime:
        params["sel"] = f"time={datetime}"
    if colormap_name:
        params["colormap_name"] = colormap_name
    if rescale:
        params["rescale"] = rescale

    return await _delegate(f"tiles/{tileMatrixSetId}/{z}/{x}/{y}", params)


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
