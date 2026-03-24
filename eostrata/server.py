"""eostrata FastAPI application.

Endpoints
---------
GET  /                          OGC landing page
GET  /conformance               OGC conformance classes
GET  /collections               OGC collections list (shim)
GET  /stac                      STAC catalogue root
GET  /stac/collections          STAC collections
GET  /stac/collections/{id}/items
GET  /stac/search
GET  /tiles/...                    titiler.xarray tile endpoints
GET  /processes                 OGC Processes list
GET  /processes/zonalstats      Process description
POST /processes/zonalstats/execution
GET  /docs                      OpenAPI docs
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from stac_fastapi.api.app import StacApi
from stac_fastapi.types.config import ApiSettings
from titiler.core.errors import DEFAULT_STATUS_CODES, add_exception_handlers

from eostrata.config import settings
from eostrata.ogc.processes import router as processes_router
from eostrata.ogc.tiles import router as tiles_router
from eostrata.catalog import PystacClient

# ── Main app ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="eostrata",
    description="One tool to fetch, store, aggregate, and serve earth observation layers.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── STAC API — mount as sub-application at /stac ──────────────────────────────

_stac_api = StacApi(
    settings=ApiSettings(
        title="eostrata STAC catalogue",
        description="STAC catalogue of earth observation layers managed by eostrata.",
    ),
    client=PystacClient(),
)
# stac_api.app is the internal FastAPI instance — mount it at /stac
app.mount("/stac", _stac_api.app)

# ── TiTiler xarray — mounted at /tiles via ogc/tiles.py ──────────────────────

app.include_router(tiles_router, prefix="/tiles", tags=["Tiles"])

# ── OGC Processes ─────────────────────────────────────────────────────────────

app.include_router(processes_router)

add_exception_handlers(app, DEFAULT_STATUS_CODES)

# ── OGC Common ────────────────────────────────────────────────────────────────

@app.get("/", tags=["OGC Common"], summary="Landing page")
def landing_page() -> dict:
    return {
        "title": "eostrata",
        "description": "Earth observation layers — tiles, zonal statistics and STAC catalogue.",
        "links": [
            {"rel": "self",        "href": "/",            "type": "application/json"},
            {"rel": "conformance", "href": "/conformance", "type": "application/json"},
            {"rel": "data",        "href": "/collections", "type": "application/json"},
            {"rel": "search",      "href": "/stac/search", "type": "application/json"},
            {"rel": "docs",        "href": "/docs",        "type": "text/html"},
        ],
    }


@app.get("/conformance", tags=["OGC Common"], summary="Conformance classes")
def conformance() -> dict:
    return {
        "conformsTo": [
            "http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/core",
            "http://www.opengis.net/spec/ogcapi-common-2/1.0/conf/collections",
            "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/core",
            "http://www.opengis.net/spec/ogcapi-processes-1/1.0/conf/core",
            "http://www.opengis.net/spec/ogcapi-processes-1/1.0/conf/ogc-process-description",
            "https://api.stacspec.org/v1.0.0/core",
        ]
    }


@app.get("/collections", tags=["OGC Common"], summary="Available collections")
def collections() -> dict:
    """OGC API - Common /collections — thin shim over the STAC catalogue."""
    import pystac
    from eostrata.catalog import load_or_create
    catalogue = load_or_create(settings.catalog_path)
    result = []
    for coll in catalogue.get_children():
        if isinstance(coll, pystac.Collection):
            result.append({
                "id": coll.id,
                "title": coll.title or coll.id,
                "description": coll.description,
                "links": [
                    {"rel": "items",     "href": f"/stac/collections/{coll.id}/items"},
                    {"rel": "tiles",     "href": f"/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}"},
                    {"rel": "processes", "href": "/processes/zonalstats"},
                ],
            })
    return {"collections": result, "links": []}
