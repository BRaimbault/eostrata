"""eostrata FastAPI application.

Endpoints
---------
GET  /                                                      OGC landing page
GET  /conformance                                           OGC conformance classes
GET  /collections                                           OGC collections list
GET  /collections/{id}/tiles/{tileMatrixSetId}/{z}/{x}/{y}  OGC tile
GET  /collections/{id}/tiles/{tileMatrixSetId}/map.html     map viewer
GET  /collections/{id}/info                                 dataset info
GET  /stac                                                  STAC catalogue root
GET  /stac/collections/{id}/items                          STAC items
GET  /tiles/...                                             raw TiTiler (direct access)
GET  /processes                                             OGC Processes list
POST /processes/zonalstats/execution                        zonal statistics
POST /processes/ingest/execution                            async ingest (worldpop/chirps/cds)
GET  /processes/jobs                                        list ingest jobs
GET  /processes/jobs/{job_id}                               poll ingest job
GET  /docs                                                  OpenAPI docs
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import pystac
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, JSONResponse
from stac_fastapi.api.app import StacApi
from stac_fastapi.types.config import ApiSettings
from starlette.exceptions import HTTPException as StarletteHTTPException
from titiler.core.errors import DEFAULT_STATUS_CODES, add_exception_handlers
from titiler.xarray.extensions import VariablesExtension
from titiler.xarray.factory import TilerFactory

from eostrata.aggregate import AggregatingReader
from eostrata.catalog import PystacClient, load_or_create
from eostrata.config import settings
from eostrata.ogc.ingest import router as ingest_router
from eostrata.ogc.processes import router as processes_router
from eostrata.ogc.tiles import router as collection_tiles_router

logger = logging.getLogger(__name__)

# ── Lifespan: start / stop the background scheduler ───────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    from eostrata.log import setup_logging

    setup_logging(rich_console=False)  # file-only; uvicorn owns the console

    scheduler = None
    try:
        from eostrata.scheduler import Scheduler

        scheduler = Scheduler()
        scheduler.start()
    except ImportError:
        logger.info(
            "APScheduler or PyYAML not installed — scheduler disabled. "
            "Run: uv add apscheduler pyyaml"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Scheduler failed to start: %s", exc)

    yield

    if scheduler is not None:
        scheduler.stop()


# ── Main app ──────────────────────────────────────────────────────────────────

_OPENAPI_TAGS = [
    {
        "name": "OGC Common",
        "description": (
            "Landing page, conformance classes, and the collections index. "
            "Start here — call **/examples** to see what data is currently available "
            "and get ready-to-use parameter values for the other endpoints."
        ),
    },
    {
        "name": "OGC Tiles",
        "description": (
            "Serve map tiles via **OGC API - Tiles**. "
            "Use `/collections/{collection_id}/tiles/WebMercatorQuad/{z}/{x}/{y}` for raw PNG tiles "
            "or the `map.html` variant for an interactive viewer. "
            "Call **/examples** first to find valid `collection_id`, `item`, and `datetime` values."
        ),
    },
    {
        "name": "OGC Processes",
        "description": (
            "Execute analytical processes. **zonalstats** computes per-polygon raster statistics. "
            "Use **/examples** to obtain the `group` and `variable` values for your ingested data."
        ),
    },
    {
        "name": "OGC Ingest",
        "description": (
            "Trigger async data ingestion jobs for WorldPop, CHIRPS, and CDS/ERA5. "
            "POST to an execution endpoint to start a job — the response includes a `job_id`. "
            "Poll `GET /processes/jobs/{job_id}` to check status (`running` → `succeeded` or `failed`)."
        ),
    },
    {
        "name": "Tiles (direct)",
        "description": (
            "Raw TiTiler xarray access at `/tiles/...` — accepts `url`, `group`, and `variable` directly "
            "without going through the collection catalog. Useful for development."
        ),
    },
]

app = FastAPI(
    title="eostrata",
    description=(
        "One tool to fetch, store, aggregate, and serve earth observation layers.\n\n"
        "**Quick start**: call [/examples](/examples) to see what data is currently available "
        "and get copy-pasteable parameter values for every endpoint."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=_OPENAPI_TAGS,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    # Skip tile/map requests to keep the log readable
    if not request.url.path.startswith("/tiles/"):
        logger.info(
            "%s %s %s %.0fms",
            request.method,
            request.url.path,
            response.status_code,
            ms,
        )
    return response


# ── STAC API — mounted at /stac ───────────────────────────────────────────────

_stac_api = StacApi(
    settings=ApiSettings(
        title="eostrata STAC catalogue",
        description="STAC catalogue of earth observation layers managed by eostrata.",
    ),
    client=PystacClient(),
)
app.mount("/stac", _stac_api.app)

# ── OGC collection tile routes ────────────────────────────────────────────────

app.include_router(collection_tiles_router)

# ── Raw TiTiler xarray — direct access at /tiles ─────────────────────────────
# Useful for development and direct Zarr access without collection resolution.

_raw_tiler = TilerFactory(
    reader=AggregatingReader,
    router_prefix="/tiles",
    extensions=[VariablesExtension()],
)
app.include_router(
    _raw_tiler.router, prefix="/tiles", tags=["Tiles (direct)"], include_in_schema=False
)

# ── OGC Processes ─────────────────────────────────────────────────────────────

app.include_router(processes_router)
app.include_router(ingest_router)

add_exception_handlers(app, DEFAULT_STATUS_CODES)

# ── RFC 7807 Problem Details error handlers ───────────────────────────────────

_HTTP_STATUS_TITLES = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    409: "Conflict",
    422: "Unprocessable Content",
    500: "Internal Server Error",
}


@app.exception_handler(StarletteHTTPException)
async def ogc_http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "type": "about:blank",
            "title": _HTTP_STATUS_TITLES.get(exc.status_code, "Error"),
            "status": exc.status_code,
            "detail": exc.detail,
        },
    )


def _serialisable_errors(exc: RequestValidationError) -> list:
    result = []
    for err in exc.errors():
        safe = {}
        for k, v in err.items():
            if k == "ctx":
                safe[k] = {ck: str(cv) for ck, cv in v.items()}
            else:
                safe[k] = v
        result.append(safe)
    return result


@app.exception_handler(RequestValidationError)
async def ogc_validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "type": "about:blank",
            "title": "Unprocessable Content",
            "status": 422,
            "detail": "Request validation failed.",
            "errors": _serialisable_errors(exc),
        },
    )


# ── OGC Common ────────────────────────────────────────────────────────────────


@app.get("/", tags=["OGC Common"], summary="Landing page")
def landing_page() -> dict:
    return {
        "title": "eostrata",
        "description": "Earth observation layers — tiles, zonal statistics and STAC catalogue.",
        "links": [
            {"rel": "self", "href": "/", "type": "application/json"},
            {"rel": "conformance", "href": "/conformance", "type": "application/json"},
            {"rel": "data", "href": "/collections", "type": "application/json"},
            {"rel": "search", "href": "/stac/search", "type": "application/json"},
            {"rel": "docs", "href": "/docs", "type": "text/html"},
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
    """OGC API - Common /collections — lists all ingested collections."""
    import pystac

    from eostrata.catalog import load_or_create

    catalogue = load_or_create(settings.catalog_path)
    result = []
    for coll in catalogue.get_children():
        if isinstance(coll, pystac.Collection):
            result.append(
                {
                    "id": coll.id,
                    "title": coll.title or coll.id,
                    "description": coll.description,
                    "links": [
                        {"rel": "items", "href": f"/stac/collections/{coll.id}/items"},
                        {
                            "rel": "tiles",
                            "href": f"/collections/{coll.id}/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}",
                        },
                        {
                            "rel": "map",
                            "href": f"/collections/{coll.id}/tiles/WebMercatorQuad/map.html",
                        },
                        {"rel": "processes", "href": "/processes/zonalstats"},
                    ],
                }
            )
    return {
        "collections": result,
        "links": [{"rel": "self", "href": "/collections", "type": "application/json"}],
    }


@app.get(
    "/examples",
    tags=["OGC Common"],
    summary="Ready-to-use parameter values for all endpoints",
)
def examples() -> dict:
    """
    Lists every ingested item with copy-pasteable parameter values for the other endpoints.

    Use the values here to fill in `collection_id`, `item`, `datetime`, `group`, and
    `variable` when testing tiles, tilejson, info, and zonalstats via the Swagger UI.

    Returns a warning when no data has been ingested yet.
    """
    import pystac

    from eostrata.catalog import load_or_create

    catalogue = load_or_create(settings.catalog_path)
    items_out = []
    for coll in catalogue.get_children():
        if not isinstance(coll, pystac.Collection):
            continue
        for item in coll.get_items():
            datetimes: list[str] = item.properties.get("eostrata:datetimes", [])
            if not datetimes:
                # Fallback: derive from start/end
                start = item.properties.get("start_datetime") or item.properties.get("datetime")
                if start:
                    datetimes = [start]
            variable = item.properties.get("eostrata:variable", "")
            zarr_group = item.properties.get("eostrata:zarr_group", "")
            first_dt = datetimes[0] if datetimes else None

            tile_qs = f"item={item.id}"
            if first_dt:
                tile_qs += f"&datetime={first_dt}"

            items_out.append(
                {
                    "collection_id": coll.id,
                    "item_id": item.id,
                    "variable": variable,
                    "zarr_group": zarr_group,
                    "available_datetimes": datetimes,
                    "endpoints": {
                        "tile": (
                            f"/collections/{coll.id}/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}"
                            f"?{tile_qs}"
                        ),
                        "map": (f"/collections/{coll.id}/tiles/WebMercatorQuad/map.html?{tile_qs}"),
                        "tilejson": (
                            f"/collections/{coll.id}/tiles/WebMercatorQuad/tilejson.json?{tile_qs}"
                        ),
                        "info": f"/collections/{coll.id}/info?item={item.id}",
                    },
                    "zonalstats_body": {
                        "inputs": {
                            "group": zarr_group,
                            "variable": variable,
                            "datetime": first_dt,
                            "features": {
                                "type": "FeatureCollection",
                                "features": [
                                    {
                                        "type": "Feature",
                                        "geometry": {
                                            "type": "Polygon",
                                            "coordinates": [
                                                [
                                                    [item.bbox[0], item.bbox[1]],
                                                    [item.bbox[2], item.bbox[1]],
                                                    [item.bbox[2], item.bbox[3]],
                                                    [item.bbox[0], item.bbox[3]],
                                                    [item.bbox[0], item.bbox[1]],
                                                ]
                                            ],
                                        },
                                        "properties": {},
                                    }
                                ],
                            },
                        }
                    },
                }
            )

    if not items_out:
        return {
            "warning": (
                "No data has been ingested yet. "
                "Run `eostrata download worldpop --iso nga --year 2020` "
                "(or another source) to add data, then refresh this endpoint."
            ),
            "items": [],
        }

    return {"items": items_out}


@app.get("/store-usage", tags=["OGC Common"], summary="Zarr store disk usage")
def store_usage() -> dict:
    """Return current on-disk usage of the Zarr store, quota, and per-group breakdown.

    ``groups`` is sorted oldest-first (i.e. eviction candidates first).
    ``last_accessed`` is an ISO 8601 timestamp, or ``null`` if the group has
    never been read via the tile or zonal-stats endpoints.
    """
    from datetime import UTC, datetime

    from eostrata.cache import list_groups, list_timestamps, store_size_mb

    used_mb = store_size_mb(settings.zarr_root)
    quota_mb = settings.store_quota_mb
    groups = []
    for g_path, g_size_mb, _ in list_groups(settings.zarr_root):
        ts_details = list_timestamps(settings.zarr_root, g_path)
        if not ts_details:
            continue  # skip empty/evicted groups with no time dimension
        timestamps = [
            {
                "datetime": ts_iso,
                "size_mb": ts_size_mb,
                "last_accessed": datetime.fromtimestamp(la, tz=UTC).isoformat() if la else None,
                "ingested": datetime.fromtimestamp(ing, tz=UTC).isoformat()
                if ing and not la
                else None,
            }
            for ts_iso, ts_size_mb, la, ing in ts_details
        ]
        groups.append({"group": g_path, "size_mb": round(g_size_mb, 2), "timestamps": timestamps})
    return {
        "used_mb": round(used_mb, 2),
        "quota_mb": quota_mb,
        "quota_unlimited": quota_mb <= 0,
        "used_pct": round(used_mb / quota_mb * 100, 1) if quota_mb > 0 else None,
        "groups": groups,
    }


# ── Map viewer ────────────────────────────────────────────────────────────────

_MAP_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>eostrata viewer</title>
  <meta name="viewport" content="initial-scale=1,maximum-scale=1,user-scalable=no"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.3/dist/leaflet.css"
        integrity="sha384-o/2yZuJZWGJ4s/adjxVW71R+EO/LyCwdQfP5UWSgX/w87iiTXuvDZaejd3TsN7mf"
        crossorigin="anonymous"/>
  <script src="https://unpkg.com/leaflet@1.9.3/dist/leaflet.js"
          integrity="sha384-okbbMvvx/qfQkmiQKfd5VifbKZ/W8p1qIsWvE1ROPUfHWsDcC8/BnHohF7vPg2T6"
          crossorigin="anonymous"></script>
  <style>
    /* ── Design tokens ───────────────────────────────────────────────────────
       Type scale : 11px (xs) · 12px (sm) · 13px (base) · 14px (md)
       Gray palette: Tailwind gray-500/400/300/100/50 + white
       Accent      : blue-600   Warning: amber-700
    ─────────────────────────────────────────────────────────────────────── */
    body { margin: 0; font-family: system-ui, -apple-system, sans-serif; }
    #map  { position: absolute; top: 0; bottom: 0; width: 100%; }

    /* Panel */
    #controls {
      position: absolute; top: 10px; left: 10px; z-index: 1000;
      background: #fff; padding: 14px 16px; border-radius: 8px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.18); min-width: 290px; max-width: 310px;
    }
    #controls h3 {
      margin: 0 0 10px; font-size: 14px; font-weight: 600; color: #111827;
    }

    /* Tab nav — proper tab appearance */
    .tab-nav {
      display: flex; gap: 3px; margin-bottom: 0;
      border-bottom: 2px solid #2563eb;
    }
    .tab-nav button {
      padding: 5px 12px; font-size: 12px; font-family: inherit;
      color: #6b7280; border: 1px solid #d1d5db; border-bottom: none;
      border-radius: 4px 4px 0 0; background: #f3f4f6;
      cursor: pointer; transition: background 0.12s, color 0.12s;
      margin-bottom: -2px; position: relative;
    }
    .tab-nav button.active {
      background: #fff; color: #2563eb; font-weight: 600;
      border-color: #d1d5db; border-bottom: 2px solid #fff;
    }
    .tab-nav button:not(.active):hover { background: #e5e7eb; color: #111827; }
    .tab-panel { padding-top: 10px; }

    /* Labels — uppercase, xs size, secondary color */
    #controls label {
      display: block; margin-top: 8px;
      font-size: 11px; font-weight: 500; color: #6b7280;
      text-transform: uppercase; letter-spacing: 0.05em;
    }

    /* Inputs & selects */
    #controls select,
    #controls input[type="text"],
    #controls input[type="date"] {
      width: 100%; margin-top: 3px; padding: 5px 7px;
      font-size: 13px; font-family: inherit; color: #111827;
      box-sizing: border-box; border: 1px solid #d1d5db; border-radius: 4px;
      background: #fff;
    }
    #controls select:focus,
    #controls input[type="text"]:focus,
    #controls input[type="date"]:focus {
      outline: none; border-color: #2563eb;
      box-shadow: 0 0 0 2px rgba(37,99,235,0.15);
    }

    /* Segmented date-mode toggle */
    .date-mode-toggle {
      display: flex; margin-top: 3px; border-radius: 4px; overflow: hidden;
      border: 1px solid #d1d5db;
    }
    .date-mode-toggle button {
      flex: 1; padding: 5px 0; font-size: 12px; font-family: inherit;
      color: #6b7280; border: none; background: #f9fafb;
      cursor: pointer; transition: background 0.12s, color 0.12s;
    }
    .date-mode-toggle button.active  { background: #2563eb; color: #fff; }
    .date-mode-toggle button:not(.active):hover { background: #f3f4f6; color: #111827; }

    /* Two-column date row */
    .row-2 { display: flex; gap: 6px; }
    .row-2 > div { flex: 1; min-width: 0; }
    .row-2 label { margin-top: 0; }

    /* Collapsible sections */
    #section-interval { display: none; }
    #section-baseline { display: none; margin-top: 2px; }

    /* Dividers */
    .section-divider {
      border: none; border-top: 1px solid #d1d5db; margin: 10px 0 4px;
    }

    /* Checkbox label — same size as small text, no uppercase */
    .label-checkbox {
      display: flex; align-items: center; gap: 6px; margin-top: 6px;
      font-size: 12px; font-weight: 400; color: #6b7280;
      text-transform: none; letter-spacing: 0; cursor: pointer;
    }
    .label-checkbox input[type="checkbox"] { width: auto; margin: 0; }
    #inp-rescale:disabled { background: #f3f4f6; color: #9ca3af; cursor: not-allowed; }

    /* Status bar */
    #status  { font-size: 11px; color: #9ca3af; margin-top: 10px; min-height: 14px; }
    #warning { font-size: 12px; color: #b45309; margin-top: 8px; display: none; }

    /* Zonal stats panel */
    #stats-section { display: none; margin-top: 2px; }
    #stats-section h4 {
      margin: 8px 0 6px; font-size: 11px; font-weight: 500; color: #6b7280;
      text-transform: uppercase; letter-spacing: 0.05em;
    }
    #stats-grid {
      display: grid; grid-template-columns: 1fr 1fr; gap: 4px 10px;
      font-size: 12px;
    }
    .stat-item  { display: flex; justify-content: space-between; }
    .stat-label { color: #9ca3af; }
    .stat-value { font-weight: 600; color: #111827; }
    #stats-msg  { font-size: 11px; color: #9ca3af; margin-top: 4px; min-height: 14px; }

    /* ── Ingest tab ─────────────────────────────────────────────────────── */
    .btn {
      display: block; width: 100%; margin-top: 8px; padding: 6px 10px;
      font-size: 13px; font-family: inherit; font-weight: 500;
      border: none; border-radius: 4px; cursor: pointer;
      background: #2563eb; color: #fff; transition: background 0.12s;
    }
    .btn:hover:not(:disabled) { background: #1d4ed8; }
    .btn:disabled { background: #93c5fd; cursor: not-allowed; }
    .btn.secondary {
      background: #f3f4f6; color: #374151; border: 1px solid #d1d5db;
    }
    .btn.secondary:hover:not(:disabled) { background: #e5e7eb; }

    .ing-msg { font-size: 11px; margin-top: 6px; min-height: 14px; }
    .ing-msg.ok  { color: #166534; }
    .ing-msg.err { color: #991b1b; }
    .ing-msg.dim { color: #9ca3af; }

    /* Job list */
    #jobs-list { margin-top: 6px; max-height: 200px; overflow-y: auto; }
    .job-card {
      border: 1px solid #e5e7eb; border-radius: 4px; padding: 6px 8px;
      margin-bottom: 5px; font-size: 12px;
    }
    .job-card-header { display: flex; justify-content: space-between; align-items: center; }
    .job-source { font-weight: 600; color: #111827; }
    .job-id     { font-size: 10px; color: #9ca3af; margin-top: 2px;
                  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .job-time   { font-size: 10px; color: #9ca3af; margin-top: 1px; }
    .job-error  { font-size: 10px; color: #991b1b; margin-top: 3px;
                  white-space: pre-wrap; word-break: break-all; max-height: 60px;
                  overflow-y: auto; }
    .badge {
      display: inline-block; padding: 1px 7px; border-radius: 9px;
      font-size: 11px; font-weight: 500; white-space: nowrap;
    }
    .badge-running   { background: #dbeafe; color: #1e40af; }
    .badge-succeeded { background: #dcfce7; color: #166534; }
    .badge-failed    { background: #fee2e2; color: #991b1b; }

    /* Ingest tab section headers */
    .ing-section-hdr {
      display: flex; justify-content: space-between; align-items: center;
      margin-top: 4px;
    }
    .ing-section-hdr span {
      font-size: 11px; font-weight: 500; color: #6b7280;
      text-transform: uppercase; letter-spacing: 0.05em;
    }
    .ing-section-hdr button {
      font-size: 11px; font-family: inherit; padding: 2px 8px;
      border: 1px solid #d1d5db; border-radius: 4px; background: #f9fafb;
      cursor: pointer; color: #374151;
    }
    .ing-section-hdr button:hover { background: #e5e7eb; }

    /* ── Config tab ─────────────────────────────────────────────────────── */
    .cfg-section { margin-bottom: 10px; }
    .cfg-hdr {
      font-size: 11px; font-weight: 500; color: #6b7280;
      text-transform: uppercase; letter-spacing: 0.05em;
      margin-bottom: 5px;
    }
    .cfg-grid {
      display: grid; grid-template-columns: auto 1fr; gap: 3px 10px;
      font-size: 12px;
    }
    .cfg-key { color: #9ca3af; white-space: nowrap; }
    .cfg-val { color: #111827; font-weight: 500; word-break: break-all; }
    .cfg-path { font-family: monospace; font-size: 11px; font-weight: 400; }
    .cfg-group-toggle {
      width: 100%; text-align: left; background: #f9fafb; border: none;
      padding: 4px 6px; cursor: pointer; font-family: monospace; font-weight: 600;
      font-size: 12px; color: #111827;
    }
    .cfg-group-toggle:hover { background: #f3f4f6; }
    .cfg-group-toggle .cfg-arrow { display: inline-block; width: 12px; color: #9ca3af; }
    #cfg-bbox-map {
      width: 100%; height: 80px; margin-top: 6px; border-radius: 4px;
      border: 1px solid #e5e7eb; background: #f9fafb;
    }
  </style>
</head>
<body>
  <div id="map"></div>
  <div id="controls">
    <h3>eostrata viewer</h3>
    <div class="tab-nav">
      <button id="tab-btn-map" class="active" onclick="showTab('map')">Map</button>
      <button id="tab-btn-ingest" onclick="showTab('ingest')">Ingest</button>
      <button id="tab-btn-config" onclick="showTab('config')">Config</button>
    </div>

    <!-- ── Map tab ──────────────────────────────────────────────────────── -->
    <div id="tab-map-content" class="tab-panel">
      <label>Collection</label>
      <select id="sel-collection"><option value="">Loading…</option></select>

      <label>Item</label>
      <select id="sel-item"><option value="">—</option></select>

      <label>Date mode</label>
      <div class="date-mode-toggle">
        <button id="btn-mode-single" class="active" onclick="setDateMode('single')">Single date</button>
        <button id="btn-mode-interval" onclick="setDateMode('interval')">Interval</button>
      </div>

      <div id="section-single">
        <label>Date</label>
        <select id="sel-datetime"><option value="">latest</option></select>
      </div>

      <div id="section-interval">
        <div class="row-2">
          <div>
            <label>Start</label>
            <input id="inp-start" type="date"/>
          </div>
          <div>
            <label>End</label>
            <input id="inp-end" type="date"/>
          </div>
        </div>
        <label>Aggregation</label>
        <select id="sel-agg">
          <option value="mean">mean</option>
          <option value="sum">sum</option>
          <option value="min">min</option>
          <option value="max">max</option>
          <option value="anomaly">anomaly (deviation from baseline)</option>
        </select>
        <div id="section-baseline">
          <hr class="section-divider"/>
          <div class="row-2">
            <div>
              <label>Baseline start</label>
              <input id="inp-baseline-start" type="date"/>
            </div>
            <div>
              <label>Baseline end</label>
              <input id="inp-baseline-end" type="date"/>
            </div>
          </div>
        </div>
      </div>

      <hr class="section-divider"/>
      <label>Colormap</label>
      <select id="sel-colormap">
        <option value="">default</option>
        <option value="viridis">viridis</option>
        <option value="plasma">plasma</option>
        <option value="inferno">inferno</option>
        <option value="magma">magma</option>
        <option value="coolwarm">coolwarm</option>
      </select>
      <label>Rescale (min,max)</label>
      <input id="inp-rescale" type="text" placeholder="e.g. 0,1000"/>
      <label class="label-checkbox">
        <input id="chk-autoscale" type="checkbox"/>
        Auto-scale from stats
      </label>

      <div id="warning"></div>
      <div id="status"></div>
      <div id="stats-section">
        <hr class="section-divider"/>
        <h4>Zonal stats — bbox</h4>
        <div id="stats-grid"></div>
        <div id="stats-msg"></div>
      </div>
    </div><!-- /tab-map-content -->

    <!-- ── Ingest tab ────────────────────────────────────────────────────── -->
    <div id="tab-ingest-content" class="tab-panel" style="display:none">

      <!-- Start job form -->
      <label>Source</label>
      <select id="ing-source" onchange="onIngSourceChange()">
        <option value="worldpop">worldpop — population rasters</option>
        <option value="chirps">chirps — precipitation</option>
        <option value="cds">cds — ERA5 reanalysis</option>
      </select>

      <div id="ing-iso3-row">
        <label>ISO3 country code</label>
        <input id="ing-iso3" type="text" placeholder="e.g. NGA" maxlength="3"
               style="text-transform:uppercase"/>
      </div>

      <div id="ing-var-row" style="display:none">
        <label>Variable</label>
        <select id="ing-variable">
          <option value="t2m">t2m — 2m temperature</option>
          <option value="tp">tp — total precipitation</option>
          <option value="u10">u10 — 10m u-wind</option>
          <option value="v10">v10 — 10m v-wind</option>
          <option value="sp">sp — surface pressure</option>
        </select>
      </div>

      <label>Years (comma-separated, blank = latest)</label>
      <input id="ing-years" type="text" placeholder="e.g. 2023,2024"/>

      <div id="ing-months-row" style="display:none">
        <label>Months (comma-separated, blank = latest)</label>
        <input id="ing-months" type="text" placeholder="e.g. 1,2,3 or ALL"/>
      </div>

      <button class="btn" id="btn-ingest" onclick="startIngest()">Start ingestion job</button>
      <div id="ing-msg" class="ing-msg dim"></div>

      <hr class="section-divider"/>

      <!-- Jobs list -->
      <div class="ing-section-hdr">
        <span>Jobs</span>
        <button onclick="loadJobs()">Refresh</button>
      </div>
      <div id="jobs-list"><span style="font-size:12px;color:#9ca3af">No jobs yet.</span></div>

      <hr class="section-divider"/>

      <!-- Catalogue tools -->
      <div class="ing-section-hdr" style="margin-top:6px">
        <span>Catalogue</span>
      </div>
      <button class="btn secondary" id="btn-rebuild" onclick="rebuildCatalog()">
        Rebuild catalog from Zarr
      </button>
      <div id="rebuild-msg" class="ing-msg dim"></div>

    </div><!-- /tab-ingest-content -->

    <!-- ── Config tab ───────────────────────────────────────────────────── -->
    <div id="tab-config-content" class="tab-panel" style="display:none">

      <div class="cfg-section">
        <div class="cfg-hdr">Bounding box</div>
        <div class="cfg-grid">
          <span class="cfg-key">West</span><span id="cfg-bbox-w" class="cfg-val"></span>
          <span class="cfg-key">South</span><span id="cfg-bbox-s" class="cfg-val"></span>
          <span class="cfg-key">East</span><span id="cfg-bbox-e" class="cfg-val"></span>
          <span class="cfg-key">North</span><span id="cfg-bbox-n" class="cfg-val"></span>
        </div>
      </div>

      <div class="cfg-section">
        <div class="cfg-hdr">Storage</div>
        <div class="cfg-grid">
          <span class="cfg-key">Zarr root</span><span id="cfg-zarr-root" class="cfg-val cfg-path"></span>
          <span class="cfg-key">Raw dir</span><span id="cfg-raw-dir" class="cfg-val cfg-path"></span>
          <span class="cfg-key">Catalog</span><span id="cfg-catalog" class="cfg-val cfg-path"></span>
        </div>
      </div>

      <div class="cfg-section">
        <div class="cfg-hdr">Cache</div>
        <div class="cfg-grid">
          <span class="cfg-key">Used</span><span id="cfg-used" class="cfg-val"></span>
          <span class="cfg-key">Quota</span><span id="cfg-quota" class="cfg-val"></span>
        </div>
        <div id="cfg-quota-bar-wrap" style="display:none;margin-top:8px">
          <div style="background:#e5e7eb;border-radius:4px;height:10px;overflow:hidden">
            <div id="cfg-quota-bar" style="height:100%;background:#2563eb;width:0%;transition:width .4s"></div>
          </div>
          <div id="cfg-quota-pct" style="font-size:11px;color:#6b7280;margin-top:3px;text-align:right"></div>
        </div>
        <div id="cfg-groups-wrap" style="display:none;margin-top:12px">
          <table style="width:100%;border-collapse:collapse;font-size:12px">
            <thead>
              <tr style="color:#6b7280;text-align:left;border-bottom:1px solid #e5e7eb">
                <th style="padding:4px 6px;font-weight:600">Group</th>
                <th style="padding:4px 6px;font-weight:600;text-align:right">Size</th>
                <th style="padding:4px 6px;font-weight:600;text-align:right">Last accessed</th>
              </tr>
            </thead>
            <tbody id="cfg-groups-tbody"></tbody>
          </table>
        </div>
      </div>

    </div><!-- /tab-config-content -->
  </div>
  <script>
    const PRESELECT = __PRESELECT__;
    const CONFIG    = __CONFIG__;

    const map = L.map('map').setView([5, 20], 4);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '\\u00a9 <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      opacity: 0.5
    }).addTo(map);

    // Draw the configured bounding box on the map
    (function drawBbox() {
      const [w, s, e, n] = CONFIG.bbox;
      const isWorld = w === -180 && s === -90 && e === 180 && n === 90;
      if (isWorld) return; // skip trivial world bbox
      L.rectangle([[s, w], [n, e]], {
        color: '#2563eb', weight: 2, dashArray: '6 4', fill: false,
      }).addTo(map).bindTooltip('Config bbox', {permanent: false});
      map.fitBounds([[s, w], [n, e]], {padding: [20, 20]});
    })();

    // Populate config tab values
    (function populateConfig() {
      const [w, s, e, n] = CONFIG.bbox;
      document.getElementById('cfg-bbox-w').textContent = w.toFixed(4) + '°';
      document.getElementById('cfg-bbox-s').textContent = s.toFixed(4) + '°';
      document.getElementById('cfg-bbox-e').textContent = e.toFixed(4) + '°';
      document.getElementById('cfg-bbox-n').textContent = n.toFixed(4) + '°';
      document.getElementById('cfg-zarr-root').textContent = CONFIG.zarr_root;
      document.getElementById('cfg-raw-dir').textContent   = CONFIG.raw_dir;
      document.getElementById('cfg-catalog').textContent   = CONFIG.catalog_path;
      document.getElementById('cfg-quota').textContent =
        CONFIG.quota_mb > 0 ? CONFIG.quota_mb.toLocaleString() + ' MB' : 'Unlimited';

      fetch('/store-usage').then(r => r.json()).then(u => {
        const fmtMb = mb => mb >= 1024
          ? (mb / 1024).toFixed(2) + ' GB'
          : mb.toFixed(1) + ' MB';
        const fmtTs = iso => iso ? iso.slice(0, 10) : '—';
        document.getElementById('cfg-used').textContent = fmtMb(u.used_mb);
        if (!u.quota_unlimited && u.used_pct !== null) {
          const pct = Math.min(u.used_pct, 100);
          const bar = document.getElementById('cfg-quota-bar');
          bar.style.width = pct + '%';
          bar.style.background = pct >= 90 ? '#dc2626' : pct >= 70 ? '#f59e0b' : '#2563eb';
          document.getElementById('cfg-quota-pct').textContent = u.used_pct + '% of quota used';
          document.getElementById('cfg-quota-bar-wrap').style.display = 'block';
        }
        if (u.groups && u.groups.length) {
          const tbody = document.getElementById('cfg-groups-tbody');
          tbody.innerHTML = '';
          u.groups.forEach((g, gi) => {
            const evictRisk = !u.quota_unlimited && gi === 0 && u.used_pct > 80;
            const groupId = 'cfg-grp-' + gi;

            // Collapsible group header row
            const trg = document.createElement('tr');
            trg.style.background = '#f9fafb';
            const trgTd = document.createElement('td');
            trgTd.colSpan = 3;
            trgTd.style.padding = '0';
            const btn = document.createElement('button');
            btn.className = 'cfg-group-toggle';
            if (evictRisk) btn.style.color = '#dc2626';
            btn.setAttribute('aria-expanded', 'false');
            btn.setAttribute('aria-controls', groupId);
            btn.innerHTML =
              '<span class="cfg-arrow">▶</span> ' + g.group +
              ' <span style="font-weight:normal;color:#6b7280">(' + fmtMb(g.size_mb) +
              ', ' + (g.timestamps || []).length + ' timestamp' +
              ((g.timestamps || []).length !== 1 ? 's' : '') + ')</span>';
            btn.onclick = function() {
              const expanded = this.getAttribute('aria-expanded') === 'true';
              this.setAttribute('aria-expanded', String(!expanded));
              this.querySelector('.cfg-arrow').textContent = expanded ? '▶' : '▼';
              document.querySelectorAll('.' + groupId).forEach(r => {
                r.style.display = expanded ? 'none' : '';
              });
            };
            trgTd.appendChild(btn);
            trg.appendChild(trgTd);
            tbody.appendChild(trg);

            // Per-timestamp sub-rows, hidden by default
            (g.timestamps || []).forEach((ts, i) => {
              const tr = document.createElement('tr');
              tr.className = groupId;
              tr.style.display = 'none';
              tr.style.borderBottom = '1px solid #f3f4f6';
              if (evictRisk && i === 0) tr.style.color = '#dc2626';
              const accessCell = ts.last_accessed
                ? 'accessed ' + fmtTs(ts.last_accessed)
                : ts.ingested ? '<span style="color:#9ca3af">ingested ' + fmtTs(ts.ingested) + '</span>' : '—';
              tr.innerHTML =
                '<td style="padding:2px 6px 2px 18px;font-family:monospace;font-size:11px">' + ts.datetime.slice(0,10) + '</td>' +
                '<td style="padding:2px 6px;text-align:right;font-size:11px">' + fmtMb(ts.size_mb) + '</td>' +
                '<td style="padding:2px 6px;text-align:right;font-size:11px">' + accessCell + '</td>';
              tbody.appendChild(tr);
            });
          });
          document.getElementById('cfg-groups-wrap').style.display = 'block';
        }
      }).catch(() => {});
    })();

    let dataLayer = null;
    const catalog = {};
    let dateMode = 'single'; // 'single' | 'interval'

    function setStatus(msg) { document.getElementById('status').textContent = msg; }

    function setDateMode(mode) {
      dateMode = mode;
      document.getElementById('btn-mode-single').classList.toggle('active', mode === 'single');
      document.getElementById('btn-mode-interval').classList.toggle('active', mode === 'interval');
      document.getElementById('section-single').style.display = mode === 'single' ? 'block' : 'none';
      document.getElementById('section-interval').style.display = mode === 'interval' ? 'block' : 'none';
      if (mode === 'interval') {
        // Populate start/end from the currently selected item's date range
        const collId = document.getElementById('sel-collection').value;
        const itemId = document.getElementById('sel-item').value;
        const range = getItemDateRange(collId, itemId);
        if (range) {
          document.getElementById('inp-start').value = range.first;
          document.getElementById('inp-end').value = range.last;
        }
      }
      updateLayer();
    }

    function getItemDateRange(collId, itemId) {
      if (!collId || !itemId) return null;
      const item = (catalog[collId] || []).find(i => i.item_id === itemId);
      if (!item) return null;
      const dts = item.available_datetimes || [];
      if (dts.length === 0) return null;
      return { first: dts[0].slice(0, 10), last: dts[dts.length - 1].slice(0, 10) };
    }

    function onAggChange() {
      const agg = document.getElementById('sel-agg').value;
      document.getElementById('section-baseline').style.display = agg === 'anomaly' ? '' : 'none';
      updateLayer();
    }

    function onCollectionChange(preItem, preDatetime) {
      const collId = document.getElementById('sel-collection').value;
      const itemSel = document.getElementById('sel-item');
      itemSel.innerHTML = '<option value="">— select item —</option>';
      document.getElementById('sel-datetime').innerHTML = '<option value="">latest</option>';
      if (!collId) { updateLayer(); return; }
      const items = catalog[collId] || [];
      for (const it of items) {
        const opt = document.createElement('option');
        opt.value = it.item_id;
        opt.textContent = it.item_id;
        itemSel.appendChild(opt);
      }
      if (preItem && items.find(i => i.item_id === preItem)) {
        itemSel.value = preItem;
      } else if (items.length === 1) {
        itemSel.value = items[0].item_id;
      }
      onItemChange(preDatetime);
    }

    function onItemChange(preDatetime) {
      const collId = document.getElementById('sel-collection').value;
      const itemId = document.getElementById('sel-item').value;
      const dtSel = document.getElementById('sel-datetime');
      dtSel.innerHTML = '<option value="">latest</option>';
      if (!collId || !itemId) { updateLayer(); return; }
      const items = catalog[collId] || [];
      const item = items.find(i => i.item_id === itemId);
      if (!item) { updateLayer(); return; }
      const dts = item.available_datetimes || [];
      for (const dt of dts) {
        const opt = document.createElement('option');
        opt.value = dt;
        opt.textContent = dt.slice(0, 10);
        dtSel.appendChild(opt);
      }
      if (preDatetime && dts.includes(preDatetime)) {
        dtSel.value = preDatetime;
      } else if (dts.length === 1) {
        dtSel.value = dts[0];
      }
      // Always update interval start/end to reflect the selected item's date range
      if (dts.length > 0) {
        document.getElementById('inp-start').value = dts[0].slice(0, 10);
        document.getElementById('inp-end').value = dts[dts.length - 1].slice(0, 10);
      }
      updateLayer();
    }

    function buildDatetimeParam() {
      if (dateMode === 'single') {
        return document.getElementById('sel-datetime').value || '';
      }
      const start = document.getElementById('inp-start').value;
      const end = document.getElementById('inp-end').value;
      if (!start && !end) return '';
      return (start || '') + '/' + (end || '');
    }

    function updateLayer() {
      const collId = document.getElementById('sel-collection').value;
      if (!collId) { setStatus(''); return; }
      const itemId = document.getElementById('sel-item').value;
      const dt = buildDatetimeParam();
      const aggVal = dateMode === 'interval' ? document.getElementById('sel-agg').value : null;
      const baselineVal = aggVal === 'anomaly'
        ? (document.getElementById('inp-baseline-start').value || '') + '/' +
          (document.getElementById('inp-baseline-end').value || '')
        : null;

      if (dataLayer) { map.removeLayer(dataLayer); dataLayer = null; }
      dataLayer = L.tileLayer(buildTileUrl(collId, itemId, dt, aggVal, baselineVal), {
        tileSize: 256, opacity: 0.8, errorTileUrl: ''
      }).addTo(map);

      let statusParts = [collId];
      if (itemId) statusParts.push(itemId);
      if (dt) statusParts.push(dt.length > 10 ? dt : dt.slice(0, 10));
      if (aggVal) statusParts.push('agg=' + aggVal);
      setStatus(statusParts.join(' / '));

      // Debounce stats fetch so rapid control changes don't spam the server
      lastStats = null;
      if (statsTimer) clearTimeout(statsTimer);
      if (itemId) {
        document.getElementById('stats-section').style.display = 'block';
        document.getElementById('stats-msg').textContent = 'Computing…';
        document.getElementById('stats-grid').innerHTML = '';
        statsTimer = setTimeout(() => fetchStats(collId, itemId, dt, aggVal, baselineVal || null), 600);
      } else {
        document.getElementById('stats-section').style.display = 'none';
      }
    }

    let statsTimer = null;
    let lastStats = null; // most recent successful stats payload

    function buildTileUrl(collId, itemId, dt, aggVal, baselineVal) {
      const cmap = document.getElementById('sel-colormap').value;
      const rescale = document.getElementById('inp-rescale').value.trim();
      const params = [];
      if (itemId) params.push('item=' + encodeURIComponent(itemId));
      if (dt) params.push('datetime=' + encodeURIComponent(dt));
      if (aggVal) params.push('agg=' + encodeURIComponent(aggVal));
      if (baselineVal) params.push('baseline=' + encodeURIComponent(baselineVal));
      if (cmap) params.push('colormap_name=' + encodeURIComponent(cmap));
      if (rescale) params.push('rescale=' + encodeURIComponent(rescale));
      return '/collections/' + collId + '/tiles/WebMercatorQuad/{z}/{x}/{y}'
        + (params.length ? '?' + params.join('&') : '');
    }

    function refreshTile(collId, itemId, dt, aggVal, baselineVal) {
      // Rebuild the tile layer only — no stats re-fetch, no status update.
      if (dataLayer) { map.removeLayer(dataLayer); dataLayer = null; }
      dataLayer = L.tileLayer(buildTileUrl(collId, itemId, dt, aggVal, baselineVal), {
        tileSize: 256, opacity: 0.8, errorTileUrl: ''
      }).addTo(map);
    }

    function fmt(v) {
      if (v === null || v === undefined) return '—';
      const n = Number(v);
      if (!isFinite(n)) return '—';
      return Math.abs(n) >= 1000 ? n.toLocaleString(undefined, {maximumFractionDigits: 0})
           : Math.abs(n) >= 1    ? n.toLocaleString(undefined, {maximumFractionDigits: 2})
           : n.toLocaleString(undefined, {maximumFractionDigits: 4});
    }

    async function fetchStats(collId, itemId, dt, aggParam, baselineParam) {
      const items = catalog[collId] || [];
      const item = items.find(i => i.item_id === itemId);
      if (!item || !item.zonalstats_body) return;

      const statsSection = document.getElementById('stats-section');
      const statsMsg = document.getElementById('stats-msg');
      const statsGrid = document.getElementById('stats-grid');
      statsSection.style.display = 'block';
      statsMsg.textContent = 'Computing…';
      statsGrid.innerHTML = '';

      const body = {
        inputs: {
          group: item.zarr_group,
          variable: item.variable,
          features: item.zonalstats_body.inputs.features,
        }
      };
      if (dt) body.inputs.datetime = dt;
      if (aggParam) body.inputs.agg = aggParam;
      if (baselineParam) body.inputs.baseline = baselineParam;

      let stats;
      try {
        const resp = await fetch('/processes/zonalstats/execution', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(body),
        });
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          statsMsg.textContent = 'Error: ' + (err.detail || resp.statusText);
          return;
        }
        const data = await resp.json();
        stats = data.features?.[0]?.statistics;
      } catch (e) {
        statsMsg.textContent = 'Request failed.';
        return;
      }

      if (!stats || stats.error) {
        statsMsg.textContent = stats?.error || 'No statistics returned.';
        return;
      }

      statsMsg.textContent = stats.count.toLocaleString() + ' valid pixels';
      const rows = [
        ['mean', stats.mean], ['min', stats.min],
        ['max', stats.max],   ['std', stats.std],
        ['p5',  stats.percentiles?.p5],  ['p95', stats.percentiles?.p95],
        ['p25', stats.percentiles?.p25], ['p75', stats.percentiles?.p75],
        ['sum', stats.sum],   ['nodata', stats.nodata_count],
      ];
      statsGrid.innerHTML = rows.map(([label, val]) =>
        `<div class="stat-item"><span class="stat-label">${label}</span><span class="stat-value">${fmt(val)}</span></div>`
      ).join('');

      lastStats = { stats, collId, itemId, dt, aggParam, baselineParam };

      // Auto-scale: update rescale input and refresh tile without re-fetching stats
      if (document.getElementById('chk-autoscale').checked) {
        applyAutoScale(stats, collId, itemId, dt, aggParam, baselineParam);
      }
    }

    function _autoScaleRange(stats) {
      // Prefer p5/p95, fall back to p25/p75, then min/max
      const p5  = stats.percentiles?.p5;
      const p95 = stats.percentiles?.p95;
      const p25 = stats.percentiles?.p25;
      const p75 = stats.percentiles?.p75;
      if (isFinite(p5) && isFinite(p95))  return [p5,  p95];
      if (isFinite(p25) && isFinite(p75)) return [p25, p75];
      if (isFinite(stats.min) && isFinite(stats.max)) return [stats.min, stats.max];
      return null;
    }

    function applyAutoScale(stats, collId, itemId, dt, aggParam, baselineParam) {
      const range = _autoScaleRange(stats);
      if (!range) return;
      const [lo, hi] = range.map(v => parseFloat(v.toFixed(4)));
      const inp = document.getElementById('inp-rescale');
      inp.value = lo + ',' + hi;
      inp.disabled = true;
      refreshTile(collId, itemId, dt, aggParam, baselineParam);
    }

    async function loadCatalog() {
      setStatus('Loading catalog…');
      let data;
      try {
        const resp = await fetch('/examples');
        data = await resp.json();
      } catch (e) {
        setStatus('Failed to load catalog.');
        return;
      }
      const collSel = document.getElementById('sel-collection');
      collSel.innerHTML = '';
      if (!data.items || data.items.length === 0) {
        collSel.innerHTML = '<option value="">No data available</option>';
        const w = document.getElementById('warning');
        w.style.display = 'block';
        w.textContent = data.warning || 'No data ingested yet.';
        setStatus('');
        return;
      }
      for (const it of data.items) {
        if (!catalog[it.collection_id]) {
          catalog[it.collection_id] = [];
          const opt = document.createElement('option');
          opt.value = it.collection_id;
          opt.textContent = it.collection_id;
          collSel.appendChild(opt);
        }
        catalog[it.collection_id].push(it);
      }
      collSel.addEventListener('change', () => onCollectionChange());
      document.getElementById('sel-item').addEventListener('change', () => onItemChange());
      document.getElementById('sel-datetime').addEventListener('change', updateLayer);
      document.getElementById('inp-start').addEventListener('change', updateLayer);
      document.getElementById('inp-end').addEventListener('change', updateLayer);
      document.getElementById('sel-agg').addEventListener('change', onAggChange);
      document.getElementById('inp-baseline-start').addEventListener('change', updateLayer);
      document.getElementById('inp-baseline-end').addEventListener('change', updateLayer);
      document.getElementById('sel-colormap').addEventListener('change', updateLayer);
      document.getElementById('inp-rescale').addEventListener('input', () => {
        // Editing the rescale field manually → untick auto-scale
        const chk = document.getElementById('chk-autoscale');
        if (chk.checked) {
          chk.checked = false;
          document.getElementById('inp-rescale').disabled = false;
        }
      });
      document.getElementById('inp-rescale').addEventListener('change', updateLayer);
      document.getElementById('chk-autoscale').addEventListener('change', () => {
        const chk = document.getElementById('chk-autoscale');
        document.getElementById('inp-rescale').disabled = chk.checked;
        if (chk.checked && lastStats) {
          const { stats, collId, itemId, dt, aggParam, baselineParam } = lastStats;
          applyAutoScale(stats, collId, itemId, dt, aggParam, baselineParam);
        }
      });

      // Apply pre-selected values from query params
      if (PRESELECT.collection && catalog[PRESELECT.collection]) {
        collSel.value = PRESELECT.collection;
      }
      if (PRESELECT.colormap_name) document.getElementById('sel-colormap').value = PRESELECT.colormap_name;
      if (PRESELECT.rescale) document.getElementById('inp-rescale').value = PRESELECT.rescale;

      // Detect interval vs single-date pre-selection
      let preItem = PRESELECT.item || null;
      let preDatetime = PRESELECT.datetime || null;
      if (PRESELECT.datetime && PRESELECT.datetime.includes('/')) {
        const parts = PRESELECT.datetime.split('/');
        setDateMode('interval');
        document.getElementById('inp-start').value = parts[0] || '';
        document.getElementById('inp-end').value = parts[1] || '';
        if (PRESELECT.agg) {
          document.getElementById('sel-agg').value = PRESELECT.agg;
          onAggChange();
        }
        if (PRESELECT.baseline && PRESELECT.baseline.includes('/')) {
          const bp = PRESELECT.baseline.split('/');
          document.getElementById('inp-baseline-start').value = bp[0] || '';
          document.getElementById('inp-baseline-end').value = bp[1] || '';
        }
        preDatetime = null; // handled above
      }
      onCollectionChange(preItem, preDatetime);
    }

    loadCatalog();

    // ── Tab switching ────────────────────────────────────────────────────────

    function showTab(tab) {
      ['map', 'ingest', 'config'].forEach(function(t) {
        document.getElementById('tab-' + t + '-content').style.display = t === tab ? '' : 'none';
        document.getElementById('tab-btn-' + t).classList.toggle('active', t === tab);
      });
      if (tab === 'ingest') loadJobs();
    }

    // ── Ingest tab ───────────────────────────────────────────────────────────

    function onIngSourceChange() {
      const src = document.getElementById('ing-source').value;
      document.getElementById('ing-iso3-row').style.display    = src === 'worldpop' ? '' : 'none';
      document.getElementById('ing-var-row').style.display     = src === 'cds'      ? '' : 'none';
      document.getElementById('ing-months-row').style.display  = src !== 'worldpop' ? '' : 'none';
    }

    function _parseInts(str) {
      return str.split(',').map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n));
    }

    async function startIngest() {
      const src    = document.getElementById('ing-source').value;
      const years  = document.getElementById('ing-years').value.trim();
      const months = document.getElementById('ing-months').value.trim();
      const iso3   = document.getElementById('ing-iso3').value.trim().toUpperCase();
      const varbl  = document.getElementById('ing-variable').value;

      const inputs = { source: src };
      if (src === 'worldpop') {
        if (!iso3 || iso3.length !== 3) {
          setIngMsg('ISO3 must be 3 characters.', 'err'); return;
        }
        inputs.iso3 = iso3;
      }
      if (src === 'cds') inputs.variable = varbl;
      if (years)  inputs.years  = _parseInts(years);
      if (months) inputs.months = months.toUpperCase() === 'ALL' ? 'ALL' : _parseInts(months);

      const btn = document.getElementById('btn-ingest');
      btn.disabled = true;
      setIngMsg('Submitting…', 'dim');
      try {
        const resp = await fetch('/processes/ingest/execution', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ inputs }),
        });
        const data = await resp.json();
        if (!resp.ok) {
          setIngMsg('Error: ' + (data.detail || resp.statusText), 'err');
        } else {
          setIngMsg('Job started: ' + data.jobID.slice(0, 8) + '…', 'ok');
          loadJobs();
          startJobPoller();
        }
      } catch (e) {
        setIngMsg('Request failed.', 'err');
      } finally {
        btn.disabled = false;
      }
    }

    function setIngMsg(msg, cls) {
      const el = document.getElementById('ing-msg');
      el.textContent = msg;
      el.className = 'ing-msg ' + (cls || 'dim');
    }

    // ── Jobs list ────────────────────────────────────────────────────────────

    let _jobPoller = null;

    function startJobPoller() {
      if (_jobPoller) return; // already polling
      _jobPoller = setInterval(async () => {
        await loadJobs();
        // Stop polling when no jobs are running
        const cards = document.querySelectorAll('.badge-running');
        if (cards.length === 0) { clearInterval(_jobPoller); _jobPoller = null; }
      }, 3000);
    }

    async function loadJobs() {
      let data;
      try {
        const resp = await fetch('/processes/jobs');
        data = await resp.json();
      } catch (e) { return; }

      const jobs = (data.jobs || []).slice().reverse(); // newest first
      const el   = document.getElementById('jobs-list');
      if (!jobs.length) {
        el.innerHTML = '<span style="font-size:12px;color:#9ca3af">No jobs yet.</span>';
        return;
      }

      el.innerHTML = jobs.map(j => {
        const badgeCls = j.status === 'running'   ? 'badge-running'
                       : j.status === 'successful' ? 'badge-succeeded'
                       :                            'badge-failed';
        const params = Object.entries(j.params || {})
          .map(([k, v]) => k + '=' + (Array.isArray(v) ? v.join(',') : v))
          .join(' ');
        const created = j.created ? new Date(j.created).toLocaleTimeString() : '';
        const errHtml = j.error
          ? '<div class="job-error">' + j.error.split('\\n').slice(-3).join('\\n') + '</div>'
          : '';
        return (
          '<div class="job-card">' +
            '<div class="job-card-header">' +
              '<span class="job-source">' + j.source + '</span>' +
              '<span class="badge ' + badgeCls + '">' + j.status + '</span>' +
            '</div>' +
            '<div class="job-id">' + j.jobID + '</div>' +
            (params ? '<div class="job-time">' + params + '</div>' : '') +
            '<div class="job-time">' + created + '</div>' +
            errHtml +
          '</div>'
        );
      }).join('');

      // Auto-start poller if any job is still running
      if (jobs.some(j => j.status === 'running')) startJobPoller();
    }

    // ── Rebuild catalog ──────────────────────────────────────────────────────

    async function rebuildCatalog() {
      const btn = document.getElementById('btn-rebuild');
      const msg = document.getElementById('rebuild-msg');
      btn.disabled = true;
      msg.textContent = 'Rebuilding…';
      msg.className = 'ing-msg dim';
      try {
        const resp = await fetch('/processes/rebuild-catalog/execution', { method: 'POST' });
        const data = await resp.json();
        if (!resp.ok) {
          msg.textContent = 'Error: ' + (data.detail || resp.statusText);
          msg.className = 'ing-msg err';
        } else {
          const n = data.total_timestamps || 0;
          const g = Object.keys(data.groups || {}).length;
          msg.textContent = 'Done — ' + g + ' groups, ' + n + ' timestamps.';
          msg.className = 'ing-msg ok';
        }
      } catch (e) {
        msg.textContent = 'Request failed.';
        msg.className = 'ing-msg err';
      } finally {
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>"""


@app.get(
    "/map",
    response_class=HTMLResponse,
    tags=["OGC Common"],
    summary="Interactive catalog viewer",
    include_in_schema=True,
)
def map_viewer(
    collection: str | None = None,
    item: str | None = None,
    datetime: str | None = None,
    agg: str | None = None,
    baseline: str | None = None,
    colormap_name: str | None = None,
    rescale: str | None = None,
) -> HTMLResponse:
    """
    Catalog-aware map viewer.

    Loads available collections, items, and datetimes from the catalog
    and renders tiles via the OGC Tiles API — no zarr paths needed.
    Optional query parameters pre-select a specific collection/item/datetime/agg.
    Pass ``datetime`` as an ISO 8601 interval (``start/end``) to activate interval mode.
    """
    import json

    preselect = json.dumps(
        {
            "collection": collection or "",
            "item": item or "",
            "datetime": datetime or "",
            "agg": agg or "",
            "baseline": baseline or "",
            "colormap_name": colormap_name or "",
            "rescale": rescale or "",
        }
    )
    config_data = json.dumps(
        {
            "bbox": list(settings.bbox),
            "quota_mb": settings.store_quota_mb,
            "zarr_root": str(settings.zarr_root),
            "raw_dir": str(settings.raw_dir),
            "catalog_path": str(settings.catalog_path),
        }
    )
    html = _MAP_HTML.replace("__PRESELECT__", preselect).replace("__CONFIG__", config_data)
    return HTMLResponse(content=html)


# ── Dynamic OpenAPI schema — inject real catalog examples ─────────────────────


_COLORMAP_EXAMPLES = {
    "viridis": {"value": "viridis", "summary": "viridis — sequential, perceptually uniform"},
    "plasma": {"value": "plasma", "summary": "plasma — sequential, high contrast"},
    "inferno": {"value": "inferno", "summary": "inferno — sequential, dark-to-bright"},
    "magma": {"value": "magma", "summary": "magma — sequential, dark-to-light"},
    "coolwarm": {"value": "coolwarm", "summary": "coolwarm — diverging, blue–red"},
}

_RESCALE_EXAMPLES = {
    "0,1000": {"value": "0,1000", "summary": "0–1000 (e.g. population density, mm precipitation)"},
    "0,100": {"value": "0,100", "summary": "0–100 (e.g. percentage, index)"},
    "0,10000": {"value": "0,10000", "summary": "0–10 000 (e.g. high population count)"},
    "-3,3": {"value": "-3,3", "summary": "-3 to 3 (anomaly in std-dev units)"},
    "-50,50": {"value": "-50,50", "summary": "-50 to 50 (anomaly, absolute units)"},
    "250,320": {"value": "250,320", "summary": "250–320 K (temperature in Kelvin)"},
    "-30,45": {"value": "-30,45", "summary": "-30 to 45 °C (temperature in Celsius)"},
}


def _catalog_openapi_examples() -> dict[str, dict[str, dict]]:
    """
    Read the catalog and return per-parameter example dicts ready for OpenAPI injection.
    Returns empty dicts for catalog-derived params if catalog is missing or empty.
    """
    examples: dict[str, dict] = {
        "collection_id": {},
        "item": {},
        "datetime": {},
        "colormap_name": _COLORMAP_EXAMPLES,
        "rescale": _RESCALE_EXAMPLES,
    }
    try:
        catalogue = load_or_create(settings.catalog_path)
        for coll in catalogue.get_children():
            if not isinstance(coll, pystac.Collection):
                continue
            coll_items = list(coll.get_items())
            if not coll_items:
                continue
            examples["collection_id"][coll.id] = {
                "value": coll.id,
                "summary": coll.title or coll.id,
            }
            for item in coll_items:
                examples["item"][item.id] = {
                    "value": item.id,
                    "summary": f"{item.id} ({coll.id})",
                }
                # Use eostrata:datetimes first, fall back to start/end interval bounds
                datetimes: list[str] = item.properties.get("eostrata:datetimes", [])
                if not datetimes:
                    for key in ("start_datetime", "end_datetime", "datetime"):
                        val = item.properties.get(key)
                        if val:
                            datetimes.append(val)
                for dt in datetimes:
                    examples["datetime"][dt] = {"value": dt, "summary": dt[:10]}
    except Exception:  # noqa: BLE001
        pass
    return examples


def _dynamic_openapi() -> dict:
    """Build the OpenAPI schema and inject live catalog examples into tile parameters."""
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=_OPENAPI_TAGS,
    )

    param_examples = _catalog_openapi_examples()
    has_data = bool(param_examples["collection_id"])

    tms_example = {
        "WebMercatorQuad": {"value": "WebMercatorQuad", "summary": "Web Mercator (standard)"}
    }

    for path, path_item in schema.get("paths", {}).items():
        if "{collection_id}" not in path:
            continue
        for operation in path_item.values():
            if not isinstance(operation, dict):
                continue
            for param in operation.get("parameters", []):
                name = param.get("name")
                if name == "tileMatrixSetId":
                    param["examples"] = tms_example
                elif (
                    name in ("colormap_name", "rescale")
                    or has_data
                    and name in param_examples
                    and param_examples[name]
                ):
                    param["examples"] = param_examples[name]

    return schema


app.openapi = _dynamic_openapi  # type: ignore[method-assign]
