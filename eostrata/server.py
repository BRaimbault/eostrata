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

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

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
from titiler.xarray.factory import TilerFactory

from eostrata.aggregate import AggregatingReader
from eostrata.catalog import PystacClient, load_or_create
from eostrata.config import settings
from eostrata.constants import PROP_DATETIMES, PROP_VARIABLE, PROP_ZARR_GROUP
from eostrata.ogc.ingest import router as ingest_router
from eostrata.ogc.processes import router as processes_router
from eostrata.ogc.scheduler_router import router as scheduler_router
from eostrata.ogc.tiles import _VariablesExtension
from eostrata.ogc.tiles import router as collection_tiles_router

logger = logging.getLogger(__name__)

# ── Lifespan: start / stop the background scheduler ───────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    from eostrata.log import setup_logging

    setup_logging(rich_console=False)  # file-only; uvicorn owns the console

    # Validate that required storage directories exist and are writable.
    # The isinstance guard lets test suites substitute mock settings without
    # tripping over this check.
    import os

    for label, path in [
        ("zarr_root", settings.zarr_root),
        ("raw_dir", settings.raw_dir),
    ]:
        if not isinstance(path, Path):
            continue
        path.mkdir(parents=True, exist_ok=True)
        if not os.access(path, os.W_OK):
            raise RuntimeError(
                f"Storage directory '{label}' is not writable: {path}. "
                "Check permissions before starting the server."
            )
    logger.info(
        "Storage directories OK — zarr_root=%s raw_dir=%s",
        settings.zarr_root,
        settings.raw_dir,
    )

    scheduler = None
    try:
        from eostrata.scheduler import Scheduler, set_scheduler

        scheduler = Scheduler()
        set_scheduler(scheduler)
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
        from eostrata.scheduler import set_scheduler

        set_scheduler(None)


# ── Main app ──────────────────────────────────────────────────────────────────

_OPENAPI_TAGS = [
    {
        "name": "Discovery",
        "description": (
            "Start here. **/examples** lists every ingested dataset with copy-pasteable parameter "
            "values for all other endpoints. **/collections** gives the OGC collections index."
        ),
    },
    {
        "name": "Map Viewer",
        "description": (
            "Browser-based catalog-aware map. "
            "Open **/map** in a browser, pick a collection and datetime, and tiles render instantly. "
            "Supports temporal aggregation (`agg`, `baseline`) and deep-linking via query parameters."
        ),
    },
    {
        "name": "Tiles",
        "description": (
            "Dynamic raster tiles served directly from the Zarr store via titiler.xarray. "
            "Use `/collections/{collection_id}/tiles/WebMercatorQuad/{z}/{x}/{y}` for PNG tiles. "
            "Call **/examples** first to find valid `collection_id`, `item`, and `datetime` values."
        ),
    },
    {
        "name": "Zonal Statistics",
        "description": (
            "Compute per-polygon raster statistics over any ingested dataset. "
            "Send a GeoJSON FeatureCollection and get back per-feature mean / sum / min / max / std / "
            "percentiles. Temporal aggregation parameters (`datetime`, `agg`, `baseline`) apply before "
            "extraction. Use **/examples** to obtain the `group` and `variable` values."
        ),
    },
    {
        "name": "Data Ingestion",
        "description": (
            "Trigger async ingestion jobs for WorldPop, CHIRPS, and CDS/ERA5. "
            "POST to an execution endpoint to start a job — the response includes a `jobID`. "
            "Poll `GET /processes/jobs/{jobID}` to check status. "
            "When nothing could be ingested (e.g. all periods returned 404) the job is marked **failed**."
        ),
    },
    {
        "name": "Store & Catalog",
        "description": (
            "Monitor Zarr store disk usage and rebuild the STAC catalogue. "
            "**/store-usage** shows per-group sizes and per-timestamp last-access times. "
            "Use **rebuild-catalog** to resync the catalogue after manual edits or partial failures."
        ),
    },
    {
        "name": "OGC",
        "description": (
            "OGC API - Common protocol endpoints: landing page and conformance classes. "
            "Primarily for OGC-native clients — most users can ignore these."
        ),
    },
    {
        "name": "Scheduler",
        "description": (
            "Manage APScheduler cron jobs at runtime. "
            "Open [/scheduler](/scheduler) for the visual dashboard. "
            "Jobs are persisted to ``schedules.yml`` so they survive restarts. "
            "Use ``POST /scheduler/jobs/{job_id}/run`` to trigger a job immediately."
        ),
    },
]

app = FastAPI(
    title="eostrata",
    description=(
        "One tool to fetch, store, aggregate, and serve earth observation layers.\n\n"
        "**Quick start**: call [/examples](/examples) to see what data is currently available "
        "and get copy-pasteable parameter values for every endpoint. "
        "Open [/map](/map) to browse layers interactively in the browser."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=_OPENAPI_TAGS,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
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
    extensions=[_VariablesExtension()],
)
app.include_router(
    _raw_tiler.router, prefix="/tiles", tags=["Tiles (direct)"], include_in_schema=False
)

# ── OGC Processes ─────────────────────────────────────────────────────────────

app.include_router(processes_router)
app.include_router(ingest_router)
app.include_router(scheduler_router)

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


@app.get("/", tags=["OGC"], summary="Landing page")
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
            {"rel": "scheduler", "href": "/scheduler", "type": "text/html"},
        ],
    }


@app.get("/conformance", tags=["OGC"], summary="Conformance classes")
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


@app.get("/collections", tags=["Discovery"], summary="Available collections")
def collections() -> dict:
    """OGC API - Common /collections — lists all ingested collections."""
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
    tags=["Discovery"],
    summary="Ready-to-use parameter values for all endpoints",
)
def examples() -> dict:
    """
    Lists every ingested item with copy-pasteable parameter values for the other endpoints.

    Use the values here to fill in `collection_id`, `item`, `datetime`, `group`, and
    `variable` when testing tiles, tilejson, info, and zonalstats via the Swagger UI.

    Returns a warning when no data has been ingested yet.
    """
    catalogue = load_or_create(settings.catalog_path)
    items_out = []
    for coll in catalogue.get_children():
        if not isinstance(coll, pystac.Collection):
            continue
        for item in coll.get_items():
            datetimes: list[str] = item.properties.get(PROP_DATETIMES, [])
            if not datetimes:
                # Fallback: derive from start/end
                start = item.properties.get("start_datetime") or item.properties.get("datetime")
                if start:
                    datetimes = [start]
            variable = item.properties.get(PROP_VARIABLE, "")
            zarr_group = item.properties.get(PROP_ZARR_GROUP, "")
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


@app.get("/store-usage", tags=["Store & Catalog"], summary="Zarr store disk usage")
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

_MAP_HTML = (Path(__file__).parent / "templates" / "map.html").read_text()
_SCHEDULER_HTML = (Path(__file__).parent / "templates" / "scheduler.html").read_text()


@app.get(
    "/map",
    response_class=HTMLResponse,
    tags=["Map Viewer"],
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
            "eviction_buffer_mb": settings.store_eviction_buffer_mb,
            "zarr_root": str(settings.zarr_root),
            "raw_dir": str(settings.raw_dir),
            "catalog_path": str(settings.catalog_path),
        }
    )
    from eostrata.ogc.ingest import INGEST_SOURCES

    sources_data = json.dumps(INGEST_SOURCES)
    html = (
        _MAP_HTML.replace("__PRESELECT__", preselect)
        .replace("__CONFIG__", config_data)
        .replace("__SOURCES__", sources_data)
    )
    return HTMLResponse(content=html)


# ── Scheduler dashboard ───────────────────────────────────────────────────────


@app.get(
    "/scheduler",
    response_class=HTMLResponse,
    tags=["Scheduler"],
    summary="Scheduler dashboard",
    include_in_schema=True,
)
def scheduler_ui() -> HTMLResponse:
    """
    Visual dashboard for managing APScheduler cron jobs.

    Lists all configured jobs with their next run times, and provides a form
    to add, edit, enable/disable, delete, and manually trigger jobs.
    Jobs are persisted to ``schedules.yml`` so they survive restarts.
    """
    from eostrata.ogc.ingest import INGEST_SOURCES

    sources_data = json.dumps(INGEST_SOURCES)
    html = _SCHEDULER_HTML.replace("__SOURCES__", sources_data)
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
                datetimes: list[str] = item.properties.get(PROP_DATETIMES, [])
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
