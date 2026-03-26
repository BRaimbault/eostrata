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
GET  /docs                                                  OpenAPI docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import pystac
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse
from stac_fastapi.api.app import StacApi
from stac_fastapi.types.config import ApiSettings
from titiler.core.errors import DEFAULT_STATUS_CODES, add_exception_handlers
from titiler.xarray.extensions import VariablesExtension
from titiler.xarray.factory import TilerFactory

from eostrata.aggregate import AggregatingReader
from eostrata.catalog import PystacClient, load_or_create
from eostrata.config import settings
from eostrata.ogc.processes import router as processes_router
from eostrata.ogc.tiles import router as collection_tiles_router

logger = logging.getLogger(__name__)

# ── Lifespan: start / stop the background scheduler ───────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
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

add_exception_handlers(app, DEFAULT_STATUS_CODES)

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
    return {"collections": result, "links": []}


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
    body { margin: 0; font-family: sans-serif; }
    #map { position: absolute; top: 0; bottom: 0; width: 100%; }
    #controls {
      position: absolute; top: 10px; left: 10px; z-index: 1000;
      background: white; padding: 14px 16px; border-radius: 8px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.25); min-width: 280px;
    }
    #controls h3 { margin: 0 0 10px; font-size: 14px; }
    #controls label {
      display: block; margin-top: 8px; font-size: 11px; color: #555;
      text-transform: uppercase; letter-spacing: 0.04em;
    }
    #controls select, #controls input {
      width: 100%; margin-top: 3px; padding: 5px 6px;
      font-size: 13px; box-sizing: border-box;
      border: 1px solid #ccc; border-radius: 4px;
    }
    #status { font-size: 11px; color: #888; margin-top: 10px; min-height: 14px; }
    #warning { font-size: 12px; color: #b45309; margin-top: 8px; display: none; }
  </style>
</head>
<body>
  <div id="map"></div>
  <div id="controls">
    <h3>eostrata viewer</h3>
    <label>Collection</label>
    <select id="sel-collection"><option value="">Loading…</option></select>
    <label>Item</label>
    <select id="sel-item"><option value="">—</option></select>
    <label>Datetime</label>
    <select id="sel-datetime"><option value="">latest</option></select>
    <label>Colormap</label>
    <select id="sel-colormap">
      <option value="">default</option>
      <option value="viridis">viridis</option>
      <option value="plasma">plasma</option>
      <option value="inferno">inferno</option>
      <option value="magma">magma</option>
      <option value="YlOrRd">YlOrRd</option>
      <option value="YlGnBu">YlGnBu</option>
      <option value="RdYlGn">RdYlGn</option>
      <option value="Blues">Blues</option>
      <option value="Greens">Greens</option>
    </select>
    <label>Rescale (min,max)</label>
    <input id="inp-rescale" type="text" placeholder="e.g. 0,1000"/>
    <div id="warning"></div>
    <div id="status"></div>
  </div>
  <script>
    const PRESELECT = __PRESELECT__;

    const map = L.map('map').setView([5, 20], 4);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '\u00a9 <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      opacity: 0.5
    }).addTo(map);

    let dataLayer = null;
    const catalog = {};

    function setStatus(msg) { document.getElementById('status').textContent = msg; }

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
      updateLayer();
    }

    function updateLayer() {
      const collId = document.getElementById('sel-collection').value;
      if (!collId) { setStatus(''); return; }
      const itemId = document.getElementById('sel-item').value;
      const dt = document.getElementById('sel-datetime').value;
      const cmap = document.getElementById('sel-colormap').value;
      const rescale = document.getElementById('inp-rescale').value.trim();

      if (dataLayer) { map.removeLayer(dataLayer); dataLayer = null; }

      const params = [];
      if (itemId) params.push('item=' + encodeURIComponent(itemId));
      if (dt) params.push('datetime=' + encodeURIComponent(dt));
      if (cmap) params.push('colormap_name=' + encodeURIComponent(cmap));
      if (rescale) params.push('rescale=' + encodeURIComponent(rescale));
      const qs = params.length ? '?' + params.join('&') : '';
      const tileUrl = '/collections/' + collId + '/tiles/WebMercatorQuad/{z}/{x}/{y}' + qs;

      dataLayer = L.tileLayer(tileUrl, {
        tileSize: 256,
        opacity: 0.8,
        errorTileUrl: ''
      }).addTo(map);

      setStatus(collId + (itemId ? ' / ' + itemId : '') + (dt ? ' @ ' + dt.slice(0, 10) : ''));
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
      document.getElementById('sel-colormap').addEventListener('change', updateLayer);
      document.getElementById('inp-rescale').addEventListener('change', updateLayer);

      if (PRESELECT.collection && catalog[PRESELECT.collection]) {
        collSel.value = PRESELECT.collection;
      }
      if (PRESELECT.colormap_name) document.getElementById('sel-colormap').value = PRESELECT.colormap_name;
      if (PRESELECT.rescale) document.getElementById('inp-rescale').value = PRESELECT.rescale;
      onCollectionChange(PRESELECT.item || null, PRESELECT.datetime || null);
    }

    loadCatalog();
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
    colormap_name: str | None = None,
    rescale: str | None = None,
) -> HTMLResponse:
    """
    Catalog-aware map viewer.

    Loads available collections, items, and datetimes from the catalog
    and renders tiles via the OGC Tiles API — no zarr paths needed.
    Optional query parameters pre-select a specific collection/item/datetime.
    """
    import json

    preselect = json.dumps(
        {
            "collection": collection or "",
            "item": item or "",
            "datetime": datetime or "",
            "colormap_name": colormap_name or "",
            "rescale": rescale or "",
        }
    )
    html = _MAP_HTML.replace("__PRESELECT__", preselect)
    return HTMLResponse(content=html)


# ── Dynamic OpenAPI schema — inject real catalog examples ─────────────────────


_COLORMAP_EXAMPLES = {
    "viridis": {"value": "viridis", "summary": "viridis — sequential, perceptually uniform"},
    "plasma": {"value": "plasma", "summary": "plasma — sequential, high contrast"},
    "inferno": {"value": "inferno", "summary": "inferno — sequential, dark-to-bright"},
    "magma": {"value": "magma", "summary": "magma — sequential, dark-to-light"},
    "YlOrRd": {"value": "YlOrRd", "summary": "YlOrRd — yellow → orange → red"},
    "YlGnBu": {"value": "YlGnBu", "summary": "YlGnBu — yellow → green → blue"},
    "RdYlGn": {"value": "RdYlGn", "summary": "RdYlGn — diverging, red–yellow–green"},
    "RdBu": {"value": "RdBu", "summary": "RdBu — diverging, red–blue (anomalies)"},
    "coolwarm": {"value": "coolwarm", "summary": "coolwarm — diverging, blue–red"},
    "Blues": {"value": "Blues", "summary": "Blues — sequential blue (precipitation)"},
    "Greens": {"value": "Greens", "summary": "Greens — sequential green (vegetation)"},
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
