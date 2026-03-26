# Getting started with eostrata

> **⚠️ Early development — only WorldPop ingestion and basic serving are implemented.**

This guide walks through downloading your first WorldPop dataset, storing it as Zarr, and serving it as map tiles, a STAC catalogue, and zonal statistics.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Downloading WorldPop data](#downloading-worldpop-data)
- [Starting the server](#starting-the-server)
- [Exploring the map viewer](#exploring-the-map-viewer)
- [STAC catalogue walkthrough](#stac-catalogue-walkthrough)
- [Zonal statistics example](#zonal-statistics-example)
- [CLI reference](#cli-reference)

---

## Prerequisites

- **Python 3.11+** — check with `python --version`
- **uv** — fast Python package manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- **git**

---

## Installation

```bash
git clone https://github.com/your-org/eostrata.git
cd eostrata
uv sync
```

Verify the CLI is available:

```bash
uv run eostrata --help
```

---

## Configuration

All settings are read from environment variables prefixed `EOSTRATA_`. The most important ones to set before downloading anything are the **bounding box** (your area of interest) and the **storage paths**.

Create a `.env` file at the project root:

```bash
# Storage
EOSTRATA_ZARR_ROOT=data/zarr
EOSTRATA_RAW_DIR=data/raw
EOSTRATA_CATALOG_PATH=data/catalog.json

# Bounding box (west, south, east, north) in EPSG:4326
# Example below covers Nigeria
EOSTRATA_BBOX_WEST=2.0
EOSTRATA_BBOX_SOUTH=4.0
EOSTRATA_BBOX_EAST=15.0
EOSTRATA_BBOX_NORTH=14.0
```

> **Tip** — keep the bounding box as small as your use case allows. Clipping to your area of interest significantly reduces storage and processing time.

---

## Downloading WorldPop data

Download a population raster for a country and year. The dataset is clipped to your configured bounding box, converted to a CF-compliant Zarr collection, and registered in the STAC catalogue automatically.

```bash
uv run eostrata download worldpop NGA --year 2020
```

The command will:
1. Resolve the WorldPop R2025A download URL for Nigeria 2020
2. Download the GeoTIFF to `data/raw/worldpop/`
3. Clip it to your bbox
4. Write a Zarr group to `data/zarr/worldpop/nga` with variable `population`
5. Register a STAC item `worldpop_nga` in `data/catalog.json`

Multiple years are appended along the `time` dimension of the same Zarr group - one group per country, all years as timesteps:

```bash
uv run eostrata download worldpop NGA --year 2021
uv run eostrata download worldpop NGA --year 2022
```

Check what is in the store at any time:

```bash
uv run eostrata list
```

---

## Starting the server

```bash
uv run eostrata serve
```

The server starts on `http://127.0.0.1:8000` and exposes:

| Interface | URL |
|---|---|
| Interactive API docs | `http://127.0.0.1:8000/docs` |
| OGC collections | `http://127.0.0.1:8000/collections` |
| STAC catalogue | `http://127.0.0.1:8000/stac` |
| OGC Processes | `http://127.0.0.1:8000/processes` |

Add `--reload` for hot-reloading during development:

```bash
uv run eostrata serve --reload
```

---

## Exploring the map viewer

The map viewer is available per collection via the OGC Tiles endpoint:

```
http://127.0.0.1:8000/collections/worldpop/tiles/WebMercatorQuad/map.html?item=worldpop_nga&datetime=2020-01-01&rescale=0,1000&colormap_name=viridis
```

**Key query parameters:**

| Parameter | Description | Example |
|---|---|---|
| `item` | STAC item id within the collection | `worldpop_nga` |
| `datetime` | ISO 8601 datetime or interval for time selection | `2021-01-01` or `2021-01-01/2022-12-31` |
| `agg` | Temporal aggregation method | `mean`, `sum`, `min`, `max`, `anomaly` |
| `rescale` | Value range mapped to 0–255 | `0,1000` |
| `colormap_name` | Matplotlib colormap name | `viridis`, `plasma`, `reds`, `ylorbr` |

---

## STAC catalogue walkthrough

The STAC catalogue is available at `/stac`. It follows the [STAC API spec](https://api.stacspec.org) and can be queried with any STAC-compatible client.

**List collections:**
```bash
curl http://127.0.0.1:8000/stac/collections | python -m json.tool
```

**List items in a collection:**
```bash
curl http://127.0.0.1:8000/stac/collections/worldpop/items | python -m json.tool
```

Each item represents one country. The `datetime` interval spans all ingested years and expands automatically as you download more. The `zarr` asset contains the open kwargs needed to load the data directly with xarray:

```json
{
  "id": "worldpop_nga",
  "properties": {
    "start_datetime": "2020-01-01T00:00:00+00:00",
    "end_datetime": "2022-01-01T00:00:00+00:00",
    "eostrata:variable": "population",
    "eostrata:zarr_group": "worldpop/nga"
  },
  "assets": {
    "zarr": {
      "href": "data/zarr/worldpop/nga",
      "type": "application/vnd+zarr",
      "xarray:open_kwargs": {
        "engine": "zarr",
        "group": "worldpop/nga",
        "consolidated": true
      },
      "xarray:variable": "population"
    }
  }
}
```

---

## Zonal statistics example

The `zonalstats` process computes per-feature summary statistics over any collection. Send a GeoJSON `FeatureCollection` as the input zones.

```bash
curl -s -X POST http://127.0.0.1:8000/processes/zonalstats/execution \
  -H "Content-Type: application/json" \
  -d '
  {
    "inputs": {
      "group": "worldpop/nga",
      "variable": "population",
      "features":{
        "type": "FeatureCollection",
        "features": [
          {
            "type":"Feature",
            "properties": {
              "name":"Lagos area"
            },
            "geometry": {
              "type":"Polygon",
              "coordinates": [
                [[3.0,6.0],[4.5,6.0],[4.5,7.0],[3.0,7.0],[3.0,6.0]]
              ]
            }
          }
        ]
      }
    }
  }' \
  | python -m json.tool
```

The response is a GeoJSON `FeatureCollection` with original properties preserved and a `statistics` object added to each feature:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "name": "Lagos area"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [[3.0, 6.0], [4.5, 6.0], [4.5, 7.0], [3.0, 7.0], [3.0, 6.0]]
        ]
      },
      "statistics": {
        "count": 10186,
        "nodata_count": 11414,
        "min": 9.722341848598562e-22,
        "max": 21217.90234375,
        "mean": 1884.0143487505113,
        "std": 3865.856879172618,
        "sum": 19190570.156372707,
        "percentiles": {
          "p5": 0.9055151790380478,
          "p25": 15.221198558807373,
          "p50": 131.11722564697266,
          "p75": 1391.2993774414062,
          "p95": 12006.97900390625
        }
      }
    }
  ]
}
```

You can pass any number of features in the `FeatureCollection`, statistics are computed per feature.

---

## CLI reference

```
eostrata --help

Commands:
  download worldpop   Download a WorldPop raster, clip to bbox, write to Zarr
  list                List datasets in the Zarr store and STAC catalogue
  serve               Start the tile server, STAC catalogue and OGC Processes API
  cleanup             Delete the store, raw downloads and catalogue (dev only)
```

**download worldpop**
```
uv run eostrata download worldpop [ISO3] [OPTIONS]

Arguments:
  ISO3                    ISO 3166-1 alpha-3 country code, e.g. NGA

Options:
  --year INTEGER          Reference year (default: latest available)
  --zarr-root PATH        Override Zarr store root
  --raw-dir PATH          Override raw download directory
  --catalog-path PATH     Override catalog.json path
  -v, --verbose           Enable debug logging
```

**serve**
```
uv run eostrata serve [OPTIONS]

Options:
  --host TEXT     Bind host (default: 127.0.0.1)
  --port INTEGER  Bind port (default: 8000)
  --reload        Enable hot-reload for development
```

**cleanup**
```
uv run eostrata cleanup [OPTIONS]

Options:
  --yes, -y   Skip confirmation prompt
```
