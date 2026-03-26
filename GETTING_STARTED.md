# Getting started with eostrata

This guide walks through downloading earth observation data, storing it as Zarr, and serving it as map tiles, a STAC catalogue, and zonal statistics.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Downloading data](#downloading-data)
  - [WorldPop — population](#worldpop--population)
  - [CHIRPS — precipitation](#chirps--precipitation)
  - [CDS / ERA5 — climate reanalysis](#cds--era5--climate-reanalysis)
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

Copy `.env.example` to `.env` and edit it, or set environment variables directly. The full set of options:

```bash
# Storage paths
EOSTRATA_ZARR_ROOT=data/zarr
EOSTRATA_RAW_DIR=data/raw
EOSTRATA_CATALOG_PATH=data/catalog.json

# Store quota — maximum size of the Zarr store in MB. 0 = unlimited (default).
# When a download would exceed this limit, the oldest groups are evicted first.
# EOSTRATA_STORE_QUOTA_MB=0

# Bounding box (west, south, east, north) in EPSG:4326
# Example below covers Nigeria
EOSTRATA_BBOX_WEST=2.0
EOSTRATA_BBOX_SOUTH=4.0
EOSTRATA_BBOX_EAST=15.0
EOSTRATA_BBOX_NORTH=14.0
```

> **Tip** — keep the bounding box as small as your use case allows. Clipping to your area of interest significantly reduces storage and processing time.

---

## Downloading data

Every `download` command follows the same pattern:

1. Fetch the raw file (GeoTIFF, NetCDF, or gzip) from the upstream source
2. Clip to your configured bounding box
3. Write a CF-compliant Zarr group — multiple downloads append along the `time` dimension
4. Register a STAC item in `data/catalog.json`

Raw files are cached in `data/raw/` and reused on subsequent runs. Check what is in the store at any time:

```bash
uv run eostrata list
```

This shows each Zarr group with its size, total store usage, and quota if one is configured:

```
Zarr store: data/zarr  [142.3 MB / 10000 MB (1%)]
┌───────────────┬──────────┐
│ Group         │ Size     │
├───────────────┼──────────┤
│ chirps/global │  89.4 MB │
│ worldpop/nga  │  52.9 MB │
└───────────────┴──────────┘
```

---

### WorldPop — population

Annual population count rasters from [WorldPop R2025A](https://www.worldpop.org). One Zarr group per country, all years as timesteps.

**Download the latest available year:**
```bash
uv run eostrata download worldpop NGA
```

**Download a specific year:**
```bash
uv run eostrata download worldpop NGA --year 2020
```

**Build a multi-year time series** in a single call:
```bash
uv run eostrata download worldpop NGA --years 2020,2021,2022
```

**Download for multiple countries:**
```bash
uv run eostrata download worldpop NGA
uv run eostrata download worldpop KEN
uv run eostrata download worldpop ETH
```

Each country gets its own Zarr group (`worldpop/nga`, `worldpop/ken`, …) and STAC item.

**What gets written:**

| | Value |
|---|---|
| Zarr group | `data/zarr/worldpop/<iso3_lower>/` |
| Variable | `population` |
| STAC item id | `worldpop_<iso3_lower>` |
| Time resolution | annual (one timestep per year) |

---

### CHIRPS — precipitation

Monthly precipitation totals from [CHIRPS v2.0](https://www.chc.ucsb.edu/data/chirps) (Climate Hazards Group InfraRed Precipitation with Station data). A single global Zarr group covers all months as timesteps.

**Download the latest available month** (typically ~45 days behind real time):
```bash
uv run eostrata download chirps
```

**Download a specific month:**
```bash
uv run eostrata download chirps --year 2023 --month 6
```

**Download multiple months at once:**
```bash
uv run eostrata download chirps --year 2023 --months 1,2,3
```

**Download multiple years and months in a single call:**
```bash
uv run eostrata download chirps --years 2022,2023 --months 1,2,3,4,5,6,7,8,9,10,11,12
```

**What gets written:**

| | Value |
|---|---|
| Zarr group | `data/zarr/chirps/global/` |
| Variable | `precipitation` |
| STAC item id | `chirps_global` |
| Time resolution | monthly (one timestep per month) |

---

### CDS / ERA5 — climate reanalysis

Monthly-averaged ERA5 reanalysis from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu). One Zarr group per variable, all years and months as timesteps.

**Setup required** — ERA5 needs a CDS account and API credentials before you can download:

1. Register at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu)
2. Install the optional dependency: `uv sync --extra cds`
3. Create `~/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api
key: <your-api-key>
```

**Download 2m air temperature for the latest available year** (~3 months behind real time):
```bash
uv run eostrata download cds --variable t2m
```

**Download a specific year:**
```bash
uv run eostrata download cds --variable t2m --year 2023
```

**Download specific months only:**
```bash
uv run eostrata download cds --variable t2m --year 2023 --months 1,2,3
```

**Supported variables:**

| Flag | ERA5 variable | Unit |
|---|---|---|
| `t2m` | 2m air temperature | K |
| `tp` | Total precipitation | m |
| `u10` | 10m U-component of wind | m/s |
| `v10` | 10m V-component of wind | m/s |
| `sp` | Surface pressure | Pa |

**Download multiple years at once:**
```bash
uv run eostrata download cds --variable t2m --years 2021,2022,2023
```

**Download multiple variables** (run once per variable):
```bash
uv run eostrata download cds --variable t2m --year 2023
uv run eostrata download cds --variable tp --year 2023
uv run eostrata download cds --variable u10 --year 2023
```

**What gets written** (example for `t2m`):

| | Value |
|---|---|
| Zarr group | `data/zarr/era5/t2m/` |
| Variable | `t2m` |
| STAC item id | `era5_t2m` |
| Time resolution | monthly (one timestep per month) |

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
  download chirps     Download a CHIRPS monthly precipitation raster
  download cds        Download ERA5 monthly reanalysis from Copernicus CDS
  download worldpop   Download a WorldPop population raster
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
  --year INTEGER          Single year (default: latest available)
  --years TEXT            Multiple years, comma-separated: 2020,2021,2022
  --zarr-root PATH        Override Zarr store root
  --raw-dir PATH          Override raw download directory
  --catalog-path PATH     Override catalog.json path
  -v, --verbose           Enable debug logging
```

**download chirps**
```
uv run eostrata download chirps [OPTIONS]

Options:
  --year INTEGER          Single year (default: latest available)
  --years TEXT            Multiple years, comma-separated: 2022,2023
  --month INTEGER         Single month 1-12 (default: latest available)
  --months TEXT           Multiple months, comma-separated: 1,2,3
  --zarr-root PATH        Override Zarr store root
  --raw-dir PATH          Override raw download directory
  --catalog-path PATH     Override catalog.json path
  -v, --verbose           Enable debug logging
```

**download cds**
```
uv run eostrata download cds [OPTIONS]

Options:
  --variable TEXT         ERA5 variable short name: t2m, tp, u10, v10, sp (default: t2m)
  --year INTEGER          Single year (default: latest available)
  --years TEXT            Multiple years, comma-separated: 2022,2023
  --months TEXT           Months to fetch, comma-separated: 1,2,3 (default: all 12)
  --zarr-root PATH        Override Zarr store root
  --raw-dir PATH          Override raw download directory
  --catalog-path PATH     Override catalog.json path
  -v, --verbose           Enable debug logging
```

**list**
```
uv run eostrata list [OPTIONS]

Options:
  --zarr-root PATH        Zarr store root
  --catalog-path PATH     Path to catalog.json
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
  --zarr-root PATH    Zarr store root
  --raw-dir PATH      Raw downloads directory
  --catalog-path PATH Path to catalog.json
  --yes, -y           Skip confirmation prompt
```
