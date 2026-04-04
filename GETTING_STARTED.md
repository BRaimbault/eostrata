# Getting started with eostrata

This guide walks through downloading earth observation data, storing it as Zarr, and serving it as map tiles, a STAC catalogue, and zonal statistics.

**Live demo: [eostrata.onrender.com](https://eostrata.onrender.com/)**

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Available data sources](#available-data-sources)
- [Downloading data](#downloading-data)
  - [WorldPop — population](#worldpop--population)
  - [CHIRPS — precipitation](#chirps--precipitation)
  - [CDS / ERA5 — climate reanalysis](#cds--era5--climate-reanalysis)
  - [CAMS — air quality reanalysis](#cams--air-quality-reanalysis)
  - [TROPOMI — Sentinel-5P air quality](#tropomi--sentinel-5p-air-quality)
  - [Sentinel NDVI — vegetation index](#sentinel-ndvi--vegetation-index)
- [Ingesting data via the API](#ingesting-data-via-the-api)
- [Starting the server](#starting-the-server)
- [Exploring the map viewer](#exploring-the-map-viewer)
- [STAC catalogue walkthrough](#stac-catalogue-walkthrough)
- [Zonal statistics example](#zonal-statistics-example)
- [Scheduled ingestion](#scheduled-ingestion)
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
git clone https://github.com/BRaimbault/eostrata.git
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
# When a download would exceed this limit, the least-recently-used timestamps
# are evicted first (oldest first, one at a time across all groups).
# EOSTRATA_STORE_QUOTA_MB=0

# Eviction headroom — MB to keep free inside the quota before each download.
# Recommended: ~10% of quota. Ignored if >= quota.
# EOSTRATA_STORE_EVICTION_BUFFER_MB=0

# Access tracking — set to false if last_access should reflect ingestion time only.
# EOSTRATA_TRACK_ACCESS=true

# Log file — daily rotating, 30 days history. Set to empty string to disable.
# EOSTRATA_LOG_FILE=data/eostrata.log

# Bounding box (west, south, east, north) in EPSG:4326
# Example below covers Nigeria
EOSTRATA_BBOX_WEST=2.0
EOSTRATA_BBOX_SOUTH=4.0
EOSTRATA_BBOX_EAST=15.0
EOSTRATA_BBOX_NORTH=14.0
```

> **Tip** — keep the bounding box as small as your use case allows. Clipping to your area of interest significantly reduces storage and processing time.

### Storage model and cache eviction

Each dataset is stored as a single Zarr group containing **all its timestamps in one array** (e.g. `worldpop/nga` holds every ingested year as a time step). This layout is deliberately chosen to make temporal aggregation fast: computing a mean, anomaly, or any other reduction over a date range is a single array operation with no cross-file joins.

**Cache eviction works at timestamp granularity**: when the store exceeds the configured quota before a new download, eostrata removes individual timestamps — oldest-accessed first — across all groups until the store fits within the quota. Removing a timestamp requires rebuilding the group in a temporary directory and atomically swapping it into place, so eviction I/O scales with the number of remaining timestamps in the group, not just the one being removed.

**Eviction headroom** — set `EOSTRATA_STORE_EVICTION_BUFFER_MB` to keep a buffer free inside the quota before each download (recommended: ~10% of quota). If quota is 10 000 MB and buffer is 1 000 MB, eviction runs until the store is at or below 9 000 MB.

**Last-access tracking** — filesystem `atime` is unreliable on most Linux mounts (`relatime`, `noatime`). eostrata instead maintains one sentinel file per timestamp inside a `.eostrata_access/` directory within each group. The sentinel is touched (debounced to once per minute) whenever that timestamp is accessed via tile serving or zonal statistics. The sentinel's modification time determines eviction order.

Set `EOSTRATA_TRACK_ACCESS=false` to disable this — last-access time will then reflect the ingestion timestamp only, which is useful if you want eviction order to match ingest order rather than recent use.

Practically this means:
- A timestamp that was ingested but never served via tiles or zonal stats shows **"never read"** in `eostrata list` — it is the first eviction candidate.
- A timestamp that is being actively served will not be evicted as long as it continues to receive requests within the debounce window.
- You can inspect per-timestamp last-access times and sizes at any time via `eostrata list` or `GET /store-usage`.

---

## Available data sources

eostrata ships with the following built-in sources. New sources can be added in a single file — see `docs/adding-a-source.md`.

| Source | What it is | Variables | Resolution | Lag |
|---|---|---|---|---|
| `worldpop` | WorldPop R2025A population count | `population` | Annual | ~1 year |
| `chirps` | CHIRPS v2.0 precipitation | `precipitation` | Monthly | ~45 days |
| `cds` | CDS / ERA5 climate reanalysis | `t2m` `tp` `u10` `v10` `sp` | Monthly | ~90 days |
| `cams` | CAMS EAC4 air quality reanalysis | `pm2p5` `pm10` `no2` `co` `o3` `so2` `aod550` | Monthly | ~120 days |
| `tropomi` | Sentinel-5P TROPOMI OFFLINE L2 | `no2` `co` `o3` `so2` `ch4` `hcho` `aer_ai` | Daily | ~3 days |
| `sentinel_ndvi` | CGLS Sentinel-3 NDVI 300m v2 | `ndvi` | Dekadal (10-day) | ~5 days |
| _your source_ | _implement `BaseSource`_ | _any_ | _any_ | — |

**Variable reference:**

| Source | Variable | Description | Unit |
|---|---|---|---|
| `worldpop` | `population` | Total population count per pixel | count |
| `chirps` | `precipitation` | Monthly precipitation total | mm |
| `cds` | `t2m` | 2m air temperature | K |
| `cds` | `tp` | Total precipitation | m |
| `cds` | `u10` | 10m U-component of wind | m/s |
| `cds` | `v10` | 10m V-component of wind | m/s |
| `cds` | `sp` | Surface pressure | Pa |
| `cams` | `pm2p5` | PM2.5 surface concentration | kg/m³ |
| `cams` | `pm10` | PM10 surface concentration | kg/m³ |
| `cams` | `no2` | Nitrogen dioxide surface concentration | kg/m³ |
| `cams` | `co` | Carbon monoxide surface concentration | kg/m³ |
| `cams` | `o3` | Ozone surface concentration | kg/m³ |
| `cams` | `so2` | Sulphur dioxide surface concentration | kg/m³ |
| `cams` | `aod550` | Total aerosol optical depth at 550 nm | dimensionless |
| `tropomi` | `no2` | Tropospheric NO₂ column | mol/m² |
| `tropomi` | `co` | Total CO column | mol/m² |
| `tropomi` | `o3` | Total O₃ column | mol/m² |
| `tropomi` | `so2` | Total SO₂ column | mol/m² |
| `tropomi` | `ch4` | CH₄ mixing ratio | ppb |
| `tropomi` | `hcho` | Tropospheric HCHO column | mol/m² |
| `tropomi` | `aer_ai` | Aerosol index | dimensionless |
| `sentinel_ndvi` | `ndvi` | Normalised Difference Vegetation Index | 0–1 |

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

This shows each Zarr group with its size, last-access time, total store usage, and quota if one is configured:

```
Zarr store: data/zarr  [142.3 MB / 10000 MB (1%)]
┌───────────────┬──────────┬──────────────────────┐
│ Group         │ Size     │ Last accessed        │
├───────────────┼──────────┼──────────────────────┤
│ chirps/global │  89.4 MB │ 2026-03-27 09:12 UTC │
│ worldpop/nga  │  52.9 MB │ never read           │
└───────────────┴──────────┴──────────────────────┘
```

"Never read" means the group was ingested but has never been opened for tile serving or zonal statistics — it is the first eviction candidate if quota is exceeded.

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
| STAC item id | `<iso3_lower>` |
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
| STAC item id | `global` |
| Time resolution | monthly (one timestep per month) |

---

### CDS / ERA5 — climate reanalysis

Monthly-averaged ERA5 reanalysis from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu). One Zarr group per variable, all years and months as timesteps.

**Setup required** — ERA5 needs a CDS account and API credentials before you can download:

1. Register at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu)
2. Create `~/.cdsapirc`:
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

**Download a specific month:**
```bash
uv run eostrata download cds --variable t2m --year 2023 --month 6
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
| STAC item id | `t2m` |
| Time resolution | monthly (one timestep per month) |

---

### CAMS — air quality reanalysis

Monthly-averaged surface air quality from the [Copernicus Atmosphere Monitoring Service](https://ads.atmosphere.copernicus.eu) (CAMS EAC4). One Zarr group per variable.

**Setup required** — CAMS needs an ADS account and API credentials:

1. Register at [ads.atmosphere.copernicus.eu](https://ads.atmosphere.copernicus.eu)
2. Create `~/.adsapirc`:
```
url: https://ads.atmosphere.copernicus.eu/api
key: <your-api-key>
```
Or set environment variables: `EOSTRATA_ADS_URL` and `EOSTRATA_ADS_KEY`.

**Download PM2.5 for the latest available year** (~4 months behind real time):
```bash
uv run eostrata download cams --variable pm2p5
```

**Download a specific year and months:**
```bash
uv run eostrata download cams --variable no2 --year 2023 --months 1,2,3
```

**Supported variables:** `pm2p5`, `pm10`, `no2`, `co`, `o3`, `so2`, `aod550` — see the variable reference table above.

**What gets written** (example for `pm2p5`):

| | Value |
|---|---|
| Zarr group | `data/zarr/cams/pm2p5/` |
| Variable | `pm2p5` |
| STAC item id | `pm2p5` |
| Time resolution | monthly (one timestep per month) |

---

### TROPOMI — Sentinel-5P air quality

Daily Level-2 atmospheric composition columns from [Sentinel-5P TROPOMI](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-5p) via the Copernicus Data Space Ecosystem (CDSE). Swath pixels are aggregated to a 0.1° grid. One Zarr group per variable.

**Setup required** — TROPOMI needs CDSE credentials:

1. Register at [dataspace.copernicus.eu](https://dataspace.copernicus.eu)
2. Set environment variables:
```bash
EOSTRATA_CDSE_USER=your@email.com
EOSTRATA_CDSE_PASSWORD=yourpassword
```

**Download NO₂ for the latest available day** (~3 days behind real time):
```bash
uv run eostrata download tropomi --variable no2
```

**Download a specific day:**
```bash
uv run eostrata download tropomi --variable no2 --year 2024 --month 3 --day 15
```

**Download a full month:**
```bash
uv run eostrata download tropomi --variable co --year 2024 --month 1 --days ALL
```

**Supported variables:** `no2`, `co`, `o3`, `so2`, `ch4`, `hcho`, `aer_ai` — see the variable reference table above.

> **Note** — TROPOMI data is not available for every day and region. Days with no matching products are silently skipped (`skip_404=true`).

**What gets written** (example for `no2`):

| | Value |
|---|---|
| Zarr group | `data/zarr/tropomi/no2/` |
| Variable | `no2` |
| STAC item id | `no2` |
| Time resolution | daily (one timestep per day) |

---

### Sentinel NDVI — vegetation index

Dekadal (10-day) 300m NDVI composites from the [Copernicus Global Land Service](https://land.copernicus.eu/global/products/ndvi) (CGLS, Sentinel-3 OLCI). A single global Zarr group holds all dekads as timesteps.

**Download the latest available dekad** (~5 days behind the end of the dekad):
```bash
uv run eostrata download sentinel_ndvi
```

**Download a specific year, month, and dekad (1, 2, or 3):**
```bash
uv run eostrata download sentinel_ndvi --year 2023 --month 6 --dekad 1
```

**Download all dekads in a month:**
```bash
uv run eostrata download sentinel_ndvi --year 2023 --month 6 --dekads ALL
```

Dekad 1 = days 1–10, dekad 2 = days 11–20, dekad 3 = days 21–end of month.

**What gets written:**

| | Value |
|---|---|
| Zarr group | `data/zarr/sentinel_ndvi/global/` |
| Variable | `ndvi` |
| STAC item id | `global` |
| Time resolution | dekadal (three timesteps per month) |

---

## Ingesting data via the API

Once the server is running you can trigger ingestion jobs via HTTP without blocking the server. Jobs run in the background on a thread pool.

**Start a WorldPop ingestion job:**
```bash
curl -s -X POST http://127.0.0.1:8000/processes/ingest/execution \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"source": "worldpop", "iso3": "NGA", "years": [2022, 2023]}}' \
  | python -m json.tool
```

**Start a CHIRPS job:**
```bash
curl -s -X POST http://127.0.0.1:8000/processes/ingest/execution \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"source": "chirps", "years": [2024], "months": [1,2,3]}}' \
  | python -m json.tool
```

**Start an ERA5 job:**
```bash
curl -s -X POST http://127.0.0.1:8000/processes/ingest/execution \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"source": "cds", "variable": "t2m", "years": [2023]}}' \
  | python -m json.tool
```

The response contains a `jobID`. Poll for status:

```bash
curl http://127.0.0.1:8000/processes/jobs/<jobID> | python -m json.tool
```

`status` will be `running`, `successful`, or `failed`. When nothing could be ingested (e.g. all requested periods returned 404), the job is marked `failed` with a descriptive `error` message.

List all jobs:
```bash
curl http://127.0.0.1:8000/processes/jobs | python -m json.tool
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
| Ready-to-use parameter values | `http://127.0.0.1:8000/examples` |
| Map viewer | `http://127.0.0.1:8000/map` |
| OGC collections | `http://127.0.0.1:8000/collections` |
| STAC catalogue | `http://127.0.0.1:8000/stac` |
| OGC Processes | `http://127.0.0.1:8000/processes` |

Add `--reload` for hot-reloading during development:

```bash
uv run eostrata serve --reload
```

Once running, open `/examples` first — it lists every ingested item with copy-pasteable `collection_id`, `item`, `datetime`, `group`, and `variable` values ready to paste into the Swagger UI at `/docs`.

---

## Exploring the map viewer

The map viewer is a catalog-aware Leaflet interface available at:

```
http://127.0.0.1:8000/map
```

The viewer has four tabs:

### Map tab

Use the dropdowns to select a collection, item, datetime, colormap and rescale range. The viewer loads all available data from the STAC catalogue automatically.

**Date modes:**
- **Single date** — select one timestep from the dropdown (default)
- **Interval** — enter a start/end date range and apply a temporal aggregation method (`mean`, `sum`, `min`, `max`, or `anomaly`). The `anomaly` method requires a separate baseline interval.

Tick **Auto-scale from stats** to set the rescale range automatically from the data's p5/p95 percentiles (falls back to p25/p75, then min/max if percentile data is unavailable). Editing the rescale field manually unticks the checkbox.

Zonal statistics (mean, min, max, std, sum, percentiles, nodata count) are computed automatically for the current bbox and displayed below the controls.

### Ingest tab

Start ingestion jobs directly from the browser. Select a source, fill in the parameters (years, months, variable, etc.), and click **Start ingestion job**. The job list below refreshes automatically and shows running/successful/failed status badges.

### Quota tab

Shows live disk usage, the quota bar (blue < 70% · amber 70–90% · red > 90%), and a per-group/timestamp breakdown. Groups and timestamps highlighted in red are oldest-accessed and will be evicted first if the quota is exceeded. Hover over a red entry for details.

### Config tab

Shows static configuration: bounding box, storage paths, and a reference table of all registered sources with their variables and temporal resolution. Also provides the **Rebuild catalog from Zarr** action for recovery if the catalog becomes out of sync.

---

You can also deep-link directly to a specific view with query parameters:

```
http://127.0.0.1:8000/map?collection=worldpop&item=nga&datetime=2020-01-01&rescale=0,1000&colormap_name=viridis
```

**Supported query parameters:**

| Parameter | Description | Example |
|---|---|---|
| `collection` | Collection id to pre-select | `worldpop`, `chirps`, `era5` |
| `item` | STAC item id within the collection | `nga` |
| `datetime` | ISO 8601 datetime or interval for time selection | `2021-01-01` or `2021-01-01/2022-12-31` |
| `agg` | Temporal aggregation method applied over `datetime` interval | `mean`, `sum`, `min`, `max`, `anomaly` |
| `baseline` | ISO 8601 interval for anomaly baseline (required when `agg=anomaly`) | `2015-01-01/2020-12-31` |
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
  "id": "nga",
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

The `zonalstats` process computes per-feature summary statistics over any collection. Send a GeoJSON `FeatureCollection` as the input zones. The `datetime`, `agg`, and `baseline` parameters apply temporal aggregation before extraction — so you can compute statistics over a mean, anomaly, or any other aggregated period.

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

## Scheduled ingestion

eostrata includes an in-process APScheduler that runs alongside the server. Edit `schedules.yml` to declare cron-based ingestion jobs. Each job supports:

- `auto_period: true` — see below
- Exponential-backoff retry (3 attempts)
- Optional webhook alert (`webhook_url`) on final failure

Example `schedules.yml` entry:

```yaml
jobs:
  - id: chirps_monthly
    source: chirps
    params: {}
    cron: "0 3 15 * *"   # 03:00 on the 15th of each month
    auto_period: true
    enabled: true
```

The scheduler starts automatically with `eostrata serve` and stops cleanly on shutdown.

### Understanding `auto_period`

When `auto_period: true` is set on a job, eostrata calls `source.latest_available()` at job run time to determine the most recent period for which data is expected to be available — based on the source's typical publication lag. This computed period overrides any `year`/`month`/`day` values in `params`.

For example, CHIRPS has a ~45-day lag. A job running on 2024-03-15 with `auto_period: true` will target 2024-01 (roughly 45 days back), not the current month.

**Per-source behaviour:**

| Source | `latest_available()` returns | Typical lag |
|---|---|---|
| `worldpop` | previous year | ~1 year |
| `chirps` | ~45 days ago (year + month) | 45 days |
| `cds` | ~90 days ago (year + month) | 90 days |
| `cams` | ~120 days ago (year + month) | 120 days |
| `tropomi` | ~3 days ago (year + month + day) | 3 days |
| `sentinel_ndvi` | ~5 days ago (year + month only) | 5 days |

> **Note for Sentinel NDVI** — `auto_period` resolves year and month automatically, but the `dekad` must still be specified explicitly in `params` since the scheduler does not yet resolve dekadal periods automatically. See the commented examples in `schedules.yml`.

When `auto_period: false` (the default), `params` must include explicit `year`/`month`/`day` values — use this when you want to backfill a specific historical period.

---

## CLI reference

```
eostrata --help

Commands:
  download chirps     Download a CHIRPS monthly precipitation raster
  download cds        Download ERA5 monthly reanalysis from Copernicus CDS
  download worldpop   Download a WorldPop population raster
  list                List datasets in the Zarr store and STAC catalogue
  rebuild-catalog     Rebuild the STAC catalogue by scanning the Zarr store
  serve               Start the tile server, STAC catalogue and OGC Processes API
  test                Run the test suite with coverage
  lint                Run ruff linter and formatter
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
  --months TEXT           Multiple months, comma-separated: 1,2,3 or ALL
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
  --month INTEGER         Single month 1-12 (default: latest available)
  --months TEXT           Months to fetch, comma-separated: 1,2,3 or ALL (default: latest available)
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

**test**
```
uv run eostrata test [OPTIONS]

Options:
  -v, --verbose           Verbose pytest output (show individual test names)
```

**lint**
```
uv run eostrata lint [OPTIONS]

Options:
  --fix / --no-fix        Auto-fix ruff lint violations (default: --fix)
```

**rebuild-catalog**
```
uv run eostrata rebuild-catalog [OPTIONS]

Options:
  --zarr-root PATH        Zarr store root
  --catalog-path PATH     Path to catalog.json
```

Scans all Zarr groups in the store and reconstructs `catalog.json` from scratch. Useful when the catalogue is missing or out of sync with the stored data (e.g. after manual edits or a partial failure). Also available as `POST /processes/rebuild-catalog/execution` via the API.

**cleanup**
```
uv run eostrata cleanup [OPTIONS]

Options:
  --zarr-root PATH    Zarr store root
  --raw-dir PATH      Raw downloads directory
  --catalog-path PATH Path to catalog.json
  --yes, -y           Skip confirmation prompt
```
