# eostrata

*One tool to fetch, store, aggregate, and serve earth observation layers.*

---

## Features

- **Multi-source ingestion**: fetches earth observation data through a unified `BaseSource` plugin interface. New sources register with a single decorator: the scheduler, catalogue, and store pick them up automatically.
  - WorldPop population rasters
  - CDS / ERA5 climate reanalysis
  - CHIRPS precipitation

- **Zarr collection store**: each ingested resource is stored as a CF-compliant (Climate and Forecast conventions - standard naming for dimensions, coordinates, units and fill values) Zarr collection with `x`, `y`, and `time` dimensions, locally or on cloud object storage. When the storage quota is reached, data is evicted before new downloads proceed.

- **STAC catalogue**: every collection is automatically registered as a STAC item in an embedded `stac-fastapi` catalogue, persisted in `catalog.json`. Primary discovery interface for datasets and their assets.

- **OGC-compliant serving**: all endpoints follow OGC API - Common conventions (`/`, `/conformance`, `/collections`) as a compatibility shim for OGC-native clients. All serving endpoints accept a `datetime` range and an `agg` parameter for **temporal aggregation** - collapsing the time dimension at request time with no pre-computed intermediates. Supported methods: `mean`, `sum`, `min`, `max`, and `anomaly` (deviation from a user-defined `baseline` period expressed as an ISO 8601 interval).
  - **OGC API - Tiles**: dynamic raster tiles served directly from the Zarr store via `titiler.xarray`, no intermediate COG export. WMS-compatible. On-the-fly styling via `colormap_name` and `rescale`. Each tile can represent a single timestep or a temporally aggregated period.
  - **OGC API - Processes - Zonal Statistics**: summarises raster values within polygon zones. The `zonalstats` process accepts a GeoJSON `FeatureCollection` and returns per-feature statistics (mean, sum, min, max, std, count, percentiles). Temporal aggregation parameters apply before zonal extraction, so statistics can be computed over a single timestep or a temporally aggregated period.
  - **OGC API - Processes - Ingest**: async ingestion jobs (`POST /processes/ingest/execution`) that download, clip, and write data to the Zarr store without blocking the server. Job status is tracked and pollable via `GET /processes/jobs/{job_id}`.

- **Automated scheduler**: an `APScheduler` instance runs in-process alongside the FastAPI server. Jobs are declared in `schedules.yml` with cron expressions. Each source exposes its typical data lag so `auto_period: true` targets the latest available interval. Failed jobs retry with exponential backoff then dispatch a webhook alert.

---

## Architecture

```mermaid
flowchart TD
    WP["WorldPop\nREST API"]:::src
    CDS["CDS / ERA5\ncdsapi"]:::src
    CH["CHIRPS\nHTTP / .gz"]:::src

    CLI["CLI / Python API\neostrata download"]:::infra
    SCH["Scheduler\nAPScheduler · cron · webhook alert"]:::infra
    SY["schedules.yml"]:::config

    DL["Downloader\nBaseSource registry · httpx · cdsapi"]:::infra

    ZS["Zarr store\nCF-compliant · eviction policy · local or cloud"]:::store
    CAT["STAC catalogue\ncatalog.json · primary discovery interface"]:::store

    TILE["OGC API - Tiles\ntitiler.xarray · WMS-compatible\ntemporal aggregation at request time"]:::serve
    ZON["OGC API - Processes\nzonalstats · per-feature statistics\ntemporal aggregation before extraction"]:::serve
    ING["OGC API - Processes\ningest · async jobs · job polling"]:::serve
    OGC["OGC API - Common\n/ · /conformance · /collections"]:::serve

    WP & CDS & CH --> DL
    CLI --> DL
    SY --> SCH
    SCH --> DL
    ING --> DL
    DL --> ZS
    ZS -->|auto-register| CAT
    CAT --> OGC
    ZS --> TILE & ZON & OGC

    classDef src    fill:#E1F5EE,stroke:#0F6E56,color:#085041
    classDef config fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A
    classDef infra  fill:#D3D1C7,stroke:#5F5E5A,color:#2C2C2A
    classDef store  fill:#FAEEDA,stroke:#854F0B,color:#412402
    classDef serve  fill:#EEEDFE,stroke:#534AB7,color:#26215C
```

### Module map

```
eostrata/
├── eostrata/
│   ├── sources/
│   │   ├── base.py          BaseSource ABC + @register_source registry + retry logic
│   │   ├── worldpop.py      WorldPopSource
│   │   ├── cds.py           CDSSource
│   │   ├── chirps.py        CHIRPSSource
│   │   └── __init__.py      populates the source registry on import
│   ├── ogc/
│   │   ├── ingest.py        OGC API - Processes: async ingest jobs + job polling
│   │   ├── tiles.py         OGC API - Tiles (wraps titiler.xarray)
│   │   └── processes.py     OGC API - Processes: zonalstats
│   ├── templates/
│   │   └── map.html         interactive map viewer (Leaflet)
│   ├── config.py            pydantic-settings · all env vars
│   ├── store.py             GeoTIFF → Zarr · clip · nodata handling
│   ├── ingestion.py         download + zarr write + STAC registration (sync)
│   ├── cache.py             quota tracking · LRU eviction · access sentinels
│   ├── catalog.py           pystac STAC catalogue backend
│   ├── aggregate.py         AggregatingReader · temporal aggregation
│   ├── jobs.py              in-memory async job store
│   ├── log.py               logging setup · rotating file handler
│   ├── scheduler.py         APScheduler · cron jobs · retry · webhook alert
│   ├── server.py            assembles all routers · OGC Common endpoints
│   └── cli.py               Typer CLI
├── schedules.yml            user-facing schedule config
└── pyproject.toml
```

---

