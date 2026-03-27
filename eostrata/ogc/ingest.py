"""OGC API - Processes: async ingestion process.

Endpoints
---------
GET  /processes/ingest              Process description
POST /processes/ingest/execution    Start an ingestion job
GET  /processes/jobs                List all jobs
GET  /processes/jobs/{job_id}       Poll a single job
"""

from __future__ import annotations

import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Path, Response
from pydantic import BaseModel, Field, field_validator, model_validator

from eostrata import ingestion, jobs
from eostrata.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/processes", tags=["Data Ingestion"])

_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="ingest")

# ── Process IDs (imported by processes.py for the unified list) ───────────────

INGEST_PROCESS_IDS = [
    {"id": "ingest", "version": "0.1.0"},
    {"id": "rebuild-catalog", "version": "0.1.0"},
]

# ── Process description ───────────────────────────────────────────────────────

_INGEST_DESCRIPTION = {
    "id": "ingest",
    "title": "Data ingestion",
    "description": (
        "Download earth observation data, clip to the configured bounding box, "
        "write to the Zarr store, and register a STAC item. "
        "Set ``source`` to select the dataset: ``worldpop``, ``chirps``, or ``cds``."
    ),
    "version": "0.1.0",
    "jobControlOptions": ["async-execute"],
    "inputs": {
        "source": {
            "title": "Source",
            "description": "Dataset to ingest: worldpop, chirps, or cds.",
            "schema": {"type": "string", "enum": ["worldpop", "chirps", "cds"]},
        },
        "iso3": {
            "title": "ISO3 country code",
            "description": "ISO 3166-1 alpha-3 country code (worldpop only), e.g. NGA.",
            "schema": {"type": "string", "minLength": 3, "maxLength": 3},
        },
        "variable": {
            "title": "Variable",
            "description": "ERA5 short name (cds only): t2m, tp, u10, v10, sp.",
            "schema": {"type": "string", "enum": ["t2m", "tp", "u10", "v10", "sp"]},
        },
        "years": {
            "title": "Years",
            "description": "List of years to ingest (default: latest available).",
            "schema": {"type": "array", "items": {"type": "integer"}},
        },
        "months": {
            "title": "Months",
            "description": (
                "List of months 1-12, or the string 'ALL' for every month "
                "(chirps/cds only; default: latest available)."
            ),
            "schema": {
                "oneOf": [
                    {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 12}},
                    {"type": "string", "enum": ["ALL"]},
                ]
            },
        },
    },
    "outputs": {"job_id": {"title": "Job ID", "schema": {"type": "string"}}},
}

# ── Pydantic model ────────────────────────────────────────────────────────────

Month = Annotated[int, Field(ge=1, le=12)]

_ALL_MONTHS = list(range(1, 13))


class IngestInputs(BaseModel):
    source: Literal["worldpop", "chirps", "cds"]
    iso3: Annotated[str, Field(min_length=3, max_length=3)] | None = Field(
        None, description="ISO 3166-1 alpha-3 country code (worldpop only)"
    )
    variable: Literal["t2m", "tp", "u10", "v10", "sp"] | None = Field(
        None, description="ERA5 variable short name (cds only)"
    )
    years: list[int] | None = Field(None, description="Years to ingest")
    months: list[Month] | Literal["ALL"] | None = Field(
        None, description="Months 1-12 to ingest, or 'ALL' for every month (chirps/cds only)"
    )

    @field_validator("months", mode="before")
    @classmethod
    def expand_all_months(cls, v):
        if isinstance(v, str) and v.strip().upper() == "ALL":
            return _ALL_MONTHS
        return v

    @model_validator(mode="after")
    def check_source_fields(self) -> IngestInputs:
        if self.source == "worldpop" and self.iso3 is None:
            raise ValueError("iso3 is required when source is 'worldpop'")
        return self


class IngestExecutionRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"inputs": {"source": "worldpop", "iso3": "NGA", "years": [2023]}},
                {"inputs": {"source": "chirps", "years": [2024], "months": [1, 2, 3]}},
                {"inputs": {"source": "cds", "variable": "t2m", "years": [2023]}},
            ]
        }
    }

    inputs: IngestInputs


# ── Job runner ────────────────────────────────────────────────────────────────


def _run_job(job_id: str, fn, **kwargs) -> None:
    fn_name = getattr(fn, "__name__", repr(fn))
    logger.info("Job %s started: %s", job_id, fn_name)
    try:
        failed, saved = fn(**kwargs)
        if not saved:
            if failed:
                msg = f"Nothing ingested — {len(failed)} period(s) failed: {', '.join(failed)}"
            else:
                msg = "Nothing ingested — all requested periods may be unavailable"
            logger.warning("Job %s failed — %s", job_id, msg)
            jobs.mark_failed(job_id, msg)
        elif failed:
            msg = f"Partial: {len(failed)} period(s) failed to download: {', '.join(failed)}"
            logger.warning("Job %s succeeded with warnings — %s", job_id, msg)
            jobs.mark_succeeded(job_id, message=msg)
        else:
            jobs.mark_succeeded(job_id)
            logger.info("Job %s succeeded", job_id)
    except Exception:
        error = traceback.format_exc()
        logger.exception("Job %s failed: %s", job_id, fn_name)
        jobs.mark_failed(job_id, error)


def _job_response(job: jobs.Job) -> dict:
    d = job.to_dict()
    d["links"] = [
        {"href": f"/processes/jobs/{job.job_id}", "rel": "monitor", "type": "application/json"}
    ]
    return d


# ── Rebuild-catalog process description ──────────────────────────────────────

_REBUILD_CATALOG_DESCRIPTION = {
    "id": "rebuild-catalog",
    "title": "Rebuild STAC catalogue",
    "description": (
        "Scan all Zarr groups in the store and reconstruct the STAC catalogue from scratch. "
        "Reads time coordinates and spatial extent directly from the data. "
        "Useful when the catalogue is missing or out of sync with the stored data."
    ),
    "version": "0.1.0",
    "jobControlOptions": ["sync-execute"],
    "inputs": {},
    "outputs": {
        "groups": {
            "title": "Rebuilt groups",
            "description": "Map of group path to number of timestamps registered.",
            "schema": {"type": "object", "additionalProperties": {"type": "integer"}},
        },
        "total_timestamps": {
            "title": "Total timestamps",
            "schema": {"type": "integer"},
        },
    },
}

# ── Process description endpoint ──────────────────────────────────────────────


@router.get("/ingest", summary="Ingest process description")
def describe_ingest() -> dict:
    return _INGEST_DESCRIPTION


# ── Execution endpoint ────────────────────────────────────────────────────────


@router.post("/ingest/execution", status_code=201, summary="Start an ingestion job")
def execute_ingest(body: IngestExecutionRequest, response: Response) -> dict:
    """
    Start a data ingestion job for WorldPop, CHIRPS, or CDS/ERA5.

    Returns a ``job_id`` immediately. Poll ``GET /processes/jobs/{job_id}`` for status.

    **source = worldpop** — requires ``iso3``
    **source = chirps** — uses ``years`` and ``months``
    **source = cds** — uses ``variable`` (default ``t2m``), ``years``, and ``months``

    All year/month values default to the latest available when omitted.
    """
    inp = body.inputs
    logger.info("API POST /processes/ingest/execution %s", inp.model_dump(exclude_none=True))

    if inp.source == "worldpop":
        from eostrata.sources import WorldPopSource

        years = inp.years or [WorldPopSource().latest_available().year]
        job = jobs.create_job("worldpop", {"iso3": inp.iso3.upper(), "years": years})
        _executor.submit(
            _run_job,
            job.job_id,
            ingestion.run_worldpop_ingest,
            iso3=inp.iso3,
            years=years,
            zarr_root=settings.zarr_root,
            raw_dir=settings.raw_dir,
            catalog_path=settings.catalog_path,
            bbox=settings.bbox,
            quota_mb=settings.store_quota_mb,
            eviction_buffer_mb=settings.store_eviction_buffer_mb,
        )

    elif inp.source == "chirps":
        from eostrata.sources.chirps import CHIRPSSource

        latest = CHIRPSSource().latest_available()
        years = inp.years or [latest.year]
        months = inp.months or [latest.month]
        job = jobs.create_job("chirps", {"years": years, "months": months})
        _executor.submit(
            _run_job,
            job.job_id,
            ingestion.run_chirps_ingest,
            years=years,
            months=months,
            zarr_root=settings.zarr_root,
            raw_dir=settings.raw_dir,
            catalog_path=settings.catalog_path,
            bbox=settings.bbox,
            quota_mb=settings.store_quota_mb,
            eviction_buffer_mb=settings.store_eviction_buffer_mb,
        )

    else:  # cds
        from eostrata.sources.cds import CDSSource

        variable = inp.variable or "t2m"
        latest = CDSSource().latest_available()
        years = inp.years or [latest.year]
        months = inp.months or [latest.month]
        job = jobs.create_job("cds", {"variable": variable, "years": years, "months": months})
        _executor.submit(
            _run_job,
            job.job_id,
            ingestion.run_cds_ingest,
            variable=variable,
            years=years,
            months=months,
            zarr_root=settings.zarr_root,
            raw_dir=settings.raw_dir,
            catalog_path=settings.catalog_path,
            bbox=settings.bbox,
            quota_mb=settings.store_quota_mb,
            eviction_buffer_mb=settings.store_eviction_buffer_mb,
        )

    response.headers["Location"] = f"/processes/jobs/{job.job_id}"
    return _job_response(job)


# ── Rebuild-catalog endpoints ─────────────────────────────────────────────────


@router.get(
    "/rebuild-catalog", tags=["Store & Catalog"], summary="Rebuild-catalog process description"
)
def describe_rebuild_catalog() -> dict:
    return _REBUILD_CATALOG_DESCRIPTION


@router.post(
    "/rebuild-catalog/execution",
    tags=["Store & Catalog"],
    summary="Rebuild STAC catalogue from Zarr store",
)
def execute_rebuild_catalog() -> dict:
    """
    Scan all Zarr groups and rebuild the STAC catalogue from scratch.

    Runs synchronously and returns the rebuilt groups with their timestamp counts.
    """
    results = ingestion.rebuild_catalog_from_zarr(
        zarr_root=settings.zarr_root,
        catalog_path=settings.catalog_path,
    )
    return {
        "status": "succeeded",
        "groups": results,
        "total_timestamps": sum(results.values()),
    }


# ── Job polling endpoints ─────────────────────────────────────────────────────


@router.get("/jobs", summary="List all ingestion jobs")
def list_jobs() -> dict:
    return {
        "jobs": [j.to_dict() for j in jobs.list_jobs()],
        "links": [{"href": "/processes/jobs", "rel": "self", "type": "application/json"}],
    }


@router.get("/jobs/{job_id}", summary="Poll a single ingestion job")
def get_job(
    job_id: str = Path(..., description="Job ID returned by an execution endpoint"),
) -> dict:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job.to_dict()
