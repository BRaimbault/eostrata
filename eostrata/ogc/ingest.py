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
from typing import Annotated, Literal  # Literal still used for months/dekads "ALL"

from fastapi import APIRouter, HTTPException, Path, Response
from pydantic import BaseModel, Field, field_validator, model_validator

from eostrata import ingestion, jobs
from eostrata.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/processes", tags=["Data Ingestion"])

_executor = ThreadPoolExecutor(max_workers=settings.ingest_max_workers, thread_name_prefix="ingest")

# ── Process IDs (imported by processes.py for the unified list) ───────────────

INGEST_PROCESS_IDS = [
    {"id": "ingest", "version": "0.1.0"},
    {"id": "rebuild-catalog", "version": "0.1.0"},
]

# ── Source registry for the UI ────────────────────────────────────────────────
# Derived from the source registry — adding a new source with ui_fields defined
# automatically makes it available here.

import eostrata.sources  # noqa: E402,F401 — triggers auto-discovery of all source modules
from eostrata.sources.base import all_sources as _all_sources  # noqa: E402

INGEST_SOURCES = [
    {
        "id": cls.id,
        "label": f"{cls.id} — {cls.collection_title}",
        "fields": cls.ui_fields,
        "variables": cls.VARIABLES if cls.VARIABLES else [cls.VARIABLE],
        "variable_descriptions": cls.VARIABLE_DESCRIPTIONS,
        "temporal_resolution": cls.temporal_resolution,
        "lag_days": cls.default_lag_days,
    }
    for cls in _all_sources()
    if cls.ui_fields
]

# Derived from INGEST_SOURCES so we never have to update the two separately.
_SOURCE_IDS = [s["id"] for s in INGEST_SOURCES]
_source_list = ", ".join(f"``{sid}``" for sid in _SOURCE_IDS)

# ── Process description ───────────────────────────────────────────────────────

_INGEST_DESCRIPTION = {
    "id": "ingest",
    "title": "Data ingestion",
    "description": (
        "Download earth observation data, clip to the configured bounding box, "
        f"write to the Zarr store, and register a STAC item. "
        f"Set ``source`` to select the dataset: {_source_list}."
    ),
    "version": "0.1.0",
    "jobControlOptions": ["async-execute"],
    "inputs": {
        "source": {
            "title": "Source",
            "description": f"Dataset to ingest: {', '.join(_SOURCE_IDS)}.",
            "schema": {"type": "string", "enum": _SOURCE_IDS},
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
                "(chirps/cds/sentinel_ndvi only; default: latest available)."
            ),
            "schema": {
                "oneOf": [
                    {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 12}},
                    {"type": "string", "enum": ["ALL"]},
                ]
            },
        },
        "dekads": {
            "title": "Dekads",
            "description": (
                "List of dekads 1-3, or the string 'ALL' for all three dekads "
                "(sentinel_ndvi only; default: latest available)."
            ),
            "schema": {
                "oneOf": [
                    {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 3}},
                    {"type": "string", "enum": ["ALL"]},
                ]
            },
        },
        "days": {
            "title": "Days",
            "description": (
                "List of days 1-31, or the string 'ALL' for every day of the month "
                "(daily sources only; default: latest available)."
            ),
            "schema": {
                "oneOf": [
                    {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 31}},
                    {"type": "string", "enum": ["ALL"]},
                ]
            },
        },
    },
    "outputs": {"job_id": {"title": "Job ID", "schema": {"type": "string"}}},
}

# ── Pydantic model ────────────────────────────────────────────────────────────

Month = Annotated[int, Field(ge=1, le=12)]
Dekad = Annotated[int, Field(ge=1, le=3)]
Day = Annotated[int, Field(ge=1, le=31)]

_ALL_MONTHS = list(range(1, 13))
_ALL_DEKADS = [1, 2, 3]
_ALL_DAYS = list(range(1, 32))


class IngestInputs(BaseModel):
    source: Annotated[str, Field(json_schema_extra={"enum": _SOURCE_IDS})]

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        if v not in _SOURCE_IDS:
            raise ValueError(f"must be one of {_SOURCE_IDS}")
        return v

    iso3: Annotated[str, Field(min_length=3, max_length=3)] | None = Field(
        None, description="ISO 3166-1 alpha-3 country code (worldpop only)"
    )
    variable: Literal["t2m", "tp", "u10", "v10", "sp"] | None = Field(
        None, description="ERA5 variable short name (cds only)"
    )
    years: list[int] | None = Field(None, description="Years to ingest")
    months: list[Month] | Literal["ALL"] | None = Field(
        None,
        description="Months 1-12 to ingest, or 'ALL' for every month (chirps/cds/sentinel_ndvi)",
    )
    dekads: list[Dekad] | Literal["ALL"] | None = Field(
        None, description="Dekads 1-3 to ingest, or 'ALL' for all three (sentinel_ndvi only)"
    )
    days: list[Day] | Literal["ALL"] | None = Field(
        None,
        description="Days 1-31 to ingest, or 'ALL' for every day of the month (daily sources only)",
    )

    @field_validator("months", mode="before")
    @classmethod
    def expand_all_months(cls, v):
        if isinstance(v, str) and v.strip().upper() == "ALL":
            return _ALL_MONTHS
        return v

    @field_validator("dekads", mode="before")
    @classmethod
    def expand_all_dekads(cls, v):
        if isinstance(v, str) and v.strip().upper() == "ALL":
            return _ALL_DEKADS
        return v

    @field_validator("days", mode="before")
    @classmethod
    def expand_all_days(cls, v):
        if isinstance(v, str) and v.strip().upper() == "ALL":
            return _ALL_DAYS
        return v

    @model_validator(mode="after")
    def check_source_fields(self) -> IngestInputs:
        from eostrata.sources.base import get_source

        try:
            source_cls = get_source(self.source)
        except ValueError:  # pragma: no cover
            return self  # Invalid source already caught by validate_source
        if "iso3" in source_cls.ui_fields and self.iso3 is None:
            raise ValueError(f"iso3 is required when source is '{self.source}'")
        return self


class IngestExecutionRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"inputs": {"source": "worldpop", "iso3": "NGA", "years": [2023]}},
                {"inputs": {"source": "chirps", "years": [2024], "months": [1, 2, 3]}},
                {"inputs": {"source": "cds", "variable": "t2m", "years": [2023]}},
                {
                    "inputs": {
                        "source": "sentinel_ndvi",
                        "years": [2024],
                        "months": [1],
                        "dekads": [1, 2, 3],
                    }
                },
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
    """Start a data ingestion job for any registered source.

    Returns a ``job_id`` immediately. Poll ``GET /processes/jobs/{job_id}`` for status.
    """
    pending_count = sum(1 for j in jobs.list_jobs() if j.status in ("accepted", "running"))
    if pending_count >= settings.ingest_max_queued:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Too many ingestion jobs queued ({pending_count}). "
                f"Maximum is {settings.ingest_max_queued}. "
                "Poll /processes/jobs and retry when existing jobs have completed."
            ),
        )

    inp = body.inputs
    logger.info("API POST /processes/ingest/execution %s", inp.model_dump(exclude_none=True))

    from eostrata.sources.base import get_source

    source_cls = get_source(inp.source)
    source = source_cls()
    latest = source.latest_available()

    source_params: dict = {}
    if "iso3" in source_cls.ui_fields:
        source_params["iso3"] = inp.iso3.upper()
    if "variable" in source_cls.ui_fields:
        source_params["variable"] = inp.variable or "t2m"
    if "years" in source_cls.ui_fields:
        source_params["years"] = inp.years or [latest.year]
    if "months" in source_cls.ui_fields:
        source_params["months"] = inp.months or [latest.month]
    if "dekads" in source_cls.ui_fields:
        default_dekad = 1 if latest.day < 11 else (2 if latest.day < 21 else 3)
        source_params["dekads"] = inp.dekads or [default_dekad]
    if "days" in source_cls.ui_fields:  # pragma: no cover
        source_params["days"] = inp.days or [latest.day]

    job = jobs.create_job(inp.source, source_params)
    _executor.submit(
        _run_job,
        job.job_id,
        ingestion.run_ingest,
        source_id=inp.source,
        zarr_root=settings.zarr_root,
        raw_dir=settings.raw_dir,
        catalog_path=settings.catalog_path,
        bbox=settings.bbox,
        quota_mb=settings.store_quota_mb,
        eviction_buffer_mb=settings.store_eviction_buffer_mb,
        **source_params,
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
        "status": "successful",
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
