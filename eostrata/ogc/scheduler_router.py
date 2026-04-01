"""REST API for the in-process APScheduler scheduler.

Endpoints
---------
GET    /scheduler/jobs                List all configured jobs (+ next run time)
POST   /scheduler/jobs                Create a new scheduled job
PUT    /scheduler/jobs/{job_id}       Replace an existing job
DELETE /scheduler/jobs/{job_id}       Remove a job
POST   /scheduler/jobs/{job_id}/run   Trigger a job immediately
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel, Field, field_validator

router = APIRouter(prefix="/scheduler", tags=["Scheduler"])


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_scheduler():
    from eostrata.scheduler import get_scheduler

    s = get_scheduler()
    if s is None:
        raise HTTPException(
            status_code=503,
            detail="Scheduler is not running. Ensure APScheduler and PyYAML are installed.",
        )
    return s


# ── Pydantic model ────────────────────────────────────────────────────────────


class JobDef(BaseModel):
    """Schema for a scheduled job definition."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "chirps_monthly",
                    "source": "chirps",
                    "params": {},
                    "cron": "0 3 15 * *",
                    "auto_period": True,
                    "enabled": True,
                },
                {
                    "id": "era5_t2m",
                    "source": "cds",
                    "params": {"variable": "t2m"},
                    "cron": "0 4 1 * *",
                    "auto_period": True,
                    "enabled": True,
                },
            ]
        }
    }

    id: str = Field(..., description="Unique job identifier", min_length=1)
    source: str = Field(..., description="Registered source id")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Extra keyword args forwarded to the source"
    )
    cron: str = Field(
        ...,
        description="Cron expression: 'minute hour day-of-month month day-of-week'",
        examples=["0 3 15 * *", "0 4 1 * *", "0 7 * * *"],
    )
    auto_period: bool = Field(
        False,
        description=(
            "When true, year/month/day are resolved from source.latest_available() "
            "at runtime and override any values in params."
        ),
    )
    enabled: bool = Field(True, description="Set false to disable without deleting.")

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        import eostrata.sources  # noqa: F401 — trigger auto-discovery
        from eostrata.sources.base import all_sources

        valid = {s.id for s in all_sources()}
        if v not in valid:
            raise ValueError(f"Unknown source '{v}'. Available: {sorted(valid)}")
        return v

    @field_validator("cron")
    @classmethod
    def validate_cron(cls, v: str) -> str:
        parts = v.strip().split()
        if len(parts) != 5:
            raise ValueError(
                "cron must have exactly 5 fields: 'minute hour day-of-month month day-of-week'"
            )
        return v.strip()


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/jobs", summary="List all scheduled jobs")
def list_scheduler_jobs() -> dict:
    """Return all jobs from schedules.yml enriched with APScheduler next_run_time."""
    s = _get_scheduler()
    return {"jobs": s.get_jobs()}


@router.post("/jobs", status_code=201, summary="Create a scheduled job")
def create_job(body: JobDef) -> dict:
    """Add a new job to the scheduler and persist it to schedules.yml."""
    s = _get_scheduler()
    existing_ids = {j["id"] for j in s.get_jobs()}
    if body.id in existing_ids:
        raise HTTPException(
            status_code=409, detail=f"Job '{body.id}' already exists. Use PUT to update."
        )
    s.save_job(body.model_dump())
    return {"status": "created", "job_id": body.id}


@router.put("/jobs/{job_id}", summary="Replace a scheduled job")
def update_job(
    job_id: str = Path(..., description="Job ID to update"),
    body: JobDef = ...,
) -> dict:
    """Replace an existing job definition and re-register it with APScheduler."""
    s = _get_scheduler()
    job_def = body.model_dump()
    job_def["id"] = job_id  # path param is authoritative
    s.save_job(job_def)
    return {"status": "updated", "job_id": job_id}


@router.delete("/jobs/{job_id}", status_code=204, summary="Delete a scheduled job")
def delete_job(
    job_id: str = Path(..., description="Job ID to delete"),
) -> None:
    """Remove a job from the scheduler and from schedules.yml."""
    s = _get_scheduler()
    s.remove_job(job_id)


@router.post("/jobs/{job_id}/run", status_code=202, summary="Trigger a job immediately")
def trigger_job(
    job_id: str = Path(..., description="Job ID to run now"),
) -> dict:
    """Run a scheduled job immediately in a background thread (ignores cron schedule)."""
    s = _get_scheduler()
    try:
        s.trigger_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "triggered", "job_id": job_id}
