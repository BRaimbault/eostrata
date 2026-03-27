"""In-memory job store for async ingestion jobs."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum


class JobStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "successful"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    source: str
    params: dict
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    message: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            # OGC API - Processes required fields
            "jobID": self.job_id,
            "type": "process",
            "processID": "ingest",
            "status": self.status,
            # OGC optional timing fields
            "created": self.created_at.isoformat(),
            "updated": self.updated_at.isoformat(),
            "message": self.message,
            # eostrata-specific fields (used by the viewer)
            "source": self.source,
            "params": self.params,
            "error": self.error,
        }


_store: dict[str, Job] = {}
_lock = threading.Lock()


def create_job(source: str, params: dict) -> Job:
    now = datetime.now(tz=UTC)
    job = Job(
        job_id=uuid.uuid4().hex,
        source=source,
        params=params,
        status=JobStatus.RUNNING,
        created_at=now,
        updated_at=now,
    )
    with _lock:
        _store[job.job_id] = job
    return job


def get_job(job_id: str) -> Job | None:
    with _lock:
        return _store.get(job_id)


def list_jobs() -> list[Job]:
    with _lock:
        return list(_store.values())


def mark_succeeded(job_id: str, message: str = "") -> None:
    with _lock:
        job = _store.get(job_id)
        if job is not None:
            job.status = JobStatus.SUCCEEDED
            job.message = message
            job.updated_at = datetime.now(tz=UTC)


def mark_failed(job_id: str, error: str) -> None:
    with _lock:
        job = _store.get(job_id)
        if job is not None:
            job.status = JobStatus.FAILED
            job.error = error
            job.updated_at = datetime.now(tz=UTC)
