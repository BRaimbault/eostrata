"""In-memory job store for async ingestion jobs."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum


class JobStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
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
            "job_id": self.job_id,
            "source": self.source,
            "params": self.params,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message": self.message,
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
