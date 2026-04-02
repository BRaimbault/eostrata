"""APScheduler-based ingestion scheduler.

Reads ``schedules.yml`` at startup and registers cron jobs that call
source.download() + source.to_zarr() + catalog.register_item().

Features
--------
- YAML-configured cron jobs with ``auto_period`` support
- Exponential-backoff retry (3 attempts by default)
- Webhook alert (HTTP POST) when a job fails after all retries
- In-process scheduler — runs alongside the FastAPI server
- Graceful start / shutdown via FastAPI lifespan
- REST-accessible CRUD (add / update / remove / trigger jobs at runtime)

Usage in server.py
------------------
    from contextlib import asynccontextmanager
    from eostrata.scheduler import Scheduler, set_scheduler

    @asynccontextmanager
    async def lifespan(app):
        scheduler = Scheduler()
        set_scheduler(scheduler)
        scheduler.start()
        yield
        scheduler.stop()
        set_scheduler(None)

    app = FastAPI(lifespan=lifespan)
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_SCHEDULES_FILE = Path(__file__).parent.parent / "schedules.yml"
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 60  # seconds

# ── Module-level singleton ────────────────────────────────────────────────────

_instance: Scheduler | None = None


def get_scheduler() -> Scheduler | None:
    """Return the running Scheduler instance, or None if not started."""
    return _instance


def set_scheduler(s: Scheduler | None) -> None:
    """Store (or clear) the running Scheduler instance."""
    global _instance
    _instance = s


# ── Internal helpers ──────────────────────────────────────────────────────────


def _load_schedules(path: Path) -> dict:
    """Load and parse schedules.yml.  Returns an empty dict on missing file."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for the scheduler.\n"
            "Install it with: uv add pyyaml  or  pip install pyyaml"
        ) from exc

    if not path.exists():
        logger.warning("schedules.yml not found at %s — no jobs scheduled.", path)
        return {}

    with open(path) as fh:
        data = yaml.safe_load(fh) or {}

    return data


def _send_webhook(url: str, payload: dict) -> None:
    """Fire-and-forget HTTP POST to *url* with *payload*."""
    try:
        resp = httpx.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.debug("Webhook delivered to %s", url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Webhook delivery failed: %s", exc)


def _run_job(
    *,
    job_id: str,
    source_id: str,
    params: dict[str, Any],
    auto_period: bool,
    webhook_url: str | None,
) -> tuple[bool, str | None]:
    """
    Execute one scheduled ingestion job with retry + webhook alert.

    Returns ``(success, error_message)``.  Called by the tracking wrapper
    registered with APScheduler and by Scheduler.trigger_job().
    """
    from eostrata.config import settings
    from eostrata.sources.base import get_source

    logger.info("[scheduler] Starting job '%s'", job_id)

    source_cls = get_source(source_id)
    source = source_cls()

    # Resolve auto_period → inject year / month from source.latest_available()
    job_params = dict(params)
    if auto_period:
        latest = source.latest_available()
        job_params.setdefault("year", latest.year)
        if source.temporal_resolution == "monthly":
            job_params.setdefault("month", latest.month)

    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            _execute_ingestion(source, job_params, settings)
            logger.info("[scheduler] Job '%s' succeeded (attempt %d)", job_id, attempt)
            return True, None
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "[scheduler] Job '%s' attempt %d/%d failed: %s — retrying in %ds",
                job_id,
                attempt,
                _MAX_RETRIES,
                exc,
                delay,
            )
            if attempt < _MAX_RETRIES:
                time.sleep(delay)

    # All retries exhausted
    error_msg = str(last_exc)
    logger.error(
        "[scheduler] Job '%s' failed after %d attempts: %s", job_id, _MAX_RETRIES, last_exc
    )
    if webhook_url:
        _send_webhook(
            webhook_url,
            {
                "event": "job_failed",
                "job_id": job_id,
                "source": source_id,
                "params": job_params,
                "error": error_msg,
                "timestamp": datetime.now(tz=UTC).isoformat(),
            },
        )
    return False, error_msg


def _execute_ingestion(source: Any, params: dict, settings: Any) -> None:
    """Run download → zarr → stac for one source + params combo."""
    from eostrata import catalog as cat

    bbox = settings.bbox

    # Download
    paths = source.download(settings.raw_dir, bbox, **params)
    if not paths:
        raise RuntimeError("Source.download() returned no paths.")
    path = paths[0]

    # Write to Zarr
    ds = source.to_zarr(path, settings.zarr_root, bbox, **params)

    zarr_group = source.zarr_group(**params)
    item_id = source.stac_item_id(**params)

    # Derive spatial bbox from written dataset
    x_dim = "x" if "x" in ds.coords else next(iter(ds.coords))
    y_dim = "y" if "y" in ds.coords else list(ds.coords)[1]
    item_bbox = (
        float(ds[x_dim].min()),
        float(ds[y_dim].min()),
        float(ds[x_dim].max()),
        float(ds[y_dim].max()),
    )

    # Register STAC item
    year = params.get("year", datetime.now(tz=UTC).year)
    month = params.get("month", 1)
    catalogue = cat.load_or_create(settings.catalog_path)
    cat.register_item(
        catalogue,
        collection_id=source.collection_id,
        item_id=item_id,
        bbox=item_bbox,
        datetime_=datetime(year, month, 1, tzinfo=UTC),
        zarr_root=settings.zarr_root,
        zarr_group=zarr_group,
        variable=getattr(source, "VARIABLE", params.get("variable", "data")),
        extra_properties=source.stac_properties(**params),
    )
    cat.save(catalogue, settings.catalog_path)
    logger.info("[scheduler] Ingestion complete: %s / %s", source.collection_id, item_id)


class Scheduler:
    """
    In-process APScheduler wrapper.

    Reads ``schedules.yml``, registers enabled cron jobs, and wires
    retry + webhook alert logic.

    Job definitions are kept in ``self._job_defs`` (dict keyed by job id)
    so that they can be listed, added, updated, and removed at runtime via
    the scheduler API without re-parsing the YAML file.
    """

    def __init__(self, schedules_path: Path | None = None) -> None:
        try:
            from apscheduler.schedulers.background import (
                BackgroundScheduler,  # type: ignore[import-untyped]
            )
        except ImportError as exc:
            raise ImportError(
                "APScheduler is required for the scheduler.\n"
                "Install it with: uv add apscheduler  or  pip install apscheduler"
            ) from exc

        self._scheduler = BackgroundScheduler()
        self._path = schedules_path or _SCHEDULES_FILE
        self._job_defs: dict[str, dict] = {}
        self._global_webhook: str | None = None
        self._job_runs: dict[str, dict] = {}  # job_id → last run record
        self._lock = threading.Lock()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _wrap_for_tracking(
        self,
        job_id: str,
        source_id: str,
        params: dict,
        auto_period: bool,
        webhook_url: str | None,
    ):
        """Return a zero-arg callable that runs the job and records its outcome."""

        def _tracked() -> None:
            started_at = datetime.now(tz=UTC).isoformat()
            with self._lock:
                self._job_runs[job_id] = {
                    "started_at": started_at,
                    "finished_at": None,
                    "status": "running",
                    "error": None,
                }
            success, error = _run_job(
                job_id=job_id,
                source_id=source_id,
                params=params,
                auto_period=auto_period,
                webhook_url=webhook_url,
            )
            with self._lock:
                self._job_runs[job_id] = {
                    "started_at": started_at,
                    "finished_at": datetime.now(tz=UTC).isoformat(),
                    "status": "success" if success else "failed",
                    "error": error,
                }

        return _tracked

    def _add_apscheduler_job(self, job_def: dict) -> None:
        """Register one job_def with APScheduler (does not touch _job_defs)."""
        job_id: str = job_def["id"]
        source_id: str = job_def["source"]
        params: dict = job_def.get("params") or {}
        cron_expr: str = job_def["cron"]
        auto_period: bool = job_def.get("auto_period", False)

        parts = cron_expr.split()
        if len(parts) != 5:
            logger.error(
                "[scheduler] Job '%s' has invalid cron '%s' — skipping.",
                job_id,
                cron_expr,
            )
            return

        minute, hour, day, month, day_of_week = parts

        tracked_fn = self._wrap_for_tracking(
            job_id, source_id, params, auto_period, self._global_webhook
        )

        self._scheduler.add_job(
            tracked_fn,
            trigger="cron",
            id=job_id,
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
            misfire_grace_time=3600,
            coalesce=True,
        )
        logger.info(
            "[scheduler] Registered job '%s' (source=%s, cron='%s')",
            job_id,
            source_id,
            cron_expr,
        )

    def _register_jobs(self) -> None:
        """Parse schedules.yml and add jobs to the APScheduler instance."""
        data = _load_schedules(self._path)
        self._global_webhook = data.get("webhook_url")
        jobs: list[dict] = data.get("jobs") or []

        for job_def in jobs:
            job_id = job_def.get("id")
            if not job_id:
                continue
            self._job_defs[job_id] = job_def
            if not job_def.get("enabled", True):
                logger.debug("[scheduler] Job '%s' is disabled — skipping.", job_id)
                continue
            self._add_apscheduler_job(job_def)

        if not self._job_defs:
            logger.info("[scheduler] No jobs found in %s.", self._path)

    def _write_schedules(self) -> None:
        """Persist current _job_defs to schedules.yml (overwrites comments)."""
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("[scheduler] PyYAML not installed — cannot persist schedules.")
            return

        data: dict = {"jobs": list(self._job_defs.values())}
        if self._global_webhook:
            data["webhook_url"] = self._global_webhook

        with open(self._path, "w") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)
        logger.debug("[scheduler] schedules.yml updated (%d jobs).", len(self._job_defs))

    # ── Public runtime API ────────────────────────────────────────────────────

    def get_jobs(self) -> list[dict]:
        """Return all job definitions enriched with APScheduler next_run_time and last_run."""
        result = []
        with self._lock:
            for job_id, job_def in self._job_defs.items():
                apjob = self._scheduler.get_job(job_id)
                next_run = (
                    apjob.next_run_time.isoformat()
                    if apjob and apjob.next_run_time
                    else None
                )
                last_run = self._job_runs.get(job_id)
                result.append({**job_def, "next_run_time": next_run, "last_run": last_run})
        return result

    def save_job(self, job_def: dict) -> None:
        """Add or update a job in memory, APScheduler, and schedules.yml."""
        job_id: str = job_def["id"]
        with self._lock:
            # Remove existing APScheduler entry if present
            with contextlib.suppress(Exception):
                self._scheduler.remove_job(job_id)

            self._job_defs[job_id] = job_def

            if job_def.get("enabled", True):
                self._add_apscheduler_job(job_def)

            self._write_schedules()

        logger.info("[scheduler] Job '%s' saved.", job_id)

    def remove_job(self, job_id: str) -> None:
        """Remove a job from memory, APScheduler, and schedules.yml."""
        with self._lock:
            with contextlib.suppress(Exception):
                self._scheduler.remove_job(job_id)

            self._job_defs.pop(job_id, None)
            self._write_schedules()

        logger.info("[scheduler] Job '%s' removed.", job_id)

    def trigger_job(self, job_id: str) -> None:
        """Run a job immediately in a background daemon thread."""
        with self._lock:
            job_def = self._job_defs.get(job_id)

        if job_def is None:
            raise KeyError(f"Job '{job_id}' not found.")

        tracked_fn = self._wrap_for_tracking(
            job_id=job_id,
            source_id=job_def["source"],
            params=job_def.get("params") or {},
            auto_period=job_def.get("auto_period", False),
            webhook_url=self._global_webhook,
        )
        t = threading.Thread(
            target=tracked_fn,
            daemon=True,
            name=f"scheduler-manual-{job_id}",
        )
        t.start()
        logger.info("[scheduler] Manual trigger for job '%s' started.", job_id)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Register all jobs and start the background scheduler."""
        self._register_jobs()
        self._scheduler.start()
        logger.info(
            "[scheduler] Started with %d job(s) from %s.",
            len(self._job_defs),
            self._path,
        )

    def stop(self) -> None:
        """Gracefully shut down the scheduler."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("[scheduler] Stopped.")
