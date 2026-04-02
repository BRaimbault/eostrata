"""Tests for the scheduler REST API router (/scheduler/jobs/*)."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Fixtures ──────────────────────────────────────────────────────────────────

_JOB_DEF = {
    "id": "chirps_monthly",
    "source": "chirps",
    "params": {},
    "cron": "0 3 15 * *",
    "auto_period": True,
    "enabled": True,
}

_JOB_WITH_RUNTIME = {
    **_JOB_DEF,
    "next_run_time": "2026-04-15T03:00:00+00:00",
    "last_run": None,
}


@pytest.fixture()
def mock_scheduler():
    s = MagicMock()
    s.get_jobs.return_value = [_JOB_WITH_RUNTIME]
    return s


@pytest.fixture()
def client(tmp_path, monkeypatch, mock_scheduler):
    monkeypatch.setenv("EOSTRATA_CATALOG_PATH", str(tmp_path / "catalog.json"))
    monkeypatch.setenv("EOSTRATA_ZARR_ROOT", str(tmp_path / "zarr"))

    import eostrata.config as cfg_mod
    importlib.reload(cfg_mod)

    from eostrata.server import app

    with patch("eostrata.scheduler._instance", mock_scheduler):
        yield TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def client_no_scheduler(tmp_path, monkeypatch):
    monkeypatch.setenv("EOSTRATA_CATALOG_PATH", str(tmp_path / "catalog.json"))
    monkeypatch.setenv("EOSTRATA_ZARR_ROOT", str(tmp_path / "zarr"))

    import eostrata.config as cfg_mod
    importlib.reload(cfg_mod)

    from eostrata.server import app

    with patch("eostrata.scheduler._instance", None):
        yield TestClient(app, raise_server_exceptions=False)


# ── GET /scheduler/jobs ───────────────────────────────────────────────────────


class TestListJobs:
    def test_returns_jobs(self, client, mock_scheduler):
        resp = client.get("/scheduler/jobs")
        assert resp.status_code == 200
        body = resp.json()
        assert "jobs" in body
        assert body["jobs"][0]["id"] == "chirps_monthly"

    def test_503_when_scheduler_not_running(self, client_no_scheduler):
        resp = client_no_scheduler.get("/scheduler/jobs")
        assert resp.status_code == 503


# ── POST /scheduler/jobs ──────────────────────────────────────────────────────


class TestCreateJob:
    def test_creates_new_job(self, client, mock_scheduler):
        mock_scheduler.get_jobs.return_value = []  # no existing jobs

        resp = client.post("/scheduler/jobs", json=_JOB_DEF)
        assert resp.status_code == 201
        mock_scheduler.save_job.assert_called_once()
        saved = mock_scheduler.save_job.call_args[0][0]
        assert saved["id"] == "chirps_monthly"

    def test_409_on_duplicate_id(self, client, mock_scheduler):
        # mock_scheduler.get_jobs already returns chirps_monthly
        resp = client.post("/scheduler/jobs", json=_JOB_DEF)
        assert resp.status_code == 409
        assert "chirps_monthly" in resp.json()["detail"]

    def test_422_on_invalid_source(self, client):
        resp = client.post("/scheduler/jobs", json={**_JOB_DEF, "id": "new", "source": "bogus"})
        assert resp.status_code == 422

    def test_422_on_bad_cron(self, client, mock_scheduler):
        mock_scheduler.get_jobs.return_value = []
        resp = client.post("/scheduler/jobs", json={**_JOB_DEF, "id": "new", "cron": "0 3 15"})
        assert resp.status_code == 422

    def test_503_when_scheduler_not_running(self, client_no_scheduler):
        resp = client_no_scheduler.post("/scheduler/jobs", json=_JOB_DEF)
        assert resp.status_code == 503


# ── PUT /scheduler/jobs/{job_id} ──────────────────────────────────────────────


class TestUpdateJob:
    def test_updates_existing_job(self, client, mock_scheduler):
        updated = {**_JOB_DEF, "cron": "0 4 15 * *"}
        resp = client.put("/scheduler/jobs/chirps_monthly", json=updated)
        assert resp.status_code == 200
        mock_scheduler.save_job.assert_called_once()
        saved = mock_scheduler.save_job.call_args[0][0]
        assert saved["id"] == "chirps_monthly"
        assert saved["cron"] == "0 4 15 * *"

    def test_path_id_overrides_body_id(self, client, mock_scheduler):
        """The job id in the URL path is authoritative."""
        resp = client.put("/scheduler/jobs/chirps_monthly", json={**_JOB_DEF, "id": "other_id"})
        assert resp.status_code == 200
        saved = mock_scheduler.save_job.call_args[0][0]
        assert saved["id"] == "chirps_monthly"

    def test_503_when_scheduler_not_running(self, client_no_scheduler):
        resp = client_no_scheduler.put("/scheduler/jobs/chirps_monthly", json=_JOB_DEF)
        assert resp.status_code == 503


# ── DELETE /scheduler/jobs/{job_id} ──────────────────────────────────────────


class TestDeleteJob:
    def test_deletes_job(self, client, mock_scheduler):
        resp = client.delete("/scheduler/jobs/chirps_monthly")
        assert resp.status_code == 204
        mock_scheduler.remove_job.assert_called_once_with("chirps_monthly")

    def test_503_when_scheduler_not_running(self, client_no_scheduler):
        resp = client_no_scheduler.delete("/scheduler/jobs/chirps_monthly")
        assert resp.status_code == 503


# ── POST /scheduler/jobs/{job_id}/run ─────────────────────────────────────────


class TestTriggerJob:
    def test_triggers_job(self, client, mock_scheduler):
        resp = client.post("/scheduler/jobs/chirps_monthly/run")
        assert resp.status_code == 202
        assert resp.json()["status"] == "triggered"
        mock_scheduler.trigger_job.assert_called_once_with("chirps_monthly")

    def test_404_when_job_not_found(self, client, mock_scheduler):
        mock_scheduler.trigger_job.side_effect = KeyError("unknown_job")
        resp = client.post("/scheduler/jobs/unknown_job/run")
        assert resp.status_code == 404

    def test_503_when_scheduler_not_running(self, client_no_scheduler):
        resp = client_no_scheduler.post("/scheduler/jobs/chirps_monthly/run")
        assert resp.status_code == 503
