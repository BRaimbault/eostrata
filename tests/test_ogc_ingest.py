"""Tests for the OGC ingest API (POST /processes/ingest/execution)
and the underlying ingestion service functions."""

from __future__ import annotations

from concurrent.futures import Future
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("EOSTRATA_CATALOG_PATH", str(tmp_path / "catalog.json"))
    monkeypatch.setenv("EOSTRATA_ZARR_ROOT", str(tmp_path / "zarr"))

    import importlib

    import eostrata.config as cfg_mod

    importlib.reload(cfg_mod)

    from eostrata.server import app

    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture(autouse=True)
def clear_job_store():
    import eostrata.jobs as job_mod

    with job_mod._lock:
        job_mod._store.clear()
    yield
    with job_mod._lock:
        job_mod._store.clear()


@pytest.fixture()
def sync_executor():
    """Run executor submissions synchronously in the test thread."""

    def _sync_submit(fn, *args, **kwargs):
        fn(*args, **kwargs)
        f = Future()
        f.set_result(None)
        return f

    with patch("eostrata.ogc.ingest._executor.submit", side_effect=_sync_submit):
        yield


# ── Process description ───────────────────────────────────────────────────────


class TestProcessDescription:
    def test_list_includes_ingest(self, client):
        data = client.get("/processes").json()
        ids = {p["id"] for p in data["processes"]}
        assert "ingest" in ids

    def test_describe_ingest(self, client):
        resp = client.get("/processes/ingest")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "ingest"
        assert "async-execute" in data["jobControlOptions"]
        assert "source" in data["inputs"]
        assert "iso3" in data["inputs"]
        assert "variable" in data["inputs"]
        assert "years" in data["inputs"]
        assert "months" in data["inputs"]


# ── Input validation ──────────────────────────────────────────────────────────


class TestInputValidation:
    def test_missing_source_returns_422(self, client):
        resp = client.post("/processes/ingest/execution", json={"inputs": {}})
        assert resp.status_code == 422

    def test_invalid_source_returns_422(self, client):
        resp = client.post("/processes/ingest/execution", json={"inputs": {"source": "sentinel"}})
        assert resp.status_code == 422

    def test_worldpop_without_iso3_returns_422(self, client):
        resp = client.post("/processes/ingest/execution", json={"inputs": {"source": "worldpop"}})
        assert resp.status_code == 422

    def test_worldpop_short_iso3_returns_422(self, client):
        resp = client.post(
            "/processes/ingest/execution", json={"inputs": {"source": "worldpop", "iso3": "NG"}}
        )
        assert resp.status_code == 422

    def test_invalid_month_returns_422(self, client):
        resp = client.post(
            "/processes/ingest/execution",
            json={"inputs": {"source": "chirps", "months": [0]}},
        )
        assert resp.status_code == 422

    def test_month_out_of_range_returns_422(self, client):
        resp = client.post(
            "/processes/ingest/execution",
            json={"inputs": {"source": "chirps", "months": [13]}},
        )
        assert resp.status_code == 422

    def test_months_all_string_is_accepted(self, client, sync_executor):
        """'ALL' string must be accepted and expanded to all 12 months."""
        with patch("eostrata.ingestion.run_ingest", return_value=([], True)) as mock_fn:
            resp = client.post(
                "/processes/ingest/execution",
                json={"inputs": {"source": "chirps", "months": "ALL"}},
            )
        assert resp.status_code == 201
        assert mock_fn.call_args.kwargs["months"] == list(range(1, 13))

    def test_cds_invalid_variable_returns_422(self, client):
        resp = client.post(
            "/processes/ingest/execution",
            json={"inputs": {"source": "cds", "variable": "xyz"}},
        )
        assert resp.status_code == 422


# ── WorldPop execution ────────────────────────────────────────────────────────


class TestWorldPopExecution:
    def test_returns_201_with_job_id(self, client, sync_executor):
        with patch("eostrata.ingestion.run_ingest", return_value=([], True)):
            resp = client.post(
                "/processes/ingest/execution",
                json={"inputs": {"source": "worldpop", "iso3": "NGA"}},
            )
        assert resp.status_code == 201
        data = resp.json()
        assert "jobID" in data
        assert any("/processes/jobs/" in link["href"] for link in data["links"])

    def test_success_path(self, client, sync_executor):
        with patch("eostrata.ingestion.run_ingest", return_value=([], True)) as mock_fn:
            resp = client.post(
                "/processes/ingest/execution",
                json={"inputs": {"source": "worldpop", "iso3": "NGA", "years": [2022]}},
            )
        job_id = resp.json()["jobID"]
        kw = mock_fn.call_args.kwargs
        assert kw["iso3"] == "NGA"
        assert kw["years"] == [2022]

        poll = client.get(f"/processes/jobs/{job_id}")
        assert poll.json()["status"] == "successful"

    def test_failure_path(self, client, sync_executor):
        with patch("eostrata.ingestion.run_ingest", side_effect=RuntimeError("network error")):
            resp = client.post(
                "/processes/ingest/execution",
                json={"inputs": {"source": "worldpop", "iso3": "NGA"}},
            )
        poll = client.get(f"/processes/jobs/{resp.json()['jobID']}")
        data = poll.json()
        assert data["status"] == "failed"
        assert "network error" in data["error"]

    def test_nothing_saved_marks_job_failed(self, client, sync_executor):
        """When saved=False (e.g. all 404), the job status must be 'failed', not 'successful'."""
        with patch("eostrata.ingestion.run_ingest", return_value=([], False)):
            resp = client.post(
                "/processes/ingest/execution",
                json={"inputs": {"source": "worldpop", "iso3": "MAU", "years": [2022]}},
            )
        job = client.get(f"/processes/jobs/{resp.json()['jobID']}").json()
        assert job["status"] == "failed"
        assert "unavailable" in job["error"]

    def test_nothing_saved_with_failures_marks_job_failed_with_details(self, client, sync_executor):
        """When saved=False and some periods failed, error message lists the failed periods."""
        with patch("eostrata.ingestion.run_ingest", return_value=(["NGA/2022"], False)):
            resp = client.post(
                "/processes/ingest/execution",
                json={"inputs": {"source": "worldpop", "iso3": "NGA", "years": [2022]}},
            )
        job = client.get(f"/processes/jobs/{resp.json()['jobID']}").json()
        assert job["status"] == "failed"
        assert "NGA/2022" in job["error"]

    def test_partial_save_marks_job_succeeded_with_message(self, client, sync_executor):
        """When saved=True but some periods failed, job is successful with a warning message."""
        with patch("eostrata.ingestion.run_ingest", return_value=(["NGA/2021"], True)):
            resp = client.post(
                "/processes/ingest/execution",
                json={"inputs": {"source": "worldpop", "iso3": "NGA", "years": [2021, 2022]}},
            )
        job = client.get(f"/processes/jobs/{resp.json()['jobID']}").json()
        assert job["status"] == "successful"
        assert "NGA/2021" in job["message"]

    def test_default_year_used_when_omitted(self, client, sync_executor):
        with patch("eostrata.ingestion.run_ingest", return_value=([], True)) as mock_fn:
            client.post(
                "/processes/ingest/execution",
                json={"inputs": {"source": "worldpop", "iso3": "NGA"}},
            )
        years = mock_fn.call_args.kwargs["years"]
        assert len(years) == 1 and isinstance(years[0], int)

    def test_iso3_uppercased_in_params(self, client, sync_executor):
        with patch("eostrata.ingestion.run_ingest", return_value=([], True)):
            resp = client.post(
                "/processes/ingest/execution",
                json={"inputs": {"source": "worldpop", "iso3": "nga"}},
            )
        job = client.get(f"/processes/jobs/{resp.json()['jobID']}").json()
        assert job["params"]["iso3"] == "NGA"


# ── CHIRPS execution ──────────────────────────────────────────────────────────


class TestCHIRPSExecution:
    def test_returns_201(self, client, sync_executor):
        with patch("eostrata.ingestion.run_ingest", return_value=([], True)):
            resp = client.post("/processes/ingest/execution", json={"inputs": {"source": "chirps"}})
        assert resp.status_code == 201

    def test_success_path(self, client, sync_executor):
        with patch("eostrata.ingestion.run_ingest", return_value=([], True)) as mock_fn:
            resp = client.post(
                "/processes/ingest/execution",
                json={"inputs": {"source": "chirps", "years": [2023], "months": [6]}},
            )
        kw = mock_fn.call_args.kwargs
        assert kw["years"] == [2023]
        assert kw["months"] == [6]
        assert (
            client.get(f"/processes/jobs/{resp.json()['jobID']}").json()["status"] == "successful"
        )

    def test_failure_path(self, client, sync_executor):
        with patch("eostrata.ingestion.run_ingest", side_effect=OSError("disk full")):
            resp = client.post("/processes/ingest/execution", json={"inputs": {"source": "chirps"}})
        assert client.get(f"/processes/jobs/{resp.json()['jobID']}").json()["status"] == "failed"


# ── CDS execution ─────────────────────────────────────────────────────────────


class TestCDSExecution:
    def test_returns_201(self, client, sync_executor):
        with patch("eostrata.ingestion.run_ingest", return_value=([], True)):
            resp = client.post(
                "/processes/ingest/execution",
                json={"inputs": {"source": "cds", "variable": "t2m"}},
            )
        assert resp.status_code == 201

    def test_success_path(self, client, sync_executor):
        with patch("eostrata.ingestion.run_ingest", return_value=([], True)) as mock_fn:
            resp = client.post(
                "/processes/ingest/execution",
                json={
                    "inputs": {"source": "cds", "variable": "tp", "years": [2023], "months": [1, 2]}
                },
            )
        kw = mock_fn.call_args.kwargs
        assert kw["variable"] == "tp"
        assert kw["years"] == [2023]
        assert kw["months"] == [1, 2]
        assert (
            client.get(f"/processes/jobs/{resp.json()['jobID']}").json()["status"] == "successful"
        )

    def test_failure_path(self, client, sync_executor):
        with patch("eostrata.ingestion.run_ingest", side_effect=RuntimeError("no credentials")):
            resp = client.post("/processes/ingest/execution", json={"inputs": {"source": "cds"}})
        assert client.get(f"/processes/jobs/{resp.json()['jobID']}").json()["status"] == "failed"

    def test_default_variable_is_t2m(self, client, sync_executor):
        with patch("eostrata.ingestion.run_ingest", return_value=([], True)) as mock_fn:
            client.post("/processes/ingest/execution", json={"inputs": {"source": "cds"}})
        assert mock_fn.call_args.kwargs["variable"] == "t2m"


# ── Job polling ───────────────────────────────────────────────────────────────


class TestJobPolling:
    def test_unknown_job_returns_404(self, client):
        assert client.get("/processes/jobs/doesnotexist").status_code == 404

    def test_list_jobs_empty(self, client):
        data = client.get("/processes/jobs").json()
        assert data["jobs"] == []
        assert any(link["rel"] == "self" for link in data["links"])

    def test_list_jobs_after_submissions(self, client, sync_executor):
        with patch("eostrata.ingestion.run_ingest", return_value=([], True)):
            client.post(
                "/processes/ingest/execution",
                json={"inputs": {"source": "worldpop", "iso3": "NGA"}},
            )
            client.post(
                "/processes/ingest/execution",
                json={"inputs": {"source": "worldpop", "iso3": "KEN"}},
            )
        assert len(client.get("/processes/jobs").json()["jobs"]) == 2

    def test_job_dict_has_expected_keys(self, client, sync_executor):
        with patch("eostrata.ingestion.run_ingest", return_value=([], True)):
            resp = client.post(
                "/processes/ingest/execution",
                json={"inputs": {"source": "worldpop", "iso3": "NGA"}},
            )
        job = client.get(f"/processes/jobs/{resp.json()['jobID']}").json()
        for key in ("jobID", "source", "params", "status", "created", "updated", "error"):
            assert key in job

    def test_params_stored_in_job(self, client, sync_executor):
        with patch("eostrata.ingestion.run_ingest", return_value=([], True)):
            resp = client.post(
                "/processes/ingest/execution",
                json={"inputs": {"source": "worldpop", "iso3": "ETH", "years": [2021]}},
            )
        job = client.get(f"/processes/jobs/{resp.json()['jobID']}").json()
        assert job["params"]["iso3"] == "ETH"
        assert job["params"]["years"] == [2021]


# ── Concurrency ───────────────────────────────────────────────────────────────


class TestConcurrency:
    def test_many_jobs_have_unique_ids(self, client, sync_executor):
        job_ids = []
        with patch("eostrata.ingestion.run_ingest", return_value=([], True)):
            for _ in range(10):
                resp = client.post(
                    "/processes/ingest/execution",
                    json={"inputs": {"source": "worldpop", "iso3": "NGA"}},
                )
                job_ids.append(resp.json()["jobID"])
        assert len(set(job_ids)) == 10


# ── ingestion.py unit tests ───────────────────────────────────────────────────


def _mock_ds(x=(0.0, 10.0), y=(0.0, 5.0)):
    ds = MagicMock()
    ds.x.min.return_value = x[0]
    ds.x.max.return_value = x[1]
    ds.y.min.return_value = y[0]
    ds.y.max.return_value = y[1]
    return ds


def _setup_source(
    stack, source_id: str, zarr_group_val: str, mock_ds, tmp_path, *, periods, **extra
):
    """Set up mocks for run_ingest tests.

    periods: list of (label, period_kwargs) tuples, same as iter_periods() would yield.
    """
    from datetime import UTC, datetime

    from eostrata.sources import base as src_base

    stack.enter_context(patch("eostrata.catalog.load_or_create", return_value=MagicMock()))
    mock_register = stack.enter_context(patch("eostrata.catalog.register_item"))
    mock_save = stack.enter_context(patch("eostrata.catalog.save"))
    stack.enter_context(patch("eostrata.cache.check_and_evict"))

    mock_source = MagicMock()
    mock_source.zarr_group.return_value = zarr_group_val
    mock_source.download.return_value = [tmp_path / "file.tif"]
    mock_source.to_zarr.return_value = mock_ds
    mock_source.extract_item_bbox.return_value = (0.0, 0.0, 10.0, 5.0)
    mock_source.stac_registrations.return_value = [
        {
            "item_id": zarr_group_val.replace("/", "_"),
            "datetime_": datetime(2020, 1, 1, tzinfo=UTC),
            "variable": extra.get("VARIABLE", "var"),
            "extra_properties": {},
        }
    ]
    for k, v in extra.items():
        setattr(mock_source, k, v)

    MockSourceCls = MagicMock()
    MockSourceCls.return_value = mock_source
    MockSourceCls.skip_404 = extra.get("skip_404", False)
    MockSourceCls.collection_id = extra.get("collection_id", source_id)
    MockSourceCls.iter_periods.return_value = periods

    stack.enter_context(patch.dict(src_base._REGISTRY, {source_id: MockSourceCls}))

    return mock_source, mock_register, mock_save


_BBOX = (0.0, 0.0, 10.0, 5.0)


class TestIngestionFunctions:
    def test_worldpop_ingest_calls_source_and_catalog(self, tmp_path):
        from eostrata import ingestion

        with ExitStack() as stack:
            src, mock_register, mock_save = _setup_source(
                stack,
                "worldpop",
                "worldpop/nga",
                _mock_ds(),
                tmp_path,
                periods=[("NGA/2022", {"iso3": "NGA", "year": 2022})],
                collection_id="worldpop",
                VARIABLE="population",
                skip_404=True,
            )
            ingestion.run_ingest(
                "worldpop",
                iso3="NGA",
                years=[2022],
                zarr_root=tmp_path / "zarr",
                raw_dir=tmp_path / "raw",
                catalog_path=tmp_path / "catalog.json",
                bbox=_BBOX,
            )

        src.download.assert_called_once_with(tmp_path / "raw", _BBOX, iso3="NGA", year=2022)
        src.to_zarr.assert_called_once()
        mock_register.assert_called_once()
        mock_save.assert_called_once()

    def test_worldpop_ingest_multiple_years(self, tmp_path):
        from eostrata import ingestion

        with ExitStack() as stack:
            src, mock_register, _ = _setup_source(
                stack,
                "worldpop",
                "worldpop/nga",
                _mock_ds(),
                tmp_path,
                periods=[
                    ("NGA/2021", {"iso3": "NGA", "year": 2021}),
                    ("NGA/2022", {"iso3": "NGA", "year": 2022}),
                    ("NGA/2023", {"iso3": "NGA", "year": 2023}),
                ],
                collection_id="worldpop",
                VARIABLE="population",
                skip_404=True,
            )
            ingestion.run_ingest(
                "worldpop",
                iso3="NGA",
                years=[2021, 2022, 2023],
                zarr_root=tmp_path / "zarr",
                raw_dir=tmp_path / "raw",
                catalog_path=tmp_path / "catalog.json",
                bbox=_BBOX,
            )

        assert src.download.call_count == 3
        assert mock_register.call_count == 3

    def test_chirps_ingest_calls_source_and_catalog(self, tmp_path):
        from eostrata import ingestion

        with ExitStack() as stack:
            src, mock_register, mock_save = _setup_source(
                stack,
                "chirps",
                "chirps/global",
                _mock_ds(),
                tmp_path,
                periods=[
                    ("2023-06", {"year": 2023, "month": 6}),
                    ("2023-07", {"year": 2023, "month": 7}),
                ],
                collection_id="chirps",
                VARIABLE="precipitation",
                skip_404=True,
            )
            ingestion.run_ingest(
                "chirps",
                years=[2023],
                months=[6, 7],
                zarr_root=tmp_path / "zarr",
                raw_dir=tmp_path / "raw",
                catalog_path=tmp_path / "catalog.json",
                bbox=_BBOX,
            )

        assert src.download.call_count == 2
        assert mock_register.call_count == 2
        mock_save.assert_called_once()

    def test_chirps_ingest_no_save_on_failure(self, tmp_path):
        """When all downloads fail, nothing is saved and the failed periods are returned."""
        from eostrata import ingestion

        with ExitStack() as stack:
            src, _, mock_save = _setup_source(
                stack,
                "chirps",
                "chirps/global",
                _mock_ds(),
                tmp_path,
                periods=[("2023-06", {"year": 2023, "month": 6})],
                collection_id="chirps",
                VARIABLE="precipitation",
            )
            src.download.side_effect = RuntimeError("fail")
            failed, saved = ingestion.run_ingest(
                "chirps",
                years=[2023],
                months=[6],
                zarr_root=tmp_path / "zarr",
                raw_dir=tmp_path / "raw",
                catalog_path=tmp_path / "catalog.json",
                bbox=_BBOX,
            )

        assert failed == ["2023-06"]
        assert not saved
        mock_save.assert_not_called()

    def test_chirps_ingest_skips_404_months(self, tmp_path):
        """A 404 for a specific month should be skipped, not abort the job."""
        import httpx

        from eostrata import ingestion

        not_found = httpx.HTTPStatusError(
            "404",
            request=MagicMock(),
            response=MagicMock(status_code=404),
        )

        with ExitStack() as stack:
            src, mock_register, mock_save = _setup_source(
                stack,
                "chirps",
                "chirps/global",
                _mock_ds(),
                tmp_path,
                periods=[
                    ("2023-06", {"year": 2023, "month": 6}),
                    ("2023-07", {"year": 2023, "month": 7}),
                ],
                collection_id="chirps",
                VARIABLE="precipitation",
                skip_404=True,
            )
            # month 6 succeeds, month 7 returns 404
            src.download.side_effect = [
                [tmp_path / "raw" / "chirps" / "chirps-v2.0.2023.06.tif"],
                not_found,
            ]
            ingestion.run_ingest(
                "chirps",
                years=[2023],
                months=[6, 7],
                zarr_root=tmp_path / "zarr",
                raw_dir=tmp_path / "raw",
                catalog_path=tmp_path / "catalog.json",
                bbox=_BBOX,
            )

        # Only month 6 registered; 404 month silently skipped
        assert mock_register.call_count == 1
        mock_save.assert_called_once()

    def test_chirps_ingest_non_404_http_errors_collected(self, tmp_path):
        """Non-404 HTTP errors are collected in the failed list, not re-raised."""
        import httpx

        from eostrata import ingestion

        server_error = httpx.HTTPStatusError(
            "500",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )

        with ExitStack() as stack:
            src, _mock_register, mock_save = _setup_source(
                stack,
                "chirps",
                "chirps/global",
                _mock_ds(),
                tmp_path,
                periods=[("2023-06", {"year": 2023, "month": 6})],
                collection_id="chirps",
                VARIABLE="precipitation",
            )
            src.download.side_effect = server_error
            failed, saved = ingestion.run_ingest(
                "chirps",
                years=[2023],
                months=[6],
                zarr_root=tmp_path / "zarr",
                raw_dir=tmp_path / "raw",
                catalog_path=tmp_path / "catalog.json",
                bbox=_BBOX,
            )

        assert failed == ["2023-06"]
        assert not saved
        mock_save.assert_not_called()

    def test_cds_ingest_calls_source_and_catalog(self, tmp_path):
        from datetime import UTC, datetime

        from eostrata import ingestion

        with ExitStack() as stack:
            src, mock_register, mock_save = _setup_source(
                stack,
                "cds",
                "era5/t2m",
                _mock_ds(),
                tmp_path,
                periods=[("t2m/2023", {"variable": "t2m", "year": 2023, "months": [1, 2]})],
                collection_id="cds",
            )
            # CDS stac_registrations returns one item per month
            src.stac_registrations.return_value = [
                {
                    "item_id": "era5_t2m",
                    "datetime_": datetime(2023, 1, 1, tzinfo=UTC),
                    "variable": "t2m",
                    "extra_properties": {},
                },
                {
                    "item_id": "era5_t2m",
                    "datetime_": datetime(2023, 2, 1, tzinfo=UTC),
                    "variable": "t2m",
                    "extra_properties": {},
                },
            ]
            ingestion.run_ingest(
                "cds",
                variable="t2m",
                years=[2023],
                months=[1, 2],
                zarr_root=tmp_path / "zarr",
                raw_dir=tmp_path / "raw",
                catalog_path=tmp_path / "catalog.json",
                bbox=_BBOX,
            )

        src.download.assert_called_once_with(
            tmp_path / "raw", _BBOX, variable="t2m", year=2023, months=[1, 2]
        )
        assert mock_register.call_count == 2
        mock_save.assert_called_once()

    def test_worldpop_ingest_skips_404_years(self, tmp_path):
        """A 404 for a WorldPop year is silently skipped (not added to failed)."""
        import httpx

        from eostrata import ingestion

        not_found = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock(status_code=404)
        )
        with ExitStack() as stack:
            src, mock_register, mock_save = _setup_source(
                stack,
                "worldpop",
                "worldpop/nga",
                _mock_ds(),
                tmp_path,
                periods=[("NGA/2099", {"iso3": "NGA", "year": 2099})],
                collection_id="worldpop",
                VARIABLE="population",
                skip_404=True,
            )
            src.download.side_effect = not_found
            failed, saved = ingestion.run_ingest(
                "worldpop",
                iso3="NGA",
                years=[2099],
                zarr_root=tmp_path / "zarr",
                raw_dir=tmp_path / "raw",
                catalog_path=tmp_path / "catalog.json",
                bbox=_BBOX,
            )

        assert failed == []  # 404 is a skip, not a failure
        assert not saved
        mock_save.assert_not_called()

    def test_worldpop_ingest_non_404_http_error_collected(self, tmp_path):
        """A non-404 HTTP error for a WorldPop year is added to the failed list."""
        import httpx

        from eostrata import ingestion

        server_error = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock(status_code=500)
        )
        with ExitStack() as stack:
            src, mock_register, mock_save = _setup_source(
                stack,
                "worldpop",
                "worldpop/nga",
                _mock_ds(),
                tmp_path,
                periods=[("NGA/2023", {"iso3": "NGA", "year": 2023})],
                collection_id="worldpop",
                VARIABLE="population",
                skip_404=True,
            )
            src.download.side_effect = server_error
            failed, saved = ingestion.run_ingest(
                "worldpop",
                iso3="NGA",
                years=[2023],
                zarr_root=tmp_path / "zarr",
                raw_dir=tmp_path / "raw",
                catalog_path=tmp_path / "catalog.json",
                bbox=_BBOX,
            )

        assert failed == ["NGA/2023"]
        assert not saved
        mock_save.assert_not_called()

    def test_cds_ingest_uses_longitude_fallback(self, tmp_path):
        """CDS extract_item_bbox falls back to longitude/latitude when x/y are absent."""
        from eostrata import ingestion
        from eostrata.sources.cds import CDSSource

        mock_ds = MagicMock()
        mock_ds.coords.__contains__ = lambda self, key: key not in ("x", "y")

        with ExitStack() as stack:
            src, _, mock_save = _setup_source(
                stack,
                "cds",
                "era5/t2m",
                mock_ds,
                tmp_path,
                periods=[("t2m/2023", {"variable": "t2m", "year": 2023, "months": [6]})],
                collection_id="cds",
            )
            # Override extract_item_bbox to call the real CDS implementation
            src.extract_item_bbox.side_effect = lambda ds: CDSSource().extract_item_bbox(ds)
            ingestion.run_ingest(
                "cds",
                variable="t2m",
                years=[2023],
                months=[6],
                zarr_root=tmp_path / "zarr",
                raw_dir=tmp_path / "raw",
                catalog_path=tmp_path / "catalog.json",
                bbox=_BBOX,
            )

        mock_save.assert_called_once()


# ── Rebuild-catalog process description ───────────────────────────────────────


class TestRebuildCatalogDescription:
    def test_list_includes_rebuild_catalog(self, client):
        data = client.get("/processes").json()
        ids = {p["id"] for p in data["processes"]}
        assert "rebuild-catalog" in ids

    def test_describe_rebuild_catalog(self, client):
        resp = client.get("/processes/rebuild-catalog")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "rebuild-catalog"
        assert "sync-execute" in data["jobControlOptions"]


# ── rebuild_catalog_from_zarr unit tests ──────────────────────────────────────

_TS_2022 = np.array(["2022-01-01"], dtype="datetime64[ns]")
_TS_MONTHLY = np.array(["2023-01-01", "2023-02-01", "2023-03-01"], dtype="datetime64[ns]")


def _mock_zarr_ds(times, x_range=(10.0, 20.0), y_range=(0.0, 10.0)):
    """Return a lightweight mock xarray Dataset for rebuild-catalog tests."""
    ds = MagicMock()
    ds.time.values = times
    ds.x.min.return_value = x_range[0]
    ds.x.max.return_value = x_range[1]
    ds.y.min.return_value = y_range[0]
    ds.y.max.return_value = y_range[1]
    return ds


class TestRebuildCatalogFromZarr:
    def test_missing_zarr_root_returns_empty(self, tmp_path):
        from eostrata.ingestion import rebuild_catalog_from_zarr

        results = rebuild_catalog_from_zarr(
            zarr_root=tmp_path / "nonexistent",
            catalog_path=tmp_path / "catalog.json",
        )
        assert results == {}

    def test_worldpop_group_registered(self, tmp_path):
        from eostrata.ingestion import rebuild_catalog_from_zarr

        zarr_root = tmp_path / "zarr"
        zarr_root.mkdir()
        catalog_path = tmp_path / "catalog.json"
        mock_ds = _mock_zarr_ds(_TS_2022)

        with (
            patch("eostrata.cache.list_groups", return_value=[("worldpop/nga", 10.0, 0.0)]),
            patch("xarray.open_zarr", return_value=mock_ds),
            patch("eostrata.catalog.save") as mock_save,
        ):
            results = rebuild_catalog_from_zarr(zarr_root=zarr_root, catalog_path=catalog_path)

        assert results == {"worldpop/nga": 1}
        mock_save.assert_called_once()

    def test_chirps_group_registered(self, tmp_path):
        from eostrata.ingestion import rebuild_catalog_from_zarr

        zarr_root = tmp_path / "zarr"
        zarr_root.mkdir()
        catalog_path = tmp_path / "catalog.json"
        mock_ds = _mock_zarr_ds(_TS_MONTHLY)

        with (
            patch("eostrata.cache.list_groups", return_value=[("chirps/global", 5.0, 0.0)]),
            patch("xarray.open_zarr", return_value=mock_ds),
            patch("eostrata.catalog.save"),
        ):
            results = rebuild_catalog_from_zarr(zarr_root=zarr_root, catalog_path=catalog_path)

        assert results == {"chirps/global": 3}

    def test_era5_group_registered(self, tmp_path):
        from eostrata.ingestion import rebuild_catalog_from_zarr

        zarr_root = tmp_path / "zarr"
        zarr_root.mkdir()
        catalog_path = tmp_path / "catalog.json"
        mock_ds = _mock_zarr_ds(_TS_MONTHLY)

        with (
            patch("eostrata.cache.list_groups", return_value=[("era5/t2m", 8.0, 0.0)]),
            patch("xarray.open_zarr", return_value=mock_ds),
            patch("eostrata.catalog.save"),
        ):
            results = rebuild_catalog_from_zarr(zarr_root=zarr_root, catalog_path=catalog_path)

        assert results == {"era5/t2m": 3}

    def test_invalid_group_path_depth_skipped(self, tmp_path):
        """Group paths with != 2 parts are skipped with a warning (lines 187-188)."""
        from eostrata.ingestion import rebuild_catalog_from_zarr

        zarr_root = tmp_path / "zarr"
        zarr_root.mkdir()
        catalog_path = tmp_path / "catalog.json"

        with (
            patch(
                "eostrata.cache.list_groups",
                return_value=[("singlepart", 5.0, 0.0), ("a/b/c", 3.0, 0.0)],
            ),
            patch("eostrata.catalog.save"),
        ):
            results = rebuild_catalog_from_zarr(zarr_root=zarr_root, catalog_path=catalog_path)

        assert results == {}

    def test_unknown_source_type_skipped(self, tmp_path):
        from eostrata.ingestion import rebuild_catalog_from_zarr

        zarr_root = tmp_path / "zarr"
        zarr_root.mkdir()
        catalog_path = tmp_path / "catalog.json"
        mock_ds = _mock_zarr_ds(_TS_2022)

        with (
            patch(
                "eostrata.cache.list_groups",
                return_value=[("sentinel2/tile", 5.0, 0.0)],
            ),
            patch("xarray.open_zarr", return_value=mock_ds),
            patch("eostrata.catalog.save"),
        ):
            results = rebuild_catalog_from_zarr(zarr_root=zarr_root, catalog_path=catalog_path)

        assert results == {}

    def test_multiple_groups_combined(self, tmp_path):
        from eostrata.ingestion import rebuild_catalog_from_zarr

        zarr_root = tmp_path / "zarr"
        zarr_root.mkdir()
        catalog_path = tmp_path / "catalog.json"

        def _open_zarr(store, group, **kwargs):
            if "worldpop" in group:
                return _mock_zarr_ds(_TS_2022)
            return _mock_zarr_ds(_TS_MONTHLY)

        groups = [
            ("worldpop/nga", 10.0, 0.0),
            ("chirps/global", 5.0, 0.0),
            ("era5/t2m", 8.0, 0.0),
        ]
        with (
            patch("eostrata.cache.list_groups", return_value=groups),
            patch("xarray.open_zarr", side_effect=_open_zarr),
            patch("eostrata.catalog.save"),
        ):
            results = rebuild_catalog_from_zarr(zarr_root=zarr_root, catalog_path=catalog_path)

        assert results == {"worldpop/nga": 1, "chirps/global": 3, "era5/t2m": 3}

    def test_bad_group_coords_skipped(self, tmp_path):
        from eostrata.ingestion import rebuild_catalog_from_zarr

        zarr_root = tmp_path / "zarr"
        zarr_root.mkdir()
        catalog_path = tmp_path / "catalog.json"

        bad_ds = MagicMock()
        bad_ds.time.values = _TS_2022
        bad_ds.x.min.side_effect = KeyError("x")

        with (
            patch("eostrata.cache.list_groups", return_value=[("worldpop/nga", 10.0, 0.0)]),
            patch("xarray.open_zarr", return_value=bad_ds),
            patch("eostrata.catalog.save"),
        ):
            results = rebuild_catalog_from_zarr(zarr_root=zarr_root, catalog_path=catalog_path)

        assert results == {}

    def test_sentinel_ndvi_group_registered(self, tmp_path):
        from eostrata.ingestion import rebuild_catalog_from_zarr

        zarr_root = tmp_path / "zarr"
        zarr_root.mkdir()
        catalog_path = tmp_path / "catalog.json"
        mock_ds = _mock_zarr_ds(_TS_MONTHLY)

        with (
            patch(
                "eostrata.cache.list_groups",
                return_value=[("sentinel_ndvi/global", 5.0, 0.0)],
            ),
            patch("xarray.open_zarr", return_value=mock_ds),
            patch("eostrata.catalog.save"),
        ):
            results = rebuild_catalog_from_zarr(zarr_root=zarr_root, catalog_path=catalog_path)

        assert results == {"sentinel_ndvi/global": 3}


# ── Sentinel NDVI OGC execution ───────────────────────────────────────────────


class TestSentinelNDVIExecution:
    def test_returns_201_with_job_id(self, client, sync_executor):
        with patch("eostrata.ingestion.run_ingest", return_value=([], True)):
            resp = client.post(
                "/processes/ingest/execution",
                json={
                    "inputs": {
                        "source": "sentinel_ndvi",
                        "years": [2024],
                        "months": [1],
                        "dekads": [1],
                    }
                },
            )
        assert resp.status_code == 201
        assert "jobID" in resp.json()

    def test_success_path_calls_ingest(self, client, sync_executor):
        with patch("eostrata.ingestion.run_ingest", return_value=([], True)) as mock_fn:
            client.post(
                "/processes/ingest/execution",
                json={
                    "inputs": {
                        "source": "sentinel_ndvi",
                        "years": [2024],
                        "months": [3],
                        "dekads": [2],
                    }
                },
            )
        kw = mock_fn.call_args.kwargs
        assert kw["years"] == [2024]
        assert kw["months"] == [3]
        assert kw["dekads"] == [2]

    def test_dekads_all_string_expands(self, client, sync_executor):
        with patch("eostrata.ingestion.run_ingest", return_value=([], True)) as mock_fn:
            client.post(
                "/processes/ingest/execution",
                json={"inputs": {"source": "sentinel_ndvi", "dekads": "ALL"}},
            )
        assert mock_fn.call_args.kwargs["dekads"] == [1, 2, 3]

    def test_days_all_string_expands(self, client, sync_executor):
        with patch("eostrata.ingestion.run_ingest", return_value=([], True)) as mock_fn:
            client.post(
                "/processes/ingest/execution",
                json={"inputs": {"source": "sentinel_ndvi", "days": "ALL"}},
            )
        assert mock_fn.call_args.kwargs.get("days") in (None, list(range(1, 32)))

    def test_days_list_passthrough(self, client, sync_executor):
        with patch("eostrata.ingestion.run_ingest", return_value=([], True)):
            resp = client.post(
                "/processes/ingest/execution",
                json={"inputs": {"source": "sentinel_ndvi", "days": [1, 5]}},
            )
        assert resp.status_code == 201

    def test_failure_path_marks_job_failed(self, client, sync_executor):
        with patch(
            "eostrata.ingestion.run_ingest",
            side_effect=RuntimeError("network error"),
        ):
            resp = client.post(
                "/processes/ingest/execution",
                json={"inputs": {"source": "sentinel_ndvi"}},
            )
        job = client.get(f"/processes/jobs/{resp.json()['jobID']}").json()
        assert job["status"] == "failed"


# ── Sentinel NDVI ingestion unit tests ────────────────────────────────────────


class TestSentinelNDVIIngestion:
    def test_ingest_calls_source_and_catalog(self, tmp_path):
        from contextlib import ExitStack

        from eostrata import ingestion

        mock_ds = _mock_ds()

        with ExitStack() as stack:
            src, mock_register, mock_save = _setup_source(
                stack,
                "sentinel_ndvi",
                "sentinel_ndvi/global",
                mock_ds,
                tmp_path,
                periods=[("2024-03-d1", {"year": 2024, "month": 3, "dekad": 1})],
                collection_id="sentinel_ndvi",
                VARIABLE="ndvi",
            )
            ingestion.run_ingest(
                "sentinel_ndvi",
                years=[2024],
                months=[3],
                dekads=[1],
                zarr_root=tmp_path / "zarr",
                raw_dir=tmp_path / "raw",
                catalog_path=tmp_path / "catalog.json",
                bbox=_BBOX,
            )

        src.download.assert_called_once()
        src.to_zarr.assert_called_once()
        mock_register.assert_called_once()
        mock_save.assert_called_once()

    def test_ingest_multiple_dekads(self, tmp_path):
        from contextlib import ExitStack

        from eostrata import ingestion

        with ExitStack() as stack:
            src, mock_register, _ = _setup_source(
                stack,
                "sentinel_ndvi",
                "sentinel_ndvi/global",
                _mock_ds(),
                tmp_path,
                periods=[
                    ("2024-01-d1", {"year": 2024, "month": 1, "dekad": 1}),
                    ("2024-01-d2", {"year": 2024, "month": 1, "dekad": 2}),
                    ("2024-01-d3", {"year": 2024, "month": 1, "dekad": 3}),
                ],
                collection_id="sentinel_ndvi",
                VARIABLE="ndvi",
            )
            ingestion.run_ingest(
                "sentinel_ndvi",
                years=[2024],
                months=[1],
                dekads=[1, 2, 3],
                zarr_root=tmp_path / "zarr",
                raw_dir=tmp_path / "raw",
                catalog_path=tmp_path / "catalog.json",
                bbox=_BBOX,
            )

        assert src.download.call_count == 3
        assert mock_register.call_count == 3

    def test_ingest_download_failure_collected(self, tmp_path):
        from contextlib import ExitStack

        from eostrata import ingestion

        with ExitStack() as stack:
            src, _, mock_save = _setup_source(
                stack,
                "sentinel_ndvi",
                "sentinel_ndvi/global",
                _mock_ds(),
                tmp_path,
                periods=[("2024-01-d1", {"year": 2024, "month": 1, "dekad": 1})],
                collection_id="sentinel_ndvi",
                VARIABLE="ndvi",
            )
            src.download.side_effect = RuntimeError("network error")
            failed, saved = ingestion.run_ingest(
                "sentinel_ndvi",
                years=[2024],
                months=[1],
                dekads=[1],
                zarr_root=tmp_path / "zarr",
                raw_dir=tmp_path / "raw",
                catalog_path=tmp_path / "catalog.json",
                bbox=_BBOX,
            )

        assert failed == ["2024-01-d1"]
        assert not saved
        mock_save.assert_not_called()

    def test_ingest_zarr_failure_collected(self, tmp_path):
        from contextlib import ExitStack

        from eostrata import ingestion

        with ExitStack() as stack:
            src, _, mock_save = _setup_source(
                stack,
                "sentinel_ndvi",
                "sentinel_ndvi/global",
                _mock_ds(),
                tmp_path,
                periods=[("2024-01-d1", {"year": 2024, "month": 1, "dekad": 1})],
                collection_id="sentinel_ndvi",
                VARIABLE="ndvi",
            )
            src.to_zarr.side_effect = RuntimeError("zarr error")
            failed, saved = ingestion.run_ingest(
                "sentinel_ndvi",
                years=[2024],
                months=[1],
                dekads=[1],
                zarr_root=tmp_path / "zarr",
                raw_dir=tmp_path / "raw",
                catalog_path=tmp_path / "catalog.json",
                bbox=_BBOX,
            )

        assert failed == ["2024-01-d1"]
        assert not saved
        mock_save.assert_not_called()

    def test_zarr_open_error_skipped(self, tmp_path):
        from eostrata.ingestion import rebuild_catalog_from_zarr

        zarr_root = tmp_path / "zarr"
        zarr_root.mkdir()
        catalog_path = tmp_path / "catalog.json"

        with (
            patch("eostrata.cache.list_groups", return_value=[("worldpop/nga", 10.0, 0.0)]),
            patch("xarray.open_zarr", side_effect=OSError("corrupt")),
            patch("eostrata.catalog.save"),
        ):
            results = rebuild_catalog_from_zarr(zarr_root=zarr_root, catalog_path=catalog_path)

        assert results == {}


# ── Rebuild-catalog API endpoint ─────────────────────────────────────────────


class TestRebuildCatalogAPI:
    def test_post_rebuild_catalog_empty_store(self, client, tmp_path):
        resp = client.post("/processes/rebuild-catalog/execution")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "successful"

    def test_post_rebuild_catalog_with_groups(self, client):
        mock_results = {"worldpop/nga": 2, "chirps/global": 6}
        with patch("eostrata.ingestion.rebuild_catalog_from_zarr", return_value=mock_results):
            resp = client.post("/processes/rebuild-catalog/execution")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "successful"
        assert data["groups"] == mock_results
        assert data["total_timestamps"] == 8
