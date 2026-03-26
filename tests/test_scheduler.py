"""Tests for the scheduler module."""

from __future__ import annotations

import textwrap
from datetime import UTC
from unittest.mock import MagicMock, patch

import pytest

from eostrata.scheduler import _load_schedules, _send_webhook


class TestLoadSchedules:
    def test_yaml_not_installed_raises(self, tmp_path):
        """_load_schedules raises ImportError when PyYAML is not installed."""
        import sys

        f = tmp_path / "schedules.yml"
        f.write_text("jobs: []\n")
        with patch.dict(sys.modules, {"yaml": None}):
            with pytest.raises(ImportError, match="PyYAML"):
                _load_schedules(f)

    def test_missing_file_returns_empty(self, tmp_path):
        result = _load_schedules(tmp_path / "nonexistent.yml")
        assert result == {}

    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "schedules.yml"
        f.write_text("")
        assert _load_schedules(f) == {}

    def test_parses_jobs(self, tmp_path):
        f = tmp_path / "schedules.yml"
        f.write_text(
            textwrap.dedent("""\
            webhook_url: https://example.com/alert
            jobs:
              - id: test_job
                source: worldpop
                params:
                  iso3: NGA
                cron: "0 2 1 2 *"
                auto_period: true
                enabled: true
        """)
        )
        data = _load_schedules(f)
        assert data["webhook_url"] == "https://example.com/alert"
        assert len(data["jobs"]) == 1
        assert data["jobs"][0]["id"] == "test_job"

    def test_disabled_jobs_parsed_but_not_omitted(self, tmp_path):
        """_load_schedules returns everything; Scheduler.start() filters enabled."""
        f = tmp_path / "schedules.yml"
        f.write_text(
            textwrap.dedent("""\
            jobs:
              - id: disabled_job
                source: chirps
                params: {}
                cron: "0 3 15 * *"
                enabled: false
        """)
        )
        data = _load_schedules(f)
        assert data["jobs"][0]["enabled"] is False


class TestSendWebhook:
    def test_posts_payload(self):
        with patch("httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            _send_webhook("https://example.com/alert", {"event": "test"})
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            assert call_kwargs[0][0] == "https://example.com/alert"

    def test_swallows_connection_error(self):
        with patch("httpx.post", side_effect=Exception("connection refused")):
            # Should not raise
            _send_webhook("https://example.com/alert", {"event": "test"})


class TestRunJob:
    """Tests for the _run_job orchestration function."""

    def _make_mock_source(self, tmp_path):
        """Return a mock source that writes a tiny real Zarr."""
        from datetime import datetime
        from unittest.mock import MagicMock

        import numpy as np
        import xarray as xr

        zarr_root = tmp_path / "zarr"
        zarr_root.mkdir(parents=True, exist_ok=True)

        ds = xr.Dataset(
            {"population": (("y", "x"), np.ones((4, 4), dtype="float32"))},
            coords={"y": np.arange(4.0), "x": np.arange(4.0)},
        )
        ds.to_zarr(str(zarr_root), group="worldpop/nga", mode="w")

        src = MagicMock()
        src.collection_id = "worldpop"
        src.temporal_resolution = "annual"
        src.VARIABLE = "population"
        src.latest_available.return_value = datetime(2020, 1, 1, tzinfo=UTC)
        src.download.return_value = [tmp_path / "dummy.tif"]
        src.to_zarr.return_value = ds
        src.zarr_group.return_value = "worldpop/nga"
        src.stac_item_id.return_value = "worldpop_nga"
        src.stac_properties.return_value = {}
        return src, zarr_root

    def test_succeeds_on_first_attempt(self, tmp_path):
        src, zarr_root = self._make_mock_source(tmp_path)
        from unittest.mock import MagicMock, patch

        from eostrata.scheduler import _run_job

        mock_settings = MagicMock()
        mock_settings.bbox = (2.0, 4.0, 15.0, 14.0)
        mock_settings.raw_dir = tmp_path / "raw"
        mock_settings.zarr_root = zarr_root
        mock_settings.catalog_path = tmp_path / "catalog.json"

        with (
            patch("eostrata.sources.base.get_source", return_value=lambda: src),
            patch("eostrata.config.settings", mock_settings),
            patch("eostrata.catalog.load_or_create", return_value=MagicMock()),
            patch("eostrata.catalog.register_item"),
            patch("eostrata.catalog.save"),
        ):
            _run_job(
                job_id="test",
                source_id="worldpop",
                params={"iso3": "NGA", "year": 2020},
                auto_period=False,
                webhook_url=None,
            )

    def test_auto_period_injects_month_for_monthly_source(self, tmp_path):
        """auto_period=True with a monthly source injects both year and month."""
        src, zarr_root = self._make_mock_source(tmp_path)
        from datetime import datetime
        from unittest.mock import MagicMock, patch

        from eostrata.scheduler import _run_job

        src.temporal_resolution = "monthly"
        src.latest_available.return_value = datetime(2023, 6, 1, tzinfo=UTC)

        mock_settings = MagicMock()
        mock_settings.bbox = (0.0, 0.0, 10.0, 10.0)
        mock_settings.raw_dir = tmp_path / "raw"
        mock_settings.zarr_root = zarr_root
        mock_settings.catalog_path = tmp_path / "catalog.json"

        with (
            patch("eostrata.sources.base.get_source", return_value=lambda: src),
            patch("eostrata.config.settings", mock_settings),
            patch("eostrata.catalog.load_or_create", return_value=MagicMock()),
            patch("eostrata.catalog.register_item"),
            patch("eostrata.catalog.save"),
        ):
            _run_job(
                job_id="test",
                source_id="worldpop",
                params={},
                auto_period=True,
                webhook_url=None,
            )

        call_kwargs = src.download.call_args[1]
        assert call_kwargs.get("month") == 6

    def test_auto_period_injects_year(self, tmp_path):
        """auto_period=True should resolve year from source.latest_available()."""
        src, zarr_root = self._make_mock_source(tmp_path)
        from datetime import datetime
        from unittest.mock import MagicMock, patch

        from eostrata.scheduler import _run_job

        src.latest_available.return_value = datetime(2023, 1, 1, tzinfo=UTC)

        mock_settings = MagicMock()
        mock_settings.bbox = (0.0, 0.0, 10.0, 10.0)
        mock_settings.raw_dir = tmp_path / "raw"
        mock_settings.zarr_root = zarr_root
        mock_settings.catalog_path = tmp_path / "catalog.json"

        with (
            patch("eostrata.sources.base.get_source", return_value=lambda: src),
            patch("eostrata.config.settings", mock_settings),
            patch("eostrata.catalog.load_or_create", return_value=MagicMock()),
            patch("eostrata.catalog.register_item"),
            patch("eostrata.catalog.save"),
        ):
            _run_job(
                job_id="test",
                source_id="worldpop",
                params={},
                auto_period=True,
                webhook_url=None,
            )

        call_kwargs = src.download.call_args[1]
        assert call_kwargs.get("year") == 2023

    def test_sends_webhook_after_exhausted_retries(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from eostrata.scheduler import _run_job

        mock_settings = MagicMock()
        mock_settings.bbox = (0.0, 0.0, 10.0, 10.0)

        failing_source = MagicMock()
        failing_source.temporal_resolution = "annual"
        failing_source.download.side_effect = RuntimeError("network error")

        webhook_calls = []

        with (
            patch("eostrata.sources.base.get_source", return_value=lambda: failing_source),
            patch("eostrata.config.settings", mock_settings),
            patch(
                "eostrata.scheduler._send_webhook",
                side_effect=lambda url, payload: webhook_calls.append(payload),
            ),
            patch("eostrata.scheduler.time.sleep"),
        ):
            _run_job(
                job_id="fail_job",
                source_id="worldpop",
                params={"iso3": "NGA", "year": 2020},
                auto_period=False,
                webhook_url="https://example.com/alert",
            )

        assert len(webhook_calls) == 1
        assert webhook_calls[0]["job_id"] == "fail_job"
        assert "network error" in webhook_calls[0]["error"]

    def test_no_webhook_when_url_is_none(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from eostrata.scheduler import _run_job

        mock_settings = MagicMock()

        failing_source = MagicMock()
        failing_source.temporal_resolution = "annual"
        failing_source.download.side_effect = RuntimeError("err")

        with (
            patch("eostrata.sources.base.get_source", return_value=lambda: failing_source),
            patch("eostrata.config.settings", mock_settings),
            patch("eostrata.scheduler._send_webhook") as mock_wh,
            patch("eostrata.scheduler.time.sleep"),
        ):
            _run_job(
                job_id="x",
                source_id="worldpop",
                params={"iso3": "NGA", "year": 2020},
                auto_period=False,
                webhook_url=None,
            )

        mock_wh.assert_not_called()


class TestExecuteIngestion:
    def test_empty_paths_raises(self):
        """_execute_ingestion raises RuntimeError when download returns no paths."""
        from unittest.mock import MagicMock

        from eostrata.scheduler import _execute_ingestion

        src = MagicMock()
        src.download.return_value = []
        with pytest.raises(RuntimeError, match="no paths"):
            _execute_ingestion(src, {}, MagicMock())


class TestSchedulerInit:
    def test_missing_apscheduler_raises(self):
        """Scheduler() raises ImportError when APScheduler is not installed."""
        import sys

        with patch.dict(
            sys.modules,
            {
                "apscheduler": None,
                "apscheduler.schedulers": None,
                "apscheduler.schedulers.background": None,
            },
        ):
            from eostrata.scheduler import Scheduler

            with pytest.raises(ImportError, match="APScheduler"):
                Scheduler()


class TestSchedulerStartStop:
    def test_start_stop_no_jobs(self, tmp_path):
        """Scheduler with no enabled jobs starts and stops cleanly."""
        f = tmp_path / "schedules.yml"
        f.write_text("jobs: []\n")

        from eostrata.scheduler import Scheduler

        s = Scheduler(schedules_path=f)
        s.start()
        s.stop()

    def test_start_enabled_job_registers(self, tmp_path):
        """An enabled job with a valid cron expression is registered."""
        f = tmp_path / "schedules.yml"
        f.write_text(
            textwrap.dedent("""\
            jobs:
              - id: worldpop_job
                source: worldpop
                params:
                  iso3: NGA
                cron: "0 2 1 * *"
                auto_period: false
                enabled: true
        """)
        )

        from eostrata.scheduler import Scheduler

        s = Scheduler(schedules_path=f)
        s.start()
        assert s._job_count == 1
        s.stop()

    def test_start_invalid_cron_skips_job(self, tmp_path):
        """A job with a malformed cron expression is skipped."""
        f = tmp_path / "schedules.yml"
        f.write_text(
            textwrap.dedent("""\
            jobs:
              - id: bad_cron
                source: worldpop
                params: {}
                cron: "0 2 1"
                enabled: true
        """)
        )

        from eostrata.scheduler import Scheduler

        s = Scheduler(schedules_path=f)
        s.start()
        assert s._job_count == 0
        s.stop()

    def test_start_skip_disabled_jobs(self, tmp_path):
        """Disabled jobs are not registered with APScheduler."""
        import textwrap

        f = tmp_path / "schedules.yml"
        f.write_text(
            textwrap.dedent("""\
            jobs:
              - id: disabled
                source: chirps
                params: {}
                cron: "0 3 15 * *"
                enabled: false
        """)
        )

        from eostrata.scheduler import Scheduler

        s = Scheduler(schedules_path=f)
        s.start()
        assert s._job_count == 0
        s.stop()
