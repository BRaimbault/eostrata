"""Tests for the scheduler module."""

from __future__ import annotations

import sys
import textwrap
from datetime import UTC

import pytest

from eostrata.scheduler import _load_schedules, _send_webhook


class TestLoadSchedules:
    def test_yaml_not_installed_raises(self, tmp_path, mocker):
        """_load_schedules raises ImportError when PyYAML is not installed."""
        import sys

        f = tmp_path / "schedules.yml"
        f.write_text("jobs: []\n")
        mocker.patch.dict(sys.modules, {"yaml": None})
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
    def test_posts_payload(self, mocker):
        mock_resp = mocker.MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_post = mocker.patch("httpx.post", return_value=mock_resp)

        _send_webhook("https://example.com/alert", {"event": "test"})
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[0][0] == "https://example.com/alert"

    def test_swallows_connection_error(self, mocker):
        mocker.patch("httpx.post", side_effect=Exception("connection refused"))
        # Should not raise
        _send_webhook("https://example.com/alert", {"event": "test"})


class TestRunJob:
    """Tests for the _run_job orchestration function."""

    def _make_mock_source(self, tmp_path, mocker):
        """Return a mock source that writes a tiny real Zarr."""
        from datetime import datetime

        import numpy as np
        import xarray as xr

        zarr_root = tmp_path / "zarr"
        zarr_root.mkdir(parents=True, exist_ok=True)

        ds = xr.Dataset(
            {"population": (("y", "x"), np.ones((4, 4), dtype="float32"))},
            coords={"y": np.arange(4.0), "x": np.arange(4.0)},
        )
        ds.to_zarr(str(zarr_root), group="worldpop/nga", mode="w")

        src = mocker.MagicMock()
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

    def test_succeeds_on_first_attempt(self, tmp_path, mocker):
        src, zarr_root = self._make_mock_source(tmp_path, mocker)

        from eostrata.scheduler import _run_job

        mock_settings = mocker.MagicMock()
        mock_settings.bbox = (2.0, 4.0, 15.0, 14.0)
        mock_settings.raw_dir = tmp_path / "raw"
        mock_settings.zarr_root = zarr_root
        mock_settings.catalog_path = tmp_path / "catalog.json"

        mocker.patch("eostrata.sources.base.get_source", return_value=lambda: src)
        mocker.patch("eostrata.config.settings", mock_settings)
        mocker.patch("eostrata.catalog.load_or_create", return_value=mocker.MagicMock())
        mocker.patch("eostrata.catalog.register_item")
        mocker.patch("eostrata.catalog.save")
        _run_job(
            job_id="test",
            source_id="worldpop",
            params={"iso3": "NGA", "year": 2020},
            auto_period=False,
            webhook_url=None,
        )

    def test_auto_period_injects_month_for_monthly_source(self, tmp_path, mocker):
        """auto_period=True with a monthly source injects both year and month."""
        from datetime import datetime

        src, zarr_root = self._make_mock_source(tmp_path, mocker)

        from eostrata.scheduler import _run_job

        src.temporal_resolution = "monthly"
        src.latest_available.return_value = datetime(2023, 6, 1, tzinfo=UTC)

        mock_settings = mocker.MagicMock()
        mock_settings.bbox = (0.0, 0.0, 10.0, 10.0)
        mock_settings.raw_dir = tmp_path / "raw"
        mock_settings.zarr_root = zarr_root
        mock_settings.catalog_path = tmp_path / "catalog.json"

        mocker.patch("eostrata.sources.base.get_source", return_value=lambda: src)
        mocker.patch("eostrata.config.settings", mock_settings)
        mocker.patch("eostrata.catalog.load_or_create", return_value=mocker.MagicMock())
        mocker.patch("eostrata.catalog.register_item")
        mocker.patch("eostrata.catalog.save")
        _run_job(
            job_id="test",
            source_id="worldpop",
            params={},
            auto_period=True,
            webhook_url=None,
        )

        call_kwargs = src.download.call_args[1]
        assert call_kwargs.get("month") == 6

    def test_auto_period_injects_year(self, tmp_path, mocker):
        """auto_period=True should resolve year from source.latest_available()."""
        from datetime import datetime

        src, zarr_root = self._make_mock_source(tmp_path, mocker)

        from eostrata.scheduler import _run_job

        src.latest_available.return_value = datetime(2023, 1, 1, tzinfo=UTC)

        mock_settings = mocker.MagicMock()
        mock_settings.bbox = (0.0, 0.0, 10.0, 10.0)
        mock_settings.raw_dir = tmp_path / "raw"
        mock_settings.zarr_root = zarr_root
        mock_settings.catalog_path = tmp_path / "catalog.json"

        mocker.patch("eostrata.sources.base.get_source", return_value=lambda: src)
        mocker.patch("eostrata.config.settings", mock_settings)
        mocker.patch("eostrata.catalog.load_or_create", return_value=mocker.MagicMock())
        mocker.patch("eostrata.catalog.register_item")
        mocker.patch("eostrata.catalog.save")
        _run_job(
            job_id="test",
            source_id="worldpop",
            params={},
            auto_period=True,
            webhook_url=None,
        )

        call_kwargs = src.download.call_args[1]
        assert call_kwargs.get("year") == 2023

    def test_sends_webhook_after_exhausted_retries(self, tmp_path, mocker):
        from eostrata.scheduler import _run_job

        mock_settings = mocker.MagicMock()
        mock_settings.bbox = (0.0, 0.0, 10.0, 10.0)

        failing_source = mocker.MagicMock()
        failing_source.temporal_resolution = "annual"
        failing_source.download.side_effect = RuntimeError("network error")

        webhook_calls = []

        mocker.patch("eostrata.sources.base.get_source", return_value=lambda: failing_source)
        mocker.patch("eostrata.config.settings", mock_settings)
        mocker.patch(
            "eostrata.scheduler._send_webhook",
            side_effect=lambda url, payload: webhook_calls.append(payload),
        )
        mocker.patch("eostrata.scheduler.time.sleep")
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

    def test_no_webhook_when_url_is_none(self, tmp_path, mocker):
        from eostrata.scheduler import _run_job

        mock_settings = mocker.MagicMock()

        failing_source = mocker.MagicMock()
        failing_source.temporal_resolution = "annual"
        failing_source.download.side_effect = RuntimeError("err")

        mocker.patch("eostrata.sources.base.get_source", return_value=lambda: failing_source)
        mocker.patch("eostrata.config.settings", mock_settings)
        mock_wh = mocker.patch("eostrata.scheduler._send_webhook")
        mocker.patch("eostrata.scheduler.time.sleep")
        _run_job(
            job_id="x",
            source_id="worldpop",
            params={"iso3": "NGA", "year": 2020},
            auto_period=False,
            webhook_url=None,
        )

        mock_wh.assert_not_called()


class TestExecuteIngestion:
    def test_empty_paths_raises(self, mocker):
        """_execute_ingestion raises RuntimeError when download returns no paths."""
        from eostrata.scheduler import _execute_ingestion

        src = mocker.MagicMock()
        src.download.return_value = []
        with pytest.raises(RuntimeError, match="no paths"):
            _execute_ingestion(src, {}, mocker.MagicMock())


class TestSchedulerInit:
    def test_missing_apscheduler_raises(self, mocker):
        """Scheduler() raises ImportError when APScheduler is not installed."""
        import sys

        mocker.patch.dict(
            sys.modules,
            {
                "apscheduler": None,
                "apscheduler.schedulers": None,
                "apscheduler.schedulers.background": None,
            },
        )
        from eostrata.scheduler import Scheduler

        with pytest.raises(ImportError, match="APScheduler"):
            Scheduler()


class TestSchedulerStartStop:
    @pytest.fixture(autouse=True)
    def _mock_apscheduler(self, mocker):
        """Inject a mock APScheduler so tests run without the package installed."""
        mock_instance = mocker.MagicMock()
        mock_instance.running = True  # so stop() calls shutdown()

        mock_bg_cls = mocker.MagicMock(return_value=mock_instance)
        mock_bg_module = mocker.MagicMock()
        mock_bg_module.BackgroundScheduler = mock_bg_cls

        mock_schedulers = mocker.MagicMock()
        mock_schedulers.background = mock_bg_module

        mock_aps = mocker.MagicMock()
        mock_aps.schedulers = mock_schedulers

        mocker.patch.dict(
            sys.modules,
            {
                "apscheduler": mock_aps,
                "apscheduler.schedulers": mock_schedulers,
                "apscheduler.schedulers.background": mock_bg_module,
            },
        )
        yield mock_bg_cls

    def test_start_stop_no_jobs(self, tmp_path):
        """Scheduler with no enabled jobs starts and stops cleanly."""
        f = tmp_path / "schedules.yml"
        f.write_text("jobs: []\n")

        from eostrata.scheduler import Scheduler

        s = Scheduler(schedules_path=f)
        s.start()
        s.stop()

    def test_start_enabled_job_registers(self, tmp_path, _mock_apscheduler):
        """An enabled job with a valid cron expression is registered with APScheduler."""
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
        assert "worldpop_job" in s._job_defs
        assert _mock_apscheduler.return_value.add_job.call_count == 1
        s.stop()

    def test_start_invalid_cron_skips_job(self, tmp_path, _mock_apscheduler):
        """A job with a malformed cron expression is loaded but not sent to APScheduler."""
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
        assert "bad_cron" in s._job_defs
        assert _mock_apscheduler.return_value.add_job.call_count == 0
        s.stop()

    def test_start_skip_disabled_jobs(self, tmp_path, _mock_apscheduler):
        """Disabled jobs are loaded into _job_defs but not registered with APScheduler."""
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
        assert "disabled" in s._job_defs
        assert _mock_apscheduler.return_value.add_job.call_count == 0
        s.stop()

    def test_start_skips_job_with_missing_id(self, tmp_path):
        """A job def with no 'id' field is silently skipped."""
        f = tmp_path / "schedules.yml"
        f.write_text(
            textwrap.dedent("""\
            jobs:
              - source: chirps
                params: {}
                cron: "0 3 15 * *"
                enabled: true
        """)
        )

        from eostrata.scheduler import Scheduler

        s = Scheduler(schedules_path=f)
        s.start()
        assert len(s._job_defs) == 0
        s.stop()

    def test_stop_skips_when_not_running(self, tmp_path):
        """stop() is a no-op when the scheduler was never started."""
        f = tmp_path / "schedules.yml"
        f.write_text("jobs: []\n")

        from eostrata.scheduler import Scheduler

        s = Scheduler(schedules_path=f)
        # Patch running=False so stop() takes the early-exit branch
        s._scheduler.running = False
        s.stop()  # should not raise


class TestRunJobReturnValue:
    """_run_job returns (bool, error_str_or_none)."""

    def _make_source(self, tmp_path, mocker):
        from datetime import datetime

        import numpy as np
        import xarray as xr

        zarr_root = tmp_path / "zarr"
        zarr_root.mkdir(parents=True, exist_ok=True)
        ds = xr.Dataset(
            {"population": (("y", "x"), np.ones((4, 4), dtype="float32"))},
            coords={"y": np.arange(4.0), "x": np.arange(4.0)},
        )
        ds.to_zarr(str(zarr_root), group="worldpop/nga", mode="w")
        src = mocker.MagicMock()
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

    def test_returns_true_none_on_success(self, tmp_path, mocker):
        from eostrata.scheduler import _run_job

        src, zarr_root = self._make_source(tmp_path, mocker)
        mock_settings = mocker.MagicMock()
        mock_settings.bbox = (0.0, 0.0, 10.0, 10.0)
        mock_settings.raw_dir = tmp_path / "raw"
        mock_settings.zarr_root = zarr_root
        mock_settings.catalog_path = tmp_path / "catalog.json"

        mocker.patch("eostrata.sources.base.get_source", return_value=lambda: src)
        mocker.patch("eostrata.config.settings", mock_settings)
        mocker.patch("eostrata.catalog.load_or_create", return_value=mocker.MagicMock())
        mocker.patch("eostrata.catalog.register_item")
        mocker.patch("eostrata.catalog.save")
        result = _run_job(
            job_id="test",
            source_id="worldpop",
            params={"iso3": "NGA", "year": 2020},
            auto_period=False,
            webhook_url=None,
        )

        assert result == (True, None)

    def test_returns_false_and_error_on_all_retries_failed(self, tmp_path, mocker):
        from eostrata.scheduler import _run_job

        failing_src = mocker.MagicMock()
        failing_src.temporal_resolution = "annual"
        failing_src.download.side_effect = RuntimeError("boom")

        mocker.patch("eostrata.sources.base.get_source", return_value=lambda: failing_src)
        mocker.patch("eostrata.config.settings", mocker.MagicMock())
        mocker.patch("eostrata.scheduler.time.sleep")
        success, error = _run_job(
            job_id="fail",
            source_id="worldpop",
            params={"iso3": "NGA", "year": 2020},
            auto_period=False,
            webhook_url=None,
        )

        assert success is False
        assert "boom" in error


class TestSchedulerSingleton:
    def test_set_get_clear(self, mocker):
        from eostrata.scheduler import get_scheduler, set_scheduler

        mock_s = mocker.MagicMock()
        set_scheduler(mock_s)
        try:
            assert get_scheduler() is mock_s
        finally:
            set_scheduler(None)
        assert get_scheduler() is None


class TestSchedulerTracking:
    """Tests for _wrap_for_tracking, get_jobs, save_job, remove_job, trigger_job."""

    @pytest.fixture()
    def scheduler(self, tmp_path, mocker):
        """A Scheduler backed by a mock APScheduler."""
        import sys

        mock_instance = mocker.MagicMock()
        mock_instance.running = True
        mock_bg_cls = mocker.MagicMock(return_value=mock_instance)
        mock_bg_module = mocker.MagicMock()
        mock_bg_module.BackgroundScheduler = mock_bg_cls
        mock_schedulers = mocker.MagicMock()
        mock_schedulers.background = mock_bg_module
        mock_aps = mocker.MagicMock()
        mock_aps.schedulers = mock_schedulers

        mocker.patch.dict(
            sys.modules,
            {
                "apscheduler": mock_aps,
                "apscheduler.schedulers": mock_schedulers,
                "apscheduler.schedulers.background": mock_bg_module,
            },
        )
        from eostrata.scheduler import Scheduler

        f = tmp_path / "schedules.yml"
        f.write_text("jobs: []\n")
        s = Scheduler(schedules_path=f)
        s._scheduler = mock_instance
        yield s

    def test_get_jobs_empty(self, scheduler):
        assert scheduler.get_jobs() == []

    def test_get_jobs_includes_next_run_and_last_run(self, scheduler, mocker):
        from datetime import datetime

        scheduler._job_defs["j1"] = {
            "id": "j1",
            "source": "chirps",
            "params": {},
            "cron": "0 3 15 * *",
            "auto_period": True,
            "enabled": True,
        }
        mock_apjob = mocker.MagicMock()
        mock_apjob.next_run_time = datetime(2026, 4, 15, 3, 0, tzinfo=UTC)
        scheduler._scheduler.get_job.return_value = mock_apjob

        jobs = scheduler.get_jobs()
        assert len(jobs) == 1
        assert jobs[0]["next_run_time"] == "2026-04-15T03:00:00+00:00"
        assert jobs[0]["last_run"] is None

    def test_get_jobs_next_run_none_when_no_apjob(self, scheduler):
        scheduler._job_defs["j1"] = {
            "id": "j1",
            "source": "chirps",
            "params": {},
            "cron": "0 3 15 * *",
            "auto_period": True,
            "enabled": False,
        }
        scheduler._scheduler.get_job.return_value = None
        jobs = scheduler.get_jobs()
        assert jobs[0]["next_run_time"] is None

    def test_save_job_adds_to_defs_and_schedules_if_enabled(self, scheduler):
        job_def = {
            "id": "new_job",
            "source": "chirps",
            "params": {},
            "cron": "0 3 15 * *",
            "auto_period": True,
            "enabled": True,
        }
        scheduler.save_job(job_def)

        assert "new_job" in scheduler._job_defs
        assert scheduler._scheduler.add_job.call_count == 1

    def test_save_job_disabled_does_not_call_add_job(self, scheduler):
        job_def = {
            "id": "new_job",
            "source": "chirps",
            "params": {},
            "cron": "0 3 15 * *",
            "auto_period": True,
            "enabled": False,
        }
        scheduler.save_job(job_def)

        assert "new_job" in scheduler._job_defs
        scheduler._scheduler.add_job.assert_not_called()

    def test_save_job_replaces_existing_apscheduler_entry(self, scheduler):
        job_def = {
            "id": "existing",
            "source": "chirps",
            "params": {},
            "cron": "0 3 15 * *",
            "auto_period": True,
            "enabled": True,
        }
        scheduler._job_defs["existing"] = job_def
        scheduler.save_job({**job_def, "cron": "0 4 15 * *"})

        # remove_job called to clear old entry, add_job called for new one
        scheduler._scheduler.remove_job.assert_called_once_with("existing")
        assert scheduler._scheduler.add_job.call_count == 1

    def test_remove_job_clears_defs_and_apscheduler(self, scheduler):
        scheduler._job_defs["to_remove"] = {
            "id": "to_remove",
            "source": "chirps",
            "params": {},
            "cron": "0 3 15 * *",
            "auto_period": True,
            "enabled": True,
        }
        scheduler.remove_job("to_remove")

        assert "to_remove" not in scheduler._job_defs
        scheduler._scheduler.remove_job.assert_called_once_with("to_remove")

    def test_remove_job_tolerates_missing_apscheduler_entry(self, scheduler):
        """remove_job on a disabled (unscheduled) job doesn't raise."""
        scheduler._job_defs["disabled"] = {
            "id": "disabled",
            "source": "chirps",
            "params": {},
            "cron": "0 3 15 * *",
            "auto_period": False,
            "enabled": False,
        }
        scheduler._scheduler.remove_job.side_effect = Exception("not found")
        scheduler.remove_job("disabled")  # must not raise
        assert "disabled" not in scheduler._job_defs

    def test_trigger_job_raises_on_unknown_id(self, scheduler):
        with pytest.raises(KeyError, match="unknown"):
            scheduler.trigger_job("unknown")

    def test_trigger_job_spawns_thread(self, scheduler, mocker):
        scheduler._job_defs["j1"] = {
            "id": "j1",
            "source": "chirps",
            "params": {},
            "cron": "0 3 15 * *",
            "auto_period": False,
            "enabled": True,
        }
        mock_t = mocker.MagicMock()
        mocker.patch("threading.Thread", return_value=mock_t)
        scheduler.trigger_job("j1")
        mock_t.start.assert_called_once()

    def test_wrap_for_tracking_records_success(self, scheduler, tmp_path, mocker):
        import numpy as np
        import xarray as xr

        zarr_root = tmp_path / "zarr"
        zarr_root.mkdir()
        ds = xr.Dataset(
            {"population": (("y", "x"), np.ones((2, 2), dtype="float32"))},
            coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
        )
        ds.to_zarr(str(zarr_root), group="worldpop/nga", mode="w")

        src = mocker.MagicMock()
        src.collection_id = "worldpop"
        src.temporal_resolution = "annual"
        src.VARIABLE = "population"
        src.download.return_value = [tmp_path / "f.tif"]
        src.to_zarr.return_value = ds
        src.zarr_group.return_value = "worldpop/nga"
        src.stac_item_id.return_value = "worldpop_nga"
        src.stac_properties.return_value = {}

        mock_settings = mocker.MagicMock()
        mock_settings.bbox = (0.0, 0.0, 10.0, 10.0)
        mock_settings.raw_dir = tmp_path / "raw"
        mock_settings.zarr_root = zarr_root
        mock_settings.catalog_path = tmp_path / "catalog.json"

        tracked = scheduler._wrap_for_tracking(
            "j1", "worldpop", {"iso3": "NGA", "year": 2020}, False, None
        )
        mocker.patch("eostrata.sources.base.get_source", return_value=lambda: src)
        mocker.patch("eostrata.config.settings", mock_settings)
        mocker.patch("eostrata.catalog.load_or_create", return_value=mocker.MagicMock())
        mocker.patch("eostrata.catalog.register_item")
        mocker.patch("eostrata.catalog.save")
        tracked()

        run = scheduler._job_runs["j1"]
        assert run["status"] == "success"
        assert run["error"] is None
        assert run["finished_at"] is not None

    def test_wrap_for_tracking_records_failure(self, scheduler, mocker):
        src = mocker.MagicMock()
        src.temporal_resolution = "annual"
        src.download.side_effect = RuntimeError("disk full")

        tracked = scheduler._wrap_for_tracking("j1", "worldpop", {}, False, None)
        mocker.patch("eostrata.sources.base.get_source", return_value=lambda: src)
        mocker.patch("eostrata.config.settings", mocker.MagicMock())
        mocker.patch("eostrata.scheduler.time.sleep")
        tracked()

        run = scheduler._job_runs["j1"]
        assert run["status"] == "failed"
        assert "disk full" in run["error"]

    def test_wrap_for_tracking_sets_running_before_executing(self, scheduler, mocker):
        """While the job is in-flight, status should be 'running'."""
        observed_status = []

        def capture_status(*args, **kwargs):
            observed_status.append(scheduler._job_runs.get("j1", {}).get("status"))
            raise RuntimeError("stop")

        src = mocker.MagicMock()
        src.temporal_resolution = "annual"
        src.download.side_effect = capture_status

        tracked = scheduler._wrap_for_tracking("j1", "worldpop", {}, False, None)
        mocker.patch("eostrata.sources.base.get_source", return_value=lambda: src)
        mocker.patch("eostrata.config.settings", mocker.MagicMock())
        mocker.patch("eostrata.scheduler.time.sleep")
        tracked()

        assert "running" in observed_status


class TestSchedulerWriteSchedules:
    @pytest.fixture()
    def scheduler_with_jobs(self, tmp_path, mocker):
        import sys

        mock_instance = mocker.MagicMock()
        mock_instance.running = False
        mock_bg_cls = mocker.MagicMock(return_value=mock_instance)
        mock_bg_module = mocker.MagicMock()
        mock_bg_module.BackgroundScheduler = mock_bg_cls
        mock_schedulers = mocker.MagicMock()
        mock_schedulers.background = mock_bg_module
        mock_aps = mocker.MagicMock()
        mock_aps.schedulers = mock_schedulers

        mocker.patch.dict(
            sys.modules,
            {
                "apscheduler": mock_aps,
                "apscheduler.schedulers": mock_schedulers,
                "apscheduler.schedulers.background": mock_bg_module,
            },
        )
        from eostrata.scheduler import Scheduler

        f = tmp_path / "schedules.yml"
        f.write_text("jobs: []\n")
        s = Scheduler(schedules_path=f)
        s._scheduler = mock_instance
        s._job_defs["j1"] = {
            "id": "j1",
            "source": "chirps",
            "params": {},
            "cron": "0 3 15 * *",
            "auto_period": True,
            "enabled": True,
        }
        yield s, f

    def test_write_persists_jobs_to_yaml(self, scheduler_with_jobs):
        import yaml

        s, f = scheduler_with_jobs
        s._write_schedules()
        data = yaml.safe_load(f.read_text())
        assert any(j["id"] == "j1" for j in (data.get("jobs") or []))

    def test_write_includes_webhook_if_set(self, scheduler_with_jobs):
        import yaml

        s, f = scheduler_with_jobs
        s._global_webhook = "https://example.com/hook"
        s._write_schedules()
        data = yaml.safe_load(f.read_text())
        assert data["webhook_url"] == "https://example.com/hook"

    def test_write_schedules_no_yaml_warns_and_returns(self, scheduler_with_jobs, mocker):
        import sys

        s, _ = scheduler_with_jobs
        mocker.patch.dict(sys.modules, {"yaml": None})
        s._write_schedules()  # must not raise
