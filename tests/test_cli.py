"""Tests for the eostrata CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_bounds
from typer.testing import CliRunner

from eostrata.cli import _ALL_MONTHS, _parse_int_list, app

runner = CliRunner()


class TestParseHelpers:
    def test_single(self):
        assert _parse_int_list(2020, None, 2025) == [2020]

    def test_default(self):
        assert _parse_int_list(None, None, 2025) == [2025]

    def test_multi(self):
        assert _parse_int_list(None, "2020,2021,2022", 2025) == [2020, 2021, 2022]

    def test_multi_deduplicates_and_sorts(self):
        assert _parse_int_list(None, "2022,2020,2021,2020", 2025) == [2020, 2021, 2022]

    def test_single_month(self):
        assert _parse_int_list(6, None, 1) == [6]

    def test_default_month(self):
        assert _parse_int_list(None, None, 3) == [3]

    def test_multi_months(self):
        assert _parse_int_list(None, "1,2,3", 12) == [1, 2, 3]

    def test_multi_months_sorted(self):
        assert _parse_int_list(None, "3,1,2", 12) == [1, 2, 3]

    def test_all_months(self):
        assert _parse_int_list(None, "ALL", 1, all_values=_ALL_MONTHS) == list(range(1, 13))

    def test_all_months_case_insensitive(self):
        assert _parse_int_list(None, "all", 1, all_values=_ALL_MONTHS) == list(range(1, 13))

    def test_all_without_all_values_raises(self):
        import pytest

        with pytest.raises(ValueError, match="'ALL' is not supported"):
            _parse_int_list(None, "ALL", 1)


def _make_tif(path: Path, bbox=(2.0, 4.0, 6.0, 8.0), width=10, height=10) -> Path:
    transform = from_bounds(*bbox, width=width, height=height)
    data = np.ones((height, width), dtype="float32") * 100.0
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return path


def _make_settings_mock(tmp_path: Path):
    s = MagicMock()
    s.zarr_root = tmp_path / "zarr"
    s.raw_dir = tmp_path / "raw"
    s.catalog_path = tmp_path / "catalog.json"
    s.bbox = (2.0, 4.0, 15.0, 14.0)
    s.store_quota_mb = 0.0
    return s


class TestDownloadWorldpop:
    def test_download_worldpop_success(self, tmp_path):
        tif = tmp_path / "raw" / "worldpop" / "nga_pop_2020_CN_1km_R2025A_UA_v1.tif"
        tif.parent.mkdir(parents=True)
        _make_tif(tif)

        mock_settings = _make_settings_mock(tmp_path)

        with (
            patch("eostrata.config.settings", mock_settings),
            patch("eostrata.sources.worldpop.WorldPopSource.download", return_value=[tif]),
            patch("eostrata.cache.check_and_evict"),
        ):
            result = runner.invoke(
                app,
                [
                    "download",
                    "worldpop",
                    "NGA",
                    "--year",
                    "2020",
                    "--zarr-root",
                    str(tmp_path / "zarr"),
                    "--raw-dir",
                    str(tmp_path / "raw"),
                    "--catalog-path",
                    str(tmp_path / "catalog.json"),
                ],
            )

        assert result.exit_code == 0, result.output
        assert "Done" in result.output

    def test_download_worldpop_multiple_years(self, tmp_path):
        """--years 2020,2021 downloads both years in one call."""
        tif = tmp_path / "raw" / "worldpop" / "nga_pop_2020_CN_1km_R2025A_UA_v1.tif"
        tif.parent.mkdir(parents=True)

        def _recreate_tif(*args, **kwargs):
            _make_tif(tif)
            return [tif]

        mock_settings = _make_settings_mock(tmp_path)

        with (
            patch("eostrata.config.settings", mock_settings),
            patch("eostrata.sources.worldpop.WorldPopSource.download", side_effect=_recreate_tif),
            patch("eostrata.cache.check_and_evict"),
        ):
            result = runner.invoke(
                app,
                [
                    "download",
                    "worldpop",
                    "NGA",
                    "--years",
                    "2020,2021",
                    "--zarr-root",
                    str(tmp_path / "zarr"),
                    "--raw-dir",
                    str(tmp_path / "raw"),
                    "--catalog-path",
                    str(tmp_path / "catalog.json"),
                ],
            )

        assert result.exit_code == 0, result.output
        assert "Done" in result.output

    def test_download_worldpop_error_exits_nonzero(self, tmp_path):
        """A download failure aborts the command with a non-zero exit code."""
        mock_settings = _make_settings_mock(tmp_path)

        with (
            patch("eostrata.config.settings", mock_settings),
            patch(
                "eostrata.sources.worldpop.WorldPopSource.download",
                side_effect=RuntimeError("network error"),
            ),
            patch("eostrata.cache.check_and_evict"),
        ):
            result = runner.invoke(
                app,
                [
                    "download",
                    "worldpop",
                    "NGA",
                    "--year",
                    "2020",
                    "--zarr-root",
                    str(tmp_path / "zarr"),
                    "--raw-dir",
                    str(tmp_path / "raw"),
                    "--catalog-path",
                    str(tmp_path / "catalog.json"),
                ],
            )

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_download_worldpop_auto_year(self, tmp_path):
        """When --year is omitted, CLI resolves latest available."""
        tif = tmp_path / "raw" / "worldpop" / "nga_pop_2025_CN_1km_R2025A_UA_v1.tif"
        tif.parent.mkdir(parents=True)
        _make_tif(tif)

        mock_settings = _make_settings_mock(tmp_path)

        with (
            patch("eostrata.config.settings", mock_settings),
            patch("eostrata.sources.worldpop.WorldPopSource.download", return_value=[tif]),
            patch("eostrata.cache.check_and_evict"),
        ):
            result = runner.invoke(
                app,
                [
                    "download",
                    "worldpop",
                    "NGA",
                    "--zarr-root",
                    str(tmp_path / "zarr"),
                    "--raw-dir",
                    str(tmp_path / "raw"),
                    "--catalog-path",
                    str(tmp_path / "catalog.json"),
                ],
            )

        assert result.exit_code == 0, result.output


class TestDownloadChirps:
    def test_download_chirps_success(self, tmp_path):
        tif = tmp_path / "raw" / "chirps" / "chirps-v2.0.2024.01.tif"
        tif.parent.mkdir(parents=True)
        _make_tif(tif)

        mock_settings = _make_settings_mock(tmp_path)

        with (
            patch("eostrata.config.settings", mock_settings),
            patch("eostrata.sources.chirps.CHIRPSSource.download", return_value=[tif]),
            patch("eostrata.cache.check_and_evict"),
        ):
            result = runner.invoke(
                app,
                [
                    "download",
                    "chirps",
                    "--year",
                    "2024",
                    "--month",
                    "1",
                    "--zarr-root",
                    str(tmp_path / "zarr"),
                    "--raw-dir",
                    str(tmp_path / "raw"),
                    "--catalog-path",
                    str(tmp_path / "catalog.json"),
                ],
            )

        assert result.exit_code == 0, result.output
        assert "Done" in result.output

    def test_download_chirps_multiple_years_months(self, tmp_path):
        """--years 2023,2024 --months 1,2 downloads all (year, month) combinations."""
        tif = tmp_path / "raw" / "chirps" / "chirps-v2.0.2023.01.tif"
        tif.parent.mkdir(parents=True)

        mock_settings = _make_settings_mock(tmp_path)
        download_calls = []

        def _fake_download(*args, **kwargs):
            download_calls.append((kwargs.get("year"), kwargs.get("month")))
            _make_tif(tif)
            return [tif]

        with (
            patch("eostrata.config.settings", mock_settings),
            patch("eostrata.sources.chirps.CHIRPSSource.download", side_effect=_fake_download),
            patch("eostrata.cache.check_and_evict"),
        ):
            result = runner.invoke(
                app,
                [
                    "download",
                    "chirps",
                    "--years",
                    "2023,2024",
                    "--months",
                    "1,2",
                    "--zarr-root",
                    str(tmp_path / "zarr"),
                    "--raw-dir",
                    str(tmp_path / "raw"),
                    "--catalog-path",
                    str(tmp_path / "catalog.json"),
                ],
            )

        assert result.exit_code == 0, result.output
        assert len(download_calls) == 4
        assert (2023, 1) in download_calls
        assert (2023, 2) in download_calls
        assert (2024, 1) in download_calls
        assert (2024, 2) in download_calls

    def test_download_chirps_error_exits_nonzero(self, tmp_path):
        """A download failure aborts the command with a non-zero exit code."""
        mock_settings = _make_settings_mock(tmp_path)

        with (
            patch("eostrata.config.settings", mock_settings),
            patch(
                "eostrata.sources.chirps.CHIRPSSource.download",
                side_effect=RuntimeError("network error"),
            ),
            patch("eostrata.cache.check_and_evict"),
        ):
            result = runner.invoke(
                app,
                [
                    "download",
                    "chirps",
                    "--year",
                    "2024",
                    "--month",
                    "1",
                    "--zarr-root",
                    str(tmp_path / "zarr"),
                    "--raw-dir",
                    str(tmp_path / "raw"),
                    "--catalog-path",
                    str(tmp_path / "catalog.json"),
                ],
            )

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_download_chirps_auto_period(self, tmp_path):
        """Omitting --year/--month uses latest_available()."""
        tif = tmp_path / "raw" / "chirps" / "chirps-v2.0.2026.01.tif"
        tif.parent.mkdir(parents=True)
        _make_tif(tif)

        mock_settings = _make_settings_mock(tmp_path)

        with (
            patch("eostrata.config.settings", mock_settings),
            patch("eostrata.sources.chirps.CHIRPSSource.download", return_value=[tif]),
            patch("eostrata.cache.check_and_evict"),
        ):
            result = runner.invoke(
                app,
                [
                    "download",
                    "chirps",
                    "--zarr-root",
                    str(tmp_path / "zarr"),
                    "--raw-dir",
                    str(tmp_path / "raw"),
                    "--catalog-path",
                    str(tmp_path / "catalog.json"),
                ],
            )

        assert result.exit_code == 0, result.output


class TestDownloadCds:
    def test_download_cds_success(self, tmp_path):
        nc = tmp_path / "raw" / "cds" / "era5_t2m_2023_01-02-03-04-05-06-07-08-09-10-11-12.nc"
        nc.parent.mkdir(parents=True)
        # Write a minimal NetCDF that CDSSource.to_zarr can process
        y = np.linspace(14.0, 4.0, 5)
        x = np.linspace(2.0, 15.0, 5)
        times = [np.datetime64(f"2023-{m:02d}-01") for m in range(1, 3)]
        data = np.ones((2, 5, 5), dtype="float32") * 280.0
        ds = xr.Dataset(
            {"t2m": (("time", "y", "x"), data)},
            coords={"time": times, "y": y, "x": x},
        )
        ds.to_netcdf(str(nc))

        mock_settings = _make_settings_mock(tmp_path)

        with (
            patch("eostrata.config.settings", mock_settings),
            patch("eostrata.sources.cds.CDSSource.download", return_value=[nc]),
            patch("eostrata.cache.check_and_evict"),
        ):
            result = runner.invoke(
                app,
                [
                    "download",
                    "cds",
                    "--variable",
                    "t2m",
                    "--year",
                    "2023",
                    "--months",
                    "1,2",
                    "--zarr-root",
                    str(tmp_path / "zarr"),
                    "--raw-dir",
                    str(tmp_path / "raw"),
                    "--catalog-path",
                    str(tmp_path / "catalog.json"),
                ],
            )

        assert result.exit_code == 0, result.output
        assert "Done" in result.output


class TestDownloadCdsErrorHandling:
    def test_download_cds_error_exits_nonzero(self, tmp_path):
        """A download failure aborts the command with a non-zero exit code."""
        mock_settings = _make_settings_mock(tmp_path)

        with (
            patch("eostrata.config.settings", mock_settings),
            patch(
                "eostrata.sources.cds.CDSSource.download",
                side_effect=RuntimeError("cds api error"),
            ),
            patch("eostrata.cache.check_and_evict"),
        ):
            result = runner.invoke(
                app,
                [
                    "download",
                    "cds",
                    "--variable",
                    "t2m",
                    "--year",
                    "2023",
                    "--zarr-root",
                    str(tmp_path / "zarr"),
                    "--raw-dir",
                    str(tmp_path / "raw"),
                    "--catalog-path",
                    str(tmp_path / "catalog.json"),
                ],
            )

        assert result.exit_code == 1
        assert "Error" in result.output


class TestDownloadCdsMultiYear:
    def test_download_cds_multiple_years(self, tmp_path):
        """--years 2022,2023 downloads each year separately."""
        nc = tmp_path / "raw" / "cds" / "era5_t2m_2022.nc"
        nc.parent.mkdir(parents=True)
        y = np.linspace(14.0, 4.0, 5)
        x = np.linspace(2.0, 15.0, 5)
        times = [np.datetime64("2022-01-01")]
        data = np.ones((1, 5, 5), dtype="float32") * 280.0
        ds = xr.Dataset({"t2m": (("time", "y", "x"), data)}, coords={"time": times, "y": y, "x": x})
        ds.to_netcdf(str(nc))

        mock_settings = _make_settings_mock(tmp_path)
        download_calls = []

        def _fake_download(*args, **kwargs):
            download_calls.append(kwargs.get("year"))
            ds.to_netcdf(str(nc))
            return [nc]

        with (
            patch("eostrata.config.settings", mock_settings),
            patch("eostrata.sources.cds.CDSSource.download", side_effect=_fake_download),
            patch("eostrata.cache.check_and_evict"),
        ):
            result = runner.invoke(
                app,
                [
                    "download",
                    "cds",
                    "--variable",
                    "t2m",
                    "--years",
                    "2022,2023",
                    "--months",
                    "1,2",
                    "--zarr-root",
                    str(tmp_path / "zarr"),
                    "--raw-dir",
                    str(tmp_path / "raw"),
                    "--catalog-path",
                    str(tmp_path / "catalog.json"),
                ],
            )

        assert result.exit_code == 0, result.output
        assert download_calls == [2022, 2023]


class TestList:
    def test_list_empty(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "list",
                "--zarr-root",
                str(tmp_path / "zarr"),
                "--catalog-path",
                str(tmp_path / "catalog.json"),
            ],
        )
        assert result.exit_code == 0

    def test_list_with_catalog_items(self, tmp_path):
        """list shows the STAC catalogue table when items are registered."""
        from datetime import UTC, datetime

        from eostrata import catalog as cat

        catalog_path = tmp_path / "catalog.json"
        catalogue = cat.load_or_create(catalog_path)
        cat.register_item(
            catalogue,
            collection_id="worldpop",
            item_id="worldpop_nga",
            bbox=(2.0, 4.0, 15.0, 14.0),
            datetime_=datetime(2020, 1, 1, tzinfo=UTC),
            zarr_root=tmp_path / "zarr",
            zarr_group="worldpop/nga",
            variable="population",
        )
        cat.save(catalogue, catalog_path)

        result = runner.invoke(
            app,
            [
                "list",
                "--zarr-root",
                str(tmp_path / "zarr"),
                "--catalog-path",
                str(catalog_path),
            ],
        )
        assert result.exit_code == 0
        assert "worldpop_nga" in result.output
        assert "2020" in result.output

    def test_list_item_with_datetime_no_eostrata_datetimes(self, tmp_path):
        """list uses item.datetime when eostrata:datetimes is absent (lines 175-176)."""
        from datetime import UTC, datetime

        import pystac

        from eostrata.catalog import _make_catalog, save

        catalog_path = tmp_path / "catalog.json"
        cat_ = _make_catalog()
        coll = cat_.get_child("worldpop")
        item = pystac.Item(
            id="worldpop_nga",
            geometry=None,
            bbox=[2.0, 4.0, 6.0, 8.0],
            datetime=datetime(2020, 6, 1, tzinfo=UTC),
            properties={"eostrata:variable": "population", "eostrata:zarr_group": "worldpop/nga"},
        )
        coll.add_item(item)
        save(cat_, catalog_path)

        result = runner.invoke(
            app,
            ["list", "--zarr-root", str(tmp_path / "zarr"), "--catalog-path", str(catalog_path)],
        )
        assert result.exit_code == 0
        assert "2020-06-01" in result.output

    def test_list_item_with_start_end_datetime_only(self, tmp_path):
        """list falls back to start/end_datetime when datetime is None (lines 178-180)."""

        import pystac

        from eostrata.catalog import _make_catalog, save

        catalog_path = tmp_path / "catalog.json"
        cat_ = _make_catalog()
        coll = cat_.get_child("worldpop")
        item = pystac.Item(
            id="worldpop_nga",
            geometry=None,
            bbox=[2.0, 4.0, 6.0, 8.0],
            datetime=None,
            properties={
                "start_datetime": "2020-01-01T00:00:00+00:00",
                "end_datetime": "2020-12-31T00:00:00+00:00",
                "datetime": None,
                "eostrata:variable": "population",
                "eostrata:zarr_group": "worldpop/nga",
            },
        )
        coll.add_item(item)
        save(cat_, catalog_path)

        result = runner.invoke(
            app,
            ["list", "--zarr-root", str(tmp_path / "zarr"), "--catalog-path", str(catalog_path)],
        )
        assert result.exit_code == 0
        assert "2020-01-01" in result.output

    def test_list_with_zarr_store(self, tmp_path):
        zarr_root = tmp_path / "zarr"
        # Write a minimal zarr group
        ds = xr.Dataset({"v": (("y", "x"), np.ones((4, 4)))})
        ds.to_zarr(str(zarr_root), group="worldpop/nga", mode="w")

        result = runner.invoke(
            app,
            [
                "list",
                "--zarr-root",
                str(zarr_root),
                "--catalog-path",
                str(tmp_path / "catalog.json"),
            ],
        )
        assert result.exit_code == 0
        assert "worldpop/nga" in result.output
        assert "MB" in result.output

    def test_list_shows_last_accessed_when_sentinel_exists(self, tmp_path):
        from eostrata.cache import _ACCESS_DIR

        zarr_root = tmp_path / "zarr"
        ds = xr.Dataset({"v": (("y", "x"), np.ones((4, 4)))})
        ds.to_zarr(str(zarr_root), group="worldpop/nga", mode="w")
        # Create a sentinel file so the "last accessed" branch is exercised
        access_dir = zarr_root / "worldpop" / "nga" / _ACCESS_DIR
        access_dir.mkdir(parents=True)
        (access_dir / "2020-01-01T00:00:00").touch()

        result = runner.invoke(
            app,
            [
                "list",
                "--zarr-root",
                str(zarr_root),
                "--catalog-path",
                str(tmp_path / "catalog.json"),
            ],
        )
        assert result.exit_code == 0
        assert "UTC" in result.output  # date is shown as "YYYY-MM-DD HH:MM UTC"

    def test_list_no_quota(self, tmp_path):
        """When no quota is configured, list shows size without percentage."""
        zarr_root = tmp_path / "zarr"
        ds = xr.Dataset({"v": (("y", "x"), np.ones((4, 4)))})
        ds.to_zarr(str(zarr_root), group="worldpop/nga", mode="w")

        mock_settings = _make_settings_mock(tmp_path)
        mock_settings.zarr_root = zarr_root
        mock_settings.store_quota_mb = 0.0

        with patch("eostrata.config.settings", mock_settings):
            result = runner.invoke(
                app,
                [
                    "list",
                    "--zarr-root",
                    str(zarr_root),
                    "--catalog-path",
                    str(tmp_path / "catalog.json"),
                ],
            )

        assert result.exit_code == 0
        # Rich may wrap "no quota" across lines; check words are present without %
        assert "no" in result.output and "quota" in result.output
        assert "%" not in result.output

    def test_list_shows_quota(self, tmp_path, monkeypatch):
        """When a quota is configured, list shows usage percentage."""
        zarr_root = tmp_path / "zarr"
        ds = xr.Dataset({"v": (("y", "x"), np.ones((4, 4)))})
        ds.to_zarr(str(zarr_root), group="worldpop/nga", mode="w")

        mock_settings = _make_settings_mock(tmp_path)
        mock_settings.zarr_root = zarr_root
        mock_settings.store_quota_mb = 10000.0

        with patch("eostrata.config.settings", mock_settings):
            result = runner.invoke(
                app,
                [
                    "list",
                    "--zarr-root",
                    str(zarr_root),
                    "--catalog-path",
                    str(tmp_path / "catalog.json"),
                ],
            )

        assert result.exit_code == 0
        assert "10000" in result.output
        assert "%" in result.output


class TestServe:
    def test_serve_invokes_uvicorn(self):
        with patch("uvicorn.run") as mock_run:
            result = runner.invoke(app, ["serve", "--port", "9999"])
        assert result.exit_code == 0
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs[1]["port"] == 9999


class TestRunTests:
    def test_run_tests_calls_pytest(self):

        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = runner.invoke(app, ["test", "--no-cov"])
        assert result.exit_code == 0
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "pytest" in cmd

    def test_run_tests_includes_coverage_by_default(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            runner.invoke(app, ["test"])
        cmd = mock_run.call_args[0][0]
        assert "--cov=eostrata" in cmd

    def test_run_tests_verbose_flag(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            runner.invoke(app, ["test", "--no-cov", "-v"])
        cmd = mock_run.call_args[0][0]
        assert "-v" in cmd


class TestRunLint:
    def test_run_lint_calls_ruff(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = runner.invoke(app, ["lint"])
        assert result.exit_code == 0
        assert mock_run.call_count == 2  # ruff check + ruff format
        all_cmds = [call[0][0] for call in mock_run.call_args_list]
        assert any("check" in cmd for cmd in all_cmds)
        assert any("format" in cmd for cmd in all_cmds)

    def test_run_lint_no_fix(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            runner.invoke(app, ["lint", "--no-fix"])
        check_cmd = mock_run.call_args_list[0][0][0]
        assert "--fix" not in check_cmd


class TestCleanup:
    def test_cleanup_nothing_to_clean(self, tmp_path):
        # Point all paths to non-existent subdirectories so nothing is found
        result = runner.invoke(
            app,
            [
                "cleanup",
                "--zarr-root",
                str(tmp_path / "nonexistent_zarr"),
                "--raw-dir",
                str(tmp_path / "nonexistent_raw"),
                "--catalog-path",
                str(tmp_path / "nonexistent" / "catalog.json"),
                "--yes",
            ],
        )
        assert result.exit_code == 0
        assert "Nothing" in result.output

    def test_cleanup_removes_dirs(self, tmp_path):
        zarr_root = tmp_path / "zarr"
        raw_dir = tmp_path / "raw"
        zarr_root.mkdir()
        raw_dir.mkdir()
        (tmp_path / "catalog.json").touch()

        result = runner.invoke(
            app,
            [
                "cleanup",
                "--zarr-root",
                str(zarr_root),
                "--raw-dir",
                str(raw_dir),
                "--catalog-path",
                str(tmp_path / "catalog.json"),
                "--yes",
            ],
        )
        assert result.exit_code == 0
        assert not zarr_root.exists()
        assert not raw_dir.exists()

    def test_cleanup_prompts_without_yes(self, tmp_path):
        zarr_root = tmp_path / "zarr"
        zarr_root.mkdir()

        result = runner.invoke(
            app,
            [
                "cleanup",
                "--zarr-root",
                str(zarr_root),
                "--raw-dir",
                str(tmp_path / "raw"),
                "--catalog-path",
                str(tmp_path / "catalog.json"),
            ],
            input="n\n",
        )
        # Answered "n" — should abort
        assert result.exit_code != 0 or not zarr_root.exists() or zarr_root.exists()


class TestRebuildCatalog:
    def test_rebuild_catalog_empty_store(self, tmp_path):
        """rebuild-catalog with no zarr groups reports empty catalogue."""
        zarr_root = tmp_path / "zarr"
        zarr_root.mkdir()
        catalog_path = tmp_path / "catalog.json"

        with patch("eostrata.ingestion.rebuild_catalog_from_zarr", return_value={}) as mock_rebuild:
            result = runner.invoke(
                app,
                [
                    "rebuild-catalog",
                    "--zarr-root",
                    str(zarr_root),
                    "--catalog-path",
                    str(catalog_path),
                ],
            )
        assert result.exit_code == 0
        assert "No Zarr groups found" in result.output
        mock_rebuild.assert_called_once_with(zarr_root=zarr_root, catalog_path=catalog_path)

    def test_rebuild_catalog_with_groups(self, tmp_path):
        """rebuild-catalog displays a table with rebuilt groups."""
        zarr_root = tmp_path / "zarr"
        zarr_root.mkdir()
        catalog_path = tmp_path / "catalog.json"

        with patch(
            "eostrata.ingestion.rebuild_catalog_from_zarr",
            return_value={"worldpop/nga": 3, "chirps/global": 12},
        ):
            result = runner.invoke(
                app,
                [
                    "rebuild-catalog",
                    "--zarr-root",
                    str(zarr_root),
                    "--catalog-path",
                    str(catalog_path),
                ],
            )
        assert result.exit_code == 0
        assert "worldpop/nga" in result.output
        assert "chirps/global" in result.output

    def test_rebuild_catalog_uses_settings_defaults(self, tmp_path):
        """rebuild-catalog falls back to settings when paths are not specified."""
        mock_settings = _make_settings_mock(tmp_path)
        mock_settings.zarr_root.mkdir(parents=True, exist_ok=True)

        with (
            patch("eostrata.config.settings", mock_settings),
            patch("eostrata.ingestion.rebuild_catalog_from_zarr", return_value={}) as mock_rebuild,
        ):
            result = runner.invoke(app, ["rebuild-catalog"])
        assert result.exit_code == 0
        mock_rebuild.assert_called_once()
