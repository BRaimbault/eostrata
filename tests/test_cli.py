"""Tests for the eostrata CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_bounds
from typer.testing import CliRunner

from eostrata.cli import _parse_months, _parse_years, app

runner = CliRunner()


class TestParseHelpers:
    def test_parse_years_single(self):
        assert _parse_years(2020, None, 2025) == [2020]

    def test_parse_years_default(self):
        assert _parse_years(None, None, 2025) == [2025]

    def test_parse_years_multi(self):
        assert _parse_years(None, "2020,2021,2022", 2025) == [2020, 2021, 2022]

    def test_parse_years_multi_deduplicates_and_sorts(self):
        assert _parse_years(None, "2022,2020,2021,2020", 2025) == [2020, 2021, 2022]

    def test_parse_months_single(self):
        assert _parse_months(6, None, 1) == [6]

    def test_parse_months_default(self):
        assert _parse_months(None, None, 3) == [3]

    def test_parse_months_multi(self):
        assert _parse_months(None, "1,2,3", 12) == [1, 2, 3]

    def test_parse_months_multi_sorted(self):
        assert _parse_months(None, "3,1,2", 12) == [1, 2, 3]


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
        _make_tif(tif)

        mock_settings = _make_settings_mock(tmp_path)
        download_calls = []

        def _fake_download(*args, **kwargs):
            download_calls.append((kwargs.get("year"), kwargs.get("month")))
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
                ["list", "--zarr-root", str(zarr_root), "--catalog-path", str(tmp_path / "catalog.json")],
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
