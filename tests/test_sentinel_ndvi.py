"""Tests for the Sentinel NDVI (CGLS v3 via Sentinel Hub BYOC) source."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
import rasterio
import rasterio.transform

from eostrata.constants import PROP_RESOLUTION, PROP_SOURCE, PROP_VARIABLE
from eostrata.sources.sentinel_ndvi import (
    SentinelNDVISource,
    _BYOC_COLLECTION_ID,
    _end_day_of_dekad,
    _fetch_ndvi_geotiff,
    _get_cdse_token,
)


# ── _end_day_of_dekad() ────────────────────────────────────────────────────────


class TestEndDayOfDekad:
    def test_dekad1(self):
        assert _end_day_of_dekad(2023, 6, 1) == 10

    def test_dekad2(self):
        assert _end_day_of_dekad(2023, 6, 2) == 20

    def test_dekad3_june(self):
        assert _end_day_of_dekad(2023, 6, 3) == 30

    def test_dekad3_february_leap(self):
        assert _end_day_of_dekad(2024, 2, 3) == 29

    def test_dekad3_february_non_leap(self):
        assert _end_day_of_dekad(2023, 2, 3) == 28

    def test_dekad3_january(self):
        assert _end_day_of_dekad(2023, 1, 3) == 31


# ── _get_cdse_token() ──────────────────────────────────────────────────────────


class TestGetCdseToken:
    def test_returns_access_token(self, mocker):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"access_token": "mytoken123"}
        mocker.patch("eostrata.sources.sentinel_ndvi.httpx.post", return_value=mock_resp)

        token = _get_cdse_token("user@example.com", "secret")
        assert token == "mytoken123"

    def test_sends_correct_payload(self, mocker):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"access_token": "tok"}
        mock_post = mocker.patch(
            "eostrata.sources.sentinel_ndvi.httpx.post", return_value=mock_resp
        )

        _get_cdse_token("u", "p")
        data = mock_post.call_args.kwargs["data"]
        assert data["grant_type"] == "password"
        assert data["username"] == "u"
        assert data["password"] == "p"
        assert data["client_id"] == "cdse-public"


# ── _fetch_ndvi_geotiff() ──────────────────────────────────────────────────────


def _make_sh_stream_mock(content: bytes, content_type: str = "image/tiff"):
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.raise_for_status = MagicMock()
    mock_resp.headers = {"content-type": content_type}
    mock_resp.iter_bytes = MagicMock(return_value=iter([content]))
    mock_resp.read = MagicMock(return_value=content)
    return mock_resp


class TestFetchNdviGeotiff:
    def test_saves_tiff_to_dest(self, tmp_path, mocker):
        content = b"TIFF_FAKE_BYTES"
        mock_resp = _make_sh_stream_mock(content)
        mocker.patch("eostrata.sources.sentinel_ndvi.httpx.stream", return_value=mock_resp)

        dest = tmp_path / "ndvi.tif"
        result = _fetch_ndvi_geotiff((0, 0, 1, 1), "2023-06-01", "2023-06-10", dest, "tok")

        assert result == dest
        assert dest.read_bytes() == content

    def test_sends_bearer_token(self, tmp_path, mocker):
        mock_resp = _make_sh_stream_mock(b"data")
        mock_stream = mocker.patch(
            "eostrata.sources.sentinel_ndvi.httpx.stream", return_value=mock_resp
        )

        _fetch_ndvi_geotiff((0, 0, 1, 1), "2023-06-01", "2023-06-10", tmp_path / "f.tif", "mytoken")
        headers = mock_stream.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer mytoken"

    def test_sends_correct_collection_id(self, tmp_path, mocker):
        mock_resp = _make_sh_stream_mock(b"data")
        mock_stream = mocker.patch(
            "eostrata.sources.sentinel_ndvi.httpx.stream", return_value=mock_resp
        )

        _fetch_ndvi_geotiff((0, 0, 1, 1), "2023-06-01", "2023-06-10", tmp_path / "f.tif", "tok")
        payload = mock_stream.call_args.kwargs["json"]
        data_type = payload["input"]["data"][0]["type"]
        assert data_type == f"byoc-{_BYOC_COLLECTION_ID}"

    def test_sends_correct_time_range(self, tmp_path, mocker):
        mock_resp = _make_sh_stream_mock(b"data")
        mock_stream = mocker.patch(
            "eostrata.sources.sentinel_ndvi.httpx.stream", return_value=mock_resp
        )

        _fetch_ndvi_geotiff((0, 0, 1, 1), "2023-06-21", "2023-06-30", tmp_path / "f.tif", "tok")
        payload = mock_stream.call_args.kwargs["json"]
        time_range = payload["input"]["data"][0]["dataFilter"]["timeRange"]
        assert time_range["from"] == "2023-06-21T00:00:00Z"
        assert time_range["to"] == "2023-06-30T23:59:59Z"

    def test_sends_correct_bbox(self, tmp_path, mocker):
        mock_resp = _make_sh_stream_mock(b"data")
        mock_stream = mocker.patch(
            "eostrata.sources.sentinel_ndvi.httpx.stream", return_value=mock_resp
        )

        _fetch_ndvi_geotiff((2.0, 4.0, 14.0, 13.0), "2023-06-01", "2023-06-10", tmp_path / "f.tif", "tok")
        payload = mock_stream.call_args.kwargs["json"]
        assert payload["input"]["bounds"]["bbox"] == [2.0, 4.0, 14.0, 13.0]

    def test_raises_on_json_error_response(self, tmp_path, mocker):
        error_body = b'{"error": {"status": 400, "reason": "Bad Request"}}'
        mock_resp = _make_sh_stream_mock(error_body, content_type="application/json")
        mocker.patch("eostrata.sources.sentinel_ndvi.httpx.stream", return_value=mock_resp)

        with pytest.raises(RuntimeError, match="unexpected content-type"):
            _fetch_ndvi_geotiff((0, 0, 1, 1), "2023-06-01", "2023-06-10", tmp_path / "f.tif", "tok")

    def test_raises_on_html_response(self, tmp_path, mocker):
        mock_resp = _make_sh_stream_mock(b"<html>login</html>", content_type="text/html")
        mocker.patch("eostrata.sources.sentinel_ndvi.httpx.stream", return_value=mock_resp)

        with pytest.raises(RuntimeError, match="unexpected content-type"):
            _fetch_ndvi_geotiff((0, 0, 1, 1), "2023-06-01", "2023-06-10", tmp_path / "f.tif", "tok")


# ── SentinelNDVISource metadata ────────────────────────────────────────────────


class TestSentinelNDVISource:
    def setup_method(self):
        self.source = SentinelNDVISource()

    def test_metadata(self):
        assert self.source.id == "cgls"
        assert self.source.collection_id == "cgls"
        assert self.source.temporal_resolution == "dekadal"
        assert self.source.VARIABLE == "ndvi"

    def test_zarr_group(self):
        assert self.source.zarr_group() == "cgls/global"

    def test_stac_item_id(self):
        assert self.source.stac_item_id() == "global"

    def test_stac_properties_dekad1(self):
        props = self.source.stac_properties(year=2023, month=6, dekad=1)
        assert props[PROP_VARIABLE] == "ndvi"
        assert "300m" in props[PROP_RESOLUTION]
        assert "Sentinel-3" in props[PROP_SOURCE]
        assert props["eostrata:period"] == "2023-06-01/2023-06-10"
        assert props["eostrata:release"] == "v3"
        assert props["eostrata:byoc_collection"] == _BYOC_COLLECTION_ID

    def test_stac_properties_dekad3(self):
        props = self.source.stac_properties(year=2023, month=6, dekad=3)
        assert props["eostrata:period"] == "2023-06-21/2023-06-30"

    def test_stac_properties_default_dekad(self):
        props = self.source.stac_properties(year=2023, month=1)
        assert props["eostrata:period"] == "2023-01-01/2023-01-10"

    def test_iter_periods(self):
        periods = list(SentinelNDVISource.iter_periods(years=[2024], months=[3], dekads=[1, 2]))
        assert periods == [
            ("2024-03-d1", {"year": 2024, "month": 3, "dekad": 1}),
            ("2024-03-d2", {"year": 2024, "month": 3, "dekad": 2}),
        ]

    def test_stac_registrations(self):
        regs = self.source.stac_registrations(None, {"year": 2024, "month": 3, "dekad": 1})
        assert len(regs) == 1
        reg = regs[0]
        assert reg["variable"] == "ndvi"
        assert reg["datetime_"] == datetime(2024, 3, 1, tzinfo=UTC)
        assert reg["item_id"] == "global"

    def test_latest_available_is_in_past(self):
        latest = self.source.latest_available()
        assert latest < datetime.now(tz=UTC)

    def test_latest_available_has_timezone(self):
        latest = self.source.latest_available()
        assert latest.tzinfo is not None

    def test_latest_available_day_is_1_11_or_21(self):
        latest = self.source.latest_available()
        assert latest.day in (1, 11, 21)

    def test_latest_available_mid_month(self, mocker):
        """From the 15th, dekad 1 of the current month should be available."""
        mock_dt = mocker.patch("eostrata.sources.sentinel_ndvi.datetime")
        mock_dt.now.return_value = datetime(2024, 6, 20, tzinfo=UTC)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = SentinelNDVISource().latest_available()
        assert result == datetime(2024, 6, 1, tzinfo=UTC)

    def test_latest_available_early_month_wraps_to_previous(self, mocker):
        """Before the 5th, return dekad 2 of the previous month."""
        mock_dt = mocker.patch("eostrata.sources.sentinel_ndvi.datetime")
        mock_dt.now.return_value = datetime(2024, 3, 3, tzinfo=UTC)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = SentinelNDVISource().latest_available()
        assert result == datetime(2024, 2, 11, tzinfo=UTC)

    def test_latest_available_january_wraps_to_december(self, mocker):
        """Early January → previous year December."""
        mock_dt = mocker.patch("eostrata.sources.sentinel_ndvi.datetime")
        mock_dt.now.return_value = datetime(2024, 1, 2, tzinfo=UTC)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = SentinelNDVISource().latest_available()
        assert result.year == 2023
        assert result.month == 12


# ── download() ─────────────────────────────────────────────────────────────────


class TestDownloadMethod:
    def setup_method(self):
        self.source = SentinelNDVISource()

    def _patch_cdse(self, mocker, tif_content: bytes = b"TIFF"):
        mocker.patch(
            "eostrata.sources.sentinel_ndvi._get_cdse_token", return_value="fake-token"
        )
        mock_resp = _make_sh_stream_mock(tif_content)
        mocker.patch("eostrata.sources.sentinel_ndvi.httpx.stream", return_value=mock_resp)

    def _patch_settings(self, mocker, user: str = "u", password: str = "p"):
        mock_settings = MagicMock()
        mock_settings.cdse_user = user
        mock_settings.cdse_password = password
        mocker.patch("eostrata.sources.sentinel_ndvi.settings", mock_settings, create=True)
        # Also patch within the download() local import path
        mocker.patch("eostrata.config.settings", mock_settings)

    def test_skips_if_tif_exists(self, tmp_path, mocker):
        dest = tmp_path / "cgls" / "ndvi_300m_v3_20230601.tif"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"existing")

        self._patch_settings(mocker)
        mock_token = mocker.patch("eostrata.sources.sentinel_ndvi._get_cdse_token")
        paths = self.source.download(tmp_path, (0, 0, 10, 10), year=2023, month=6, dekad=1)
        assert paths == [dest]
        mock_token.assert_not_called()

    def test_filename_encodes_dekad_start_day(self, tmp_path, mocker):
        """Each dekad gets a distinct filename based on its start day."""
        self._patch_settings(mocker)
        for start_day in (1, 11, 21):
            dest = tmp_path / "cgls" / f"ndvi_300m_v3_202306{start_day:02d}.tif"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"fake")

        assert self.source.download(tmp_path, (0, 0, 1, 1), year=2023, month=6, dekad=1) == [
            tmp_path / "cgls" / "ndvi_300m_v3_20230601.tif"
        ]
        assert self.source.download(tmp_path, (0, 0, 1, 1), year=2023, month=6, dekad=2) == [
            tmp_path / "cgls" / "ndvi_300m_v3_20230611.tif"
        ]
        assert self.source.download(tmp_path, (0, 0, 1, 1), year=2023, month=6, dekad=3) == [
            tmp_path / "cgls" / "ndvi_300m_v3_20230621.tif"
        ]

    def test_raises_without_credentials(self, tmp_path, mocker):
        self._patch_settings(mocker, user="", password="")
        with pytest.raises(RuntimeError, match="CDSE credentials"):
            self.source.download(tmp_path, (0, 0, 1, 1), year=2023, month=6, dekad=1)

    def test_download_fetches_and_returns_tif(self, tmp_path, mocker):
        self._patch_settings(mocker)
        self._patch_cdse(mocker, b"TIFF_DATA")
        paths = self.source.download(tmp_path, (0, 0, 10, 10), year=2023, month=6, dekad=1)

        assert len(paths) == 1
        assert paths[0].suffix == ".tif"
        assert paths[0].read_bytes() == b"TIFF_DATA"


# ── to_zarr() ──────────────────────────────────────────────────────────────────


def _write_test_geotiff(path: Path, bbox: tuple, nx: int = 20, ny: int = 20) -> None:
    """Write a minimal GeoTIFF FLOAT32 file for testing to_zarr."""
    west, south, east, north = bbox
    transform = rasterio.transform.from_bounds(west, south, east, north, nx, ny)
    data = np.random.rand(ny, nx).astype("float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=ny,
        width=nx,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


class TestToZarr:
    def setup_method(self):
        self.source = SentinelNDVISource()

    def test_to_zarr_writes_group(self, tmp_path):
        tif_path = tmp_path / "ndvi.tif"
        bbox = (0.0, 0.0, 5.0, 5.0)
        _write_test_geotiff(tif_path, bbox)

        zarr_root = tmp_path / "zarr"
        ds = self.source.to_zarr(tif_path, zarr_root, bbox, year=2023, month=6, dekad=1)

        assert "ndvi" in ds
        assert "time" in ds.dims
        assert (zarr_root / "cgls" / "global").exists()

    def test_to_zarr_timestep_is_dekad_start(self, tmp_path):
        import pandas as pd

        tif_path = tmp_path / "ndvi.tif"
        bbox = (0.0, 0.0, 5.0, 5.0)
        _write_test_geotiff(tif_path, bbox)

        zarr_root = tmp_path / "zarr"
        ds = self.source.to_zarr(tif_path, zarr_root, bbox, year=2023, month=6, dekad=2)

        ts = pd.Timestamp(ds.time.values[0])
        assert ts.year == 2023
        assert ts.month == 6
        assert ts.day == 11  # dekad 2 starts on the 11th

    def test_to_zarr_variable_named_ndvi(self, tmp_path):
        tif_path = tmp_path / "ndvi.tif"
        bbox = (0.0, 0.0, 5.0, 5.0)
        _write_test_geotiff(tif_path, bbox)

        zarr_root = tmp_path / "zarr"
        ds = self.source.to_zarr(tif_path, zarr_root, bbox, year=2024, month=1, dekad=3)
        assert "ndvi" in ds.data_vars


# ── latest_available() mid-month branch ───────────────────────────────────────


class TestLatestAvailableMidMonth:
    def test_latest_available_day_5_to_14_returns_dekad3_prev_month(self, mocker):
        """Between day 5 and 14 inclusive, returns dekad 3 (day 21) of the previous month."""
        mock_dt = mocker.patch("eostrata.sources.sentinel_ndvi.datetime")
        mock_dt.now.return_value = datetime(2024, 4, 10, tzinfo=UTC)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = SentinelNDVISource().latest_available()
        assert result == datetime(2024, 3, 21, tzinfo=UTC)


# ── is_configured() ───────────────────────────────────────────────────────────


class TestIsConfigured:
    def test_configured_with_cdse_credentials(self, mocker):
        mock_settings = MagicMock()
        mock_settings.cdse_user = "user@x.com"
        mock_settings.cdse_password = "pass"
        mocker.patch("eostrata.config.settings", mock_settings)
        ok, reason = SentinelNDVISource.is_configured()
        assert ok is True
        assert reason == ""

    def test_not_configured_without_credentials(self, mocker):
        mock_settings = MagicMock()
        mock_settings.cdse_user = ""
        mock_settings.cdse_password = ""
        mocker.patch("eostrata.config.settings", mock_settings)
        ok, reason = SentinelNDVISource.is_configured()
        assert ok is False
        assert "CDSE" in reason
