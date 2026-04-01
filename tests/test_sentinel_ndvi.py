"""Tests for the Sentinel NDVI (CGLS) source."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest

from eostrata.sources.sentinel_ndvi import (
    SentinelNDVISource,
    _build_wcs_url,
    _download,
    _end_day_of_dekad,
)


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


class TestBuildWcsUrl:
    def test_contains_wcs_params(self):
        url = _build_wcs_url((0.0, -10.0, 10.0, 10.0), 2023, 6, 1)
        assert "SERVICE=WCS" in url
        assert "REQUEST=GetCoverage" in url
        assert "VERSION=2.0.1" in url

    def test_contains_coverage_id(self):
        url = _build_wcs_url((0.0, -10.0, 10.0, 10.0), 2023, 6, 1)
        assert "ndvi_300m_v2_10daily" in url

    def test_dekad1_date_range(self):
        url = _build_wcs_url((0.0, -10.0, 10.0, 10.0), 2023, 6, 1)
        assert "2023-06-01T00:00:00.000Z" in url
        assert "2023-06-10T23:59:59.999Z" in url

    def test_dekad2_date_range(self):
        url = _build_wcs_url((0.0, -10.0, 10.0, 10.0), 2023, 6, 2)
        assert "2023-06-11T00:00:00.000Z" in url
        assert "2023-06-20T23:59:59.999Z" in url

    def test_dekad3_date_range(self):
        url = _build_wcs_url((0.0, -10.0, 10.0, 10.0), 2023, 6, 3)
        assert "2023-06-21T00:00:00.000Z" in url
        assert "2023-06-30T23:59:59.999Z" in url

    def test_bbox_in_url(self):
        url = _build_wcs_url((5.0, -3.0, 15.0, 7.0), 2023, 1, 1)
        assert "Lat(-3.0,7.0)" in url
        assert "Long(5.0,15.0)" in url

    def test_format_geotiff(self):
        url = _build_wcs_url((0.0, -10.0, 10.0, 10.0), 2023, 6, 1)
        assert "FORMAT=image/tiff" in url


class TestSentinelNDVISource:
    def setup_method(self):
        self.source = SentinelNDVISource()

    def test_metadata(self):
        assert self.source.id == "sentinel_ndvi"
        assert self.source.collection_id == "sentinel_ndvi"
        assert self.source.temporal_resolution == "dekadal"
        assert self.source.VARIABLE == "ndvi"

    def test_zarr_group(self):
        assert self.source.zarr_group() == "sentinel_ndvi/global"

    def test_stac_item_id(self):
        assert self.source.stac_item_id() == "sentinel_ndvi_global"

    def test_stac_properties_dekad1(self):
        props = self.source.stac_properties(year=2023, month=6, dekad=1)
        assert props["eostrata:variable"] == "ndvi"
        assert "300m" in props["eostrata:resolution"]
        assert "Sentinel-3" in props["eostrata:source"]
        assert props["eostrata:period"] == "2023-06-01/2023-06-10"

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
        assert reg["item_id"] == "sentinel_ndvi_global"

    def test_latest_available_is_in_past(self):
        latest = self.source.latest_available()
        assert latest < datetime.now(tz=UTC)

    def test_latest_available_has_timezone(self):
        latest = self.source.latest_available()
        assert latest.tzinfo is not None

    def test_latest_available_day_is_1_11_or_21(self):
        latest = self.source.latest_available()
        assert latest.day in (1, 11, 21)

    def test_latest_available_mid_month(self):
        """From the 15th, dekad 1 of the current month should be available."""
        with patch("eostrata.sources.sentinel_ndvi.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 6, 20, tzinfo=UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = SentinelNDVISource().latest_available()
        assert result == datetime(2024, 6, 1, tzinfo=UTC)

    def test_latest_available_early_month_wraps_to_previous(self):
        """Before the 5th, return dekad 2 of the previous month."""
        with patch("eostrata.sources.sentinel_ndvi.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 3, 3, tzinfo=UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = SentinelNDVISource().latest_available()
        assert result == datetime(2024, 2, 11, tzinfo=UTC)

    def test_latest_available_january_wraps_to_december(self):
        """Early January → previous year December."""
        with patch("eostrata.sources.sentinel_ndvi.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 2, tzinfo=UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = SentinelNDVISource().latest_available()
        assert result.year == 2023
        assert result.month == 12

    def test_download_skips_if_tif_exists(self, tmp_path):
        """If the .tif already exists locally, no HTTP request should be made."""
        dest_tif = tmp_path / "sentinel_ndvi" / "ndvi_300m_v2_20230601.tif"
        dest_tif.parent.mkdir(parents=True)
        dest_tif.write_bytes(b"fake tif data")

        paths = self.source.download(tmp_path, (0, 0, 10, 10), year=2023, month=6, dekad=1)
        assert paths == [dest_tif]

    def test_download_filename_encodes_dekad_start_day(self, tmp_path):
        """Each dekad gets a distinct filename based on its start day."""
        dest1 = tmp_path / "sentinel_ndvi" / "ndvi_300m_v2_20230601.tif"
        dest2 = tmp_path / "sentinel_ndvi" / "ndvi_300m_v2_20230611.tif"
        dest3 = tmp_path / "sentinel_ndvi" / "ndvi_300m_v2_20230621.tif"
        dest1.parent.mkdir(parents=True)
        for d in (dest1, dest2, dest3):
            d.write_bytes(b"fake")

        assert self.source.download(tmp_path, (0, 0, 1, 1), year=2023, month=6, dekad=1) == [dest1]
        assert self.source.download(tmp_path, (0, 0, 1, 1), year=2023, month=6, dekad=2) == [dest2]
        assert self.source.download(tmp_path, (0, 0, 1, 1), year=2023, month=6, dekad=3) == [dest3]

    def test_to_zarr_writes_group(self, tmp_path):
        """to_zarr should produce a Zarr group at sentinel_ndvi/global."""
        import rasterio
        from rasterio.transform import from_bounds

        tif_path = tmp_path / "test.tif"
        bbox = (0.0, 0.0, 5.0, 5.0)
        transform = from_bounds(*bbox, width=10, height=10)
        data = np.random.rand(10, 10).astype("float32")

        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=10,
            width=10,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data, 1)

        zarr_root = tmp_path / "zarr"
        ds = self.source.to_zarr(tif_path, zarr_root, bbox, year=2023, month=6, dekad=1)

        assert "ndvi" in ds
        assert "time" in ds.dims
        assert (zarr_root / "sentinel_ndvi" / "global").exists()

    def test_to_zarr_timestep_is_dekad_start(self, tmp_path):
        """The time coordinate should be the first day of the dekad."""
        import rasterio
        from rasterio.transform import from_bounds

        tif_path = tmp_path / "test.tif"
        bbox = (0.0, 0.0, 5.0, 5.0)
        transform = from_bounds(*bbox, width=10, height=10)
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=10,
            width=10,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(np.ones((10, 10), dtype="float32"), 1)

        zarr_root = tmp_path / "zarr"
        ds = self.source.to_zarr(tif_path, zarr_root, bbox, year=2023, month=6, dekad=2)

        import pandas as pd

        ts = pd.Timestamp(ds.time.values[0])
        assert ts.year == 2023
        assert ts.month == 6
        assert ts.day == 11  # dekad 2 starts on the 11th


# ── _download() unit tests ─────────────────────────────────────────────────────


def _make_httpx_stream_mock(content: bytes):
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.raise_for_status = MagicMock()
    mock_resp.headers = {"content-length": str(len(content))}
    mock_resp.iter_bytes = MagicMock(return_value=iter([content]))
    return mock_resp


class TestDownloadFunction:
    def test_streams_content_to_file(self, tmp_path):
        dest = tmp_path / "sub" / "out.tif"
        content = b"fake tif bytes"
        mock_resp = _make_httpx_stream_mock(content)

        with patch("eostrata.sources.sentinel_ndvi.httpx.stream", return_value=mock_resp):
            result = _download("http://example.com/ndvi.tif", dest)

        assert result == dest
        assert dest.read_bytes() == content

    def test_skips_if_file_already_exists(self, tmp_path):
        dest = tmp_path / "out.tif"
        dest.write_bytes(b"existing")

        with patch("eostrata.sources.sentinel_ndvi.httpx.stream") as mock_stream:
            result = _download("http://example.com/ndvi.tif", dest)

        mock_stream.assert_not_called()
        assert result == dest

    def test_sends_bearer_auth_header(self, tmp_path):
        dest = tmp_path / "out.tif"
        content = b"data"
        mock_resp = _make_httpx_stream_mock(content)

        with patch(
            "eostrata.sources.sentinel_ndvi.httpx.stream", return_value=mock_resp
        ) as mock_stream:
            _download("http://example.com/ndvi.tif", dest, api_key="mytoken")

        call_kwargs = mock_stream.call_args
        headers = call_kwargs.kwargs["headers"]
        assert headers["Authorization"] == "Bearer mytoken"

    def test_no_auth_header_without_api_key(self, tmp_path):
        dest = tmp_path / "out.tif"
        content = b"data"
        mock_resp = _make_httpx_stream_mock(content)

        with patch(
            "eostrata.sources.sentinel_ndvi.httpx.stream", return_value=mock_resp
        ) as mock_stream:
            _download("http://example.com/ndvi.tif", dest)

        headers = mock_stream.call_args.kwargs["headers"]
        assert "Authorization" not in headers

    def test_retries_on_transport_error_then_succeeds(self, tmp_path):
        dest = tmp_path / "out.tif"
        content = b"recovered"
        good_resp = _make_httpx_stream_mock(content)
        error = httpx.TransportError("connection reset")

        call_count = 0

        def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise error
            return good_resp

        with (
            patch("eostrata.sources.sentinel_ndvi.httpx.stream", side_effect=_side_effect),
            patch("eostrata.sources.sentinel_ndvi.time.sleep"),
        ):
            result = _download("http://example.com/ndvi.tif", dest)

        assert result == dest
        assert call_count == 2

    def test_raises_after_all_retries_exhausted(self, tmp_path):
        dest = tmp_path / "out.tif"
        error = httpx.TransportError("timeout")

        with (
            patch(
                "eostrata.sources.sentinel_ndvi.httpx.stream",
                side_effect=error,
            ),
            patch("eostrata.sources.sentinel_ndvi.time.sleep"),
            pytest.raises(httpx.TransportError),
        ):
            _download("http://example.com/ndvi.tif", dest)

    def test_download_method_calls_download_helper(self, tmp_path):
        """SentinelNDVISource.download() calls _download when file is missing."""
        source = SentinelNDVISource()
        content = b"tif bytes"
        mock_resp = _make_httpx_stream_mock(content)

        with (
            patch("eostrata.sources.sentinel_ndvi.httpx.stream", return_value=mock_resp),
            patch("eostrata.config.settings") as mock_settings,
        ):
            mock_settings.cgls_api_key = ""
            paths = source.download(tmp_path, (0, 0, 10, 10), year=2023, month=6, dekad=1)

        assert len(paths) == 1
        assert paths[0].exists()


# ── latest_available() mid-month branch ───────────────────────────────────────


class TestLatestAvailableMidMonth:
    def test_latest_available_day_5_to_14_returns_dekad3_prev_month(self):
        """Between day 5 and 14 inclusive, returns dekad 3 (day 21) of the previous month."""
        with patch("eostrata.sources.sentinel_ndvi.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 4, 10, tzinfo=UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = SentinelNDVISource().latest_available()
        assert result == datetime(2024, 3, 21, tzinfo=UTC)
