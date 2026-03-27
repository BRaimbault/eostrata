"""Tests for download helpers in worldpop and chirps sources.

These tests mock the HTTP layer so no real network calls are made.
"""

from __future__ import annotations

import gzip
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from eostrata.sources.base import _stream_download
from eostrata.sources.chirps import CHIRPSSource, _decompress_gz

# ── Shared helpers ─────────────────────────────────────────────────────────────


def _make_tif_bytes(bbox=(0.0, 0.0, 5.0, 5.0), width=4, height=4) -> bytes:
    """Return raw bytes of a minimal in-memory GeoTIFF."""
    import os
    import tempfile

    transform = from_bounds(*bbox, width=width, height=height)
    # rasterio can't write to BytesIO directly for GTiff; use a tmp file
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        tmp = f.name
    with rasterio.open(
        tmp,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(np.ones((height, width), dtype="float32"), 1)
    with open(tmp, "rb") as f:
        data = f.read()
    os.unlink(tmp)
    return data


def _make_httpx_stream_mock(content: bytes):
    """Return a context manager mock that yields chunks of *content*."""
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.raise_for_status = MagicMock()
    mock_resp.headers = {"content-length": str(len(content))}
    mock_resp.iter_bytes = MagicMock(return_value=iter([content]))
    return mock_resp


# ── _stream_download ──────────────────────────────────────────────────────────


class TestStreamDownload:
    def test_skips_if_already_exists(self, tmp_path):
        dest = tmp_path / "test.tif"
        dest.write_bytes(b"existing")
        result = _stream_download("http://example.com/test.tif", dest)
        assert result == dest

    def test_downloads_and_saves(self, tmp_path):
        dest = tmp_path / "test.tif"
        tif_bytes = _make_tif_bytes()

        mock_stream = _make_httpx_stream_mock(tif_bytes)
        with patch("eostrata.sources.base.httpx.stream", return_value=mock_stream):
            result = _stream_download("http://example.com/test.tif", dest)

        assert result == dest
        assert dest.exists()
        assert dest.stat().st_size == len(tif_bytes)

    def test_raises_on_http_error(self, tmp_path):
        dest = tmp_path / "fail.tif"

        mock_stream = MagicMock()
        mock_stream.__enter__ = lambda s: s
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.raise_for_status = MagicMock(side_effect=Exception("404 Not Found"))

        with (
            patch("eostrata.sources.base.httpx.stream", return_value=mock_stream),
            pytest.raises(Exception, match="404"),
        ):
            _stream_download("http://example.com/missing.tif", dest)

    def test_transport_error_retries_then_raises(self, tmp_path):
        """TransportError triggers retries; after all attempts the error is re-raised."""
        dest = tmp_path / "retry_fail.tif"
        err = httpx.TransportError("connection reset")

        with (
            patch(
                "eostrata.sources.base.httpx.stream",
                side_effect=err,
            ),
            patch("eostrata.sources.base.time.sleep"),
            pytest.raises(httpx.TransportError),
        ):
            _stream_download("http://example.com/test.tif", dest)

        # Partial file must be cleaned up after each failed attempt
        assert not dest.exists()

    def test_transport_error_first_attempt_then_success(self, tmp_path):
        """First attempt raises TransportError; second attempt succeeds."""
        dest = tmp_path / "retry_ok.tif"
        tif_bytes = _make_tif_bytes()
        err = httpx.TransportError("timeout")
        ok_stream = _make_httpx_stream_mock(tif_bytes)

        call_count = 0

        def _stream_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise err
            return ok_stream

        with (
            patch("eostrata.sources.base.httpx.stream", side_effect=_stream_side_effect),
            patch("eostrata.sources.base.time.sleep"),
        ):
            result = _stream_download("http://example.com/test.tif", dest)

        assert result == dest
        assert dest.exists()
        assert call_count == 2


class TestWorldPopDownloadIntegration:
    def test_download_calls_http(self, tmp_path):
        """WorldPopSource.download should call _stream_download."""
        from eostrata.sources.worldpop import WorldPopSource

        source = WorldPopSource()
        tif_bytes = _make_tif_bytes()
        mock_stream = _make_httpx_stream_mock(tif_bytes)

        with patch("eostrata.sources.base.httpx.stream", return_value=mock_stream):
            paths = source.download(tmp_path, (0, 0, 10, 10), iso3="NGA", year=2020)

        assert len(paths) == 1
        assert paths[0].exists()


# ── CHIRPS download helpers ────────────────────────────────────────────────────


class TestCHIRPSDownloadHelpers:
    def test_download_gz_skips_if_exists(self, tmp_path):
        dest = tmp_path / "chirps.tif.gz"
        dest.write_bytes(b"existing gz")
        result = _stream_download("http://example.com/data.tif.gz", dest)
        assert result == dest

    def test_download_gz_saves_file(self, tmp_path):
        dest = tmp_path / "chirps.tif.gz"
        tif_bytes = _make_tif_bytes()
        gz_bytes = gzip.compress(tif_bytes)
        mock_stream = _make_httpx_stream_mock(gz_bytes)

        with patch("eostrata.sources.base.httpx.stream", return_value=mock_stream):
            result = _stream_download("http://example.com/data.tif.gz", dest)

        assert result == dest
        assert dest.exists()

    def test_decompress_gz_skips_if_tif_exists(self, tmp_path):
        src = tmp_path / "data.tif.gz"
        dest = tmp_path / "data.tif"
        dest.write_bytes(b"existing tif")
        result = _decompress_gz(src, dest)
        assert result == dest

    def test_decompress_gz_extracts(self, tmp_path):
        tif_bytes = _make_tif_bytes()
        src = tmp_path / "data.tif.gz"
        dest = tmp_path / "data.tif"
        src.write_bytes(gzip.compress(tif_bytes))

        result = _decompress_gz(src, dest)
        assert result == dest
        assert dest.read_bytes() == tif_bytes


class TestCHIRPSSourceDownloadIntegration:
    def test_download_fetches_and_decompresses(self, tmp_path):
        source = CHIRPSSource()
        tif_bytes = _make_tif_bytes()
        gz_bytes = gzip.compress(tif_bytes)
        mock_stream = _make_httpx_stream_mock(gz_bytes)

        with patch("eostrata.sources.base.httpx.stream", return_value=mock_stream):
            paths = source.download(tmp_path, (0, 0, 10, 10), year=2023, month=6)

        assert len(paths) == 1
        assert paths[0].suffix == ".tif"
        assert paths[0].exists()
        # gz file should have been removed
        assert not paths[0].with_suffix(".tif.gz").exists()

    def test_download_returns_existing_tif(self, tmp_path):
        """If the .tif exists, no network call is made."""
        source = CHIRPSSource()
        tif = tmp_path / "chirps" / "chirps-v2.0.2023.06.tif"
        tif.parent.mkdir(parents=True)
        tif.write_bytes(b"fake tif")

        with patch("eostrata.sources.base.httpx.stream") as mock_http:
            paths = source.download(tmp_path, (0, 0, 10, 10), year=2023, month=6)

        mock_http.assert_not_called()
        assert paths[0] == tif
