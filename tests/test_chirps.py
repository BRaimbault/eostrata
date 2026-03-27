"""Tests for the CHIRPS source."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import numpy as np

from eostrata.sources.chirps import CHIRPSSource, _build_url


class TestBuildUrl:
    def test_format(self):
        url = _build_url(2023, 6)
        assert "chirps-v2.0.2023.06.tif.gz" in url
        assert "global_monthly" in url

    def test_zero_padded_month(self):
        assert "2024.01" in _build_url(2024, 1)
        assert "2024.12" in _build_url(2024, 12)


class TestCHIRPSSource:
    def setup_method(self):
        self.source = CHIRPSSource()

    def test_metadata(self):
        assert self.source.id == "chirps"
        assert self.source.collection_id == "chirps"
        assert self.source.temporal_resolution == "monthly"
        assert self.source.VARIABLE == "precipitation"

    def test_zarr_group(self):
        assert self.source.zarr_group() == "chirps/global"

    def test_stac_item_id(self):
        assert self.source.stac_item_id() == "chirps_global"

    def test_stac_properties(self):
        props = self.source.stac_properties(year=2023, month=6)
        assert props["eostrata:variable"] == "precipitation"
        assert "mm/month" in props["eostrata:units"]

    def test_latest_available_is_in_past(self):
        latest = self.source.latest_available()
        assert latest < datetime.now(tz=UTC)

    def test_latest_available_wraps_year_in_january(self):
        """When current month is January (month-2 <= 0), wraps to previous year."""
        with patch("eostrata.sources.chirps.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 2, 15, tzinfo=UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = CHIRPSSource().latest_available()
        assert result.year == 2023
        assert result.month == 12

    def test_latest_available_is_datetime_with_tz(self):
        latest = self.source.latest_available()
        assert isinstance(latest, datetime)
        assert latest.tzinfo is not None

    def test_download_skips_if_tif_exists(self, tmp_path):
        """If the .tif already exists, no HTTP request should be made."""
        dest_tif = tmp_path / "chirps" / "chirps-v2.0.2023.06.tif"
        dest_tif.parent.mkdir(parents=True)
        dest_tif.write_bytes(b"fake tif data")

        paths = self.source.download(tmp_path, (0, 0, 10, 10), year=2023, month=6)
        assert paths == [dest_tif]

    def test_to_zarr_writes_group(self, tmp_path):
        """to_zarr should produce a Zarr group at chirps/global."""
        import rasterio
        from rasterio.transform import from_bounds

        # Create a minimal GeoTIFF for testing
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
            nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)

        zarr_root = tmp_path / "zarr"
        ds = self.source.to_zarr(tif_path, zarr_root, bbox, year=2023, month=6)

        assert "precipitation" in ds
        assert "time" in ds.dims
        assert (zarr_root / "chirps" / "global").exists()
