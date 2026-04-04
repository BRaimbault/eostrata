"""Tests for the WorldPop source."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from eostrata.constants import PROP_VARIABLE
from eostrata.sources.worldpop import WorldPopSource, _build_url


class TestBuildUrl:
    def test_format(self):
        url = _build_url("NGA", 2020)
        assert "NGA" in url
        assert "nga" in url
        assert "2020" in url
        assert url.endswith(".tif")

    def test_case_normalisation(self):
        url_upper = _build_url("NGA", 2020)
        url_lower = _build_url("nga", 2020)
        assert url_upper == url_lower

    def test_r2025a_release(self):
        url = _build_url("NGA", 2020)
        assert "R2025A" in url


class TestWorldPopSource:
    def setup_method(self):
        self.source = WorldPopSource()

    def test_metadata(self):
        assert self.source.id == "worldpop"
        assert self.source.collection_id == "worldpop"
        assert self.source.temporal_resolution == "annual"
        assert self.source.VARIABLE == "population"

    def test_zarr_group(self):
        assert self.source.zarr_group(iso3="NGA") == "worldpop/nga"
        assert self.source.zarr_group(iso3="nga") == "worldpop/nga"

    def test_stac_item_id(self):
        assert self.source.stac_item_id(iso3="NGA") == "nga"
        assert self.source.stac_item_id(iso3="nga") == "nga"

    def test_stac_properties(self):
        props = self.source.stac_properties(iso3="NGA", year=2020)
        assert props["eostrata:iso3"] == "NGA"
        assert props[PROP_VARIABLE] == "population"

    def test_latest_available_is_previous_year(self):
        latest = self.source.latest_available()
        current_year = datetime.now(tz=UTC).year
        assert latest.year == current_year - 1

    def test_latest_available_has_timezone(self):
        latest = self.source.latest_available()
        assert latest.tzinfo is not None

    def test_download_skips_if_file_exists(self, tmp_path):
        """No HTTP request when file is already present."""
        dest = tmp_path / "worldpop" / "nga_pop_2020_CN_1km_R2025A_UA_v1.tif"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"fake tif")

        paths = self.source.download(tmp_path, (0, 0, 10, 10), iso3="NGA", year=2020)
        assert paths[0] == dest

    def test_to_zarr_writes_group(self, tmp_path):
        tif = tmp_path / "test.tif"
        bbox = (2.0, 4.0, 6.0, 8.0)
        transform = from_bounds(*bbox, width=10, height=10)
        data = np.random.rand(10, 10).astype("float32")
        with rasterio.open(
            tif,
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
        ds = self.source.to_zarr(tif, zarr_root, bbox, iso3="NGA", year=2021)
        assert "population" in ds.data_vars
        assert "time" in ds.dims
        assert (zarr_root / "worldpop" / "nga").exists()
