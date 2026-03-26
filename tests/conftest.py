"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds


@pytest.fixture()
def fake_geotiff(tmp_path):
    """Create a minimal GeoTIFF clipped to a small bbox and return its path."""
    bbox = (2.0, 4.0, 6.0, 8.0)
    tif_path = tmp_path / "test.tif"
    transform = from_bounds(*bbox, width=20, height=20)
    data = (np.random.rand(20, 20) * 1000).astype("float32")

    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=20,
        width=20,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)

    return tif_path, bbox
