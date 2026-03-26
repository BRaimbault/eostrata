"""CHIRPS v2.0 monthly precipitation raster source.

CHIRPS — Climate Hazards Group InfraRed Precipitation with Station data.
Monthly global precipitation at 0.05-degree (~5 km) resolution.
Coverage: 1981-present, 50°S–50°N.

Data URL pattern:
  https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/
      chirps-v2.0.{year}.{month:02d}.tif.gz
"""

from __future__ import annotations

import gzip
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from eostrata.sources.base import BaseSource, _stream_download, register_source
from eostrata.store import geotiff_to_zarr

logger = logging.getLogger(__name__)

_BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs"
_NODATA = -9999.0


def _build_url(year: int, month: int) -> str:
    return f"{_BASE_URL}/chirps-v2.0.{year}.{month:02d}.tif.gz"


def _decompress_gz(src: Path, dest: Path) -> Path:
    """Decompress *src* (.gz) to *dest* (.tif), skipping if dest already exists."""
    if dest.exists():
        logger.info("Already decompressed: %s", dest.name)
        return dest

    logger.info("Decompressing %s -> %s", src.name, dest.name)
    with gzip.open(src, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    logger.info("Decompressed: %s", dest)
    return dest


@register_source
class CHIRPSSource(BaseSource):
    """CHIRPS v2.0 monthly global precipitation at 0.05-degree resolution."""

    id = "chirps"
    collection_id = "chirps"
    temporal_resolution = "monthly"
    default_lag_days = 45
    VARIABLE = "precipitation"

    def download(
        self,
        raw_dir: Path,
        bbox: tuple[float, float, float, float],
        *,
        year: int,
        month: int,
        **_: Any,
    ) -> list[Path]:
        """Download and decompress the CHIRPS monthly GeoTIFF for *year*/*month*."""
        url = _build_url(year, month)
        gz_name = Path(url).name  # chirps-v2.0.YYYY.MM.tif.gz
        tif_name = gz_name[: -len(".gz")]  # chirps-v2.0.YYYY.MM.tif
        dest_dir = Path(raw_dir) / "chirps"
        dest_gz = dest_dir / gz_name
        dest_tif = dest_dir / tif_name

        if dest_tif.exists():
            logger.info("Already available: %s", dest_tif.name)
            return [dest_tif]

        _stream_download(url, dest_gz)
        _decompress_gz(dest_gz, dest_tif)
        # Remove the gz to save space — the TIF is authoritative
        dest_gz.unlink(missing_ok=True)
        return [dest_tif]

    def to_zarr(
        self,
        path: Path,
        zarr_root: Path,
        bbox: tuple[float, float, float, float],
        *,
        year: int,
        month: int,
        **_: Any,
    ) -> Any:  # xr.Dataset
        """Clip *path* to *bbox* and append to the global CHIRPS Zarr group."""
        time_coord = np.datetime64(f"{year}-{month:02d}-01", "ns")
        return geotiff_to_zarr(
            path,
            zarr_root,
            self.zarr_group(),
            bbox=bbox,
            time_coord=time_coord,
            variable_name=self.VARIABLE,
        )

    def zarr_group(self, **_: Any) -> str:
        """Single global Zarr group — all months as timesteps."""
        return "chirps/global"

    def stac_item_id(self, **_: Any) -> str:
        """Single STAC item for the global CHIRPS collection."""
        return "chirps_global"

    def stac_properties(self, *, year: int, month: int, **_: Any) -> dict:
        return {
            "eostrata:variable": self.VARIABLE,
            "eostrata:resolution": "0.05deg",
            "eostrata:release": "v2.0",
            "eostrata:units": "mm/month",
        }

    def latest_available(self) -> datetime:
        """CHIRPS monthly data lags ~45 days; return 2 months prior to today."""
        now = datetime.now(tz=UTC)
        month = now.month - 2
        year = now.year
        if month <= 0:
            month += 12
            year -= 1
        return datetime(year, month, 1, tzinfo=UTC)
