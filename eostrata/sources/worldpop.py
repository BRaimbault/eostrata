"""WorldPop population raster source."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from tqdm import tqdm

from eostrata.sources.base import BaseSource, register_source
from eostrata.store import geotiff_to_zarr

logger = logging.getLogger(__name__)

# URL pattern (R2025A release, 1km constrained unconstrained-adjusted):
# https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/{year}/
#   {ISO3}/v1/1km_ua/constrained/{iso3lower}_pop_{year}_CN_1km_R2025A_UA_v1.tif
# Example (Nigeria 2026):
# .../R2025A/2026/NGA/v1/1km_ua/constrained/nga_pop_2026_CN_1km_R2025A_UA_v1.tif

_BASE_URL = "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A"
_RELEASE = "R2025A"


def _build_url(iso3: str, year: int) -> str:
    """Build the WorldPop R2025A download URL from iso3 and year."""
    iso3_upper = iso3.upper()
    iso3_lower = iso3.lower()
    filename = f"{iso3_lower}_pop_{year}_CN_1km_{_RELEASE}_UA_v1.tif"
    return f"{_BASE_URL}/{year}/{iso3_upper}/v1/1km_ua/constrained/{filename}"


def _download_file(url: str, dest: Path) -> Path:
    """Download *url* to *dest*, skipping if already present."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        logger.info("Already downloaded: %s", dest.name)
        return dest

    logger.info("Downloading %s", url)
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with (
            open(dest, "wb") as fh,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            for chunk in resp.iter_bytes(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))

    logger.info("Saved to %s", dest)
    return dest


@register_source
class WorldPopSource(BaseSource):
    """WorldPop global population rasters at 1km constrained resolution (R2025A)."""

    id = "worldpop"
    collection_id = "worldpop"
    temporal_resolution = "annual"
    default_lag_days = 365
    VARIABLE = "population"

    def download(
        self,
        raw_dir: Path,
        bbox: tuple[float, float, float, float],
        *,
        iso3: str,
        year: int,
        **_: Any,
    ) -> list[Path]:
        """Download the WorldPop GeoTIFF for *iso3* / *year*."""
        url = _build_url(iso3, year)
        filename = Path(url).name
        dest = Path(raw_dir) / "worldpop" / filename
        return [_download_file(url, dest)]

    def to_zarr(
        self,
        path: Path,
        zarr_root: Path,
        bbox: tuple[float, float, float, float],
        *,
        iso3: str,
        year: int,
        **_: Any,
    ) -> Any:  # xr.Dataset
        """Clip *path* to *bbox* and append to the country's Zarr group."""
        time_coord = np.datetime64(f"{year}-01-01", "ns")
        return geotiff_to_zarr(
            path,
            zarr_root,
            self.zarr_group(iso3=iso3),
            bbox=bbox,
            time_coord=time_coord,
            variable_name=self.VARIABLE,
        )

    def zarr_group(self, *, iso3: str, **_: Any) -> str:
        """One Zarr group per country — years stored as time steps."""
        return f"worldpop/{iso3.lower()}"

    def stac_item_id(self, *, iso3: str, **_: Any) -> str:
        """One STAC item per country — datetime interval updated on each download."""
        return f"worldpop_{iso3.lower()}"

    def stac_properties(self, *, iso3: str, year: int, **_: Any) -> dict:
        return {
            "eostrata:iso3": iso3.upper(),
            "eostrata:variable": self.VARIABLE,
            "eostrata:resolution": "1km",
            "eostrata:release": _RELEASE,
        }

    def latest_available(self) -> datetime:
        """WorldPop R2025A covers 2015-2030 — default to previous year."""
        year = datetime.now(tz=UTC).year - 1
        return datetime(year, 1, 1, tzinfo=UTC)
