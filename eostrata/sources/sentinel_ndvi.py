"""Sentinel-3 NDVI source via Copernicus Global Land Service (CGLS) WCS.

Product:    CGLS NDVI 300m v2 — Sentinel-3 OLCI 10-day composites
Resolution: 300 m spatial, 10-day (dekadal) temporal
Coverage:   Global, 2014-present
Provider:   VITO / Copernicus Global Land Service
WCS:        https://globalland.vito.be/geoserver/ows

The public WCS endpoint is accessible without credentials for standard use.
For higher-throughput or restricted datasets, set EOSTRATA_CGLS_API_KEY to
a valid Copernicus Land Service API token; it is sent as a Bearer token.

URL pattern (WCS 2.0.1):
    SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage
    &COVERAGEID=ndvi_300m_v2_10daily
    &FORMAT=image/tiff
    &SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326
    &SUBSET=Lat(south,north)
    &SUBSET=Long(west,east)
    &SUBSET=ansi("YYYY-MM-DDT00:00:00.000Z","YYYY-MM-DDT23:59:59.999Z")

Dekad conventions:
    dekad=1  → days  1-10  of the month (timestep: YYYY-MM-01)
    dekad=2  → days 11-20  of the month (timestep: YYYY-MM-11)
    dekad=3  → days 21-end of the month (timestep: YYYY-MM-21)
"""

from __future__ import annotations

import calendar
import logging
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from tqdm import tqdm

from eostrata.constants import PROP_RESOLUTION, PROP_SOURCE, PROP_VARIABLE
from eostrata.sources.base import BaseSource, register_source
from eostrata.store import geotiff_to_zarr

logger = logging.getLogger(__name__)

_WCS_BASE = "https://globalland.vito.be/geoserver/ows"
_COVERAGE_ID = "ndvi_300m_v2_10daily"
_DEKAD_START_DAYS = {1: 1, 2: 11, 3: 21}
_DOWNLOAD_RETRIES = 2
_RETRY_DELAYS = (5, 15)


def _end_day_of_dekad(year: int, month: int, dekad: int) -> int:
    """Return the last calendar day of *dekad* in *year*/*month*."""
    if dekad == 1:
        return 10
    if dekad == 2:
        return 20
    return calendar.monthrange(year, month)[1]


def _build_wcs_url(
    bbox: tuple[float, float, float, float],
    year: int,
    month: int,
    dekad: int,
) -> str:
    """Build a WCS 2.0.1 GetCoverage URL for the CGLS NDVI 300m product."""
    west, south, east, north = bbox
    start_day = _DEKAD_START_DAYS[dekad]
    end_day = _end_day_of_dekad(year, month, dekad)
    t0 = f"{year:04d}-{month:02d}-{start_day:02d}T00:00:00.000Z"
    t1 = f"{year:04d}-{month:02d}-{end_day:02d}T23:59:59.999Z"
    return (
        f"{_WCS_BASE}?SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
        f"&COVERAGEID={_COVERAGE_ID}"
        f"&FORMAT=image/tiff"
        f"&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
        f"&SUBSET=Lat({south},{north})"
        f"&SUBSET=Long({west},{east})"
        f'&SUBSET=ansi("{t0}","{t1}")'
    )


def _download(url: str, dest: Path, *, api_key: str = "") -> Path:
    """Stream *url* to *dest* with optional Bearer auth and retry logic."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("Already downloaded: %s", dest.name)
        return dest

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_exc: Exception | None = None
    for attempt in range(1, _DOWNLOAD_RETRIES + 2):
        try:
            logger.info("Downloading %s (attempt %d)", url, attempt)
            with httpx.stream(
                "GET", url, headers=headers, follow_redirects=True, timeout=None
            ) as resp:
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
        except httpx.TransportError as exc:
            last_exc = exc
            dest.unlink(missing_ok=True)
            if attempt <= _DOWNLOAD_RETRIES:
                delay = _RETRY_DELAYS[attempt - 1]
                logger.warning(
                    "Download failed (attempt %d/%d): %s — retrying in %ds",
                    attempt,
                    _DOWNLOAD_RETRIES + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error("Download failed after %d attempts: %s", _DOWNLOAD_RETRIES + 1, exc)

    raise last_exc  # type: ignore[misc]


@register_source
class SentinelNDVISource(BaseSource):
    """Sentinel-3 NDVI 300m dekadal composites from the Copernicus Global Land Service."""

    id = "sentinel_ndvi"
    collection_id = "sentinel_ndvi"
    collection_title = "Sentinel-3 NDVI — CGLS composites (300 m, dekadal)"
    collection_description = (
        "Sentinel-3 NDVI 300m dekadal composites from the Copernicus Global Land Service"
    )
    zarr_prefix = "sentinel_ndvi"
    temporal_resolution = "dekadal"
    default_lag_days = 5  # ~5 days after dekad end before publication
    VARIABLE = "ndvi"
    ui_fields = ["years", "months", "dekads"]

    @classmethod
    def catalog_meta(cls, dataset_name: str) -> dict:
        return {
            "item_id": "global",
            "variable": cls.VARIABLE,
            "extra": {PROP_VARIABLE: cls.VARIABLE, PROP_SOURCE: "Sentinel-3 OLCI"},
        }

    def download(
        self,
        raw_dir: Path,
        bbox: tuple[float, float, float, float],
        *,
        year: int,
        month: int,
        dekad: int = 1,
        **_: Any,
    ) -> list[Path]:
        """Download a single CGLS NDVI 300m dekadal GeoTIFF.

        Parameters
        ----------
        year:   Reference year.
        month:  Month (1-12).
        dekad:  Dekad number: 1 (days 1-10), 2 (days 11-20), 3 (days 21-end).
        """
        start_day = _DEKAD_START_DAYS[dekad]
        filename = f"ndvi_300m_v2_{year:04d}{month:02d}{start_day:02d}.tif"
        dest = Path(raw_dir) / "sentinel_ndvi" / filename

        if dest.exists():
            logger.info("Already available: %s", dest.name)
            return [dest]

        from eostrata.config import settings

        url = _build_wcs_url(bbox, year, month, dekad)
        _download(url, dest, api_key=settings.cgls_api_key)
        return [dest]

    def to_zarr(
        self,
        path: Path,
        zarr_root: Path,
        bbox: tuple[float, float, float, float],
        *,
        year: int,
        month: int,
        dekad: int = 1,
        **_: Any,
    ) -> Any:  # xr.Dataset
        """Clip *path* to *bbox* and append to the global sentinel_ndvi Zarr group."""
        start_day = _DEKAD_START_DAYS[dekad]
        time_coord = np.datetime64(f"{year:04d}-{month:02d}-{start_day:02d}", "ns")
        return geotiff_to_zarr(
            path,
            zarr_root,
            self.zarr_group(),
            bbox=bbox,
            time_coord=time_coord,
            variable_name=self.VARIABLE,
        )

    def zarr_group(self, **_: Any) -> str:
        """Single global Zarr group — all dekads as time-steps."""
        return "sentinel_ndvi/global"

    def stac_item_id(self, **_: Any) -> str:
        """Single STAC item for the global Sentinel NDVI collection."""
        return "global"

    def stac_properties(self, *, year: int, month: int, dekad: int = 1, **_: Any) -> dict:
        start_day = _DEKAD_START_DAYS[dekad]
        end_day = _end_day_of_dekad(year, month, dekad)
        return {
            PROP_VARIABLE: self.VARIABLE,
            PROP_RESOLUTION: "300m",
            "eostrata:release": "v2",
            "eostrata:product": _COVERAGE_ID,
            PROP_SOURCE: "Sentinel-3 OLCI",
            "eostrata:period": (
                f"{year:04d}-{month:02d}-{start_day:02d}/{year:04d}-{month:02d}-{end_day:02d}"
            ),
        }

    def latest_available(self) -> datetime:
        """Return the start date of the most recently published CGLS NDVI dekad.

        CGLS NDVI publishes each dekad approximately 5 days after the dekad ends:
          Dekad 1 (days  1-10) → available ~15th of the same month
          Dekad 2 (days 11-20) → available ~25th of the same month
          Dekad 3 (days 21-end)→ available ~5th of the following month
        """
        now = datetime.now(tz=UTC)
        year, month, day = now.year, now.month, now.day

        if day >= 15:
            # Dekad 1 of current month is published
            return datetime(year, month, 1, tzinfo=UTC)
        if day >= 5:
            # Dekad 3 of previous month is published
            prev_month = month - 1 or 12
            prev_year = year if month > 1 else year - 1
            return datetime(prev_year, prev_month, 21, tzinfo=UTC)
        # Very early in the month — fall back to dekad 2 of previous month
        prev_month = month - 1 or 12
        prev_year = year if month > 1 else year - 1
        return datetime(prev_year, prev_month, 11, tzinfo=UTC)

    @classmethod
    def iter_periods(
        cls, *, years: list[int], months: list[int], dekads: list[int], **_
    ) -> Iterator[tuple[str, dict]]:
        for year in years:
            for month in months:
                for dekad in dekads:
                    yield (
                        f"{year}-{month:02d}-d{dekad}",
                        {"year": year, "month": month, "dekad": dekad},
                    )

    def stac_registrations(self, ds, period_kwargs: dict) -> list[dict]:
        from datetime import UTC, datetime

        year = period_kwargs["year"]
        month = period_kwargs["month"]
        dekad = period_kwargs.get("dekad", 1)
        start_day = _DEKAD_START_DAYS[dekad]
        return [
            {
                "item_id": self.stac_item_id(),
                "datetime_": datetime(year, month, start_day, tzinfo=UTC),
                "variable": self.VARIABLE,
                "extra_properties": self.stac_properties(**period_kwargs),
            }
        ]
