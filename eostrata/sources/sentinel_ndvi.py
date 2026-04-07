"""Sentinel-3 NDVI source via Sentinel Hub BYOC / Copernicus Data Space.

Product:    CGLS NDVI 300m v3 — Sentinel-3 OLCI 10-day composites
Resolution: ~300 m spatial, 10-day (dekadal) temporal
Coverage:   Global, 2014-present
Provider:   JRC / EEA / Copernicus Global Land Service

Access:
    Sentinel Hub BYOC Process API on the Copernicus Data Space Ecosystem (CDSE).
    BYOC collection ID: 6303088f-3c19-4967-9038-119267c6d090

    Requires CDSE credentials (same account used for TROPOMI):
      EOSTRATA_CDSE_USER and EOSTRATA_CDSE_PASSWORD

Band encoding (UINT8):
    NDVI = raw_DN / 250.0 - 0.08    (range: -0.08 to 0.92)
    DN 255 is the fill/nodata value.

Dekad conventions:
    dekad=1  → days  1-10  of the month (timestep: YYYY-MM-01)
    dekad=2  → days 11-20  of the month (timestep: YYYY-MM-11)
    dekad=3  → days 21-end of the month (timestep: YYYY-MM-21)
"""

from __future__ import annotations

import calendar
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from eostrata.constants import PROP_RESOLUTION, PROP_SOURCE, PROP_VARIABLE
from eostrata.sources.base import BaseSource, register_source
from eostrata.store import geotiff_to_zarr

logger = logging.getLogger(__name__)

# ── Sentinel Hub BYOC constants ────────────────────────────────────────────────

_SH_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
_BYOC_COLLECTION_ID = "6303088f-3c19-4967-9038-119267c6d090"

# CGLS NDVI 300m grid spacing (1/336 °, ≈ 300 m at the equator)
_NDVI_RESOLUTION_DEG = 1.0 / 336.0

# DN 255 → nodata; remaining: physical_value = DN / 250.0 - 0.08
_EVALSCRIPT = """\
//VERSION=3
function setup() {
  return {
    input: [{bands: ["NDVI"], units: "DN"}],
    output: {bands: 1, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(sample) {
  if (sample.NDVI === 255) return [NaN];
  return [sample.NDVI / 250.0 - 0.08];
}
"""

_DEKAD_START_DAYS = {1: 1, 2: 11, 3: 21}

# Sentinel Hub Process API hard limit (pixels per dimension)
_SH_MAX_PIXELS = 2500


def _clamp_resolution(
    bbox: tuple[float, float, float, float],
    resolution_deg: float,
) -> float:
    """Return the coarsest of *resolution_deg* and the minimum needed to fit within 2500 px.

    Logs a warning when the resolution has to be reduced.
    """
    west, south, east, north = bbox
    min_resx = (east - west) / _SH_MAX_PIXELS
    min_resy = (north - south) / _SH_MAX_PIXELS
    clamped = max(resolution_deg, min_resx, min_resy)
    if clamped > resolution_deg:
        logger.warning(
            "Requested resolution %.6f° would exceed Sentinel Hub's %d-pixel limit "
            "for this bbox — automatically coarsened to %.6f° (≈ %.0f m at equator)",
            resolution_deg,
            _SH_MAX_PIXELS,
            clamped,
            clamped * 111_320,
        )
    return clamped


# ── CDSE authentication ────────────────────────────────────────────────────────

_CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)


def _get_cdse_token(user: str, password: str) -> str:
    """Obtain an OAuth2 Bearer token from the CDSE identity service."""
    resp = httpx.post(
        _CDSE_TOKEN_URL,
        data={
            "grant_type": "password",
            "username": user,
            "password": password,
            "client_id": "cdse-public",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _end_day_of_dekad(year: int, month: int, dekad: int) -> int:
    """Return the last calendar day of *dekad* in *year*/*month*."""
    if dekad == 1:
        return 10
    if dekad == 2:
        return 20
    return calendar.monthrange(year, month)[1]


def _fetch_ndvi_geotiff(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    dest: Path,
    token: str,
) -> Path:
    """Request a NDVI GeoTIFF from the Sentinel Hub Process API and save to *dest*.

    Parameters
    ----------
    bbox:
        (west, south, east, north) in EPSG:4326.
    start_date / end_date:
        ISO date strings ``YYYY-MM-DD`` for the dekad window.
    dest:
        Destination path for the downloaded GeoTIFF.
    token:
        CDSE OAuth2 Bearer token.
    """
    west, south, east, north = bbox
    resolution = _clamp_resolution(bbox, _NDVI_RESOLUTION_DEG)
    payload = {
        "input": {
            "bounds": {
                "bbox": [west, south, east, north],
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [
                {
                    "type": f"byoc-{_BYOC_COLLECTION_ID}",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{start_date}T00:00:00Z",
                            "to": f"{end_date}T23:59:59Z",
                        },
                        "mosaickingOrder": "mostRecent",
                    },
                }
            ],
        },
        "output": {
            "resx": resolution,
            "resy": resolution,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": _EVALSCRIPT,
    }

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Requesting NDVI GeoTIFF from Sentinel Hub for %s – %s", start_date, end_date)

    with httpx.stream(
        "POST",
        _SH_PROCESS_URL,
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
        follow_redirects=True,
        timeout=300,
    ) as resp:
        if not resp.is_success:
            resp.read()  # populate body while still inside the streaming context
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type or "text/html" in content_type:
            body = resp.read()
            raise RuntimeError(
                f"Sentinel Hub Process API returned unexpected content-type {content_type!r}. "
                f"Response: {body[:500]!r}"
            )
        with open(dest, "wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=1 << 20):
                fh.write(chunk)

    logger.info("Saved NDVI GeoTIFF to %s", dest)
    return dest


# ── Source class ───────────────────────────────────────────────────────────────


@register_source
class SentinelNDVISource(BaseSource):
    """Sentinel-3 NDVI 300m dekadal composites — CGLS v3 via Sentinel Hub BYOC."""

    id = "cgls"
    collection_id = "cgls"
    collection_title = "Sentinel-3 NDVI — CGLS v3 composites (300 m, dekadal)"
    collection_description = (
        "Sentinel-3 NDVI 300m dekadal composites from the Copernicus Global Land Service "
        "(v3, 2014-present), accessed via the Sentinel Hub BYOC API on CDSE."
    )
    zarr_prefix = "cgls"
    temporal_resolution = "dekadal"
    default_lag_days = 5  # ~5 days after dekad end before publication
    VARIABLE = "ndvi"
    ui_fields = ["years", "months", "dekads"]

    @classmethod
    def is_configured(cls) -> tuple[bool, str]:
        from eostrata.config import settings

        if settings.cdse_user and settings.cdse_password:
            return True, ""
        return (
            False,
            "CDSE credentials missing — set EOSTRATA_CDSE_USER + EOSTRATA_CDSE_PASSWORD "
            "(register for free at https://dataspace.copernicus.eu)",
        )

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
        """Download a single CGLS NDVI 300m dekadal GeoTIFF via Sentinel Hub.

        Parameters
        ----------
        year:   Reference year.
        month:  Month (1-12).
        dekad:  Dekad number: 1 (days 1-10), 2 (days 11-20), 3 (days 21-end).
        """
        from eostrata.config import settings

        user = settings.cdse_user
        password = settings.cdse_password
        if not user or not password:
            raise RuntimeError(
                "CDSE credentials are required for the CGLS NDVI source.\n"
                "Set EOSTRATA_CDSE_USER and EOSTRATA_CDSE_PASSWORD in your .env file.\n"
                "Register for free at https://dataspace.copernicus.eu"
            )

        start_day = _DEKAD_START_DAYS[dekad]
        end_day = _end_day_of_dekad(year, month, dekad)
        start_date = f"{year:04d}-{month:02d}-{start_day:02d}"
        end_date = f"{year:04d}-{month:02d}-{end_day:02d}"

        filename = f"ndvi_300m_v3_{year:04d}{month:02d}{start_day:02d}.tif"
        dest = Path(raw_dir) / "cgls" / filename

        if dest.exists():
            logger.info("Already available: %s", dest.name)
            return [dest]

        token = _get_cdse_token(user, password)
        _fetch_ndvi_geotiff(bbox, start_date, end_date, dest, token)
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
        """Clip *path* to *bbox* and append to the global ndvi Zarr group."""
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
        return "cgls/global"

    def stac_item_id(self, **_: Any) -> str:
        """Single STAC item for the global Sentinel NDVI collection."""
        return "global"

    def stac_properties(self, *, year: int, month: int, dekad: int = 1, **_: Any) -> dict:
        start_day = _DEKAD_START_DAYS[dekad]
        end_day = _end_day_of_dekad(year, month, dekad)
        return {
            PROP_VARIABLE: self.VARIABLE,
            PROP_RESOLUTION: "300m",
            "eostrata:release": "v3",
            "eostrata:byoc_collection": _BYOC_COLLECTION_ID,
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
