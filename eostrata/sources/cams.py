"""CAMS Global Reanalysis EAC4 monthly air quality source.

Fetches CAMS EAC4 monthly-averaged surface-level data from the Copernicus
Atmosphere Data Store (ADS) via the ``cdsapi`` library.

Requires:
  - ADS account at https://ads.atmosphere.copernicus.eu
  - ``cdsapi`` package installed  (already a project dependency)
  - ~/.adsapirc credentials file:
      url: https://ads.atmosphere.copernicus.eu/api
      key: <your-uid>:<your-api-key>
    OR set EOSTRATA_ADS_URL and EOSTRATA_ADS_KEY environment variables.

Supported variables (``--variable`` / ``variable`` param):
  pm2p5   — particulate_matter_2.5um    (kg/m³, surface concentration)
  pm10    — particulate_matter_10um     (kg/m³, surface concentration)
  no2     — nitrogen_dioxide            (kg/m³, surface concentration)
  co      — carbon_monoxide             (kg/m³, surface concentration)
  o3      — ozone                       (kg/m³, surface concentration)
  so2     — sulphur_dioxide             (kg/m³, surface concentration)
  aod550  — total_aerosol_optical_depth_at_550nm  (dimensionless, single-level)

The default variable is ``pm2p5`` (PM2.5 surface concentration).

Dataset: cams-global-reanalysis-eac4-monthly
Resolution: 0.75° (~80 km)
Coverage: Global, 2003–present
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

import eostrata.config as _eostrata_config
from eostrata.constants import PROP_RESOLUTION, PROP_VARIABLE
from eostrata.sources.base import BaseSource, register_source

logger = logging.getLogger(__name__)

# Maps short variable names to CAMS EAC4 long names used in the ADS request
_VARIABLE_MAP: dict[str, str] = {
    "pm2p5": "particulate_matter_2.5um",
    "pm10": "particulate_matter_10um",
    "no2": "nitrogen_dioxide",
    "co": "carbon_monoxide",
    "o3": "ozone",
    "so2": "sulphur_dioxide",
    "aod550": "total_aerosol_optical_depth_at_550nm",
}

# Variables that require a pressure_level selection (surface = 1000 hPa).
# aod550 is a single-level (column-integrated) field.
_MULTI_LEVEL_VARS: frozenset[str] = frozenset({"pm2p5", "pm10", "no2", "co", "o3", "so2"})

_ADS_DATASET = "cams-global-reanalysis-eac4-monthly"
_DEFAULT_ADS_URL = "https://ads.atmosphere.copernicus.eu/api"


def _get_cdsapi():
    """Import cdsapi lazily — only fails if the user tries to use this source."""
    try:
        import cdsapi  # type: ignore[import-untyped]

        return cdsapi
    except ImportError as exc:
        raise ImportError(
            "The 'cdsapi' package is required for the CAMS source.\n"
            "It should already be installed as a project dependency.\n"
            "Configure ~/.adsapirc with your ADS credentials:\n"
            "  https://ads.atmosphere.copernicus.eu/how-to-api"
        ) from exc


def _download_cams(
    dest: Path,
    *,
    variable: str,
    year: int,
    months: list[int],
    bbox: tuple[float, float, float, float],
) -> Path:
    """Download CAMS EAC4 monthly data for *variable* / *year* / *months* to *dest*.

    Returns *dest* (reused if it already exists).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("Already downloaded: %s", dest.name)
        return dest

    from eostrata.config import settings

    cdsapi = _get_cdsapi()
    west, south, east, north = bbox
    cads_variable = _VARIABLE_MAP.get(variable, variable)
    needs_level = variable in _MULTI_LEVEL_VARS

    request: dict[str, Any] = {
        "variable": cads_variable,
        "year": str(year),
        "month": [f"{m:02d}" for m in months],
        "time": "00:00",
        "format": "netcdf",
        "area": [north, west, south, east],  # ADS uses N/W/S/E order
    }
    if needs_level:
        request["pressure_level"] = ["1000"]  # surface level (hPa)

    url = settings.ads_url or _DEFAULT_ADS_URL
    key = settings.ads_key

    logger.info(
        "Requesting CAMS EAC4 %s year=%d months=%s bbox=%s",
        variable,
        year,
        months,
        bbox,
    )

    if key:
        c = cdsapi.Client(url=url, key=key, quiet=True)
    else:
        # Falls back to ~/.adsapirc
        c = cdsapi.Client(url=url, quiet=True)

    c.retrieve(_ADS_DATASET, request, str(dest))
    logger.info("CAMS file saved: %s", dest)
    return dest


def _cams_netcdf_to_zarr(
    nc_path: Path,
    zarr_root: Path,
    zarr_group: str,
    *,
    variable: str,
    bbox: tuple[float, float, float, float],
) -> xr.Dataset:
    """Read a CAMS EAC4 NetCDF, normalise coordinates, clip to bbox, write Zarr."""
    ds = xr.open_dataset(nc_path)

    # Rename spatial and time coordinates to (x, y, time)
    rename: dict[str, str] = {}
    if "longitude" in ds.coords:
        rename["longitude"] = "x"
    elif "lon" in ds.coords:
        rename["lon"] = "x"
    if "latitude" in ds.coords:
        rename["latitude"] = "y"
    elif "lat" in ds.coords:
        rename["lat"] = "y"
    if "valid_time" in ds.coords and "time" not in ds.coords:
        rename["valid_time"] = "time"
    if rename:
        ds = ds.rename(rename)

    # For multi-level variables, select the surface pressure level and drop it
    for level_dim in ("pressure_level", "level"):
        if level_dim in ds.dims:
            ds = ds.sel({level_dim: 1000}, method="nearest").drop_vars(level_dim, errors="ignore")
            break

    # Clip to bbox
    west, south, east, north = bbox
    ds = ds.sel(x=slice(west, east), y=slice(north, south))

    # Ensure the data variable is named by its short name
    short = zarr_group.split("/")[-1]  # e.g. "pm2p5" from "cams/pm2p5"
    cads_variable = _VARIABLE_MAP.get(variable, variable)
    for candidate in (short, cads_variable, variable):
        if candidate in ds and candidate != short:
            ds = ds.rename({candidate: short})
            break

    # Drop auxiliary metadata variables that would break Zarr append
    for var in ["expver", "time_bnds"]:
        if var in ds:
            ds = ds.drop_vars(var)

    zarr_root_path = Path(zarr_root)
    zarr_root_path.mkdir(parents=True, exist_ok=True)
    store_path = str(zarr_root_path)
    group_exists = (zarr_root_path / zarr_group).exists()

    cy = cx = _eostrata_config.settings.zarr_chunk_size
    encoding: dict = {
        short: {
            "chunks": (1, cy, cx),
            "_FillValue": float("nan"),
        },
    }

    try:
        if group_exists:
            # Skip timestamps already in the store to avoid duplicates
            try:
                existing = xr.open_zarr(store_path, group=zarr_group, consolidated=False)
                existing_times_s = existing["time"].values.astype("datetime64[s]")
                existing.close()
                new_times_s = ds["time"].values.astype("datetime64[s]")
                new_mask = ~np.isin(new_times_s, existing_times_s)
                if not new_mask.any():
                    logger.info(
                        "All timestamps already present in '%s' — skipping append", zarr_group
                    )
                    return ds
                if not new_mask.all():
                    skipped = (~new_mask).sum()
                    logger.info(
                        "Skipping %d already-present timestamp(s) for '%s'", skipped, zarr_group
                    )
                    ds = ds.isel(time=np.where(new_mask)[0])
            except Exception as exc:
                logger.debug(
                    "Could not check existing timestamps in '%s', appending anyway: %s",
                    zarr_group,
                    exc,
                )
            logger.info("Appending CAMS '%s' to existing Zarr group", zarr_group)
            ds.to_zarr(
                store_path,
                group=zarr_group,
                mode="a",
                append_dim="time",
                consolidated=True,
            )
        else:
            logger.info("Writing new CAMS Zarr group '%s'", zarr_group)
            ds.to_zarr(
                store_path,
                group=zarr_group,
                mode="w",
                encoding=encoding,
                consolidated=True,
            )
    finally:
        ds.close()

    return ds


@register_source
class CAMSSource(BaseSource):
    """CAMS Global Reanalysis EAC4 monthly air quality from the Atmosphere Data Store."""

    id = "cams"
    collection_id = "cams"
    collection_title = "CAMS EAC4 — Air quality reanalysis (0.75°, monthly)"
    collection_description = (
        "CAMS Global Reanalysis EAC4 monthly surface air quality "
        "from the Copernicus Atmosphere Data Store"
    )
    zarr_prefix = "cams"
    temporal_resolution = "monthly"
    default_lag_days = 120  # EAC4 has ~4-month production lag
    VARIABLE = "pm2p5"  # default variable short name
    VARIABLES = ["pm2p5", "pm10", "no2", "co", "o3", "so2", "aod550"]
    VARIABLE_DESCRIPTIONS = {
        "pm2p5": "PM2.5 surface concentration (kg/m³)",
        "pm10": "PM10 surface concentration (kg/m³)",
        "no2": "nitrogen dioxide surface concentration (kg/m³)",
        "co": "carbon monoxide surface concentration (kg/m³)",
        "o3": "ozone surface concentration (kg/m³)",
        "so2": "sulphur dioxide surface concentration (kg/m³)",
        "aod550": "total aerosol optical depth at 550 nm",
    }
    ui_fields = ["variable", "years", "months"]

    @classmethod
    def is_configured(cls) -> tuple[bool, str]:
        from pathlib import Path

        from eostrata.config import settings

        if settings.ads_key:
            return True, ""
        if Path.home().joinpath(".adsapirc").exists():
            return True, ""
        return False, "ADS credentials missing — set EOSTRATA_ADS_KEY or add ~/.adsapirc"

    @classmethod
    def catalog_meta(cls, dataset_name: str) -> dict:
        # dataset_name IS the variable (e.g. "pm2p5" from "cams/pm2p5")
        return {
            "item_id": dataset_name,
            "variable": dataset_name,
            "extra": {PROP_VARIABLE: dataset_name},
        }

    def download(
        self,
        raw_dir: Path,
        bbox: tuple[float, float, float, float],
        *,
        variable: str = "pm2p5",
        year: int,
        months: list[int] | None = None,
        **_: Any,
    ) -> list[Path]:
        """Download CAMS EAC4 monthly data for *variable* / *year*.

        Parameters
        ----------
        variable:
            Short variable name — one of: pm2p5, pm10, no2, co, o3, so2, aod550.
            Defaults to ``pm2p5`` (PM2.5 surface concentration).
        year:
            Reference year.
        months:
            List of months to fetch (1-12).  Defaults to all 12.
        """
        if variable not in _VARIABLE_MAP:
            raise ValueError(
                f"Unknown CAMS variable '{variable}'. Available: {list(_VARIABLE_MAP)}"
            )
        _months = months or list(range(1, 13))
        month_tag = "-".join(f"{m:02d}" for m in _months)
        filename = f"cams_{variable}_{year}_{month_tag}.nc"
        dest = Path(raw_dir) / "cams" / filename

        return [
            _download_cams(
                dest,
                variable=variable,
                year=year,
                months=_months,
                bbox=bbox,
            )
        ]

    def to_zarr(
        self,
        path: Path,
        zarr_root: Path,
        bbox: tuple[float, float, float, float],
        *,
        variable: str = "pm2p5",
        year: int,
        **_: Any,
    ) -> Any:  # xr.Dataset
        """Clip *path* to *bbox* and append to the variable's CAMS Zarr group."""
        return _cams_netcdf_to_zarr(
            path,
            zarr_root,
            self.zarr_group(variable=variable),
            variable=variable,
            bbox=bbox,
        )

    def zarr_group(self, *, variable: str = "pm2p5", **_: Any) -> str:
        """One Zarr group per CAMS variable — all months/years as timesteps."""
        return f"cams/{variable}"

    def stac_item_id(self, *, variable: str = "pm2p5", **_: Any) -> str:
        """One STAC item per CAMS variable."""
        return variable

    def stac_properties(self, *, variable: str = "pm2p5", year: int, **_: Any) -> dict:
        cads_name = _VARIABLE_MAP.get(variable, variable)
        return {
            PROP_VARIABLE: variable,
            "eostrata:cams_variable": cads_name,
            PROP_RESOLUTION: "0.75deg",
            "eostrata:dataset": _ADS_DATASET,
            "eostrata:pressure_level": "1000hPa" if variable in _MULTI_LEVEL_VARS else "single",
        }

    def latest_available(self) -> datetime:
        """CAMS EAC4 production lag is ~4 months; return 4 months prior to today."""
        now = datetime.now(tz=UTC)
        month = now.month - 4
        year = now.year
        while month <= 0:
            month += 12
            year -= 1
        return datetime(year, month, 1, tzinfo=UTC)

    @classmethod
    def iter_periods(
        cls, *, variable: str = "pm2p5", years: list[int], months: list[int], **_
    ) -> Iterator[tuple[str, dict]]:
        for year in years:
            yield (f"{variable}/{year}", {"variable": variable, "year": year, "months": months})

    def stac_registrations(self, ds, period_kwargs: dict) -> list[dict]:
        variable = period_kwargs["variable"]
        year = period_kwargs["year"]
        months = period_kwargs["months"]
        return [
            {
                "item_id": self.stac_item_id(variable=variable),
                "datetime_": datetime(year, month, 1, tzinfo=UTC),
                "variable": variable,
                "extra_properties": self.stac_properties(variable=variable, year=year),
            }
            for month in months
        ]

    def extract_item_bbox(self, ds) -> tuple[float, float, float, float]:
        x_dim = "x" if "x" in ds.coords else "longitude"
        y_dim = "y" if "y" in ds.coords else "latitude"
        return (
            float(ds[x_dim].min()),
            float(ds[y_dim].min()),
            float(ds[x_dim].max()),
            float(ds[y_dim].max()),
        )
