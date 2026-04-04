"""CDS / ERA5 monthly reanalysis source.

Fetches ERA5 monthly-averaged single-level data from the Copernicus
Climate Data Store via the ``cdsapi`` library.

Requires:
  - CDS account and API key at https://cds.climate.copernicus.eu
  - ``cdsapi`` package installed  (pip install cdsapi  OR  uv add cdsapi)
  - ~/.cdsapirc credentials file:
      url: https://cds.climate.copernicus.eu/api
      key: <your-uid>:<your-api-key>

Supported ERA5 variables (``--variable`` option):
  2m_temperature          → t2m        (K, monthly mean)
  total_precipitation     → tp         (m, monthly total)
  10m_u_component_of_wind → u10        (m/s, monthly mean)
  10m_v_component_of_wind → v10        (m/s, monthly mean)
  surface_pressure        → sp         (Pa, monthly mean)

The default variable is ``2m_temperature``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from eostrata.constants import PROP_RESOLUTION, PROP_VARIABLE
from eostrata.sources.base import BaseSource, register_source
from eostrata.store import _group_lock

logger = logging.getLogger(__name__)

# Maps short variable names to the ERA5 CDS variable long name
_VARIABLE_MAP: dict[str, str] = {
    "t2m": "2m_temperature",
    "tp": "total_precipitation",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "sp": "surface_pressure",
}

_CDS_DATASET = "reanalysis-era5-single-levels-monthly-means"
_PRODUCT_TYPE = "monthly_averaged_reanalysis"


def _get_cdsapi():
    """Import cdsapi lazily — only fails if the user tries to use this source."""
    try:
        import cdsapi  # type: ignore[import-untyped]

        return cdsapi
    except ImportError as exc:
        raise ImportError(
            "The 'cdsapi' package is required for the CDS/ERA5 source.\n"
            "Install it with:  uv add cdsapi  or  pip install cdsapi\n"
            "Then configure ~/.cdsapirc with your CDS credentials:\n"
            "  https://cds.climate.copernicus.eu/how-to-api"
        ) from exc


def _download_era5(
    dest: Path,
    *,
    variable: str,
    year: int,
    months: list[int],
    bbox: tuple[float, float, float, float],
) -> Path:
    """
    Download ERA5 monthly data for *variable* / *year* / *months* to *dest*.

    The ERA5 API returns a NetCDF file.  If *dest* already exists it is reused.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        logger.info("Already downloaded: %s", dest.name)
        return dest

    cdsapi = _get_cdsapi()
    west, south, east, north = bbox

    logger.info(
        "Requesting ERA5 %s year=%d months=%s bbox=%s",
        variable,
        year,
        months,
        bbox,
    )

    c = cdsapi.Client(quiet=True)
    c.retrieve(
        _CDS_DATASET,
        {
            "product_type": _PRODUCT_TYPE,
            "variable": variable,
            "year": str(year),
            "month": [f"{m:02d}" for m in months],
            "time": "00:00",
            "format": "netcdf",
            "area": [north, west, south, east],  # ERA5 uses N/W/S/E
        },
        str(dest),
    )

    logger.info("ERA5 file saved: %s", dest)
    return dest


def _netcdf_to_zarr(
    nc_path: Path,
    zarr_root: Path,
    zarr_group: str,
    *,
    variable: str,
    bbox: tuple[float, float, float, float],
) -> xr.Dataset:
    """
    Read an ERA5 NetCDF, rename coords to (x, y, time), clip to bbox,
    and write to the Zarr store.
    """
    ds = xr.open_dataset(nc_path)

    # ERA5 NetCDF may use 'longitude'/'latitude' or 'lon'/'lat'
    rename = {}
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

    # Clip to bbox
    west, south, east, north = bbox
    ds = ds.sel(x=slice(west, east), y=slice(north, south))

    # Drop ERA5 metadata variables that break Zarr append
    for var in ["expver", "time_bnds"]:
        if var in ds:
            ds = ds.drop_vars(var)

    # Ensure variable name matches the short name stored in zarr_group
    short = zarr_group.split("/")[-1]  # e.g. "t2m" from "era5/t2m"
    if short not in ds and variable in ds:
        ds = ds.rename({variable: short})

    zarr_root_path = Path(zarr_root)
    zarr_root_path.mkdir(parents=True, exist_ok=True)
    store_path = str(zarr_root_path)
    group_exists = (zarr_root_path / zarr_group).exists()

    cy, cx = 512, 512
    encoding: dict = {
        short: {
            "chunks": (1, cy, cx),
            "_FillValue": float("nan"),
        },
    }

    try:
        with _group_lock(zarr_root_path, zarr_group):
            if group_exists:
                # Filter out timestamps already present to avoid duplicate time values
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
                            "Skipping %d already-present timestamp(s) for '%s'",
                            skipped,
                            zarr_group,
                        )
                        ds = ds.isel(time=np.where(new_mask)[0])
                except (OSError, KeyError, ValueError) as exc:
                    logger.debug(
                        "Could not check existing timestamps in '%s', appending anyway: %s",
                        zarr_group,
                        exc,
                    )
                logger.info("Appending ERA5 '%s' to existing Zarr group", zarr_group)
                ds.to_zarr(
                    store_path,
                    group=zarr_group,
                    mode="a",
                    append_dim="time",
                    consolidated=True,
                )
            else:
                logger.info("Writing new ERA5 Zarr group '%s'", zarr_group)
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
class CDSSource(BaseSource):
    """ERA5 monthly reanalysis from the Copernicus Climate Data Store."""

    id = "cds"
    collection_id = "era5"
    collection_title = "ERA5 — Climate reanalysis (0.25°, monthly)"
    collection_description = (
        "Monthly climate reanalysis from the Copernicus Climate Data Store (ERA5)"
    )
    zarr_prefix = "era5"
    temporal_resolution = "monthly"
    default_lag_days = 90  # ERA5 has ~3-month production lag
    VARIABLE = "t2m"  # default variable short name
    VARIABLES = ["t2m", "tp", "u10", "v10", "sp"]
    VARIABLE_DESCRIPTIONS = {
        "t2m": "2m temperature (K)",
        "tp": "total precipitation (m)",
        "u10": "10m U-component of wind (m/s)",
        "v10": "10m V-component of wind (m/s)",
        "sp": "surface pressure (Pa)",
    }
    ui_fields = ["variable", "years", "months"]

    @classmethod
    def catalog_meta(cls, dataset_name: str) -> dict:
        # dataset_name IS the variable (e.g. "t2m" from "era5/t2m")
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
        variable: str = "t2m",
        year: int,
        months: list[int] | None = None,
        **_: Any,
    ) -> list[Path]:
        """
        Download ERA5 monthly data for *variable* / *year*.

        Parameters
        ----------
        variable:
            Short variable name — one of: t2m, tp, u10, v10, sp.
            Defaults to ``t2m`` (2m temperature).
        year:
            Reference year.
        months:
            List of months to fetch (1-12).  Defaults to all 12.
        """
        _months = months or list(range(1, 13))
        cds_variable = _VARIABLE_MAP.get(variable, variable)
        month_tag = "-".join(f"{m:02d}" for m in _months)
        filename = f"era5_{variable}_{year}_{month_tag}.nc"
        dest = Path(raw_dir) / "cds" / filename

        return [
            _download_era5(
                dest,
                variable=cds_variable,
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
        variable: str = "t2m",
        year: int,
        **_: Any,
    ) -> Any:  # xr.Dataset
        """Clip *path* to *bbox* and append to the variable's Zarr group."""
        cds_variable = _VARIABLE_MAP.get(variable, variable)
        return _netcdf_to_zarr(
            path,
            zarr_root,
            self.zarr_group(variable=variable),
            variable=cds_variable,
            bbox=bbox,
        )

    def zarr_group(self, *, variable: str = "t2m", **_: Any) -> str:
        """One Zarr group per ERA5 variable — all months/years as timesteps."""
        return f"era5/{variable}"

    def stac_item_id(self, *, variable: str = "t2m", **_: Any) -> str:
        """One STAC item per ERA5 variable."""
        return variable

    def stac_properties(self, *, variable: str = "t2m", year: int, **_: Any) -> dict:
        cds_name = _VARIABLE_MAP.get(variable, variable)
        return {
            PROP_VARIABLE: variable,
            "eostrata:cds_variable": cds_name,
            PROP_RESOLUTION: "0.25deg",
            "eostrata:dataset": _CDS_DATASET,
            "eostrata:product_type": _PRODUCT_TYPE,
        }

    def latest_available(self) -> datetime:
        """ERA5 production lag is ~3 months; return 3 months prior to today."""
        now = datetime.now(tz=UTC)
        month = now.month - 3
        year = now.year
        if month <= 0:
            month += 12
            year -= 1
        return datetime(year, month, 1, tzinfo=UTC)

    @classmethod
    def iter_periods(
        cls, *, variable: str = "t2m", years: list[int], months: list[int], **_
    ) -> Iterator[tuple[str, dict]]:
        for year in years:
            yield (f"{variable}/{year}", {"variable": variable, "year": year, "months": months})

    def stac_registrations(self, ds, period_kwargs: dict) -> list[dict]:
        from datetime import UTC, datetime

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
