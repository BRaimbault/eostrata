"""TODO: Replace this module docstring with a description of your data source.

Example:
    MODIS LST — Terra/Aqua land surface temperature monthly composites.
    Resolution: 1 km, Coverage: global, 2000-present.
    Provider: NASA EOSDIS / LP DAAC.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from eostrata.sources.base import BaseSource, _stream_download, register_source
from eostrata.store import geotiff_to_zarr

logger = logging.getLogger(__name__)

# TODO: Define your data URL pattern here
_BASE_URL = "https://example.org/data"


# TODO: Add any helper functions (URL builders, decompressors, etc.) here


@register_source
class TemplateSource(BaseSource):
    """TODO: One-line description of the source."""

    # ── Identity ───────────────────────────────────────────────────────────────

    #: Unique snake_case source identifier used in the CLI, API, and UI.
    id = "template"  # TODO: change to your source id, e.g. "modis_lst"

    #: STAC collection id (usually the same as `id`).
    collection_id = "template"  # TODO

    #: Human-readable title shown in the STAC catalogue and UI.
    collection_title = "Template source"  # TODO: e.g. "MODIS Land Surface Temperature"

    #: One-sentence description for the STAC collection.
    collection_description = "TODO: describe your dataset"

    # ── Zarr / variable ────────────────────────────────────────────────────────

    #: First path component in the Zarr store.  Usually the same as `id` but
    #: can differ (e.g. CDSSource uses "era5" not "cds").
    zarr_prefix = "template"  # TODO

    #: Short variable name stored in Zarr (e.g. "lst", "ndvi", "population").
    VARIABLE = "myvar"  # TODO

    # ── Temporal ───────────────────────────────────────────────────────────────

    #: "monthly", "annual", "dekadal", "daily", etc.
    temporal_resolution = "monthly"  # TODO

    #: Typical days between period end and data availability.
    default_lag_days = 30  # TODO

    # ── Ingest behaviour ───────────────────────────────────────────────────────

    #: Set True if HTTP 404 means "data not yet published" (skip silently).
    #: Set False if 404 is a real error that should be reported.
    skip_404 = False  # TODO

    #: Form fields shown in the map UI ingest tab.
    #: Recognised values: "iso3", "variable", "years", "months", "dekads"
    ui_fields = ["years", "months"]  # TODO: adjust for your source

    # ── Catalog metadata ───────────────────────────────────────────────────────

    @classmethod
    def catalog_meta(cls, dataset_name: str) -> dict:
        """Return STAC metadata inferred from a Zarr group path component.

        Called by ``rebuild-catalog``.  ``dataset_name`` is the second part of
        the group path (e.g. ``"nga"`` from ``worldpop/nga``).

        Override only if the default ``f"{zarr_prefix}_{dataset_name}"`` item id
        does not suit your source.  Delete this method to use the default.
        """
        # TODO: customise or delete this method
        return {
            "item_id": f"{cls.zarr_prefix}_{dataset_name}",
            "variable": cls.VARIABLE,
            "extra": {"eostrata:variable": cls.VARIABLE},
        }

    # ── Download ───────────────────────────────────────────────────────────────

    def download(
        self,
        raw_dir: Path,
        bbox: tuple[float, float, float, float],
        *,
        year: int,
        month: int,
        **_: Any,
    ) -> list[Path]:
        """Download raw data for one period to *raw_dir*.

        TODO: adapt the signature to match the keys yielded by iter_periods.
        Returns a list of downloaded file paths; the first element is passed to to_zarr.
        """
        # TODO: build the URL for this period
        url = f"{_BASE_URL}/{year}/{month:02d}.tif"
        filename = f"template_{year}_{month:02d}.tif"
        dest = Path(raw_dir) / "template" / filename  # TODO: use your source id
        return [_stream_download(url, dest)]

    # ── Zarr conversion ────────────────────────────────────────────────────────

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
        """Convert downloaded file to a Zarr group clipped to *bbox*.

        TODO: adapt to your file format (GeoTIFF, NetCDF, etc.).
        """
        # TODO: set the correct time coordinate for this period
        time_coord = np.datetime64(f"{year}-{month:02d}-01", "ns")
        return geotiff_to_zarr(
            path,
            zarr_root,
            self.zarr_group(),
            bbox=bbox,
            time_coord=time_coord,
            variable_name=self.VARIABLE,
        )

    # ── Zarr group ─────────────────────────────────────────────────────────────

    def zarr_group(self, **_: Any) -> str:
        """Return the Zarr group path for this source.

        Examples:
          "worldpop/nga"   — one group per country
          "chirps/global"  — single global group
          "era5/t2m"       — one group per variable
        """
        # TODO: return the appropriate group path
        return "template/global"

    # ── STAC ───────────────────────────────────────────────────────────────────

    def stac_item_id(self, **_: Any) -> str:
        """Return the STAC item id for the given params."""
        # TODO: return a stable, unique item id
        return "template_global"

    def stac_properties(self, *, year: int, month: int, **_: Any) -> dict:
        """Return extra STAC item properties for the given params."""
        # TODO: add source-specific metadata fields
        return {
            "eostrata:variable": self.VARIABLE,
            # "eostrata:resolution": "...",
            # "eostrata:release":    "...",
        }

    def latest_available(self) -> datetime:
        """Return the most recent datetime for which data is reliably available.

        Used as the default when the user does not specify a year/month.
        """
        # TODO: implement the actual lag calculation for your source
        now = datetime.now(tz=UTC)
        month = now.month - 1 or 12
        year = now.year if now.month > 1 else now.year - 1
        return datetime(year, month, 1, tzinfo=UTC)

    # ── Generic ingest hooks ────────────────────────────────────────────────────

    @classmethod
    def iter_periods(
        cls,
        *,
        years: list[int],
        months: list[int],
        **_,
    ) -> Iterator[tuple[str, dict]]:
        """Yield (label, period_kwargs) for every period to download.

        ``label`` appears in log messages and error output.
        ``period_kwargs`` is passed to download(), to_zarr(), stac_item_id(),
        stac_properties(), and stac_registrations().

        TODO: add / remove loop variables to match your source's period structure.
        Examples:
          WorldPop  — loops over years only (annual data)
          CHIRPS    — loops over years × months
          Sentinel  — loops over years × months × dekads
        """
        for year in years:
            for month in months:
                yield (f"{year}-{month:02d}", {"year": year, "month": month})

    def stac_registrations(self, ds, period_kwargs: dict) -> list[dict]:
        """Return STAC items to register after one period is written to Zarr.

        ``ds`` is the xr.Dataset returned by to_zarr().
        ``period_kwargs`` is the second element of the tuple yielded by iter_periods.

        Each dict must have: item_id, datetime_, variable, extra_properties.

        Most sources return a list with a single entry.  CDS/ERA5 is an exception:
        one download covers a full year but produces one STAC item per month.

        TODO: set the correct datetime_ for your period.
        """
        year = period_kwargs["year"]
        month = period_kwargs["month"]
        return [{
            "item_id": self.stac_item_id(**period_kwargs),
            "datetime_": datetime(year, month, 1, tzinfo=UTC),
            "variable": self.VARIABLE,
            "extra_properties": self.stac_properties(**period_kwargs),
        }]
