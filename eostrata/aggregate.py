"""Temporal aggregation reader — subclass of titiler.xarray.io.Reader.

Intercepts get_variable() to apply a time slice and optional reduction
before TiTiler renders the tile. No pre-computed intermediates.
"""
from __future__ import annotations

import logging
from typing import List, Literal, Optional

import numpy as np
import xarray as xr
from titiler.xarray.io import Reader

logger = logging.getLogger(__name__)

AggMethod = Literal["mean", "sum", "min", "max", "anomaly"]


def _parse_datetime_interval(
    datetime_str: str | None,
) -> tuple[str | None, str | None]:
    """
    Parse an ISO 8601 datetime or interval string into (start, end).

    Examples
    --------
    ``"2021-01-01"``              → ``("2021-01-01", "2021-01-01")``
    ``"2021-01-01/2022-12-31"``   → ``("2021-01-01", "2022-12-31")``
    ``None``                       → ``(None, None)``
    """
    if not datetime_str:
        return None, None
    if "/" in datetime_str:
        parts = datetime_str.split("/", 1)
        return parts[0] or None, parts[1] or None
    return datetime_str, datetime_str


def apply_temporal_aggregation(
    da: xr.DataArray,
    *,
    datetime_str: str | None = None,
    agg: AggMethod | None = None,
    baseline: str | None = None,
) -> xr.DataArray:
    """
    Slice and optionally reduce the time dimension of *da*.

    Parameters
    ----------
    da:
        Input DataArray — must have a ``time`` dimension.
    datetime_str:
        ISO 8601 datetime or interval, e.g. ``"2021-01-01/2022-12-31"``.
        If None, no time slicing is applied.
    agg:
        Aggregation method. If None, the last timestep is selected.
        ``"anomaly"`` requires *baseline*.
    baseline:
        ISO 8601 interval defining the reference period for anomaly
        calculation, e.g. ``"2015-01-01/2020-12-31"``.

    Returns
    -------
    xr.DataArray
        A 2D (y, x) DataArray.
    """
    if "time" not in da.dims:
        return da

    t0, t1 = _parse_datetime_interval(datetime_str)

    # Time slice
    if t0 and t1:
        da = da.sel(time=slice(t0, t1))
    elif t0:
        da = da.sel(time=t0, method="nearest")

    if da.sizes.get("time", 1) == 0:
        raise ValueError(f"No data found for datetime='{datetime_str}'.")

    # Reduction
    if agg is None:
        # No aggregation — use last available timestep
        return da.isel(time=-1) if "time" in da.dims else da

    if agg == "mean":
        return da.mean("time")
    elif agg == "sum":
        return da.sum("time")
    elif agg == "min":
        return da.min("time")
    elif agg == "max":
        return da.max("time")
    elif agg == "anomaly":
        if not baseline:
            raise ValueError("'anomaly' aggregation requires a 'baseline' interval.")
        b0, b1 = _parse_datetime_interval(baseline)
        baseline_da = da.sel(time=slice(b0, b1))
        if baseline_da.sizes.get("time", 0) == 0:
            raise ValueError(f"No data found for baseline='{baseline}'.")
        return da.mean("time") - baseline_da.mean("time")
    else:
        raise ValueError(f"Unknown agg method '{agg}'. Use: mean, sum, min, max, anomaly.")


class AggregatingReader(Reader):
    """
    titiler.xarray Reader subclass that applies temporal aggregation
    before handing the DataArray to TiTiler's tile renderer.

    Extra kwargs accepted (passed via titiler's reader_options / dependency):
        datetime   - ISO 8601 datetime or interval string
        agg        - aggregation method (mean|sum|min|max|anomaly)
        baseline   - ISO 8601 interval for anomaly baseline
    """

    def get_variable(
        self,
        ds: xr.Dataset,
        variable: str,
        sel: Optional[List[str]] = None,
        method: Optional[str] = None,
    ) -> xr.DataArray:
        """Override to apply temporal aggregation after variable selection."""
        da = super().get_variable(ds, variable, sel=sel, method=method)

        # Pull aggregation params injected via reader_options
        datetime_str: str | None = getattr(self, "_agg_datetime", None)
        agg: str | None = getattr(self, "_agg_method", None)
        baseline: str | None = getattr(self, "_agg_baseline", None)

        if "time" in da.dims:
            da = apply_temporal_aggregation(
                da,
                datetime_str=datetime_str,
                agg=agg,
                baseline=baseline,
            )

        return da
