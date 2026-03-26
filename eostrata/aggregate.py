"""Temporal aggregation reader — subclass of titiler.xarray.io.Reader.

Intercepts tile rendering to apply a time slice and optional reduction
before TiTiler encodes the tile.  No pre-computed intermediates.
"""

from __future__ import annotations

import logging
from typing import Literal

import xarray as xr
from titiler.xarray.io import Reader
from titiler.xarray.io import get_variable as _base_get_variable

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

    TiTiler's Reader.__attrs_post_init__ calls the module-level get_variable()
    directly (not self.get_variable), so we must override __attrs_post_init__
    to intercept before that call.  We replicate Reader's init logic so we can:

      1. Rename ERA5's ``valid_time`` coordinate to ``time`` before get_variable
         tries to process a ``sel=time=...`` parameter.
      2. Collapse any remaining time dimension after get_variable returns, so
         TiTiler always receives a 2D (y, x) array for tile encoding.

    The get_variable() instance method is retained for direct callers (tests,
    zonal-stats) and respects ``_agg_datetime``, ``_agg_method``, ``_agg_baseline``
    instance attributes.
    """

    def __attrs_post_init__(self) -> None:
        """Open dataset, normalise coords, run get_variable, then collapse time."""
        from rio_tiler.io.xarray import XarrayReader as _XarrayReader

        # Open the dataset (mirrors Reader.__attrs_post_init__)
        self.ds = self.opener(
            self.src_path,
            group=self.group,
            decode_times=self.decode_times,
        )

        # Normalise ERA5 time coordinate: valid_time → time
        if "valid_time" in self.ds.coords and "time" not in self.ds.coords:
            self.ds = self.ds.rename({"valid_time": "time"})

        # Select variable — sel handles any explicit datetime parameter
        self.input = _base_get_variable(
            self.ds,
            self.variable,
            sel=self.sel,
            method=self.method,
        )

        # Let XarrayReader set bounds, CRS, _dims from self.input
        _XarrayReader.__attrs_post_init__(self)

        # Collapse any remaining time dimension (sel already handled specific datetimes)
        if "time" in self.input.dims:
            self.input = apply_temporal_aggregation(
                self.input,
                datetime_str=None,
                agg=None,  # default: last timestep
            )
            self._dims = [
                d for d in self.input.dims if d not in (self.input.rio.x_dim, self.input.rio.y_dim)
            ]

    def get_variable(
        self,
        ds: xr.Dataset,
        variable: str,
        sel: list[str] | None = None,
        method: str | None = None,
    ) -> xr.DataArray:
        """
        Select *variable* from *ds* and apply temporal aggregation.

        Respects ``_agg_datetime``, ``_agg_method``, and ``_agg_baseline``
        instance attributes set by callers (e.g. tests, zonal-stats endpoint).
        """
        da = _base_get_variable(ds, variable, sel=sel, method=method)

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
