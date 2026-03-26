"""Temporal aggregation reader — subclass of titiler.xarray.io.Reader.

Intercepts get_variable() to apply a time slice and optional reduction
before TiTiler renders the tile. No pre-computed intermediates.
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

    TiTiler's Reader sets ``self.input`` via the module-level ``get_variable``
    inside ``__attrs_post_init__``, so overriding ``get_variable`` as an instance
    method has no effect.  We therefore override ``__attrs_post_init__`` and
    reduce the time dimension there, after the parent has finished initialising.

    Time selection via ``sel`` (e.g. ``sel=["time=2020-01-01"]``) is handled
    by TiTiler before we get here and already produces a 2D array, so we only
    need to act when the array is still 3D after the parent init.
    """

    def __attrs_post_init__(self) -> None:
        """Initialise parent, then collapse any remaining time dimension."""
        super().__attrs_post_init__()
        if "time" in self.input.dims:
            self.input = apply_temporal_aggregation(
                self.input,
                datetime_str=None,  # sel already handled datetime selection
                agg=None,  # default: last timestep
            )
            # Re-sync _dims now that time is gone
            self._dims = [
                d
                for d in self.input.dims
                if d
                not in (
                    self.input.rio.x_dim,
                    self.input.rio.y_dim,
                )
            ]
