"""Temporal aggregation reader — subclass of titiler.xarray.io.Reader.

Intercepts tile rendering to apply a time slice and optional reduction
before TiTiler encodes the tile.  No pre-computed intermediates.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Literal

import numpy as np
import xarray as xr
from titiler.xarray.io import Reader
from titiler.xarray.io import get_variable as _base_get_variable

logger = logging.getLogger(__name__)

AggMethod = Literal["mean", "sum", "min", "max", "anomaly"]

# ---------------------------------------------------------------------------
# Context variables — set by tiles.py before delegating to TiTiler so that
# AggregatingReader.__attrs_post_init__ can read them without relying on
# TiTiler's fixed parameter set.  Using ContextVar is safe for async code:
# each asyncio task inherits its own copy of the context, and ASGITransport
# runs the inner ASGI app in the same coroutine (no new Task), so the values
# are visible inside AggregatingReader.
# ---------------------------------------------------------------------------

_CTX_AGG_DATETIME: ContextVar[str | None] = ContextVar("agg_datetime", default=None)
_CTX_AGG_METHOD: ContextVar[str | None] = ContextVar("agg_method", default=None)
_CTX_AGG_BASELINE: ContextVar[str | None] = ContextVar("agg_baseline", default=None)


def _strip_tz(dt: str) -> str:
    """Remove timezone suffix from an ISO 8601 datetime string.

    xarray uses timezone-naive ``datetime64[ns]`` indices; passing a string
    that includes a UTC offset (e.g. ``+00:00``) or a ``Z`` suffix causes
    ``TypeError: The index must be timezone aware when indexing with a date
    string with a UTC offset``.  Since all eostrata data is stored in UTC,
    stripping the suffix is safe.
    """
    # Use C-level str.find() instead of a Python char-by-char loop.
    # Z is the most common suffix for UTC strings; check it first.
    if dt.endswith("Z"):
        return dt[:-1]
    # Search for + or - only after position 10 (skips the date dashes).
    for sep in ("+", "-"):
        idx = dt.find(sep, 11)
        if idx != -1:
            return dt[:idx]
    return dt


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

    # Sort the time axis so that slice operations work correctly.
    # ERA5 zarr data appended across multiple download runs can produce a
    # non-monotonic DatetimeIndex, causing pandas to reject label-based slices.
    # Cache the index in a local variable — da.indexes["time"] is looked up
    # multiple times below and each lookup involves a dict access + property call.
    time_idx = da.indexes["time"]
    if not time_idx.is_monotonic_increasing:
        da = da.sortby("time")
        time_idx = da.indexes["time"]
        if not time_idx.is_monotonic_increasing:
            raise ValueError(
                "Time axis is not monotonic increasing even after sort — "
                "the dataset may be corrupt."
            )

    # Deduplicate the time axis — re-ingesting the same year produces duplicate
    # timestamps that cause .sel(method="nearest") to raise InvalidIndexError.
    if not time_idx.is_unique:
        n_dups = len(time_idx) - time_idx.nunique()
        logger.warning(
            "Found %d duplicate timestamp(s) in time axis — keeping first occurrence. "
            "Re-ingest may have produced duplicates; consider rebuilding the catalogue.",
            n_dups,
        )
        _, first_occurrence = np.unique(time_idx, return_index=True)
        da = da.isel(time=first_occurrence)

    t0, t1 = _parse_datetime_interval(datetime_str)
    if t0:
        t0 = _strip_tz(t0)
    if t1:
        t1 = _strip_tz(t1)

    # Time slice / selection
    if t0 and t1:
        if t0 == t1:
            # Single point — use nearest-neighbour to avoid partial-slice errors
            # on non-monotonic or non-matching DatetimeIndexes.
            da = da.sel(time=t0, method="nearest")
        else:
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
        if b0:
            b0 = _strip_tz(b0)
        if b1:
            b1 = _strip_tz(b1)
        baseline_da = da.sel(time=slice(b0, b1))
        if baseline_da.sizes.get("time", 0) == 0:
            raise ValueError(f"No data found for baseline='{baseline}'.")
        return da.mean("time") - baseline_da.mean("time")
    else:
        raise ValueError(f"Unknown agg method '{agg}'. Use: mean, sum, min, max, anomaly.")


def resolve_accessed_times(
    ds,
    datetime_str: str | None,
    agg_method: str | None = None,
    baseline: str | None = None,
) -> list:
    """Return the numpy datetime64 values from *ds* that a request will access.

    For a single-point datetime, returns the nearest timestamp.
    For an interval (or temporal aggregation), returns all timestamps in range.
    For ``agg="anomaly"``, also includes all timestamps in the baseline range.
    For ``datetime_str=None``, returns the last timestamp (default behaviour).

    Used to record per-timestamp last-access sentinels before the time
    dimension is collapsed by aggregation.
    """
    # Support both Dataset and DataArray
    if "time" not in getattr(ds, "dims", {}) and "time" not in getattr(ds, "coords", {}):
        return []
    try:
        times = ds["time"].values
    except (KeyError, AttributeError):
        return []
    if len(times) == 0:
        return []

    # Pre-compute once — reused by both the main datetime and the baseline range
    # when agg="anomaly" calls _in_range() twice.
    times_s = times.astype("datetime64[s]")

    def _in_range(dt_str: str | None) -> list:
        if not dt_str:
            return [times[-1]]
        t0, t1 = _parse_datetime_interval(dt_str)
        if not t0:
            return [times[-1]]
        t0s = _strip_tz(t0)
        t1s = _strip_tz(t1) if t1 else t0s
        if t0s == t1s:
            target = np.datetime64(t0s, "s")
            idx = int(np.argmin(np.abs(times_s - target)))
            return [times[idx]]
        start = np.datetime64(t0s, "s")
        end = np.datetime64(t1s, "s")
        mask = (times_s >= start) & (times_s <= end)
        return list(times[mask])

    accessed = _in_range(datetime_str)
    if agg_method == "anomaly" and baseline:
        baseline_times = _in_range(baseline)
        seen = {t.tobytes() for t in accessed}
        for t in baseline_times:
            if t.tobytes() not in seen:
                accessed.append(t)
                seen.add(t.tobytes())
    return accessed


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
        from pathlib import Path

        from rio_tiler.io.xarray import XarrayReader as _XarrayReader

        from eostrata.cache import record_access

        # Open the dataset (mirrors Reader.__attrs_post_init__)
        self.ds = self.opener(
            self.src_path,
            group=self.group,
            decode_times=self.decode_times,
        )

        # Normalise ERA5 time coordinate: valid_time → time
        if "valid_time" in self.ds.coords and "time" not in self.ds.coords:
            self.ds = self.ds.rename({"valid_time": "time"})

        # Read aggregation parameters from context (set by collection_tile before
        # delegating to the internal TiTiler app via ASGITransport).
        agg_datetime = _CTX_AGG_DATETIME.get()
        agg_method = _CTX_AGG_METHOD.get()
        agg_baseline = _CTX_AGG_BASELINE.get()

        # Record per-timestamp access AFTER time coord is normalised and BEFORE
        # apply_temporal_aggregation collapses the time dimension.
        if self.group:
            accessed = resolve_accessed_times(self.ds, agg_datetime, agg_method, agg_baseline)
            if accessed:
                record_access(Path(self.src_path), self.group, accessed)

        # Strip any time-related sel entries — we handle time ourselves via
        # apply_temporal_aggregation so that range queries and agg methods work.
        # Non-time sel entries (other dimensions) are forwarded as-is.
        non_time_sel = [s for s in (self.sel or []) if not s.startswith("time=")]
        self.input = _base_get_variable(
            self.ds,
            self.variable,
            sel=non_time_sel or None,
        )

        # Let XarrayReader set bounds, CRS, _dims from self.input
        _XarrayReader.__attrs_post_init__(self)

        # Collapse any remaining time dimension using the aggregation parameters
        # from context.  Defaults (all None) yield the last available timestep.
        if "time" in self.input.dims:
            self.input = apply_temporal_aggregation(
                self.input,
                datetime_str=agg_datetime,
                agg=agg_method,
                baseline=agg_baseline,
            )
            self.input = self.input.rio.write_crs(self.crs)
            self._dims = [
                d for d in self.input.dims if d not in (self.input.rio.x_dim, self.input.rio.y_dim)
            ]

    def get_variable(
        self,
        ds: xr.Dataset,
        variable: str,
        sel: list[str] | None = None,
    ) -> xr.DataArray:
        """
        Select *variable* from *ds* and apply temporal aggregation.

        Respects ``_agg_datetime``, ``_agg_method``, and ``_agg_baseline``
        instance attributes set by callers (e.g. tests, zonal-stats endpoint).
        """
        da = _base_get_variable(ds, variable, sel=sel)

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
