"""Temporal aggregation reader — subclass of titiler.xarray.io.Reader.

Intercepts tile rendering to apply a time slice and optional reduction
before TiTiler encodes the tile.  No pre-computed intermediates.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr
from morecantile import Tile as _Tile
from rasterio.warp import transform_bounds as _transform_bounds
from rio_tiler.errors import TileOutsideBounds as _TileOutsideBounds
from rio_tiler.io.xarray import XarrayReader as _XarrayReader
from titiler.xarray.io import Reader
from titiler.xarray.io import get_variable as _base_get_variable

import eostrata.config as _eostrata_config
from eostrata.cache import record_access

# Semaphore that caps concurrent heavy aggregation operations.
# Initialised lazily from settings so tests can override max_concurrent_aggregations.
_agg_semaphore: threading.Semaphore | None = None
_agg_semaphore_lock = threading.Lock()


def _get_agg_semaphore() -> threading.Semaphore | None:
    """Return the global aggregation semaphore, creating it on first call."""
    global _agg_semaphore
    limit = _eostrata_config.settings.max_concurrent_aggregations
    if limit <= 0:
        return None
    if _agg_semaphore is None or _agg_semaphore._value != limit:  # type: ignore[attr-defined]
        with _agg_semaphore_lock:
            if _agg_semaphore is None or _agg_semaphore._value != limit:  # type: ignore[attr-defined]
                _agg_semaphore = threading.Semaphore(limit)
    return _agg_semaphore


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


class _nullctx:
    """No-op context manager used when max_concurrent_aggregations=0 (unlimited)."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


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


def _chunked_reduce(
    da: xr.DataArray,
    batch_op: Callable[[xr.DataArray], xr.DataArray],
    combine: Callable[[xr.DataArray, xr.DataArray], xr.DataArray],
    batch_size: int,
) -> xr.DataArray:
    """Reduce *da* along time in sequential batches to bound peak RAM.

    *batch_op* is applied to each batch (e.g. ``lambda b: b.sum("time")``).
    *combine* merges partial results (e.g. ``lambda a, b: a + b``).
    Each batch is materialised independently; at most one batch is in memory
    at a time.
    """
    result: xr.DataArray | None = None
    n = da.sizes["time"]
    for start in range(0, n, batch_size):
        part = batch_op(da.isel(time=slice(start, start + batch_size))).compute()
        result = part if result is None else combine(result, part)
    if result is None:
        raise ValueError("Cannot reduce a DataArray with no time steps")
    return result


def _chunked_mean(da: xr.DataArray, batch_size: int) -> xr.DataArray:
    """Compute da.mean("time") in batches, correctly weighted for unequal final batch."""
    n = da.sizes["time"]
    total = _chunked_reduce(da, lambda b: b.sum("time"), lambda a, b: a + b, batch_size)
    return total / n


def _chunked_aggregate(da: xr.DataArray, agg: AggMethod, batch_size: int) -> xr.DataArray:
    """Compute a reduction over time in batches of *batch_size* to bound peak RAM."""
    if agg == "mean":
        return _chunked_mean(da, batch_size)
    if agg == "sum":
        return _chunked_reduce(da, lambda b: b.sum("time"), lambda a, b: a + b, batch_size)
    if agg == "min":
        return _chunked_reduce(
            da, lambda b: b.min("time"), lambda a, b: a.where(a <= b, b), batch_size
        )
    if agg == "max":
        return _chunked_reduce(
            da, lambda b: b.max("time"), lambda a, b: a.where(a >= b, b), batch_size
        )
    raise ValueError(f"Unknown agg method '{agg}'. Use: mean, sum, min, max, anomaly.")


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

    n_ts = da.sizes.get("time", 1)
    max_ts = _eostrata_config.settings.max_aggregation_timesteps
    batch_size = _eostrata_config.settings.aggregation_batch_size
    use_batched = max_ts > 0 and "time" in da.dims and n_ts > max_ts
    if use_batched:
        logger.info(
            "Time range spans %d timesteps (limit %d) — using batched aggregation (batch_size=%d)",
            n_ts,
            max_ts,
            batch_size,
        )
    else:
        logger.info("Aggregating %d timestep(s) with agg=%s", n_ts, agg)

    sem = _get_agg_semaphore()
    with sem if sem is not None else _nullctx():
        if agg == "mean":
            return _chunked_mean(da, batch_size) if use_batched else da.mean("time").compute()
        elif agg == "sum":
            return (
                _chunked_aggregate(da, "sum", batch_size)
                if use_batched
                else da.sum("time").compute()
            )
        elif agg == "min":
            return (
                _chunked_aggregate(da, "min", batch_size)
                if use_batched
                else da.min("time").compute()
            )
        elif agg == "max":
            return (
                _chunked_aggregate(da, "max", batch_size)
                if use_batched
                else da.max("time").compute()
            )
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
            mean_fn: Callable[[xr.DataArray], xr.DataArray] = (
                (lambda d: _chunked_mean(d, batch_size))
                if use_batched
                else (lambda d: d.mean("time").compute())
            )
            return mean_fn(da) - mean_fn(baseline_da)
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
        seen = set(accessed)
        for t in baseline_times:
            if t not in seen:
                accessed.append(t)
                seen.add(t)
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

        # When the data has a time dimension, defer temporal aggregation to tile().
        # This allows tile() to spatially clip to the tile bbox *before* aggregating,
        # which avoids loading the full global raster for each tile (crucial on
        # memory-constrained instances).  We replace self.input with the last
        # timestep so that reader metadata (bounds, dtype, count) is always 2D.
        if "time" in self.input.dims:
            self._unagg_input: xr.DataArray | None = self.input  # lazy 3D zarr array
            self._tile_datetime: str | None = agg_datetime
            self._tile_method: str | None = agg_method
            self._tile_baseline: str | None = agg_baseline
            # Use last timestep as a lightweight 2D placeholder for reader metadata.
            self.input = self.input.isel(time=-1)
            self.input = self.input.rio.write_crs(self.crs)
            # Override _dims: the time dimension is now collapsed, so no extra dims.
            self._dims = []
        else:
            self._unagg_input = None
            self._tile_datetime = None
            self._tile_method = None
            self._tile_baseline = None

    def tile(self, tile_x: int, tile_y: int, tile_z: int, tilesize: int | None = None, **kwargs):  # type: ignore[override]
        """Render a map tile, clipping spatially to the tile bbox before aggregating.

        For datasets with a time dimension this clips the lazy 3D zarr array to
        the tile's geographic bounding box first, then aggregates only the
        ~256×256 pixel region needed — rather than aggregating the full global
        raster and discarding all but the tile pixels afterward.  This reduces
        peak RAM from O(global_extent × timesteps) to O(tile_pixels × timesteps).
        """
        if self._unagg_input is None:
            # No time dimension — use the standard reader implementation.
            return super().tile(tile_x, tile_y, tile_z, tilesize=tilesize, **kwargs)

        if not self.tile_exists(tile_x, tile_y, tile_z):
            raise _TileOutsideBounds(f"Tile(x={tile_x}, y={tile_y}, z={tile_z}) is outside bounds")

        # Get tile bounds in WGS84 (the coordinate system of eostrata zarr data).
        tile_bounds_tms = self.tms.xy_bounds(_Tile(x=tile_x, y=tile_y, z=tile_z))
        w, s, e, n = _transform_bounds(
            self.tms.rasterio_crs,
            "EPSG:4326",
            tile_bounds_tms.left,
            tile_bounds_tms.bottom,
            tile_bounds_tms.right,
            tile_bounds_tms.top,
        )

        # Clip the lazy 3D array to the tile bbox plus a 1-pixel buffer to
        # avoid gaps at tile edges from rounding during reprojection.
        da = self._unagg_input
        try:
            xbuf = abs(float(da.x[1] - da.x[0])) if da.x.size > 1 else 1.0
            ybuf = abs(float(da.y[1] - da.y[0])) if da.y.size > 1 else 1.0
        except Exception:
            xbuf = ybuf = 0.1

        y_vals = da.y.values
        if len(y_vals) > 1 and float(y_vals[0]) > float(y_vals[-1]):
            # Descending y (north→south storage, typical for raster grids)
            clipped = da.sel(x=slice(w - xbuf, e + xbuf), y=slice(n + ybuf, s - ybuf))
        else:
            clipped = da.sel(x=slice(w - xbuf, e + xbuf), y=slice(s - ybuf, n + ybuf))

        # Aggregate only the tile-sized spatial region.
        agg_2d = apply_temporal_aggregation(
            clipped,
            datetime_str=self._tile_datetime,
            agg=self._tile_method,
            baseline=self._tile_baseline,
        )
        agg_2d = agg_2d.rio.write_crs(self.crs)

        # Render via a temporary XarrayReader wrapping the small 2D result.
        matrix = self.tms.matrix(tile_z)
        tmp = _XarrayReader(agg_2d, tms=self.tms, options=self.options)
        return tmp.tile(
            tile_x,
            tile_y,
            tile_z,
            tilesize=tilesize or matrix.tileHeight,
            **kwargs,
        )

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
