"""OGC API - Processes: zonalstats process."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor on xarray
import rioxarray.exceptions as rio_exc
import xarray as xr
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from eostrata.aggregate import apply_temporal_aggregation, resolve_accessed_times
from eostrata.cache import record_access
from eostrata.config import settings
from eostrata.log import current_job_id
from eostrata.ogc.ingest import INGEST_PROCESS_IDS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/processes", tags=["Zonal Statistics"])

# ── OGC Process description ───────────────────────────────────────────────────

_PROCESS_DESCRIPTION = {
    "id": "zonalstats",
    "title": "Zonal statistics",
    "description": (
        "Summarises raster values from a Zarr collection within polygon zones. "
        "Returns per-feature statistics over a single timestep or a temporally "
        "aggregated period."
    ),
    "version": "0.1.0",
    "jobControlOptions": ["sync-execute"],
    "inputs": {
        "url": {
            "title": "Zarr store URL",
            "description": "Path to the Zarr store root.",
            "schema": {"type": "string"},
        },
        "variable": {
            "title": "Variable name",
            "description": "Data variable to extract from the Zarr group.",
            "schema": {"type": "string"},
        },
        "group": {
            "title": "Zarr group",
            "description": "Group path inside the store, e.g. worldpop/nga_2021_1km.",
            "schema": {"type": "string"},
        },
        "features": {
            "title": "GeoJSON FeatureCollection",
            "description": "Zones to compute statistics for.",
            "schema": {"type": "object"},
        },
        "datetime": {
            "title": "Datetime",
            "description": "ISO 8601 datetime or interval for time selection (optional).",
            "schema": {"type": "string"},
        },
        "agg": {
            "title": "Aggregation method",
            "description": (
                "Temporal aggregation method applied before zonal extraction. "
                "One of: mean, sum, min, max, anomaly. "
                "If omitted, the last available timestep is used."
            ),
            "schema": {"type": "string", "enum": ["mean", "sum", "min", "max", "anomaly"]},
        },
        "baseline": {
            "title": "Baseline interval",
            "description": (
                "ISO 8601 interval defining the reference period for anomaly aggregation, "
                "e.g. 2015-01-01/2020-12-31. Required when agg=anomaly."
            ),
            "schema": {"type": "string"},
        },
    },
    "outputs": {
        "results": {
            "title": "Zonal statistics FeatureCollection",
            "schema": {"type": "object"},
        }
    },
}


# ── Request / response models ─────────────────────────────────────────────────


class ZonalStatsInputs(BaseModel):
    url: str | None = Field(
        default=None, description="Zarr store root path (leave blank to use the server default)"
    )
    variable: str = Field(
        ...,
        description="Variable name in the Zarr group (see /examples for valid values)",
        json_schema_extra={"example": "population"},
    )
    group: str = Field(
        ...,
        description="Zarr group path inside the store (see /examples for valid values)",
        json_schema_extra={"example": "worldpop/nga"},
    )
    features: dict = Field(
        ..., description="GeoJSON FeatureCollection, Feature, or Polygon geometry"
    )
    datetime: str | None = Field(
        None,
        description="ISO 8601 datetime or interval for time selection",
        json_schema_extra={"example": "2020-01-01T00:00:00+00:00"},
    )
    agg: str | None = Field(
        None,
        description="Temporal aggregation method: mean, sum, min, max, anomaly",
        json_schema_extra={"example": "mean"},
    )
    baseline: str | None = Field(
        None,
        description="ISO 8601 interval for anomaly baseline, e.g. 2015-01-01/2020-12-31",
        json_schema_extra={"example": "2015-01-01/2020-12-31"},
    )


_EXECUTION_EXAMPLE = {
    "inputs": {
        "group": "worldpop/nga",
        "variable": "population",
        "datetime": "2020-01-01T00:00:00+00:00",
        "features": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [2.0, 4.0],
                                [15.0, 4.0],
                                [15.0, 14.0],
                                [2.0, 14.0],
                                [2.0, 4.0],
                            ]
                        ],
                    },
                    "properties": {"name": "example zone"},
                }
            ],
        },
    }
}


class ExecutionRequest(BaseModel):
    model_config = {"json_schema_extra": {"examples": [_EXECUTION_EXAMPLE]}}

    inputs: ZonalStatsInputs


# ── Computation helpers ───────────────────────────────────────────────────────


def _load_array(
    url: str,
    group: str,
    variable: str,
    *,
    datetime: str | None = None,
    agg: str | None = None,
    baseline: str | None = None,
    clip_bbox: tuple[float, float, float, float] | None = None,
) -> xr.DataArray:
    """Open the Zarr group and return the requested variable as a loaded 2D DataArray.

    Applies temporal aggregation when the array has a ``time`` dimension.
    When ``agg_cache_maxsize > 0`` the aggregated result is cached in memory
    and reused on subsequent calls with the same parameters (including from
    tile requests), making repeated queries free.  When caching is disabled
    (``agg_cache_maxsize=0``) the *clip_bbox* spatial pre-clip is applied to
    bound memory usage to the features' bounding box.
    The DataArray is fully materialised into memory before returning so that
    callers (e.g. the zonal-stats feature loop) can clip it N times without
    triggering N separate zarr reads.  The underlying dataset is closed before
    this function returns to avoid open file-handle accumulation under load.
    """
    # Deferred import to avoid circular dependency (aggregate → cache → aggregate).
    from eostrata.aggregate import _agg_cache_key, _get_agg_cache, _put_agg_cache

    store_path = url or str(settings.zarr_root)

    # ── Cache check ────────────────────────────────────────────────────────────
    if settings.agg_cache_maxsize > 0:
        cache_key = _agg_cache_key(store_path, group, variable, datetime, agg, baseline)
        hit = _get_agg_cache(cache_key)
        if hit is not None:
            cached_da, accessed = hit
            if accessed:
                record_access(Path(store_path), group, accessed)
            return cached_da

    ds = xr.open_zarr(store_path, group=group, consolidated=True)
    try:
        if variable not in ds:
            available = [v for v in ds.data_vars if v != "crs"]
            raise HTTPException(
                status_code=422,
                detail=f"Variable '{variable}' not found. Available: {available}",
            )
        da = ds[variable]

        # Normalise ERA5 time coordinate: valid_time → time
        if "valid_time" in da.coords and "time" not in da.dims:
            da = da.assign_coords(time=da["valid_time"]).swap_dims({"valid_time": "time"})

        # Spatial pre-clip: only applied when caching is disabled.
        # When caching is on we compute the full-extent result (so it is
        # reusable by any tile/stats request) and the clip_bbox is irrelevant.
        if clip_bbox is not None and settings.agg_cache_maxsize == 0 and "x" in da.dims and "y" in da.dims:
            w, s, e, n = clip_bbox
            y_vals = da.y.values
            # y may be descending (north→south) — slice direction must match
            if len(y_vals) > 1 and y_vals[0] > y_vals[-1]:
                da = da.sel(x=slice(w, e), y=slice(n, s))
            else:
                da = da.sel(x=slice(w, e), y=slice(s, n))

        # Record per-timestamp access BEFORE apply_temporal_aggregation collapses the time dim.
        accessed = resolve_accessed_times(da, datetime, agg, baseline)
        if accessed:
            record_access(Path(store_path), group, accessed)

        if "time" in da.dims:
            try:
                da = apply_temporal_aggregation(
                    da,
                    datetime_str=datetime,
                    agg=agg,
                    baseline=baseline,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

        da = da.squeeze()
        # Write CRS from the dataset's crs variable if present
        if "crs" in ds and "crs_wkt" in ds["crs"].attrs:
            da = da.rio.write_crs(ds["crs"].attrs["crs_wkt"])
        else:
            da = da.rio.write_crs("EPSG:4326")

        # Materialise into memory so the zarr store can be released and callers
        # can clip the same in-memory array N times (once per input polygon).
        da = da.load()

        # Store in cache for subsequent tile and stats requests with same params.
        if settings.agg_cache_maxsize > 0:
            _put_agg_cache(cache_key, da, accessed)  # type: ignore[possibly-undefined]
    finally:
        ds.close()
    return da


def _features_bbox(
    features: list[dict],
) -> tuple[float, float, float, float] | None:
    """Return the combined (west, south, east, north) bbox of all feature geometries.

    Returns None when no valid coordinate pairs are found.
    """
    xs: list[float] = []
    ys: list[float] = []

    def _collect(coords) -> None:
        for item in coords:
            if isinstance(item[0], (int, float)):
                xs.append(float(item[0]))
                ys.append(float(item[1]))
            else:
                _collect(item)

    for feat in features:
        geom = feat.get("geometry") or {}
        raw = geom.get("coordinates")
        if raw:
            _collect(raw)

    if not xs:
        return None
    # Add a small buffer (0.5°) so edge pixels aren't clipped by floating-point rounding
    buf = 0.5
    return min(xs) - buf, min(ys) - buf, max(xs) + buf, max(ys) + buf


def _feature_stats(da: xr.DataArray, geometry: dict) -> dict:
    """Clip *da* to *geometry* and return summary statistics."""
    try:
        clipped = da.rio.clip([geometry], crs="EPSG:4326", drop=True, all_touched=False)
    except (ValueError, rio_exc.RioXarrayError) as exc:
        logger.warning("Clip failed: %s", exc)
        return {"error": str(exc)}

    values = clipped.values.astype("float64", copy=False)  # no-op if already float64
    valid = values[np.isfinite(values)]

    if valid.size == 0:
        return {"count": 0, "nodata_count": int(values.size)}

    p5, p25, p50, p75, p95 = np.percentile(valid, [5, 25, 50, 75, 95])
    return {
        "count": int(valid.size),
        "nodata_count": int(values.size - valid.size),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "sum": float(valid.sum()),
        "percentiles": {
            "p5": float(p5),
            "p25": float(p25),
            "p50": float(p50),
            "p75": float(p75),
            "p95": float(p95),
        },
    }


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get("", summary="List available processes")
def list_processes() -> dict:
    return {
        "processes": [{"id": "zonalstats", "version": "0.1.0"}, *INGEST_PROCESS_IDS],
        "links": [{"href": "/processes", "rel": "self", "type": "application/json"}],
    }


@router.get("/zonalstats", summary="Process description")
def describe_process() -> dict:
    return _PROCESS_DESCRIPTION


@router.post(
    "/zonalstats/execution",
    summary="Execute zonal statistics",
    response_class=JSONResponse,
)
def execute_zonalstats(body: ExecutionRequest) -> dict:
    """
    Compute per-feature zonal statistics over a Zarr dataset.

    The request body follows the OGC API - Processes execution schema:
    ```json
    {
      "inputs": {
        "group": "worldpop/nga_2021_1km",
        "variable": "nga_2021_1km",
        "features": { "type": "FeatureCollection", "features": [...] }
      }
    }
    ```
    """
    job_id = uuid.uuid4().hex[:8]
    _tok = current_job_id.set(job_id)
    try:
        return _execute_zonalstats(job_id, body)
    finally:
        current_job_id.reset(_tok)


def _execute_zonalstats(job_id: str, body: ExecutionRequest) -> dict:
    inp = body.inputs

    n_features = len(
        (inp.features.get("features") or [inp.features])
        if isinstance(inp.features, dict)
        else []
    )
    logger.info(
        "Job %s started: zonalstats group=%s variable=%s datetime=%s agg=%s features=%d",
        job_id,
        inp.group,
        inp.variable,
        inp.datetime,
        inp.agg,
        n_features,
    )
    fc = inp.features

    if fc.get("type") != "FeatureCollection":
        # Accept a bare Geometry or Feature too
        if fc.get("type") == "Feature":
            fc = {"type": "FeatureCollection", "features": [fc]}
        elif fc.get("type") in {"Polygon", "MultiPolygon"}:
            fc = {
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "geometry": fc, "properties": {}}],
            }
        else:
            logger.warning("Job %s failed — invalid features type", job_id)
            return JSONResponse(
                status_code=422,
                content={
                    "jobID": job_id,
                    "detail": "'features' must be a GeoJSON FeatureCollection, Feature or Polygon.",
                },
            )

    features = fc.get("features", [])
    if not features:
        logger.warning("Job %s failed — FeatureCollection has no features", job_id)
        return JSONResponse(
            status_code=422,
            content={"jobID": job_id, "detail": "FeatureCollection has no features."},
        )

    clip_bbox = _features_bbox(features)
    try:
        da = _load_array(
            inp.url,
            inp.group,
            inp.variable,
            datetime=inp.datetime,
            agg=inp.agg,
            baseline=inp.baseline,
            clip_bbox=clip_bbox,
        )
    except HTTPException as exc:
        logger.warning("Job %s failed — %s", job_id, exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={"jobID": job_id, "detail": exc.detail},
        )
    except Exception:
        logger.exception(
            "Job %s failed: loading group=%s variable=%s datetime=%s",
            job_id,
            inp.group,
            inp.variable,
            inp.datetime,
        )
        return JSONResponse(
            status_code=500,
            content={"jobID": job_id, "detail": "Failed to load dataset."},
        )

    try:
        result_features = []
        for feat in features:
            geom = feat.get("geometry")
            if geom is None:
                result_features.append({**feat, "statistics": {"error": "no geometry"}})
                continue
            stats = _feature_stats(da, geom)
            result_features.append({**feat, "statistics": stats})
    except Exception:
        logger.exception("Job %s failed: computing feature statistics", job_id)
        return JSONResponse(
            status_code=500,
            content={"jobID": job_id, "detail": "Failed to compute statistics."},
        )

    logger.info("Job %s succeeded", job_id)
    return {
        "jobID": job_id,
        "type": "FeatureCollection",
        "features": result_features,
    }
