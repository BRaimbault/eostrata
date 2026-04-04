"""OGC API - Processes: zonalstats process."""

from __future__ import annotations

import logging

import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor on xarray
import rioxarray.exceptions as rio_exc
import xarray as xr
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from pathlib import Path

from eostrata.aggregate import apply_temporal_aggregation, resolve_accessed_times
from eostrata.cache import record_access
from eostrata.config import settings
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
) -> xr.DataArray:
    """Open the Zarr group and return the requested variable as a loaded 2D DataArray.

    Applies temporal aggregation when the array has a ``time`` dimension.
    The DataArray is fully materialised into memory before returning so that
    callers (e.g. the zonal-stats feature loop) can clip it N times without
    triggering N separate zarr reads.  The underlying dataset is closed before
    this function returns to avoid open file-handle accumulation under load.
    """
    store_path = url or str(settings.zarr_root)
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
    finally:
        ds.close()
    return da


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
    inp = body.inputs
    logger.info(
        "API POST /processes/zonalstats/execution group=%s variable=%s datetime=%s agg=%s features=%d",
        inp.group,
        inp.variable,
        inp.datetime,
        inp.agg,
        len(
            (inp.features.get("features") or [inp.features])
            if isinstance(inp.features, dict)
            else []
        ),
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
            raise HTTPException(
                status_code=422,
                detail="'features' must be a GeoJSON FeatureCollection, Feature or Polygon.",
            )

    features = fc.get("features", [])
    if not features:
        raise HTTPException(status_code=422, detail="FeatureCollection has no features.")

    # Load array once, clip to total bbox for efficiency
    da = _load_array(
        inp.url,
        inp.group,
        inp.variable,
        datetime=inp.datetime,
        agg=inp.agg,
        baseline=inp.baseline,
    )

    result_features = []
    for feat in features:
        geom = feat.get("geometry")
        if geom is None:
            result_features.append({**feat, "statistics": {"error": "no geometry"}})
            continue
        stats = _feature_stats(da, geom)
        result_features.append({**feat, "statistics": stats})

    return {
        "type": "FeatureCollection",
        "features": result_features,
    }
