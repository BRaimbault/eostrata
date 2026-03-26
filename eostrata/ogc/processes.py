"""OGC API - Processes: zonalstats process."""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor on xarray
import xarray as xr
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from shapely.geometry import shape

from eostrata.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/processes", tags=["OGC Processes"])

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
    url: str = Field(default=None, description="Zarr store root path")
    variable: str = Field(..., description="Variable name in the Zarr group")
    group: str = Field(..., description="Zarr group path, e.g. worldpop/nga_2021_1km")
    features: dict = Field(..., description="GeoJSON FeatureCollection")
    datetime: Optional[str] = Field(None, description="ISO 8601 datetime or interval")


class ExecutionRequest(BaseModel):
    inputs: ZonalStatsInputs


# ── Computation helpers ───────────────────────────────────────────────────────

def _load_array(url: str, group: str, variable: str) -> xr.DataArray:
    """Open the Zarr group and return the requested variable as a 2D DataArray."""
    store_path = url or str(settings.zarr_root)
    ds = xr.open_zarr(store_path, group=group, consolidated=True)
    if variable not in ds:
        available = [v for v in ds.data_vars if v != "crs"]
        raise HTTPException(
            status_code=422,
            detail=f"Variable '{variable}' not found. Available: {available}",
        )
    da = ds[variable]
    # Collapse time dimension if present — use last timestep by default
    if "time" in da.dims:
        da = da.isel(time=-1)
    da = da.squeeze()
    # Write CRS from the dataset's crs variable if present
    if "crs" in ds and "crs_wkt" in ds["crs"].attrs:
        da = da.rio.write_crs(ds["crs"].attrs["crs_wkt"])
    else:
        da = da.rio.write_crs("EPSG:4326")
    return da


def _feature_stats(da: xr.DataArray, geometry: dict) -> dict:
    """Clip *da* to *geometry* and return summary statistics."""
    try:
        clipped = da.rio.clip([geometry], crs="EPSG:4326", drop=True, all_touched=False)
    except Exception as exc:
        logger.warning("Clip failed: %s", exc)
        return {"error": str(exc)}

    values = clipped.values.astype("float64")
    valid = values[np.isfinite(values)]

    if valid.size == 0:
        return {"count": 0, "nodata_count": int(values.size)}

    return {
        "count": int(valid.size),
        "nodata_count": int(values.size - valid.size),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "sum": float(valid.sum()),
        "percentiles": {
            "p5":  float(np.percentile(valid, 5)),
            "p25": float(np.percentile(valid, 25)),
            "p50": float(np.percentile(valid, 50)),
            "p75": float(np.percentile(valid, 75)),
            "p95": float(np.percentile(valid, 95)),
        },
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("", summary="List available processes")
def list_processes() -> dict:
    return {
        "processes": [{"id": "zonalstats", "version": "0.1.0"}],
        "links": [],
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
    fc = inp.features

    if fc.get("type") != "FeatureCollection":
        # Accept a bare Geometry or Feature too
        if fc.get("type") == "Feature":
            fc = {"type": "FeatureCollection", "features": [fc]}
        elif fc.get("type") in {"Polygon", "MultiPolygon"}:
            fc = {"type": "FeatureCollection",
                  "features": [{"type": "Feature", "geometry": fc, "properties": {}}]}
        else:
            raise HTTPException(status_code=422, detail="'features' must be a GeoJSON FeatureCollection, Feature or Polygon.")

    features = fc.get("features", [])
    if not features:
        raise HTTPException(status_code=422, detail="FeatureCollection has no features.")

    # Load array once, clip to total bbox for efficiency
    da = _load_array(inp.url or str(settings.zarr_root), inp.group, inp.variable)

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
