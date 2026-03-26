"""Tests for OGC API - Processes (zonalstats)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient


def _write_zarr(zarr_root: Path, group: str = "worldpop/nga") -> None:
    """Write a tiny Zarr dataset for zonal stats testing."""
    import rioxarray  # noqa: F401

    y = np.linspace(14.0, 4.0, 20)
    x = np.linspace(2.0, 15.0, 20)
    data = np.ones((20, 20), dtype="float32") * 100.0
    da = xr.DataArray(data, dims=("y", "x"), coords={"y": y, "x": x}, name="population")
    da = da.rio.write_crs("EPSG:4326")
    ds = da.to_dataset()
    ds.to_zarr(str(zarr_root), group=group, mode="w", consolidated=True)


@pytest.fixture()
def zarr_root(tmp_path):
    root = tmp_path / "zarr"
    _write_zarr(root)
    return root


@pytest.fixture()
def app_client(zarr_root, monkeypatch):
    monkeypatch.setenv("EOSTRATA_ZARR_ROOT", str(zarr_root))
    monkeypatch.setenv("EOSTRATA_CATALOG_PATH", str(zarr_root.parent / "catalog.json"))

    import importlib

    import eostrata.config as cfg_mod

    importlib.reload(cfg_mod)

    from eostrata.server import app

    return TestClient(app)


# ── Unit tests for helpers ─────────────────────────────────────────────────────


class TestFeatureStats:
    def _make_da(self) -> xr.DataArray:
        import rioxarray  # noqa: F401

        y = np.linspace(10.0, 0.0, 20)
        x = np.linspace(0.0, 10.0, 20)
        data = np.full((20, 20), 50.0, dtype="float32")
        da = xr.DataArray(data, dims=("y", "x"), coords={"y": y, "x": x})
        return da.rio.write_crs("EPSG:4326")

    def test_returns_expected_keys(self):
        from eostrata.ogc.processes import _feature_stats

        da = self._make_da()
        geom = {
            "type": "Polygon",
            "coordinates": [[[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0], [1.0, 1.0]]],
        }
        stats = _feature_stats(da, geom)
        assert "count" in stats
        assert "mean" in stats
        assert "percentiles" in stats

    def test_mean_value(self):
        from eostrata.ogc.processes import _feature_stats

        da = self._make_da()
        geom = {
            "type": "Polygon",
            "coordinates": [[[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0], [1.0, 1.0]]],
        }
        stats = _feature_stats(da, geom)
        assert stats["mean"] == pytest.approx(50.0, abs=1.0)

    def test_all_nodata(self):
        import rioxarray  # noqa: F401

        from eostrata.ogc.processes import _feature_stats

        y = np.linspace(10.0, 0.0, 10)
        x = np.linspace(0.0, 10.0, 10)
        data = np.full((10, 10), np.nan, dtype="float64")
        da = xr.DataArray(data, dims=("y", "x"), coords={"y": y, "x": x})
        da = da.rio.write_crs("EPSG:4326")
        geom = {
            "type": "Polygon",
            "coordinates": [[[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0], [1.0, 1.0]]],
        }
        stats = _feature_stats(da, geom)
        assert stats["count"] == 0
        assert "nodata_count" in stats


# ── Route tests ────────────────────────────────────────────────────────────────


class TestListProcesses:
    def test_status_200(self, app_client):
        resp = app_client.get("/processes")
        assert resp.status_code == 200

    def test_contains_zonalstats(self, app_client):
        data = app_client.get("/processes").json()
        ids = [p["id"] for p in data["processes"]]
        assert "zonalstats" in ids


class TestDescribeProcess:
    def test_status_200(self, app_client):
        resp = app_client.get("/processes/zonalstats")
        assert resp.status_code == 200

    def test_has_inputs(self, app_client):
        data = app_client.get("/processes/zonalstats").json()
        assert "inputs" in data


class TestExecuteZonalStats:
    def _payload(self, zarr_root: str) -> dict:
        return {
            "inputs": {
                "url": zarr_root,
                "group": "worldpop/nga",
                "variable": "population",
                "features": {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"name": "test"},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [
                                    [[3.0, 5.0], [8.0, 5.0], [8.0, 9.0], [3.0, 9.0], [3.0, 5.0]]
                                ],
                            },
                        }
                    ],
                },
            }
        }

    def test_returns_feature_collection(self, app_client, zarr_root):
        payload = self._payload(str(zarr_root))
        resp = app_client.post("/processes/zonalstats/execution", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "FeatureCollection"

    def test_statistics_present(self, app_client, zarr_root):
        payload = self._payload(str(zarr_root))
        resp = app_client.post("/processes/zonalstats/execution", json=payload)
        feat = resp.json()["features"][0]
        assert "statistics" in feat
        stats = feat["statistics"]
        assert "mean" in stats
        assert "count" in stats

    def test_accepts_bare_feature(self, app_client, zarr_root):
        payload = {
            "inputs": {
                "url": str(zarr_root),
                "group": "worldpop/nga",
                "variable": "population",
                "features": {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[3.0, 5.0], [8.0, 5.0], [8.0, 9.0], [3.0, 9.0], [3.0, 5.0]]
                        ],
                    },
                },
            }
        }
        resp = app_client.post("/processes/zonalstats/execution", json=payload)
        assert resp.status_code == 200

    def test_accepts_bare_polygon(self, app_client, zarr_root):
        payload = {
            "inputs": {
                "url": str(zarr_root),
                "group": "worldpop/nga",
                "variable": "population",
                "features": {
                    "type": "Polygon",
                    "coordinates": [[[3.0, 5.0], [8.0, 5.0], [8.0, 9.0], [3.0, 9.0], [3.0, 5.0]]],
                },
            }
        }
        resp = app_client.post("/processes/zonalstats/execution", json=payload)
        assert resp.status_code == 200

    def test_unknown_type_returns_422(self, app_client, zarr_root):
        payload = {
            "inputs": {
                "url": str(zarr_root),
                "group": "worldpop/nga",
                "variable": "population",
                "features": {"type": "LineString", "coordinates": []},
            }
        }
        resp = app_client.post("/processes/zonalstats/execution", json=payload)
        assert resp.status_code == 422

    def test_empty_feature_collection_returns_422(self, app_client, zarr_root):
        payload = {
            "inputs": {
                "url": str(zarr_root),
                "group": "worldpop/nga",
                "variable": "population",
                "features": {"type": "FeatureCollection", "features": []},
            }
        }
        resp = app_client.post("/processes/zonalstats/execution", json=payload)
        assert resp.status_code == 422

    def test_missing_variable_returns_422(self, app_client, zarr_root):
        payload = {
            "inputs": {
                "url": str(zarr_root),
                "group": "worldpop/nga",
                "variable": "nonexistent_var",
                "features": {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [
                                    [[3.0, 5.0], [8.0, 5.0], [8.0, 9.0], [3.0, 9.0], [3.0, 5.0]]
                                ],
                            },
                        }
                    ],
                },
            }
        }
        resp = app_client.post("/processes/zonalstats/execution", json=payload)
        assert resp.status_code == 422

    def test_feature_without_geometry(self, app_client, zarr_root):
        """Features without geometry should return an error in statistics."""
        payload = {
            "inputs": {
                "url": str(zarr_root),
                "group": "worldpop/nga",
                "variable": "population",
                "features": {
                    "type": "FeatureCollection",
                    "features": [
                        {"type": "Feature", "properties": {"name": "no-geom"}, "geometry": None}
                    ],
                },
            }
        }
        resp = app_client.post("/processes/zonalstats/execution", json=payload)
        assert resp.status_code == 200
        feat = resp.json()["features"][0]
        assert feat["statistics"] == {"error": "no geometry"}
