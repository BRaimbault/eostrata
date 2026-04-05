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


def _write_zarr_with_time(zarr_root: Path, group: str = "chirps/global") -> None:
    """Write a Zarr dataset with a time dimension for aggregation testing."""
    import rioxarray  # noqa: F401

    y = np.linspace(14.0, 4.0, 20)
    x = np.linspace(2.0, 15.0, 20)
    times = np.array(
        [np.datetime64("2020-01-01"), np.datetime64("2021-01-01"), np.datetime64("2022-01-01")]
    )
    # Values equal to the year (2020, 2021, 2022) for easy assertion
    data = np.stack([np.full((20, 20), float(y), dtype="float32") for y in [2020, 2021, 2022]])
    ds = xr.Dataset(
        {"precipitation": (("time", "y", "x"), data)},
        coords={"time": times, "y": y, "x": x},
    )
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

    def test_has_self_link(self, app_client):
        data = app_client.get("/processes").json()
        assert any(link["rel"] == "self" for link in data["links"])


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


# ── Temporal aggregation tests ─────────────────────────────────────────────────


@pytest.fixture()
def zarr_root_time(tmp_path):
    root = tmp_path / "zarr"
    _write_zarr_with_time(root)
    return root


_GEOM = {
    "type": "Polygon",
    "coordinates": [[[3.0, 5.0], [8.0, 5.0], [8.0, 9.0], [3.0, 9.0], [3.0, 5.0]]],
}


class TestZonalStatsTemporalAggregation:
    def _payload(self, zarr_root: str, **extra) -> dict:
        return {
            "inputs": {
                "url": zarr_root,
                "group": "chirps/global",
                "variable": "precipitation",
                "features": {
                    "type": "FeatureCollection",
                    "features": [{"type": "Feature", "properties": {}, "geometry": _GEOM}],
                },
                **extra,
            }
        }

    def test_no_datetime_returns_last_timestep(self, app_client, zarr_root_time):
        resp = app_client.post(
            "/processes/zonalstats/execution", json=self._payload(str(zarr_root_time))
        )
        assert resp.status_code == 200
        stats = resp.json()["features"][0]["statistics"]
        # No agg → last timestep → 2022
        assert stats["mean"] == pytest.approx(2022.0, abs=1.0)

    def test_datetime_single_selects_timestep(self, app_client, zarr_root_time):
        resp = app_client.post(
            "/processes/zonalstats/execution",
            json=self._payload(str(zarr_root_time), datetime="2020-01-01"),
        )
        assert resp.status_code == 200
        stats = resp.json()["features"][0]["statistics"]
        assert stats["mean"] == pytest.approx(2020.0, abs=1.0)

    def test_agg_mean_over_interval(self, app_client, zarr_root_time):
        resp = app_client.post(
            "/processes/zonalstats/execution",
            json=self._payload(
                str(zarr_root_time),
                datetime="2020-01-01/2021-12-31",
                agg="mean",
            ),
        )
        assert resp.status_code == 200
        stats = resp.json()["features"][0]["statistics"]
        # mean(2020, 2021) = 2020.5
        assert stats["mean"] == pytest.approx(2020.5, abs=0.1)

    def test_agg_sum_over_interval(self, app_client, zarr_root_time):
        resp = app_client.post(
            "/processes/zonalstats/execution",
            json=self._payload(
                str(zarr_root_time),
                datetime="2020-01-01/2021-12-31",
                agg="sum",
            ),
        )
        assert resp.status_code == 200
        stats = resp.json()["features"][0]["statistics"]
        # sum(2020, 2021) per pixel = 4041.0
        assert stats["mean"] == pytest.approx(4041.0, abs=1.0)

    def test_agg_anomaly(self, app_client, zarr_root_time):
        resp = app_client.post(
            "/processes/zonalstats/execution",
            json=self._payload(
                str(zarr_root_time),
                datetime="2020-01-01/2022-12-31",
                agg="anomaly",
                baseline="2020-01-01/2020-12-31",
            ),
        )
        assert resp.status_code == 200
        stats = resp.json()["features"][0]["statistics"]
        # mean(2020,2021,2022)=2021, baseline mean(2020)=2020 → anomaly=1.0
        assert stats["mean"] == pytest.approx(1.0, abs=0.1)

    def test_agg_anomaly_missing_baseline_returns_422(self, app_client, zarr_root_time):
        resp = app_client.post(
            "/processes/zonalstats/execution",
            json=self._payload(str(zarr_root_time), agg="anomaly"),
        )
        assert resp.status_code == 422

    def test_out_of_range_interval_returns_422(self, app_client, zarr_root_time):
        # An interval that matches zero timesteps raises a 422.
        resp = app_client.post(
            "/processes/zonalstats/execution",
            json=self._payload(str(zarr_root_time), datetime="1900-01-01/1900-12-31", agg="mean"),
        )
        assert resp.status_code == 422


class TestPrepareDaEdgeCases:
    """Edge cases in _prepare_da (lines 188, 204)."""

    def _geom(self) -> dict:
        return {
            "type": "Polygon",
            "coordinates": [[[3.0, 5.0], [8.0, 5.0], [8.0, 9.0], [3.0, 9.0], [3.0, 5.0]]],
        }

    def test_valid_time_coord_renamed(self, app_client, tmp_path):
        """valid_time dimension is renamed to time before aggregation (line 188)."""
        zarr_root = tmp_path / "era5"
        y = np.linspace(14.0, 4.0, 20)
        x = np.linspace(2.0, 15.0, 20)
        times = np.array([np.datetime64("2020-01-01")], dtype="datetime64[ns]")
        data = np.ones((1, 20, 20), dtype="float32") * 99.0
        ds = xr.Dataset(
            {"t2m": (("valid_time", "y", "x"), data)},
            coords={"valid_time": times, "y": y, "x": x},
        )
        ds.to_zarr(str(zarr_root), group="era5/t2m", mode="w", consolidated=True)

        payload = {
            "inputs": {
                "url": str(zarr_root),
                "group": "era5/t2m",
                "variable": "t2m",
                "features": {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {},
                            "geometry": self._geom(),
                        }
                    ],
                },
            }
        }
        resp = app_client.post("/processes/zonalstats/execution", json=payload)
        assert resp.status_code == 200
        stats = resp.json()["features"][0]["statistics"]
        assert stats["mean"] == pytest.approx(99.0, abs=1.0)

    def test_crs_wkt_from_crs_variable(self, app_client, tmp_path):
        """CRS is read from dataset's crs variable crs_wkt attr (line 204)."""
        import rioxarray  # noqa: F401

        zarr_root = tmp_path / "with_crs_var"
        y = np.linspace(14.0, 4.0, 20)
        x = np.linspace(2.0, 15.0, 20)
        data = np.ones((20, 20), dtype="float32") * 55.0
        ds = xr.Dataset(
            {"population": (("y", "x"), data)},
            coords={"y": y, "x": x},
        )
        # Add a 'crs' variable with crs_wkt attribute (mimics CF conventions)
        from pyproj import CRS

        crs_obj = CRS.from_epsg(4326)
        ds["crs"] = xr.DataArray(0, attrs={"crs_wkt": crs_obj.to_wkt()})
        ds.to_zarr(str(zarr_root), group="worldpop/test", mode="w", consolidated=True)

        payload = {
            "inputs": {
                "url": str(zarr_root),
                "group": "worldpop/test",
                "variable": "population",
                "features": {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {},
                            "geometry": self._geom(),
                        }
                    ],
                },
            }
        }
        resp = app_client.post("/processes/zonalstats/execution", json=payload)
        assert resp.status_code == 200
        stats = resp.json()["features"][0]["statistics"]
        assert stats["mean"] == pytest.approx(55.0, abs=1.0)


class TestFeatureStatsClipFailure:
    """_feature_stats returns error dict when clip raises (lines 214-216)."""

    def test_geometry_outside_raster_returns_error(self):
        import rioxarray  # noqa: F401

        from eostrata.ogc.processes import _feature_stats

        y = np.linspace(10.0, 0.0, 20)
        x = np.linspace(0.0, 10.0, 20)
        data = np.full((20, 20), 50.0, dtype="float32")
        da = xr.DataArray(data, dims=("y", "x"), coords={"y": y, "x": x})
        da = da.rio.write_crs("EPSG:4326")

        geom_outside = {
            "type": "Polygon",
            "coordinates": [
                [[100.0, 50.0], [120.0, 50.0], [120.0, 70.0], [100.0, 70.0], [100.0, 50.0]]
            ],
        }
        result = _feature_stats(da, geom_outside)
        assert "error" in result


class TestZonalStatsUnexpectedError:
    def test_unexpected_load_error_returns_500(self, app_client, monkeypatch):
        """Non-HTTP exceptions from _load_array are logged and re-raised as 500."""
        from eostrata.ogc import processes

        monkeypatch.setattr(
            processes, "_load_array", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        resp = app_client.post(
            "/processes/zonalstats/execution",
            json={
                "inputs": {
                    "group": "worldpop/nga",
                    "variable": "population",
                    "features": {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [[[2, 4], [15, 4], [15, 14], [2, 14], [2, 4]]],
                                },
                                "properties": {},
                            }
                        ],
                    },
                }
            },
        )
        assert resp.status_code == 500
        assert "Failed to load dataset" in resp.json()["detail"]
