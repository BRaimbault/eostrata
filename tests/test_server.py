"""Tests for the FastAPI server — OGC Common endpoints."""

from __future__ import annotations

import sys
from datetime import UTC, datetime

import pystac
import pytest
from fastapi.testclient import TestClient

from eostrata.constants import PROP_VARIABLE, PROP_ZARR_GROUP


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient with catalog and zarr_root pointed at tmp_path."""
    monkeypatch.setenv("EOSTRATA_CATALOG_PATH", str(tmp_path / "catalog.json"))
    monkeypatch.setenv("EOSTRATA_ZARR_ROOT", str(tmp_path / "zarr"))

    # Reload settings so env vars take effect
    import importlib

    import eostrata.config as cfg_mod

    importlib.reload(cfg_mod)

    from eostrata.server import app

    return TestClient(app, raise_server_exceptions=True)


class TestLandingPage:
    def test_status_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_has_title(self, client):
        data = client.get("/").json()
        assert "title" in data

    def test_has_links(self, client):
        data = client.get("/").json()
        assert "links" in data
        rels = {link["rel"] for link in data["links"]}
        assert "conformance" in rels
        assert "data" in rels


class TestConformance:
    def test_status_200(self, client):
        assert client.get("/conformance").status_code == 200

    def test_conforms_to_contains_ogc_common(self, client):
        data = client.get("/conformance").json()
        conforms = data["conformsTo"]
        assert any("ogcapi-common" in c for c in conforms)

    def test_conforms_to_contains_tiles(self, client):
        data = client.get("/conformance").json()
        conforms = data["conformsTo"]
        assert any("ogcapi-tiles" in c for c in conforms)


class TestExamples:
    def test_empty_catalog_returns_warning(self, client):
        resp = client.get("/examples")
        assert resp.status_code == 200
        data = resp.json()
        assert "warning" in data
        assert data["items"] == []

    def test_with_items_returns_item_list(self, tmp_path, monkeypatch, mocker):
        from eostrata import catalog as cat

        catalog_path = tmp_path / "catalog.json"
        catalogue = cat.load_or_create(catalog_path)
        cat.register_item(
            catalogue,
            collection_id="worldpop",
            item_id="worldpop_nga",
            bbox=(2.0, 4.0, 15.0, 14.0),
            datetime_=datetime(2020, 1, 1, tzinfo=UTC),
            zarr_root=tmp_path / "zarr",
            zarr_group="worldpop/nga",
            variable="population",
        )
        cat.save(catalogue, catalog_path)

        mock_settings = mocker.MagicMock()
        mock_settings.catalog_path = catalog_path

        from eostrata.server import app

        mocker.patch("eostrata.server.settings", mock_settings)
        with TestClient(app) as c:
            resp = c.get("/examples")

        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert len(data["items"]) == 1
        item = data["items"][0]
        assert item["collection_id"] == "worldpop"
        assert item["item_id"] == "worldpop_nga"
        assert "endpoints" in item
        assert "zonalstats_body" in item


class TestLifespanStorageCheck:
    def test_unwritable_storage_dir_raises(self, tmp_path, monkeypatch):
        """lifespan should raise RuntimeError if a storage directory is not writable."""
        import importlib
        from unittest.mock import patch

        monkeypatch.setenv("EOSTRATA_CATALOG_PATH", str(tmp_path / "catalog.json"))
        monkeypatch.setenv("EOSTRATA_ZARR_ROOT", str(tmp_path / "zarr"))

        import eostrata.config as cfg_mod

        importlib.reload(cfg_mod)

        from eostrata.server import app

        with (
            patch("os.access", return_value=False),
            pytest.raises(RuntimeError, match="not writable"),
            TestClient(app),
        ):
            pass


class TestMapViewer:
    def test_returns_html(self, client):
        resp = client.get("/map")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "<html" in resp.text.lower()

    def test_preselect_params_accepted(self, client):
        resp = client.get(
            "/map",
            params={
                "collection": "worldpop",
                "item": "worldpop_nga",
                "datetime": "2020-01-01",
                "agg": "mean",
                "baseline": "2015-01-01/2019-12-31",
                "colormap_name": "viridis",
                "rescale": "0,1000",
            },
        )
        assert resp.status_code == 200
        assert "worldpop" in resp.text
        assert "viridis" in resp.text

    def test_ingest_tab_present(self, client):
        resp = client.get("/map")
        assert resp.status_code == 200
        assert "tab-ingest-content" in resp.text
        assert "startIngest" in resp.text
        assert "loadJobs" in resp.text
        assert "rebuildCatalog" in resp.text


class TestSchedulerUI:
    def test_scheduler_returns_html(self, client):
        """GET /scheduler returns the static scheduler HTML (line 595)."""
        resp = client.get("/scheduler")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


class TestDynamicOpenAPI:
    def test_openapi_schema_accessible(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "eostrata"
        assert "paths" in schema

    def test_openapi_cache_hit_on_second_call(self, mocker):
        """Second /openapi.json call returns cached schema (line 679)."""
        import eostrata.server as server_mod

        mock_settings = mocker.MagicMock()
        mock_settings.catalog_path.stat.return_value.st_mtime = 12345.0

        mocker.patch("eostrata.server.settings", mock_settings)
        mocker.patch("eostrata.server.load_or_create", return_value=mocker.MagicMock())

        # Reset module-level cache so the first call populates it
        server_mod._openapi_schema_cache = None
        server_mod._openapi_catalog_mtime = 0.0

        from eostrata.server import app

        with TestClient(app) as c:
            r1 = c.get("/openapi.json")
            r2 = c.get("/openapi.json")

        assert r1.status_code == 200
        assert r2.status_code == 200

    def test_openapi_with_catalog_items(self, tmp_path, mocker):
        from eostrata import catalog as cat

        catalog_path = tmp_path / "catalog.json"
        catalogue = cat.load_or_create(catalog_path)
        cat.register_item(
            catalogue,
            collection_id="chirps",
            item_id="chirps_global",
            bbox=(-180.0, -50.0, 180.0, 50.0),
            datetime_=datetime(2023, 6, 1, tzinfo=UTC),
            zarr_root=tmp_path / "zarr",
            zarr_group="chirps/global",
            variable="precipitation",
        )
        cat.save(catalogue, catalog_path)

        mock_settings = mocker.MagicMock()
        mock_settings.catalog_path = catalog_path

        from eostrata.server import app

        mocker.patch("eostrata.server.settings", mock_settings)
        with TestClient(app) as c:
            resp = c.get("/openapi.json")

        assert resp.status_code == 200
        schema = resp.json()
        assert any("{collection_id}" in p for p in schema["paths"])


class TestCollections:
    def test_empty_store_returns_predefined_collections(self, client):
        from eostrata.sources.base import all_sources

        data = client.get("/collections").json()
        assert "collections" in data
        assert isinstance(data["collections"], list)
        expected_ids = {cls.collection_id for cls in all_sources()}
        assert len(data["collections"]) == len(expected_ids)
        assert {c["id"] for c in data["collections"]} == expected_ids

    def test_with_registered_item(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EOSTRATA_CATALOG_PATH", str(tmp_path / "catalog.json"))
        monkeypatch.setenv("EOSTRATA_ZARR_ROOT", str(tmp_path / "zarr"))

        import importlib

        import eostrata.config as cfg_mod

        importlib.reload(cfg_mod)

        from eostrata import catalog as cat
        from eostrata.config import settings

        catalogue = cat.load_or_create(settings.catalog_path)
        cat.register_item(
            catalogue,
            collection_id="worldpop",
            item_id="worldpop_nga",
            bbox=(2.0, 4.0, 15.0, 14.0),
            datetime_=datetime(2020, 1, 1, tzinfo=UTC),
            zarr_root=tmp_path / "zarr",
            zarr_group="worldpop/nga",
            variable="population",
        )
        cat.save(catalogue, settings.catalog_path)

        from eostrata.server import app

        with TestClient(app) as c:
            data = c.get("/collections").json()
        coll_ids = [c["id"] for c in data["collections"]]
        assert "worldpop" in coll_ids


class TestStoreUsage:
    def test_status_200(self, client):
        assert client.get("/store-usage").status_code == 200

    def test_unlimited_when_quota_zero(self, tmp_path, mocker):
        mock_settings = mocker.MagicMock()
        mock_settings.store_quota_mb = 0
        mock_settings.zarr_root = tmp_path / "zarr"

        from eostrata.server import app

        mocker.patch("eostrata.server.settings", mock_settings)
        with TestClient(app) as c:
            data = c.get("/store-usage").json()
        assert data["quota_unlimited"] is True
        assert data["used_pct"] is None
        assert data["used_mb"] >= 0.0

    def test_quota_set_returns_percent(self, tmp_path, mocker):
        mock_settings = mocker.MagicMock()
        mock_settings.store_quota_mb = 1000
        mock_settings.zarr_root = tmp_path / "zarr"

        from eostrata.server import app

        mocker.patch("eostrata.server.settings", mock_settings)
        mocker.patch("eostrata.cache.store_size_mb", return_value=250.0)
        with TestClient(app) as c:
            data = c.get("/store-usage").json()
        assert data["quota_unlimited"] is False
        assert data["quota_mb"] == 1000
        assert data["used_pct"] == 25.0

    def test_groups_field_present(self, client):
        data = client.get("/store-usage").json()
        assert "groups" in data
        assert isinstance(data["groups"], list)

    def test_groups_include_timestamps(self, tmp_path, mocker):
        import numpy as np
        import xarray as xr

        zarr_root = tmp_path / "zarr"
        times = np.array([np.datetime64("2020-01-01"), np.datetime64("2021-01-01")])
        ds = xr.Dataset({"v": (("time", "y", "x"), np.zeros((2, 4, 4)))}, coords={"time": times})
        ds.to_zarr(str(zarr_root), group="worldpop/nga", mode="w")

        mock_settings = mocker.MagicMock()
        mock_settings.store_quota_mb = 0
        mock_settings.zarr_root = zarr_root

        from eostrata.server import app

        mocker.patch("eostrata.server.settings", mock_settings)
        with TestClient(app) as c:
            data = c.get("/store-usage").json()
        assert len(data["groups"]) == 1
        g = data["groups"][0]
        assert g["group"] == "worldpop/nga"
        assert g["size_mb"] >= 0
        assert "timestamps" in g
        assert len(g["timestamps"]) == 2
        t = g["timestamps"][0]
        assert "datetime" in t
        assert "size_mb" in t
        assert "last_accessed" in t

    def test_group_without_time_dimension_excluded(self, tmp_path, mocker):
        """A Zarr group with no time dimension should be skipped in the response."""
        import numpy as np
        import xarray as xr

        zarr_root = tmp_path / "zarr"
        # Write a group with no time dimension
        ds = xr.Dataset({"v": (("y", "x"), np.zeros((4, 4)))})
        ds.to_zarr(str(zarr_root), group="worldpop/nga", mode="w")

        mock_settings = mocker.MagicMock()
        mock_settings.store_quota_mb = 0
        mock_settings.zarr_root = zarr_root

        from eostrata.server import app

        mocker.patch("eostrata.server.settings", mock_settings)
        with TestClient(app) as c:
            data = c.get("/store-usage").json()
        assert data["groups"] == []


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_catalog_with_item_no_datetimes() -> pystac.Catalog:
    """Return a catalog with a sub-catalog child and a collection whose item
    lacks ``eostrata:datetimes`` but has ``start_datetime``."""
    catalog = pystac.Catalog(id="root", description="root")
    # non-Collection child — exercises the isinstance guard
    catalog.add_child(pystac.Catalog(id="sub", description="sub"))
    collection = pystac.Collection(
        id="worldpop",
        description="test",
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent([[-180, -90, 180, 90]]),
            temporal=pystac.TemporalExtent([[None, None]]),
        ),
    )
    item = pystac.Item(
        id="test_item",
        geometry=None,
        bbox=[0.0, 0.0, 10.0, 10.0],
        datetime=None,
        properties={
            "start_datetime": "2021-01-01T00:00:00+00:00",
            "end_datetime": "2021-12-31T00:00:00+00:00",
            "datetime": None,
            PROP_VARIABLE: "population",
            PROP_ZARR_GROUP: "worldpop/test",
            # deliberately NO eostrata:datetimes
        },
    )
    collection.add_item(item)
    catalog.add_child(collection)
    return catalog


# ── Lifespan ──────────────────────────────────────────────────────────────────


class TestLifespan:
    def test_scheduler_import_error_is_handled(self, mocker):

        from eostrata.server import app

        mocker.patch.dict(sys.modules, {"eostrata.scheduler": None})
        with TestClient(app) as c:
            assert c.get("/").status_code == 200

    def test_scheduler_runtime_error_is_handled(self, mocker):

        from eostrata.server import app

        mock_mod = mocker.MagicMock()
        mock_mod.Scheduler.return_value.start.side_effect = RuntimeError("boom")
        mocker.patch.dict(sys.modules, {"eostrata.scheduler": mock_mod})
        with TestClient(app) as c:
            assert c.get("/").status_code == 200

    def test_non_writable_storage_dir_raises(self, tmp_path, mocker):
        """Lifespan raises RuntimeError when a storage directory is not writable (line 95)."""

        from eostrata.server import app

        mock_settings = mocker.MagicMock()
        mock_settings.zarr_root = tmp_path / "zarr"
        mock_settings.raw_dir = tmp_path / "raw"

        mocker.patch("eostrata.server.settings", mock_settings)
        mocker.patch("os.access", return_value=False)

        with pytest.raises(RuntimeError, match="not writable"), TestClient(app):
            pass


# ── examples() edge cases ─────────────────────────────────────────────────────


class TestExamplesEdgeCases:
    def test_non_collection_child_skipped(self, mocker):
        """Catalog children that are not pystac.Collection instances are skipped."""
        catalog = _make_catalog_with_item_no_datetimes()
        mock_settings = mocker.MagicMock()

        from eostrata.server import app

        mocker.patch("eostrata.server.settings", mock_settings)
        mocker.patch("eostrata.server.load_or_create", return_value=catalog)
        with TestClient(app) as c:
            resp = c.get("/examples")

        assert resp.status_code == 200
        # Only the collection item should appear (sub-catalog child is skipped)
        assert len(resp.json()["items"]) == 1

    def test_fallback_datetime_from_start_datetime(self, mocker):
        """Items missing eostrata:datetimes fall back to start_datetime."""
        catalog = _make_catalog_with_item_no_datetimes()
        mock_settings = mocker.MagicMock()

        from eostrata.server import app

        mocker.patch("eostrata.server.settings", mock_settings)
        mocker.patch("eostrata.server.load_or_create", return_value=catalog)
        with TestClient(app) as c:
            resp = c.get("/examples")

        data = resp.json()
        assert len(data["items"]) == 1
        assert "2021-01-01" in data["items"][0]["available_datetimes"][0]


# ── _catalog_openapi_examples() and _dynamic_openapi() edge cases ─────────────


class TestOpenAPIEdgeCases:
    def test_openapi_fallback_datetime_from_properties(self, mocker):
        """_catalog_openapi_examples falls back to start_datetime when
        eostrata:datetimes is absent; non-Collection children are also skipped."""
        catalog = _make_catalog_with_item_no_datetimes()
        mock_settings = mocker.MagicMock()

        from eostrata.server import app

        # _catalog_openapi_examples uses the module-level load_or_create binding
        mocker.patch("eostrata.server.settings", mock_settings)
        mocker.patch("eostrata.server.load_or_create", return_value=catalog)
        with TestClient(app) as c:
            resp = c.get("/openapi.json")

        assert resp.status_code == 200
        assert "paths" in resp.json()

    def test_openapi_catalog_exception_is_silenced(self, mocker):
        """_catalog_openapi_examples swallows any catalog read error."""
        mock_settings = mocker.MagicMock()

        from eostrata.server import app

        mocker.patch("eostrata.server.settings", mock_settings)
        mocker.patch("eostrata.server.load_or_create", side_effect=OSError("no catalog"))
        with TestClient(app) as c:
            resp = c.get("/openapi.json")

        assert resp.status_code == 200

    def test_openapi_non_dict_operation_skipped(self, mocker):
        """Non-dict values inside a path item (e.g. a 'summary' string) are
        skipped without error."""
        from eostrata.server import _dynamic_openapi

        mock_schema = {
            "info": {"title": "eostrata", "version": "0.1.0"},
            "paths": {
                "/collections/{collection_id}/tiles/{tileMatrixSetId}/{z}/{x}/{y}": {
                    "summary": "top-level string, not an operation dict",
                    "get": {
                        "parameters": [
                            {"name": "collection_id", "in": "path"},
                            {"name": "tileMatrixSetId", "in": "path"},
                        ]
                    },
                }
            },
        }

        mocker.patch("eostrata.server.get_openapi", return_value=mock_schema)
        schema = _dynamic_openapi()

        assert schema["paths"] is not None
        # The string "summary" value was skipped; the get operation was processed
        get_op = schema["paths"][
            "/collections/{collection_id}/tiles/{tileMatrixSetId}/{z}/{x}/{y}"
        ]["get"]
        assert "parameters" in get_op


# ── RFC 7807 error handlers ───────────────────────────────────────────────────


class TestOGCErrorHandlers:
    def test_404_returns_problem_details(self, client):
        resp = client.get("/does-not-exist")
        assert resp.status_code == 404
        data = resp.json()
        assert data["type"] == "about:blank"
        assert data["status"] == 404
        assert data["title"] == "Not Found"
        assert "detail" in data

    def test_validation_error_returns_problem_details(self, client):
        # POST to ingest with invalid body triggers RequestValidationError
        resp = client.post(
            "/processes/ingest/execution",
            json={"inputs": {"source": "worldpop"}},  # missing iso3
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["type"] == "about:blank"
        assert data["status"] == 422
        assert data["title"] == "Unprocessable Content"
        assert "errors" in data
