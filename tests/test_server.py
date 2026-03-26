"""Tests for the FastAPI server — OGC Common endpoints."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient


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


class TestCollections:
    def test_empty_store_returns_predefined_collections(self, client):
        data = client.get("/collections").json()
        assert "collections" in data
        # The catalog always has 3 predefined collections (worldpop, cds, chirps)
        # even when no data has been ingested yet
        assert isinstance(data["collections"], list)
        assert len(data["collections"]) == 3
        ids = {c["id"] for c in data["collections"]}
        assert ids == {"worldpop", "cds", "chirps"}

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
