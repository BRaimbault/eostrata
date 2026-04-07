"""Tests for OGC API - Tiles routes and _resolve helper."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from eostrata.constants import PROP_VARIABLE, PROP_ZARR_GROUP


def _register_item(catalog_path: Path, zarr_root: Path) -> None:
    from eostrata import catalog as cat

    catalogue = cat.load_or_create(catalog_path)
    cat.register_item(
        catalogue,
        collection_id="worldpop",
        item_id="worldpop_nga",
        bbox=(2.0, 4.0, 15.0, 14.0),
        datetime_=datetime(2020, 1, 1, tzinfo=UTC),
        zarr_root=zarr_root,
        zarr_group="worldpop/nga",
        variable="population",
    )
    cat.save(catalogue, catalog_path)


def _write_zarr(zarr_root: Path) -> None:
    import rioxarray  # noqa: F401

    y = np.linspace(14.0, 4.0, 20)
    x = np.linspace(2.0, 15.0, 20)
    times = np.array([np.datetime64("2020-01-01"), np.datetime64("2021-01-01")])
    data = np.ones((2, 20, 20), dtype="float32") * 42.0
    ds = xr.Dataset(
        {"population": (("time", "y", "x"), data)},
        coords={"time": times, "y": y, "x": x},
    )
    ds.to_zarr(str(zarr_root), group="worldpop/nga", mode="w", consolidated=True)


@pytest.fixture()
def setup(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    zarr_root = tmp_path / "zarr"
    _write_zarr(zarr_root)
    _register_item(catalog_path, zarr_root)
    return tmp_path, catalog_path, zarr_root


@pytest.fixture()
def app_client(setup, mocker):
    tmp_path, catalog_path, zarr_root = setup

    mock_settings = mocker.MagicMock()
    mock_settings.catalog_path = catalog_path
    mock_settings.zarr_root = zarr_root
    mock_settings.bbox = (2.0, 4.0, 15.0, 14.0)

    mocker.patch("eostrata.ogc.tiles.settings", mock_settings)
    mocker.patch("eostrata.ogc.processes.settings", mock_settings)
    mocker.patch("eostrata.server.settings", mock_settings)
    from eostrata.server import app

    yield TestClient(app)


# ── Unit tests for _resolve ────────────────────────────────────────────────────


class TestResolve:
    def test_found_with_item(self, setup, mocker):
        tmp_path, catalog_path, zarr_root = setup

        from eostrata.ogc.tiles import _resolve

        mock_settings = mocker.MagicMock()
        mock_settings.catalog_path = catalog_path
        mock_settings.zarr_root = zarr_root

        mocker.patch("eostrata.ogc.tiles.settings", mock_settings)
        result = _resolve("worldpop", "worldpop_nga")
        assert result["zarr_group"] == "worldpop/nga"
        assert result["variable"] == "population"

    def test_missing_collection_raises_404(self, setup, mocker):
        tmp_path, catalog_path, zarr_root = setup

        from fastapi import HTTPException

        from eostrata.ogc.tiles import _resolve

        mock_settings = mocker.MagicMock()
        mock_settings.catalog_path = catalog_path

        mocker.patch("eostrata.ogc.tiles.settings", mock_settings)
        with pytest.raises(HTTPException) as exc_info:
            _resolve("nonexistent", None)
        assert exc_info.value.status_code == 404

    def test_missing_item_raises_404(self, setup, mocker):
        tmp_path, catalog_path, zarr_root = setup

        from fastapi import HTTPException

        from eostrata.ogc.tiles import _resolve

        mock_settings = mocker.MagicMock()
        mock_settings.catalog_path = catalog_path

        mocker.patch("eostrata.ogc.tiles.settings", mock_settings)
        with pytest.raises(HTTPException) as exc_info:
            _resolve("worldpop", "unknown_item")
        assert exc_info.value.status_code == 404

    def test_no_item_picks_first(self, setup, mocker):
        tmp_path, catalog_path, zarr_root = setup

        from eostrata.ogc.tiles import _resolve

        mock_settings = mocker.MagicMock()
        mock_settings.catalog_path = catalog_path
        mock_settings.zarr_root = zarr_root

        mocker.patch("eostrata.ogc.tiles.settings", mock_settings)
        result = _resolve("worldpop", None)
        assert result["zarr_group"] == "worldpop/nga"

    def test_empty_collection_raises_404(self, tmp_path, mocker):
        """A collection with no items should raise 404."""
        from fastapi import HTTPException

        from eostrata import catalog as cat
        from eostrata.ogc.tiles import _resolve

        cat_path = tmp_path / "catalog.json"
        cat.save(cat._make_catalog(), cat_path)

        mock_settings = mocker.MagicMock()
        mock_settings.catalog_path = cat_path

        mocker.patch("eostrata.ogc.tiles.settings", mock_settings)
        with pytest.raises(HTTPException) as exc_info:
            _resolve("worldpop", None)  # worldpop collection exists but has no items
        assert exc_info.value.status_code == 404

    def test_item_without_zarr_asset_raises_422(self, tmp_path, mocker):
        """An item without a zarr asset should raise 422."""
        from datetime import datetime

        import pystac
        from fastapi import HTTPException

        from eostrata import catalog as cat
        from eostrata.ogc.tiles import _resolve

        cat_path = tmp_path / "catalog.json"
        catalogue = cat._make_catalog()
        # Add item without zarr asset
        coll = catalogue.get_child("worldpop")
        item = pystac.Item(
            id="bare_item",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
            bbox=[0, 0, 1, 1],
            datetime=datetime(2020, 1, 1, tzinfo=UTC),
            properties={
                PROP_ZARR_GROUP: "worldpop/x",
                PROP_VARIABLE: "v",
            },
        )
        # Do NOT add zarr asset
        coll.add_item(item)
        cat.save(catalogue, cat_path)

        mock_settings = mocker.MagicMock()
        mock_settings.catalog_path = cat_path

        mocker.patch("eostrata.ogc.tiles.settings", mock_settings)
        with pytest.raises(HTTPException) as exc_info:
            _resolve("worldpop", "bare_item")
        assert exc_info.value.status_code == 422

    def test_resolve_cache_hit_on_second_call(self, setup, mocker):
        """Second _resolve call with same args returns cached result (line 105-106)."""
        tmp_path, catalog_path, zarr_root = setup

        import eostrata.ogc.tiles as tiles_mod
        from eostrata.ogc.tiles import _resolve

        mock_settings = mocker.MagicMock()
        mock_settings.catalog_path = catalog_path
        mock_settings.zarr_root = zarr_root

        mocker.patch("eostrata.ogc.tiles.settings", mock_settings)

        # Force cache miss on first call by resetting the cache
        tiles_mod._resolve_cache = {}
        tiles_mod._resolve_cache_catalog_id = 0

        result1 = _resolve("worldpop", "worldpop_nga")
        # Second call should return cached result
        result2 = _resolve("worldpop", "worldpop_nga")

        assert result1 == result2
        assert result1["zarr_group"] == "worldpop/nga"


# ── Internal app — variables endpoint ─────────────────────────────────────────


class TestVariablesEndpoint:
    def test_variables_lists_data_vars(self, setup, mocker):
        """GET /internal/variables returns list of dataset variables (lines 53-54)."""
        tmp_path, catalog_path, zarr_root = setup

        from eostrata.ogc.tiles import _internal_app

        zarr_url = str(zarr_root)
        client = TestClient(_internal_app)
        resp = client.get(
            "/internal/variables",
            params={"url": zarr_url, "group": "worldpop/nga"},
        )
        assert resp.status_code == 200
        assert "population" in resp.json()


# ── Route tests ────────────────────────────────────────────────────────────────


class TestTileRoutes:
    def test_tilejson_200(self, app_client):
        resp = app_client.get(
            "/collections/worldpop/tiles/WebMercatorQuad/tilejson.json?item=worldpop_nga"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "tiles" in data
        assert data["tilejson"] == "2.2.0"

    def test_tilejson_with_optional_params(self, app_client):
        resp = app_client.get(
            "/collections/worldpop/tiles/WebMercatorQuad/tilejson.json"
            "?item=worldpop_nga&datetime=2020-01-01&agg=mean"
            "&colormap_name=viridis&rescale=0,100"
        )
        assert resp.status_code == 200
        tile_url = resp.json()["tiles"][0]
        assert "datetime" in tile_url
        assert "agg" in tile_url

    def test_tilejson_with_baseline(self, app_client):
        """baseline param is included in tile URL (line 162)."""
        resp = app_client.get(
            "/collections/worldpop/tiles/WebMercatorQuad/tilejson.json"
            "?item=worldpop_nga&agg=anomaly&baseline=2019-01-01/2019-12-31"
        )
        assert resp.status_code == 200
        tile_url = resp.json()["tiles"][0]
        assert "baseline" in tile_url

    def test_tilejson_unknown_collection_404(self, app_client):
        resp = app_client.get("/collections/unknown/tiles/WebMercatorQuad/tilejson.json")
        assert resp.status_code == 404

    def test_map_html_returns_meta_refresh(self, app_client):
        resp = app_client.get(
            "/collections/worldpop/tiles/WebMercatorQuad/map.html?item=worldpop_nga",
            follow_redirects=False,
        )
        assert resp.status_code == 200
        assert "meta" in resp.text.lower()

    def test_map_html_with_optional_params(self, app_client):
        resp = app_client.get(
            "/collections/worldpop/tiles/WebMercatorQuad/map.html"
            "?item=worldpop_nga&colormap_name=viridis&rescale=0,1000"
        )
        assert resp.status_code == 200

    def test_map_html_with_datetime_agg_baseline(self, app_client):
        """datetime, agg, baseline are forwarded to /map URL (lines 118, 120, 122)."""
        resp = app_client.get(
            "/collections/worldpop/tiles/WebMercatorQuad/map.html"
            "?item=worldpop_nga&datetime=2020-01-01/2021-12-31"
            "&agg=anomaly&baseline=2019-01-01/2019-12-31",
            follow_redirects=False,
        )
        assert resp.status_code == 200
        assert "datetime" in resp.text
        assert "agg" in resp.text
        assert "baseline" in resp.text

    def test_map_html_unknown_collection_404(self, app_client):
        resp = app_client.get("/collections/missing/tiles/WebMercatorQuad/map.html")
        assert resp.status_code == 404

    def test_info_endpoint_real_delegate(self, app_client):
        """_delegate body is exercised (lines 81-86) via info endpoint without mocking."""
        resp = app_client.get("/collections/worldpop/info?item=worldpop_nga")
        # TiTiler processes the real zarr — any non-server-error response is acceptable
        assert resp.status_code != 500

    def test_info_endpoint_calls_delegate(self, setup, mocker):
        """Collection info delegates to TiTiler — verify _delegate is invoked."""
        tmp_path, catalog_path, zarr_root = setup

        from fastapi.responses import Response

        mock_settings = mocker.MagicMock()
        mock_settings.catalog_path = catalog_path
        mock_settings.zarr_root = zarr_root

        async def fake_delegate(path, params):
            return Response(content=b'{"info": "ok"}', media_type="application/json")

        mocker.patch("eostrata.ogc.tiles.settings", mock_settings)
        mocker.patch("eostrata.ogc.tiles._delegate", side_effect=fake_delegate)
        from eostrata.server import app

        client = TestClient(app)
        resp = client.get("/collections/worldpop/info?item=worldpop_nga")

        assert resp.status_code == 200

    def test_tile_endpoint_calls_delegate(self, setup, mocker):
        """Collection tile delegates to TiTiler — verify _delegate is invoked."""
        tmp_path, catalog_path, zarr_root = setup

        from fastapi.responses import Response

        mock_settings = mocker.MagicMock()
        mock_settings.catalog_path = catalog_path
        mock_settings.zarr_root = zarr_root

        async def fake_delegate(path, params):
            return Response(content=b"\x89PNG", media_type="image/png")

        mocker.patch("eostrata.ogc.tiles.settings", mock_settings)
        mocker.patch("eostrata.ogc.tiles._delegate", side_effect=fake_delegate)
        from eostrata.server import app

        client = TestClient(app)
        # z=1,x=1,y=0 covers (0°,0°,180°,85°N) — overlaps Nigeria bbox (2,4,15,14)
        resp = client.get("/collections/worldpop/tiles/WebMercatorQuad/1/1/0?item=worldpop_nga")

        assert resp.status_code == 200

    def test_tile_with_optional_params(self, setup, mocker):
        """Optional colormap/rescale params are forwarded to TiTiler."""
        tmp_path, catalog_path, zarr_root = setup

        from fastapi.responses import Response

        mock_settings = mocker.MagicMock()
        mock_settings.catalog_path = catalog_path
        mock_settings.zarr_root = zarr_root

        captured = {}

        async def fake_delegate(path, params):
            captured.update(params)
            return Response(content=b"\x89PNG", media_type="image/png")

        mocker.patch("eostrata.ogc.tiles.settings", mock_settings)
        mocker.patch("eostrata.ogc.tiles._delegate", side_effect=fake_delegate)
        from eostrata.server import app

        client = TestClient(app)
        # z=1,x=1,y=0 covers (0°,0°,180°,85°N) — overlaps Nigeria bbox (2,4,15,14)
        client.get(
            "/collections/worldpop/tiles/WebMercatorQuad/1/1/0"
            "?item=worldpop_nga&datetime=2020-01-01&colormap_name=viridis&rescale=0,100"
        )

        assert "colormap_name" in captured
        assert "rescale" in captured
        # time is handled via context vars, not forwarded as sel
        assert "sel" not in captured

    def test_tile_agg_params_set_context_vars(self, setup, mocker):
        """agg/baseline/datetime are propagated to AggregatingReader via ContextVar."""
        tmp_path, catalog_path, zarr_root = setup

        from fastapi.responses import Response

        from eostrata.aggregate import _CTX_AGG_BASELINE, _CTX_AGG_DATETIME, _CTX_AGG_METHOD

        mock_settings = mocker.MagicMock()
        mock_settings.catalog_path = catalog_path
        mock_settings.zarr_root = zarr_root

        ctx_snapshot = {}

        async def fake_delegate(path, params):
            ctx_snapshot["datetime"] = _CTX_AGG_DATETIME.get()
            ctx_snapshot["agg"] = _CTX_AGG_METHOD.get()
            ctx_snapshot["baseline"] = _CTX_AGG_BASELINE.get()
            return Response(content=b"\x89PNG", media_type="image/png")

        mocker.patch("eostrata.ogc.tiles.settings", mock_settings)
        mocker.patch("eostrata.ogc.tiles._delegate", side_effect=fake_delegate)
        from eostrata.server import app

        client = TestClient(app)
        # z=1,x=1,y=0 covers (0°,0°,180°,85°N) — overlaps Nigeria bbox (2,4,15,14)
        client.get(
            "/collections/worldpop/tiles/WebMercatorQuad/1/1/0"
            "?item=worldpop_nga"
            "&datetime=2020-01-01/2021-12-31"
            "&agg=mean"
            "&baseline=2019-01-01/2019-12-31"
        )

        assert ctx_snapshot["datetime"] == "2020-01-01/2021-12-31"
        assert ctx_snapshot["agg"] == "mean"
        assert ctx_snapshot["baseline"] == "2019-01-01/2019-12-31"

    def test_tile_outside_bbox_returns_empty_png(self, setup, mocker):
        """Tile entirely outside item bbox returns 200 with transparent PNG, no delegate call."""
        tmp_path, catalog_path, zarr_root = setup

        mock_settings = mocker.MagicMock()
        mock_settings.catalog_path = catalog_path
        mock_settings.zarr_root = zarr_root

        delegate_called = []

        async def fake_delegate(path, params):
            delegate_called.append(True)
            from fastapi.responses import Response

            return Response(content=b"\x89PNG", media_type="image/png")

        mocker.patch("eostrata.ogc.tiles.settings", mock_settings)
        mocker.patch("eostrata.ogc.tiles._delegate", side_effect=fake_delegate)
        from eostrata.server import app

        client = TestClient(app)
        # z=1,x=0,y=0 covers (-180°,0°,0°,85°N) — does NOT overlap Nigeria bbox (2,4,15,14)
        resp = client.get("/collections/worldpop/tiles/WebMercatorQuad/1/0/0?item=worldpop_nga")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.content[:4] == b"\x89PNG"
        assert not delegate_called, "_delegate should not be called for out-of-bbox tile"


# ── _tile_bbox unit tests ──────────────────────────────────────────────────────


class TestTileBbox:
    def test_zoom0_covers_world(self):
        from eostrata.ogc.tiles import _tile_bbox

        west, south, east, north = _tile_bbox(0, 0, 0)
        assert west == pytest.approx(-180.0)
        assert east == pytest.approx(180.0)
        assert north == pytest.approx(85.0511, abs=0.01)
        assert south == pytest.approx(-85.0511, abs=0.01)

    def test_zoom1_quadrants(self):
        from eostrata.ogc.tiles import _tile_bbox

        # Top-left quadrant: west hemisphere, north
        w, s, e, n = _tile_bbox(1, 0, 0)
        assert w == pytest.approx(-180.0)
        assert e == pytest.approx(0.0)
        assert s == pytest.approx(0.0)
        assert n > 0.0

        # Top-right quadrant: east hemisphere, north — overlaps Nigeria
        w, s, e, n = _tile_bbox(1, 1, 0)
        assert w == pytest.approx(0.0)
        assert e == pytest.approx(180.0)
