"""Tests for catalog.py — STAC catalogue management."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pystac
import pytest

from eostrata.catalog import (
    PystacClient,
    _make_catalog,
    load_or_create,
    register_item,
    resolve_item,
    save,
)

_BBOX = (2.0, 4.0, 6.0, 8.0)
_DT = datetime(2020, 1, 1, tzinfo=UTC)


def _catalog_with_item(tmp_path: Path, item_id: str = "worldpop_nga") -> pystac.Catalog:
    cat = _make_catalog()
    register_item(
        cat,
        collection_id="worldpop",
        item_id=item_id,
        bbox=_BBOX,
        datetime_=_DT,
        zarr_root=tmp_path / "zarr",
        zarr_group="worldpop/nga",
        variable="population",
        extra_properties={"eostrata:iso3": "NGA"},
    )
    return cat


class TestMakeCatalog:
    def test_has_three_collections(self):
        cat = _make_catalog()
        colls = list(cat.get_children())
        assert len(colls) == 3

    def test_collection_ids(self):
        cat = _make_catalog()
        ids = {c.id for c in cat.get_children()}
        assert ids == {"worldpop", "cds", "chirps"}


class TestLoadOrCreate:
    def test_creates_new_when_missing(self, tmp_path):
        cat = load_or_create(tmp_path / "catalog.json")
        assert cat.id == "eostrata"

    def test_loads_existing(self, tmp_path):
        cat = _make_catalog()
        save(cat, tmp_path / "catalog.json")
        loaded = load_or_create(tmp_path / "catalog.json")
        assert loaded.id == "eostrata"


class TestSave:
    def test_writes_json(self, tmp_path):
        cat = _make_catalog()
        save(cat, tmp_path / "catalog.json")
        assert (tmp_path / "catalog.json").exists()

    def test_creates_parent_dirs(self, tmp_path):
        cat = _make_catalog()
        save(cat, tmp_path / "nested" / "deep" / "catalog.json")
        assert (tmp_path / "nested" / "deep" / "catalog.json").exists()


class TestRegisterItem:
    def test_creates_item_in_collection(self, tmp_path):
        cat = _catalog_with_item(tmp_path)
        coll = cat.get_child("worldpop")
        item = coll.get_item("worldpop_nga")
        assert item is not None

    def test_item_has_zarr_asset(self, tmp_path):
        cat = _catalog_with_item(tmp_path)
        item = cat.get_child("worldpop").get_item("worldpop_nga")
        assert "zarr" in item.assets
        asset = item.assets["zarr"]
        assert "xarray:open_kwargs" in asset.extra_fields

    def test_item_extra_properties(self, tmp_path):
        cat = _catalog_with_item(tmp_path)
        item = cat.get_child("worldpop").get_item("worldpop_nga")
        assert item.properties["eostrata:iso3"] == "NGA"

    def test_item_has_datetimes_property(self, tmp_path):
        cat = _catalog_with_item(tmp_path)
        item = cat.get_child("worldpop").get_item("worldpop_nga")
        assert item.properties["eostrata:datetimes"] == [_DT.isoformat()]

    def test_extending_existing_item_expands_interval(self, tmp_path):
        cat = _catalog_with_item(tmp_path)
        dt2 = datetime(2022, 1, 1, tzinfo=UTC)
        register_item(
            cat,
            collection_id="worldpop",
            item_id="worldpop_nga",
            bbox=_BBOX,
            datetime_=dt2,
            zarr_root=tmp_path / "zarr",
            zarr_group="worldpop/nga",
            variable="population",
        )
        item = cat.get_child("worldpop").get_item("worldpop_nga")
        assert item.common_metadata.start_datetime == _DT
        assert item.common_metadata.end_datetime == dt2

    def test_gap_in_series_is_preserved_in_datetimes(self, tmp_path):
        """Registering 2020 and 2022 (not 2021) keeps both timestamps, exposing the gap."""
        cat = _catalog_with_item(tmp_path)  # 2020
        dt2022 = datetime(2022, 1, 1, tzinfo=UTC)
        register_item(
            cat,
            collection_id="worldpop",
            item_id="worldpop_nga",
            bbox=_BBOX,
            datetime_=dt2022,
            zarr_root=tmp_path / "zarr",
            zarr_group="worldpop/nga",
            variable="population",
        )
        item = cat.get_child("worldpop").get_item("worldpop_nga")
        datetimes = item.properties["eostrata:datetimes"]
        assert len(datetimes) == 2
        assert _DT.isoformat() in datetimes
        assert dt2022.isoformat() in datetimes
        # Bounding interval spans 2020–2022 but only two points exist
        assert item.common_metadata.start_datetime == _DT
        assert item.common_metadata.end_datetime == dt2022

    def test_duplicate_datetime_not_added_twice(self, tmp_path):
        """Registering the same datetime twice only stores one entry."""
        cat = _catalog_with_item(tmp_path)
        register_item(
            cat,
            collection_id="worldpop",
            item_id="worldpop_nga",
            bbox=_BBOX,
            datetime_=_DT,
            zarr_root=tmp_path / "zarr",
            zarr_group="worldpop/nga",
            variable="population",
        )
        item = cat.get_child("worldpop").get_item("worldpop_nga")
        assert item.properties["eostrata:datetimes"].count(_DT.isoformat()) == 1

    def test_existing_item_with_null_datetimes_uses_new_datetime(self, tmp_path):
        """When existing item has no start/end/datetime, new values are used (lines 156-157)."""
        cat_ = _make_catalog()
        coll = cat_.get_child("worldpop")
        # Create an item with all date fields set to None
        item = pystac.Item(
            id="worldpop_nga",
            geometry=None,
            bbox=list(_BBOX),
            datetime=None,
            properties={
                "start_datetime": None,
                "end_datetime": None,
                "datetime": None,
                "eostrata:variable": "population",
                "eostrata:zarr_group": "worldpop/nga",
                "eostrata:zarr_root": str(tmp_path / "zarr"),
            },
        )
        coll.add_item(item)

        register_item(
            cat_,
            collection_id="worldpop",
            item_id="worldpop_nga",
            bbox=_BBOX,
            datetime_=_DT,
            zarr_root=tmp_path / "zarr",
            zarr_group="worldpop/nga",
            variable="population",
        )
        updated = cat_.get_child("worldpop").get_item("worldpop_nga")
        assert updated.common_metadata.start_datetime == _DT

    def test_unknown_collection_raises(self, tmp_path):
        cat = _make_catalog()
        with pytest.raises(ValueError, match="not found"):
            register_item(
                cat,
                collection_id="nonexistent",
                item_id="x",
                bbox=_BBOX,
                datetime_=_DT,
                zarr_root=tmp_path,
                zarr_group="x/y",
                variable="v",
            )


class TestResolveItem:
    def test_found(self, tmp_path):
        cat = _catalog_with_item(tmp_path)
        save(cat, tmp_path / "catalog.json")
        result = resolve_item(tmp_path / "catalog.json", "worldpop", "worldpop_nga")
        assert result["zarr_group"] == "worldpop/nga"
        assert result["variable"] == "population"

    def test_missing_collection(self, tmp_path):
        save(_make_catalog(), tmp_path / "catalog.json")
        with pytest.raises(ValueError, match="Collection"):
            resolve_item(tmp_path / "catalog.json", "unknown", "x")

    def test_missing_item(self, tmp_path):
        save(_make_catalog(), tmp_path / "catalog.json")
        with pytest.raises(ValueError, match="Item"):
            resolve_item(tmp_path / "catalog.json", "worldpop", "missing_item")

    def test_item_without_zarr_asset_raises(self, tmp_path):
        """resolve_item raises ValueError when item has no zarr asset (line 250)."""
        cat_ = _make_catalog()
        coll = cat_.get_child("worldpop")
        item = pystac.Item(
            id="worldpop_nga",
            geometry=None,
            bbox=list(_BBOX),
            datetime=_DT,
            properties={
                "eostrata:variable": "population",
                "eostrata:zarr_group": "worldpop/nga",
            },
        )
        coll.add_item(item)
        save(cat_, tmp_path / "catalog.json")
        with pytest.raises(ValueError, match="no zarr asset"):
            resolve_item(tmp_path / "catalog.json", "worldpop", "worldpop_nga")


class TestPystacClient:
    def _client(self, tmp_path: Path) -> PystacClient:
        cat = _catalog_with_item(tmp_path)
        save(cat, tmp_path / "catalog.json")
        return PystacClient(catalog_path=str(tmp_path / "catalog.json"))

    def test_all_collections(self, tmp_path):
        client = self._client(tmp_path)
        result = client.all_collections()
        assert len(result["collections"]) == 3

    def test_get_collection_found(self, tmp_path):
        client = self._client(tmp_path)
        coll = client.get_collection("worldpop")
        assert coll["id"] == "worldpop"

    def test_get_collection_not_found(self, tmp_path):
        client = self._client(tmp_path)
        with pytest.raises(Exception, match="not found"):
            client.get_collection("nonexistent")

    def test_item_collection(self, tmp_path):
        client = self._client(tmp_path)
        ic = client.item_collection("worldpop")
        assert ic["type"] == "FeatureCollection"
        assert len(ic["features"]) == 1

    def test_get_item_found(self, tmp_path):
        client = self._client(tmp_path)
        item = client.get_item("worldpop_nga", "worldpop")
        assert item["id"] == "worldpop_nga"

    def test_get_item_not_found(self, tmp_path):
        client = self._client(tmp_path)
        with pytest.raises(Exception, match="not found"):
            client.get_item("missing", "worldpop")

    def test_get_search_returns_all_items(self, tmp_path):
        client = self._client(tmp_path)
        result = client.get_search()
        assert result["type"] == "FeatureCollection"
        assert len(result["features"]) == 1

    def test_post_search_same_as_get(self, tmp_path):
        client = self._client(tmp_path)
        assert client.post_search(None) == client.get_search()

    def test_item_collection_nonexistent_collection(self, tmp_path):
        """item_collection raises NotFoundError for unknown collection (line 305)."""
        client = self._client(tmp_path)
        with pytest.raises(Exception, match="not found"):
            client.item_collection("nonexistent")

    def test_get_item_nonexistent_collection(self, tmp_path):
        """get_item raises NotFoundError when collection doesn't exist (line 313)."""
        client = self._client(tmp_path)
        with pytest.raises(Exception, match="not found"):
            client.get_item("worldpop_nga", "nonexistent")
