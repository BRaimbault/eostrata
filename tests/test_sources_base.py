"""Tests for the source registry and BaseSource interface."""

from __future__ import annotations

import pytest

from eostrata.sources.base import _REGISTRY, all_sources, get_source, register_source


class TestRegistry:
    def test_get_known_source(self):
        src = get_source("worldpop")
        assert src.id == "worldpop"

    def test_get_chirps(self):
        src = get_source("chirps")
        assert src.id == "chirps"

    def test_get_cds(self):
        src = get_source("cds")
        assert src.id == "cds"

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unknown source"):
            get_source("nonexistent_xyz")

    def test_all_sources_returns_list(self):
        sources = all_sources()
        assert isinstance(sources, list)
        assert len(sources) >= 1

    def test_all_sources_ids(self):
        ids = {s.id for s in all_sources()}
        assert {"worldpop", "chirps", "cds", "cgls"}.issubset(ids)

    def test_register_source_decorator(self):
        """A dynamically-registered source should appear in the registry."""
        from datetime import datetime

        from eostrata.sources.base import BaseSource

        @register_source
        class _TestSource(BaseSource):
            id = "_test_register"
            collection_id = "_test"
            temporal_resolution = "annual"
            default_lag_days = 0

            def download(self, raw_dir, bbox, **kw):
                return []

            def to_zarr(self, path, zarr_root, bbox, **kw): ...
            def stac_item_id(self, **kw):
                return "_test_item"

            def stac_properties(self, **kw):
                return {}

            def latest_available(self):
                return datetime(2020, 1, 1)

        assert get_source("_test_register") is _TestSource

        # Cleanup so as not to pollute other tests
        del _REGISTRY["_test_register"]
