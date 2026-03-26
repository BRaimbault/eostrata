"""BaseSource abstract class and source registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import xarray as xr

# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type[BaseSource]] = {}


def register_source(cls: type[BaseSource]) -> type[BaseSource]:
    """Class decorator — adds the source to the global registry."""
    _REGISTRY[cls.id] = cls
    return cls


def get_source(source_id: str) -> type[BaseSource]:
    """Return a registered source class by id."""
    if source_id not in _REGISTRY:
        available = list(_REGISTRY)
        raise ValueError(f"Unknown source '{source_id}'. Available: {available}")
    return _REGISTRY[source_id]


def all_sources() -> list[type[BaseSource]]:
    """Return all registered source classes."""
    return list(_REGISTRY.values())


# ── Abstract base ─────────────────────────────────────────────────────────────


class BaseSource(ABC):
    """
    Base class for all eostrata data sources.

    Subclass and decorate with @register_source to make a source
    available to the CLI, scheduler, and store.
    """

    #: Unique source identifier, e.g. ``"worldpop"``.
    id: str

    #: STAC collection this source belongs to.
    collection_id: str

    #: Temporal resolution: ``"monthly"``, ``"annual"``, etc.
    temporal_resolution: str

    #: Typical number of days between a period ending and data availability.
    default_lag_days: int

    @abstractmethod
    def download(
        self,
        raw_dir: Path,
        bbox: tuple[float, float, float, float],
        **params: Any,
    ) -> list[Path]:
        """
        Download source data to *raw_dir*, clipped to *bbox*.

        Returns a list of downloaded local file paths.
        """

    @abstractmethod
    def to_zarr(
        self,
        path: Path,
        zarr_root: Path,
        bbox: tuple[float, float, float, float],
    ) -> xr.Dataset:
        """Convert a downloaded file to a Zarr group. Returns the dataset."""

    @abstractmethod
    def stac_item_id(self, **params: Any) -> str:
        """Return the STAC item id for the given params."""

    @abstractmethod
    def stac_properties(self, **params: Any) -> dict:
        """Return extra STAC item properties for the given params."""

    @abstractmethod
    def latest_available(self) -> datetime:
        """Return the most recent period for which data is available."""
