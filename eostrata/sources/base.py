"""BaseSource abstract class and source registry."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import xarray as xr
from tqdm import tqdm

from eostrata.constants import PROP_VARIABLE

logger = logging.getLogger(__name__)

# ── Shared download helper ─────────────────────────────────────────────────────

_DOWNLOAD_RETRIES = 2
_RETRY_DELAYS = (5, 15)  # seconds between attempts 1→2 and 2→3


def _stream_download(url: str, dest: Path) -> Path:
    """Stream *url* to *dest*, skipping if the file already exists.

    Retries up to ``_DOWNLOAD_RETRIES`` times on transient network errors
    (connection reset, timeout, etc.), removing any partial file before each
    retry so the next attempt starts clean.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("Already downloaded: %s", dest.name)
        return dest

    last_exc: Exception | None = None
    for attempt in range(1, _DOWNLOAD_RETRIES + 2):
        try:
            logger.info("Downloading %s (attempt %d)", url, attempt)
            with httpx.stream("GET", url, follow_redirects=True, timeout=None) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                with (
                    open(dest, "wb") as fh,
                    tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
                ):
                    for chunk in resp.iter_bytes(chunk_size=1 << 20):
                        fh.write(chunk)
                        bar.update(len(chunk))
            logger.info("Saved to %s", dest)
            return dest
        except httpx.TransportError as exc:
            last_exc = exc
            dest.unlink(missing_ok=True)  # remove partial file before retry
            if attempt <= _DOWNLOAD_RETRIES:
                delay = _RETRY_DELAYS[attempt - 1]
                logger.warning(
                    "Download failed (attempt %d/%d): %s — retrying in %ds",
                    attempt,
                    _DOWNLOAD_RETRIES + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error("Download failed after %d attempts: %s", _DOWNLOAD_RETRIES + 1, exc)

    raise last_exc  # type: ignore[misc]


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

    #: Human-readable title for the STAC collection.
    collection_title: str

    #: Human-readable description for the STAC collection.
    collection_description: str

    #: First component of the Zarr group path (may differ from ``id``, e.g. ``"era5"`` for CDS).
    zarr_prefix: str

    #: Primary data variable name stored in the Zarr group.
    VARIABLE: str

    #: Temporal resolution: ``"monthly"``, ``"annual"``, etc.
    temporal_resolution: str

    #: Typical number of days between a period ending and data availability.
    default_lag_days: int

    #: Whether 404 HTTP errors should be silently skipped (True for WorldPop, CHIRPS).
    skip_404: bool = False

    #: Form fields shown in the ingest UI for this source.
    ui_fields: list[str] = []

    #: All supported variable short names. Empty means only ``VARIABLE`` is supported.
    VARIABLES: list[str] = []

    #: Human-readable descriptions for each variable, keyed by short name.
    VARIABLE_DESCRIPTIONS: dict[str, str] = {}

    @classmethod
    def catalog_meta(cls, dataset_name: str) -> dict:
        """Return catalog registration metadata for a Zarr group with this source's prefix.

        ``dataset_name`` is the second path component of the Zarr group
        (e.g. ``"nga"`` from ``worldpop/nga``, ``"t2m"`` from ``era5/t2m``).

        Returns a dict with keys ``item_id``, ``variable``, and ``extra``.
        Override in subclasses when the default derivation doesn't apply.
        """
        return {
            "item_id": f"{cls.zarr_prefix}_{dataset_name}",
            "variable": cls.VARIABLE,
            "extra": {PROP_VARIABLE: cls.VARIABLE},
        }

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

    def extract_item_bbox(self, ds) -> tuple[float, float, float, float]:
        """Extract (west, south, east, north) bounding box from a dataset."""
        return (float(ds.x.min()), float(ds.y.min()), float(ds.x.max()), float(ds.y.max()))

    @classmethod
    @abstractmethod
    def iter_periods(cls, **source_params) -> Iterator[tuple[str, dict]]:
        """Yield (label, period_kwargs) for each period to ingest."""

    @abstractmethod
    def stac_registrations(self, ds, period_kwargs: dict) -> list[dict]:
        """Return list of STAC registration dicts for one downloaded period.

        Each dict has keys: item_id, datetime_, variable, extra_properties.
        """
