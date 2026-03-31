# Adding a New Data Source

Adding a new earth observation source to eostrata requires **a single Python file** in
`eostrata/sources/`. No other files need to be changed — the source is auto-discovered and
becomes available in the CLI, ingest API, map UI, tile server, STAC catalogue, and OGC
Processes interface automatically.

---

## Quick start

1. Copy `eostrata/sources/_template.py` to `eostrata/sources/<your_source_id>.py`.
2. Fill in every `TODO` section.
3. Run the tests: `uv run pytest tests/ -x -q`.

That's it.

---

## What auto-discovery gives you

When your source file is placed in `eostrata/sources/` and decorated with
`@register_source`, the framework automatically:

| Feature | What happens |
|---|---|
| **CLI** | `eostrata download <source_id> [options]` works immediately |
| **Ingest API** | `POST /processes/ingest/execution` accepts `"source": "<source_id>"` |
| **Map UI** | `/map` ingest tab shows a new entry with the correct form fields |
| **STAC catalogue** | A new collection appears under `GET /stac/collections` |
| **Tile server** | `GET /tiles/…` serves data once it is ingested |
| **OGC Processes** | `GET /processes/ingest` lists the source in its description |

---

## Required class attributes

| Attribute | Type | Description |
|---|---|---|
| `id` | `str` | Unique snake_case identifier, e.g. `"modis_lst"` |
| `collection_id` | `str` | STAC collection id (usually same as `id`) |
| `collection_title` | `str` | Human-readable collection name |
| `collection_description` | `str` | One-sentence description |
| `zarr_prefix` | `str` | First path component in the Zarr store (may differ from `id`) |
| `VARIABLE` | `str` | Short variable name stored in Zarr (e.g. `"lst"`) |
| `temporal_resolution` | `str` | `"monthly"`, `"annual"`, `"dekadal"`, etc. |
| `default_lag_days` | `int` | Typical days between period end and data availability |
| `ui_fields` | `list[str]` | Form fields the ingest UI should show for this source |
| `skip_404` | `bool` | Set `True` if HTTP 404 means "data not yet available" (not an error) |

### Recognised `ui_fields` values

| Value | UI element shown |
|---|---|
| `"iso3"` | ISO3 country-code text input |
| `"variable"` | ERA5 variable `<select>` |
| `"years"` | Years free-text input |
| `"months"` | Months free-text input |
| `"days"` | Days free-text input (daily sources) |
| `"dekads"` | Dekads free-text input |

---

## Required methods

### `download(self, raw_dir, bbox, **params) -> list[Path]`

Download the raw data for one period.  `params` contains the keys yielded by
`iter_periods` for this period (e.g. `year`, `month`, `iso3`).  Return a list of
downloaded file paths; the first element is passed to `to_zarr`.

### `to_zarr(self, path, zarr_root, bbox, **params) -> xr.Dataset`

Convert the downloaded file to a Zarr group and return the resulting dataset.  Use
`eostrata.store.geotiff_to_zarr` for GeoTIFF sources or write NetCDF directly for
others.

### `zarr_group(self, **params) -> str`

Return the Zarr group path for the given params (e.g. `"worldpop/nga"`,
`"chirps/global"`).

### `stac_item_id(self, **params) -> str`

Return the STAC item id for the given params.

### `stac_properties(self, **params) -> dict`

Return extra STAC item properties stored alongside the standard fields.

### `latest_available(self) -> datetime`

Return the most recent datetime for which data is reliably available.  Used as the
default when the user does not supply a specific year/month/dekad.

### `iter_periods(cls, **source_params) -> Iterator[tuple[str, dict]]`  *(classmethod)*

Yield `(label, period_kwargs)` for every period that should be downloaded.
`label` is a human-readable string used in log messages and error reporting.
`period_kwargs` is passed directly to `download`, `to_zarr`, `stac_item_id`, and
`stac_properties`.

### `stac_registrations(self, ds, period_kwargs) -> list[dict]`

Return a list of STAC items to register after one period is written to Zarr.  Each
dict must have:

```python
{
    "item_id":        str,       # STAC item id
    "datetime_":      datetime,  # UTC datetime for this period
    "variable":       str,       # variable name stored in Zarr
    "extra_properties": dict,    # extra STAC properties
}
```

Most sources return a list with a single item.  CDS/ERA5 is the exception: one
download covers a whole year but produces one STAC item *per month*.

---

## Optional overrides

### `catalog_meta(cls, dataset_name) -> dict`

Called by `rebuild-catalog` to infer STAC metadata from a Zarr group path.
`dataset_name` is the second component of the group path (e.g. `"nga"` from
`worldpop/nga`).  Returns `{item_id, variable, extra}`.  Override only if the default
`f"{zarr_prefix}_{dataset_name}"` derivation does not apply.

### `extract_item_bbox(self, ds) -> tuple[float, float, float, float]`

Return `(west, south, east, north)` from the written dataset.  The default reads
`ds.x` / `ds.y` coordinates.  Override if your dataset uses `longitude` / `latitude`
(as ERA5 does before coordinate renaming).

---

## Example: minimal monthly GeoTIFF source

```python
"""My new source."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from eostrata.sources.base import BaseSource, _stream_download, register_source
from eostrata.store import geotiff_to_zarr

logger = logging.getLogger(__name__)

_BASE_URL = "https://example.org/data"


@register_source
class MySource(BaseSource):
    id = "mysource"
    collection_id = "mysource"
    collection_title = "My dataset"
    collection_description = "A short description of this dataset."
    zarr_prefix = "mysource"
    VARIABLE = "myvar"
    temporal_resolution = "monthly"
    default_lag_days = 30
    skip_404 = False
    ui_fields = ["years", "months"]

    def download(self, raw_dir: Path, bbox, *, year: int, month: int, **_) -> list[Path]:
        url = f"{_BASE_URL}/{year}/{month:02d}.tif"
        dest = Path(raw_dir) / "mysource" / f"mysource_{year}_{month:02d}.tif"
        return [_stream_download(url, dest)]

    def to_zarr(self, path: Path, zarr_root: Path, bbox, *, year: int, month: int, **_: Any):
        time_coord = np.datetime64(f"{year}-{month:02d}-01", "ns")
        return geotiff_to_zarr(
            path, zarr_root, self.zarr_group(),
            bbox=bbox, time_coord=time_coord, variable_name=self.VARIABLE,
        )

    def zarr_group(self, **_) -> str:
        return "mysource/global"

    def stac_item_id(self, **_) -> str:
        return "mysource_global"

    def stac_properties(self, *, year: int, month: int, **_) -> dict:
        return {"eostrata:variable": self.VARIABLE}

    def latest_available(self) -> datetime:
        now = datetime.now(tz=UTC)
        month = now.month - 1 or 12
        year = now.year if now.month > 1 else now.year - 1
        return datetime(year, month, 1, tzinfo=UTC)

    @classmethod
    def iter_periods(cls, *, years: list[int], months: list[int], **_) -> Iterator[tuple[str, dict]]:
        for year in years:
            for month in months:
                yield (f"{year}-{month:02d}", {"year": year, "month": month})

    def stac_registrations(self, ds, period_kwargs: dict) -> list[dict]:
        year, month = period_kwargs["year"], period_kwargs["month"]
        return [{
            "item_id": self.stac_item_id(),
            "datetime_": datetime(year, month, 1, tzinfo=UTC),
            "variable": self.VARIABLE,
            "extra_properties": self.stac_properties(**period_kwargs),
        }]
```

---

## Testing your source

Add a test file `tests/test_mysource.py`.  At minimum, test:

- `download()` with a mocked HTTP response
- `to_zarr()` with a real (tiny) GeoTIFF written to `tmp_path`
- `iter_periods()` yields the expected labels and kwargs
- `stac_registrations()` returns the correct datetime and item id
- `latest_available()` returns a plausible datetime

See `tests/test_sentinel_ndvi.py` as a reference.

---

## Checklist

- [ ] File placed in `eostrata/sources/<source_id>.py`
- [ ] Class decorated with `@register_source`
- [ ] All required attributes defined
- [ ] All required methods implemented
- [ ] `iter_periods` yields correct `(label, period_kwargs)` tuples
- [ ] `stac_registrations` returns correct registration dicts
- [ ] `ui_fields` lists the correct form fields
- [ ] Tests written in `tests/test_<source_id>.py`
- [ ] `uv run pytest tests/ -x -q` passes
