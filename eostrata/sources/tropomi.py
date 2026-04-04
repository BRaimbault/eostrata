"""Sentinel-5P TROPOMI air quality source via Copernicus Data Space (CDSE).

Downloads Sentinel-5P OFFLINE Level-2 products from the Copernicus Data Space
Ecosystem (CDSE) OData API, grids the swath data to a regular lat/lon grid
within the configured bbox, and writes daily means to the Zarr store.

Requires:
  - A free CDSE account at https://dataspace.copernicus.eu
  - Set EOSTRATA_CDSE_USER and EOSTRATA_CDSE_PASSWORD environment variables
    (or add them to your .env file).

Supported variables (``--variable`` / ``variable`` param):
  no2     — nitrogendioxide_tropospheric_column  (mol/m², tropospheric column)
  co      — carbonmonoxide_total_column          (mol/m², total column)
  o3      — ozone_total_column                   (mol/m², total column)
  so2     — sulfurdioxide_total_vertical_column  (mol/m², total column)
  ch4     — methane_mixing_ratio_bias_corrected  (ppb, dry-air column avg)
  hcho    — formaldehyde_tropospheric_vertical_column  (mol/m², tropospheric)
  aer_ai  — aerosol_index_354_388               (dimensionless, absorbing aerosol)

The default variable is ``no2`` (tropospheric NO₂ column).

Gridding:
  Swath pixels are binned to a regular 0.1° lat/lon grid within the bbox.
  Quality filter: qa_value ≥ 0.75 (ESA recommended threshold).

Dataset: Sentinel-5P OFFLINE L2 products
Resolution: native ~3.5×5.5 km, gridded to 0.1°
Coverage: Global, 2018-present (S5P launched Oct 2017, data from May 2018)
"""

from __future__ import annotations

import logging
import zipfile
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import h5py
import httpx
import numpy as np

from eostrata.constants import PROP_RESOLUTION, PROP_SOURCE, PROP_VARIABLE
from eostrata.sources.base import BaseSource, register_source
from eostrata.store import _group_lock, geotiff_to_zarr  # noqa: F401 — imported for type hints only

logger = logging.getLogger(__name__)

# ── CDSE API constants ─────────────────────────────────────────────────────────

_CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)
_CDSE_SEARCH_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
_CDSE_DOWNLOAD_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"

# Quality value threshold (ESA recommended: 0.75 for most variables)
_QA_THRESHOLD = 0.75

# Default output grid resolution in degrees
_GRID_RESOLUTION = 0.1

# Maps short variable names to (S5P product type, HDF5 variable path)
_VARIABLE_MAP: dict[str, tuple[str, str]] = {
    "no2": ("L2__NO2___", "PRODUCT/nitrogendioxide_tropospheric_column"),
    "co": ("L2__CO____", "PRODUCT/carbonmonoxide_total_column"),
    "o3": ("L2__O3____", "PRODUCT/ozone_total_column"),
    "so2": ("L2__SO2___", "PRODUCT/sulfurdioxide_total_vertical_column"),
    "ch4": ("L2__CH4___", "PRODUCT/methane_mixing_ratio_bias_corrected"),
    "hcho": ("L2__HCHO__", "PRODUCT/formaldehyde_tropospheric_vertical_column"),
    "aer_ai": ("L2__AER_AI", "PRODUCT/aerosol_index_354_388"),
}

# Human-readable units for STAC metadata
_VARIABLE_UNITS: dict[str, str] = {
    "no2": "mol/m2",
    "co": "mol/m2",
    "o3": "mol/m2",
    "so2": "mol/m2",
    "ch4": "ppb",
    "hcho": "mol/m2",
    "aer_ai": "dimensionless",
}


# ── CDSE authentication ────────────────────────────────────────────────────────


def _get_cdse_token(user: str, password: str) -> str:
    """Obtain an OAuth2 Bearer token from the CDSE identity service."""
    resp = httpx.post(
        _CDSE_TOKEN_URL,
        data={
            "grant_type": "password",
            "username": user,
            "password": password,
            "client_id": "cdse-public",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


# ── CDSE product search ────────────────────────────────────────────────────────


def _build_bbox_wkt(bbox: tuple[float, float, float, float]) -> str:
    """Return a WKT POLYGON for *bbox* (west, south, east, north)."""
    w, s, e, n = bbox
    return f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"


def _search_products(
    product_type: str,
    day: date,
    bbox: tuple[float, float, float, float],
) -> list[dict]:
    """Return a list of CDSE product metadata dicts that intersect *bbox* on *day*."""
    start = f"{day.isoformat()}T00:00:00.000Z"
    end = f"{day.isoformat()}T23:59:59.999Z"
    wkt = _build_bbox_wkt(bbox)

    odata_filter = (
        f"Collection/Name eq 'SENTINEL-5P' and "
        f"Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq 'productType' and "
        f"att/OData.CSC.StringAttribute/Value eq '{product_type}') and "
        f"ContentDate/Start ge {start} and "
        f"ContentDate/Start le {end} and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')"
    )

    products: list[dict] = []
    url: str | None = (
        f"{_CDSE_SEARCH_URL}?$filter={odata_filter}&$top=100&$orderby=ContentDate/Start"
    )

    while url:
        resp = httpx.get(url, timeout=60)
        resp.raise_for_status()
        body = resp.json()
        products.extend(body.get("value", []))
        url = body.get("@odata.nextLink")

    logger.info(
        "Found %d TROPOMI %s products for %s intersecting bbox %s",
        len(products),
        product_type,
        day,
        bbox,
    )
    return products


# ── CDSE product download ──────────────────────────────────────────────────────


def _download_product(product_id: str, dest: Path, token: str) -> Path:
    """Download a CDSE product to *dest*, skipping if already present.

    The CDSE zipper returns either a raw NetCDF4 file or a ZIP archive.
    Both cases are handled: if a ZIP is received it is extracted in-place
    and the first .nc member is returned.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("Already downloaded: %s", dest.name)
        return dest

    url = f"{_CDSE_DOWNLOAD_URL}('{product_id}')/$value"
    logger.info("Downloading product %s", product_id)

    tmp = dest.with_suffix(".tmp")
    try:
        with httpx.stream(
            "GET",
            url,
            headers={"Authorization": f"Bearer {token}"},
            follow_redirects=True,
            timeout=None,
        ) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1 << 20):
                    fh.write(chunk)

        # Detect ZIP vs raw NetCDF4 (HDF5 magic bytes: \x89HDF)
        with open(tmp, "rb") as fh:
            magic = fh.read(4)

        if magic[:2] == b"PK":  # ZIP archive
            with zipfile.ZipFile(tmp) as zf:
                nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
                if not nc_names:
                    raise RuntimeError(f"No .nc file found in product ZIP for {product_id}")
                zf.extract(nc_names[0], dest.parent)
                extracted = dest.parent / nc_names[0]
                extracted.rename(dest)
            tmp.unlink(missing_ok=True)
        else:
            tmp.rename(dest)

    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    logger.info("Saved product to %s", dest)
    return dest


# ── Swath processing and gridding ─────────────────────────────────────────────


def _read_swath(nc_path: Path, var_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read (latitude, longitude, values) from a TROPOMI L2 NetCDF4/HDF5 file.

    Applies the ESA-recommended quality filter (qa_value ≥ 0.75) and
    returns flat 1-D arrays of valid pixel coordinates and values.
    """
    with h5py.File(nc_path, "r") as f:
        lat = f["PRODUCT/latitude"][0].astype("float32")  # (scanline, pixel)
        lon = f["PRODUCT/longitude"][0].astype("float32")
        qa = f["PRODUCT/qa_value"][0].astype("float32")

        # Retrieve the data variable; handle missing datasets gracefully
        if var_path not in f:
            raise KeyError(f"Variable '{var_path}' not found in {nc_path.name}")
        data = f[var_path][0].astype("float64")

        # Apply fill-value mask from HDF5 attributes
        ds = f[var_path]
        fill = ds.attrs.get("_FillValue", None)
        if fill is not None:
            data = np.where(data == fill, np.nan, data)

    # Quality filter
    qa_mask = qa >= _QA_THRESHOLD
    flat_mask = qa_mask.ravel()
    lat_f = lat.ravel()[flat_mask]
    lon_f = lon.ravel()[flat_mask]
    val_f = data.ravel()[flat_mask]

    # Remove NaN values
    valid = np.isfinite(val_f)
    return lat_f[valid], lon_f[valid], val_f[valid]


def _grid_swath_data(
    lat: np.ndarray,
    lon: np.ndarray,
    values: np.ndarray,
    bbox: tuple[float, float, float, float],
    resolution: float = _GRID_RESOLUTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin swath pixels to a regular lat/lon grid using simple averaging.

    Returns (grid_2d, lats_1d, lons_1d) where grid_2d has shape
    (len(lats), len(lons)) and NaN where no valid pixel was binned.
    """
    west, south, east, north = bbox

    # Build output grid axes (cell centres)
    lons = np.arange(west + resolution / 2, east, resolution, dtype="float64")
    lats = np.arange(south + resolution / 2, north, resolution, dtype="float64")
    ny, nx = len(lats), len(lons)

    if ny == 0 or nx == 0:
        return np.full((1, 1), np.nan, dtype="float32"), lats, lons

    # Filter pixels to bbox — inclusive on all sides so edge pixels are not lost.
    # np.clip below ensures computed grid indices stay in [0, n-1].
    in_bbox = (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)
    lat_b, lon_b, val_b = lat[in_bbox], lon[in_bbox], values[in_bbox]

    if lat_b.size == 0:
        return np.full((ny, nx), np.nan, dtype="float32"), lats, lons

    # Map pixel centres to nearest grid cell indices
    xi = np.floor((lon_b - west) / resolution).astype(np.intp)
    yi = np.floor((lat_b - south) / resolution).astype(np.intp)
    xi = np.clip(xi, 0, nx - 1)
    yi = np.clip(yi, 0, ny - 1)

    # Accumulate sum and count for averaging
    grid_sum = np.zeros((ny, nx), dtype="float64")
    count = np.zeros((ny, nx), dtype=np.int32)
    np.add.at(grid_sum, (yi, xi), val_b)
    np.add.at(count, (yi, xi), 1)

    grid = np.where(count > 0, grid_sum / count, np.nan).astype("float32")
    return grid, lats, lons


def _write_daily_grid(
    grid: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    zarr_root: Path,
    zarr_group: str,
    time_coord: np.datetime64,
    variable_name: str,
) -> Any:  # xr.Dataset
    """Write a 2-D daily grid to the Zarr store (append or create)."""
    import xarray as xr

    cy, cx = 512, 512

    coords = {
        "time": np.array([time_coord], dtype="datetime64[ns]"),
        "y": lats.astype("float64"),
        "x": lons.astype("float64"),
    }
    da = xr.DataArray(
        grid[np.newaxis, ...],  # add time axis
        dims=("time", "y", "x"),
        coords=coords,
        name=variable_name,
    )
    da.attrs["grid_mapping"] = "crs"
    da.attrs["long_name"] = zarr_group

    from rasterio.crs import CRS

    ds = da.to_dataset()
    ds["crs"] = xr.DataArray(
        np.int32(0),
        attrs={
            "grid_mapping_name": "latitude_longitude",
            "crs_wkt": CRS.from_epsg(4326).to_wkt(),
            "spatial_ref": CRS.from_epsg(4326).to_wkt(),
        },
    )
    ds.attrs["Conventions"] = "CF-1.8"

    zarr_root = Path(zarr_root)
    zarr_root.mkdir(parents=True, exist_ok=True)
    store_path = str(zarr_root)

    encoding = {
        variable_name: {
            "chunks": (1, cy, cx),
            "_FillValue": float("nan"),
        },
    }

    with _group_lock(zarr_root, zarr_group):
        group_exists = (zarr_root / zarr_group).exists()

        if group_exists:
            try:
                existing = xr.open_zarr(store_path, group=zarr_group, consolidated=False)
                try:
                    already_present = "time" in existing and time_coord in existing["time"].values
                finally:
                    existing.close()
                if already_present:
                    logger.info(
                        "Timestamp %s already exists in '%s' — skipping", time_coord, zarr_group
                    )
                    return ds
            except (OSError, KeyError, ValueError):
                logger.debug(
                    "Could not read existing Zarr group '%s', proceeding with append", zarr_group
                )
            logger.info("Appending TROPOMI '%s' daily grid", zarr_group)
            ds.to_zarr(
                store_path,
                group=zarr_group,
                mode="a",
                append_dim="time",
                consolidated=True,
            )
        else:
            logger.info("Writing new TROPOMI Zarr group '%s'", zarr_group)
            ds.to_zarr(
                store_path,
                group=zarr_group,
                mode="w",
                encoding=encoding,
                consolidated=True,
            )

    return ds


# ── Source class ───────────────────────────────────────────────────────────────


@register_source
class TROPOMISource(BaseSource):
    """Sentinel-5P TROPOMI daily L2 air quality columns from Copernicus Data Space."""

    id = "tropomi"
    collection_id = "tropomi"
    collection_title = "Sentinel-5P TROPOMI air quality"
    collection_description = (
        "Daily surface-level air quality columns from the Sentinel-5P "
        "TROPOMI instrument via the Copernicus Data Space"
    )
    zarr_prefix = "tropomi"
    temporal_resolution = "daily"
    default_lag_days = 3  # OFFLINE products available ~1-3 days after sensing
    VARIABLE = "no2"  # default variable short name
    VARIABLES = ["no2", "co", "o3", "so2", "ch4", "hcho", "aer_ai"]
    VARIABLE_DESCRIPTIONS = {
        "no2": "tropospheric NO₂ column (mol/m²)",
        "co": "total CO column (mol/m²)",
        "o3": "total O₃ column (mol/m²)",
        "so2": "total SO₂ column (mol/m²)",
        "ch4": "CH₄ mixing ratio (ppb)",
        "hcho": "tropospheric HCHO column (mol/m²)",
        "aer_ai": "aerosol index (dimensionless)",
    }
    skip_404 = True  # no data for some days/regions is expected
    ui_fields = ["variable", "years", "months", "days"]

    @classmethod
    def catalog_meta(cls, dataset_name: str) -> dict:
        # dataset_name IS the variable (e.g. "no2" from "tropomi/no2")
        return {
            "item_id": f"tropomi_{dataset_name}",
            "variable": dataset_name,
            "extra": {PROP_VARIABLE: dataset_name},
        }

    def download(
        self,
        raw_dir: Path,
        bbox: tuple[float, float, float, float],
        *,
        variable: str = "no2",
        year: int,
        month: int,
        day: int,
        **_: Any,
    ) -> list[Path]:
        """Download all TROPOMI L2 swath files for *variable* on *year*-*month*-*day*.

        Parameters
        ----------
        variable:
            Short variable name — one of: no2, co, o3, so2, ch4, hcho, aer_ai.
        year, month, day:
            UTC sensing date.

        Returns a list of downloaded .nc file paths (one per orbit).
        """
        if variable not in _VARIABLE_MAP:
            raise ValueError(
                f"Unknown TROPOMI variable '{variable}'. Available: {list(_VARIABLE_MAP)}"
            )

        from eostrata.config import settings

        user = settings.cdse_user
        password = settings.cdse_password
        if not user or not password:
            raise RuntimeError(
                "CDSE credentials are required for the TROPOMI source.\n"
                "Set EOSTRATA_CDSE_USER and EOSTRATA_CDSE_PASSWORD in your .env file\n"
                "or as environment variables.  Register for free at:\n"
                "  https://dataspace.copernicus.eu"
            )

        product_type, _ = _VARIABLE_MAP[variable]
        sensing_date = date(year, month, day)

        products = _search_products(product_type, sensing_date, bbox)
        if not products:
            logger.info(
                "No TROPOMI %s products found for %s in bbox %s — skipping",
                variable,
                sensing_date,
                bbox,
            )
            return []

        token = _get_cdse_token(user, password)
        dest_dir = Path(raw_dir) / "tropomi" / f"{year:04d}" / f"{month:02d}" / f"{day:02d}"
        dest_dir.mkdir(parents=True, exist_ok=True)

        downloaded: list[Path] = []
        for product in products:
            pid = product["Id"]
            name = product.get("Name", pid)
            dest = dest_dir / f"{name}.nc"
            try:
                path = _download_product(pid, dest, token)
                downloaded.append(path)
            except Exception as exc:
                logger.warning("Failed to download product %s: %s — skipping", pid, exc)

        return downloaded

    def to_zarr(
        self,
        path: Path,
        zarr_root: Path,
        bbox: tuple[float, float, float, float],
        *,
        variable: str = "no2",
        year: int,
        month: int,
        day: int,
        **_: Any,
    ) -> Any:
        """Process a single TROPOMI swath file and merge into the daily Zarr group.

        Note: ``download()`` may return multiple swath files for a single day.
        The ingestion pipeline calls ``to_zarr`` once per file; each call
        contributes pixels to the daily average.  Because the Zarr store is
        append-only, subsequent calls for the same day are detected by the
        duplicate-timestamp guard and skipped.  To merge all swaths, the
        daily grid is computed afresh from *all* available swath files in the
        raw directory before writing.

        **First-write-wins limitation**: the daily grid is computed from
        whichever swath files are present on disk at the time of the *first*
        successful write for that day.  Any orbit files downloaded after that
        point are silently skipped by the duplicate-timestamp guard and will
        not be incorporated.  To regenerate the grid with all orbits, delete
        the existing Zarr timestep and re-run ``to_zarr`` on any swath file
        for that day (or use ``eostrata rebuild-catalog`` after deleting the
        timestep).
        """
        if variable not in _VARIABLE_MAP:
            raise ValueError(f"Unknown TROPOMI variable '{variable}'")

        _, var_path = _VARIABLE_MAP[variable]
        time_coord = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}", "ns")

        # Collect all swath files for this day (already downloaded siblings)
        swath_dir = path.parent
        swath_files = sorted(swath_dir.glob("*.nc"))

        all_lat: list[np.ndarray] = []
        all_lon: list[np.ndarray] = []
        all_val: list[np.ndarray] = []

        for swath_path in swath_files:
            try:
                lat_s, lon_s, val_s = _read_swath(swath_path, var_path)
                all_lat.append(lat_s)
                all_lon.append(lon_s)
                all_val.append(val_s)
            except Exception as exc:
                logger.warning("Could not read swath %s: %s — skipping", swath_path.name, exc)

        if not all_lat:
            logger.info(
                "No valid pixels in swaths for %s %s — writing empty grid", variable, time_coord
            )
            lat_empty: np.ndarray = np.array([], dtype="float32")
            lon_empty: np.ndarray = np.array([], dtype="float32")
            val_empty: np.ndarray = np.array([], dtype="float64")
            grid, lats, lons = _grid_swath_data(lat_empty, lon_empty, val_empty, bbox)
        else:
            lat_all = np.concatenate(all_lat)
            lon_all = np.concatenate(all_lon)
            val_all = np.concatenate(all_val)
            grid, lats, lons = _grid_swath_data(lat_all, lon_all, val_all, bbox)

        return _write_daily_grid(
            grid,
            lats,
            lons,
            zarr_root,
            self.zarr_group(variable=variable),
            time_coord,
            variable_name=variable,
        )

    def zarr_group(self, *, variable: str = "no2", **_: Any) -> str:
        """One Zarr group per TROPOMI variable — all days as timesteps."""
        return f"tropomi/{variable}"

    def stac_item_id(self, *, variable: str = "no2", **_: Any) -> str:
        """One STAC item per TROPOMI variable (all days as timesteps)."""
        return f"tropomi_{variable}"

    def stac_properties(
        self, *, variable: str = "no2", year: int, month: int, day: int, **_: Any
    ) -> dict:
        product_type, var_path = _VARIABLE_MAP.get(variable, ("", ""))
        return {
            PROP_VARIABLE: variable,
            "eostrata:tropomi_product": product_type,
            "eostrata:hdf5_path": var_path,
            PROP_RESOLUTION: "0.1deg",
            "eostrata:units": _VARIABLE_UNITS.get(variable, ""),
            "eostrata:qa_threshold": _QA_THRESHOLD,
            PROP_SOURCE: "Sentinel-5P TROPOMI OFFLINE L2",
        }

    def latest_available(self) -> datetime:
        """TROPOMI OFFLINE products lag ~1-3 days; return 3 days before today."""
        dt = datetime.now(tz=UTC) - timedelta(days=3)
        return datetime(dt.year, dt.month, dt.day, tzinfo=UTC)

    @classmethod
    def iter_periods(
        cls,
        *,
        variable: str = "no2",
        years: list[int],
        months: list[int],
        days: list[int],
        **_,
    ) -> Iterator[tuple[str, dict]]:
        for year in years:
            for month in months:
                for day in days:
                    yield (
                        f"{variable}/{year}-{month:02d}-{day:02d}",
                        {"variable": variable, "year": year, "month": month, "day": day},
                    )

    def stac_registrations(self, ds, period_kwargs: dict) -> list[dict]:
        variable = period_kwargs["variable"]
        year = period_kwargs["year"]
        month = period_kwargs["month"]
        day = period_kwargs["day"]
        return [
            {
                "item_id": self.stac_item_id(variable=variable),
                "datetime_": datetime(year, month, day, tzinfo=UTC),
                "variable": variable,
                "extra_properties": self.stac_properties(**period_kwargs),
            }
        ]

    def extract_item_bbox(self, ds) -> tuple[float, float, float, float]:
        return (
            float(ds["x"].min()),
            float(ds["y"].min()),
            float(ds["x"].max()),
            float(ds["y"].max()),
        )
