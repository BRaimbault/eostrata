"""Zarr store cache management — quota checks and LRU eviction.

The eviction policy works at the **timestamp** granularity within a Zarr group
(one group = one dataset, e.g. ``worldpop/nga`` or ``chirps/global``).  When
the store exceeds the configured quota, the least-recently-accessed timestamps
are removed (one by one, oldest first) until the store fits within the quota.

Last-access tracking
--------------------
File-system ``atime`` (access time) is unreliable — many Linux mounts use
``relatime`` or ``noatime``, and it is never updated on reads via ``mmap`` or
memory-mapped Zarr stores.  Instead, eostrata maintains lightweight sentinel
files inside a per-group subdirectory named ``.eostrata_access/``.  One file
per timestamp (named after the ISO 8601 datetime) is created/touched whenever
that timestamp is accessed.  ``list_groups`` uses the *maximum* mtime of all
files in the group directory (sentinel mtimes are included naturally via rglob)
as the "last access" timestamp for the group, giving a correct LRU order.

Per-timestamp sentinels live at::

    <zarr_root.parent>/.<zarr_root.name>/access/<group_path>/<timestamp_iso>

This location is **outside** the zarr store so the zarr hierarchy stays clean
— zarr 3's ``consolidate_metadata`` no longer finds foreign directories and
emits spurious "Object not recognised" warnings.

Configuration
-------------
Add to your .env file (all sizes in megabytes):

    EOSTRATA_STORE_QUOTA_MB=10000          # 10 GB — 0 means unlimited (default)
    EOSTRATA_STORE_EVICTION_BUFFER_MB=1000 # keep 1 GB headroom (optional, recommend ~10%)
    EOSTRATA_TRACK_ACCESS=true             # false → last_access = ingestion time only

Concurrency
-----------
Two FileLock types protect the store against concurrent access:

Per-group lock  (``.<zarr_root.name>/locks/<group>__<name>.lock``)
    Held by *both* ingest writes (``geotiff_to_zarr``, ``_write_daily_grid``,
    ``_netcdf_to_zarr``) and ``evict_timestamp``.  Ensures that an eviction
    cannot rename a group directory out from under a concurrent ingest write,
    and that two concurrent evictions of the same group cannot produce
    interleaved renames that silently drop timestamps.

Store-wide eviction lock (``.<zarr_root.name>/locks/__eviction__.lock``)
    Held for the entire duration of a ``check_and_evict`` pass.  Ensures
    that when several ingest jobs start simultaneously, only one of them
    performs the quota check + eviction loop.  Without this lock, all N jobs
    would independently measure an over-quota store, each building a full
    eviction list and collectively evicting N times as much data as needed.

Public API
----------
    from eostrata.cache import (
        check_and_evict, record_access, store_size_mb,
        list_groups, list_timestamps, evict_timestamp,
    )

    # Record that specific timestamps were read (called automatically):
    record_access(zarr_root, "worldpop/nga", [np.datetime64("2020-01-01")])

    # Before a download, ensure there is room:
    check_and_evict(zarr_root, required_mb=500)

    # Query current usage:
    print(store_size_mb(zarr_root))
"""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from filelock import FileLock

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

_DEBOUNCE_S = 60  # minimum seconds between sentinel touches for the same timestamp

# TTL cache for store_size_mb() — avoids rescanning all zarr chunks on every
# quota check.  Key = str(zarr_root), value = (size_mb, monotonic_time).
# Invalidated by evict_timestamp() so the post-eviction final check is accurate.
_SIZE_CACHE: dict[str, tuple[float, float]] = {}
_SIZE_CACHE_TTL = 30.0  # seconds


def _meta_root(zarr_root: Path) -> Path:
    """Hidden state directory *alongside* the zarr store (not inside it).

    Placing lock files and access sentinels here keeps the zarr store a clean
    zarr hierarchy — no foreign directories that zarr 3 would warn about during
    ``consolidate_metadata``.

    Convention: if ``zarr_root`` is ``data/zarr``, the meta directory is
    ``data/.zarr`` (same parent, dot-prefixed name).
    """
    zarr_root = Path(zarr_root)
    meta = zarr_root.parent / f".{zarr_root.name}"
    meta.mkdir(parents=True, exist_ok=True)
    return meta


def _lock_dir(zarr_root: Path) -> Path:
    d = _meta_root(zarr_root) / "locks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _access_dir(zarr_root: Path, group_path: str) -> Path:
    """Per-group access-sentinel directory, outside the zarr store."""
    return _meta_root(zarr_root) / "access" / group_path


def _group_lock(zarr_root: Path, group_path: str) -> FileLock:
    """Per-group FileLock — shared with store.py to serialise ingest vs eviction."""
    lock_name = group_path.replace("/", "__") + ".lock"
    return FileLock(str(_lock_dir(zarr_root) / lock_name))


def _store_eviction_lock(zarr_root: Path) -> FileLock:
    """Store-wide FileLock that serialises concurrent check_and_evict calls.

    Held for the full duration of a quota check + eviction pass so that only
    one pass runs at a time.  This prevents:
      - Two jobs both measuring an over-quota store and evicting the same data.
      - Two jobs evicting more combined data than necessary.
    """
    return FileLock(str(_lock_dir(zarr_root) / "__eviction__.lock"))


def _consolidate_metadata_with_timeout(zarr_root: Path, timeout_s: int = 30) -> None:
    """Call zarr.consolidate_metadata with a timeout, logging a warning if it hangs.

    Uses concurrent.futures so the timeout works from any thread (signal.SIGALRM
    is restricted to the main thread of the main interpreter).
    The consolidation thread is allowed to finish naturally in the background if
    it exceeds the timeout — the caller simply stops waiting.

    Because the zarr store is a clean hierarchy (no lock/access directories),
    zarr 3 consolidation runs without spurious ZarrUserWarnings.
    """
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="zarr_consolidate") as ex:
        future = ex.submit(zarr.consolidate_metadata, str(zarr_root))
        try:
            future.result(timeout=timeout_s)
        except FuturesTimeoutError:
            logger.warning(
                "zarr.consolidate_metadata timed out after %d s — metadata may be stale",
                timeout_s,
            )
        except OSError as exc:
            logger.warning("Could not consolidate zarr metadata after eviction: %s", exc)


def _eviction_sort_key(ts_iso: str, last_access: float, ingestion_time: float) -> tuple:
    """Sort key for LRU eviction: never-accessed timestamps sort first.

    Priority:
    1. Unaccessed (last_access=0) — sorted by ingestion_time, then ts_iso.
    2. Accessed — sorted by last_access ascending (oldest first), then ts_iso.
    """
    if last_access > 0:
        return (1, last_access, ts_iso)
    if ingestion_time > 0:
        return (0, ingestion_time, ts_iso)
    return (0, 0.0, ts_iso)


def _ts_to_iso(ts) -> str:
    """Convert a numpy datetime64 value to ``"YYYY-MM-DDTHH:MM:SS"``."""
    return ts.astype("datetime64[s]").item().strftime("%Y-%m-%dT%H:%M:%S")


def record_access(zarr_root: Path, group_path: str, timestamps: list) -> None:
    """Touch per-timestamp sentinel files inside *group_path* to record the access time.

    The sentinels' mtimes are picked up by ``list_groups`` as the last-access
    timestamp for LRU eviction.  This avoids the unreliable filesystem
    ``atime`` (disabled on many Linux mounts).

    Calls within ``_DEBOUNCE_S`` seconds of the previous touch for the same
    timestamp are skipped to avoid unnecessary write chatter on tile-heavy
    workloads.

    Errors are silently logged so that a read-only filesystem never breaks a
    tile or zonal-stats request.

    Does nothing when ``EOSTRATA_TRACK_ACCESS=false`` — in that case last-access
    time reflects the ingestion timestamp only.

    Parameters
    ----------
    zarr_root:
        Root directory of the Zarr store.
    group_path:
        Group path relative to *zarr_root*, e.g. ``"worldpop/nga"``.
    timestamps:
        List of numpy datetime64 values that were touched by the request.
    """
    from eostrata.config import settings

    if not settings.track_access:
        return

    try:
        adir = _access_dir(Path(zarr_root), group_path)
        adir.mkdir(parents=True, exist_ok=True)
        now = time.time()
        for ts in timestamps:
            ts_iso = _ts_to_iso(ts)
            sentinel = adir / ts_iso
            if sentinel.exists() and now - sentinel.stat().st_mtime < _DEBOUNCE_S:
                continue
            sentinel.touch()
    except OSError:
        logger.debug("Could not update access sentinel for group '%s'", group_path)


def _invalidate_size_cache(zarr_root: Path) -> None:
    """Evict the cached store size for *zarr_root* so the next call re-measures."""
    _SIZE_CACHE.pop(str(Path(zarr_root)), None)


def store_size_mb(zarr_root: Path) -> float:
    """Return the total on-disk size of *zarr_root* in megabytes.

    Results are cached for ``_SIZE_CACHE_TTL`` seconds to avoid rescanning
    thousands of zarr chunk files on every quota check.  The cache is
    invalidated by ``evict_timestamp`` so post-eviction measurements are fresh.
    """
    root = Path(zarr_root)
    cache_key = str(root)
    cached = _SIZE_CACHE.get(cache_key)
    if cached is not None:
        size_mb, cached_at = cached
        if time.monotonic() - cached_at < _SIZE_CACHE_TTL:
            return size_mb
    if not root.exists():
        return 0.0
    total = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    size_mb = total / (1024**2)
    _SIZE_CACHE[cache_key] = (size_mb, time.monotonic())
    return size_mb


def list_groups(zarr_root: Path) -> list[tuple[str, float, float]]:
    """
    Return all top-level Zarr groups in *zarr_root* sorted oldest-first for eviction.

    Sort priority:

    1. Sentinel mtime (``_ACCESSED_SENTINEL``) — reflects when the group was
       last read via tiles or zonal stats.
    2. Minimum data-file mtime — proxy for ingestion time (when data was first
       written), used when the group has never been accessed via tiles.
    3. Group path — alphabetical tiebreaker when both are unavailable.

    Returns
    -------
    List of ``(group_path, size_mb, last_access_time)`` tuples.
    ``group_path`` is relative, e.g. ``"worldpop/nga"``.
    ``last_access_time`` is a Unix timestamp (float).

    Notes
    -----
    ``size_mb`` excludes sentinel files (files whose parent dir name equals
    ``_ACCESS_DIR``).  ``last_access_time`` is computed from the maximum mtime
    of all files in the group directory, including sentinel files — so the most
    recently touched sentinel (= most recently accessed timestamp) automatically
    propagates to the group's last-access time.
    """
    root = Path(zarr_root)
    if not root.exists():
        return []

    groups: list[tuple[str, float, float]] = []

    # Walk two levels deep: source_type/dataset (e.g. worldpop/nga)
    for source_dir in sorted(root.iterdir()):
        if not source_dir.is_dir() or source_dir.name.startswith("."):
            continue
        for dataset_dir in sorted(source_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            group_path = f"{source_dir.name}/{dataset_dir.name}"

            # Single traversal: collect stats once for size + mtime computation
            data_stats = [f.stat() for f in dataset_dir.rglob("*") if f.is_file()]
            size_mb = sum(s.st_size for s in data_stats) / (1024**2)

            # Last access: max mtime across zarr data files AND access sentinels.
            # Sentinels live in the sibling meta directory, outside the zarr store.
            # Compute max directly from data_stats (no extra list allocation).
            data_max_mtime = max((s.st_mtime for s in data_stats), default=0.0)
            adir = _access_dir(zarr_root, group_path)
            sentinel_max_mtime = (
                max((f.stat().st_mtime for f in adir.rglob("*") if f.is_file()), default=0.0)
                if adir.exists()
                else 0.0
            )
            last_access = max(data_max_mtime, sentinel_max_mtime)

            groups.append((group_path, size_mb, last_access))

    # Sort oldest-first; groups with last_access=0 sort by path as tiebreaker
    groups.sort(key=lambda t: (t[2], t[0]))
    return groups


def list_timestamps(zarr_root: Path, group_path: str) -> list[tuple[str, float, float, float]]:
    """
    Return per-timestamp details for a Zarr group, sorted oldest-first.

    Opens the group with ``consolidated=False`` to avoid stale metadata issues.

    Parameters
    ----------
    zarr_root:
        Root directory of the Zarr store.
    group_path:
        Group path relative to *zarr_root*, e.g. ``"worldpop/nga"``.

    Returns
    -------
    List of ``(timestamp_iso, size_mb, last_access, ingestion_time)`` tuples
    sorted oldest-first using the following priority:

    1. ``last_access`` (sentinel mtime) if the timestamp has been accessed.
    2. ``ingestion_time`` (minimum mtime of zarr data files in the group, a
       proxy for when data was first written) when ``last_access`` is 0.
    3. ``timestamp_iso`` lexicographic order (oldest data timestamp first)
       when both access and ingestion times are unavailable.

    ``last_access`` is 0.0 if no sentinel exists for that timestamp.
    ``ingestion_time`` is 0.0 if no zarr data files are found.
    Duplicate timestamps are deduplicated, keeping the first occurrence.
    """
    try:
        ds = xr.open_zarr(str(zarr_root), group=group_path, consolidated=False)
    except Exception:
        return []

    if "time" not in ds:
        ds.close()
        return []

    try:
        times = ds["time"].values
    except Exception:
        ds.close()
        return []
    finally:
        # Close zarr handles as soon as we have the time values in memory;
        # the rest of the function works on plain Python/numpy objects.
        ds.close()

    if len(times) == 0:
        return []

    # Deduplicate while preserving chronological order
    seen: set[str] = set()
    unique_times = []
    for ts in times:
        iso = _ts_to_iso(ts)
        if iso not in seen:
            seen.add(iso)
            unique_times.append((iso, ts))

    group_dir = Path(zarr_root) / group_path

    # Zarr data files only — no sentinel directories in the store.
    # Single stat() pass: collect size and mtime together to halve syscall count.
    # Build data_stats directly (no intermediate data_files list).
    data_stats = (
        [f.stat() for f in group_dir.rglob("*") if f.is_file()]
        if group_dir.exists()
        else []
    )
    if data_stats:
        total_size_bytes = sum(s.st_size for s in data_stats)
        ingestion_time = min(s.st_mtime for s in data_stats)
    else:
        total_size_bytes = 0
        ingestion_time = 0.0
    total_group_size_mb = total_size_bytes / (1024**2)
    per_ts_mb = total_group_size_mb / len(unique_times) if unique_times else 0.0

    # Access sentinels live in the sibling meta directory
    adir = _access_dir(Path(zarr_root), group_path)
    result: list[tuple[str, float, float, float]] = []
    for ts_iso, _ts in unique_times:
        sentinel = adir / ts_iso
        last_access = sentinel.stat().st_mtime if sentinel.exists() else 0.0
        result.append((ts_iso, per_ts_mb, last_access, ingestion_time))

    result.sort(key=lambda t: _eviction_sort_key(t[0], t[2], t[3]))
    return result


def evict_group(zarr_root: Path, group_path: str) -> float:
    """
    Remove the Zarr group at *zarr_root*/*group_path* from disk.

    Returns the number of megabytes freed.
    """
    target = Path(zarr_root) / group_path
    if not target.exists():
        logger.warning("Evict: path does not exist: %s", target)
        return 0.0

    size_mb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / (1024**2)

    shutil.rmtree(target)

    # Remove access sentinels for this group from the meta directory
    adir = _access_dir(zarr_root, group_path)
    if adir.exists():
        shutil.rmtree(adir)

    logger.info("Evicted Zarr group '%s' (freed %.1f MB)", group_path, size_mb)
    return size_mb


def evict_timestamp(
    zarr_root: Path,
    group_path: str,
    timestamp_iso: str,
    *,
    catalog_path: Path | None = None,
) -> float:
    """
    Remove a single timestamp from the Zarr group at *zarr_root*/*group_path*.

    Uses a write-to-temp + atomic rename strategy so the original group is
    never left in a partially-written state.

    Parameters
    ----------
    zarr_root:
        Root directory of the Zarr store.
    group_path:
        Group path relative to *zarr_root*, e.g. ``"worldpop/nga"``.
    timestamp_iso:
        ISO 8601 timestamp to remove, e.g. ``"2021-01-01T00:00:00"``.
    catalog_path:
        If given, the STAC catalog is updated to remove this timestamp.

    Returns
    -------
    Estimated megabytes freed (``total_group_size / n_timestamps``).
    Returns 0.0 if the group or timestamp was not found.
    """
    zarr_root = Path(zarr_root)

    with _group_lock(zarr_root, group_path):
        # Re-open inside the lock so we see the latest committed state.
        # Any concurrent ingest that held this lock before us has already
        # finished its to_zarr() call by the time we get here.
        try:
            ds = xr.open_zarr(str(zarr_root), group=group_path, consolidated=False)
        except Exception:
            return 0.0

        if "time" not in ds:
            ds.close()
            return 0.0

        times = ds["time"].values
        n_times = len(times)
        if n_times == 0:
            ds.close()
            return 0.0

        # Compare at second precision
        target_np = np.datetime64(timestamp_iso, "s")
        times_s = times.astype("datetime64[s]")
        mask = times_s != target_np

        if mask.all():
            # Timestamp not found (already evicted by a concurrent call)
            ds.close()
            return 0.0

        # Estimate freed size (pure zarr data files only)
        group_dir = zarr_root / group_path
        total_size_bytes = sum(f.stat().st_size for f in group_dir.rglob("*") if f.is_file())
        freed_mb = (total_size_bytes / (1024**2)) / n_times

        # Select remaining timestamps by position to avoid issues with duplicate time values
        remaining = ds.isel(time=np.where(mask)[0])

        # Write to a temp group path in the same parent dir (same filesystem).
        # drop_encoding() clears format-specific codec keys so the zarr default
        # compressor is applied cleanly regardless of the store's existing format.
        tmp_name = f"._tmp_{Path(group_path).name}_{uuid.uuid4().hex[:8]}"
        tmp_group_path = str(Path(group_path).parent / tmp_name)
        remaining.drop_encoding().to_zarr(str(zarr_root), group=tmp_group_path, mode="w")
        ds.close()  # release zarr handles after write completes

        target = zarr_root / group_path
        tmp_target = zarr_root / tmp_group_path

        # Copy access sentinels for remaining timestamps into a parallel tmp dir,
        # then atomically swap both the zarr group and the sentinel directory.
        src_access = _access_dir(zarr_root, group_path)
        dst_access = _access_dir(zarr_root, tmp_group_path)
        if src_access.exists():
            shutil.copytree(str(src_access), str(dst_access))
            evicted_sentinel = dst_access / timestamp_iso
            if evicted_sentinel.exists():
                evicted_sentinel.unlink()

        # Atomic swap for zarr data: old → ._old_*, tmp → original
        old = target.parent / f"._old_{uuid.uuid4().hex[:8]}"
        target.rename(old)
        tmp_target.rename(target)
        shutil.rmtree(old)

        # Atomic swap for access sentinels: old_access → gone, tmp_access → original
        old_access = src_access.parent / f"._old_{uuid.uuid4().hex[:8]}"
        if src_access.exists():
            src_access.rename(old_access)
        if dst_access.exists():
            dst_access.rename(src_access)
        if old_access.exists():
            shutil.rmtree(old_access)

    # Invalidate the size cache so the next store_size_mb() call re-measures.
    _invalidate_size_cache(zarr_root)

    # Refresh the root consolidated metadata so tile requests don't read stale
    # time encoding from before the eviction.
    # Uses a thread + Future so the 30-second timeout works from any thread
    # (signal.SIGALRM is restricted to the main thread).
    _consolidate_metadata_with_timeout(zarr_root)

    logger.info(
        "Evicted timestamp '%s' from group '%s' (freed ~%.1f MB)",
        timestamp_iso,
        group_path,
        freed_mb,
    )

    if catalog_path is not None:
        from eostrata import catalog as cat

        catalogue = cat.load_or_create(catalog_path)
        cat.remove_timestamp(catalogue, group_path, timestamp_iso)
        cat.save(catalogue, catalog_path)

    return freed_mb


def check_and_evict(
    zarr_root: Path,
    *,
    quota_mb: float = 0.0,
    required_mb: float = 0.0,
    catalog_path: Path | None = None,
) -> None:
    """
    Evict the oldest timestamps until the store fits within *quota_mb*
    with *required_mb* of headroom.

    Parameters
    ----------
    zarr_root:
        Root directory of the Zarr store.
    quota_mb:
        Maximum allowed store size in megabytes.  ``0`` means unlimited.
    required_mb:
        Extra headroom to reserve before a new download.
    catalog_path:
        If given, the STAC catalog is updated when timestamps are evicted.

    Raises
    ------
    RuntimeError
        If no timestamps are found to evict, or if the store still exceeds
        the quota after evicting everything available.
    """
    if quota_mb <= 0:
        return  # Unlimited — nothing to do

    zarr_root = Path(zarr_root)

    # Acquire the store-wide eviction lock before measuring size.
    # This ensures that only one check_and_evict pass runs at a time, preventing:
    #   • Two concurrent jobs both measuring an over-quota store and each
    #     independently evicting a full pass worth of data (double-eviction).
    #   • Two concurrent passes trying to evict the same timestamp simultaneously
    #     (which would corrupt the group via conflicting renames).
    # evict_timestamp also holds the per-group lock, so ingest jobs that happen
    # to be writing to a group being evicted are safely serialised.
    with _store_eviction_lock(zarr_root):
        # Ignore the buffer if it equals or exceeds the quota (misconfiguration guard)
        effective_required_mb = required_mb if required_mb < quota_mb else 0.0

        current_mb = store_size_mb(zarr_root)
        target_mb = quota_mb - effective_required_mb

        if current_mb <= target_mb:
            logger.debug(
                "Store size %.1f MB is within quota %.1f MB — no eviction needed.",
                current_mb,
                quota_mb,
            )
            return

        logger.info(
            "Store size %.1f MB exceeds target %.1f MB — evicting oldest timestamps.",
            current_mb,
            target_mb,
        )

        # Build flat list of all (group_path, ts_iso, ts_size_mb, last_access, ingestion_time)
        all_timestamps: list[tuple[str, str, float, float, float]] = []
        for group_path, _group_size_mb, _last_access in list_groups(zarr_root):
            for ts_iso, ts_size_mb, ts_last_access, ts_ingestion in list_timestamps(
                zarr_root, group_path
            ):
                all_timestamps.append(
                    (group_path, ts_iso, ts_size_mb, ts_last_access, ts_ingestion)
                )

        if not all_timestamps:
            raise RuntimeError(
                f"Store exceeds quota ({current_mb:.1f} MB > {quota_mb:.1f} MB) "
                "but no timestamps found to evict."
            )

        # Sort oldest-first using same priority as list_timestamps:
        # unaccessed first, then by last_access ascending, then ts_iso
        all_timestamps.sort(key=lambda t: _eviction_sort_key(t[1], t[3], t[4]))

        for group_path, ts_iso, ts_size_mb, last_access, ingestion_time in all_timestamps:
            if current_mb <= target_mb:
                break
            if last_access:
                age_desc = f"last accessed {(time.time() - last_access) / 3600:.1f} h ago"
            elif ingestion_time:
                age_desc = (
                    f"ingested {(time.time() - ingestion_time) / 3600:.1f} h ago, never accessed"
                )
            else:
                age_desc = "no access or ingestion time recorded"
            logger.info(
                "Evicting timestamp '%s' from '%s' (~%.1f MB, %s)",
                ts_iso,
                group_path,
                ts_size_mb,
                age_desc,
            )
            freed = evict_timestamp(zarr_root, group_path, ts_iso, catalog_path=catalog_path)
            current_mb -= freed

        # Re-measure after all evictions (cache was invalidated by evict_timestamp)
        remaining = store_size_mb(zarr_root)
        if remaining > quota_mb:
            raise RuntimeError(
                f"Could not reduce store to quota ({remaining:.1f} MB > {quota_mb:.1f} MB) "
                "after evicting all available timestamps."
            )

        logger.info(
            "Eviction complete — estimated store size now %.1f MB.",
            remaining,
        )
