"""Zarr store cache management — quota checks and LRU eviction.

The eviction policy works at the Zarr *group* level (one group = one dataset,
e.g. ``worldpop/nga`` or ``chirps/global``).  When the store exceeds the
configured quota, the least-recently-accessed groups are removed until the
store fits within the quota again.

Configuration
-------------
Add to your .env file (all sizes in megabytes):

    EOSTRATA_STORE_QUOTA_MB=10000   # 10 GB — 0 means unlimited (default)

Public API
----------
    from eostrata.cache import check_and_evict, store_size_mb, list_groups

    # Before a download, ensure there is room:
    check_and_evict(zarr_root, required_mb=500)

    # Query current usage:
    print(store_size_mb(zarr_root))
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def store_size_mb(zarr_root: Path) -> float:
    """Return the total on-disk size of *zarr_root* in megabytes."""
    root = Path(zarr_root)
    if not root.exists():
        return 0.0
    total = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    return total / (1024**2)


def list_groups(zarr_root: Path) -> list[tuple[str, float, float]]:
    """
    Return all top-level Zarr groups in *zarr_root* sorted by last-access time
    (oldest first — candidates for eviction).

    Returns
    -------
    List of ``(group_path, size_mb, last_access_time)`` tuples.
    ``group_path`` is relative, e.g. ``"worldpop/nga"``.
    ``last_access_time`` is a Unix timestamp (float).
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
            size_mb = sum(f.stat().st_size for f in dataset_dir.rglob("*") if f.is_file()) / (
                1024**2
            )
            # Use the latest modification time of any file as a proxy for
            # last access (atime is not reliable on all filesystems/OSes).
            mtimes = [f.stat().st_mtime for f in dataset_dir.rglob("*") if f.is_file()]
            last_access = max(mtimes) if mtimes else 0.0
            groups.append((group_path, size_mb, last_access))

    # Sort oldest-access first
    groups.sort(key=lambda t: t[2])
    return groups


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
    logger.info("Evicted Zarr group '%s' (freed %.1f MB)", group_path, size_mb)
    return size_mb


def check_and_evict(
    zarr_root: Path,
    *,
    quota_mb: float = 0.0,
    required_mb: float = 0.0,
) -> None:
    """
    Evict the oldest Zarr groups until the store fits within *quota_mb*
    with *required_mb* of headroom.

    Parameters
    ----------
    zarr_root:
        Root directory of the Zarr store.
    quota_mb:
        Maximum allowed store size in megabytes.  ``0`` means unlimited.
    required_mb:
        Extra headroom to reserve before a new download.

    Raises
    ------
    RuntimeError
        If the store still exceeds the quota after evicting every group
        (should not happen in practice).
    """
    if quota_mb <= 0:
        return  # Unlimited — nothing to do

    current_mb = store_size_mb(zarr_root)
    target_mb = quota_mb - required_mb

    if current_mb <= target_mb:
        logger.debug(
            "Store size %.1f MB is within quota %.1f MB — no eviction needed.",
            current_mb,
            quota_mb,
        )
        return

    logger.info(
        "Store size %.1f MB exceeds target %.1f MB — evicting oldest groups.",
        current_mb,
        target_mb,
    )

    groups = list_groups(zarr_root)
    if not groups:
        raise RuntimeError(
            f"Store exceeds quota ({current_mb:.1f} MB > {quota_mb:.1f} MB) "
            "but no groups found to evict."
        )

    freed_mb = 0.0
    for group_path, size_mb, last_access in groups:
        if current_mb - freed_mb <= target_mb:
            break
        age_hours = (time.time() - last_access) / 3600
        logger.info(
            "Evicting '%s' (%.1f MB, last accessed %.1f h ago)",
            group_path,
            size_mb,
            age_hours,
        )
        evict_group(zarr_root, group_path)
        freed_mb += size_mb  # size already known from list_groups; avoid re-walking

    remaining = current_mb - freed_mb
    if remaining > quota_mb:
        raise RuntimeError(
            f"Could not reduce store to quota ({remaining:.1f} MB > {quota_mb:.1f} MB) "
            "after evicting all available groups."
        )

    logger.info(
        "Eviction complete — freed %.1f MB, estimated store size now %.1f MB.",
        freed_mb,
        remaining,
    )
