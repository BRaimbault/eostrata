"""eostrata command-line interface."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from eostrata.constants import PROP_DATETIMES

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="eostrata",
    help="One tool to fetch, store, aggregate, and serve earth observation layers.",
    no_args_is_help=True,
)
console = Console()

_ALL_MONTHS = list(range(1, 13))
_ALL_DAYS = list(range(1, 32))


def _parse_int_list(
    single: int | None,
    multi: str | None,
    default: int,
    *,
    all_values: list[int] | None = None,
) -> list[int]:
    """Resolve a single-value / comma-separated option into a sorted list of ints.

    Passing ``"ALL"`` (case-insensitive) as *multi* expands to *all_values*.
    """
    if multi is not None:
        if multi.strip().upper() == "ALL":
            if all_values is None:
                raise ValueError("'ALL' is not supported for this parameter")
            return all_values
        return sorted({int(v.strip()) for v in multi.split(",")})
    return [single if single is not None else default]


def _setup_logging(verbose: bool) -> None:
    from eostrata.log import setup_logging

    setup_logging(verbose=verbose)


_ALL_DEKADS = [1, 2, 3]


@app.command("download")
def download(
    source_id: str = typer.Argument(
        ..., help="Source ID: worldpop, chirps, cds, sentinel_ndvi (or sentinel-ndvi)"
    ),
    iso3: str = typer.Option(
        None, help="ISO 3166-1 alpha-3 country code (worldpop only), e.g. NGA"
    ),
    variable: str | None = typer.Option(
        None, help="ERA5 variable short name (cds only): t2m, tp, u10, v10, sp"
    ),
    year: int = typer.Option(None, help="Single year (default: latest available)"),
    years: str = typer.Option(None, help="Multiple years, comma-separated: 2020,2021,2022"),
    month: int = typer.Option(None, help="Single month 1-12 (default: latest available)"),
    months: str = typer.Option(None, help="Multiple months, comma-separated: 1,2,3 or ALL"),
    dekad: int = typer.Option(None, help="Single dekad 1-3 (sentinel_ndvi only)"),
    dekads: str = typer.Option(None, help="Multiple dekads: 1,2,3 or ALL (sentinel_ndvi only)"),
    day: int = typer.Option(None, help="Single day 1-31 (daily sources only)"),
    days: str = typer.Option(
        None, help="Multiple days, comma-separated: 1,2,3 or ALL (daily sources only)"
    ),
    zarr_root: Path | None = typer.Option(None, help="Override Zarr store root"),
    raw_dir: Path | None = typer.Option(None, help="Override raw download directory"),
    catalog_path: Path | None = typer.Option(None, help="Override catalog.json path"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Download data from a source, clip to bbox, write to Zarr and register in STAC.

    SOURCE_ID can be: worldpop, chirps, cds, sentinel_ndvi (or sentinel-ndvi).
    """
    _setup_logging(verbose)

    # Normalize hyphens to underscores (e.g. sentinel-ndvi → sentinel_ndvi)
    source_id = source_id.replace("-", "_")

    from eostrata.config import settings
    from eostrata.ingestion import run_ingest
    from eostrata.sources.base import get_source

    _zarr_root = zarr_root or settings.zarr_root
    _raw_dir = raw_dir or settings.raw_dir
    _catalog_path = catalog_path or settings.catalog_path

    try:
        source_cls = get_source(source_id)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from None

    source = source_cls()
    latest = source.latest_available()

    source_params: dict = {}
    if "iso3" in source_cls.ui_fields:
        if iso3 is None:
            console.print(f"[red]Error:[/red] --iso3 is required for source '{source_id}'")
            raise typer.Exit(1) from None
        source_params["iso3"] = iso3
    if "variable" in source_cls.ui_fields:
        source_params["variable"] = variable or "t2m"
    if "years" in source_cls.ui_fields:
        source_params["years"] = _parse_int_list(year, years, latest.year)
    if "months" in source_cls.ui_fields:
        source_params["months"] = _parse_int_list(
            month, months, latest.month, all_values=_ALL_MONTHS
        )
    if "dekads" in source_cls.ui_fields:
        _default_dekad = 1 if latest.day < 11 else (2 if latest.day < 21 else 3)
        source_params["dekads"] = _parse_int_list(
            dekad, dekads, _default_dekad, all_values=_ALL_DEKADS
        )
    if "days" in source_cls.ui_fields:  # pragma: no cover
        source_params["days"] = _parse_int_list(day, days, latest.day, all_values=_ALL_DAYS)

    logger.info("CLI download %s params=%s bbox=%s", source_id, source_params, settings.bbox)
    console.print(
        f"[bold]Downloading {source_id}[/bold] params=[cyan]{source_params}[/cyan] "
        f"bbox=[cyan]{settings.bbox}[/cyan]"
    )

    try:
        failed, saved = run_ingest(
            source_id,
            zarr_root=_zarr_root,
            raw_dir=_raw_dir,
            catalog_path=_catalog_path,
            bbox=settings.bbox,
            quota_mb=settings.store_quota_mb,
            eviction_buffer_mb=settings.store_eviction_buffer_mb,
            **source_params,
        )
    except Exception as exc:
        logger.exception("CLI command failed: %s", exc)
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from None

    if not saved:
        if failed:
            console.print(
                f"[red]Error: nothing ingested — {len(failed)} period(s) failed: "
                f"{', '.join(failed)}[/red]"
            )
        else:
            console.print(
                "[red]Error: nothing ingested — all requested periods may be unavailable[/red]"
            )
        raise typer.Exit(1) from None
    if failed:
        console.print(
            f"[yellow]Warning: {len(failed)} period(s) failed to download: "
            f"{', '.join(failed)}[/yellow]"
        )
    console.print("[bold green]Done.[/bold green]")


# ── list ──────────────────────────────────────────────────────────────────────


@app.command("list")
def list_datasets(
    zarr_root: Path | None = typer.Option(None, help="Zarr store root"),
    catalog_path: Path | None = typer.Option(None, help="Path to catalog.json"),
) -> None:
    """List all datasets currently in the Zarr store and STAC catalogue."""
    from eostrata import catalog as cat
    from eostrata.cache import list_groups, store_size_mb
    from eostrata.config import settings

    _zarr_root = zarr_root or settings.zarr_root
    _catalog_path = catalog_path or settings.catalog_path
    _quota_mb = settings.store_quota_mb

    # Zarr store
    if Path(_zarr_root).exists():
        total_mb = store_size_mb(_zarr_root)
        groups = list_groups(_zarr_root)

        # Header: size + quota
        if _quota_mb > 0:
            pct = total_mb / _quota_mb * 100
            size_summary = f"{total_mb:.1f} MB / {_quota_mb:.0f} MB ({pct:.0f}%)"
        else:
            size_summary = f"{total_mb:.1f} MB (no quota)"

        from datetime import UTC, datetime

        from eostrata.cache import _access_dir

        table = Table(
            "Group", "Size", "Last accessed", title=f"Zarr store: {_zarr_root}  [{size_summary}]"
        )
        for group_path, size_mb, _ in sorted(groups, key=lambda t: t[0]):
            adir = _access_dir(Path(_zarr_root), group_path)
            if adir.exists():
                try:
                    newest_mtime = max(f.stat().st_mtime for f in adir.iterdir())
                    ts = datetime.fromtimestamp(newest_mtime, tz=UTC)
                    last_read = ts.strftime("%Y-%m-%d %H:%M UTC")
                except ValueError:  # empty directory — max() of empty sequence
                    last_read = "[dim]never read[/dim]"
            else:
                last_read = "[dim]never read[/dim]"
            table.add_row(group_path, f"{size_mb:.1f} MB", last_read)
        console.print(table)
    else:
        console.print(f"[yellow]No Zarr store found at {_zarr_root}[/yellow]")

    # STAC catalogue
    if Path(_catalog_path).exists():
        catalogue = cat.load_or_create(_catalog_path)
        stac_table = Table("Collection", "Item ID", "Timestamps", title="STAC catalogue")
        for collection in catalogue.get_children():
            for item in collection.get_items():
                timestamps: list[str] = item.properties.get(PROP_DATETIMES, [])
                if timestamps:
                    # Show as short dates; flag gaps with a marker
                    dates = [t[:10] for t in timestamps]
                    dt = ", ".join(dates)
                elif item.datetime:
                    dt = item.datetime.strftime("%Y-%m-%d")
                else:
                    start = item.common_metadata.start_datetime
                    end = item.common_metadata.end_datetime
                    dt = (
                        f"{start.strftime('%Y-%m-%d')} / {end.strftime('%Y-%m-%d')}"
                        if start
                        else "-"
                    )
                stac_table.add_row(collection.id, item.id, dt)
        console.print(stac_table)
    else:
        console.print(f"[yellow]No catalog found at {_catalog_path}[/yellow]")


# ── test ──────────────────────────────────────────────────────────────────────


@app.command("test")
def run_tests(
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose pytest output"),
    no_cov: bool = typer.Option(False, "--no-cov", help="Skip coverage reporting"),
) -> None:
    """Run the test suite with coverage (uv run pytest tests/ --cov=eostrata)."""
    import subprocess

    cmd = ["uv", "run", "pytest", "tests/"]
    if not no_cov:
        cmd += ["--cov=eostrata", "--cov-report=term-missing"]
    if verbose:
        cmd.append("-v")
    result = subprocess.run(cmd)
    raise typer.Exit(result.returncode)


# ── lint ──────────────────────────────────────────────────────────────────────


@app.command("lint")
def run_lint(
    fix: bool = typer.Option(True, "--fix/--no-fix", help="Auto-fix ruff issues"),
) -> None:
    """Lint and format the codebase with ruff."""
    import subprocess

    check_cmd = ["uv", "run", "ruff", "check", "eostrata/", "tests/"]
    if fix:
        check_cmd.append("--fix")
    r1 = subprocess.run(check_cmd)

    fmt_cmd = ["uv", "run", "ruff", "format", "eostrata/", "tests/"]
    r2 = subprocess.run(fmt_cmd)

    raise typer.Exit(max(r1.returncode, r2.returncode))


# ── serve ─────────────────────────────────────────────────────────────────────


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", envvar="HOST", help="Bind host"),
    port: int = typer.Option(8000, envvar="PORT", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Hot-reload (dev only)"),
) -> None:
    """Start the tile server, STAC catalogue and OGC Processes API."""
    import uvicorn

    console.print(f"[cyan]Starting eostrata server on http://{host}:{port}[/cyan]")
    console.print(f"  [dim]Docs:      http://{host}:{port}/docs[/dim]")
    console.print(f"  [dim]STAC:      http://{host}:{port}/stac[/dim]")
    console.print(
        f"  [dim]Tiles:     http://{host}:{port}/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}[/dim]"
    )
    console.print(f"  [dim]Processes: http://{host}:{port}/processes[/dim]")
    console.print(f"  [dim]Viewer:    http://{host}:{port}/map[/dim]")
    uvicorn.run(
        "eostrata.server:app",
        host=host,
        port=port,
        reload=reload,
    )


# ── cleanup ───────────────────────────────────────────────────────────────────


@app.command("cleanup")
def cleanup(
    zarr_root: Path | None = typer.Option(None, help="Zarr store root"),
    raw_dir: Path | None = typer.Option(None, help="Raw downloads directory"),
    catalog_path: Path | None = typer.Option(None, help="Path to catalog.json"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Delete the Zarr store, raw downloads and STAC catalogue. For dev/testing only."""
    import shutil

    from eostrata.config import settings

    _zarr_root = Path(zarr_root or settings.zarr_root)
    _raw_dir = Path(raw_dir or settings.raw_dir)
    _catalog_path = Path(catalog_path or settings.catalog_path)

    targets = [t for t in [_zarr_root, _raw_dir, _catalog_path.parent] if t.exists()]

    if not targets:
        console.print("[yellow]Nothing to clean up.[/yellow]")
        raise typer.Exit()

    console.print("[bold red]The following will be permanently deleted:[/bold red]")
    for t in [_zarr_root, _raw_dir, _catalog_path.parent]:
        console.print(f"  [dim]{t}[/dim]")

    if not yes:
        typer.confirm("Proceed?", abort=True)

    if _zarr_root.exists():
        shutil.rmtree(_zarr_root)
        console.print(f"[red]Deleted[/red] {_zarr_root}")

    if _raw_dir.exists():
        shutil.rmtree(_raw_dir)
        console.print(f"[red]Deleted[/red] {_raw_dir}")

    if _catalog_path.parent.exists():
        shutil.rmtree(_catalog_path.parent)
        console.print(f"[red]Deleted[/red] {_catalog_path.parent}")

    console.print("[bold]Cleanup complete.[/bold]")


# ── rebuild-catalog ───────────────────────────────────────────────────────────


@app.command("rebuild-catalog")
def rebuild_catalog(
    zarr_root: Path | None = typer.Option(None, help="Zarr store root"),
    catalog_path: Path | None = typer.Option(None, help="Path to catalog.json"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Rebuild the STAC catalogue from scratch by scanning the Zarr store.

    Reads time coordinates and spatial extent directly from each Zarr group.
    Useful when the catalogue is missing or out of sync with the stored data.
    """
    _setup_logging(verbose)

    from eostrata.config import settings
    from eostrata.ingestion import rebuild_catalog_from_zarr

    _zarr_root = zarr_root or settings.zarr_root
    _catalog_path = catalog_path or settings.catalog_path

    console.print(f"[bold]Rebuilding catalogue[/bold] from [cyan]{_zarr_root}[/cyan]")

    results = rebuild_catalog_from_zarr(zarr_root=_zarr_root, catalog_path=_catalog_path)

    if not results:
        console.print("[yellow]No Zarr groups found — catalogue is empty.[/yellow]")
    else:
        table = Table("Group", "Timestamps", title="Rebuilt STAC catalogue")
        for group_path, n_ts in sorted(results.items()):
            table.add_row(group_path, str(n_ts))
        console.print(table)
        console.print(f"[green]Catalogue saved to[/green] {_catalog_path}")

    console.print("[bold green]Done.[/bold green]")


if __name__ == "__main__":  # pragma: no cover
    app()
