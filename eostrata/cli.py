"""eostrata command-line interface."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="eostrata",
    help="One tool to fetch, store, aggregate, and serve earth observation layers.",
    no_args_is_help=True,
)
console = Console()

download_app = typer.Typer(help="Download data from a source.", no_args_is_help=True)
app.add_typer(download_app, name="download")


def _parse_int_list(single: int | None, multi: str | None, default: int) -> list[int]:
    """Resolve a single-value / comma-separated option into a sorted list of ints."""
    if multi is not None:
        return sorted({int(v.strip()) for v in multi.split(",")})
    return [single if single is not None else default]


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        format="%(message)s",
        datefmt="[%X]",
    )


# ── download worldpop ─────────────────────────────────────────────────────────


@download_app.command("worldpop")
def download_worldpop(
    iso3: str = typer.Argument(..., help="ISO 3166-1 alpha-3 country code, e.g. NGA"),
    year: int = typer.Option(None, help="Single year (default: latest available)"),
    years: str = typer.Option(None, help="Multiple years, comma-separated: 2020,2021,2022"),
    zarr_root: Path | None = typer.Option(None, help="Override Zarr store root"),
    raw_dir: Path | None = typer.Option(None, help="Override raw download directory"),
    catalog_path: Path | None = typer.Option(None, help="Override catalog.json path"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Download WorldPop population rasters, clip to bbox, write to Zarr and register in STAC.

    Supports multiple years in a single call: --years 2020,2021,2022
    """
    _setup_logging(verbose)

    from eostrata.config import settings
    from eostrata.ingestion import run_worldpop_ingest
    from eostrata.sources import WorldPopSource

    _zarr_root = zarr_root or settings.zarr_root
    _raw_dir = raw_dir or settings.raw_dir
    _catalog_path = catalog_path or settings.catalog_path

    _years = _parse_int_list(year, years, WorldPopSource().latest_available().year)

    console.print(
        f"[bold]Downloading WorldPop[/bold] iso3=[cyan]{iso3.upper()}[/cyan] "
        f"years=[cyan]{_years}[/cyan] bbox=[cyan]{settings.bbox}[/cyan]"
    )

    try:
        run_worldpop_ingest(
            iso3=iso3,
            years=_years,
            zarr_root=_zarr_root,
            raw_dir=_raw_dir,
            catalog_path=_catalog_path,
            bbox=settings.bbox,
            quota_mb=settings.store_quota_mb,
        )
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from None

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

        from eostrata.cache import _ACCESSED_SENTINEL

        table = Table(
            "Group", "Size", "Last accessed", title=f"Zarr store: {_zarr_root}  [{size_summary}]"
        )
        for group_path, size_mb, _ in sorted(groups, key=lambda t: t[0]):
            sentinel = Path(_zarr_root) / group_path / _ACCESSED_SENTINEL
            if sentinel.exists():
                ts = datetime.fromtimestamp(sentinel.stat().st_mtime, tz=UTC)
                last_read = ts.strftime("%Y-%m-%d %H:%M UTC")
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
                timestamps: list[str] = item.properties.get("eostrata:datetimes", [])
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
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
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


# ── download chirps ───────────────────────────────────────────────────────────


@download_app.command("chirps")
def download_chirps(
    year: int = typer.Option(None, help="Single year (default: latest available)"),
    years: str = typer.Option(None, help="Multiple years, comma-separated: 2022,2023"),
    month: int = typer.Option(None, help="Single month 1-12 (default: latest available)"),
    months: str = typer.Option(None, help="Multiple months, comma-separated: 1,2,3"),
    zarr_root: Path | None = typer.Option(None, help="Override Zarr store root"),
    raw_dir: Path | None = typer.Option(None, help="Override raw download directory"),
    catalog_path: Path | None = typer.Option(None, help="Override catalog.json path"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Download CHIRPS monthly precipitation rasters, clip to bbox, write to Zarr and register in STAC.

    Supports multiple years and months in a single call:
      --years 2022,2023 --months 1,2,3
    """
    _setup_logging(verbose)

    from eostrata.config import settings
    from eostrata.ingestion import run_chirps_ingest
    from eostrata.sources.chirps import CHIRPSSource

    _zarr_root = zarr_root or settings.zarr_root
    _raw_dir = raw_dir or settings.raw_dir
    _catalog_path = catalog_path or settings.catalog_path

    latest = CHIRPSSource().latest_available()
    _years = _parse_int_list(year, years, latest.year)
    _months = _parse_int_list(month, months, latest.month)

    console.print(
        f"[bold]Downloading CHIRPS[/bold] years=[cyan]{_years}[/cyan] "
        f"months=[cyan]{_months}[/cyan] bbox=[cyan]{settings.bbox}[/cyan]"
    )

    try:
        run_chirps_ingest(
            years=_years,
            months=_months,
            zarr_root=_zarr_root,
            raw_dir=_raw_dir,
            catalog_path=_catalog_path,
            bbox=settings.bbox,
            quota_mb=settings.store_quota_mb,
        )
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from None

    console.print("[bold green]Done.[/bold green]")


# ── download cds ──────────────────────────────────────────────────────────────


@download_app.command("cds")
def download_cds(
    variable: str = typer.Option("t2m", help="ERA5 variable short name: t2m, tp, u10, v10, sp"),
    year: int = typer.Option(None, help="Single year (default: latest available)"),
    years: str = typer.Option(None, help="Multiple years, comma-separated: 2022,2023"),
    month: int = typer.Option(None, help="Single month 1-12 (default: latest available)"),
    months: str = typer.Option(
        None, help="Months to fetch, comma-separated: 1,2,3 (default: latest available)"
    ),
    zarr_root: Path | None = typer.Option(None, help="Override Zarr store root"),
    raw_dir: Path | None = typer.Option(None, help="Override raw download directory"),
    catalog_path: Path | None = typer.Option(None, help="Override catalog.json path"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Download ERA5 monthly reanalysis from the Copernicus Climate Data Store.

    Supports multiple years in a single call: --years 2022,2023

    Requires a CDS account and ~/.cdsapirc credentials file.
    See: https://cds.climate.copernicus.eu/how-to-api
    """
    _setup_logging(verbose)

    from eostrata.config import settings
    from eostrata.ingestion import run_cds_ingest
    from eostrata.sources.cds import CDSSource

    _zarr_root = zarr_root or settings.zarr_root
    _raw_dir = raw_dir or settings.raw_dir
    _catalog_path = catalog_path or settings.catalog_path

    latest = CDSSource().latest_available()
    _years = _parse_int_list(year, years, latest.year)
    _months = _parse_int_list(month, months, latest.month)

    console.print(
        f"[bold]Downloading ERA5[/bold] variable=[cyan]{variable}[/cyan] "
        f"years=[cyan]{_years}[/cyan] months=[cyan]{_months}[/cyan] bbox=[cyan]{settings.bbox}[/cyan]"
    )

    try:
        run_cds_ingest(
            variable=variable,
            years=_years,
            months=_months,
            zarr_root=_zarr_root,
            raw_dir=_raw_dir,
            catalog_path=_catalog_path,
            bbox=settings.bbox,
            quota_mb=settings.store_quota_mb,
        )
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from None

    console.print("[bold green]Done.[/bold green]")


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
