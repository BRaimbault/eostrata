"""eostrata command-line interface."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

app = typer.Typer(
    name="eostrata",
    help="One tool to fetch, store, aggregate, and serve earth observation layers.",
    no_args_is_help=True,
)
console = Console()

download_app = typer.Typer(help="Download data from a source.", no_args_is_help=True)
app.add_typer(download_app, name="download")


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
    year: int = typer.Option(None, help="Reference year (default: latest available)"),
    zarr_root: Optional[Path] = typer.Option(None, help="Override Zarr store root"),
    raw_dir: Optional[Path] = typer.Option(None, help="Override raw download directory"),
    catalog_path: Optional[Path] = typer.Option(None, help="Override catalog.json path"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Download a WorldPop population raster, clip to bbox, write to Zarr and register in STAC."""
    _setup_logging(verbose)

    from eostrata.config import settings
    from eostrata.sources import WorldPopSource
    from eostrata import catalog as cat

    source = WorldPopSource()

    # Resolve defaults
    _zarr_root = zarr_root or settings.zarr_root
    _raw_dir = raw_dir or settings.raw_dir
    _catalog_path = catalog_path or settings.catalog_path
    _bbox = settings.bbox

    # Resolve year
    if year is None:
        year = source.latest_available().year
        console.print(f"[dim]No year specified — using latest available: {year}[/dim]")

    console.print(
        f"[bold]Downloading WorldPop[/bold] iso3=[cyan]{iso3.upper()}[/cyan] "
        f"year=[cyan]{year}[/cyan] bbox=[cyan]{_bbox}[/cyan]"
    )

    # Download
    paths = source.download(_raw_dir, _bbox, iso3=iso3, year=year)
    tif_path = paths[0]
    console.print(f"[green]Downloaded[/green] {tif_path}")

    # Write to Zarr
    ds = source.to_zarr(tif_path, _zarr_root, _bbox, iso3=iso3, year=year)
    zarr_group = f"worldpop/{iso3.lower()}_{year}_1km"
    console.print(f"[green]Zarr written[/green] {_zarr_root}/{zarr_group}")

    # Register STAC item
    catalogue = cat.load_or_create(_catalog_path)

    # Derive bbox from the actual written dataset
    item_bbox = (
        float(ds.x.min()),
        float(ds.y.min()),
        float(ds.x.max()),
        float(ds.y.max()),
    )

    cat.register_item(
        catalogue,
        collection_id=source.collection_id,
        item_id=source.stac_item_id(iso3=iso3, year=year),
        bbox=item_bbox,
        datetime_=datetime(year, 1, 1, tzinfo=timezone.utc),
        zarr_root=_zarr_root,
        zarr_group=zarr_group,
        variable=zarr_group.split("/")[-1],
        extra_properties=source.stac_properties(iso3=iso3, year=year),
    )
    cat.save(catalogue, _catalog_path)
    console.print(f"[green]STAC item registered[/green] {_catalog_path}")

    console.print("[bold green]Done.[/bold green]")


# ── list ──────────────────────────────────────────────────────────────────────

@app.command("list")
def list_datasets(
    zarr_root: Optional[Path] = typer.Option(None, help="Zarr store root"),
    catalog_path: Optional[Path] = typer.Option(None, help="Path to catalog.json"),
) -> None:
    """List all datasets currently in the Zarr store and STAC catalogue."""
    import zarr as zarr_lib
    from eostrata.config import settings
    from eostrata import catalog as cat

    _zarr_root = zarr_root or settings.zarr_root
    _catalog_path = catalog_path or settings.catalog_path

    # Zarr store
    if Path(_zarr_root).exists():
        store = zarr_lib.open_group(str(_zarr_root), mode="r")
        table = Table("Group", "Arrays", title=f"Zarr store: {_zarr_root}")
        for key in sorted(store.keys()):
            grp = store[key]
            arrays = ", ".join(sorted(grp.keys())) if hasattr(grp, "keys") else "-"
            table.add_row(key, arrays)
        console.print(table)
    else:
        console.print(f"[yellow]No Zarr store found at {_zarr_root}[/yellow]")

    # STAC catalogue
    if Path(_catalog_path).exists():
        catalogue = cat.load_or_create(_catalog_path)
        stac_table = Table("Collection", "Item ID", "Datetime", title="STAC catalogue")
        for collection in catalogue.get_children():
            for item in collection.get_items():
                dt = item.datetime.strftime("%Y-%m-%d") if item.datetime else "-"
                stac_table.add_row(collection.id, item.id, dt)
        console.print(stac_table)
    else:
        console.print(f"[yellow]No catalog found at {_catalog_path}[/yellow]")


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
    console.print(f"  [dim]Tiles:     http://{host}:{port}/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}[/dim]")
    console.print(f"  [dim]Processes: http://{host}:{port}/processes[/dim]")
    uvicorn.run(
        "eostrata.server:app",
        host=host,
        port=port,
        reload=reload,
    )


# ── cleanup ───────────────────────────────────────────────────────────────────

@app.command("cleanup")
def cleanup(
    zarr_root: Optional[Path] = typer.Option(None, help="Zarr store root"),
    raw_dir: Optional[Path] = typer.Option(None, help="Raw downloads directory"),
    catalog_path: Optional[Path] = typer.Option(None, help="Path to catalog.json"),
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


if __name__ == "__main__":
    app()
