"""Shared STAC item property keys used across multiple eostrata modules."""

# Core properties read back by catalog, server, tiles, and CLI
PROP_VARIABLE = "eostrata:variable"
PROP_SOURCE = "eostrata:source"
PROP_ZARR_GROUP = "eostrata:zarr_group"
PROP_ZARR_ROOT = "eostrata:zarr_root"
PROP_DATETIMES = "eostrata:datetimes"

# Metadata property written by all data sources
PROP_RESOLUTION = "eostrata:resolution"
