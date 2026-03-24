"""OGC API - Tiles router — wraps titiler.xarray TilerFactory."""
from __future__ import annotations

from titiler.xarray.extensions import VariablesExtension
from titiler.xarray.factory import TilerFactory

tiler = TilerFactory(
    router_prefix="/md",
    extensions=[VariablesExtension()],
)

router = tiler.router
