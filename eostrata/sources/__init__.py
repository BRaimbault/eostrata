"""Auto-discover and import all source modules to populate the registry."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

# Import every module in this package (except base) so that each
# @register_source decorator runs and the class enters the registry.
_here = Path(__file__).parent
for _mod in pkgutil.iter_modules([str(_here)]):
    if _mod.name != "base":
        importlib.import_module(f"eostrata.sources.{_mod.name}")

# Expose all registered source classes at the package level so that
# `from eostrata.sources import WorldPopSource` keeps working.
from eostrata.sources.base import _REGISTRY  # noqa: E402

globals().update({cls.__name__: cls for cls in _REGISTRY.values()})
__all__ = [cls.__name__ for cls in _REGISTRY.values()]
