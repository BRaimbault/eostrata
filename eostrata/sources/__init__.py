"""Import all sources to populate the registry on package import."""

from eostrata.sources.cds import CDSSource
from eostrata.sources.chirps import CHIRPSSource
from eostrata.sources.sentinel_ndvi import SentinelNDVISource
from eostrata.sources.worldpop import WorldPopSource

__all__ = ["WorldPopSource", "CHIRPSSource", "CDSSource", "SentinelNDVISource"]
