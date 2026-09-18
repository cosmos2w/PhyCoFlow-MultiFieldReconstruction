"""Graph cross-spectrum coherence family and its training objectives."""

from .family import CrossSpectrumFamily
from .schema import CONFIG_SCHEMA

__all__ = ["CONFIG_SCHEMA", "CrossSpectrumFamily"]
