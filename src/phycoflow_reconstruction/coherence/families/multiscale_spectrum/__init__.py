"""Signed-band and joint spectral scores with frozen geometry and training scales."""

from .config import COMPONENT_KEYS, FAMILY_KEYS, OPT_IN_COMPONENTS, parse_multiscale_config
from .family import MultiscaleSpectrumFamily

__all__ = [
    "COMPONENT_KEYS",
    "FAMILY_KEYS",
    "OPT_IN_COMPONENTS",
    "MultiscaleSpectrumFamily",
    "parse_multiscale_config",
]
