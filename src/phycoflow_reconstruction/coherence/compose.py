"""Build the enabled family composition without flattening its taxonomy."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..contracts import DataSpec
from ..data.normalization import FieldNormalizer
from .registry import build_coherence_family


def build_enabled_families(
    config: Mapping[str, Any],
    data_spec: DataSpec,
    normalizer: FieldNormalizer,
) -> dict[str, Any]:
    families = {}
    for name, settings in config.get("families", {}).items():
        if bool(settings.get("enabled", True)):
            families[name] = build_coherence_family(name, settings, data_spec, normalizer)
    if not families:
        raise ValueError("post-training coherence must enable at least one family")
    return families


def three_family_order(names: Iterable[str]) -> tuple[str, str, str]:
    """Resolve the three ConFIG rows for legacy or signed-multiscale Scale."""
    enabled = set(names)
    for scale in ("cross_spectrum", "multiscale_spectrum"):
        order = ("global_distribution", scale, "topology")
        if enabled == set(order):
            return order
    raise ValueError(
        "coherence_aggregation requires enabled coherence families global_distribution, "
        "one of cross_spectrum or multiscale_spectrum, and topology"
    )
