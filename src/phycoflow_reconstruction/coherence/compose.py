"""Build the enabled family composition without flattening its taxonomy."""

from __future__ import annotations

from collections.abc import Mapping
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
