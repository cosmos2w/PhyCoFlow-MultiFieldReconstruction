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
    from .relation_queries import QueryBank, resolve_coverage_config

    config = resolve_coverage_config(config, data_spec.field_names)
    families = {}
    for name, settings in config.get("families", {}).items():
        if bool(settings.get("enabled", True)):
            family = build_coherence_family(name, settings, data_spec, normalizer)
            if name == "multiscale_spectrum" and config.get("query_coverage", {}).get("enabled", False):
                bank = QueryBank(family.field_names, seed=int(config["query_coverage"].get("relation_seed", 0)))
                indices = {field: i for i, field in enumerate(family.field_names)}
                family.coverage_pairs = tuple((indices[a], indices[b]) for a, b in bank.pairs)
            families[name] = family
    if not families:
        raise ValueError("post-training coherence must enable at least one family")
    return families
