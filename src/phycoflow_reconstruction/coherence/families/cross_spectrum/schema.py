"""Config contract of the cross-spectrum family."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...family_schema import FamilySchema, reject_unknown_keys


def _component_active(family: Mapping[str, Any], name: str) -> bool:
    settings = family.get("components", {}).get(name, {})
    return (
        bool(family.get("enabled", True))
        and float(family.get("weight", 1.0)) > 0
        and bool(settings.get("enabled", True))
        and float(settings.get("weight", 1.0)) > 0
    )


def _check(family: Mapping[str, Any], compute: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    graph = family.get("graph", {})
    reject_unknown_keys(
        graph,
        {"k_neighbors", "sigma", "num_modes", "exclude_zero", "bands"},
        "cross_spectrum.graph",
    )
    if int(graph.get("k_neighbors", 16)) < 1 or int(graph.get("num_modes", 64)) < 1:
        raise ValueError("cross_spectrum graph sizes must be positive")
    cross_active = _component_active(family, "cross_frequency")
    if cross_active and int(compute["batch_size"]) < 3:
        raise ValueError("cross-frequency coherence requires compute batch_size>=3")
    same_active = _component_active(family, "same_frequency")
    if same_active and int(compute["batch_size"]) < 2:
        raise ValueError("same-frequency coherence requires compute batch_size>=2")
    evaluation_minimum = 3 if cross_active else 2 if same_active else 1
    if int(config.get("evaluation", {}).get("max_samples", 1)) < evaluation_minimum:
        raise ValueError(
            f"evaluation.max_samples must be >= {evaluation_minimum} for active spectral components"
        )


CONFIG_SCHEMA = FamilySchema(
    keys=frozenset({"pairs", "graph", "eps"}),
    components={
        "self_spectrum": frozenset({"enabled", "weight"}),
        "same_frequency": frozenset({"enabled", "weight"}),
        "cross_frequency": frozenset({"enabled", "weight"}),
        "band_energy": frozenset({"enabled", "weight"}),
    },
    # The raw-power self spectrum is pending development and never enabled implicitly.
    opt_in_components=frozenset({"self_spectrum"}),
    requires_fixed_shared=True,
    check=_check,
)
