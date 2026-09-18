"""Coherence-family construction.

Three families are always registered. The Betti-curve ``topology_betti`` family
is an optional sibling package: when ``families/topology_betti`` is present it
registers here under that name and joins config validation through its schema,
and when it is absent nothing else in the repository refers to it.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from importlib.util import find_spec
from typing import Any

from ..contracts import DataSpec
from ..data.normalization import FieldNormalizer
from ..registry import COHERENCE_FAMILY_REGISTRY
from .families import cross_spectrum, global_distribution, topology
from .family_schema import FamilySchema

RESERVED_FAMILIES: dict[str, str] = {}

#: Registered name -> (package, family class name, registration metadata).
_BUILTIN_FAMILIES: tuple[tuple[str, Any, str, dict[str, Any]], ...] = (
    (
        "global_distribution",
        global_distribution,
        "GlobalDistributionFamily",
        {
            "components": (
                "self.marginal_w2",
                "mutual.pairwise_swd",
                "cross.joint_topk_swd",
            ),
            "license": "repository-local scientific implementation",
        },
    ),
    (
        "cross_spectrum",
        cross_spectrum,
        "CrossSpectrumFamily",
        {
            "components": (
                "self_spectrum.auto_spectrum",
                "same_frequency.magnitude_squared",
                "cross_frequency.band_energy_coupling",
                "band_energy.log_power",
            ),
            "aggregation": "ensemble",
        },
    ),
    (
        "topology",
        topology,
        "TopologyFamily",
        {
            "components": (
                "self.persistence_matching",
                "mutual.fibered_matching",
            ),
            "aggregation": "per_sample",
            "status": "experimental; the topology term",
        },
    ),
)

#: The optional Betti-curve package, registered only when it is on disk.
_OPTIONAL_BETTI = f"{__package__}.families.topology_betti"


def _register(name: str, package: Any, class_name: str, metadata: dict[str, Any]) -> None:
    if name in COHERENCE_FAMILY_REGISTRY.names():
        return
    COHERENCE_FAMILY_REGISTRY.register(
        name,
        getattr(package, class_name),
        version="1",
        metadata={**metadata, "schema": package.CONFIG_SCHEMA},
    )


def _register_defaults() -> None:
    for name, package, class_name, metadata in _BUILTIN_FAMILIES:
        _register(name, package, class_name, metadata)
    if find_spec(_OPTIONAL_BETTI) is not None:
        _register(
            "topology_betti",
            import_module(_OPTIONAL_BETTI),
            "BettiCurvesFamily",
            {
                "components": (
                    "self.betti_curves",
                    "mutual.fibered_betti_curves",
                ),
                "aggregation": "per_sample",
                "status": "experimental; optional Betti-curve alternative to topology",
            },
        )


def family_schemas() -> dict[str, FamilySchema]:
    """Config contracts of every registered family, by name."""
    _register_defaults()
    return {
        name: COHERENCE_FAMILY_REGISTRY.get(name).metadata["schema"]
        for name in COHERENCE_FAMILY_REGISTRY.names()
    }


def build_coherence_family(
    name: str,
    config: Mapping[str, Any],
    data_spec: DataSpec,
    normalizer: FieldNormalizer,
):
    _register_defaults()
    normalized = str(name).strip().lower()
    return COHERENCE_FAMILY_REGISTRY.build(
        normalized,
        config=config,
        data_spec=data_spec,
        normalizer=normalizer,
    )


_register_defaults()
