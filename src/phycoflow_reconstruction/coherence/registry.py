"""Coherence-family construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts import DataSpec
from ..data.normalization import FieldNormalizer
from ..registry import COHERENCE_FAMILY_REGISTRY
from .families.cross_spectrum import CrossSpectrumFamily
from .families.global_distribution import GlobalDistributionFamily
from .families.topology import TopologyFamily

RESERVED_FAMILIES: dict[str, str] = {}


def _register_defaults() -> None:
    if "global_distribution" not in COHERENCE_FAMILY_REGISTRY.names():
        COHERENCE_FAMILY_REGISTRY.register(
            "global_distribution",
            GlobalDistributionFamily,
            version="1",
            metadata={
                "components": (
                    "self.marginal_w2",
                    "mutual.pairwise_swd",
                    "cross.joint_topk_swd",
                    "cross.joint_copula_cvar",
                ),
                "definitions": {"legacy_v1": "1", "marginal_copula_v2": "2"},
                "default_definition": "legacy_v1",
                "license": "repository-local scientific implementation",
            },
        )
    if "cross_spectrum" not in COHERENCE_FAMILY_REGISTRY.names():
        COHERENCE_FAMILY_REGISTRY.register(
            "cross_spectrum",
            CrossSpectrumFamily,
            version="3",
            metadata={
                "components": (
                    "self_spectrum.auto_spectrum",
                    "same_frequency.magnitude_squared",
                    "cross_frequency.band_energy_coupling",
                    "band_energy.log_power",
                    "same_frequency.second_order_blocks",
                    "cross_frequency.second_order_blocks",
                ),
                "definitions": {"legacy_v3": "3", "second_order_blocks_v4": "4"},
                "default_definition": "legacy_v3",
                "aggregation": "ensemble",
            },
        )
    if "topology" not in COHERENCE_FAMILY_REGISTRY.names():
        COHERENCE_FAMILY_REGISTRY.register(
            "topology",
            TopologyFamily,
            version="3",
            metadata={
                "components": (
                    "self.betti_curves",
                    "mutual.fibered_betti_curves",
                    "self.region", "self.connectivity", "anchor_self.spatial", "mutual.spatial",
                    "self.persistence", "mutual.persistence",
                ),
                "aggregation": "per_sample",
                "strategies": ("betti_curves", "spatial_self_mutual", "cubical_persistence"),
            },
        )


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
