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
                ),
                "license": "repository-local scientific implementation",
            },
        )
    if "cross_spectrum" not in COHERENCE_FAMILY_REGISTRY.names():
        COHERENCE_FAMILY_REGISTRY.register(
            "cross_spectrum",
            CrossSpectrumFamily,
            version="1",
            metadata={
                "components": (
                    "same_frequency.magnitude_squared",
                    "cross_frequency.band_energy_coupling",
                    "band_energy.log_power",
                ),
                "aggregation": "ensemble",
            },
        )
    if "topology" not in COHERENCE_FAMILY_REGISTRY.names():
        COHERENCE_FAMILY_REGISTRY.register(
            "topology",
            TopologyFamily,
            version="1",
            metadata={
                "components": (
                    "self.betti_curves",
                    "mutual.fibered_betti_curves",
                ),
                "aggregation": "per_sample",
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
