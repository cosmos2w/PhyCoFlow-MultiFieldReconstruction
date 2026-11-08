"""Coherence-family construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts import DataSpec
from ..data.normalization import FieldNormalizer
from ..registry import COHERENCE_FAMILY_REGISTRY
from .families.cross_spectrum import CrossSpectrumFamily
from .families.global_distribution import GlobalDistributionFamily
from .families.multiscale_spectrum import MultiscaleSpectrumFamily
from .families.multiscale_spectrum.config import OBJECTIVES
from .families.multiscale_spectrum.objectives import (
    BAND_COHERENCE_PATH,
    ENERGY_SCORE_PATH,
    JOINT_ENERGY_PATH,
    SIGNED_BAND_PATH,
)
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
            version="2",
            metadata={
                "components": (
                    "self_spectrum.auto_spectrum",
                    "same_frequency.magnitude_squared",
                    "cross_frequency.band_energy_coupling",
                    "band_energy.log_power",
                ),
                "aggregation": "ensemble",
            },
        )
    if "multiscale_spectrum" not in COHERENCE_FAMILY_REGISTRY.names():
        COHERENCE_FAMILY_REGISTRY.register(
            "multiscale_spectrum",
            MultiscaleSpectrumFamily,
            version=MultiscaleSpectrumFamily.version,
            metadata={
                "components": (
                    "log_band_power.squared_error",
                    "signed_band_cospectrum.squared_error",
                    "third_order_product_moments.squared_error",
                    ENERGY_SCORE_PATH, SIGNED_BAND_PATH, JOINT_ENERGY_PATH, BAND_COHERENCE_PATH,
                ),
                "aggregation": "per_sample",
                "objectives": OBJECTIVES,
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
    if normalized == "global_distribution" and config.get("definition") == "conditional_quantile_fair_v1":
        from .families.global_distribution.conditional import ConditionalQuantileFairFamily

        return ConditionalQuantileFairFamily(config, data_spec, normalizer)
    if normalized == "multiscale_spectrum" and "relation_extension" in config:
        extension = config["relation_extension"]
        if isinstance(extension, Mapping) and extension.get("definition") == "self_rho_native_quadrature_v1":
            from .families.native_relation import NativeRelationFamily

            return NativeRelationFamily(config, data_spec, normalizer)
        from .families.multiscale_relation import MultiscaleRelationFamily

        return MultiscaleRelationFamily(config, data_spec, normalizer)
    if normalized == "multiscale_spectrum" and "gram_extension" in config:
        # Keep the frozen v4 implementation and its artifact hashes untouched.
        from .families.multiscale_gram import MultiscaleGramFamily

        return MultiscaleGramFamily(config, data_spec, normalizer)
    return COHERENCE_FAMILY_REGISTRY.build(
        normalized,
        config=config,
        data_spec=data_spec,
        normalizer=normalizer,
    )


_register_defaults()
