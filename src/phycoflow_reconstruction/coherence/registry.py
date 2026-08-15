"""Coherence-family construction and reserved extension declarations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts import DataSpec
from ..data.normalization import FieldNormalizer
from ..registry import COHERENCE_FAMILY_REGISTRY
from .families.global_distribution import GlobalDistributionFamily

RESERVED_FAMILIES = {
    "cross_spectrum": "Scientific definitions pending co-worker contribution.",
    "topology": "Scientific definitions pending co-worker contribution.",
}


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


def build_coherence_family(
    name: str,
    config: Mapping[str, Any],
    data_spec: DataSpec,
    normalizer: FieldNormalizer,
):
    _register_defaults()
    normalized = str(name).strip().lower()
    if normalized in RESERVED_FAMILIES:
        raise NotImplementedError(
            f"coherence family {normalized!r} is reserved: {RESERVED_FAMILIES[normalized]}"
        )
    return COHERENCE_FAMILY_REGISTRY.build(
        normalized,
        config=config,
        data_spec=data_spec,
        normalizer=normalizer,
    )


_register_defaults()
