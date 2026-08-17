"""Composable global-distribution coherence family.

The family owns denormalization, field-name resolution, fixed projection banks,
component weighting, and namespaced diagnostics. Reference selection and point
subsampling remain explicit responsibilities of the post-training context.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from typing import Any

import torch
from torch import nn

from ....contracts import CoherenceFamilySpec, DataSpec, FamilyResult, TermResult
from ....data.normalization import FieldNormalizer
from ...base import require_field_tensor
from .components import CrossJointTopKSWD, MutualPairwiseSWD, SelfMarginalW2


class GlobalDistributionFamily(nn.Module):
    """Global empirical-distribution coherence with three nested components."""

    family_name = "global_distribution"
    version = "1"

    def __init__(
        self,
        config: Mapping[str, Any],
        data_spec: DataSpec,
        normalizer: FieldNormalizer,
    ) -> None:
        super().__init__()
        self.config = dict(config)
        self.target_use = str(config.get("target_use", "training_reference"))
        self.units = str(config.get("units", "model_units"))
        self.family_weight = float(config.get("weight", 1.0))
        if self.family_weight < 0:
            raise ValueError("global_distribution.weight must be non-negative")
        if self.target_use not in {"training_reference", "paired_supervised"}:
            raise ValueError(
                "global_distribution.target_use must be training_reference or paired_supervised"
            )
        if self.units not in {"model_units", "physical_units"}:
            raise ValueError("global_distribution.units must be model_units or physical_units")

        field_names = tuple(config.get("fields") or data_spec.field_names)
        if len(set(field_names)) != len(field_names):
            raise ValueError("global-distribution fields must be unique")
        lookup = {name: index for index, name in enumerate(data_spec.field_names)}
        unknown = sorted(set(field_names) - set(lookup))
        if unknown:
            raise KeyError(f"unknown global-distribution fields: {unknown}")
        self.field_names = field_names
        self.field_ids = tuple(lookup[name] for name in field_names)
        self.register_buffer("normalization_offset", normalizer.offset.clone())
        self.register_buffer("normalization_scale", normalizer.scale.clone())

        components = config.get("components", {})
        self.component_weights: dict[str, float] = {}
        self.components_by_key = nn.ModuleDict()

        self_settings = components.get("self", {})
        if bool(self_settings.get("enabled", True)):
            weights = self_settings.get("channel_weights")
            if isinstance(weights, Mapping):
                weights = [float(weights[name]) for name in field_names]
            self.components_by_key["self_marginal_w2"] = SelfMarginalW2(
                self.field_ids,
                weights,
                target_use=self.target_use,
                units=self.units,
            )
            self.component_weights["self_marginal_w2"] = float(self_settings.get("weight", 1.0))

        mutual_settings = components.get("mutual", {})
        if bool(mutual_settings.get("enabled", len(field_names) >= 2)):
            configured_pairs = mutual_settings.get("pairs")
            name_pairs = configured_pairs or list(combinations(field_names, 2))
            pairs = []
            for left, right in name_pairs:
                if left not in lookup or right not in lookup:
                    raise KeyError(f"unknown mutual field pair {(left, right)}")
                pairs.append((lookup[left], lookup[right]))
            self.components_by_key["mutual_pairwise_swd"] = MutualPairwiseSWD(
                pairs,
                int(mutual_settings.get("directions", 16)),
                int(mutual_settings.get("seed", 1234)),
                target_use=self.target_use,
                units=self.units,
            )
            self.component_weights["mutual_pairwise_swd"] = float(
                mutual_settings.get("weight", 1.0)
            )

        cross_settings = components.get("cross", {})
        if bool(cross_settings.get("enabled", len(field_names) >= 2)):
            self.components_by_key["cross_joint_topk_swd"] = CrossJointTopKSWD(
                self.field_ids,
                int(cross_settings.get("directions", 32)),
                float(cross_settings.get("top_fraction", 0.1)),
                int(cross_settings.get("seed", 1234)),
                bool(cross_settings.get("include_axes", True)),
                bool(cross_settings.get("qmc", True)),
                target_use=self.target_use,
                units=self.units,
            )
            self.component_weights["cross_joint_topk_swd"] = float(
                cross_settings.get("weight", 1.0)
            )

        if not self.components_by_key:
            raise ValueError("global_distribution must enable at least one component")
        if any(weight < 0 for weight in self.component_weights.values()):
            raise ValueError("global-distribution component weights must be non-negative")
        if not any(weight > 0 for weight in self.component_weights.values()):
            raise ValueError("at least one global-distribution component weight must be positive")
        self.spec = CoherenceFamilySpec(
            self.family_name,
            self.version,
            tuple(component.spec for component in self.components_by_key.values()),
            metadata={"target_use": self.target_use, "units": self.units},
        )
        self.spec.validate()

    def _in_declared_units(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.units == "model_units":
            return generated, reference
        offset = self.normalization_offset.to(device=generated.device, dtype=generated.dtype)
        scale = self.normalization_scale.to(device=generated.device, dtype=generated.dtype)
        return generated * scale + offset, reference * scale + offset

    def forward(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
        *,
        coordinates: torch.Tensor | None = None,
        context: Any | None = None,
    ) -> FamilyResult:
        require_field_tensor("generated", generated)
        require_field_tensor("reference", reference)
        if generated.shape != reference.shape:
            raise ValueError(
                f"generated/reference shapes differ: {tuple(generated.shape)} vs {tuple(reference.shape)}"
            )
        generated, reference = self._in_declared_units(generated, reference)
        per_sample = generated.sum(dim=(1, 2)) * 0.0
        component_results: dict[str, TermResult] = {}
        for key, component in self.components_by_key.items():
            result = component(generated, reference)
            path = f"{self.family_name}.{component.spec.name}"
            component_results[path] = result
            per_sample = per_sample + self.component_weights[key] * result.per_sample_cost
        if not torch.isfinite(per_sample).all():
            raise FloatingPointError("global-distribution family produced a non-finite cost")
        return FamilyResult(
            component_results=component_results,
            per_sample_cost=per_sample,
            scalar_loss=per_sample.mean(),
            diagnostics={
                "family": self.family_name,
                "version": self.version,
                "target_use": self.target_use,
                "units": self.units,
                "fields": self.field_names,
                "component_weights": dict(self.component_weights),
            },
        )

    def state_artifact(self) -> dict[str, Any]:
        """Serializable fixed banks and scientific settings for run provenance."""
        return {
            "family": self.family_name,
            "version": self.version,
            "config": self.config,
            "field_names": self.field_names,
            "field_ids": self.field_ids,
            "target_use": self.target_use,
            "units": self.units,
            "state_dict": self.state_dict(),
        }

    def load_state_artifact(self, artifact: Mapping[str, Any]) -> None:
        if artifact.get("family") != self.family_name or artifact.get("version") != self.version:
            raise ValueError("global-distribution family artifact identity mismatch")
        if dict(artifact.get("config", {})) != self.config:
            raise ValueError("global-distribution family artifact config mismatch")
        self.load_state_dict(artifact["state_dict"], strict=True)
