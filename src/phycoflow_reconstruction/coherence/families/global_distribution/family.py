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
from .components import (
    CrossJointCopulaCVaR,
    CrossJointTopKSWD,
    MutualPairwiseSWD,
    SelfMarginalW2,
)
from .components.point_masks import validate_point_mask


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
        self.definition = str(config.get("definition", "legacy_v1"))
        if self.definition not in {"legacy_v1", "marginal_copula_v2"}:
            raise ValueError(
                "global_distribution.definition must be legacy_v1 or marginal_copula_v2"
            )
        self.version = "2" if self.definition == "marginal_copula_v2" else "1"
        self.target_use = str(config.get("target_use", "training_reference"))
        self.units = str(config.get("units", "model_units"))
        self.family_weight = float(config.get("weight", 1.0))
        if self.family_weight <= 0:
            raise ValueError("global_distribution.weight must be positive")
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
        mutual_enabled_default = len(field_names) >= 2 and self.definition == "legacy_v1"
        if bool(mutual_settings.get("enabled", mutual_enabled_default)):
            configured_pairs = mutual_settings.get("pairs")
            name_pairs = configured_pairs or list(combinations(field_names, 2))
            normalized_pairs = [tuple(pair) for pair in name_pairs]
            if any(len(pair) != 2 for pair in normalized_pairs):
                raise ValueError("mutual field pairs must each contain exactly two fields")
            if any(left == right for left, right in normalized_pairs):
                raise ValueError("mutual field pairs must contain distinct fields")
            canonical_pairs = [frozenset(pair) for pair in normalized_pairs]
            if len(set(canonical_pairs)) != len(canonical_pairs):
                raise ValueError("mutual field pairs must be unique regardless of order")
            undeclared = sorted(
                {name for pair in normalized_pairs for name in pair} - set(field_names)
            )
            if undeclared:
                raise KeyError(
                    f"mutual field pairs contain fields outside global-distribution fields: {undeclared}"
                )
            pairs = []
            for left, right in normalized_pairs:
                pairs.append((lookup[left], lookup[right]))
            self.components_by_key["mutual_pairwise_swd"] = MutualPairwiseSWD(
                pairs,
                int(mutual_settings.get("directions", 16)),
                int(mutual_settings.get("seed", 1234)),
                target_use=self.target_use,
                units=self.units,
            )
            mutual_default_weight = 1.0 if self.definition == "legacy_v1" else 0.0
            self.component_weights["mutual_pairwise_swd"] = float(
                mutual_settings.get("weight", mutual_default_weight)
            )

        cross_settings = components.get("cross", {})
        if bool(cross_settings.get("enabled", len(field_names) >= 2)):
            if self.definition == "marginal_copula_v2":
                copula_settings = cross_settings.get("copula", {})
                tail_settings = cross_settings.get("tail", {})
                self.components_by_key["cross_joint_copula_cvar"] = (
                    CrossJointCopulaCVaR(
                        self.field_ids,
                        int(cross_settings.get("directions", 32)),
                        int(cross_settings.get("seed", 1234)),
                        include_axes=bool(cross_settings.get("include_axes", False)),
                        qmc=bool(cross_settings.get("qmc", True)),
                        canonicalizer=str(
                            copula_settings.get(
                                "canonicalizer", "quantile_landmark_smooth_cdf"
                            )
                        ),
                        landmarks=int(copula_settings.get("landmarks", 64)),
                        bandwidth=float(copula_settings.get("bandwidth", 0.1)),
                        scale_floor=float(copula_settings.get("scale_floor", 1e-6)),
                        chunk_size=int(copula_settings.get("chunk_size", 512)),
                        alpha=float(tail_settings.get("alpha", 0.25)),
                        rho=float(tail_settings.get("rho", 0.10)),
                        temperature=float(tail_settings.get("temperature", 1e-3)),
                        eta_tolerance=float(
                            tail_settings.get("eta_tolerance", 1e-6)
                        ),
                        eta_max_iterations=int(
                            tail_settings.get("eta_max_iterations", 80)
                        ),
                        pairwise_diagnostics=bool(
                            cross_settings.get("pairwise_diagnostics", False)
                        ),
                        target_use=self.target_use,
                        units=self.units,
                    )
                )
                self.component_weights["cross_joint_copula_cvar"] = float(
                    cross_settings.get("weight", 1.0)
                )
            else:
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
            metadata={
                "definition": self.definition,
                "target_use": self.target_use,
                "units": self.units,
            },
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
        point_mask: torch.Tensor | None = None,
    ) -> FamilyResult:
        require_field_tensor("generated", generated)
        require_field_tensor("reference", reference)
        if generated.shape != reference.shape:
            raise ValueError(
                f"generated/reference shapes differ: {tuple(generated.shape)} vs {tuple(reference.shape)}"
            )
        validate_point_mask(
            point_mask,
            batch_size=generated.shape[0],
            point_count=generated.shape[1],
        )
        generated, reference = self._in_declared_units(generated, reference)
        per_sample = generated.sum(dim=(1, 2)) * 0.0
        component_results: dict[str, TermResult] = {}
        component_diagnostics: dict[str, dict[str, Any]] = {}
        for key, component in self.components_by_key.items():
            weight = self.component_weights[key]
            path = f"{self.family_name}.{component.spec.name}"
            if weight == 0.0:
                component_diagnostics[path] = {
                    "weight": weight,
                    "executed": False,
                    "raw_scalar_loss": None,
                    "weighted_scalar_contribution": None,
                }
                continue
            runtime = context if isinstance(context, Mapping) else {}
            if isinstance(component, CrossJointCopulaCVaR) and runtime.get("execution_mode") == "r4_exact":
                result = component(generated, reference, point_mask=point_mask,
                                   batch_tail_roots=True, diagnostics=runtime.get("diagnostics", True))
            else:
                result = component(generated, reference, point_mask=point_mask)
            component_results[path] = result
            weighted_contribution = weight * result.scalar_loss
            component_diagnostics[path] = {
                "weight": weight,
                "executed": True,
                "raw_scalar_loss": float(result.scalar_loss.detach()),
                "weighted_scalar_contribution": float(weighted_contribution.detach()),
            }
            per_sample = per_sample + weight * result.per_sample_cost
        if not torch.isfinite(per_sample).all():
            raise FloatingPointError("global-distribution family produced a non-finite cost")
        return FamilyResult(
            component_results=component_results,
            per_sample_cost=per_sample,
            scalar_loss=per_sample.mean(),
            diagnostics={
                "family": self.family_name,
                "version": self.version,
                "definition": self.definition,
                "target_use": self.target_use,
                "units": self.units,
                "fields": self.field_names,
                "component_weights": dict(self.component_weights),
                "components": component_diagnostics,
            },
        )

    def state_artifact(self) -> dict[str, Any]:
        """Serializable fixed banks and scientific settings for run provenance."""
        return {
            "family": self.family_name,
            "version": self.version,
            "definition": self.definition,
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
        artifact_config = dict(artifact.get("config", {}))
        artifact_definition = str(
            artifact.get("definition", artifact_config.get("definition", "legacy_v1"))
        )
        artifact_config.setdefault("definition", artifact_definition)
        current_config = dict(self.config)
        current_config.setdefault("definition", self.definition)
        if artifact_definition != self.definition or artifact_config != current_config:
            raise ValueError("global-distribution family artifact config mismatch")
        self.load_state_dict(artifact["state_dict"], strict=True)
