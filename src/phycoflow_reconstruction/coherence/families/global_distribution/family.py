"""Composable global-distribution coherence family.

The family owns denormalization, field-name resolution, fixed projection banks,
component weighting, and namespaced diagnostics. Reference selection and point
subsampling remain explicit responsibilities of the post-training context.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from itertools import combinations
from typing import Any

import torch
from torch import nn

from ....contracts import CoherenceFamilySpec, DataSpec, FamilyResult, TermResult
from ....data.normalization import FieldNormalizer
from ...base import require_field_tensor
from ...measures import measure_fingerprint, normalize_measure_weights
from .components import CrossJointTopKSWD, MutualPairwiseSWD, SelfMarginalW2
from .diagnostics import named_pair_occupations


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
        self.score_definition = str(config.get("definition", "global_distribution_v1"))
        if self.score_definition not in {
            "global_distribution_v1",
            "marginal_mixed_mean_v2",
            "marginal_mixed_weighted_v3",
        }:
            raise ValueError("unknown global_distribution.definition")
        self.weighted_measure = self.score_definition == "marginal_mixed_weighted_v3"
        self._measure_artifact: dict[str, Any] | None = None
        self._measure_support: dict[str, torch.Tensor | None] | None = None
        self.target_use = str(config.get("target_use", "training_reference"))
        self.units = str(config.get("units", "model_units"))
        self.family_weight = float(config.get("weight", 1.0))
        if self.family_weight <= 0:
            raise ValueError("global_distribution.weight must be positive")
        if self.target_use not in {"training_reference", "paired_supervised"}:
            raise ValueError(
                "global_distribution.target_use must be training_reference or paired_supervised"
            )
        if self.weighted_measure and self.target_use != "paired_supervised":
            raise ValueError(
                "marginal_mixed_weighted_v3 requires paired_supervised; reference banks do not provide endpoint masses"
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
            self.component_weights["mutual_pairwise_swd"] = float(
                mutual_settings.get("weight", 1.0)
            )

        cross_settings = components.get("cross", {})
        mean_bank = self.score_definition in {
            "marginal_mixed_mean_v2",
            "marginal_mixed_weighted_v3",
        }
        if mean_bank and (
            not bool(self_settings.get("enabled", True))
            or bool(mutual_settings.get("enabled", len(field_names) >= 2))
            or float(cross_settings.get("top_fraction", 0.1)) != 1.0
            or bool(cross_settings.get("include_axes", True))
            or not bool(cross_settings.get("qmc", True))
        ):
            raise ValueError(
                f"{self.score_definition} requires marginals, no mutual, and a full mixed QMC mean bank"
            )
        if (
            self.weighted_measure
            and len(field_names) >= 2
            and not bool(cross_settings.get("enabled", True))
        ):
            raise ValueError("marginal_mixed_weighted_v3 requires the mixed projection bank")
        if bool(cross_settings.get("enabled", len(field_names) >= 2)) and (
            not mean_bank or len(field_names) >= 2
        ):
            self.components_by_key["cross_joint_topk_swd"] = CrossJointTopKSWD(
                self.field_ids,
                int(cross_settings.get("directions", 32)),
                float(cross_settings.get("top_fraction", 0.1)),
                int(cross_settings.get("seed", 1234)),
                bool(cross_settings.get("include_axes", True)),
                bool(cross_settings.get("qmc", True)),
                target_use=self.target_use,
                units=self.units,
                mean_reduction=mean_bank,
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

    def _endpoint_measures(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
        context: Mapping[str, Any],
        *,
        required: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left, right = context.get("generated_weights"), context.get("reference_weights")
        if left is None and right is None and not required:
            left = generated.new_ones(generated.shape[:2])
            right = reference.new_ones(reference.shape[:2])
        if left is None or right is None:
            raise ValueError("weighted State requires generated_weights and reference_weights")
        return (
            normalize_measure_weights(
                left,
                context.get("generated_mask"),
                batch_size=generated.shape[0],
                point_count=generated.shape[1],
                device=generated.device,
                dtype=generated.dtype,
            ),
            normalize_measure_weights(
                right,
                context.get("reference_mask"),
                batch_size=reference.shape[0],
                point_count=reference.shape[1],
                device=reference.device,
                dtype=reference.dtype,
            ),
        )

    def pair_diagnostics(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
        *,
        context: Mapping[str, Any] | None = None,
        pairs: Any | None = None,
        bins: int = 16,
    ) -> dict[str, dict]:
        """Return raw and own-rank occupations for named pairs, with no training loss."""
        require_field_tensor("generated", generated)
        require_field_tensor("reference", reference)
        if generated.shape != reference.shape:
            raise ValueError("pair diagnostics require matching endpoint shapes")
        context = context or {}
        left_weights, right_weights = self._endpoint_measures(
            generated,
            reference,
            context,
            required=self.weighted_measure,
        )
        lookup = dict(zip(self.field_names, self.field_ids))
        selected = list(combinations(self.field_names, 2)) if pairs is None else list(pairs)
        resolved = []
        for pair in selected:
            if len(pair) != 2 or pair[0] == pair[1] or any(name not in lookup for name in pair):
                raise ValueError("diagnostic pairs must name two distinct declared fields")
            resolved.append((pair[0], pair[1], lookup[pair[0]], lookup[pair[1]]))
        generated, reference = self._in_declared_units(generated, reference)
        return named_pair_occupations(
            generated, reference, left_weights, right_weights, resolved, bins=bins
        )

    def bind_measure_artifact(
        self,
        receipt: Mapping[str, Any],
        *,
        weights: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        point_indices: torch.Tensor | None = None,
    ) -> None:
        """Freeze a native-measure receipt and optional actual support tensors."""
        if not self.weighted_measure:
            raise ValueError("native measure artifacts apply to weighted State only")
        candidate = copy.deepcopy(dict(receipt))
        digest = candidate.get("sha256")
        if (
            candidate.get("version") != "physical_measure_v1"
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError("invalid native measure fingerprint receipt")
        if self._measure_artifact is not None and candidate != self._measure_artifact:
            raise ValueError(
                "native State measure artifact is already frozen to a different receipt"
            )
        if weights is None and (mask is not None or point_indices is not None):
            raise ValueError("measure support mask and point indices require weights")
        if weights is not None:
            weights = torch.as_tensor(weights).detach().cpu().clone()
            mask = None if mask is None else torch.as_tensor(mask).detach().cpu().clone()
            point_indices = (
                None
                if point_indices is None
                else torch.as_tensor(point_indices).detach().cpu().clone()
            )
            verified = measure_fingerprint(
                weights,
                mask=mask,
                point_indices=point_indices,
                metadata=candidate.get("metadata"),
            )
            if verified != candidate:
                raise ValueError("native measure support does not match its fingerprint receipt")
            self._measure_support = {
                "weights": weights,
                "mask": mask,
                "point_indices": point_indices,
            }
        self._measure_artifact = candidate

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
        if context is not None and not isinstance(context, Mapping):
            if self.weighted_measure:
                raise TypeError("weighted State context must be a mapping")
            # Historical State ignored arbitrary context objects.
            context = None
        context = context or {}
        measure_metadata = context.get("measure_metadata", {})
        if self.weighted_measure and not isinstance(measure_metadata, Mapping):
            raise TypeError("measure_metadata must be a mapping")
        if self.weighted_measure:
            generated_weights, reference_weights = self._endpoint_measures(
                generated,
                reference,
                context,
                required=True,
            )
            claimed_digest = measure_metadata.get("native_measure_sha256")
            if (
                self._measure_artifact is not None
                and claimed_digest is not None
                and claimed_digest != self._measure_artifact["sha256"]
            ):
                raise ValueError("runtime State measure differs from the frozen native measure")
        else:
            generated_weights = reference_weights = None
        pair_diagnostics = (
            self.pair_diagnostics(
                generated,
                reference,
                context=context,
                pairs=context.get("diagnostic_pairs"),
                bins=int(context.get("diagnostic_bins", 16)),
            )
            if bool(context.get("include_pair_occupations", False))
            else None
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
            if self.weighted_measure:
                result = component(
                    generated,
                    reference,
                    generated_weights=generated_weights,
                    reference_weights=reference_weights,
                )
            else:
                result = component(generated, reference)
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
        diagnostics = {
            "family": self.family_name,
            "version": self.version,
            "score_definition": self.score_definition,
            "target_use": self.target_use,
            "units": self.units,
            "fields": self.field_names,
            "component_weights": dict(self.component_weights),
            "components": component_diagnostics,
        }
        if self.weighted_measure:
            diagnostics["measure"] = {
                "policy": "fixed_separate_endpoint_weights",
                "tie_handling": "stable_value_sort_piecewise_value_gradients",
                "metadata": dict(measure_metadata),
                "native_artifact": copy.deepcopy(self._measure_artifact),
            }
        if pair_diagnostics is not None:
            diagnostics["pair_occupations"] = pair_diagnostics
        return FamilyResult(
            component_results=component_results,
            per_sample_cost=per_sample,
            scalar_loss=per_sample.mean(),
            diagnostics=diagnostics,
        )

    def state_artifact(self) -> dict[str, Any]:
        """Serializable fixed banks and scientific settings for run provenance."""
        artifact = {
            "family": self.family_name,
            "version": self.version,
            "score_definition": self.score_definition,
            "config": self.config,
            "field_names": self.field_names,
            "field_ids": self.field_ids,
            "target_use": self.target_use,
            "units": self.units,
            "state_dict": self.state_dict(),
        }
        if self.weighted_measure:
            artifact["measure"] = copy.deepcopy(self._measure_artifact)
            artifact["measure_support"] = copy.deepcopy(self._measure_support)
        return artifact

    def load_state_artifact(self, artifact: Mapping[str, Any]) -> None:
        if artifact.get("family") != self.family_name or artifact.get("version") != self.version:
            raise ValueError("global-distribution family artifact identity mismatch")
        if dict(artifact.get("config", {})) != self.config:
            raise ValueError("global-distribution family artifact config mismatch")
        if self.weighted_measure:
            support = artifact.get("measure_support")
            if artifact.get("measure") is None or support is None:
                raise ValueError(
                    "weighted State artifact requires its frozen native measure and support"
                )
            self.bind_measure_artifact(artifact["measure"], **support)
        self.load_state_dict(artifact["state_dict"], strict=True)
