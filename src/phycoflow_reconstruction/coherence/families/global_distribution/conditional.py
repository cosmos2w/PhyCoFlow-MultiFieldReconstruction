"""Conditional fair energy score of the frozen weighted v3 State representation."""
from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

import torch

from ....contracts import CoherenceComponentSpec, CoherenceFamilySpec, FamilyResult, TermResult
from ...base import require_field_tensor
from .family import GlobalDistributionFamily


def _implementation_identity() -> dict[str, Any]:
    """Fingerprint the candidate, its v3 numerical dependencies and draw routing."""
    here = Path(__file__).resolve()
    coherence = here.parents[2]
    package = here.parents[3]
    files = {
        "conditional": here,
        "v3_family": here.with_name("family.py"),
        "self_marginal": here.parent / "components" / "self_marginal.py",
        "cross_joint": here.parent / "components" / "cross_joint.py",
        "numerical_base": coherence / "base.py",
        "measures": coherence / "measures.py",
        "registry": coherence / "registry.py",
        "draw_routing": package / "training" / "coherence_sampling.py",
    }
    return {"format": "conditional_state_sources_sha256_v1",
            "sources": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for name, path in files.items()}}


class ConditionalQuantileFairFamily(GlobalDistributionFamily):
    """Joint Hilbert-norm energy score; endpoint occupation W2 remains exact."""

    definition = "conditional_quantile_fair_v1"
    base_definition = "marginal_mixed_weighted_v3"

    def __init__(self, config, data_spec, normalizer):
        actual = copy.deepcopy(dict(config))
        if actual.get("base_definition", self.base_definition) != self.base_definition:
            raise ValueError("conditional State requires marginal_mixed_weighted_v3 base")
        draws = actual.get("conditional_draws", 2)
        if isinstance(draws, bool) or not isinstance(draws, int) or draws < 2:
            raise ValueError("conditional_draws must be an integer of at least 2")
        self.conditional_draws = draws
        base = copy.deepcopy(actual)
        base["definition"] = self.base_definition
        base.pop("base_definition", None)
        base.pop("conditional_draws", None)
        super().__init__(base, data_spec, normalizer)
        if self.component_weights["self_marginal_w2"] <= 0 or (
            len(self.field_names) >= 2 and self.component_weights["cross_joint_topk_swd"] <= 0
        ):
            raise ValueError("conditional State requires positive available component weights")
        self._base_config = base
        self.config = actual
        self.score_definition = self.definition
        self.spec = CoherenceFamilySpec(
            self.family_name, self.version,
            (CoherenceComponentSpec("conditional_quantile_fair", self.target_use, self.units, True),),
            metadata={"target_use": self.target_use, "units": self.units,
                      "base_definition": self.base_definition},
        )
        self.spec.validate()

    def paired_squared(self, generated, reference, *, context: Mapping[str, Any]):
        """Return the exact v3 squared occupation distance and component leaves."""
        return super().forward(generated, reference, context=context)

    def load_base_artifact(self, artifact: Mapping[str, Any]) -> None:
        """Bind a v3 artifact only when its complete base configuration matches."""
        if artifact.get("score_definition") != self.base_definition or dict(artifact.get("config", {})) != self._base_config:
            raise ValueError("conditional State base artifact configuration mismatch")
        converted = dict(artifact)
        converted["config"] = self.config
        converted["score_definition"] = self.definition
        converted["base_definition"] = self.base_definition
        converted["conditional_draws"] = self.conditional_draws
        converted["implementation_identity"] = _implementation_identity()
        self.load_state_artifact(converted)

    def state_artifact(self):
        artifact = super().state_artifact()
        artifact["base_definition"] = self.base_definition
        artifact["conditional_draws"] = self.conditional_draws
        artifact["implementation_identity"] = _implementation_identity()
        return artifact

    def load_state_artifact(self, artifact):
        if (artifact.get("score_definition") != self.definition
                or artifact.get("base_definition") != self.base_definition
                or artifact.get("conditional_draws") != self.conditional_draws
                or artifact.get("implementation_identity") != _implementation_identity()):
            raise ValueError("conditional State artifact identity mismatch")
        super().load_state_artifact(artifact)

    def forward(self, generated, reference, *, coordinates=None, context=None):
        require_field_tensor("generated", generated)
        require_field_tensor("reference", reference)
        if not isinstance(context, Mapping):
            raise TypeError("conditional State context must be a mapping")
        extras = context.get("conditional_draws", ())
        if not isinstance(extras, (tuple, list)):
            raise TypeError("conditional_draws must be a tuple or list of extra members")
        if len(extras) < self.conditional_draws - 1:
            raise ValueError("conditional State requires all configured draws")
        if not context.get("score_all_conditional_draws", False):
            extras = extras[:self.conditional_draws - 1]
        members = (generated, *extras)
        if any(not isinstance(x, torch.Tensor) or x.shape != generated.shape or x.device != generated.device or x.dtype != generated.dtype for x in members):
            raise ValueError("conditional members must match generated shape, device and dtype")
        for member in members:
            require_field_tensor("conditional member", member)

        pair_context = dict(context)
        pair_context.pop("include_pair_occupations", None)
        reference_pairs = [self.paired_squared(member, reference, context=pair_context) for member in members]
        # All generated members share the supplied generated support and measure.
        member_context = dict(pair_context)
        member_context["reference_weights"] = pair_context.get("generated_weights")
        member_context["reference_mask"] = pair_context.get("generated_mask")
        separation_pairs = [self.paired_squared(members[i], members[j], context=member_context)
                            for i, j in combinations(range(len(members)), 2)]

        def distance(square):
            # Exact sqrt, with an explicitly zero derivative at coincident endpoints.
            positive = square > 0
            return torch.where(positive, torch.sqrt(torch.where(positive, square, torch.ones_like(square))), torch.zeros_like(square))

        j = len(members)
        attraction = torch.stack([distance(r.per_sample_cost) for r in reference_pairs]).mean(0)
        separation = torch.stack([distance(r.per_sample_cost) for r in separation_pairs]).sum(0) / (j * (j - 1))
        score = attraction - separation
        result = TermResult(score, score.mean())
        key = f"{self.family_name}.conditional_quantile_fair"
        old = reference_pairs[0]
        components = {key: result, **old.component_results}
        for name, values in (("attraction", attraction), ("separation", separation),
                             ("paired_squared", old.per_sample_cost)):
            components[f"{self.family_name}.{name}"] = TermResult(
                values, values.mean(), diagnostics={"evaluation_only": True})
        component_diagnostics = {key: {"weight": 1.0, "executed": True,
                                         "raw_scalar_loss": float(score.mean().detach()),
                                         "weighted_scalar_contribution": float(score.mean().detach())}}
        for name, term in components.items():
            if name == key:
                continue
            component_diagnostics[name] = {"weight": 0.0, "executed": True,
                                           "raw_scalar_loss": float(term.scalar_loss.detach()),
                                           "weighted_scalar_contribution": 0.0}
        return FamilyResult(components, score, score.mean(), {
            "family": self.family_name, "version": self.version,
            "score_definition": self.definition, "base_definition": self.base_definition,
            "conditional_draws": j, "target_use": self.target_use, "units": self.units,
            "fields": self.field_names, "component_weights": {name: (1.0 if name == key else 0.0) for name in components},
            "components": component_diagnostics, "attraction": attraction.detach(),
            "separation": separation.detach(), "paired_squared": old.per_sample_cost.detach(),
            "measure": old.diagnostics.get("measure"),
        })
