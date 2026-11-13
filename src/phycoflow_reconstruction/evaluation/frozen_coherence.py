"""Reusable frozen, matched State/Scale/Shape comparison on supplied endpoints.

This observer evaluates an explicitly supplied panel. It performs no model
inference, data selection, fitting, sampling adaptation, or optimizer updates.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict

import torch

from ..coherence.query_runtime import ScalableQueryAdapter
from ..coherence.query_sampling import SharedQuerySampler
from ..coherence.state_compute import score_family as state_score
from ..training.scalable_fitting import _digest, validate_scalable_artifacts


class FrozenCoherenceObserver:
    """Fix the TRAIN observer and evaluation panel once for matched comparisons.

    Families must already contain their fitted TRAIN geometry and scales. The
    artifact provenance is checked against the caller's independently derived
    TRAIN identity. ``reference``, coordinates and endpoint occupation masses
    are fixed for every model; the caller selects the panel and supplies its IDs.
    ``mode`` is either ``deterministic`` (one member) or ``conditional`` (two
    independently generated members). Coefficients are common to every call;
    defaults are unit coefficients, independent of each model's training state.
    """

    def __init__(
        self,
        artifacts,
        families,
        *,
        expected_provenance,
        reference,
        coordinates,
        sample_ids: Sequence[str],
        member_weights,
        reference_weights,
        mode: str,
        query_seed: int = 0,
        query_batches: int = 8,
        coefficients: Mapping[str, float] | None = None,
    ):
        validate_scalable_artifacts(artifacts, expected_provenance)
        if mode not in {"deterministic", "conditional"}:
            raise ValueError("explicit deterministic or conditional scoring mode required")
        if (
            not isinstance(query_batches, int)
            or isinstance(query_batches, bool)
            or query_batches < 1
        ):
            raise ValueError("query_batches must be a positive integer")
        if not isinstance(reference, torch.Tensor) or reference.ndim != 3:
            raise ValueError("reference must be [B,N,C]")
        if (
            not isinstance(coordinates, torch.Tensor)
            or coordinates.ndim != 3
            or coordinates.shape[:2] != reference.shape[:2]
        ):
            raise ValueError("coordinates must align with the reference panel")
        ids = tuple(sample_ids)
        if (
            len(ids) != len(reference)
            or any(not isinstance(x, str) or not x for x in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("one unique evaluation sample ID per reference is required")
        fields = tuple(expected_provenance["field_order"])
        keys = ("global_distribution", "multiscale_spectrum", "topology")
        if set(families) != set(keys) or any(
            tuple(families[k].field_names) != fields for k in keys
        ):
            raise ValueError("all three fitted families must share the declared field order")
        if reference.shape[-1] != len(fields):
            raise ValueError("reference fields do not match the TRAIN observer")
        if not getattr(families["multiscale_spectrum"], "fitted", False) or not getattr(
            families["topology"], "geometry_sha256", None
        ):
            raise ValueError("fitted Scale and Shape geometry required")
        self.families = deepcopy(dict(families))
        self.artifacts = deepcopy(artifacts)
        self.reference = reference.detach().clone()
        self.coordinates = coordinates.detach().clone()
        self.member_weights = (
            torch.as_tensor(member_weights, device=reference.device, dtype=reference.dtype)
            .detach()
            .clone()
        )
        self.reference_weights = (
            torch.as_tensor(reference_weights, device=reference.device, dtype=reference.dtype)
            .detach()
            .clone()
        )
        # Reuse the family validation for masks, dimensions, and positive masses.
        self.context = {
            "generated_weights": self.member_weights,
            "reference_weights": self.reference_weights,
        }
        self.families["global_distribution"]._endpoint_measures(
            self.reference, self.reference, self.context, required=True
        )
        self.mode, self.fields = mode, fields
        self.adapter = ScalableQueryAdapter(self.families, self.artifacts["shape_artifact"])
        self.coefficients = {k: 1.0 for k in self.artifacts["active"]}
        if coefficients is not None:
            if set(coefficients) != set(self.coefficients):
                raise ValueError("common coefficients must cover all semantic blocks")
            self.coefficients = {k: float(v) for k, v in coefficients.items()}
        if any(not math.isfinite(v) or v <= 0 for v in self.coefficients.values()):
            raise ValueError("common coefficients must be finite and positive")
        sampler = SharedQuerySampler(fields, seed=int(query_seed), epsilon=1.0)
        batches = []
        for _ in range(query_batches):
            batches.append(sampler.draw())
            sampler.finish()
        self.batches = tuple(batches)
        families_identity = {
            k: {
                "config": dict(self.families[k].config),
                "state": self.families[k].state_dict(),
                "geometry": getattr(self.families[k], "geometry_sha256", None),
            }
            for k in keys
        }
        receipt = {
            "version": "frozen_coherence_observer_v1",
            "artifact_sha256": artifacts["sha256"],
            "train_provenance": deepcopy(dict(expected_provenance)),
            "sample_ids": ids,
            "mode": mode,
            "query_seed": int(query_seed),
            "queries": [asdict(q) for q in batches],
            "coefficients": dict(self.coefficients),
            "families_sha256": _digest(families_identity),
            "panel_sha256": _digest(
                {
                    "reference": self.reference,
                    "coordinates": self.coordinates,
                    "member_weights": self.member_weights,
                    "reference_weights": self.reference_weights,
                }
            ),
        }
        receipt["sha256"] = _digest(receipt)
        self.receipt = receipt
        self._frozen_sha256 = self._content_digest()

    def _content_digest(self):
        return _digest(
            {
                "artifact": self.artifacts,
                "receipt": self.receipt,
                "mode": self.mode,
                "fields": self.fields,
                "context": self.context,
                "reference": self.reference,
                "coordinates": self.coordinates,
                "member_weights": self.member_weights,
                "reference_weights": self.reference_weights,
                "coefficients": self.coefficients,
                "queries": [asdict(q) for q in self.batches],
                "shape": self.adapter.shape.state_dict(),
                "families": {
                    k: {
                        "state": v.state_dict(),
                        "config": dict(v.config),
                        "geometry": getattr(v, "geometry_sha256", None),
                    }
                    for k, v in self.families.items()
                },
            }
        )

    @torch.no_grad()
    def score(self, members: torch.Tensor):
        """Score [J,B,N,C] supplied members on the fixed panel and query pack."""
        if self._content_digest() != self._frozen_sha256:
            raise ValueError("frozen observer content changed")
        expected_draws = 1 if self.mode == "deterministic" else 2
        if not isinstance(members, torch.Tensor) or members.shape != (
            expected_draws,
            *self.reference.shape,
        ):
            raise ValueError("member tensor must match the fixed panel and declared draw count")
        if (
            members.device != self.reference.device
            or members.dtype != self.reference.dtype
            or not torch.isfinite(members).all()
        ):
            raise ValueError("members must be finite and match the panel device and dtype")
        family = self.families["global_distribution"]
        converted = [family._in_declared_units(x, self.reference)[0] for x in members]
        target = family._in_declared_units(members[0], self.reference)[1]
        left, right = family._endpoint_measures(
            members[0], self.reference, self.context, required=True
        )
        state, _ = state_score(
            torch.stack(converted),
            target,
            left,
            right,
            self.artifacts["state_bank"],
            channel_weights=family.components_by_key["self_marginal_w2"].channel_weights,
            marginal_weight=family.component_weights["self_marginal_w2"],
            mixed_weight=family.component_weights.get("cross_joint_topk_swd", 0.0),
            marginal_coefficient=self.coefficients["state.marginal"],
            mixed_coefficient=self.coefficients["state.mixed"],
            mode="paired" if self.mode == "deterministic" else "conditional",
        )
        context = {
            "scoring_mode": self.mode,
            "field_order": self.fields,
            "conditional_draws": tuple(members[1:]),
        }
        sums = {"scale": [], "shape": []}
        definitions = {"state": state.diagnostics["score_definition"]}
        for batch in self.batches:
            for label, name in (("scale", "multiscale_spectrum"), ("shape", "topology")):
                result, _, _ = self.adapter.score(
                    name,
                    members[0],
                    self.reference,
                    coordinates=self.coordinates,
                    context=context,
                    batch=batch,
                    coefficients=self.coefficients,
                )
                sums[label].append(result.per_sample_cost)
                definitions[label] = result.diagnostics["score_definition"]
        values = {
            "state": state.per_sample_cost,
            **{k: torch.stack(v).mean(0) for k, v in sums.items()},
        }
        return {
            "observer_sha256": self.receipt["sha256"],
            "sample_ids": self.receipt["sample_ids"],
            "score_definitions": definitions,
            "per_sample": {k: v.detach().cpu().clone() for k, v in values.items()},
            "means": {k: float(v.mean()) for k, v in values.items()},
        }
