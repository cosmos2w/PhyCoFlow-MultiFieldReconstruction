"""Topology calibration, regularized Adam, and projected-update compatibility.

RegularizedTopologyAdam is the current single-backward optimizer. The projected
controller remains available for explicitly selected historical configurations.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .gradient_balance import _assign_flat_gradient, _flat_gradient, _trainable_parameters
from .gradients import stable_clip_grad_norm_

CONSTRAINT_DEFAULTS = {
    "calibration_batches": 8,
    "scale_floor": 1e-4,
    "max_corrections": 2,
    "max_active_gradients": 12,
    "proposal_objective": "topology",
    "numerical_tolerance": 1e-5,
    "fidelity_numerical_tolerance": 1e-7,
    "minimum_relative_descent": 1e-6,
    "projection_margin": 0.1,
    "max_source_anchor_nmse": 0.01,
    "correction_selection": "all_violations",
    "topology_reduction": "per_sample",
    "fidelity_penalty_weight": 1.0,
    "anchor_soft_fraction": 0.5,
    "endpoint_soft_fraction": 0.9,
}


def constraint_settings(config):
    return {**CONSTRAINT_DEFAULTS, **config["optimization"].get("component_constraints", {})}


def leaf_components(result):
    return {
        name: term.scalar_loss
        for name, term in result.component_results.items()
        if name.startswith(("topology.self.", "topology.mutual."))
        and ".persistence" not in name
        and name.rsplit(".", 1)[-1] in {"h0", "h1"}
    }


def optimizer_source_snapshot():
    root = Path(__file__).parent
    return {
        name: (root / name).read_text()
        for name in (
            "topology_constraints.py",
            "post_training.py",
            "rollout.py",
            "checkpointing.py",
            "gradients.py",
            "gradient_balance.py",
            "retention.py",
            "model_lifecycle.py",
            "topology_selection.py",
            "run_store.py",
            "update_budget.py",
            "../data/topology_subset.py",
        )
    }


def scalar_values(values):
    """Copy related scalar tensors to the host in one device transfer."""
    names = [name for name, value in values.items() if isinstance(value, torch.Tensor)]
    result = {name: float(value) for name, value in values.items() if name not in names}
    if names:
        result.update(
            zip(names, torch.stack([values[name].detach() for name in names]).cpu().tolist())
        )
    return result


def optimizer_source_digest():
    return sha256("".join(optimizer_source_snapshot().values()).encode()).hexdigest()


def restore_constraint_audit(path, completed_step):
    """Drop uncheckpointed/replayed attempts after a preemption (atomic replace)."""
    path = Path(path)
    if not path.exists():
        return
    lines = path.read_text().splitlines()
    kept = []
    previous = 0
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break  # A preempted final write can be incomplete.
            raise
        step = int(row["step"])
        if step > completed_step:
            break
        if step <= previous:
            raise ValueError("constraint audit has repeated or unordered committed steps")
        kept.append(line)
        previous = step
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("\n".join(kept) + ("\n" if kept else ""))
    os.replace(temporary, path)


def sample_leaf_components(result):
    leaves = leaf_components(result)
    return {
        f"{name}#{i}": value
        for name in leaves
        for i, value in enumerate(result.component_results[name].per_sample_cost)
    }


def topology_constraint_components(result, reduction):
    """Keep component identities while choosing their sample aggregation.

    Batch means use the actual number of samples, including a short final batch.
    Fidelity constraints are constructed separately and remain per sample.
    """
    if reduction == "per_sample":
        return sample_leaf_components(result)
    if reduction == "batch_mean":
        return {
            name: result.component_results[name].per_sample_cost.mean()
            for name in leaf_components(result)
        }
    raise ValueError("topology_reduction must be per_sample or batch_mean")


def calibrated_score(components: Mapping, calibration: Mapping):
    if set(components) != set(calibration["scales"]):
        raise ValueError("topology calibration components differ from current objective")
    return sum(components[name] * calibration["coefficients"][name] for name in components)


def make_calibration(records, *, floor, weights, source_hash, objective_weight=1.0):
    names = set(records[0]["components"])
    if any(set(r["components"]) != names for r in records):
        raise ValueError("inconsistent calibration components")
    scales = {
        name: max(float(np.mean([r["components"][name] for r in records])), floor)
        for name in sorted(names)
    }
    if not all(np.isfinite(v) and v > 0 for v in scales.values()):
        raise ValueError("nonfinite topology component calibration")
    # Equal relative contributions within self and mutual, with the existing
    # explicit category weights. Preserve the source scalar's rough magnitude.
    magnitude = float(np.mean([r["raw_total"] for r in records])) / sum(weights.values())
    magnitude = max(magnitude, floor) * objective_weight
    coefficients = {
        name: magnitude
        * weights[name.split(".")[1] + ".persistence"]
        / (sum(n.split(".")[1] == name.split(".")[1] for n in names) * scales[name])
        for name in scales
    }
    return {
        "version": 1,
        "source_checkpoint_sha256": source_hash,
        "scale_floor": floor,
        "scales": scales,
        "coefficients": coefficients,
        "outer_coherence_weight": objective_weight,
        "calibration_batches": records,
        "units": "source_scaled_persistence",
        "optimizer_source_sha256": optimizer_source_digest(),
    }


def fidelity_components(prediction, reference, source_prediction, field_names, settings):
    """Return differentiable per-field NMSE and fixed same-noise source bounds."""
    variance = reference.detach().var(dim=1, unbiased=False).clamp_min(1e-8)
    endpoint = (prediction - reference.detach()).square().mean(1) / variance
    source = (source_prediction - reference.detach()).square().mean(1) / variance
    source_values = source.detach().cpu().tolist()
    anchor = (prediction - source_prediction).square().mean(1) / variance
    losses, bounds = {}, {}
    for sample in range(prediction.shape[0]):
        for i, field in enumerate(field_names):
            losses[f"fidelity.endpoint.{field}#{sample}"] = endpoint[sample, i]
            bounds[f"fidelity.endpoint.{field}#{sample}"] = source_values[sample][i] * (
                1 + settings["max_relative_field_mse_increase"]
            )
            losses[f"fidelity.anchor.{field}#{sample}"] = anchor[sample, i]
            bounds[f"fidelity.anchor.{field}#{sample}"] = settings["max_source_anchor_nmse"]
    return losses, bounds


def regularized_fidelity_loss(fidelity, bounds, settings):
    """Dimensionless soft fidelity penalties, averaged over actual samples/fields.

    These guide one stochastic gradient; they are not hard per-update guarantees.
    Endpoint pressure starts inside the source-relative error budget. Anchor
    pressure grows quadratically before the original anchor limit is reached.
    """
    endpoint = torch.stack(
        [
            value / max(bounds[name], 1e-8)
            for name, value in fidelity.items()
            if name.startswith("fidelity.endpoint.")
        ]
    )
    anchor = torch.stack(
        [
            value / bounds[name]
            for name, value in fidelity.items()
            if name.startswith("fidelity.anchor.")
        ]
    )
    endpoint_penalty = (endpoint - settings["endpoint_soft_fraction"]).relu().square().mean()
    anchor_penalty = (anchor / settings["anchor_soft_fraction"]).square().mean()
    return endpoint_penalty + anchor_penalty


class RegularizedTopologyAdam:
    """One ordinary Adam backward pass; fixed validation gates select checkpoints.

    There is no candidate model, repeated PH, projection, or per-update rejection.
    Accepted counters mean finite optimizer steps, not certified constraint passes.
    """

    def __init__(self, model, optimizer, settings, *, state=None):
        self.model, self.optimizer, self.settings = model, optimizer, dict(settings)
        self.parameters = _trainable_parameters(model)
        self.counts = dict(state or {"attempted": 0, "accepted": 0, "rejected": 0})
        if (
            set(self.counts) != {"attempted", "accepted", "rejected"}
            or any(not isinstance(v, int) or v < 0 for v in self.counts.values())
            or self.counts["attempted"] != self.counts["accepted"] + self.counts["rejected"]
        ):
            raise ValueError("invalid regularized optimizer counters")

    def state_dict(self):
        return dict(self.counts)

    def step(
        self,
        score,
        constraints,
        bounds,
        tolerances,
        evaluate,
        *,
        grad_clip=None,
        proposal_loss=None,
    ):
        started = perf_counter()
        if proposal_loss is None:
            raise ValueError("regularized topology requires an explicit combined objective")
        values = scalar_values(constraints)
        if not math.isfinite(float(proposal_loss.detach())) or not all(
            math.isfinite(v) for v in values.values()
        ):
            raise FloatingPointError("nonfinite regularized topology objective")
        self.optimizer.zero_grad(set_to_none=True)
        proposal_loss.backward()
        norm = stable_clip_grad_norm_(
            self.parameters, grad_clip if grad_clip is not None else 1e100
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.counts["attempted"] += 1
        self.counts["accepted"] += 1
        fidelity_names = [name for name in values if name.startswith("fidelity.")]
        return {
            "update_mode": "topology_regularized",
            "update_accepted": True,
            "acceptance_definition": "finite_optimizer_step_not_constraint_certification",
            "constraint_reason": "regularized_step_unchecked",
            "constraint_checks": 0,
            "constraint_active_gradients": 0,
            "backward_passes": 1,
            "balanced_topology_before": float(score.detach()),
            "regularized_loss_before": float(proposal_loss.detach()),
            "balanced_topology_candidate": None,
            "balanced_topology_committed": None,
            "constraint_seconds": perf_counter() - started,
            "constraint_baseline": values,
            "fidelity_violations_before": sum(
                values[n] > bounds[n] + tolerances[n] for n in fidelity_names
            ),
            "fidelity_max_bound_ratio_before": max(
                values[n] / max(bounds[n], 1e-8) for n in fidelity_names
            ),
            "combined_grad_norm": float(norm),
            "accepted_updates": self.counts["accepted"],
            "rejected_updates": self.counts["rejected"],
        }


def project_halfspaces(proposal, gradients, upper, *, sweeps=250):
    """Euclidean projection of the *actual* Adam displacement (Hildreth dual).

    Only the small Gram matrix goes to CPU; model-sized vectors stay on device.
    A failed/infeasible solve never authorizes an unchecked optimizer step.
    """
    matrix = torch.stack(gradients)
    norms = torch.linalg.vector_norm(matrix, dim=1)
    if not torch.isfinite(matrix).all() or bool((norms < 1e-20).any()):
        return None
    matrix = matrix / norms[:, None]
    bound = torch.as_tensor(upper, device=proposal.device, dtype=proposal.dtype) / norms
    gram = (matrix @ matrix.T).double().cpu().numpy()
    violation = (matrix @ proposal - bound).double().cpu().numpy()
    dual = np.zeros(len(gradients), dtype=np.float64)
    for _ in range(sweeps):
        previous = dual.copy()
        for i in range(len(dual)):
            dual[i] = max(0.0, dual[i] + (violation[i] - gram[i] @ dual) / max(gram[i, i], 1e-15))
        if np.max(np.abs(dual - previous)) < 1e-10 * max(1.0, np.max(np.abs(dual))):
            break
    projected = (
        proposal - torch.as_tensor(dual, device=proposal.device, dtype=proposal.dtype) @ matrix
    )
    residual = matrix @ projected - bound
    if not torch.isfinite(projected).all() or float(residual.max()) > max(
        1e-9, float(proposal.norm()) * 1e-5
    ):
        return None
    return projected


class ComponentConstrainedAdam:
    """A private candidate model keeps live parameters and autograd graphs intact.

    Rejection changes neither live Adam moments/counters nor model parameters.
    On acceptance we commit the projected parameters and tentative Adam moments.
    No weighted-sum fallback is used. Correction work is capped per attempt.
    """

    def __init__(self, model, optimizer, settings, *, state=None):
        if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1:
            raise ValueError("component constraints currently require single-group AdamW")
        self.model, self.optimizer, self.settings = model, optimizer, dict(settings)
        self.candidate = deepcopy(model).eval()
        self.parameters = _trainable_parameters(model)
        self.candidate_parameters = _trainable_parameters(self.candidate)
        if any(p.is_complex() for p in self.parameters):
            raise ValueError("component constraints currently require real parameters")
        self.candidate_optimizer = torch.optim.AdamW(
            self.candidate_parameters, lr=optimizer.param_groups[0]["lr"]
        )
        self.counts = dict(state or {"attempted": 0, "accepted": 0, "rejected": 0})
        if (
            set(self.counts) != {"attempted", "accepted", "rejected"}
            or any(not isinstance(v, int) or v < 0 for v in self.counts.values())
            or self.counts["attempted"] != self.counts["accepted"] + self.counts["rejected"]
        ):
            raise ValueError("invalid constrained optimizer counters")

    def state_dict(self):
        return dict(self.counts)

    @torch.no_grad()
    def _sync(self):
        for a, b in zip(self.candidate.parameters(), self.model.parameters()):
            a.copy_(b)
        for a, b in zip(self.candidate.buffers(), self.model.buffers()):
            a.copy_(b)
        # load_state_dict can share state tensor storage: deep copy is essential.
        self.candidate_optimizer.load_state_dict(deepcopy(self.optimizer.state_dict()))

    @torch.no_grad()
    def _set_displacement(self, displacement):
        offset = 0
        for candidate, live in zip(self.candidate_parameters, self.parameters):
            count = live.numel()
            candidate.copy_(live + displacement[offset : offset + count].view_as(live))
            offset += count

    def step(
        self,
        score,
        constraints,
        bounds,
        tolerances,
        evaluate,
        *,
        grad_clip=None,
        proposal_loss=None,
    ):
        started = perf_counter()
        if set(constraints) != set(bounds) or set(constraints) != set(tolerances):
            raise ValueError("constraint bounds/tolerances differ from component set")
        baseline_values = scalar_values(constraints)
        if (
            not np.isfinite(float(score.detach()))
            or not all(np.isfinite(v) for v in baseline_values.values())
            or not all(np.isfinite(v) for v in bounds.values())
            or not all(np.isfinite(v) and v >= 0 for v in tolerances.values())
        ):
            raise FloatingPointError("nonfinite baseline constraints or invalid tolerance")
        self._sync()
        self.candidate_optimizer.zero_grad(set_to_none=True)
        gradient = _flat_gradient(
            score if proposal_loss is None else proposal_loss, self.parameters
        )
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("nonfinite constrained topology gradient")
        _assign_flat_gradient(self.candidate_parameters, gradient)
        if grad_clip:
            stable_clip_grad_norm_(self.candidate_parameters, grad_clip)
        self.candidate_optimizer.step()
        proposal = torch.cat(
            [
                (p.detach() - q.detach()).flatten()
                for p, q in zip(self.candidate_parameters, self.parameters)
            ]
        )
        nominal_norm = float(proposal.norm())
        active, gradients, uppers = [], [], []
        accepted, reason, checks = False, "constraint_violation", 0
        baseline = float(score.detach())
        after_score = baseline
        max_violation = 0.0
        for correction in range(self.settings["max_corrections"] + 1):
            # Candidate inference must not advance training RNG. The callback
            # also restarts its explicit rollout generator at the same noise.
            devices = [proposal.device.index] if proposal.is_cuda else []
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                for candidate_buffer, live_buffer in zip(
                    self.candidate.buffers(), self.model.buffers()
                ):
                    candidate_buffer.copy_(live_buffer)
                values, after_score = evaluate(self.candidate)
            checks += 1
            if set(values) != set(constraints):
                raise ValueError("candidate constraint set differs from baseline")
            values = scalar_values(values)
            after_score = (
                float(after_score.detach())
                if isinstance(after_score, torch.Tensor)
                else float(after_score)
            )
            if not np.isfinite(after_score) or not all(np.isfinite(v) for v in values.values()):
                reason = "nonfinite_candidate"
                break
            violations = [
                name for name, value in values.items() if value > bounds[name] + tolerances[name]
            ]
            max_violation = max((values[n] - bounds[n]) / max(tolerances[n], 1e-16) for n in values)
            if not violations and after_score < baseline - self.settings[
                "minimum_relative_descent"
            ] * max(abs(baseline), 1e-12):
                accepted, reason = True, "accepted"
                break
            if not violations:
                reason = "no_topology_descent"
                break
            if correction == self.settings["max_corrections"]:
                break
            new_violations = [name for name in violations if name not in active]
            if not new_violations:
                # The same projection would reproduce the same rejected point.
                reason = "unchanged_linearization"
                break
            remaining = self.settings["max_active_gradients"] - len(active)
            if len(new_violations) > remaining:
                if (
                    remaining <= 0
                    or self.settings.get("correction_selection", "all_violations")
                    == "all_violations"
                ):
                    reason = "correction_gradient_budget"
                    break
                # A large batch need not be rejected before attempting correction.
                # Reserve part of the total gradient budget for the next round.
                # Every candidate still has to pass ALL original constraints.
                count = math.ceil(remaining / (self.settings["max_corrections"] - correction))
                new_violations = sorted(
                    new_violations,
                    key=lambda name: (values[name] - bounds[name]) / max(tolerances[name], 1e-16),
                    reverse=True,
                )[:count]
            for name in new_violations:
                active.append(name)
                g = _flat_gradient(constraints[name], self.parameters)
                gradients.append(g)
                # A small inward margin protects against nonlinear remainder
                # and FP32 rounding; feasibility is decided by actual evaluation.
                gap = bounds[name] - baseline_values[name]
                inward = self.settings["projection_margin"] * abs(float(g @ proposal))
                uppers.append(gap - inward)
            corrected = project_halfspaces(proposal, gradients, uppers)
            if corrected is None:
                reason = "infeasible_linearization"
                break
            # A bounded trust region; do not silently enlarge an Adam step.
            norm = float(corrected.norm())
            if norm > 2 * max(nominal_norm, 1e-20):
                reason = "correction_exceeds_trust_region"
                break
            self._set_displacement(corrected)
        self.counts["attempted"] += 1
        self.counts["accepted" if accepted else "rejected"] += 1
        if accepted:
            with torch.no_grad():
                for live, candidate in zip(self.parameters, self.candidate_parameters):
                    live.copy_(candidate)
            self.optimizer.load_state_dict(deepcopy(self.candidate_optimizer.state_dict()))
        self.optimizer.zero_grad(set_to_none=True)
        return {
            "update_mode": "component_constrained",
            "update_accepted": accepted,
            "constraint_reason": reason,
            "constraint_checks": checks,
            "constraint_active_gradients": len(active),
            "constraint_max_tolerance_units": max_violation,
            "balanced_topology_before": baseline,
            "balanced_topology_candidate": after_score if np.isfinite(after_score) else None,
            "balanced_topology_committed": after_score if accepted else baseline,
            "constraint_seconds": perf_counter() - started,
            "accepted_max_tolerance_units": max_violation if accepted else None,
            "active_constraints": active,
            "constraint_baseline": baseline_values,
            "constraint_bounds": bounds,
            "constraint_tolerances": tolerances,
            "constraint_candidate": {n: v if np.isfinite(v) else None for n, v in values.items()},
            "combined_grad_norm": float(gradient.norm()),
            "accepted_updates": self.counts["accepted"],
            "rejected_updates": self.counts["rejected"],
        }
