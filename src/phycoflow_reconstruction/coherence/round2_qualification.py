"""Measured fixed-target allocation and task-conditioned query selection.

These helpers consume actual parameter gradients and measured execution costs.
Endpoint gradients or reduced condition panels cannot earn a native policy claim.
Complete nonlinear anchors and enumerated extras belong to the fixed cost term.
"""
from __future__ import annotations

import copy
import math
import time

import torch

from .adaptive_fitting import _digest, _geometry


def importance_mse(gram, weights, probabilities, draws):
    """Exact uncentered MSE of an IID with-replacement gradient estimator."""
    gram = torch.as_tensor(gram, dtype=torch.float64)
    w = torch.as_tensor(weights, dtype=torch.float64)
    p = torch.as_tensor(probabilities, dtype=torch.float64)
    if (gram.shape != (len(w), len(w)) or p.shape != w.shape
        or draws < 1 or not torch.isfinite(gram).all()
        or not torch.isfinite(w).all() or (w <= 0).any()
        or not torch.isfinite(p).all() or (p <= 0).any()
        or not torch.isclose(p.sum(), p.new_tensor(1.0))):
        raise ValueError("finite full gradient Gram, positive target/proposal and draws required")
    if not torch.allclose(gram, gram.T, atol=1e-10, rtol=1e-10):
        raise ValueError("gradient Gram must be symmetric")
    value = ((w.square() * gram.diag() / p).sum() - w @ gram @ w) / draws
    tolerance = 1e-10 * max(1., float(gram.diag().abs().max()))
    if float(value) < -tolerance:
        raise ValueError("invalid gradient Gram produces negative estimator MSE")
    return max(0., float(value))


def second_moment_proposal(weights, moments, costs, *, uniform_fraction=.25):
    w, v, c = [torch.as_tensor(x, dtype=torch.float64).detach() for x in (weights, moments, costs)]
    if (w.ndim != 1 or not len(w) or v.shape != w.shape or c.shape != w.shape
        or any(not torch.isfinite(x).all() for x in (w, v, c))
        or (w <= 0).any() or (v < 0).any() or (c <= 0).any()
        or not .25 <= uniform_fraction <= 1):
        raise ValueError("positive weights/costs, nonnegative uncentered moments and >=25% exploration required")
    priority = w * (v / c).sqrt()
    if priority.sum() == 0:
        return torch.full_like(w, 1 / len(w))
    return uniform_fraction / len(w) + (1 - uniform_fraction) * priority / priority.sum()


def full_parameter_leaf_gram(leaves, parameters):
    """Differentiate each complete additive leaf over ALL trainable parameters."""
    parameters = tuple(p for p in parameters if p.requires_grad)
    if not parameters or not leaves:
        raise ValueError("nonempty actual trainable parameters and additive leaf graphs required")
    gradients = []
    for leaf in leaves:
        if leaf.ndim != 0 or not torch.isfinite(leaf):
            raise ValueError("each complete leaf must be a finite batch-mean scalar")
        values = torch.autograd.grad(leaf, parameters, retain_graph=True, allow_unused=True)
        vector = torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                            for p, g in zip(parameters, values)]).double().cpu()
        if not torch.isfinite(vector).all():
            raise ValueError("nonfinite actual parameter gradient")
        gradients.append(vector)
    matrix = torch.stack(gradients)
    return matrix @ matrix.T, matrix.shape[1]


def audit_parameter_panel(panel_factory, *, sample_ids, population, weights,
                          c8_probabilities, draws, source_family, parameterization,
                          preparation_seconds, training_seconds, replicates=128, seed=2047):
    """Audit a frozen proposal on a streamed matched ~10% TRAIN panel.

    Each fresh panel batch supplies ids, actual parameter-leaf Gram, leaf_seconds,
    fixed_seconds (model/anchors/enumerated leaves), and parameter_count. The
    callable runs a second pass at identical parameters, optimizer/RNG state and
    endpoints after fitting a detached proposal. Costs are synchronized measured
    seconds per *estimator*, including all fixed work, never score-only timing.
    """
    if (source_family not in {"ffm", "senseiver"} or parameterization != "residual_head"
        or len(sample_ids) != max(1, math.ceil(.1 * population))
        or len(set(sample_ids)) != len(sample_ids) or replicates < 2
        or preparation_seconds < 0 or training_seconds <= 0):
        raise ValueError("actual residual-head ~10% TRAIN audit and complete charged timers required")
    started = time.perf_counter()
    w = torch.as_tensor(weights, dtype=torch.float64)
    previous = torch.as_tensor(c8_probabilities, dtype=torch.float64)
    if previous.shape != w.shape:
        raise ValueError("C8 proposal and target shapes differ")
    moments, costs, seen = torch.zeros_like(w), torch.zeros_like(w), []
    parameter_count = None
    for row in panel_factory():
        gram = torch.as_tensor(row["gram"], dtype=torch.float64)
        fraction = len(row["ids"]) / len(sample_ids)
        moments += fraction * gram.diag()
        costs += fraction * torch.as_tensor(row["leaf_seconds"], dtype=torch.float64)
        seen.extend(row["ids"])
        if parameter_count is not None and parameter_count != row["parameter_count"]:
            raise ValueError("trainable parameter space changed during audit")
        parameter_count = row["parameter_count"]
    if seen != list(sample_ids):
        raise ValueError("proposal fit did not cover the declared matched TRAIN panel")
    probabilities = {
        "uniform": torch.full_like(w, 1 / len(w)),
        "c8_priorities": previous,
        "lagged_gradient_cost": second_moment_proposal(w, moments, costs),
    }
    totals = {name: {"mse": 0., "full_seconds": 0., "mse_cost": 0., "replicates": [0.] * replicates}
              for name in probabilities}
    seen = []
    rng = torch.Generator().manual_seed(seed)
    for row in panel_factory():
        seen.extend(row["ids"])
        fraction = len(row["ids"]) / len(sample_ids)
        gram = torch.as_tensor(row["gram"], dtype=torch.float64)
        if row["fixed_seconds"] < 0:
            raise ValueError("complete nonnegative model/anchor/enumeration cost required")
        for name, proposal in probabilities.items():
            mse = importance_mse(gram, w, proposal, draws)
            seconds = float(row["fixed_seconds"]) + draws * float(proposal @ torch.as_tensor(row["leaf_seconds"], dtype=torch.float64))
            if seconds <= 0:
                raise ValueError("positive full charged estimator cost required")
            item = totals[name]
            item["mse"] += fraction * mse
            item["full_seconds"] += fraction * seconds
            item["mse_cost"] += fraction * mse * seconds
            selections = torch.multinomial(proposal, draws * replicates, replacement=True, generator=rng).view(replicates, draws)
            for k, selection in enumerate(selections):
                counts = torch.bincount(selection, minlength=len(w)).double()
                error_weights = counts * w / (draws * proposal) - w
                error = max(0., float(error_weights @ gram @ error_weights))
                item["replicates"][k] += fraction * error * seconds
    if seen != list(sample_ids):
        raise ValueError("parameter audit did not cover the declared matched TRAIN panel")
    elapsed = time.perf_counter() - started
    overhead = (preparation_seconds + elapsed) / training_seconds
    baseline = totals["uniform"]["mse_cost"]
    candidate = totals["lagged_gradient_cost"]["mse_cost"]
    ratio = candidate / baseline if baseline > 0 else None
    qualified = ratio is not None and ratio <= .8 and overhead <= .05
    for item in totals.values():
        values = torch.tensor(item.pop("replicates"), dtype=torch.float64)
        item["descriptive_replicate_p05_p95"] = torch.quantile(values, torch.tensor([.05, .95], dtype=torch.float64)).tolist()
    record = {"schema": "car_round2_parameter_allocation_v1", "split": "TRAIN",
              "sample_ids": list(sample_ids), "population": population,
              "source_family": source_family, "parameterization": parameterization,
              "parameter_count": parameter_count, "probabilities": {k: v.tolist() for k, v in probabilities.items()},
              "target_weights": w.tolist(), "draws": draws, "policies": totals,
              "candidate_uniform_mse_cost_ratio": ratio, "overhead_fraction": overhead,
              "preparation_seconds": preparation_seconds, "audit_seconds": elapsed,
              "applicable_training_seconds": training_seconds, "qualified": qualified,
              "expanded_leaves": "enumerated and charged in fixed_seconds",
              "scope": "residual-head fixed-target parameter estimation; no backbone or learning-speed claim"}
    record["sha256"] = _digest(record)
    return record


def select_task_conditioned_bank(bank, measurements, *, repeatability_threshold,
                                 response_threshold, material_cost_fraction=.01):
    """Apply the TRAIN-frozen lexicographic rule to measured candidate responses.

    Measurements must include actual full-model AND head feasible response and
    development-task/Scale relevance; an endpoint-gradient complement is never
    substituted. A redundant leaf is removed only with a material saved full
    update cost. Every retained/rejected candidate and threshold is saved.
    """
    rows = [item["candidate"] for item in bank["selection_audit"]]
    if set(measurements) != {r["id"] for r in rows}:
        raise ValueError("every bounded candidate requires declared measured evidence")
    chosen, audits = [], []
    for row in rows:
        evidence = measurements[row["id"]]
        required = ("discrepancy", "replay_error", "repeatability", "development_response",
                    "full_model_response", "residual_head_response", "complete_seconds", "saved_update_fraction")
        if any(k not in evidence or not math.isfinite(evidence[k]) for k in required):
            raise ValueError("complete finite task/attainability/cost evidence required")
        if (min(evidence["discrepancy"], evidence["replay_error"], evidence["saved_update_fraction"]) < 0
            or evidence["complete_seconds"] <= 0 or not 0 <= evidence["repeatability"] <= 1
            or evidence["saved_update_fraction"] > 1):
            raise ValueError("nonnegative discrepancy/error, positive measured cost and valid fractions required")
    def useful(row):
        m = measurements[row["id"]]
        return min(m["development_response"], m["full_model_response"], m["residual_head_response"])
    starter = {row["id"] for row in bank.get("selected", ())}
    ordered = sorted(rows, key=lambda r: (r["id"] not in starter,
                     -useful(r) / max(measurements[r["id"]]["complete_seconds"], 1e-30), r["id"]))
    for row in ordered:
        m = measurements[row["id"]]
        threshold = max(response_threshold, 5 * m["replay_error"])
        if m["discrepancy"] <= 5 * m["replay_error"] or m["repeatability"] < repeatability_threshold:
            decision = "inactive_or_nonrepeatable"
        elif useful(row) <= threshold:
            decision = "no_measured_feasible_useful_response"
        elif m.get("redundant_with") in {q["id"] for q in chosen} and m["saved_update_fraction"] >= material_cost_fraction:
            decision = "redundant_with_material_complete_update_cost"
        elif sum(q["kind"] == row["kind"] for q in chosen) >= (4 if row["kind"] == "state.group_projection" else 8):
            decision = "bank_limit"
        else:
            decision = "selected"
            chosen.append(row)
        audits.append({"id": row["id"], "candidate": row, "decision": decision, "evidence": m})
    result = copy.deepcopy(bank)
    result.pop("sha256", None)
    result["selected"] = chosen
    result["selection_audit"] = audits
    result["selection_rule"] = {"version": "car_round2_task_repairability_v1",
        "repeatability_threshold": repeatability_threshold, "response_threshold": response_threshold,
        "material_cost_fraction": material_cost_fraction, "reserved_task_used": False,
        "retention_preference": "eligible source-specific starter leaves precede eligible replacements"}
    for kind, key in (("state.group_projection", "directions"), ("scale.third_order", "triples")):
        leaves = [q for q in chosen if q["kind"] == kind]
        result[key] = [_geometry(q, result["field_order"], result["field_scales"]) for q in leaves]
        result["target_weights"][kind] = [1 / len(leaves)] * len(leaves) if leaves else []
        values = [measurements[q["id"]]["discrepancy"] for q in leaves]
        result["block_units"][kind] = max(result["floor"], sum(values) / len(values)) if values else 1.
    result["sha256"] = _digest(result)
    return result
