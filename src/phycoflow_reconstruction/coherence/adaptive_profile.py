"""Fixed-SOURCE panel benchmark for five-field outer query allocation.

The finite target averages every outer pair and every Shape inner sign/line;
complete anchors, unary channels and expanded blocks are paid once per bundle.
"""

from __future__ import annotations

import math
import tempfile
import time

import torch


def _parameter_gradient_gram(callback, target, *, budget_check=None, spill_directory=None, chunk_size=8192):
    """Compute the exact full-parameter cell Gram via bounded disk-backed storage."""
    import shutil
    from pathlib import Path

    import numpy as np

    if chunk_size < 1:
        raise ValueError("positive Gram chunk size required")
    if budget_check is not None:
        budget_check()
    first = torch.zeros_like(target)
    first[0] = 1
    gradient = torch.as_tensor(callback(first)).detach().flatten().double().cpu()
    if not gradient.numel() or not torch.isfinite(gradient).all():
        raise ValueError("finite nonempty parameter gradient required")
    width = gradient.numel()
    gram = torch.zeros(len(target), len(target), dtype=torch.float64)
    with tempfile.TemporaryDirectory(prefix="car_param_gram_", dir=spill_directory) as directory:
        if shutil.disk_usage(directory).free < len(target) * width * 8:
            raise OSError("insufficient disk space for exact parameter-gradient Gram spill")
        mapped = np.memmap(Path(directory) / "gradients.bin", mode="w+", dtype=np.float64, shape=(len(target), width))
        try:
            mapped[0] = gradient.numpy()
            for index in range(1, len(target)):
                if budget_check is not None:
                    budget_check()
                hot = torch.zeros_like(target)
                hot[index] = 1
                gradient = torch.as_tensor(callback(hot)).detach().flatten().double().cpu()
                if gradient.numel() != width or not torch.isfinite(gradient).all():
                    raise ValueError("inconsistent or nonfinite parameter gradient")
                mapped[index] = gradient.numpy()
            mapped.flush()
            if budget_check is not None:
                budget_check()
            direct = torch.as_tensor(callback(target)).detach().flatten().double().cpu()
            if direct.numel() != width or not torch.isfinite(direct).all():
                raise ValueError("invalid direct parameter gradient")
            for start in range(0, width, chunk_size):
                if budget_check is not None:
                    budget_check()
                stop = min(width, start + chunk_size)
                block = torch.from_numpy(np.array(mapped[:, start:stop], copy=True))
                gram += block @ block.T
                if not torch.allclose(direct[start:stop], target @ block, rtol=1e-5, atol=1e-8):
                    raise ValueError("parameter callback must produce linear full-parameter gradients")
            if float(gram[0, 0].clamp_min(0).sqrt()) > 1e-8 * max(1.0, float(gram.diag().clamp_min(0).sqrt().max())):
                raise ValueError("parameter callback must return gradients relative to cell (0,0)")
        finally:
            del mapped
    return gram, width

from .query_runtime import ScalableQueryAdapter
from .query_sampling import (
    LINE_WEIGHTS,
    PAIR_SIGNS,
    Draw,
    HierarchicalQuerySampler,
    QueryBatch,
    pair_count,
)


def profile_source_panel(
    panel,
    *,
    families,
    artifacts,
    filter_fn,
    source_family,
    family_weights,
    parameter_gradient_fn,
    bundle_cost_fn,
    research_costs,
    policy_overhead_fraction,
    amortization_updates=13000,
    panel_batch_size=2,
    budget_check=None,
    spill_directory=None,
    coefficients=None,
    draws=3,
    replicates=128,
    seed=0,
):
    """Profile actual outer/inner sampled Shape design on fixed SOURCE endpoints.

    J2 is the first two stored independent members. parameter_gradient_fn takes
    flat [K*12+2*C] coefficients for pair and unary cells, and returns true
    parameter gradients relative to the fixed pair/unary reference cell. It
    may use a declared representative subpanel; endpoint gradients use all panel
    conditions. bundle_cost_fn(pair) returns synchronized incremental and fixed
    bundle seconds including model, anchors, unary, added leaves and SOURCE
    amortization. research_costs declares screening_seconds, fitting_seconds,
    audit_seconds, refresh_seconds and refresh_count over 13000 A1 updates.
    """
    import random
    from collections import Counter

    if (
        source_family not in {"ffm", "edm"}
        or draws < 1
        or replicates < 2
        or amortization_updates != 13000
        or panel_batch_size < 1
    ):
        raise ValueError("invalid source family or replicate budget")
    names = (
        "screening_seconds",
        "fitting_seconds",
        "audit_seconds",
        "refresh_seconds",
        "refresh_count",
    )
    if set(research_costs) != set(names) or any(
        not 0 <= float(research_costs[k]) < float("inf") for k in names
    ):
        raise ValueError("complete measured research costs required")
    if not 0 <= policy_overhead_fraction < float("inf"):
        raise ValueError("measured policy overhead required")
    if set(family_weights) != {"multiscale_spectrum", "topology"} or any(
        not math.isfinite(float(v)) or v <= 0 for v in family_weights.values()
    ):
        raise ValueError("both positive finite effective family weights required")
    started = time.perf_counter()
    x, y = panel["members"].detach(), panel["reference"].detach()
    coords, masses = panel["coordinates"], panel["masses"]
    if (
        x.ndim != 4
        or x.shape[0] != y.shape[0]
        or x.shape[2:] != y.shape[1:]
        or x.shape[1] < 2
        or coords.shape[:2] != y.shape[:2]
        or masses.shape != y.shape[:2]
    ):
        raise ValueError("invalid fixed SOURCE endpoint panel")
    population = pair_count(x.shape[-1])
    if not 2 <= x.shape[-1] <= 5:
        raise ValueError("exhaustive profiling is bounded to two through five fields")
    sample_ids = list(panel["sample_ids"])
    if (
        len(sample_ids) != len(y)
        or len(set(sample_ids)) != len(sample_ids)
        or sample_ids != list(artifacts["provenance"]["sample_ids"])
    ):
        raise ValueError("panel must contain the exact declared TRAIN population")
    c = x.shape[-1]
    fields = tuple(artifacts["provenance"]["field_order"])
    if len(fields) != c:
        raise ValueError("field order differs from panel")
    adapter = ScalableQueryAdapter(families, artifacts["shape_artifact"])
    coefficients = coefficients or {}
    inner = [
        (sign, line, weight / 4) for sign in PAIR_SIGNS for line, weight in enumerate(LINE_WEIGHTS)
    ]
    pair_cells = population * len(inner)
    cell_count = pair_cells + 2 * c
    endpoint_gram = torch.zeros(cell_count, cell_count, dtype=torch.float64)
    loss_acc = torch.zeros(population, len(inner), dtype=torch.float64)
    # Gradients for each condition are concatenated, not averaged across
    # conditions. Accumulate their exact Gram matrix without retaining the panel.
    for start in range(0, len(y), panel_batch_size):
        if budget_check is not None:
            budget_check()
        stop = min(len(y), start + panel_batch_size)
        xb, yb = x[start:stop], y[start:stop]
        cb, mb = coords[start:stop], masses[start:stop]
        cell_gradients = []
        fraction = (stop - start) / len(y)
        for index in range(population):
            for j, (sign, line, probability) in enumerate(inner):
                a = xb[:, 0].clone().requires_grad_(True)
                b = xb[:, 1].clone().requires_grad_(True)
                self_draw = Draw(0, 1, 1 / c, c, 1, (1,), None, 0.5)
                pair_draw = Draw(index, 1, 1 / population, population, 1, sign, line, probability)
                query_batch = QueryBatch((self_draw,), (pair_draw,), 1)
                context = {
                    "scoring_mode": "conditional",
                    "field_order": fields,
                    "conditional_draws": (b,),
                    "generated_weights": mb,
                    "reference_weights": mb,
                }
                total = a.new_zeros(())
                for name in ("multiscale_spectrum", "topology"):
                    result, _, _ = adapter.score(
                        name,
                        a,
                        yb,
                        coordinates=cb,
                        context=context,
                        batch=query_batch,
                        coefficients=coefficients,
                    )
                    total = total + float(family_weights[name]) * result.scalar_loss
                grads = torch.autograd.grad(total, (a, b), allow_unused=True)
                cell_gradients.append(
                    torch.cat(
                        [
                            g.detach().flatten().double().cpu()
                            if g is not None
                            else torch.zeros_like(v).flatten().double().cpu()
                            for g, v in zip(grads, (a, b))
                        ]
                    )
                    * fraction
                )
                loss_acc[index, j] += float(total.detach()) * fraction
        # Unary field/sign draws form a separate sampled stratum.
        fixed_sign, fixed_line, fixed_probability = inner[0]
        for field in range(c):
            for unary_sign in (1, -1):
                a = xb[:, 0].clone().requires_grad_(True)
                b = xb[:, 1].clone().requires_grad_(True)
                self_draw = Draw(field, 1, 1 / c, c, 1, (unary_sign,), None, 0.5)
                pair_draw = Draw(
                    0, 1, 1 / population, population, 1, fixed_sign, fixed_line, fixed_probability
                )
                query_batch = QueryBatch((self_draw,), (pair_draw,), 1)
                context = {
                    "scoring_mode": "conditional",
                    "field_order": fields,
                    "conditional_draws": (b,),
                    "generated_weights": mb,
                    "reference_weights": mb,
                }
                total = a.new_zeros(())
                for name in ("multiscale_spectrum", "topology"):
                    result, _, _ = adapter.score(
                        name,
                        a,
                        yb,
                        coordinates=cb,
                        context=context,
                        batch=query_batch,
                        coefficients=coefficients,
                    )
                    total = total + float(family_weights[name]) * result.scalar_loss
                grads = torch.autograd.grad(total, (a, b), allow_unused=True)
                cell_gradients.append(
                    torch.cat(
                        [
                            g.detach().flatten().double().cpu()
                            if g is not None
                            else torch.zeros_like(v).flatten().double().cpu()
                            for g, v in zip(grads, (a, b))
                        ]
                    )
                    * fraction
                )
        local = torch.stack(cell_gradients)
        local = local - local[:1]
        endpoint_gram += local @ local.T
    incremental, fixed = [], []
    for index in range(population):
        cost, base = bundle_cost_fn(index)
        if not (0 < cost < float("inf") and 0 <= base < float("inf")):
            raise ValueError("measured positive incremental and finite fixed costs required")
        incremental.append(float(cost))
        fixed.append(float(base))
    loss_rows = loss_acc.tolist()
    inner_prob = torch.tensor([v[2] for v in inner], dtype=torch.float64)
    target = torch.cat(
        (
            (inner_prob[None, :].expand(population, -1) / population).flatten(),
            torch.full((2 * c,), 1 / (2 * c), dtype=torch.float64),
        )
    )
    difficulty = [sum(abs(value) * inner[j][2] for j, value in enumerate(row)) for row in loss_rows]
    difficulty_sum = sum(difficulty)
    uniform = [1 / population] * population
    difficulty_proposal = (
        [0.25 / population + 0.75 * v / difficulty_sum for v in difficulty]
        if difficulty_sum
        else uniform
    )
    tree = HierarchicalQuerySampler(fields, epsilon=0.25, hierarchy_weight=0.5)
    hierarchy = [tree.pair_probability(i) for i in range(population)]
    endpoint_diagonal = endpoint_gram.diagonal()[:pair_cells].reshape(population, len(inner))
    proxy = [
        float((endpoint_diagonal[i] * inner_prob).sum().sqrt()) / incremental[i] ** 0.5
        for i in range(population)
    ]
    priority = [v / sum(proxy) for v in proxy] if sum(proxy) else uniform
    group = [(p - 0.5 / population) / 0.5 for p in hierarchy]
    gradient_cost = [0.25 / population + 0.5 * g + 0.25 * p for g, p in zip(group, priority)]
    proposals = {
        "uniform": uniform,
        "c8_difficulty": difficulty_proposal,
        "hierarchy": hierarchy,
        "gradient_cost": gradient_cost,
    }
    parameter_gram, parameter_count = _parameter_gradient_gram(
        parameter_gradient_fn, target, budget_check=budget_check, spill_directory=spill_directory
    )
    fixed_seconds = sum(fixed) / len(fixed)
    research_seconds = (
        research_costs["screening_seconds"]
        + research_costs["fitting_seconds"]
        + research_costs["audit_seconds"]
        + research_costs["refresh_seconds"] * research_costs["refresh_count"]
    )
    amortized = float(research_seconds) / amortization_updates
    results = {}
    for label, p in proposals.items():
        rng = random.Random(seed)
        errors_e, errors_p, charged = [], [], []
        for _ in range(replicates):
            counts = Counter(rng.choices(range(population), weights=p, k=draws))
            coeff = torch.zeros_like(target)
            for i, count in counts.items():
                j = rng.choices(range(len(inner)), weights=inner_prob.tolist(), k=1)[0]
                coeff[i * len(inner) + j] = count / (draws * population * p[i])
            unary = rng.randrange(2 * c)
            coeff[pair_cells + unary] = 1.0
            endpoint_delta = (coeff - target).flatten()
            endpoint_sq = float(endpoint_delta @ endpoint_gram @ endpoint_delta)
            parameter_delta = coeff - target
            parameter_sq = float(parameter_delta @ parameter_gram @ parameter_delta)
            errors_e.append(max(0.0, endpoint_sq))
            errors_p.append(max(0.0, parameter_sq))
            charged.append(sum(incremental[i] for i in counts))
        e = torch.tensor(errors_e).sqrt()
        g = torch.tensor(errors_p).sqrt()
        ec, gc = e - e.mean(), g - g.mean()
        denom = ec.norm() * gc.norm()
        mse = sum(errors_p) / replicates
        incremental_cost = sum(charged) / replicates
        extra = 0.0 if label == "uniform" else amortized
        total_cost = fixed_seconds + incremental_cost + extra
        results[label] = {
            "endpoint_gradient_mse": sum(errors_e) / replicates,
            "parameter_gradient_mse": mse,
            "mean_charged_cost": incremental_cost,
            "fixed_bundle_seconds": fixed_seconds,
            "research_amortized_seconds": extra,
            "total_charged_seconds": total_cost,
            "parameter_mse_times_total_cost": mse * total_cost,
            "maximum_importance_weight": max(1 / (population * v) for v in p),
            "proxy_error_correlation": float(ec @ gc / denom) if denom > 0 else None,
        }
    output = {
        "source_identity": artifacts["provenance"]["source_checkpoint_sha256"],
        "source_family": source_family,
        "proposals": results,
        "proposal_probabilities": proposals,
        "incremental_leaf_seconds": incremental,
        "fixed_bundle_seconds": fixed_seconds,
        "fixed_cost_scope": "model, anchors, unary, expanded leaves, declared SOURCE amortization",
        "research_costs": dict(research_costs),
        "research_amortization_updates": amortization_updates,
        "total_policy_research_seconds": float(research_seconds),
        "endpoint_panel_batch_size": panel_batch_size,
        "policy_overhead_fraction": policy_overhead_fraction,
        "parameter_gradient_audits": cell_count + 1,
        "parameter_count": parameter_count,
        "parameter_gradient_storage": "temporary disk-backed float64 cell matrix; exact chunked Gram",
        "outer_population": population,
        "inner_shape_enumerated": True,
        "members_used": 2,
        "panel_conditions": len(y),
        "panel_sample_ids": sample_ids,
        "variance_scope": "joint pair and unary outer and inner draws; duplicate pair IDs share one inner draw",
        "estimator_scope": "pre-optimizer gradient, no ConFIG or optimizer unbiasedness claim",
        "total_audit_wall_seconds": time.perf_counter() - started,
    }
    return output


def qualify_policy(bank, profiles, *, minimum_improvement=0.20):
    """Freeze a qualified policy only when both distinct SOURCE audits earn it."""
    import copy

    from .adaptive_fitting import _digest

    if len(profiles) != 2 or not 0.20 <= minimum_improvement < 1:
        raise ValueError("two SOURCE audits and a valid improvement threshold required")
    if {p.get("source_family") for p in profiles} != {"ffm", "edm"}:
        raise ValueError("separate FFM and EDM profiles required")
    sources = [p.get("source_identity") for p in profiles]
    if (
        any(not s for s in sources)
        or len(set(sources)) != 2
        or bank["source_identity"] not in sources
    ):
        raise ValueError("both distinct SOURCE identities must be represented")
    summaries = []
    accepted = True
    for profile in profiles:
        uniform = profile["proposals"]["uniform"]["parameter_mse_times_total_cost"]
        candidate = profile["proposals"]["gradient_cost"]["parameter_mse_times_total_cost"]
        correlation = profile["proposals"]["gradient_cost"]["proxy_error_correlation"]
        tail = profile["proposals"]["gradient_cost"]["maximum_importance_weight"]
        overhead = profile.get("policy_overhead_fraction")
        costs = profile.get("incremental_leaf_seconds", [])
        probabilities = profile.get("proposal_probabilities", {}).get("gradient_cost", [])
        population = profile.get("outer_population")
        valid_probabilities = (
            isinstance(population, int) and 1 <= population <= 10
            and len(probabilities) == population and len(costs) == population
            and all(isinstance(p, (int, float)) and math.isfinite(p) and p >= 0.25 / population for p in probabilities)
            and math.isclose(sum(probabilities), 1., rel_tol=1e-10, abs_tol=1e-10)
            and isinstance(tail, (int, float)) and math.isfinite(tail)
            and math.isclose(tail, max(1 / (population * p) for p in probabilities), rel_tol=1e-9)
        )
        if (
            profile.get("variance_scope")
            != "joint pair and unary outer and inner draws; duplicate pair IDs share one inner draw"
            or profile.get("research_amortization_updates") != 13000
            or not math.isfinite(uniform)
            or not math.isfinite(candidate)
            or uniform <= 0
            or candidate < 0
            or correlation is None
            or not math.isfinite(correlation)
            or overhead is None
            or not 0 <= overhead <= 0.05
            or not isinstance(tail, (int, float)) or not 0 < tail <= 4
            or not costs
            or any(not 0 < c < float("inf") for c in costs)
            or candidate > (1 - minimum_improvement) * uniform
            or correlation <= 0
            or not valid_probabilities
        ):
            accepted = False
        summaries.append(
            {
                "source_identity": profile["source_identity"],
                "uniform": uniform,
                "gradient_cost": candidate,
                "proxy_error_correlation": correlation,
                "maximum_importance_weight": tail,
                "policy_overhead_fraction": overhead,
            }
        )
    result = copy.deepcopy(bank)
    result["policy_qualified"] = accepted
    if accepted:
        match = next(p for p in profiles if p["source_identity"] == bank["source_identity"])
        result["policy_incremental_pair_costs"] = list(match["incremental_leaf_seconds"])
        result["policy_mode"] = "fixed_source_profile"
        result["policy_pair_probabilities"] = list(match["proposal_probabilities"]["gradient_cost"])
        result["policy_profile_sha256"] = _digest(match)
    else:
        for key in ("policy_mode", "policy_pair_probabilities", "policy_profile_sha256", "policy_incremental_pair_costs"):
            result.pop(key, None)
    result["policy_qualification"] = {
        "minimum_improvement": minimum_improvement,
        "sources": summaries,
        "scope": "fixed SOURCE endpoint allocation; mutable calibration and ConFIG require separate assessment",
    }
    result.pop("sha256", None)
    result["sha256"] = _digest(result)
    return result
