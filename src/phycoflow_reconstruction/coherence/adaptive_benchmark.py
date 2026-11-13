"""Frozen-target replicated estimator and gradient proxy audits."""

from __future__ import annotations

import math
import random
import time

import torch


def benchmark(
    target_weights,
    proposals,
    leaf_costs,
    *,
    draws,
    replicates,
    seed,
    endpoint_gradient_fn,
    parameter_gradient_fn,
):
    """Compare importance estimators to exact full-target gradient callbacks.

    Each callback maps a vector of leaf coefficients to the corresponding
    gradient vector, at fixed endpoint graph/parameters. Parameter callback
    must recompute the actual model graph as needed. It is intentionally not a
    descriptor-gradient proxy. Reported charged costs are additive incremental
    cost estimates; callback timings include complete audit overhead separately.
    """
    w = torch.as_tensor(target_weights, dtype=torch.float64)
    costs = torch.as_tensor(leaf_costs, dtype=torch.float64)
    if (
        w.ndim != 1
        or not len(w)
        or costs.shape != w.shape
        or not torch.isfinite(w).all()
        or (w < 0).any()
        or w.sum() <= 0
        or not torch.isfinite(costs).all()
        or (costs <= 0).any()
        or draws < 1
        or replicates < 2
    ):
        raise ValueError("invalid target, costs or sampling budget")

    def run(fn, weights):
        value = torch.as_tensor(fn(weights)).detach().flatten().double().cpu()
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite audit gradient")
        return value

    start = time.perf_counter()
    exact_endpoint = run(endpoint_gradient_fn, w)
    exact_parameter = run(parameter_gradient_fn, w)
    out = {}
    for label, raw in proposals.items():
        p = torch.as_tensor(raw, dtype=torch.float64)
        if (
            p.shape != w.shape
            or not torch.isfinite(p).all()
            or (p < 0).any()
            or not torch.isclose(p.sum(), p.new_tensor(1.0))
            or ((p <= 0) & (w > 0)).any()
        ):
            raise ValueError("proposal must support every positive target leaf")
        rng = random.Random(seed)
        endpoint_mse, parameter_mse, charged, endpoint_errors, parameter_errors = [], [], [], [], []
        for _ in range(replicates):
            ids = rng.choices(range(len(w)), weights=p.tolist(), k=draws)
            coeff = torch.zeros_like(w)
            for i in ids:
                coeff[i] += w[i] / p[i] / draws
            ge = run(endpoint_gradient_fn, coeff) - exact_endpoint
            gp = run(parameter_gradient_fn, coeff) - exact_parameter
            endpoint_mse.append(float(ge.square().sum()))
            parameter_mse.append(float(gp.square().sum()))
            endpoint_errors.append(math.sqrt(endpoint_mse[-1]))
            parameter_errors.append(math.sqrt(parameter_mse[-1]))
            charged.append(float(costs[ids].sum()))
        e = torch.tensor(endpoint_errors, dtype=torch.float64)
        g = torch.tensor(parameter_errors, dtype=torch.float64)
        centered_e, centered_g = e - e.mean(), g - g.mean()
        denominator = centered_e.norm() * centered_g.norm()
        out[label] = {
            "endpoint_gradient_mse": sum(endpoint_mse) / replicates,
            "parameter_gradient_mse": sum(parameter_mse) / replicates,
            "mean_charged_cost": sum(charged) / replicates,
            "parameter_mse_times_cost": sum(parameter_mse) / replicates * sum(charged) / replicates,
            "proxy_error_correlation": float(centered_e @ centered_g / denominator)
            if denominator > 0
            else None,
        }
    return {
        "seed": seed,
        "draws": draws,
        "replicates": replicates,
        "proposals": out,
        "audit_wall_seconds": time.perf_counter() - start,
    }
