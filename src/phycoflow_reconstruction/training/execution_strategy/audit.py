"""Sparse fixed TRAIN probes; complete descriptors and bounded gradient audits."""

import math
import random
from contextlib import contextmanager
from time import perf_counter

import numpy as np
import torch


@contextmanager
def preserve_rng_and_modes(model):
    cpu, python, numpy = torch.get_rng_state(), random.getstate(), np.random.get_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    modes = {module: module.training for module in model.modules()}
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        random.setstate(python)
        np.random.set_state(numpy)
        if cuda:
            torch.cuda.set_rng_state_all(cuda)
        for module, mode in modes.items():
            module.training = mode
        with torch.no_grad():
            for name, value in model.named_buffers():
                if name in buffers:
                    value.copy_(buffers[name])


def nested_parameter_variance(gradients, inner_variances, probabilities, support=()):
    """Exact conditional covariance, including correlated outer/nested choices."""
    names = tuple(gradients)
    if any(
        not math.isfinite(float(value)) or float(value) < 0 for value in inner_variances.values()
    ):
        raise ValueError("inner variances must be finite and nonnegative")
    if any(not 0 < float(probabilities.get(name, 1.0)) <= 1 for name in names):
        raise ValueError("inclusion probabilities must be positive and at most one")
    if support:
        if any(
            not math.isfinite(float(mass))
            or mass <= 0
            or not included
            or not set(included).issubset(names)
            or len(set(included)) != len(included)
            for mass, included in support
        ):
            raise ValueError("outer support needs positive masses and nonempty known family sets")
        if not math.isclose(sum(mass for mass, _ in support), 1.0, abs_tol=1e-10):
            raise ValueError("outer support masses must sum to one")
        for name in names:
            marginal = sum(mass for mass, included in support if name in included)
            if not math.isclose(marginal, probabilities.get(name, 1.0), abs_tol=1e-10):
                raise ValueError("outer support does not match its marginal probabilities")
    variance = 0.0
    for left in names:
        p_left = float(probabilities.get(left, 1.0))
        variance += float(inner_variances.get(left, 0.0)) / p_left
        for right in names:
            p_right = float(probabilities.get(right, 1.0))
            if left == right:
                joint = p_left
            elif support:
                joint = sum(
                    mass for mass, included in support if left in included and right in included
                )
            else:
                joint = p_left * p_right
            variance += (joint / (p_left * p_right) - 1.0) * float(
                torch.dot(gradients[left].double(), gradients[right].double())
            )
    if variance < -1e-8:
        raise ValueError("negative execution variance indicates invalid inclusion support")
    return max(0.0, variance)


def _flat_parameter_gradient(loss, parameters, *, multiplier=1.0):
    """Flatten a scalar objective gradient after its fixed outer coefficient."""
    parts = torch.autograd.grad(
        loss * float(multiplier), parameters, retain_graph=True, allow_unused=True
    )
    return torch.cat(
        [
            (torch.zeros_like(parameter) if part is None else part).reshape(-1)
            for parameter, part in zip(parameters, parts)
        ]
    ).detach()


def fixed_train_probe(model, batches, families, banks, config, runtime, epoch):
    from ...coherence.families.topology.filtration_sampling import (
        TopologyFiltrationSampler,
        sampled_family_forward,
    )
    from ..fidelity_budget import field_mse
    from ..post_training import _coherence_objective

    device = next(model.parameters()).device
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    field_names = tuple(name for name, _, _ in runtime.task.field_bounds)
    expensive = epoch % runtime.plan.audit_interval_epochs == 0
    full_gradients, execution_variances, reports = [], [], []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()
    with preserve_rng_and_modes(model):
        for index, batch in enumerate(batches[:2] if expensive else batches[:1]):
            endpoint, losses = {}, {}
            with torch.set_grad_enabled(expensive):
                result, _ = _coherence_objective(
                    model,
                    batch,
                    families,
                    banks,
                    config,
                    step=0,
                    generator=torch.Generator(device=device).manual_seed(7700313 + index),
                    family_losses=losses,
                    endpoint_context=endpoint,
                )
                risks = field_mse(endpoint["prediction"], endpoint["target"], endpoint["mask"])
                reports.append(
                    {
                        "families": {name: float(loss.detach()) for name, loss in losses.items()},
                        "field_risks": dict(zip(field_names, risks.detach().cpu().tolist())),
                        "sample_ids": list(batch.sample_ids),
                    }
                )
                if expensive:

                    gradients = {
                        name: _flat_parameter_gradient(
                            loss / runtime.task.c0,
                            parameters,
                            multiplier=float(getattr(families[name], "family_weight", 1.0)),
                        )
                        for name, loss in losses.items()
                    }
                    full_gradients.append(sum(gradients.values()))
                    inner_variances = {}
                    for name, specification in runtime.plan.within_family.items():
                        sampler = TopologyFiltrationSampler(
                            families[name].spatial_objective,
                            strategy=specification["estimator"],
                            seed=700211,
                        )
                        differences = []
                        for choice in sampler.enumerate_choices():
                            estimate = sampled_family_forward(
                                families[name],
                                endpoint["prediction"],
                                endpoint["target"],
                                coordinates=batch.query_coords,
                                sampler=sampler,
                                choice=choice,
                            )
                            sampled_gradient = _flat_parameter_gradient(
                                estimate.scalar_loss / runtime.task.c0,
                                parameters,
                                multiplier=float(
                                    getattr(families[name], "family_weight", 1.0)
                                ),
                            )
                            differences.append(
                                float((sampled_gradient.double() - gradients[name]).square().sum())
                            )
                        inner_variances[name] = sum(differences) / len(differences)
                    execution_variances.append(
                        nested_parameter_variance(
                            gradients,
                            inner_variances,
                            runtime.plan.family_probabilities,
                            runtime.plan.family_support,
                        )
                    )
                del result, losses, endpoint
    if expensive:
        stacked = torch.stack(full_gradients).double()
        intrinsic = float((stacked - stacked.mean(0)).square().sum(1).mean())
        execution = sum(execution_variances) / len(execution_variances)
    else:
        intrinsic = runtime.profile.intrinsic_noise_variance
        execution = runtime.plan.extra_variance
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    fields = {
        name: sum(report["field_risks"][name] for report in reports) / len(reports)
        for name in field_names
    }
    runtime.observe_audit(
        epoch,
        elapsed,
        intrinsic + execution,
        intrinsic,
        fields if expensive else None,
        split="TRAIN",
    )
    before = runtime.plan.plan_id
    runtime.refresh(epoch)
    return {
        "epoch": epoch,
        "split": "TRAIN",
        "full_metrics": reports,
        "expensive_parameter_audit": expensive,
        "intrinsic_parameter_variance": intrinsic,
        "execution_parameter_variance": execution,
        "seconds": elapsed,
        "plan_before": before,
        "plan_after": runtime.plan.plan_id,
        "variance_scope": "exact conditional support; intrinsic variance from two fixed TRAIN batches; finite-probe estimate",
    }
