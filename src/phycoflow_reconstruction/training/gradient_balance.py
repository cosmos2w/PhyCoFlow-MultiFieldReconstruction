"""Two-objective weighted-sum and optional ConFIG optimizer updates."""

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import nn

from .gradients import stable_clip_grad_norm_


def _trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("post-training selected no trainable parameters")
    return parameters


def _flat_gradient(loss: torch.Tensor, parameters: list[nn.Parameter]) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    flattened = []
    for parameter, gradient in zip(parameters, gradients):
        value = torch.zeros_like(parameter) if gradient is None else gradient
        if torch.is_complex(value):
            value = torch.view_as_real(value)
        flattened.append(value.reshape(-1))
    return torch.cat(flattened)


def _assign_flat_gradient(parameters: list[nn.Parameter], gradient: torch.Tensor) -> None:
    offset = 0
    for parameter in parameters:
        count = parameter.numel() * (2 if torch.is_complex(parameter) else 1)
        value = gradient[offset : offset + count]
        if torch.is_complex(parameter):
            real_dtype = parameter.real.dtype
            value = torch.view_as_complex(
                value.to(real_dtype).clone().reshape(*parameter.shape, 2).contiguous()
            )
        else:
            value = value.to(parameter.dtype).view_as(parameter)
        parameter.grad = value.clone()
        offset += count
    if offset != gradient.numel():
        raise ValueError("combined gradient length does not match trainable parameters")


def data_only_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    *,
    weight: float,
    grad_clip: float | None,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    (float(weight) * loss).backward()
    if grad_clip:
        norm = stable_clip_grad_norm_(model.parameters(), grad_clip)
    else:
        norms = [p.grad.detach().abs().double().square().sum() for p in model.parameters()
                 if p.grad is not None]
        norm = torch.stack(norms).sum().sqrt() if norms else loss.new_tensor(0.)
        if not torch.isfinite(norm):
            raise FloatingPointError("data-only gradient is non-finite; refusing optimizer update")
    optimizer.step()
    return {
        "update_mode": "data_only",
        "data_grad_norm": float(norm),
        "coherence_grad_norm": None,
        "gradient_cosine": None,
        "gradient_conflict": False,
        "combined_grad_norm": float(norm),
        "config_fallback_used": False,
    }


def two_objective_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loss: torch.Tensor,
    coherence_loss: torch.Tensor,
    *,
    mode: str,
    data_weight: float,
    coherence_weight: float,
    grad_clip: float | None,
    config_missing_behavior: str = "error",
) -> dict[str, Any]:
    """Update once while recording the relationship between both gradients."""
    parameters = _trainable_parameters(model)
    weighted_data = float(data_weight) * data_loss
    weighted_coherence = float(coherence_weight) * coherence_loss
    data_gradient = _flat_gradient(weighted_data, parameters)
    coherence_gradient = _flat_gradient(weighted_coherence, parameters)
    if not torch.isfinite(data_gradient).all() or not torch.isfinite(coherence_gradient).all():
        raise FloatingPointError("post-training objective gradient contains non-finite values")
    data_norm = torch.linalg.vector_norm(data_gradient)
    coherence_norm = torch.linalg.vector_norm(coherence_gradient)
    denominator = (data_norm * coherence_norm).clamp_min(1e-12)
    cosine = torch.dot(data_gradient, coherence_gradient) / denominator
    weighted_sum = data_gradient + coherence_gradient
    selected = weighted_sum
    update_mode = "weighted_sum"
    fallback = False

    normalized_mode = str(mode).lower()
    if normalized_mode not in {"weighted_sum", "config"}:
        raise ValueError("gradient balance mode must be weighted_sum or config")
    if normalized_mode == "config" and float(coherence_norm) > 0:
        try:
            from conflictfree.grad_operator import ConFIG_update
        except ImportError as error:
            if config_missing_behavior != "weighted_sum":
                raise ImportError(
                    "gradient balance mode=config requires the optional conflictfree package"
                ) from error
            warnings.warn(
                "conflictfree unavailable; using weighted_sum", RuntimeWarning, stacklevel=2
            )
            fallback = True
            update_mode = "weighted_sum_missing_config"
        else:
            if float(cosine) >= 0:
                update_mode = "weighted_sum_aligned"
            else:
                candidate = ConFIG_update([data_gradient, coherence_gradient])
                descends_data = torch.dot(candidate, data_gradient) > 0
                descends_coherence = torch.dot(candidate, coherence_gradient) > 0
                if torch.isfinite(candidate).all() and descends_data and descends_coherence:
                    selected = candidate
                    update_mode = "config"
                else:
                    fallback = True
                    update_mode = "weighted_sum_nondescent_config"

    optimizer.zero_grad(set_to_none=True)
    _assign_flat_gradient(parameters, selected)
    if grad_clip:
        stable_clip_grad_norm_(parameters, float(grad_clip))
    optimizer.step()
    return {
        "update_mode": update_mode,
        "data_grad_norm": float(data_norm.detach().cpu()),
        "coherence_grad_norm": float(coherence_norm.detach().cpu()),
        "gradient_cosine": float(cosine.detach().cpu()),
        "gradient_conflict": bool(float(cosine.detach().cpu()) < 0),
        "combined_grad_norm": float(torch.linalg.vector_norm(selected).detach().cpu()),
        "config_fallback_used": fallback,
    }


def combine_coherence_gradients(
    gradients_by_family: dict[str, torch.Tensor],
    *,
    method: str = "config",
    cagrad_alpha: float = 0.5,
    cagrad_rescale: int = 1,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Combine separately calibrated family gradients, keeping fidelity outside.

    CAGrad solves the simplex dual in the authors' NYUv2 implementation at
    Cranial-XIX/CAGrad dc3d48152b6196945cfd56144879b9d42353b095, utils.py.
    Rows here are tasks; upstream uses columns. Rescale=1 divides by 1+alpha².
    Weighted_sum returns the sum (the historical family-weight convention).
    ConFIG uses conflictfree==0.1.8; a failed common-descent check is explicit.
    """
    import math

    import numpy as np
    from scipy.optimize import minimize

    if method not in {"weighted_sum", "config", "cagrad"}:
        raise ValueError("unknown coherence gradient method")
    if not gradients_by_family:
        raise ValueError("at least one coherence gradient is required")
    names = list(gradients_by_family)
    matrix = torch.stack([gradients_by_family[name].detach() for name in names])
    if matrix.ndim != 2 or not torch.isfinite(matrix).all():
        raise FloatingPointError("coherence gradients must be finite flat real vectors")
    norms = torch.linalg.vector_norm(matrix.double(), dim=1)
    active = norms > 0
    selected = matrix.sum(dim=0)
    fallback = None
    rank = int(torch.linalg.matrix_rank(matrix.double() @ matrix.double().T).item())
    diagnostics: dict[str, Any] = {"method": method, "families": names,
                                  "rank": rank, "active_count": int(active.sum()),
                                  "zero_gradient_families": [n for n, a in zip(names, active) if not a]}
    if method == "config" and int(active.sum()) > 1:
        from conflictfree.grad_operator import ConFIG_update
        from conflictfree.weight_model import EqualWeight

        class DtypeEqualWeight(EqualWeight):
            def get_weights(self, gradients, losses=None, device=None):
                # conflictfree 0.1.8's default hard-codes float32 weights.
                return gradients.new_ones(gradients.shape[0])

        active_matrix = matrix[active]
        # CUDA lstsq assumes full row rank. The audited pseudoinverse route
        # remains defined for duplicate/opposing family gradients.
        candidate = ConFIG_update(active_matrix, weight_model=DtypeEqualWeight(), use_least_square=False)
        products = active_matrix.double() @ candidate.double()
        if torch.isfinite(candidate).all() and bool((products > 0).all()):
            selected = candidate
        else:
            fallback = "config_no_strict_common_descent_weighted_sum"
    elif method == "cagrad" and bool(active.any()):
        alpha = float(cagrad_alpha)
        if not math.isfinite(alpha) or alpha < 0 or cagrad_rescale not in {0, 1, 2}:
            raise ValueError("invalid CAGrad alpha/rescale")
        # Upstream simplex formulation. Keeping zero objectives in the
        # ensemble preserves the declared arithmetic-mean objective.
        gram = (matrix.double() @ matrix.double().T).cpu().numpy()
        count = len(names)
        uniform = np.full(count, 1.0 / count)
        coefficient = alpha * np.sqrt(max(float(gram.mean()), 0.0) + 1e-8) + 1e-8

        def objective(weights):
            return float(weights @ gram @ uniform + coefficient *
                         np.sqrt(max(float(weights @ gram @ weights), 0.0) + 1e-8))

        solution = minimize(objective, uniform, method="SLSQP",
                            bounds=[(0.0, 1.0)] * count,
                            constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
                            options={"ftol": 1e-10, "maxiter": 200})
        if not solution.success or not np.isfinite(solution.x).all():
            raise RuntimeError(f"CAGrad simplex solve failed: {solution.message}")
        weights = torch.as_tensor(solution.x, device=matrix.device, dtype=matrix.dtype)
        weighted = (weights[:, None] * matrix).sum(dim=0)
        multiplier = coefficient / (float(torch.linalg.vector_norm(weighted.double())) + 1e-8)
        selected = matrix.mean(dim=0) + multiplier * weighted
        if cagrad_rescale == 1:
            selected = selected / (1.0 + alpha**2)
        elif cagrad_rescale == 2:
            selected = selected / (1.0 + alpha)
        diagnostics.update(cagrad_weights=solution.x.tolist(), cagrad_alpha=alpha,
                           cagrad_rescale=cagrad_rescale, cagrad_solver_iterations=int(solution.nit))
    if not torch.isfinite(selected).all():
        raise FloatingPointError("coherence combiner produced a non-finite direction")
    diagnostics["fallback_reason"] = fallback
    diagnostics["combined_dot"] = {n: float(torch.dot(g.double(), selected.double()))
                                   for n, g in gradients_by_family.items()}
    diagnostics["strict_common_descent"] = all(
        diagnostics["combined_dot"][n] > 0 for n, a in zip(names, active) if a
    ) and bool(active.any())
    return selected, diagnostics


def coherence_primal_dual_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    family_losses: dict[str, torch.Tensor],
    fidelity_loss: torch.Tensor,
    *,
    method: str,
    grad_clip: float | None,
    diagnostics: bool = False,
    constraint_losses: dict[str, torch.Tensor] | None = None,
    cagrad_alpha: float = 0.5,
    cagrad_rescale: int = 1,
) -> dict[str, Any]:
    """Assign/clip once, then apply exactly one optimizer step.

    All flat gradients share the existing interleaved real/complex layout.
    The final fidelity pressure is never normalized by the coherence combiner.
    """
    parameters = _trainable_parameters(model)

    def objective_gradient(loss):
        # A disabled scalar objective can be detached. Retain the established
        # layout and unused-parameter handling without asking autograd to
        # differentiate a tensor that has no graph.
        if not loss.requires_grad:
            loss = loss + 0 * parameters[0].real.sum()
        return _flat_gradient(loss, parameters)

    families = {name: objective_gradient(loss) for name, loss in family_losses.items()}
    fidelity = objective_gradient(fidelity_loss)
    if not torch.isfinite(fidelity).all():
        raise FloatingPointError("fidelity gradient is non-finite")
    coherence, report = combine_coherence_gradients(
        families, method=method, cagrad_alpha=cagrad_alpha, cagrad_rescale=cagrad_rescale
    )
    final = coherence + fidelity
    row: dict[str, Any] = {"update_mode": f"coherence_primal_dual/{method}",
                          "update_accepted": True, "update/fallback_reason": report["fallback_reason"],
                          "coherence_combiner": report,
                          "combined_grad_norm": float(torch.linalg.vector_norm(final.double())),
                          "fidelity_grad_norm": float(torch.linalg.vector_norm(fidelity.double()))}
    for name, gradient in families.items():
        row[f"gradient/{name}/norm"] = float(torch.linalg.vector_norm(gradient.double()))
        row[f"gradient/combined_coherence_dot/{name}"] = report["combined_dot"][name]
        row[f"gradient/final_direction_dot/{name}"] = float(torch.dot(gradient.double(), final.double()))
    if diagnostics:
        vectors = {**families, "fidelity": fidelity}
        vectors.update({f"fidelity/{n}": _flat_gradient(v, parameters)
                        for n, v in (constraint_losses or {}).items()})
        before = torch.cat([(torch.view_as_real(p.detach()) if p.is_complex() else p.detach())
                            .reshape(-1).clone() for p in parameters])
        for left, g_left in vectors.items():
            for right, g_right in vectors.items():
                norm = torch.linalg.vector_norm(g_left.double()) * torch.linalg.vector_norm(g_right.double())
                key = f"gradient/{left}/{right}/cosine"
                row[key] = float(torch.dot(g_left.double(), g_right.double()) / norm) if norm > 0 else None
                if norm == 0:
                    row[key + "/undefined_reason"] = "zero_gradient"
    optimizer.zero_grad(set_to_none=True)
    _assign_flat_gradient(parameters, final)
    if grad_clip:
        stable_clip_grad_norm_(parameters, float(grad_clip))
    optimizer.step()
    if diagnostics:
        after = torch.cat([(torch.view_as_real(p.detach()) if p.is_complex() else p.detach())
                           .reshape(-1) for p in parameters])
        displacement = after - before
        vectors.update(combined_coherence=coherence, final=final)
        row["update/actual_displacement_norm"] = float(torch.linalg.vector_norm(displacement.double()))
        for name, gradient in vectors.items():
            row[f"update/actual_dot/{name}"] = float(torch.dot(gradient.double(), displacement.double()))
    return row
