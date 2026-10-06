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


def _packed_diagnostic_gram(vectors):
    """One small float64 host transfer; aliases/explicit zeros share no work."""
    import math
    names = list(vectors)
    unique, positions, lookup = [], {}, {}
    for name, vector in vectors.items():
        if vector is None:
            positions[name] = None
            continue
        if vector.ndim != 1 or vector.is_complex():
            raise ValueError("diagnostic vectors require the existing flat real layout")
        key = id(vector)
        if key not in lookup:
            lookup[key] = len(unique)
            unique.append(vector)
        positions[name] = lookup[key]
    if not unique:
        raise ValueError("at least one diagnostic vector is required")
    matrix = torch.stack([vector.to(torch.float64) for vector in unique])
    small = (matrix @ matrix.T).detach().cpu()
    if not bool(torch.isfinite(small).all()):
        raise FloatingPointError("non-finite diagnostic Gram")
    values = small.tolist()
    def dot(left, right):
        i, j = positions[left], positions[right]
        return 0.0 if i is None or j is None else values[i][j]
    norms = {name: math.sqrt(max(0.0, dot(name, name))) for name in names}
    cosines = {}
    for left in names:
        for right in names:
            denominator = norms[left] * norms[right]
            cosines[(left, right)] = dot(left, right) / denominator if denominator > 0 else None
    return {"names": names, "positions": positions, "matrix": matrix,
            "norms": norms, "cosines": cosines, "dot": dot,
            "unique_vectors": len(unique), "gram_host_transfers": 1}


def _packed_actual_displacement_products(packed, displacement):
    unique_dots = (packed["matrix"] @ displacement.to(torch.float64)).detach().cpu().tolist()
    return {name: 0.0 if index is None else unique_dots[index]
            for name, index in packed["positions"].items()}


def make_shared_family_diagnostic_callback(parameters, raw_family_losses, *, data_weight, coherence_weight):
    """Diagnostics only; no model forwards or SOURCE calibration changes."""
    import math
    if not (math.isfinite(data_weight) and data_weight > 0 and
            math.isfinite(coherence_weight) and coherence_weight > 0):
        raise ValueError("reconstructed diagnostic gradients require declared positive finite weights")
    parameters = list(parameters)
    def callback(weighted_native, weighted_coherence):
        family_gradients = {name: _flat_gradient(loss, parameters) for name, loss in raw_family_losses.items()}
        native = weighted_native / data_weight
        combined = weighted_coherence / coherence_weight
        vectors = {"native": native, "combined": combined,
                   **{f"family/{name}": gradient for name, gradient in family_gradients.items()}}
        packed = _packed_diagnostic_gram(vectors)
        names = list(family_gradients)
        norms = packed["norms"]
        return {
            "family_losses": dict(zip(names, torch.stack([raw_family_losses[n].detach() for n in names]).cpu().tolist())),
            "native_data_gradient_norm": norms["native"],
            "family_gradient_norms": {name: norms[f"family/{name}"] for name in names},
            "family_family_cosines": {left: {right: packed["cosines"][(f"family/{left}", f"family/{right}")]
                                             for right in names} for left in names},
            "data_family_cosines": {name: packed["cosines"][("native", f"family/{name}")] for name in names},
            "total_coherence_gradient_norm": norms["combined"],
            "diagnostic_gradient_identity": "native/combined reconstructed from cached positive-weight flat gradients; complex view_as_real Euclidean layout",
            "diagnostic_native_weight": data_weight, "diagnostic_coherence_weight": coherence_weight,
            "diagnostic_forwards": 0, "diagnostic_family_backward_requests": len(family_gradients),
            "diagnostic_gram_host_transfers": packed["gram_host_transfers"],
            "SOURCE_calibration_changed": False,
        }
    return callback


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


def _scalar_objective_update(parameters, optimizer, loss, grad_clip):
    """Exact linearity shortcut; keep the oracle's unused-parameter semantics.

    The oracle assigns zeros rather than None to unused parameters. This
    matters for AdamW decay and moments, so the scalar path does the same.
    Individual objective gradient telemetry belongs to diagnostic batches.
    """
    import math

    if grad_clip and (not math.isfinite(float(grad_clip)) or float(grad_clip) < 0):
        raise ValueError("max_norm must be finite and non-negative")
    optimizer.zero_grad(set_to_none=True)
    if not loss.requires_grad:
        loss = loss + 0 * parameters[0].real.sum()
    loss.backward()
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
    squared = torch.stack([p.grad.detach().abs().double().square().sum() for p in parameters]).sum()
    norm = squared.sqrt()
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError("post-training objective gradient contains non-finite values")
    if grad_clip:
        coefficient = (float(grad_clip) / (norm + 1.0e-6)).clamp(max=1.0)
        for parameter in parameters:
            work_dtype = torch.complex128 if parameter.grad.is_complex() else torch.float64
            clipped = parameter.grad.to(work_dtype) * coefficient.to(parameter.grad.device)
            parameter.grad.copy_(clipped.to(parameter.grad.dtype))
    else:
        coefficient = norm.new_tensor(1.)
    norm_value, coefficient_value = torch.stack([norm, coefficient]).detach().cpu().tolist()
    optimizer.step()
    return {
        "combined_grad_norm": norm_value,
        "gradient/preclip_norm": norm_value,
        "gradient/clip_limit": float(grad_clip) if grad_clip else None,
        "gradient/clip_coefficient": coefficient_value,
        "gradient/clipping_applied": coefficient_value < 1.,
        "gradient/component_diagnostics_sampled": False,
    }


def _config_gram_direction(matrix: torch.Tensor, *, use_least_square_oracle=False) -> tuple[torch.Tensor, str | None]:
    """Installed ConFIG pinv + ProjectionLength through a small Gram solve.

    For unit rows U, U+ 1 = U.T (U U.T)+ 1. Match the oracle's singular
    cutoff, which depends on the full parameter width, not just task count.
    Near the cutoff, defer to the wide SVD oracle rather than square its
    conditioning. Units and final projection scaling keep the input dtype.
    """
    norms = matrix.norm(dim=1)
    units = torch.nan_to_num(matrix / norms[:, None], 0)
    wide_units = units.double()
    gram = wide_units @ wide_units.T
    eigenvalues, eigenvectors = torch.linalg.eigh((gram + gram.T) / 2)
    relative_cutoff = max(matrix.shape) * torch.finfo(matrix.dtype).eps
    cutoff = eigenvalues[-1].clamp_min(0) * relative_cutoff**2
    # A materially ill-conditioned retained system is not a safe exact
    # shortcut. Wide SVD remains the declared oracle for this case.
    positive = eigenvalues > cutoff
    near_cutoff = ((eigenvalues > cutoff / 4) & (eigenvalues < cutoff * 4)).any()
    ill_conditioned = ((eigenvalues > cutoff) & (eigenvalues < eigenvalues[-1] * 1e-8)).any()
    singular_lstsq = (positive.sum() < matrix.shape[0]) if use_least_square_oracle else False
    if bool(near_cutoff | ill_conditioned | singular_lstsq):
        return _config_oracle_direction(matrix, use_least_square=use_least_square_oracle), "gram_cutoff_oracle"
    inverse = torch.where(positive, eigenvalues.clamp_min(torch.finfo(torch.float64).tiny).reciprocal(), 0)
    weights = eigenvectors @ (inverse * (eigenvectors.T @ gram.new_ones(matrix.shape[0])))
    target = (units.T @ weights.to(matrix.dtype))
    unit_target = torch.nan_to_num(target / target.norm(), 0)
    length = torch.stack([torch.dot(row, unit_target) for row in matrix]).sum()
    return unit_target * length, None


def _config_oracle_direction(matrix, *, use_least_square=False):
    from conflictfree.grad_operator import ConFIG_update
    from conflictfree.weight_model import EqualWeight

    class DtypeEqualWeight(EqualWeight):
        def get_weights(self, gradients, losses=None, device=None):
            return gradients.new_ones(gradients.shape[0])

    return ConFIG_update(matrix, weight_model=DtypeEqualWeight(), use_least_square=use_least_square)


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
    execution: str = "legacy",
    diagnostics: bool = True,
    config_backend: str = "legacy",
    diagnostic_callback=None,
) -> dict[str, Any]:
    """Update once while recording the relationship between both gradients."""
    parameters = _trainable_parameters(model)
    if diagnostic_callback is not None and not diagnostics:
        raise ValueError("diagnostic callback requires a diagnostic batch")
    weighted_data = float(data_weight) * data_loss
    weighted_coherence = float(coherence_weight) * coherence_loss
    if execution not in {"legacy", "exact", "scalar"}:
        raise ValueError("gradient execution must be legacy, exact or scalar")
    if config_backend not in {"legacy", "gram"}:
        raise ValueError("ConFIG backend must be legacy or gram")
    if execution != "legacy" and mode == "weighted_sum" and not diagnostics:
        row = _scalar_objective_update(parameters, optimizer, weighted_data + weighted_coherence, grad_clip)
        row.update(update_mode="weighted_sum", data_grad_norm=None, coherence_grad_norm=None,
                   gradient_cosine=None, gradient_conflict=None, config_fallback_used=False)
        return row
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
                use_lstsq = data_gradient.dtype == torch.float32 or data_gradient.is_cuda
                candidate = (_config_gram_direction(torch.stack([data_gradient, coherence_gradient]),
                                                    use_least_square_oracle=use_lstsq)[0]
                             if config_backend == "gram" else
                             _config_oracle_direction(torch.stack([data_gradient, coherence_gradient]),
                                                      use_least_square=use_lstsq))
                descends_data = torch.dot(candidate, data_gradient) > 0
                descends_coherence = torch.dot(candidate, coherence_gradient) > 0
                if torch.isfinite(candidate).all() and descends_data and descends_coherence:
                    selected = candidate
                    update_mode = "config"
                else:
                    fallback = True
                    update_mode = "weighted_sum_nondescent_config"

    callback_report = (diagnostic_callback(data_gradient.detach(), coherence_gradient.detach())
                       if diagnostic_callback is not None else None)
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
        **({"family_gradient_diagnostics": callback_report} if callback_report is not None else {}),
    }


def combine_coherence_gradients(
    gradients_by_family: dict[str, torch.Tensor],
    *,
    method: str = "config",
    cagrad_alpha: float = 0.5,
    cagrad_rescale: int = 1,
    config_backend: str = "legacy",
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
    if config_backend not in {"legacy", "gram"}:
        raise ValueError("ConFIG backend must be legacy or gram")
    if not gradients_by_family:
        raise ValueError("at least one coherence gradient is required")
    names = list(gradients_by_family)
    matrix = torch.stack([gradients_by_family[name].detach() for name in names])
    if matrix.ndim != 2 or not torch.isfinite(matrix).all():
        raise FloatingPointError("coherence gradients must be finite flat real vectors")
    wide_matrix = matrix.double()
    norms = torch.linalg.vector_norm(wide_matrix, dim=1)
    active = norms > 0
    active_flags = active.detach().cpu().tolist()
    active_count = sum(active_flags)
    selected = matrix.sum(dim=0)
    fallback = None
    rank = int(torch.linalg.matrix_rank(wide_matrix @ wide_matrix.T).item())
    diagnostics: dict[str, Any] = {"method": method, "families": names,
                                  "rank": rank, "active_count": active_count,
                                  "zero_gradient_families": [n for n, a in zip(names, active_flags) if not a]}
    if method == "config" and active_count > 1:
        active_matrix = matrix[active]
        # CUDA lstsq assumes full row rank. The audited pseudoinverse route
        # remains defined for duplicate/opposing family gradients.
        if config_backend == "gram":
            candidate, backend_fallback = _config_gram_direction(active_matrix)
            diagnostics["backend_fallback"] = backend_fallback
        else:
            candidate = _config_oracle_direction(active_matrix)
        products = wide_matrix[active] @ candidate.double()
        if torch.isfinite(candidate).all() and bool((products > 0).all()):
            selected = candidate
        else:
            fallback = "config_no_strict_common_descent_weighted_sum"
    elif method == "cagrad" and active_count:
        alpha = float(cagrad_alpha)
        if not math.isfinite(alpha) or alpha < 0 or cagrad_rescale not in {0, 1, 2}:
            raise ValueError("invalid CAGrad alpha/rescale")
        # Upstream simplex formulation. Keeping zero objectives in the
        # ensemble preserves the declared arithmetic-mean objective.
        gram = (wide_matrix @ wide_matrix.T).cpu().numpy()
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
    selected_wide = selected.double()
    products = torch.stack([torch.dot(g, selected_wide) for g in wide_matrix]).detach().cpu().tolist()
    diagnostics["combined_dot"] = dict(zip(names, products))
    diagnostics["strict_common_descent"] = all(
        diagnostics["combined_dot"][n] > 0 for n, a in zip(names, active_flags) if a
    ) and bool(active_count)
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
    native_loss: torch.Tensor | None = None,
    coherence_direction_scale: float = 1.0,
    execution: str = "legacy",
    config_backend: str = "legacy",
    diagnostic_active_constraints: tuple[str, ...] | None = None,
    diagnostic_endpoint_loss: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Assign/clip once, then apply exactly one optimizer step.

    All flat gradients share the existing interleaved real/complex layout.
    The final fidelity pressure is never normalized by the coherence combiner.
    """
    parameters = _trainable_parameters(model)
    if execution not in {"legacy", "exact", "scalar"}:
        raise ValueError("gradient execution must be legacy, exact or scalar")
    import math
    if not math.isfinite(coherence_direction_scale) or coherence_direction_scale <= 0:
        raise ValueError("fixed coherence direction scale must be finite and positive")
    if execution != "legacy" and method == "weighted_sum" and not diagnostics:
        if not family_losses:
            raise ValueError("at least one coherence gradient is required")
        scalar = sum(family_losses.values()) * coherence_direction_scale + fidelity_loss
        row = _scalar_objective_update(parameters, optimizer, scalar, grad_clip)
        row.update(update_mode="coherence_primal_dual/weighted_sum", update_accepted=True,
                   **{"update/fallback_reason": None,
                      "gradient/coherence_direction_scale": coherence_direction_scale,
                      "coherence_combiner": {"method": method, "families": list(family_losses),
                                             "fallback_reason": None, "diagnostics_sampled": False}})
        return row

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
        families, method=method, cagrad_alpha=cagrad_alpha, cagrad_rescale=cagrad_rescale,
        config_backend=config_backend,
    )
    import math

    if not math.isfinite(coherence_direction_scale) or coherence_direction_scale <= 0:
        raise ValueError("fixed coherence direction scale must be finite and positive")
    raw_coherence_norm = float(torch.linalg.vector_norm(coherence.double()))
    coherence = coherence * coherence_direction_scale
    final = coherence + fidelity
    row: dict[str, Any] = {"update_mode": f"coherence_primal_dual/{method}",
                          "update_accepted": True, "update/fallback_reason": report["fallback_reason"],
                          "coherence_combiner": report,
                          "combined_grad_norm": float(torch.linalg.vector_norm(final.double())),
                          "fidelity_grad_norm": float(torch.linalg.vector_norm(fidelity.double())),
                          "gradient/coherence_raw_norm": raw_coherence_norm,
                          "gradient/coherence_direction_scale": coherence_direction_scale,
                          "gradient/coherence_calibrated_norm": float(torch.linalg.vector_norm(coherence.double()))}
    packed = None
    if diagnostics and diagnostic_active_constraints is not None:
        # Caller guarantees fidelity_loss is exactly the fixed linear sum of
        # constraint_losses (epoch PI v3). Single-active alias is valid only
        # under this identity, not for nonlinear/projection-based objectives.
        # Caller supplies fixed-coefficient activity; NEVER infer it from a
        # loss value, because an active zero-valued risk can have a gradient.
        terms = constraint_losses or {}
        active = tuple(diagnostic_active_constraints)
        if len(set(active)) != len(active) or not set(active) <= set(terms):
            raise ValueError("active diagnostic constraints must be unique declared term names")
        vectors = {**families, "fidelity": fidelity}
        if native_loss is not None:
            vectors["native"] = objective_gradient(native_loss)
        if diagnostic_endpoint_loss is not None:
            vectors["endpoint.raw"] = objective_gradient(diagnostic_endpoint_loss)
        for name, value in terms.items():
            vectors[f"fidelity/{name}"] = (None if name not in active else
                fidelity if len(active) == 1 else objective_gradient(value))
        vectors.update(combined_coherence=coherence, final=final)
        packed = _packed_diagnostic_gram(vectors)
        for name, gradient in families.items():
            row[f"gradient/{name}/norm"] = packed["norms"][name]
            row[f"gradient/raw_combined_coherence_dot/{name}"] = report["combined_dot"][name]
            row[f"gradient/combined_coherence_dot/{name}"] = coherence_direction_scale * report["combined_dot"][name]
            row[f"gradient/final_direction_dot/{name}"] = packed["dot"](name, "final")
        for name in vectors:
            row[f"gradient/{name}/norm"] = packed["norms"][name]
            row[f"gradient/final_direction_dot/{name}"] = packed["dot"](name, "final")
            for other in vectors:
                key = f"gradient/{name}/{other}/cosine"
                row[key] = packed["cosines"][(name, other)]
                if row[key] is None:
                    row[key + "/undefined_reason"] = "zero_gradient"
        row["diagnostic/active_constraints"] = list(active)
        row["diagnostic/skipped_zero_coefficient_constraints"] = [name for name in terms if name not in active]
        row["diagnostic/reused_fidelity_constraint"] = active[0] if len(active) == 1 else None
        row["diagnostic/gram_host_transfers"] = 1
        row["diagnostic/unique_gram_vectors"] = packed["unique_vectors"]
        before = torch.cat([(torch.view_as_real(p.detach()) if p.is_complex() else p.detach())
                            .reshape(-1).clone() for p in parameters])
    else:
        for name, gradient in families.items():
            row[f"gradient/{name}/norm"] = float(torch.linalg.vector_norm(gradient.double()))
            row[f"gradient/raw_combined_coherence_dot/{name}"] = report["combined_dot"][name]
            row[f"gradient/combined_coherence_dot/{name}"] = coherence_direction_scale * report["combined_dot"][name]
            row[f"gradient/final_direction_dot/{name}"] = float(torch.dot(gradient.double(), final.double()))
        if diagnostics:
            vectors = {**families, "fidelity": fidelity}
            if native_loss is not None:
                vectors["native"] = objective_gradient(native_loss)
                row["gradient/native/norm"] = float(torch.linalg.vector_norm(vectors["native"].double()))
                row["gradient/final_direction_dot/native"] = float(torch.dot(vectors["native"].double(), final.double()))
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
    clip_coefficient = 1.0
    if grad_clip:
        preclip_norm = float(stable_clip_grad_norm_(parameters, float(grad_clip)))
        clip_coefficient = min(1.0, float(grad_clip) / (preclip_norm + 1.0e-6))
    else:
        preclip_norm = row["combined_grad_norm"]
    row.update({
        "gradient/preclip_norm": preclip_norm,
        "gradient/clip_limit": float(grad_clip) if grad_clip else None,
        "gradient/clip_coefficient": clip_coefficient,
        "gradient/clipping_applied": clip_coefficient < 1.0,
    })
    optimizer.step()
    if diagnostics:
        after = torch.cat([(torch.view_as_real(p.detach()) if p.is_complex() else p.detach())
                           .reshape(-1) for p in parameters])
        displacement = after - before
        vectors.update(combined_coherence=coherence, final=final)
        row["update/actual_displacement_norm"] = float(torch.linalg.vector_norm(displacement.double()))
        if packed is not None:
            products = _packed_actual_displacement_products(packed, displacement)
            row.update({f"update/actual_dot/{name}": value for name, value in products.items()})
            row["diagnostic/displacement_host_transfers"] = 1
        else:
            for name, gradient in vectors.items():
                row[f"update/actual_dot/{name}"] = float(torch.dot(gradient.double(), displacement.double()))
    return row


def calibrate_coherence_direction(grams, names, scales, *, cagrad_alpha=.5, cagrad_rescale=1):
    """Fixed median norm matching using exact TRAIN gradient inner products.

    An isometric Gram factor preserves raw combiner norms without retaining
    large parameter vectors. No per-update normalization or optimizer change
    occurs. All methods use the same calibrated source family gradients.
    """
    import statistics

    norms = {method: [] for method in ("weighted_sum", "config", "cagrad")}
    factors = torch.tensor([scales[name] for name in names], dtype=torch.float64)
    for gram in grams:
        raw = torch.as_tensor(gram, dtype=torch.float64)
        if raw.shape != (len(names), len(names)) or not torch.isfinite(raw).all():
            raise ValueError("invalid source TRAIN gradient Gram")
        weighted = raw * factors[:, None] * factors[None, :]
        eigenvalues, eigenvectors = torch.linalg.eigh((weighted + weighted.T) / 2)
        if float(eigenvalues.min()) < -1e-10 * max(1.0, float(weighted.abs().max())):
            raise ValueError("source gradient Gram must be positive semidefinite")
        rows = eigenvectors * eigenvalues.clamp_min(0).sqrt()[None, :]
        bank = dict(zip(names, rows))
        for method, values in norms.items():
            direction, _ = combine_coherence_gradients(
                bank, method=method, cagrad_alpha=cagrad_alpha, cagrad_rescale=cagrad_rescale)
            values.append(float(torch.linalg.vector_norm(direction)))
    if not grams:
        raise ValueError("source TRAIN direction calibration needs batches")
    medians = {method: statistics.median(values) for method, values in norms.items()}
    if any(value <= 1e-12 for value in medians.values()):
        raise ValueError("cannot norm-match a vanishing source coherence direction")
    reference = medians["weighted_sum"]
    return {"version": "train_combined_direction_calibration_v1", "split": "train",
            "gradient_representation": "exact_source_inner_product_isometry",
            "reference_method": "weighted_sum", "reference_norm": reference,
            "raw_norms_by_batch": norms, "raw_median_norms": medians,
            "resolved_scales": {method: reference / value for method, value in medians.items()}}
