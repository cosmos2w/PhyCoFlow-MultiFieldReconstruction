"""Linear physical-measure field risk with explicit sampling correction."""

from __future__ import annotations

import torch
from torch import Tensor


def physical_field_mse(
    prediction: Tensor,
    reference: Tensor,
    masses: Tensor,
    *,
    total_mass: Tensor | float | None = None,
    sampling: str = "full",
    probabilities: Tensor | float | None = None,
) -> Tensor:
    """Return [..., C] mean squared error over the native physical measure.

    Inputs broadcast as [..., Q, C]; selected unnormalized physical ``masses``
    broadcast as [..., Q]. Full support may infer total mass. A sparse sample
    must supply its *native* total mass, never the random selected mass sum.
    Without-replacement probabilities are inclusion probabilities; with-
    replacement probabilities are the proposal probabilities of each draw.
    Repeated draws contribute separately in the latter estimator.
    """
    if sampling not in {"full", "without_replacement", "with_replacement"}:
        raise ValueError("unknown spatial sampling design")
    prediction, reference = torch.broadcast_tensors(prediction, reference)
    if prediction.ndim < 2 or not prediction.is_floating_point() or not prediction.shape[-2]:
        raise ValueError("floating [..., Q, C] fields required")
    error = prediction - reference
    return _integral(
        error.square(),
        masses,
        total_mass=total_mass,
        sampling=sampling,
        probabilities=probabilities,
    )


def _integral(values, masses, *, total_mass, sampling, probabilities):
    if sampling not in {"full", "without_replacement", "with_replacement"}:
        raise ValueError("unknown spatial sampling design")
    error = values
    if not torch.isfinite(error).all():
        raise ValueError("finite field values required")
    mass = torch.as_tensor(masses, device=error.device, dtype=error.dtype).detach()
    try:
        mass = torch.broadcast_to(mass, error.shape[:-1])
    except RuntimeError as exc:
        raise ValueError("physical masses must align with sampled sites") from exc
    if not torch.isfinite(mass).all() or (mass < 0).any():
        raise ValueError("finite nonnegative physical masses required")
    if total_mass is None:
        if sampling != "full":
            raise ValueError("sampled risk requires the native total mass")
        total_mass = mass.sum(-1)
    total = torch.as_tensor(total_mass, device=error.device, dtype=error.dtype).detach()
    try:
        total = torch.broadcast_to(total, error.shape[:-2])
    except RuntimeError as exc:
        raise ValueError("native total mass must align with field batches") from exc
    if not torch.isfinite(total).all() or (total <= 0).any():
        raise ValueError("positive finite native total mass required")
    if sampling == "full":
        if probabilities is not None:
            raise ValueError("full quadrature does not accept sampling probabilities")
    else:
        if probabilities is None:
            raise ValueError("sampled risk requires actual sampling probabilities")
        probability = torch.as_tensor(
            probabilities, device=error.device, dtype=error.dtype
        ).detach()
        try:
            probability = torch.broadcast_to(probability, mass.shape)
        except RuntimeError as exc:
            raise ValueError("sampling probabilities must align with sampled sites") from exc
        if (
            not torch.isfinite(probability).all()
            or (probability <= 0).any()
            or (probability > 1).any()
        ):
            raise ValueError("sampling probabilities must lie in (0, 1]")
        mass = mass / probability
        if sampling == "with_replacement":
            mass = mass / error.shape[-2]
    return (error * mass[..., None]).sum(-2) / total[..., None]


def blend_risk_quadratic(
    source: Tensor, child: Tensor, reference: Tensor, masses: Tensor, **sampling
) -> tuple[Tensor, Tensor, Tensor]:
    """Return R0, B, A for exact R(alpha) = R0 + 2 alpha B + alpha² A.

    This computes the risk polynomial on the declared matched finite support;
    selecting/qualifying alpha and held-out assessment belong to the case.
    """
    residual, change = source - reference, child - source
    zero = torch.zeros_like(residual)
    r0 = physical_field_mse(residual, zero, masses, **sampling)
    a = physical_field_mse(change, zero, masses, **sampling)
    b = _integral(
        residual * change,
        masses,
        total_mass=sampling.get("total_mass"),
        sampling=sampling.get("sampling", "full"),
        probabilities=sampling.get("probabilities"),
    )
    return r0, b, a
