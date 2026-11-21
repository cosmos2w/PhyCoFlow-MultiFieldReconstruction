"""Finite-law SSS teacher scoring and native physical feasibility.

The caller owns sampling, immutable SOURCE, bank identities, and checkpointing.
The teacher is deliberately separate from independent-draw fair assessment.
"""
from __future__ import annotations

import torch

FAMILIES = ("global_distribution", "multiscale_spectrum", "topology")


def empirical_family_scores(runtime, native_members, native_reference, support_operator, *, coordinates, context):
    """Return family V scores [B] for K4 native atoms on a frozen support.

    ``support_operator`` maps either [B,N,C] or [K,B,N,C] to the exact
    frozen scoring support and must preserve autograd. ``runtime.begin_step``
    must already have fixed the query. Coordinates and context describe that
    scoring support. All four atoms and six pairs use that same query.

    The empirical reduction uses all four live endpoint transforms and all six
    pair distances, including expanded leaves and non-compiled Shape.
    """
    if native_members.ndim != 4 or native_members.shape[0] != 4 or native_members.shape[1:] != native_reference.shape:
        raise ValueError("teacher requires four native atoms [4,B,N,C]")
    if runtime.batch is None or runtime.scoring_mode != "conditional":
        raise ValueError("a pending conditional runtime query is required")
    members = support_operator(native_members)
    reference = support_operator(native_reference)
    if members.ndim != 4 or members.shape[0] != 4 or members.shape[1:] != reference.shape:
        raise ValueError("support operator changed the member or condition dimensions")
    if any(int(runtime.families[name].conditional_draws) != 2 for name in FAMILIES if hasattr(runtime.families[name], "conditional_draws")):
        raise ValueError("teacher pair reduction requires the qualified two-draw runtime")
    output = {}
    difficulties = {}
    scoring_context = {**context, "conditional_draws": tuple(members[1:]),
                       "score_all_conditional_draws": True, "_empirical_teacher": True}
    for name in FAMILIES:
        result, _, observed = runtime._score(name, runtime.families[name], members[0],
                                              reference, coordinates, scoring_context, runtime.batch)
        output[name] = result.per_sample_cost
        if observed is not None:
            difficulties[name] = observed
    if hasattr(runtime, "difficulties"):
        runtime.difficulties.update(difficulties)
    return output


def normalized_teacher_loss(scores, normalizers):
    """Per-condition mean of the three positive frozen SOURCE V units."""
    if set(scores) != set(FAMILIES) or set(normalizers) != set(FAMILIES):
        raise ValueError("all three teacher families and normalizers are required")
    values = []
    for name in FAMILIES:
        unit = float(normalizers[name])
        if not (0 < unit < float("inf")):
            raise ValueError("teacher normalizers must be finite and positive")
        values.append(scores[name] / unit)
    return torch.stack(values).mean(0)


def teacher_step(source, current, reference, masses, epsilon, observed_mask, loss_fn):
    """One SOURCE-anchored, per-condition clipped feasible endpoint update.

    Arrays use [4,B,N,C], [B,N,C], masses [N] or [B,N], epsilon [C]
    or [B,C] in squared native field units, and observed mask [B,N,C].
    Zero-mass sites and observations stay exactly at SOURCE. ``loss_fn`` takes
    native physical fields and returns one score per condition; it must not mix
    conditions. All feasibility arithmetic and returned values are float64.
    """
    x = source.detach().to(dtype=torch.float64)
    y = current.detach().to(device=x.device, dtype=x.dtype)
    z = reference.detach().to(device=x.device, dtype=x.dtype)
    if x.ndim != 4 or x.shape[0] != 4 or y.shape != x.shape or z.shape != x.shape[1:]:
        raise ValueError("incompatible teacher field shapes")
    k, b, n, c = x.shape
    w = torch.as_tensor(masses, device=x.device, dtype=x.dtype)
    if w.ndim == 1:
        w = w.expand(b, -1)
    e = torch.as_tensor(epsilon, device=x.device, dtype=x.dtype)
    if e.ndim == 1:
        e = e.expand(b, -1)
    obs = torch.as_tensor(observed_mask, device=x.device, dtype=torch.bool)
    if w.shape != (b, n) or e.shape != (b, c) or obs.shape != (b, n, c):
        raise ValueError("incompatible physical mass, allowance or observation shape")
    if not all(torch.isfinite(t).all() for t in (x, y, z, w, e)) or (w < 0).any() or (w.sum(-1) <= 0).any() or (e <= 0).any():
        raise ValueError("finite fields, nonnegative masses and positive allowances required")
    w = w / w.sum(-1, keepdim=True)
    free = (w > 0).unsqueeze(-1) & ~obs
    if not torch.equal(torch.where(free.unsqueeze(0), x, y), x):
        raise ValueError("current teacher must preserve observed and invalid entries")
    sqrt_e = e.sqrt()[None, :, None, :]
    u = ((y - x) / sqrt_e).detach().requires_grad_(True)
    losses = loss_fn(x + sqrt_e * u)
    if losses.shape != (b,) or not torch.isfinite(losses).all():
        raise ValueError("teacher loss must return finite per-condition scores")
    derivative, = torch.autograd.grad(losses.sum(), u)
    safe_w = torch.where(w > 0, w, torch.ones_like(w))
    gradient = derivative * (k * c) / safe_w[None, :, :, None]
    gradient = torch.where(free.unsqueeze(0), gradient, torch.zeros_like(gradient))
    if not torch.isfinite(gradient).all():
        raise ValueError("nonfinite teacher gradient")
    norm = (gradient.square() * w[None, :, :, None]).sum((0, 2, 3)).div(k * c).sqrt()
    gradient = gradient / norm.clamp_min(1)[None, :, None, None]
    delta = (sqrt_e * (u.detach() - gradient)).detach()
    delta = torch.where(free.unsqueeze(0), delta, torch.zeros_like(delta))
    if not torch.isfinite(delta).all():
        raise ValueError("nonfinite teacher proposal")

    def risks(atoms):
        err = atoms - z.unsqueeze(0)
        mean = (err.mean(0).square() * w[:, :, None]).sum(1)
        member = (err.square() * w[None, :, :, None]).sum(2).mean(0)
        displacement = ((atoms - x).square() * w[None, :, :, None]).sum(2).mean(0)
        return mean, member, displacement

    prior_mean, prior_member, prior_displacement = risks(y)
    source_mean, source_member, _ = risks(x)
    if not ((prior_mean <= source_mean + e).all() and
            (prior_member <= source_member + e).all() and
            (prior_displacement <= e).all()):
        raise ValueError("current teacher is not feasible")

    # Each constraint is A alpha^2 + B alpha - epsilon <= 0,
    # because the immutable SOURCE is the reference point on every ray.
    error = x - z.unsqueeze(0)
    mean_error, mean_delta = error.mean(0), delta.mean(0)
    wm = w[:, :, None]
    wk = w[None, :, :, None]
    coefficients = (
        ((mean_delta.square() * wm).sum(1), (2 * mean_error * mean_delta * wm).sum(1)),
        ((delta.square() * wk).sum(2).mean(0), (2 * error * delta * wk).sum(2).mean(0)),
        ((delta.square() * wk).sum(2).mean(0), torch.zeros_like(e)),
    )
    bounds = []
    for quadratic, linear in coefficients:
        discriminant = (linear.square() + 4 * quadratic * e).sqrt()
        # Stable positive root for both signs, including zero curvature.
        safe_quadratic = torch.where(quadratic > 0, quadratic, torch.ones_like(quadratic))
        safe_denom = torch.where(discriminant + linear > 0, discriminant + linear, torch.ones_like(linear))
        root_positive = 2 * e / safe_denom
        root_negative = (discriminant - linear) / (2 * safe_quadratic)
        root = torch.where(linear >= 0, root_positive, root_negative)
        root = torch.where((quadratic == 0) & (linear <= 0), torch.full_like(root, float("inf")), root)
        bounds.append(root.amin(-1))
    alpha = torch.stack(bounds).amin(0).clamp(max=1)
    if not torch.isfinite(alpha).all() or (alpha < 0).any():
        raise ValueError("invalid teacher quadratic boundary")
    # Reserve a few representable values at an active boundary so a saved
    # correction can be added back to SOURCE without crossing it by rounding.
    for _ in range(8):
        alpha = torch.where(alpha < 1, torch.nextafter(alpha, torch.zeros_like(alpha)), alpha)
    limits = (source_mean + e, source_member + e, e)
    for _ in range(8):
        result = x + alpha[None, :, None, None] * delta
        actual = risks(result)
        feasible = torch.stack([(value <= limit).all(-1) for value, limit in zip(actual, limits)]).all(0)
        if feasible.all():
            break
        alpha = torch.where(feasible, alpha, torch.nextafter(alpha, torch.zeros_like(alpha)))
    else:
        raise ValueError("numerical feasibility failure")
    return result.detach(), {"alpha": alpha.detach(), "gradient_norm": norm.detach(),
                             "mean_error": risks(result)[0].detach(),
                             "member_error": risks(result)[1].detach(),
                             "displacement": risks(result)[2].detach()}
