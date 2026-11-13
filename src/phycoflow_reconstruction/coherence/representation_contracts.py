"""Pointwise representation and physical-threshold qualification utilities."""

import torch


def retained_scale_constant(filters, scales, piece_masses, mean_scales, mean_weight=1.0):
    """Return c_min for all declared M-orthonormal modes and piece means.

    The physical norm is sum(m*x**2)/sum(m), summed over fields. Piece
    masses must partition the same physical mass used for spectral coefficients.
    Zero modal weights remain included and make the inverse bound unavailable.
    """
    f = torch.as_tensor(filters, dtype=torch.float64)
    s = torch.as_tensor(scales, dtype=torch.float64)
    pm = torch.as_tensor(piece_masses, dtype=torch.float64)
    ms = torch.as_tensor(mean_scales, dtype=torch.float64)
    if (
        f.ndim != 2
        or s.ndim != 2
        or s.shape[0] != f.shape[0]
        or pm.ndim != 1
        or ms.shape != (len(pm), s.shape[1])
        or len(pm) == 0
        or not all(torch.isfinite(x).all() for x in (f, s, pm, ms))
        or (s <= 0).any()
        or (pm <= 0).any()
        or (ms <= 0).any()
        or not 0 < mean_weight < float("inf")
    ):
        raise ValueError("invalid retained Scale geometry or normalization")
    modal = torch.einsum("sk,sc->kc", f.square(), s.reciprocal().square())
    means = mean_weight / ms.square()
    minimum = min(float(modal.min()) if modal.numel() else float("inf"), float(means.min()))
    return {
        "c_min": minimum,
        "modal_weights": modal,
        "mean_weights": means,
        "inverse_available": minimum > 0,
    }


def task_distance_bound(embedding_distance, unresolved_norm, c_min, lipschitz=1.0):
    """Lipschitz bound from pointwise distance; not from an energy score."""
    vals = [float(embedding_distance), float(unresolved_norm), float(c_min), float(lipschitz)]
    if any(not 0 <= v < float("inf") for v in vals) or c_min <= 0:
        raise ValueError("finite nonnegative arguments and positive c_min required")
    return lipschitz * (embedding_distance / c_min**0.5 + unresolved_norm)


def threshold_margin_bound(prediction, reference, masses, thresholds, margins, *, operators=None):
    """Bound a physical threshold intersection with explicit cut directions.

    Operators are ordered with the fields and preserve strict equality behavior.
    The default is >= for each field; event callers must declare their cuts.
    """
    x, z = torch.as_tensor(prediction), torch.as_tensor(reference)
    m = torch.as_tensor(masses, device=z.device, dtype=z.dtype)
    t = torch.as_tensor(thresholds, device=z.device, dtype=z.dtype)
    g = torch.as_tensor(margins, device=z.device, dtype=z.dtype)
    if (
        x.shape != z.shape
        or z.ndim != 2
        or m.shape != z.shape[:1]
        or t.shape != z.shape[1:]
        or g.shape != t.shape
        or not all(torch.isfinite(v).all() for v in (x, z, m, t, g))
        or (m < 0).any()
        or m.sum() <= 0
        or (g <= 0).any()
    ):
        raise ValueError("invalid threshold population")
    operators = tuple(operators) if operators is not None else (">=",) * z.shape[1]
    if len(operators) != z.shape[1] or any(op not in {">", ">=", "<", "<="} for op in operators):
        raise ValueError("one supported threshold operator is required per field")
    def event(value):
        comparisons = {
            ">": torch.gt, ">=": torch.ge, "<": torch.lt, "<=": torch.le,
        }
        return torch.stack([
            comparisons[op](value[:, index], t[index])
            for index, op in enumerate(operators)
        ], dim=-1).all(-1)
    m = m / m.sum()
    near = ((z - t).abs() <= g).any(-1)
    margin_mass = (m * near).sum()
    markov = (((x - z).square() * m[:, None]).sum(0) / g.square()).sum()
    disagreement = (m * (event(x) != event(z))).sum()
    return {
        "margin_mass": margin_mass,
        "markov": markov,
        "bound": margin_mass + markov,
        "disagreement": disagreement,
    }
