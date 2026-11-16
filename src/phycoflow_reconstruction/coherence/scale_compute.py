"""All-pairs signed and rho estimators with complete nonadditive anchors."""

from __future__ import annotations

import math

import torch

from .families.multiscale_spectrum.losses import (
    almost_fair_crps,
    joint_spectral_energy_score,
    paired_joint_spectral_mean_distance,
    vector_energy_score,
)
from .query_sampling import index_pair

VERSION = "round8_all_pairs_scale_v1"
PAIRED_VERSION = "paired_sampled_scale_v1"


def sampled_scale(
    draws,
    target,
    batch,
    *,
    filters,
    scales,
    mass,
    power_floor,
    band_weights,
    self_descriptors,
    target_self_descriptor,
    mean_kwargs,
    mode="conditional",
):
    """Return separate signed, rho, joint and log-power-self [B] blocks.

    draws [J,B,K,C], target [B,K,C]. The two self descriptors must be the
    *complete* normalized log-power vector [J,B,S*C] and [B,S*C].
    ``mean_kwargs`` supplies all frozen arguments to the joint mean anchor.
    The returned blocks are unweighted; caller applies fixed physical weights.
    """
    if draws.ndim != 4 or draws.shape[1:] != target.shape or draws.shape[0] < 1:
        raise ValueError("draws [J,B,K,C] and target [B,K,C] with J>=1 required")
    c = draws.shape[-1]
    if (
        filters.shape[1] != draws.shape[2]
        or scales.shape != (filters.shape[0], c)
        or power_floor.shape != scales.shape
    ):
        raise ValueError("spectral shapes do not match")
    if (
        not torch.isfinite(scales).all()
        or not (scales > 0).all()
        or not torch.isfinite(power_floor).all()
        or not (power_floor > 0).all()
    ):
        raise ValueError("positive finite frozen scales and power floors required")
    if (
        len(batch.self_queries) == 0
        or (c > 1 and not batch.pair_queries)
        or (c == 1 and batch.pair_queries)
    ):
        raise ValueError("query batch does not match population")
    if (
        self_descriptors.shape[:2] != draws.shape[:2]
        or self_descriptors.shape[2:] != target_self_descriptor.shape[1:]
        or self_descriptors.shape[1:] != target_self_descriptor.shape
    ):
        raise ValueError("self anchor descriptor shapes mismatch")
    if target_self_descriptor.ndim != 2 or target_self_descriptor.shape[-1] != filters.shape[0] * c:
        raise ValueError("complete band/field log-power descriptor required")
    bw = band_weights.to(draws)
    if (
        bw.shape != (filters.shape[0],)
        or not torch.isfinite(bw).all()
        or (bw < 0).any()
        or not torch.isclose(bw.sum(), bw.new_tensor(1.0))
    ):
        raise ValueError("band weights must be nonnegative and sum to one")
    f = filters.to(draws)
    s = scales.to(draws)
    norm = mass.to(draws).sum().sqrt()
    if (
        mode not in ("paired", "conditional")
        or (mode == "paired" and len(draws) != 1)
        or (mode == "conditional" and len(draws) < 2)
    ):
        raise ValueError(
            "paired mode needs one prediction; conditional mode needs at least two draws"
        )
    paired = mode == "paired"

    def vector_score(x, y):
        return (
            torch.linalg.vector_norm(x[0] - y, dim=-1) if paired else vector_energy_score(x, y)[0]
        )

    def leaf(a, b=None):
        x = draws[..., a].unsqueeze(-2) / s[:, a, None]  # [J,B,S,K]
        y = target[..., a].unsqueeze(-2) / s[:, a, None]
        if b is None:
            return ((vector_score(x * f / norm, y * f / norm)) * bw).sum(-1)
        xb = draws[..., b].unsqueeze(-2) / s[:, b, None]
        yb = target[..., b].unsqueeze(-2) / s[:, b, None]
        plus = vector_score(
            (x + xb) * f / (math.sqrt(2) * norm), (y + yb) * f / (math.sqrt(2) * norm)
        )
        minus = vector_score(
            (x - xb) * f / (math.sqrt(2) * norm), (y - yb) * f / (math.sqrt(2) * norm)
        )
        return ((plus + minus) * 0.5 * bw).sum(-1)

    self_leaves = {d.index: leaf(d.index) for d in batch.self_queries}
    pair_leaves = {d.index: leaf(*index_pair(c, d.index)) for d in batch.pair_queries}
    signed_self = sum(d.weight * self_leaves[d.index] for d in batch.self_queries)
    signed_pair = sum(
        (d.weight * pair_leaves[d.index] for d in batch.pair_queries),
        start=target.new_zeros(target.shape[0]),
    )
    signed = signed_self / c + (1 - 1 / c) * signed_pair

    def rho(x, a, b):
        xa, xb = x[..., a], x[..., b]
        # [J,B,S] or [B,S]; no dense field-pair Gram.
        pa = torch.einsum("...k,sk->...s", xa.square(), f.square())
        pb = torch.einsum("...k,sk->...s", xb.square(), f.square())
        cross = torch.einsum("...k,sk->...s", xa * xb, f.square())
        return cross / ((pa + power_floor[:, a].to(x)) * (pb + power_floor[:, b].to(x))).sqrt()

    rho_leaves = {
        d.index: (
            (rho(draws, *index_pair(c, d.index))[0] - rho(target, *index_pair(c, d.index))).abs()
            if paired
            else almost_fair_crps(
                rho(draws, *index_pair(c, d.index)), rho(target, *index_pair(c, d.index))
            )
        ).mean(-1)
        for d in batch.pair_queries
    }
    rho_score = None if c == 1 else sum(d.weight * rho_leaves[d.index] for d in batch.pair_queries)
    if paired:
        kwargs = dict(mean_kwargs)
        kwargs["prediction_piece_means"] = kwargs.pop("draw_piece_means")[0]
        joint = paired_joint_spectral_mean_distance(
            draws[0], target, filters=f, scales=s, mass=mass, **kwargs
        )
        self_anchor = torch.linalg.vector_norm(self_descriptors[0] - target_self_descriptor, dim=-1)
    else:
        joint = joint_spectral_energy_score(
            draws, target, filters=f, scales=s, mass=mass, **mean_kwargs
        )[0]
        self_anchor = vector_energy_score(self_descriptors, target_self_descriptor)[0]
    return {
        "version": PAIRED_VERSION if paired else VERSION,
        "signed": signed,
        "rho": rho_score,
        "joint": joint,
        "log_power_self": self_anchor,
        "leaves": {"self": self_leaves, "pair_signed": pair_leaves, "pair_rho": rho_leaves},
    }


def third_order_scores(members, reference, queries, region_masses, scales):
    """Regional centered triple fair leaves on already filtered physical fields.

    Inputs members [J,B,S,N,C], reference [B,S,N,C], queries iterable of
    (region, ((band,field),(band,field),(band,field))), masses [R,N] and
    strictly positive scales [R,S,C]. Distinct fields are required. Returns
    [B,Q], including every member; J=1 is paired scoring.
    """
    if members.ndim != 5 or members.shape[1:] != reference.shape or len(members) < 1:
        raise ValueError("invalid filtered endpoints")
    mass = torch.as_tensor(region_masses, device=members.device, dtype=members.dtype).detach()
    scale = torch.as_tensor(scales, device=members.device, dtype=members.dtype).detach()
    if (
        mass.ndim != 2
        or mass.shape[1] != reference.shape[-2]
        or scale.shape != (mass.shape[0], reference.shape[1], reference.shape[-1])
        or not torch.isfinite(mass).all()
        or (mass < 0).any()
        or (mass.sum(-1) <= 0).any()
        or not torch.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError("invalid region masses or scales")
    mass = mass / mass.sum(-1, keepdim=True)
    leaves = []
    for region, terms in queries:
        if (
            not 0 <= region < len(mass)
            or len(terms) != 3
            or len({field for _, field in terms}) != 3
            or any(
                not (0 <= band < reference.shape[1] and 0 <= field < reference.shape[-1])
                for band, field in terms
            )
        ):
            raise ValueError("invalid regional triple query")
        w = mass[region]

        def descriptor(x, terms=terms, w=w, region=region):
            factors = []
            for band, field in terms:
                value = x[..., band, :, field]
                center = (value * w).sum(-1, keepdim=True)
                factors.append((value - center) / scale[region, band, field])
            return torch.asinh((factors[0] * factors[1] * factors[2] * w).sum(-1))

        x, y = descriptor(members), descriptor(reference)
        leaf = (x - y).abs().mean(0)
        if len(members) > 1:
            leaf = leaf - sum(
                (
                    (x[i] - x[k]).abs()
                    for i in range(len(members))
                    for k in range(i + 1, len(members))
                ),
                torch.zeros_like(leaf),
            ) / (len(members) * (len(members) - 1))
        leaves.append(leaf)
    return torch.stack(leaves, -1) if leaves else members.new_empty((members.shape[1], 0))
