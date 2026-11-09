"""Finite-ensemble scores in fixed descriptor and signed-field representations.

``weighted_distance`` is the norm ``||W^{1/2}(u - v)||_2`` for a fixed positive
diagonal ``W``. Its gradient at coincident vectors is the zero subgradient
(``torch.linalg.vector_norm`` returns exactly zero there), with no smoothing
term, so the energy score below keeps its exact form.
"""

from __future__ import annotations

import torch

from .config import LEGACY_SCORE_VERSION

# The existing public constant continues to identify the mean-free metric.
SCORE_DEFINITION_VERSION = LEGACY_SCORE_VERSION


def vector_energy_score(
    draws: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 1.0,
    pair_chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Euclidean score for draws [J,...,D], target [...,D].

    Returns score, attraction and weighted repulsion with shape [...]. At alpha=1
    this is the fair U-statistic for independent conditional draws. Alpha<1 adds
    a positive dispersion penalty and is not a proper distributional score.
    Chunking limits pairwise temporary storage without detaching any draw.
    """
    if draws.ndim < 2 or draws.shape[1:] != target.shape:
        raise ValueError("energy-score draws [J,...,D] must match target [...,D]")
    count = draws.shape[0]
    factor = almost_fair_repulsion_factor(count, alpha)
    if pair_chunk_size < 1:
        raise ValueError("pair_chunk_size must be positive")
    attraction = torch.linalg.vector_norm(draws - target[None], dim=-1).mean(dim=0)
    upper = torch.triu_indices(count, count, offset=1, device=draws.device)
    pair_sum = torch.zeros_like(attraction)
    for left, right in upper.split(pair_chunk_size, dim=1):
        pair_sum = pair_sum + torch.linalg.vector_norm(draws[left] - draws[right], dim=-1).sum(
            dim=0
        )
    repulsion = factor * pair_sum / (count * (count - 1))
    return attraction - repulsion, attraction, repulsion


def weighted_distance(
    left: torch.Tensor, right: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    return torch.linalg.vector_norm((left - right) * weights.sqrt(), dim=-1)


def paired_descriptor_loss(
    prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Per-state weighted squared descriptor discrepancy ``||Phi(X) - Phi(Y)||_W^2``; [B]."""
    return ((prediction - target).square() * weights).sum(dim=-1)


def conditional_energy_score(
    draws: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unbiased per-state Monte Carlo conditional energy score.

    ``draws`` [B,J,D] are independent reconstructions for the same observations
    and ``target`` [B,D] is the paired reference. Returns
    ``(score, attraction, repulsion)`` with
    ``score = mean_j d(z_j, y) - sum_{j != l} d(z_j, z_l) / (2 J (J - 1))``.
    The repulsive term is what prevents reward for collapsing every draw onto
    one descriptor vector; both draws keep their gradients.
    """
    if draws.ndim != 3 or target.ndim != 2 or draws.shape[::2] != target.shape:
        raise ValueError("energy-score inputs must align as draws [B,J,D] and target [B,D]")
    count = draws.shape[1]
    if count < 2:
        raise ValueError("the conditional energy score needs at least two draws per condition")
    return vector_energy_score(draws.transpose(0, 1) * weights.sqrt(), target * weights.sqrt())


# ----------------------------------------------------------------------------
# Signed-band scores and their legacy almost-fair variant.
#
# ``signed_band_energy_score`` scores, per band, the conditional law of the signed
# band-filtered fields (field axes plus fixed field-space mixtures) with
# an almost-fair two-or-more-draw energy score. ``almost_fair_crps`` scores a
# per-snapshot scalar descriptor (here the signed normalized band coherence).
# Both keep every draw differentiable; the repulsion weight ``1 - (1 - alpha)/J``
# (alpha = 1: the exact fair U-statistic) biases alpha<1 toward reduced spread.
# ----------------------------------------------------------------------------


def almost_fair_repulsion_factor(draws: int, alpha: float) -> float:
    """Weight of the pairwise term: ``1 - eps`` with ``eps = (1 - alpha) / J``."""
    draws = int(draws)
    alpha = float(alpha)
    if draws < 2:
        raise ValueError("almost-fair scores need at least two draws")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("almost_fair_alpha must lie in (0, 1]")
    return 1.0 - (1.0 - alpha) / draws


def field_projections(
    field_count: int,
    kind: str,
    *,
    pairs: list[tuple[int, int]] | None = None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Fixed unit directions in field space, one scalar band field each; [P,C].

    ``axes`` scores every field separately; ``axes_and_pairs`` adds the mixtures
    ``(e_f + e_g)/sqrt2`` and ``(e_f - e_g)/sqrt2`` for every field pair. These
    mixtures add dependence sensitivity, but finitely many projections do not
    identify arbitrary joint laws. A joint Euclidean score does identify the
    represented joint law; simultaneous rotation invariance does not remove its
    sensitivity to changing forecast dependence against a fixed truth.
    """
    axes = torch.eye(int(field_count), dtype=dtype, device=device)
    if kind == "axes":
        return axes
    if kind != "axes_and_pairs":
        raise ValueError(f"unknown field projection kind {kind!r}")
    if pairs is None:
        pairs = [(i, j) for i in range(field_count) for j in range(i + 1, field_count)]
    rows = [axes]
    root = 2.0**0.5
    for left, right in pairs:
        rows.append(((axes[left] + axes[right]) / root)[None])
        rows.append(((axes[left] - axes[right]) / root)[None])
    return torch.cat(rows, dim=0)


def signed_band_energy_score(
    draw_fields: torch.Tensor,
    target_fields: torch.Tensor,
    *,
    mass: torch.Tensor,
    scales: torch.Tensor,
    projections: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-band almost-fair energy score of signed band fields; returns ``[B,S]`` tensors.

    ``draw_fields`` [J,B,S,N,C] are the band fields of J independent conditional
    draws, ``target_fields`` [B,S,N,C] those of the paired target. Fields are divided
    by the frozen band-field scales ``scales`` [S,C]; each projection ``p`` in
    ``projections`` [P,C] defines the scalar band field ``s = sum_f p_f U_bf`` and the
    mass-weighted distance ``d(x, y) = sqrt(sum_i q_i (s(x)_i - s(y)_i)^2)``. The band
    score averages the per-projection scores
    ``mean_j d(x_j, y) - (1 - eps) sum_{j<k} d(x_j, x_k) / (J (J - 1))``.
    Returns ``(score, attraction, repulsion)``, each averaged over projections.
    """
    if draw_fields.ndim != 5 or target_fields.ndim != 4:
        raise ValueError("signed band energy score expects draws [J,B,S,N,C] and target [B,S,N,C]")
    if draw_fields.shape[1:] != target_fields.shape:
        raise ValueError("signed band draws and target shapes differ")
    draws = draw_fields.shape[0]
    if draws < 2:
        raise ValueError("signed band energy score needs at least two draws")
    if scales.shape != draw_fields.shape[2:3] + draw_fields.shape[4:]:
        raise ValueError("band scales must have shape [S,C]")
    if projections.ndim != 2 or projections.shape[1] != draw_fields.shape[-1]:
        raise ValueError("projections must have shape [P,C]")
    weights = (mass / mass.sum()).sqrt().to(draw_fields)
    scaled_draws = draw_fields / scales[None, None, :, None, :]
    scaled_target = target_fields / scales[None, :, None, :]
    projected = torch.einsum("jbsnc,pc->jbspn", scaled_draws, projections.to(draw_fields)) * weights
    target = torch.einsum("bsnc,pc->bspn", scaled_target, projections.to(draw_fields)) * weights
    return tuple(
        value.mean(dim=-1) for value in vector_energy_score(projected, target, alpha=alpha)
    )


def _projected_band_coefficients(
    coefficients: torch.Tensor,
    *,
    filters: torch.Tensor,
    scales: torch.Tensor,
    projections: torch.Tensor,
    mass: torch.Tensor,
) -> torch.Tensor:
    """Embed coefficients as [J,B,S,P,K] in the normalized spatial metric.

    The fitted basis is M-orthonormal. Thus the Euclidean norm on the last
    axis equals the normalized mass-weighted norm of the corresponding signed
    spatial band, including each fixed field projection and scale.
    """
    if coefficients.ndim != 4:
        raise ValueError("band coefficients must have shape [J,B,K,C]")
    if filters.ndim != 2 or filters.shape[1] != coefficients.shape[2]:
        raise ValueError("band filters must have shape [S,K]")
    if scales.shape != (filters.shape[0], coefficients.shape[-1]):
        raise ValueError("band scales must have shape [S,C]")
    if projections.ndim != 2 or projections.shape[1] != coefficients.shape[-1]:
        raise ValueError("projections must have shape [P,C]")
    projected = torch.einsum(
        "jbkc,sc,pc->jbspk",
        coefficients,
        scales.to(coefficients).reciprocal(),
        projections.to(coefficients),
    )
    return (
        projected
        * filters.to(coefficients)[None, None, :, None, :]
        / mass.to(coefficients).sum().sqrt()
    )


def signed_band_coefficient_energy_score(
    draw_coefficients: torch.Tensor,
    target_coefficients: torch.Tensor,
    *,
    filters: torch.Tensor,
    mass: torch.Tensor,
    scales: torch.Tensor,
    projections: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The signed-band energy score in the fitted basis; returns [B,S] tensors.

    This is the same metric as :func:`signed_band_energy_score`, without
    synthesizing spatial band fields. The zero subgradient at coincidence is
    supplied by the same unsmoothed ``vector_norm`` convention.
    """
    if draw_coefficients.ndim != 4 or draw_coefficients.shape[1:] != target_coefficients.shape:
        raise ValueError("signed band coefficients must align as [J,B,K,C] and [B,K,C]")
    # Form differences first. Even a single batched matrix multiplication can
    # round equal rows differently, which otherwise gives a spurious nonzero
    # gradient at coincidence; this linear rearrangement needs no smoothing.
    factor = almost_fair_repulsion_factor(draw_coefficients.shape[0], alpha)
    attraction = torch.linalg.vector_norm(
        _projected_band_coefficients(
            draw_coefficients - target_coefficients.unsqueeze(0),
            filters=filters,
            scales=scales,
            projections=projections,
            mass=mass,
        ),
        dim=-1,
    ).mean(dim=0)
    count = draw_coefficients.shape[0]
    left, right = torch.triu_indices(count, count, offset=1, device=draw_coefficients.device)
    pair_sum = torch.zeros_like(attraction)
    for left_chunk, right_chunk in zip(left.split(32), right.split(32)):
        embedded = _projected_band_coefficients(
            draw_coefficients[left_chunk] - draw_coefficients[right_chunk],
            filters=filters,
            scales=scales,
            projections=projections,
            mass=mass,
        )
        pair_sum = pair_sum + torch.linalg.vector_norm(embedded, dim=-1).sum(dim=0)
    repulsion = factor * pair_sum / (count * (count - 1))
    return tuple(value.mean(dim=-1) for value in (attraction - repulsion, attraction, repulsion))


def paired_signed_band_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    filters: torch.Tensor,
    mass: torch.Tensor,
    scales: torch.Tensor,
    projections: torch.Tensor,
) -> torch.Tensor:
    """Paired projected signed-band point distance; [B,S], with no ensemble claim."""
    if prediction.ndim != 3 or prediction.shape != target.shape:
        raise ValueError("paired band coefficients must have shape [B,K,C]")
    embedded = _projected_band_coefficients(
        (prediction - target).unsqueeze(0),
        filters=filters,
        scales=scales,
        projections=projections,
        mass=mass,
    )
    return torch.linalg.vector_norm(embedded[0], dim=-1).mean(dim=-1)


def joint_spectral_energy_score(
    draws: torch.Tensor,
    target: torch.Tensor,
    *,
    filters: torch.Tensor,
    scales: torch.Tensor,
    mass: torch.Tensor,
    draw_piece_means: torch.Tensor | None = None,
    target_piece_means: torch.Tensor | None = None,
    piece_weights: torch.Tensor | None = None,
    mean_scales: torch.Tensor | None = None,
    mean_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fair score of the full signed-band vector, evaluated in coefficient space.

    Draw coefficients [J,B,K,C] and target [B,K,C] use an M-orthonormal basis.
    For filters [S,K] and frozen scales [S,C], the squared distance equals
    sum_{s,n,c} (mass_n / sum mass) * ((F_s(x)-F_s(y))_nc / scale_sc)^2.
    This retains cross-field and cross-band dependence. Optional piece means are
    appended to the *same* vector, not scored independently: each [P,C] entry
    uses sqrt(mean_weight * piece_weights[P]) / mean_scales[P,C]. Their fixed
    TRAIN scales and positive task-unit floors are determined before scoring.
    Strict propriety applies to this represented vector's law, not excluded
    nodes or unresolved modes. Omitting means preserves the legacy score exactly.
    """
    draw_embedding, target_embedding = _joint_spectral_embeddings(
        draws,
        target,
        filters=filters,
        scales=scales,
        mass=mass,
        draw_piece_means=draw_piece_means,
        target_piece_means=target_piece_means,
        piece_weights=piece_weights,
        mean_scales=mean_scales,
        mean_weight=mean_weight,
    )
    return vector_energy_score(draw_embedding, target_embedding)


def _joint_spectral_embeddings(
    draws: torch.Tensor,
    target: torch.Tensor,
    *,
    filters: torch.Tensor,
    scales: torch.Tensor,
    mass: torch.Tensor,
    draw_piece_means: torch.Tensor | None,
    target_piece_means: torch.Tensor | None,
    piece_weights: torch.Tensor | None,
    mean_scales: torch.Tensor | None,
    mean_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if draws.ndim != 4 or draws.shape[1:] != target.shape:
        raise ValueError("joint spectral score expects coefficients [J,B,K,C] and [B,K,C]")
    if filters.ndim != 2 or filters.shape[1] != target.shape[1]:
        raise ValueError("joint spectral filters must have shape [S,K]")
    if scales.shape != (filters.shape[0], target.shape[-1]):
        raise ValueError("joint spectral scales must have shape [S,C]")
    weights = torch.einsum(
        "sk,sc->kc", filters.to(draws).square(), scales.to(draws).reciprocal().square()
    )
    root = (weights / mass.to(draws).sum()).sqrt()
    draw_embedding = (draws * root).flatten(-2)
    target_embedding = (target * root).flatten(-2)
    mean_args = (draw_piece_means, target_piece_means, piece_weights, mean_scales)
    if any(item is not None for item in mean_args):
        if any(item is None for item in mean_args):
            raise ValueError("joint score needs all piece-mean tensors")
        if (
            draw_piece_means.ndim != 4
            or draw_piece_means.shape[1:] != target_piece_means.shape
            or target_piece_means.shape[0] != target.shape[0]
            or target_piece_means.shape[-1] != target.shape[-1]
            or draw_piece_means.shape[0] != draws.shape[0]
        ):
            raise ValueError("piece means must have shape [J,B,P,C] and [B,P,C]")
        if (
            piece_weights.shape != target_piece_means.shape[1:2]
            or mean_scales.shape != target_piece_means.shape[1:]
        ):
            raise ValueError("piece weights [P] and mean scales [P,C] must align")
        if (
            not torch.isfinite(piece_weights).all()
            or not (piece_weights > 0).all()
            or not torch.isfinite(mean_scales).all()
            or not (mean_scales > 0).all()
            or not 0.0 < mean_weight < float("inf")
        ):
            raise ValueError(
                "piece weights, mean scales and mean weight must be positive and finite"
            )
        weights = piece_weights.to(draws)
        root_mean = (mean_weight * weights / weights.sum()).sqrt()[:, None] / mean_scales.to(draws)
        draw_embedding = torch.cat(
            (draw_embedding, (draw_piece_means * root_mean).flatten(-2)), dim=-1
        )
        target_embedding = torch.cat(
            (target_embedding, (target_piece_means * root_mean).flatten(-2)), dim=-1
        )
    return draw_embedding, target_embedding


def paired_joint_spectral_mean_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    filters: torch.Tensor,
    scales: torch.Tensor,
    mass: torch.Tensor,
    prediction_piece_means: torch.Tensor,
    target_piece_means: torch.Tensor,
    piece_weights: torch.Tensor,
    mean_scales: torch.Tensor,
    mean_weight: float = 1.0,
) -> torch.Tensor:
    """Paired point distance of the joint signed-band and piece-mean vector; [B]."""
    prediction_embedding, target_embedding = _joint_spectral_embeddings(
        prediction.unsqueeze(0),
        target,
        filters=filters,
        scales=scales,
        mass=mass,
        draw_piece_means=prediction_piece_means.unsqueeze(0),
        target_piece_means=target_piece_means,
        piece_weights=piece_weights,
        mean_scales=mean_scales,
        mean_weight=mean_weight,
    )
    return torch.linalg.vector_norm(prediction_embedding[0] - target_embedding, dim=-1)


def band_coherence(gram: torch.Tensor, floor: torch.Tensor) -> torch.Tensor:
    """Signed normalized band coherence of every field pair; [B,S,C,C] -> [B,S,P2].

    ``rho = G_fg / sqrt((G_ff + delta_f)(G_gg + delta_g))`` with the frozen power floors
    ``delta`` [S,C]: bounded in [-1, 1], approximately invariant to positive gains
    only where power dominates the fixed floors, non-trivial
    because the Gram matrix already averages over the modes of a band. Pairs follow
    ``torch.triu_indices(C, C, 1)``.
    """
    power = torch.diagonal(gram, dim1=-2, dim2=-1)
    count = power.shape[-1]
    left, right = torch.triu_indices(count, count, offset=1, device=gram.device)
    denominator = (
        (power[:, :, left] + floor[:, left].to(power))
        * (power[:, :, right] + floor[:, right].to(power))
    ).sqrt()
    return gram[:, :, left, right] / denominator


def almost_fair_crps(
    draw_values: torch.Tensor, target: torch.Tensor, alpha: float = 1.0
) -> torch.Tensor:
    """Almost-fair CRPS of scalar descriptors; ``draw_values`` [J,B,...], ``target`` [B,...]."""
    if draw_values.ndim < 2 or draw_values.shape[1:] != target.shape:
        raise ValueError("almost-fair CRPS expects draws [J,B,...] matching the target [B,...]")
    return vector_energy_score(draw_values[..., None], target[..., None], alpha=alpha)[0]


def selected_band_coherence(statistics, floor):
    """Rho for explicitly declared pairs, without constructing a dense Gram."""
    if "gram" in statistics:
        return band_coherence(statistics["gram"], floor)
    power, pairs = statistics["power"], statistics["pair_indices"]
    left, right = pairs[:, 0], pairs[:, 1]
    denominator = ((power[:, :, left] + floor[:, left].to(power)) *
                   (power[:, :, right] + floor[:, right].to(power))).sqrt()
    return statistics["pair_gram"] / denominator
