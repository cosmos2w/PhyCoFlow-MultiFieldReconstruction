"""Signed-band scoring, spread discrimination and ensemble power diagnostics."""

from __future__ import annotations

import math

import pytest
import torch
from helpers.spectral import (
    _band_noise,
    _config,
    _fitted,
    _grid,
    _signed_config,
    _smooth_states,
)

from phycoflow_reconstruction.coherence.families.multiscale_spectrum import (
    parse_multiscale_config,
)


def test_signed_band_energy_score_is_zero_at_coincidence_separable_and_almost_fair():
    from phycoflow_reconstruction.coherence.families.multiscale_spectrum.losses import (
        almost_fair_repulsion_factor,
        signed_band_energy_score,
    )

    grid = _grid(10)
    states = _smooth_states(grid, 12, fields=3, seed=3)
    family = _fitted(_signed_config(), grid, states)
    coords = grid[None].expand(4, -1, -1)
    target = states[:4]
    exact = family(
        target.clone(), target, coordinates=coords, context={"conditional_draws": (target.clone(),)}
    )
    assert (
        float(
            exact.component_results[
                "multiscale_spectrum.signed_band_energy_score.conditional"
            ].scalar_loss
        )
        == 0.0
    )
    assert (
        float(
            exact.component_results[
                "multiscale_spectrum.band_coherence_crps.conditional"
            ].scalar_loss
        )
        == 0.0
    )
    # The descriptor energy score is reported but carries no weight and no gradient.
    assert exact.component_results[
        "multiscale_spectrum.descriptor_energy_score.conditional"
    ].diagnostics["evaluation_only"]
    assert not exact.component_results[
        "multiscale_spectrum.descriptor_energy_score.conditional"
    ].scalar_loss.requires_grad

    generator = torch.Generator().manual_seed(1)
    draws = torch.randn(2, 3, 2, 7, 3, generator=generator)
    truth = torch.randn(3, 2, 7, 3, generator=generator)
    mass = torch.rand(7, generator=generator) + 0.5
    scales = torch.rand(2, 3, generator=generator) + 0.5
    projections = torch.eye(3)
    score, _attraction, _repulsion = signed_band_energy_score(
        draws, truth, mass=mass, scales=scales, projections=projections, alpha=0.95
    )
    # Manual two-draw almost-fair score for band 0, projection (field) 2.
    q = mass / mass.sum()

    def distance(a, b):
        return torch.sqrt((q * (a - b).square()).sum())

    manual = []
    for s in range(2):
        per_projection = []
        for f in range(3):
            x1, x2, y = (
                draws[0, 0, s, :, f] / scales[s, f],
                draws[1, 0, s, :, f] / scales[s, f],
                truth[0, s, :, f] / scales[s, f],
            )
            factor = almost_fair_repulsion_factor(2, 0.95)
            per_projection.append(
                0.5 * distance(x1, y) + 0.5 * distance(x2, y) - 0.5 * factor * distance(x1, x2)
            )
        manual.append(torch.stack(per_projection).mean())
    assert torch.allclose(score[0], torch.stack(manual), atol=1e-6)
    assert (
        almost_fair_repulsion_factor(2, 1.0) == 1.0
        and abs(almost_fair_repulsion_factor(2, 0.95) - 0.975) < 1e-12
    )
    # Separable across bands: scaling one band's fields leaves the other band's score unchanged.
    scaled = draws.clone()
    scaled[:, :, 1] *= 10.0
    scaled_truth = truth.clone()
    scaled_truth[:, 1] *= 10.0
    other, _, _ = signed_band_energy_score(
        scaled, scaled_truth, mass=mass, scales=scales, projections=projections, alpha=0.95
    )
    assert torch.allclose(other[:, 0], score[:, 0])
    assert torch.allclose(other[:, 1], 10.0 * score[:, 1], rtol=1e-5)


def test_signed_band_score_prefers_calibrated_spread_and_does_not_reward_deterministic_inflation():
    grid = _grid(12)
    states = _smooth_states(grid, 40, fields=3, seed=7)
    family = _fitted(_signed_config(), grid, states)
    descriptor_family = _fitted(
        _config(objective={"kind": "conditional_energy_score", "conditional_draws": 2}),
        grid,
        states,
    )
    count, fine = 16, len(family.settings.bands.names) - 1
    coords = grid[None].expand(count, -1, -1)
    mean = states[:count]
    sigma = float(mean.std()) * 0.5
    truth = mean + sigma * _band_noise(family, grid, count, fine, seed=11)

    def signed_score(first, second):
        result = family(first, truth, coordinates=coords, context={"conditional_draws": (second,)})
        return float(
            result.component_results[
                "multiscale_spectrum.signed_band_energy_score.conditional"
            ].scalar_loss
        )

    # Stochastic generator: independent fine noise at the true amplitude beats the collapsed mean.
    noise_a, noise_b = (
        _band_noise(family, grid, count, fine, seed=21),
        _band_noise(family, grid, count, fine, seed=22),
    )
    collapsed = signed_score(mean.clone(), mean.clone())
    calibrated = signed_score(mean + sigma * noise_a, mean + sigma * noise_b)
    assert calibrated < collapsed
    # Deterministic generator: adding the same fine pattern to both draws only raises the score,
    # while the descriptor score falls (it rewards restoring power in the mean).
    pattern = _band_noise(family, grid, count, fine, seed=31)
    signed_values, descriptor_values = [], []
    for amplitude in (0.0, 0.5 * sigma, sigma):
        draw = mean + amplitude * pattern
        signed_values.append(signed_score(draw, draw.clone()))
        descriptor_values.append(
            float(
                descriptor_family(
                    draw, truth, coordinates=coords, context={"conditional_draws": (draw.clone(),)}
                ).scalar_loss
            )
        )
    assert signed_values[0] < signed_values[1] < signed_values[2]
    assert descriptor_values[2] < descriptor_values[0]


def test_band_coherence_crps_is_gain_invariant_and_sign_sensitive_and_both_draws_get_gradients():
    from phycoflow_reconstruction.coherence.families.multiscale_spectrum.losses import (
        almost_fair_crps,
        band_coherence,
    )

    grid = _grid(10)
    states = _smooth_states(grid, 12, fields=3, seed=5)
    family = _fitted(_signed_config(coherence_crps_weight=0.3), grid, states)
    coords = grid[None].expand(4, -1, -1)
    statistics = family._statistics(states[:4])
    rho = band_coherence(statistics["gram"], family.power_floor)
    gained = states[:4].clone()
    gained[..., 1] *= 3.0
    rho_gained = band_coherence(family._statistics(gained)["gram"], family.power_floor)
    # Exactly gain invariant without floors; the frozen floors only matter where a band's
    # power is within a few floors of zero (the smooth fixture's finest band).
    zero = torch.zeros_like(family.power_floor)
    assert torch.allclose(
        band_coherence(statistics["gram"], zero),
        band_coherence(family._statistics(gained)["gram"], zero),
        atol=1e-5,
    )
    assert torch.allclose(rho, rho_gained, atol=0.05)
    flipped = states[:4].clone()
    flipped[..., 1] *= -1.0
    rho_flipped = band_coherence(family._statistics(flipped)["gram"], family.power_floor)
    # triu order (0,1), (0,2), (1,2): the pairs containing field 1 flip sign, (0,2) does not.
    assert torch.allclose(rho_flipped[..., 0], -rho[..., 0], atol=1e-6)
    assert torch.allclose(rho_flipped[..., 1], rho[..., 1], atol=1e-6)
    assert torch.allclose(rho_flipped[..., 2], -rho[..., 2], atol=1e-6)
    assert float(rho.abs().max()) <= 1.0 + 1e-6
    values = torch.tensor([[[0.2, -0.4]], [[0.6, 0.0]]])
    target = torch.tensor([[0.4, -0.1]])
    expected = (
        0.5 * ((values[0] - target).abs() + (values[1] - target).abs())
        - 0.5 * 0.975 * (values[0] - values[1]).abs()
    )
    assert torch.allclose(almost_fair_crps(values, target, 0.95), expected)
    first = (states[4:8] + 0.2 * torch.randn(4, grid.shape[0], 3)).requires_grad_(True)
    second = (states[8:12] + 0.2 * torch.randn(4, grid.shape[0], 3)).requires_grad_(True)
    result = family(first, states[:4], coordinates=coords, context={"conditional_draws": (second,)})
    coherence = result.component_results["multiscale_spectrum.band_coherence_crps.conditional"]
    assert coherence.scalar_loss.requires_grad and "evaluation_only" not in coherence.diagnostics
    result.scalar_loss.backward()
    assert float(first.grad.norm()) > 0 and float(second.grad.norm()) > 0
    assert math.isclose(
        float(result.scalar_loss),
        float(
            result.component_results[
                "multiscale_spectrum.signed_band_energy_score.conditional"
            ].scalar_loss
        )
        + 0.3 * float(coherence.scalar_loss),
        rel_tol=1e-6,
    )


def test_signed_band_ensemble_report_splits_member_power_into_shared_and_varying():
    grid = _grid(10)
    states = _smooth_states(grid, 24, fields=3, seed=9)
    family = _fitted(_signed_config(), grid, states)
    coords = grid[None].expand(4, -1, -1)
    mean, truth = states[:4], states[4:8]
    fine = len(family.settings.bands.names) - 1
    draws = [mean + 0.5 * _band_noise(family, grid, 4, fine, seed=100 + k) for k in range(6)]
    result = family(
        draws[0],
        truth,
        coordinates=coords,
        context={
            "conditional_draws": (draws[1],),
            "report_detail": True,
            "ensemble_draws": tuple(draws[1:]),
        },
    )
    report = result.diagnostics["report"]["band_ensemble"]
    assert report["members"] == 6
    member = torch.tensor(report["member_power_over_truth"])
    shared = torch.tensor(report["shared_power_over_truth"])
    varying = torch.tensor(report["varying_power_over_truth"])
    # Sample identity: mean member power = debiased shared power + unbiased spread.
    assert torch.allclose(member, shared + varying, rtol=1e-4, atol=1e-6)
    # The injected noise lives in the finest band (and its overlapping neighbour of the tight
    # frame): bands further away see no spread.
    assert float(varying[: max(fine - 1, 0)].abs().max()) < 1e-5 and float(varying[fine].min()) > 0


def test_signed_band_objective_config_contract():
    with pytest.raises(ValueError, match="standardize_descriptors must be false"):
        parse_multiscale_config(_signed_config(standardize_descriptors=True), ("u", "v", "w"))
    with pytest.raises(ValueError, match="apply to signed_band_energy_score only"):
        parse_multiscale_config(
            _config(
                objective={
                    "kind": "conditional_energy_score",
                    "conditional_draws": 2,
                    "almost_fair_alpha": 0.9,
                }
            ),
            ("u", "v", "w"),
        )
    for bad in (
        {"almost_fair_alpha": 1.2},
        {"projections": "random"},
        {"band_weights": [1.0]},
        {"band_weights": [1.0, -1.0, 1.0, 1.0]},
        {"coherence_crps_weight": -0.1},
        {"conditional_draws": 1},
    ):
        with pytest.raises((ValueError, TypeError)):
            parse_multiscale_config(_signed_config(**bad), ("u", "v", "w"))
    settings = parse_multiscale_config(_signed_config(), ("u", "v", "w"))
    assert settings.almost_fair_alpha == 0.95 and settings.projections == "axes_and_pairs"
    assert settings.band_weights is None and settings.coherence_crps_weight == 0.0
