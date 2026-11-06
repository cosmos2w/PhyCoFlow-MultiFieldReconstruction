"""Spectral family descriptors, invariances, calibration and artifact contracts."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from helpers.spectral import (
    _config,
    _descriptor_blocks,
    _family,
    _fitted,
    _geometry,
    _grid,
    _smooth_states,
    _third,
    _u_shaped_grid,
)

from phycoflow_reconstruction.coherence import build_coherence_family
from phycoflow_reconstruction.coherence.families.cross_spectrum.statistics import (
    graph_fourier,
    spectral_coherence,
)
from phycoflow_reconstruction.coherence.families.multiscale_spectrum import (
    MultiscaleSpectrumFamily,
    parse_multiscale_config,
)
from phycoflow_reconstruction.coherence.families.multiscale_spectrum.losses import (
    conditional_energy_score,
    weighted_distance,
)
from phycoflow_reconstruction.coherence.families.multiscale_spectrum.statistics import (
    regional_centered,
    third_central_moments,
)
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def test_identity_is_zero_and_per_state_descriptors_ignore_batch_composition():
    coordinates = _grid(10)
    states = _smooth_states(coordinates, 12)
    family = _fitted(_config(), coordinates, states)
    batch = states[:4]
    result = family(batch, batch, coordinates=coordinates.expand(4, -1, -1))
    assert result.scalar_loss.item() == 0.0
    assert all(term.scalar_loss.item() == 0.0 for term in result.component_results.values())
    alone = family.descriptors(batch[:1], coordinates=coordinates[None])
    grouped = family.descriptors(batch, coordinates=coordinates.expand(4, -1, -1))
    regrouped = family.descriptors(batch.flip(0), coordinates=coordinates.expand(4, -1, -1))
    torch.testing.assert_close(alone[0], grouped[0])
    torch.testing.assert_close(grouped, regrouped.flip(0))


def test_gain_error_changes_log_power_but_not_normalized_band_coherence():
    coordinates = _grid(10)
    states = _smooth_states(coordinates, 12)
    family = _fitted(_config(), coordinates, states)
    reference = states[:3]
    doubled = reference.clone()
    doubled[..., 0] *= 2.0
    power = _descriptor_blocks(family, reference)["log_band_power"].reshape(3, 3, 3)
    change = _descriptor_blocks(family, doubled)["log_band_power"].reshape(3, 3, 3) - power
    # Field u gains ~log(4) where its power is well above the floor; v and w are unchanged.
    well_resolved = power[..., 0] > 4.0
    assert well_resolved.any()
    torch.testing.assert_close(
        change[..., 0][well_resolved],
        torch.full_like(change[..., 0][well_resolved], math.log(4.0)),
        atol=2e-2,
        rtol=0.0,
    )
    torch.testing.assert_close(change[..., 1:], torch.zeros_like(change[..., 1:]))
    report = family(
        reference,
        doubled,
        coordinates=coordinates.expand(3, -1, -1),
        context={"report_detail": True},
    ).diagnostics["report"]

    def as_array(rows):
        return np.array([[np.nan if value is None else value for value in row] for row in rows])

    np.testing.assert_allclose(
        as_array(report["reference"]["normalized_band_coherence_mean"]),
        as_array(report["prediction"]["normalized_band_coherence_mean"]),
        atol=1e-5,
        equal_nan=True,
    )


def test_relative_sign_reversal_flips_signed_cospectra_but_not_legacy_squared_coherence():
    coordinates = _grid(10)
    states = _smooth_states(coordinates, 12)
    config = _config(
        components={
            "signed_band_cospectrum": {"enabled": True, "weight": 1.0},
            "third_order_product_moments": _third(
                [[["u", 0], ["v", 1], ["w", 2]], [["u", 0], ["u", 1], ["w", 2]]]
            ),
        },
    )
    family = _fitted(config, coordinates, states)
    reference = states[:4]
    flipped = reference.clone()
    flipped[..., 1] *= -1.0
    blocks = _descriptor_blocks(family, reference)
    mirrored = _descriptor_blocks(family, flipped)
    # Pairs (u,v), (u,w), (v,w): those containing v flip sign; (u,w) does not.
    cospectrum = blocks["signed_band_cospectrum"].reshape(4, 3, 3)
    flipped_cospectrum = mirrored["signed_band_cospectrum"].reshape(4, 3, 3)
    torch.testing.assert_close(flipped_cospectrum[..., [0, 2]], -cospectrum[..., [0, 2]])
    torch.testing.assert_close(flipped_cospectrum[..., 1], cospectrum[..., 1])
    third = blocks["third_order_product_moments"]
    torch.testing.assert_close(mirrored["third_order_product_moments"][:, 0], -third[:, 0])
    torch.testing.assert_close(mirrored["third_order_product_moments"][:, 1], third[:, 1])
    assert cospectrum[..., 0].abs().max() > 0.05
    legacy_basis = family.basis.to(torch.float32)
    legacy = spectral_coherence(graph_fourier(reference.float(), legacy_basis), 1e-8)
    legacy_flipped = spectral_coherence(graph_fourier(flipped.float(), legacy_basis), 1e-8)
    torch.testing.assert_close(legacy, legacy_flipped)


def test_constant_offsets_are_invisible_to_every_descriptor():
    coordinates = _grid(10)
    states = _smooth_states(coordinates, 12)
    config = _config(
        components={
            "log_band_power": {"enabled": True, "weight": 1.0},
            "signed_band_cospectrum": {"enabled": True, "weight": 1.0},
            "third_order_product_moments": _third([[["u", 0], ["v", 1], ["w", 2]]]),
        }
    )
    family = _fitted(config, coordinates, states)
    shifted = states[:3] + torch.tensor([0.7, -1.2, 5.0], dtype=torch.float64)
    grid = coordinates.expand(3, -1, -1)
    torch.testing.assert_close(
        family.descriptors(shifted, coordinates=grid),
        family.descriptors(states[:3], coordinates=grid),
    )


def test_tight_bank_energy_identity_and_reported_unresolved_energy():
    coordinates = _grid(12)
    states = _smooth_states(coordinates, 8)
    exact = {"numerics": {"dtype": "float64"}}
    # A cutoff far above the mesh's finest mode: two bands, no tapered mode.
    resolved = _fitted(
        _config(bands={"count": 2, "min_wavelength": 1e-3}, **exact), coordinates, states
    )
    truncated = _fitted(_config(bands={"count": 3}, **exact), coordinates, states)
    assert resolved.basis.shape[1] == coordinates.shape[0] - 1
    assert truncated.geometry_record["tapered_modes"] > 0
    for family in (resolved, truncated):
        statistics = family._statistics(states)
        power = torch.diagonal(statistics["gram"], dim1=-2, dim2=-1).sum(dim=1)
        centered = states - states.mean(dim=1, keepdim=True)
        weighted_mean = torch.einsum("n,bnc->bc", family.mass, states)
        centered = states - weighted_mean[:, None, :]
        coefficients = torch.einsum("nk,bnc->bkc", family.basis * family.mass[:, None], centered)
        coverage = family.filters.square().sum(dim=0)
        torch.testing.assert_close(
            power, torch.einsum("k,bkc->bc", coverage, coefficients.square())
        )
    full = resolved._statistics(states)
    torch.testing.assert_close(
        torch.diagonal(full["gram"], dim1=-2, dim2=-1).sum(dim=1), full["fluctuation_power"]
    )
    assert resolved.calibration_record["reference_resolved_fraction"] == pytest.approx([1.0] * 3)
    assert all(
        0.0 < value < 1.0 for value in truncated.calibration_record["reference_resolved_fraction"]
    )


def test_descriptors_are_invariant_to_eigenvector_signs_and_degenerate_rotations():
    coordinates = _grid(9)
    states = _smooth_states(coordinates, 10)
    config = _config(
        numerics={"dtype": "float64"},
        regions={"grid": [2, 2]},
        components={
            "log_band_power": {"enabled": True, "weight": 1.0},
            "signed_band_cospectrum": {"enabled": True, "weight": 1.0},
            "third_order_product_moments": _third([[["u", 0], ["v", 1], ["v", 1]]]),
        },
    )
    family = _fitted(config, coordinates, states)
    values = family.eigenvalues.numpy()
    reference = family.descriptors(states[:3], coordinates=coordinates.expand(3, -1, -1))
    clusters = np.split(
        np.arange(values.size), np.flatnonzero(np.diff(values) > 1e-8 * values[1:]) + 1
    )
    assert any(len(cluster) > 1 for cluster in clusters)
    rotation = np.zeros((values.size, values.size))
    generator = np.random.default_rng(3)
    for cluster in clusters:
        block, _ = np.linalg.qr(generator.standard_normal((len(cluster), len(cluster))))
        rotation[np.ix_(cluster, cluster)] = block * generator.choice([-1.0, 1.0], len(cluster))
    family.basis = family.basis @ torch.from_numpy(rotation)
    rotated = family.descriptors(states[:3], coordinates=coordinates.expand(3, -1, -1))
    torch.testing.assert_close(rotated, reference, atol=1e-9, rtol=0.0)


def test_consistent_node_permutation_leaves_global_descriptors_unchanged():
    coordinates = _grid(10, jitter=0.3, seed=5)
    states = _smooth_states(coordinates, 10)
    config = _config(
        numerics={"dtype": "float64"},
        components={
            "log_band_power": {"enabled": True, "weight": 1.0},
            "signed_band_cospectrum": {"enabled": True, "weight": 1.0},
            "third_order_product_moments": _third([[["u", 0], ["v", 1], ["w", 1]]]),
        },
    )
    permutation = torch.from_numpy(np.random.default_rng(11).permutation(coordinates.shape[0]))
    family = _fitted(config, coordinates, states)
    permuted = _fitted(config, coordinates[permutation], states[:, permutation])
    original = family.descriptors(states[:4], coordinates=coordinates.expand(4, -1, -1))
    moved = permuted.descriptors(
        states[:4, permutation], coordinates=coordinates[permutation].expand(4, -1, -1)
    )
    torch.testing.assert_close(moved, original, atol=1e-8, rtol=1e-8)


def test_zero_power_prediction_is_finite_and_coherence_is_marked_undefined():
    coordinates = _grid(8)
    states = _smooth_states(coordinates, 6)
    family = _fitted(_config(), coordinates, states)
    flat = torch.zeros_like(states[:2]) + 0.25
    result = family(
        flat, states[:2], coordinates=coordinates.expand(2, -1, -1), context={"report_detail": True}
    )
    assert torch.isfinite(result.scalar_loss)
    report = result.diagnostics["report"]["prediction"]
    assert all(value is None for row in report["normalized_band_coherence_mean"] for value in row)
    assert all(
        value == 0.0 for row in report["normalized_band_coherence_valid_fraction"] for value in row
    )
    assert all(abs(value) < 1e-10 for row in report["band_power_mean"] for value in row)


def test_phase_locked_triad_is_detected_and_quadrature_sign_is_not():
    count = 512
    coordinates = torch.arange(count, dtype=torch.float64)[:, None] / count
    x = 2.0 * math.pi * coordinates[:, 0]

    def triad(theta: float) -> torch.Tensor:
        return torch.stack((torch.cos(2 * x), torch.cos(7 * x), torch.cos(9 * x + theta)), dim=-1)

    config = {
        "fields": ["u", "v", "w"],
        "geometry": {
            "operator": "fem_p1",
            "coordinate_scaling": [2.0 * math.pi],
            "length_unit": "radian_length",
            "periodic": True,
            "periods": [1.0],
        },
        "bands": {
            "wavelengths": [math.pi, 2 * math.pi / 7, 2 * math.pi / 9],
            "names": ["k2", "k7", "k9"],
        },
        "numerics": {"dtype": "float64"},
        "components": {
            "log_band_power": {"enabled": True, "weight": 1.0},
            "third_order_product_moments": _third([[["u", "k2"], ["v", "k7"], ["w", "k9"]]]),
        },
    }
    thetas = (0.0, 0.5 * math.pi, math.pi, -0.5 * math.pi)
    states = torch.stack([triad(theta) for theta in thetas])
    family = _fitted(config, coordinates, states, dimension=1)
    statistics = family._statistics(states)
    power = torch.diagonal(statistics["gram"], dim1=-2, dim2=-1)
    torch.testing.assert_close(
        torch.diagonal(power, dim1=-2, dim2=-1),
        torch.full((4, 3), 0.5, dtype=power.dtype),
        atol=1e-3,
        rtol=0,
    )
    global_measure = family.measures[:1]
    centered = regional_centered(statistics["factor_series"], global_measure)
    moments = third_central_moments(centered, global_measure, torch.tensor([[0, 1, 2]]))[:, 0, 0]
    torch.testing.assert_close(
        moments, torch.tensor([0.25, 0.0, -0.25, 0.0], dtype=moments.dtype), atol=2e-3, rtol=0
    )
    # The power descriptor cannot see the phase; the quadrature signs are indistinguishable.
    blocks = _descriptor_blocks(family, states)
    torch.testing.assert_close(blocks["log_band_power"][0], blocks["log_band_power"][2])
    assert abs(moments[1] - moments[3]) < 2e-3


def test_sign_symmetric_dependence_is_a_documented_blind_spot():
    values = torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=torch.float64)
    x = values.repeat_interleave(2)
    y = x * torch.tensor([1.0, -1.0], dtype=torch.float64).repeat(4)
    measures = torch.full((1, 8), 1.0 / 8, dtype=torch.float64)
    centered = regional_centered(torch.stack((x, y))[None], measures)
    third = third_central_moments(
        centered, measures, torch.tensor([[0, 0, 1], [0, 1, 1], [1, 1, 1]])
    )
    # Dependent (|y| = |x|) yet every third moment vanishes.
    torch.testing.assert_close(third, torch.zeros_like(third))


def test_autograd_matches_finite_differences_for_both_objectives():
    coordinates = _grid(5, jitter=0.2)
    states = _smooth_states(coordinates, 6, fields=2)
    base = {
        "fields": ["u", "v"],
        "geometry": _geometry(),
        "bands": {"count": 2, "min_wavelength": 0.2},
        "regions": {"grid": [2, 1]},
        "numerics": {"dtype": "float64"},
        "components": {
            "log_band_power": {"enabled": True, "weight": 1.0},
            "signed_band_cospectrum": {"enabled": True, "weight": 1.0},
            "third_order_product_moments": _third([[["u", 0], ["v", 1], ["v", 1]]]),
        },
    }
    paired = _fitted(base, coordinates, states, fields=("u", "v"))
    energy = _fitted(
        {**base, "objective": {"kind": "conditional_energy_score", "conditional_draws": 2}},
        coordinates,
        states,
        fields=("u", "v"),
    )
    target = states[:1]
    first = (states[2:3] * 0.9).clone().requires_grad_(True)
    second = (states[3:4] * 1.1).clone().requires_grad_(True)
    grid = coordinates[None]
    torch.autograd.gradcheck(
        lambda value: paired(value, target, coordinates=grid).scalar_loss, (first,)
    )
    torch.autograd.gradcheck(
        lambda left, right: (
            energy(
                left, target, coordinates=grid, context={"conditional_draws": (right,)}
            ).scalar_loss
        ),
        (first, second),
    )


def test_physical_units_are_formed_in_the_compute_precision():
    coordinates = _grid(8)
    states = _smooth_states(coordinates, 6) * 1e-3
    normalizer = FieldNormalizer(offset=[101325.0] * 3, scale=[50.0] * 3)
    config = _config(units="physical_units", numerics={"dtype": "float64"})
    family = MultiscaleSpectrumFamily(
        config, DataSpec(("u", "v", "w"), ("1",) * 3, 2, (64,)), normalizer
    )
    family.fit_reference_tensors(coordinates, states.float(), sample_ids=list("abcdef"))
    model_units = _fitted(_config(numerics={"dtype": "float64"}), coordinates, states.float())
    physical = family._statistics(states[:2].float())["gram"]
    reference = model_units._statistics(states[:2].float())["gram"]
    torch.testing.assert_close(physical, reference * 50.0**2, rtol=1e-6, atol=0.0)


def test_values_at_excluded_points_never_enter_a_descriptor():
    _, support = _u_shaped_grid()
    # Two points inside the unsupported notch join the query set: no supported
    # element reaches them, so the components policy excludes them.
    coordinates = torch.from_numpy(np.concatenate([support, [[0.5, 0.9], [0.5, 0.75]]]))
    states = _smooth_states(coordinates, 6)
    config = _config(
        geometry=_geometry(support_factor=0.75, connectivity="components"),
        components={
            "log_band_power": {"enabled": True, "weight": 1.0},
            "third_order_product_moments": _third([[["u", 0], ["v", 1], ["w", 2]]]),
        },
    )
    family = _family(config, coordinates.shape[0])
    family.fit_reference_tensors(
        coordinates,
        states,
        sample_ids=[f"s{index}" for index in range(6)],
        support_coordinates=torch.from_numpy(support),
    )
    assert family.geometry_record["excluded_points"] == 2
    assert family.analysis_index.numel() == coordinates.shape[0] - 2
    perturbed = states[:2].clone()
    perturbed[:, -2:] += 100.0
    grid = coordinates.expand(2, -1, -1)
    torch.testing.assert_close(
        family.descriptors(perturbed, coordinates=grid),
        family.descriptors(states[:2], coordinates=grid),
    )
    restored = _family(config, coordinates.shape[0])
    restored.load_state_artifact(family.state_artifact())
    torch.testing.assert_close(restored.analysis_index, family.analysis_index)


def test_per_piece_offsets_are_invisible_on_a_disconnected_domain():
    full = np.stack(np.meshgrid(np.arange(15), np.arange(15), indexing="ij"), axis=-1).reshape(
        -1, 2
    )
    points = full[(full[:, 0] < 5) | (full[:, 0] > 9)] / 14.0
    coordinates = torch.from_numpy(points)
    states = _smooth_states(coordinates, 6)
    config = _config(
        geometry=_geometry(max_edge_factor=1.8, spacing_neighbors=4, connectivity="components"),
        bands={"count": 2},
        numerics={"dtype": "float64"},
    )
    family = _fitted(config, coordinates, states)
    assert family.geometry_record["analysis_components"] == 2
    right = torch.from_numpy(points[:, 0] > 0.5)
    shifted = states[:2].clone()
    shifted[:, right] += torch.tensor([3.0, -2.0, 1.0], dtype=torch.float64)
    grid = coordinates.expand(2, -1, -1)
    torch.testing.assert_close(
        family.descriptors(shifted, coordinates=grid),
        family.descriptors(states[:2], coordinates=grid),
    )


def test_energy_score_prefers_the_correct_bimodal_distribution_over_collapse():
    weights = torch.ones(1, dtype=torch.float64)
    scores = []
    for target in (-1.0, 1.0):
        for left in (-1.0, 1.0):
            for right in (-1.0, 1.0):
                draws = torch.tensor([[[left], [right]]], dtype=torch.float64)
                score, _, _ = conditional_energy_score(
                    draws, torch.tensor([[target]], dtype=torch.float64), weights
                )
                scores.append(score.item())
    collapsed = [
        conditional_energy_score(
            torch.zeros(1, 2, 1, dtype=torch.float64),
            torch.tensor([[target]], dtype=torch.float64),
            weights,
        )[0].item()
        for target in (-1.0, 1.0)
    ]
    assert sum(scores) / len(scores) == pytest.approx(0.5)
    assert sum(collapsed) / len(collapsed) == pytest.approx(1.0)


def test_energy_score_distance_has_the_zero_subgradient_at_coincidence():
    point = torch.tensor([[0.3, -0.2]], dtype=torch.float64, requires_grad=True)
    distance = weighted_distance(
        point, point.detach(), torch.tensor([1.0, 4.0], dtype=torch.float64)
    )
    distance.sum().backward()
    assert distance.item() == 0.0
    torch.testing.assert_close(point.grad, torch.zeros_like(point))
    draws = torch.zeros(2, 3, 4, dtype=torch.float64, requires_grad=True)
    score, attraction, repulsion = conditional_energy_score(
        draws, torch.zeros(2, 4, dtype=torch.float64), torch.ones(4, dtype=torch.float64)
    )
    score.sum().backward()
    assert torch.all(score == 0) and torch.all(attraction == 0) and torch.all(repulsion == 0)
    assert torch.isfinite(draws.grad).all()


def test_config_contract_rejects_ambiguous_or_unsupported_settings():
    fields = ("u", "v", "w")
    parse_multiscale_config(_config(), fields)
    for broken, message in (
        ({"components": {"third_order_product_moments": {"enabled": True}}}, "explicit tuples"),
        ({"components": {"spatial_mean": {"enabled": True}}}, "unknown"),
        ({"geometry": {"operator": "knn_graph"}, "bands": {"wavelengths": [1.0]}}, "graph scales"),
        (
            {"geometry": {"operator": "knn_graph", "support_factor": 1.0}, "bands": {"count": 2}},
            "do not apply",
        ),
        ({"geometry": _geometry(k_neighbors=8)}, "do not apply"),
        ({"target_use": "training_reference"}, "paired_supervised"),
        ({"bands": {"wavelengths": [0.5, 1.0]}}, "strictly decreasing"),
        ({"bands": {"wavelengths": [1.0, 0.5], "min_wavelength": 0.5}}, "below the finest"),
        ({"geometry": _geometry(periodic=True)}, "geometry.periods"),
        ({"geometry": _geometry(min_component_fraction=0.1)}, "connectivity=components"),
        ({"objective": {"kind": "paired_descriptor_mse", "conditional_draws": 2}}, "one draw"),
        (
            {
                "components": {
                    "third_order_product_moments": _third(
                        [[["u", 0], ["v", 1], ["v", 1]], [["v", 1], ["u", 0], ["v", 1]]]
                    )
                }
            },
            "duplicate",
        ),
        (
            {
                "components": {
                    "third_order_product_moments": _third(
                        [[["u", 0], ["v", 1], ["v", 1]]], regional=True
                    )
                }
            },
            "regions",
        ),
        ({"regions": {"grid": [2, 2]}}, "no component is regional"),
        ({"bands": {"count": 2, "names": ["coarse", "coarse"]}}, "unique"),
    ):
        with pytest.raises((ValueError, KeyError, TypeError), match=message):
            parse_multiscale_config({**_config(), **broken}, fields)


def test_single_field_defaults_to_band_power_only():
    settings = parse_multiscale_config({"geometry": _geometry(), "bands": {"count": 2}}, ("u",))
    assert [component.key for component in settings.components] == ["log_band_power"]
    with pytest.raises(ValueError, match="two fields"):
        parse_multiscale_config(
            {"bands": {"count": 2}, "components": {"signed_band_cospectrum": {"enabled": True}}},
            ("u",),
        )


def test_state_artifact_round_trips_and_unfitted_family_refuses_to_score(tmp_path):
    coordinates = _grid(8)
    states = _smooth_states(coordinates, 6)
    config = _config(
        regions={"grid": [2, 2]},
        components={
            "log_band_power": {"enabled": True, "weight": 1.0},
            "third_order_product_moments": _third([[["u", 0], ["v", 1], ["w", 2]]]),
        },
    )
    family = _fitted(config, coordinates, states)
    path = tmp_path / "family.pt"
    torch.save(family.state_artifact(), path)
    restored = _family(config, coordinates.shape[0])
    with pytest.raises(RuntimeError, match="frozen training-reference"):
        restored(states[:1], states[:1], coordinates=coordinates[None])
    restored.load_state_artifact(torch.load(path, weights_only=True))
    torch.testing.assert_close(
        restored.descriptors(states[:2], coordinates=coordinates.expand(2, -1, -1)),
        family.descriptors(states[:2], coordinates=coordinates.expand(2, -1, -1)),
    )
    assert restored.region_names == family.region_names and len(restored.region_names) == 4
    with pytest.raises(ValueError, match="fitted geometry"):
        restored(states[:1], states[:1], coordinates=coordinates.flip(0)[None])
    with pytest.raises(RuntimeError, match="already frozen"):
        restored.fit_reference_tensors(coordinates, states, sample_ids=list("abcdef"))


def test_standardized_descriptors_weight_each_descriptor_by_its_training_variance(tmp_path):
    coordinates = _grid(8)
    states = _smooth_states(coordinates, 12)
    ids = [f"s{index}" for index in range(12)]
    objective = {"kind": "conditional_energy_score", "conditional_draws": 2}
    plain = _fitted(_config(objective=objective), coordinates, states)
    assert "descriptor_variance" not in plain.state_dict()
    config = _config(objective={**objective, "standardize_descriptors": True})
    family = _fitted(config, coordinates, states)
    streamed = _family(config, coordinates.shape[0])
    streamed.fit_reference_tensors(coordinates, iter(states.split(5)), sample_ids=ids)
    torch.testing.assert_close(streamed.descriptor_variance, family.descriptor_variance)

    grid = coordinates.expand(12, -1, -1)
    variance = plain.descriptors(states, coordinates=grid).double().var(dim=0)
    counts = plain.layout.counts
    floors = torch.cat(
        [torch.full((n,), float(v.mean()) * 1e-3) for v, n in zip(variance.split(counts), counts)]
    )
    torch.testing.assert_close(
        family.descriptor_variance.double(), torch.maximum(variance, floors.double())
    )
    assert family.calibration_record["descriptor_variance"] == pytest.approx(
        variance.tolist(), rel=1e-5
    )

    # Only the energy-score weights change: W = lambda_c / (d_c * variance).
    target, first, second = states[:2], states[2:4] * 0.9, states[4:6] * 1.1
    result = family(first, target, coordinates=grid[:2], context={"conditional_draws": (second,)})
    stack = torch.stack(
        [plain.descriptors(draw, coordinates=grid[:2]) for draw in (first, second)], 1
    )
    weights = (
        plain.layout.weight_vector(dtype=stack.dtype, device=stack.device)
        / family.descriptor_variance
    )
    expected, _, _ = conditional_energy_score(
        stack, plain.descriptors(target, coordinates=grid[:2]), weights
    )
    torch.testing.assert_close(result.scalar_loss, expected.mean())

    path = tmp_path / "family.pt"
    torch.save(family.state_artifact(), path)
    restored = _family(config, coordinates.shape[0])
    restored.load_state_artifact(torch.load(path, weights_only=True))
    torch.testing.assert_close(restored.descriptor_variance, family.descriptor_variance)
    with pytest.raises(TypeError, match="standardize_descriptors must be boolean"):
        parse_multiscale_config(
            _config(objective={**objective, "standardize_descriptors": 1}), ("u", "v", "w")
        )


def test_reference_fit_is_training_only_and_rejects_featureless_fields():
    class Selection:
        split = "validation"

    class Dataset:
        split_name = "validation"
        selection = Selection()

    family = _family(_config(), 64)
    with pytest.raises(ValueError, match="training split"):
        family.fit_training_reference(Dataset(), query_indices=torch.arange(64))
    coordinates = _grid(8)
    flat = _smooth_states(coordinates, 4)
    flat[..., 2] = 1.0
    with pytest.raises(ValueError, match="no spatial fluctuation"):
        _fitted(_config(), coordinates, flat)


def test_registry_builds_the_family_with_synchronized_version():
    family = build_coherence_family(
        "multiscale_spectrum",
        _config(),
        DataSpec(("u", "v", "w"), ("1",) * 3, 2, (16,)),
        FieldNormalizer.identity(3),
    )
    from phycoflow_reconstruction.registry import COHERENCE_FAMILY_REGISTRY

    assert COHERENCE_FAMILY_REGISTRY.get("multiscale_spectrum").version == family.version == "1"


def test_registry_advertises_every_supported_spectral_objective():
    from phycoflow_reconstruction.coherence.families.multiscale_spectrum.config import OBJECTIVES
    from phycoflow_reconstruction.coherence.families.multiscale_spectrum.objectives import (
        BAND_COHERENCE_PATH,
        JOINT_ENERGY_PATH,
        SIGNED_BAND_PATH,
    )
    from phycoflow_reconstruction.registry import COHERENCE_FAMILY_REGISTRY

    metadata = COHERENCE_FAMILY_REGISTRY.get("multiscale_spectrum").metadata
    assert metadata["objectives"] == OBJECTIVES
    assert {BAND_COHERENCE_PATH, JOINT_ENERGY_PATH, SIGNED_BAND_PATH}.issubset(metadata["components"])
