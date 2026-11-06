"""Exact signed-band coefficient-space scoring and paired point semantics."""

import copy
import json
from pathlib import Path

import pytest
import torch
from helpers.spectral import _fitted, _geometry, _smooth_states

from phycoflow_reconstruction.coherence.families.multiscale_spectrum import (
    MultiscaleSpectrumFamily,
    parse_multiscale_config,
)
from phycoflow_reconstruction.coherence.families.multiscale_spectrum.losses import (
    field_projections,
    paired_joint_spectral_mean_distance,
    paired_signed_band_distance,
    signed_band_coefficient_energy_score,
    signed_band_energy_score,
)
from phycoflow_reconstruction.coherence.families.multiscale_spectrum.statistics import band_fields
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("case", ["distinct", "repeated_draws", "all_coincident"])
def test_projected_coefficient_score_matches_spatial_values_and_all_draw_gradients(dtype, case):
    generator = torch.Generator().manual_seed(104)
    mass = torch.rand(17, generator=generator, dtype=dtype) + 0.15
    q, _ = torch.linalg.qr(torch.randn(17, 9, generator=generator, dtype=dtype))
    basis = q / mass.sqrt()[:, None]
    # Overlapping bands, three fields including a weak field and its frozen floor.
    filters = torch.rand(3, 9, generator=generator, dtype=dtype)
    scales = torch.tensor([[0.3, 0.8, 0.001], [0.5, 0.4, 0.002], [1.0, 0.6, 0.003]], dtype=dtype)
    projection = field_projections(3, "axes_and_pairs", dtype=dtype, device=torch.device("cpu"))
    target = torch.randn(2, 9, 3, generator=generator, dtype=dtype)
    target[..., 2] *= 1.0e-6
    values = torch.randn(4, 2, 9, 3, generator=generator, dtype=dtype)
    values[..., 2] *= 1.0e-6
    if case in {"repeated_draws", "all_coincident"}:
        values = values[:1].expand_as(values).clone()
    if case == "all_coincident":
        values = target[None].expand_as(values).clone()
    draws = values.requires_grad_()
    actual = signed_band_coefficient_energy_score(
        draws,
        target,
        filters=filters,
        mass=mass,
        scales=scales,
        projections=projection,
        alpha=0.95,
    )
    # The legacy family synthesizes each draw and the target separately.
    spatial_draws = torch.stack([band_fields(draw, basis, filters) for draw in draws])
    spatial_target = band_fields(target, basis, filters)
    expected = signed_band_energy_score(
        spatial_draws,
        spatial_target,
        mass=mass,
        scales=scales,
        projections=projection,
        alpha=0.95,
    )
    tolerance = 5e-5 if dtype == torch.float32 else 1e-12
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=tolerance, atol=tolerance)
        got_grad = torch.autograd.grad(got.sum(), draws, retain_graph=True)[0]
        want_grad = torch.autograd.grad(want.sum(), draws, retain_graph=True)[0]
        torch.testing.assert_close(got_grad, want_grad, rtol=tolerance, atol=tolerance)
        assert torch.isfinite(got_grad).all()
        if case == "all_coincident":
            assert torch.count_nonzero(got_grad) == 0
    if case == "distinct":
        grad = torch.autograd.grad(actual[0].sum(), draws)[0]
        assert all(item.norm() > 0 for item in grad)


def _point_config():
    return {
        "fields": ["u", "v"],
        "geometry": {"operator": "fem_p1", "coordinate_scaling": "none"},
        "bands": {"count": 2},
        "calibration": {"field_unit_floor": {"u": 0.01, "v": 0.02}},
        "numerics": {"dtype": "float64"},
        "objective": {
            "kind": "paired_signed_band_mean_distance",
            "conditional_draws": 1,
            "score_definition_version": "paired_signed_spectral_mean_v1",
            "projections": "axes_and_pairs",
            "joint_distance_weight": 0.25,
            "mean_weight": 1.0,
        },
    }


def test_point_objective_uses_one_paired_prediction_and_restores_strict_artifact():
    config = _point_config()
    coordinates = torch.linspace(0, 1, 17, dtype=torch.float64)[:, None] ** 1.4
    rng = torch.Generator().manual_seed(28)
    states = torch.randn(9, 17, 2, dtype=torch.float64, generator=rng)
    spec = DataSpec(("u", "v"), ("1", "1"), 1, (17,))
    family = MultiscaleSpectrumFamily(config, spec, FieldNormalizer.identity(2))
    family.fit_reference_tensors(coordinates, states, sample_ids=list("abcdefghi"))
    first = states[:2].clone().requires_grad_()
    reference = states[2:4]
    coords = coordinates[None].expand(2, -1, -1)
    result = family(first, reference, coordinates=coords)
    keys = result.component_results
    primary = "multiscale_spectrum.paired_signed_band_distance.paired"
    joint = "multiscale_spectrum.paired_joint_spectral_mean_distance.paired"
    assert primary in keys and joint in keys
    assert not any("energy_score" in key or "crps" in key for key in keys)
    assert result.diagnostics["paired_signed_band"]["interpretation"] == "paired_point_distance"
    torch.testing.assert_close(
        result.scalar_loss, keys[primary].scalar_loss + 0.25 * keys[joint].scalar_loss
    )
    result.scalar_loss.backward()
    assert first.grad is not None and torch.isfinite(first.grad).all() and first.grad.norm() > 0
    json.dumps(result.diagnostics, allow_nan=False)
    # Additional draws in a shared family context cannot turn this point query into a score.
    with_extra = family(
        first.detach(), reference, coordinates=coords, context={"conditional_draws": (states[4:6],)}
    )
    torch.testing.assert_close(result.scalar_loss.detach(), with_extra.scalar_loss)
    artifact = family.state_artifact()
    restored = MultiscaleSpectrumFamily(config, spec, FieldNormalizer.identity(2))
    restored.load_state_artifact(artifact)
    torch.testing.assert_close(
        result.scalar_loss.detach(),
        restored(first.detach(), reference, coordinates=coords).scalar_loss,
        rtol=0,
        atol=0,
    )
    changed = copy.deepcopy(artifact)
    changed["implementation_identity"]["sources"]["losses.py"] = "0" * 64
    with pytest.raises(ValueError, match="implementation/backend identity mismatch"):
        restored.load_state_artifact(changed)

    identity = family(reference, reference, coordinates=coords)
    assert identity.scalar_loss == 0
    shifted = family(reference + 0.5, reference, coordinates=coords)
    assert shifted.component_results[joint].scalar_loss > 0
    assert shifted.component_results[primary].scalar_loss < 1e-12


@pytest.mark.parametrize(
    "change", ["extra_draw", "no_floor", "wrong_version", "alpha", "zero_joint"]
)
def test_point_objective_rejects_ambiguous_contracts(change):
    config = _point_config()
    if change == "extra_draw":
        config["objective"]["conditional_draws"] = 2
    elif change == "no_floor":
        config["calibration"].clear()
    elif change == "wrong_version":
        config["objective"]["score_definition_version"] = "signed_spectral_mean_augmented_v4"
    elif change == "alpha":
        config["objective"]["almost_fair_alpha"] = 1
    else:
        config["objective"]["joint_distance_weight"] = 0
    with pytest.raises(ValueError):
        parse_multiscale_config(config, ("u", "v"))


def test_complete_deterministic_case_profile_does_not_inherit_energy_score_keys():
    from phycoflow_reconstruction.config.load import load_config

    path = (
        Path(__file__).resolve().parents[1]
        / "cases/turbulent_combustion/configs/posttrain/coherence_deterministic.yaml"
    )
    config = load_config(path)
    families = config["coherence"]["families"]
    assert {name for name, settings in families.items() if settings.get("enabled", True)} == {
        "global_distribution",
        "multiscale_spectrum",
        "topology",
    }
    objective = families["multiscale_spectrum"]["objective"]
    assert objective["kind"] == "paired_signed_band_mean_distance"
    assert objective["conditional_draws"] == 1
    assert (
        not {"almost_fair_alpha", "joint_energy_score_weight", "coherence_crps_weight"}
        & objective.keys()
    )
    parsed = parse_multiscale_config(
        families["multiscale_spectrum"], tuple(config["dataset"]["field_names"])
    )
    assert parsed.conditional_draws == 1


def test_disconnected_piece_point_distances_match_spatial_fields_and_gradients():
    index = torch.cartesian_prod(torch.arange(12), torch.arange(12))
    coordinates = index[(index[:, 0] < 4) | (index[:, 0] > 7)].to(torch.float64) / 11
    states = _smooth_states(coordinates, 8, fields=2, seed=19)
    config = _point_config()
    config["geometry"] = _geometry(
        max_edge_factor=1.8, spacing_neighbors=4, connectivity="components"
    )
    family = _fitted(config, coordinates, states, fields=("u", "v"))
    assert family.piece_measures.shape[0] == 2
    prediction = states[:2].clone().requires_grad_()
    target = states[2:4]
    pred = family._statistics(prediction)
    ref = family._statistics(target)
    projection = family._projections(torch.float64, torch.device("cpu"))
    coefficients = pred["coefficients"]
    expected_fields = band_fields(coefficients, family.basis, family.filters)
    reference_fields = band_fields(ref["coefficients"], family.basis, family.filters)
    scaled = (expected_fields - reference_fields) / family.band_scale[None, :, None, :]
    projected = torch.einsum("bsnc,pc->bspn", scaled, projection)
    spatial = torch.linalg.vector_norm(
        projected * (family.mass / family.mass.sum()).sqrt(), dim=-1
    ).mean(-1)
    actual = paired_signed_band_distance(
        coefficients,
        ref["coefficients"],
        filters=family.filters,
        mass=family.mass,
        scales=family.band_scale,
        projections=projection,
    )
    torch.testing.assert_close(actual, spatial)
    piece_weights = family.piece_measures.sum(1)
    means = (
        (pred["piece_means"] - ref["piece_means"])
        * (piece_weights / piece_weights.sum()).sqrt()[None, :, None]
        / family.piece_mean_scale
    )
    expected_joint = torch.linalg.vector_norm(
        torch.cat(
            (
                (scaled * (family.mass / family.mass.sum()).sqrt()[None, None, :, None]).flatten(1),
                means.flatten(1),
            ),
            dim=-1,
        ),
        dim=-1,
    )
    actual_joint = paired_joint_spectral_mean_distance(
        coefficients,
        ref["coefficients"],
        filters=family.filters,
        scales=family.band_scale,
        mass=family.mass,
        prediction_piece_means=pred["piece_means"],
        target_piece_means=ref["piece_means"],
        piece_weights=piece_weights,
        mean_scales=family.piece_mean_scale,
    )
    torch.testing.assert_close(actual_joint, expected_joint)
    for got, expected in ((actual, spatial), (actual_joint, expected_joint)):
        got_grad = torch.autograd.grad(got.sum(), prediction, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected.sum(), prediction, retain_graph=True)[0]
        torch.testing.assert_close(got_grad, expected_grad)
        assert torch.isfinite(got_grad).all()
