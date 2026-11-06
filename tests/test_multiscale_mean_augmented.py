"""Scale score qualification: means, fixed task-unit floors, and restore."""

import copy
import itertools
import json

import numpy as np
import pytest
import torch
from helpers.spectral import _config, _fitted, _geometry, _smooth_states

from phycoflow_reconstruction.coherence.families.multiscale_spectrum import (
    MultiscaleSpectrumFamily,
    parse_multiscale_config,
)
from phycoflow_reconstruction.coherence.families.multiscale_spectrum.losses import (
    field_projections,
    joint_spectral_energy_score,
    vector_energy_score,
)
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer

VERSION = "signed_spectral_mean_augmented_v4"
JOINT = "multiscale_spectrum.joint_spectral_mean_energy_score.conditional"
LEGACY_JOINT = "multiscale_spectrum.joint_spectral_energy_score.conditional"


def _new_config(fields=("u",), floor=0.01, **overrides):
    config = {
        "fields": list(fields),
        "geometry": {"operator": "fem_p1", "coordinate_scaling": "none"},
        "bands": {"count": 2},
        "calibration": {"field_unit_floor": floor},
        "numerics": {"dtype": "float64"},
        "objective": {
            "kind": "signed_band_energy_score",
            "conditional_draws": 2,
            "almost_fair_alpha": 1.0,
            "joint_energy_score_weight": 0.25,
            "score_definition_version": VERSION,
            "mean_weight": 1.0,
        },
    }
    config.update(overrides)
    return config


def _line_family(config, states, normalizer=None):
    coordinates = torch.linspace(0, 1, states.shape[1], dtype=torch.float64)[:, None]
    fields = tuple(config["fields"])
    family = MultiscaleSpectrumFamily(
        config,
        DataSpec(fields, ("1",) * len(fields), 1, (states.shape[1],)),
        normalizer or FieldNormalizer.identity(len(fields)),
    )
    family.fit_reference_tensors(
        coordinates, states, sample_ids=[str(i) for i in range(len(states))]
    )
    return family, coordinates


def _score(family, coordinates, first, second, target):
    return family(
        first,
        target,
        coordinates=coordinates[None].expand(first.shape[0], -1, -1),
        context={"conditional_draws": (second,)},
    )


def test_augmented_coefficient_score_matches_spatial_fields_and_gradients():
    rng = torch.Generator().manual_seed(812)
    opts = {"generator": rng, "dtype": torch.float64}
    mass = torch.rand(11, **opts) + 0.2
    q, _ = torch.linalg.qr(torch.randn(11, 6, **opts))
    basis = q / mass.sqrt()[:, None]
    filters, scales = torch.rand(3, 6, **opts), torch.rand(3, 2, **opts) + 0.1
    coefficients = torch.randn(3, 2, 6, 2, **opts).requires_grad_()
    target = torch.randn(2, 6, 2, **opts)
    means = torch.randn(3, 2, 2, 2, **opts).requires_grad_()
    target_means = torch.randn(2, 2, 2, **opts)
    fractions = torch.tensor([0.3, 0.7], dtype=torch.float64)
    mean_scales = torch.rand(2, 2, **opts) + 0.1
    actual = joint_spectral_energy_score(
        coefficients,
        target,
        filters=filters,
        scales=scales,
        mass=mass,
        draw_piece_means=means,
        target_piece_means=target_means,
        piece_weights=fractions,
        mean_scales=mean_scales,
        mean_weight=1.7,
    )
    fields = torch.einsum("nk,sk,jbkc->jbsnc", basis, filters, coefficients)
    truth = torch.einsum("nk,sk,bkc->bsnc", basis, filters, target)
    field_root = (mass / mass.sum()).sqrt()[None, :, None] / scales[:, None, :]
    mean_root = (1.7 * fractions).sqrt()[:, None] / mean_scales
    expected = vector_energy_score(
        torch.cat(((fields * field_root).flatten(2), (means * mean_root).flatten(2)), -1),
        torch.cat(((truth * field_root).flatten(1), (target_means * mean_root).flatten(1)), -1),
    )
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right)
    actual_grad = torch.autograd.grad(actual[0].sum(), (coefficients, means), retain_graph=True)
    expected_grad = torch.autograd.grad(expected[0].sum(), (coefficients, means))
    for left, right in zip(actual_grad, expected_grad):
        torch.testing.assert_close(left, right)
        assert torch.isfinite(left).all() and all(row.norm() > 0 for row in left)


def test_joint_embedding_detects_mean_fluctuation_dependence():
    # Each law has the same mean and fluctuation marginals, but opposite pairing.
    support = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], dtype=torch.float64)
    swapped = support * torch.tensor([1.0, -1.0])
    triplets = torch.tensor(list(itertools.product(range(2), repeat=3)))
    target = support[triplets[:, 2]]

    def expected_score(law):
        draws = law[triplets[:, :2]].transpose(0, 1)
        return joint_spectral_energy_score(
            draws[:, :, None, :1],
            target[:, None, :1],
            filters=torch.ones(1, 1),
            scales=torch.ones(1, 1),
            mass=torch.ones(1),
            draw_piece_means=draws[:, :, None, 1:],
            target_piece_means=target[:, None, 1:],
            piece_weights=torch.ones(1),
            mean_scales=torch.ones(1, 1),
        )[0].mean()

    assert expected_score(swapped) > expected_score(support) + 0.1


def test_disconnected_piece_shifts_are_scored_only_by_new_joint_term():
    full = np.stack(np.meshgrid(np.arange(15), np.arange(15), indexing="ij"), -1).reshape(-1, 2)
    points = np.concatenate([full[(full[:, 0] < 5) | (full[:, 0] > 9)] / 14, [[0.5, 0.5]]])
    coordinates = torch.from_numpy(points)
    states = _smooth_states(coordinates, 8)
    base = _config(
        geometry=_geometry(max_edge_factor=1.8, spacing_neighbors=4, connectivity="components"),
        bands={"count": 2},
        numerics={"dtype": "float64"},
        calibration={"field_unit_floor": {"u": 0.01, "v": 0.02, "w": 0.03}},
        objective=_new_config()["objective"],
    )
    family = _fitted(base, coordinates, states)
    assert family.geometry_record["analysis_components"] == 2
    assert family.geometry_record["excluded_points"] == 1
    selected_means = family._statistics(states)["piece_means"]
    floors = torch.tensor([0.01, 0.02, 0.03])
    torch.testing.assert_close(
        family.piece_mean_scale,
        (selected_means.var(dim=0, unbiased=False) + floors.square()).sqrt(),
        atol=2e-9,
        rtol=2e-9,
    )
    target = states[:1]
    shifted = target.clone()
    shifted[:, coordinates[:, 0] > 0.7] += torch.tensor([2.0, -1.0, 0.5])
    shifted[:, -1] = 999.0  # excluded point contributes to neither means nor spectrum
    first, second = shifted.clone().requires_grad_(), shifted.clone().requires_grad_()
    result = _score(family, coordinates, first, second, target)
    delta = family._statistics(first)["piece_means"] - family._statistics(target)["piece_means"]
    pi = family.piece_measures.sum(-1) / family.piece_measures.sum()
    expected = ((delta / family.piece_mean_scale).square() * pi[None, :, None]).sum().sqrt()
    torch.testing.assert_close(result.component_results[JOINT].scalar_loss, expected)
    result.scalar_loss.backward()
    assert torch.isfinite(first.grad).all() and first.grad.abs().sum() > 0
    assert torch.isfinite(second.grad).all() and second.grad.abs().sum() > 0
    assert first.grad[:, -1].abs().sum() == 0
    assert result.diagnostics["signed_band"]["score_definition_version"] == VERSION
    mean_report = result.diagnostics["signed_band"]["joint_energy_score"]
    assert len(mean_report["first_draw_mean_absolute_error_by_piece_field"]) == 2
    json.dumps(result.diagnostics, allow_nan=False)

    legacy = copy.deepcopy(base)
    legacy["objective"].pop("score_definition_version")
    legacy["objective"].pop("mean_weight")
    old = _fitted(legacy, coordinates, states)
    old_result = _score(old, coordinates, shifted, shifted, target)
    assert old_result.component_results[LEGACY_JOINT].scalar_loss < 1e-12
    assert old_result.diagnostics["signed_band"]["score_definition_version"] == "signed_spectral_v3"
    assert "piece_mean_scale" not in old.state_dict()


def test_zero_weak_train_fields_identity_gradients_and_exact_artifact_restore(tmp_path):
    coords = torch.linspace(0, 1, 16, dtype=torch.float64)
    train = torch.zeros(8, 16, 2, dtype=torch.float64)
    train[..., 1] = 1e-7 * torch.sin(2 * torch.pi * coords)[None, :]
    config = _new_config(("zero", "weak"), {"zero": 0.01, "weak": 0.02})
    family, coordinates = _line_family(config, train)
    torch.testing.assert_close(
        family.piece_mean_scale, torch.tensor([[0.01, 0.02]], dtype=torch.float64)
    )
    assert (family.band_scale > 0).all() and (family.power_floor > 0).all()
    assert family.calibration_record["field_unit_floor"] == [0.01, 0.02]
    zero = torch.zeros(2, 16, 2, dtype=torch.float64)
    first, second = zero.clone().requires_grad_(), zero.clone().requires_grad_()
    identity = _score(family, coordinates, first, second, zero)
    assert identity.scalar_loss.item() == 0.0
    identity.scalar_loss.backward()
    assert torch.equal(first.grad, torch.zeros_like(first))
    assert torch.equal(second.grad, torch.zeros_like(second))

    changed = (0.02 + 0.01 * torch.cos(2 * torch.pi * coords))[:, None].expand(-1, 2)
    first = changed[None].repeat(2, 1, 1).clone().requires_grad_()
    second = first.detach().clone().requires_grad_()
    result = _score(family, coordinates, first, second, zero)
    assert result.scalar_loss > 0
    result.scalar_loss.backward()
    assert torch.isfinite(first.grad).all() and first.grad.abs().sum() > 0
    assert torch.isfinite(second.grad).all() and second.grad.abs().sum() > 0

    artifact = tmp_path / "family.pt"
    torch.save(family.state_artifact(), artifact)
    restored = MultiscaleSpectrumFamily(
        config, DataSpec(("zero", "weak"), ("1", "1"), 1, (16,)), FieldNormalizer.identity(2)
    )
    restored.load_state_artifact(torch.load(artifact, weights_only=True))
    assert restored.calibration_record == family.calibration_record
    for name, value in family.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name]), name
    replay = _score(restored, coordinates, first.detach(), second.detach(), zero)
    assert torch.equal(result.scalar_loss.detach(), replay.scalar_loss)


@pytest.mark.parametrize("identity_change", ["missing", "source", "backend"])
def test_v4_rejects_missing_or_changed_implementation_and_backend(identity_change):
    rng = torch.Generator().manual_seed(93)
    train = torch.randn(8, 16, 1, generator=rng, dtype=torch.float64)
    config = _new_config()
    family, _ = _line_family(config, train)
    artifact = copy.deepcopy(family.state_artifact())
    identity = artifact["implementation_identity"]
    assert set(identity["backends"]) == {"torch", "numpy", "scipy"}
    assert {"family.py", "losses.py", "operators.py", "objectives.py"} <= set(identity["sources"])
    if identity_change == "missing":
        del artifact["implementation_identity"]
    elif identity_change == "source":
        identity["sources"]["losses.py"] = "0" * 64
    else:
        identity["backends"]["scipy"] = "different"
    restored = MultiscaleSpectrumFamily(
        config, DataSpec(("u",), ("1",), 1, (16,)), FieldNormalizer.identity(1)
    )
    with pytest.raises(ValueError, match="implementation/backend identity mismatch"):
        restored.load_state_artifact(artifact)
    assert not restored.fitted


@pytest.mark.parametrize(
    "field_names,field_units", [(("v", "u"), ("1", "1")), (("u", "v"), ("K", "1"))]
)
def test_v4_rejects_changed_input_field_order_or_units(field_names, field_units):
    config = _new_config()
    spec = DataSpec(("u", "v"), ("1", "1"), 1, (16,))
    family = MultiscaleSpectrumFamily(config, spec, FieldNormalizer.identity(2))
    coordinates = torch.linspace(0, 1, 16, dtype=torch.float64)[:, None]
    states = torch.randn(
        8, 16, 2, generator=torch.Generator().manual_seed(191), dtype=torch.float64
    )
    family.fit_reference_tensors(coordinates, states, sample_ids=list("abcdefgh"))
    artifact = family.state_artifact()
    changed = DataSpec(field_names, field_units, 1, (16,))
    restored = MultiscaleSpectrumFamily(config, changed, FieldNormalizer.identity(2))
    with pytest.raises(ValueError, match="input field order/units mismatch"):
        restored.load_state_artifact(artifact)


def test_field_unit_rescaling_preserves_score_and_changes_mean_units():
    rng = torch.Generator().manual_seed(192)
    train = torch.randn(10, 16, 2, generator=rng, dtype=torch.float64)
    base = _new_config(("u", "v"), {"u": 0.01, "v": 0.03})
    family, coordinates = _line_family(base, train)
    factors = torch.tensor([4.0, 0.2], dtype=torch.float64)
    scaled = _new_config(("u", "v"), {"u": 0.04, "v": 0.006})
    converted, _ = _line_family(scaled, train * factors)
    torch.testing.assert_close(converted.piece_mean_scale, family.piece_mean_scale * factors)
    a = _score(family, coordinates, train[:2], train[2:4], train[4:6])
    b = _score(
        converted, coordinates, train[:2] * factors, train[2:4] * factors, train[4:6] * factors
    )
    torch.testing.assert_close(a.scalar_loss, b.scalar_loss)


def test_physical_large_offset_uses_float64_before_scaling_and_centering():
    rng = torch.Generator().manual_seed(401)
    train = 1e-3 * torch.randn(8, 16, 1, generator=rng, dtype=torch.float64)
    normalizer = FieldNormalizer(offset=[101325.0], scale=[50.0])
    physical, _ = _line_family(_new_config(floor=5e-6, units="physical_units"), train, normalizer)
    model, _ = _line_family(_new_config(floor=1e-7), train)
    torch.testing.assert_close(
        physical.piece_mean_scale, model.piece_mean_scale * 50, rtol=1e-8, atol=1e-10
    )
    torch.testing.assert_close(
        physical._statistics(train[:2])["gram"],
        model._statistics(train[:2])["gram"] * 50**2,
        rtol=1e-8,
        atol=1e-12,
    )
    bad = _new_config(floor=0.5, units="physical_units", numerics={"dtype": "float32"})
    with pytest.raises(ValueError, match="float64"):
        parse_multiscale_config(bad, ("u",))


def test_train_mean_variance_streaming_with_large_physical_offset():
    rng = torch.Generator().manual_seed(27)
    train = 1e-3 * torch.randn(9, 16, 1, generator=rng, dtype=torch.float64)
    config = _new_config(floor=5e-6, units="physical_units")
    normalizer = FieldNormalizer(offset=[101325.0], scale=[50.0])
    dense, coordinates = _line_family(config, train, normalizer)
    spec = DataSpec(("u",), ("1",), 1, (16,))
    streamed = MultiscaleSpectrumFamily(config, spec, normalizer)
    streamed.fit_reference_tensors(
        coordinates, [train[:2], train[2:5], train[5:]], sample_ids=[str(i) for i in range(9)]
    )
    torch.testing.assert_close(
        streamed.piece_mean_scale, dense.piece_mean_scale, rtol=1e-8, atol=1e-10
    )
    means = dense._statistics(train)["piece_means"]
    expected = (means.var(dim=0, unbiased=False) + (5e-6) ** 2).sqrt()
    torch.testing.assert_close(streamed.piece_mean_scale, expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("geometry", ["periodic", "masked_nonuniform"])
def test_augmented_score_uses_fixed_periodic_or_masked_geometry(geometry):
    axis = np.arange(9)
    indices = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
    if geometry == "periodic":
        points = indices / 9.0
        geometry_config = _geometry(periodic=True, periods=[1.0, 1.0])
        grid_indices = None
    else:
        indices = indices[~((indices[:, 0] >= 4) & (indices[:, 0] <= 5) & (indices[:, 1] >= 4))]
        points = indices / 8.0
        points = points.copy()
        points[:, 0] += 0.03 * np.sin(np.pi * points[:, 0]) * np.sin(2 * np.pi * points[:, 1])
        geometry_config = _geometry(triangulation="structured_indices", structured_indices="ij")
        grid_indices = torch.from_numpy(indices)
    coords = torch.from_numpy(points)
    states = _smooth_states(coords, 8, fields=1)
    config = _new_config(geometry=geometry_config)
    spec = DataSpec(("u",), ("1",), 2, (len(coords),))
    family = MultiscaleSpectrumFamily(config, spec, FieldNormalizer.identity(1))
    if geometry == "masked_nonuniform":
        missing_mask_family = MultiscaleSpectrumFamily(config, spec, FieldNormalizer.identity(1))
        with pytest.raises(ValueError, match="grid indices"):
            missing_mask_family.fit_reference_tensors(coords, states, sample_ids=list("abcdefgh"))
    family.fit_reference_tensors(
        coords, states, sample_ids=list("abcdefgh"), grid_indices=grid_indices
    )
    if geometry == "masked_nonuniform":
        assert family.geometry_record["operator_diagnostics"]["total_measure"] < 0.95
    assert family.geometry_record["active_periodic"] == (
        [True, True] if geometry == "periodic" else [False, False]
    )
    first, second = states[:2].clone().requires_grad_(), states[2:4].clone().requires_grad_()
    result = _score(family, coords, first, second, states[4:6])
    result.scalar_loss.backward()
    assert torch.isfinite(result.scalar_loss)
    assert torch.isfinite(first.grad).all() and torch.isfinite(second.grad).all()
    changed_coords = coords.clone()
    changed_coords[0, 0] += 0.001
    with pytest.raises(ValueError, match="differ from the fitted geometry"):
        _score(family, changed_coords, first, second, states[4:6])


@pytest.mark.parametrize("floor", [0, -1, float("nan"), True, {"other": 1.0}])
def test_rejects_invalid_or_incomplete_task_unit_floor(floor):
    with pytest.raises((ValueError, TypeError), match="field_unit_floor"):
        parse_multiscale_config(_new_config(floor=floor), ("u",))


def test_new_version_requires_explicit_floor_and_positive_fair_joint():
    config = _new_config()
    config["calibration"].clear()
    with pytest.raises(ValueError, match="field_unit_floor"):
        parse_multiscale_config(config, ("u",))
    config = _new_config()
    config["objective"]["joint_energy_score_weight"] = 0
    with pytest.raises(ValueError, match="joint_energy_score_weight"):
        parse_multiscale_config(config, ("u",))
    config = _new_config()
    config["objective"]["mean_weight"] = 0
    with pytest.raises(ValueError, match="mean_weight"):
        parse_multiscale_config(config, ("u",))


def test_selected_projection_pairs_remain_fixed_and_explicit():
    config = _new_config(("u", "v", "w"))
    config["objective"]["projection_pairs"] = [["u", "w"]]
    spec = DataSpec(("u", "v", "w"), ("1",) * 3, 1, (16,))
    family = MultiscaleSpectrumFamily(config, spec, FieldNormalizer.identity(3))
    actual = family._projections(torch.float64, torch.device("cpu"))
    expected = field_projections(
        3, "axes_and_pairs", pairs=[(0, 2)], dtype=torch.float64, device=torch.device("cpu")
    )
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (5, 3)
    config["objective"]["projections"] = "axes"
    with pytest.raises(ValueError, match="projection_pairs"):
        parse_multiscale_config(config, ("u", "v", "w"))


def test_all_draw_evaluation_preserves_frozen_j2_training_contract():
    rng = torch.Generator().manual_seed(64)
    train = torch.randn(9, 16, 1, generator=rng, dtype=torch.float64)
    family, coords = _line_family(_new_config(), train)
    saved = {name: value.clone() for name, value in family.state_dict().items()}
    draws = [train[i : i + 1].clone().requires_grad_() for i in range(4)]
    target = train[4:5]
    context = {"conditional_draws": tuple(draws[1:])}
    kwargs = {"coordinates": coords[None], "context": context}
    training = family(draws[0], target, **kwargs)
    eval_context = {**context, "score_all_conditional_draws": True}
    result = family(draws[0], target, coordinates=coords[None], context=eval_context)
    assert training.diagnostics["conditional_draws"] == 2
    assert result.diagnostics["conditional_draws"] == 4
    assert result.diagnostics["signed_band"]["joint_energy_score"]["conditional_draws"] == 4
    pred_stats = [family._statistics(draw) for draw in draws]
    target_stats = family._statistics(target)
    expected = joint_spectral_energy_score(
        torch.stack([item["coefficients"] for item in pred_stats]),
        target_stats["coefficients"],
        filters=family.filters,
        scales=family.band_scale,
        mass=family.mass,
        draw_piece_means=torch.stack([item["piece_means"] for item in pred_stats]),
        target_piece_means=target_stats["piece_means"],
        piece_weights=family.piece_measures.sum(dim=1),
        mean_scales=family.piece_mean_scale,
    )[0].mean()
    torch.testing.assert_close(result.component_results[JOINT].scalar_loss, expected)
    result.scalar_loss.backward()
    assert all(
        draw.grad is not None and torch.isfinite(draw.grad).all() and draw.grad.norm() > 0
        for draw in draws
    )
    assert family.conditional_draws == 2
    assert all(torch.equal(saved[name], tensor) for name, tensor in family.state_dict().items())


def test_autocast_does_not_reduce_declared_float32_score_precision():
    rng = torch.Generator().manual_seed(918)
    train = torch.randn(8, 16, 2, generator=rng, dtype=torch.float32)
    family, coords = _line_family(_new_config(("u", "v"), numerics={"dtype": "float32"}), train)
    baseline = _score(family, coords, train[:2], train[2:4], train[4:6]).scalar_loss
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = _score(family, coords, train[:2], train[2:4], train[4:6]).scalar_loss
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
