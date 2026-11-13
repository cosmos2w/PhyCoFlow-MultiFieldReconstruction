"""Weighted State is an explicit occupation law, with legacy replay intact."""

from itertools import product

import pytest
import torch

from phycoflow_reconstruction.coherence import build_coherence_family
from phycoflow_reconstruction.coherence.base import empirical_w2_columns, weighted_w2_columns
from phycoflow_reconstruction.coherence.measures import (
    measure_fingerprint,
    rectilinear_nodal_weights,
    subset_inclusion_weights,
)
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def family(*, weighted=True, fields=("a", "b", "c"), units="model_units", normalizer=None):
    cfg = {
        "definition": "marginal_mixed_weighted_v3" if weighted else "marginal_mixed_mean_v2",
        "target_use": "paired_supervised",
        "components": {
            "self": {"enabled": True},
            "mutual": {"enabled": False},
            "cross": {
                "enabled": True,
                "directions": 32,
                "top_fraction": 1.0,
                "seed": 1234,
                "include_axes": False,
                "qmc": True,
            },
        },
    }
    cfg["units"] = units
    spec = DataSpec(fields, ("1",) * len(fields), 2, (8,))
    return build_coherence_family(
        "global_distribution", cfg, spec, normalizer or FieldNormalizer.identity(len(fields))
    )


def context(generated_weights, reference_weights=None, **extra):
    return {
        "generated_weights": generated_weights,
        "reference_weights": generated_weights if reference_weights is None else reference_weights,
        **extra,
    }


def test_two_mass_exact_quantile_integral_and_endpoint_gradients():
    # After independent value sorts the breakpoints are .25, .75, and 1:
    # 0*.25 + (0-4)^2*.5 + (2-4)^2*.25 = 9.
    x = torch.tensor([[2.0], [0.0]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[4.0], [0.0]], dtype=torch.float64, requires_grad=True)
    loss = weighted_w2_columns(x, y, torch.tensor([0.25, 0.75]), torch.tensor([0.75, 0.25]))
    torch.testing.assert_close(loss, torch.tensor([9.0], dtype=torch.float64))
    dx, dy = torch.autograd.grad(loss.sum(), (x, y))
    torch.testing.assert_close(dx, torch.tensor([[-1.0], [-4.0]], dtype=torch.float64))
    torch.testing.assert_close(dy, torch.tensor([[5.0], [0.0]], dtype=torch.float64))


def test_separate_support_sizes_and_uniform_mass_exact_legacy_parity():
    x = torch.tensor([[0.0], [3.0]], dtype=torch.float64)
    y = torch.tensor([[0.0], [1.0], [3.0]], dtype=torch.float64)
    torch.testing.assert_close(
        weighted_w2_columns(x, y, torch.tensor([0.5, 0.5]), torch.tensor([0.5, 0.25, 0.25])),
        torch.tensor([1.0], dtype=torch.float64),
    )
    torch.manual_seed(44)
    x, y = torch.randn(13, 9), torch.randn(13, 9)
    actual = weighted_w2_columns(x, y, torch.ones(13), torch.full((13,), 7.0))
    assert torch.equal(actual, empirical_w2_columns(x, y))


def test_both_endpoint_permutations_preserve_aligned_values_and_masses():
    torch.manual_seed(12)
    x, y = torch.randn(9, 4, dtype=torch.float64), torch.randn(9, 4, dtype=torch.float64)
    wx = torch.tensor([1, 3, 2, 5, 7, 1, 9, 2, 4], dtype=torch.float64)
    wy = torch.tensor([2, 1, 3, 4, 1, 2, 8, 5, 1], dtype=torch.float64)
    p, q = torch.randperm(9), torch.randperm(9)
    torch.testing.assert_close(
        weighted_w2_columns(x[p], y[q], wx[p], wy[q]),
        weighted_w2_columns(x, y, wx, wy),
        rtol=1e-14,
        atol=1e-14,
    )


def test_value_gradcheck_covers_all_columns_and_both_endpoints():
    torch.manual_seed(61)
    x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    wx = torch.tensor([0.1, 0.15, 0.3, 0.17, 0.28], dtype=torch.float64)
    wy = torch.tensor([0.21, 0.1, 0.15, 0.17, 0.14, 0.23], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda a, b: weighted_w2_columns(a, b, wx, wy), (x, y))


def test_family_uniform_parity_joint_gradients_and_independent_field_permutation():
    torch.manual_seed(71)
    x = torch.randn(2, 8, 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, 8, 3, dtype=torch.float64, requires_grad=True)
    v2, v3 = family(weighted=False), family()
    legacy = v2(x, y)
    weighted = v3(x, y, context=context(torch.ones(8)))
    assert torch.equal(legacy.per_sample_cost, weighted.per_sample_cost)
    assert torch.equal(legacy.scalar_loss, weighted.scalar_loss)
    gx, gy = torch.autograd.grad(weighted.scalar_loss, (x, y))
    assert torch.isfinite(gx).all() and torch.isfinite(gy).all()
    assert torch.count_nonzero(gx).item() == x.numel()
    assert torch.count_nonzero(gy).item() == y.numel()
    plus = torch.tensor(list(product((-1.0, 1.0), repeat=3)), dtype=torch.float64)
    plus = plus[plus.prod(dim=1) > 0].repeat(2, 1)[None]
    minus = plus.clone()
    minus[..., 1] *= -1
    result = v3(plus, minus, context=context(torch.ones(8)))
    assert result.component_results["global_distribution.self.marginal_w2"].scalar_loss == 0
    assert result.component_results["global_distribution.cross.joint_topk_swd"].scalar_loss > 0
    perm = torch.tensor([4, 0, 7, 1, 6, 2, 5, 3])
    assert v3(plus, plus[:, perm], context=context(torch.ones(8))).scalar_loss == 0


def test_weighted_family_gradient_all_channels_and_normalizer_equivalence():
    torch.manual_seed(191)
    x = torch.randn(1, 8, 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn(1, 8, 3, dtype=torch.float64, requires_grad=True)
    wx = torch.tensor([0.4, 0.2, 0.1, 0.3, 0.8, 0.7, 0.2, 0.1], dtype=torch.float64)
    wy = torch.tensor([0.3, 0.5, 0.2, 0.4, 0.9, 0.1, 0.3, 0.6], dtype=torch.float64)
    f = family()
    assert torch.autograd.gradcheck(
        lambda a, b: f(a, b, context=context(wx, wy)).scalar_loss, (x, y)
    )
    offset = torch.tensor([20.0, -3.0, 101.0], dtype=torch.float64)
    scale = torch.tensor([0.01, 70.0, 2.0], dtype=torch.float64)
    torch.testing.assert_close(
        f(
            (x * scale + offset - offset) / scale,
            (y * scale + offset - offset) / scale,
            context=context(wx, wy),
        ).scalar_loss,
        f(x, y, context=context(wx, wy)).scalar_loss,
    )
    normalizer = FieldNormalizer(offset, scale)
    physical = family(units="physical_units", normalizer=normalizer)
    physical_identity = family(units="physical_units")
    torch.testing.assert_close(
        physical(x, y, context=context(wx, wy)).scalar_loss,
        physical_identity(
            normalizer.decode(x), normalizer.decode(y), context=context(wx, wy)
        ).scalar_loss,
    )
    one = family(fields=("a",))
    assert set(one.components_by_key) == {"self_marginal_w2"}
    assert one(x[..., :1], x[..., :1], context=context(wx)).scalar_loss == 0


def test_zero_mass_masks_invalid_measure_and_bank_reference_rejected():
    f = family()
    x = torch.tensor(
        [[[1.0, 2.0, 3.0], [2.0, 5.0, 1.0], [50.0, -5.0, 3.0], [6.0, 1.0, 2.0]]], requires_grad=True
    )
    y = x.detach().clone()
    y[:, 2] = 999.0
    mask = torch.tensor([True, True, False, True])
    w = torch.ones(4)
    masked = f(x, y, context=context(w, generated_mask=mask, reference_mask=mask)).scalar_loss
    assert masked == 0
    grad = torch.autograd.grad(masked, x)[0]
    assert torch.equal(grad[:, 2], torch.zeros_like(grad[:, 2]))
    weighted_zero = w.clone()
    weighted_zero[2] = 0
    assert f(x, y, context=context(weighted_zero)).scalar_loss == 0
    with pytest.raises(ValueError, match="generated_weights and reference_weights"):
        f(x, y)
    with pytest.raises(ValueError, match="nonnegative"):
        f(x, y, context=context(torch.tensor([1.0, -1.0, 1.0, 1.0])))
    with pytest.raises(ValueError, match="positive finite mass"):
        f(x, y, context=context(torch.zeros(4)))
    with pytest.raises(TypeError, match="boolean"):
        f(x, y, context=context(w, generated_mask=torch.ones(4)))
    cfg = dict(f.config)
    cfg["target_use"] = "training_reference"
    with pytest.raises(ValueError, match="reference banks"):
        build_coherence_family(
            "global_distribution",
            cfg,
            DataSpec(("a", "b", "c"), ("1",) * 3, 2, (4,)),
            FieldNormalizer.identity(3),
        )
    cfg = {
        **f.config,
        "components": {
            **f.config["components"],
            "cross": {**f.config["components"]["cross"], "enabled": False},
        },
    }
    with pytest.raises(ValueError, match="requires the mixed projection bank"):
        build_coherence_family(
            "global_distribution",
            cfg,
            DataSpec(("a", "b", "c"), ("1",) * 3, 2, (4,)),
            FieldNormalizer.identity(3),
        )


def test_pair_diagnostics_available_without_pair_training_and_handle_ties():
    f = family()
    x = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 1.0, 2.0], [2.0, 2.0, 3.0], [3.0, 3.0, 4.0]]])
    y = x.clone()
    y[..., 1] = y[..., 1].flip(1)
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = f(x, y, context=context(weights, include_pair_occupations=True, diagnostic_bins=4))
    assert "mutual_pairwise_swd" not in f.components_by_key
    assert set(result.diagnostics["pair_occupations"]) == {"a::b", "a::c", "b::c"}
    pair = result.diagnostics["pair_occupations"]["a::b"]
    assert pair["rank_definition"] == "endpoint_weighted_midrank_ties_share_rank"
    torch.testing.assert_close(pair["generated_own_rank_occupation"].sum(), torch.tensor(1.0))
    torch.testing.assert_close(pair["reference_occupation"].sum(), torch.tensor(1.0))
    assert not torch.equal(
        pair["generated_own_rank_occupation"], pair["reference_own_rank_occupation"]
    )
    assert set(f.pair_diagnostics(x, y, context=context(weights), pairs=[("b", "c")])) == {"b::c"}


def test_rectilinear_area_fingerprint_and_sparse_inclusion_estimator():
    y = torch.tensor([0.0, 2.0, 5.0], dtype=torch.float64)
    x = torch.tensor([0.0, 1.0, 4.0, 10.0], dtype=torch.float64)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack((xx, yy), -1).reshape(-1, 2)
    weights = rectilinear_nodal_weights(coords, (3, 4))
    torch.testing.assert_close(weights.sum(), torch.tensor(50.0, dtype=torch.float64))
    torch.testing.assert_close(
        (weights * coords[:, 0]).sum() / weights.sum(), torch.tensor(5.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        (weights * coords[:, 1]).sum() / weights.sum(), torch.tensor(2.5, dtype=torch.float64)
    )
    selected = torch.tensor([[1, 4, 6], [0, 8, 11]])
    torch.testing.assert_close(
        subset_inclusion_weights(weights, selected, 0.25), 4 * weights[selected]
    )
    receipt = measure_fingerprint(
        weights, metadata={"coordinate_units": "m", "provider": "rectilinear"}
    )
    assert receipt["total_mass"] == 50
    assert (
        receipt["sha256"]
        == measure_fingerprint(
            weights.clone(), metadata={"coordinate_units": "m", "provider": "rectilinear"}
        )["sha256"]
    )
    assert (
        receipt["sha256"]
        != measure_fingerprint(
            weights.flip(0), metadata={"coordinate_units": "m", "provider": "rectilinear"}
        )["sha256"]
    )
    with pytest.raises(ValueError, match="unique"):
        subset_inclusion_weights(weights, torch.tensor([1, 1]), 0.25)
    with pytest.raises(ValueError, match="rectilinear"):
        rectilinear_nodal_weights(coords + torch.randn_like(coords) * 0.01, (3, 4))


def test_nonuniform_two_resolutions_and_frozen_measure_receipt():
    errors = []
    for n in (9, 33):
        coords = torch.linspace(0.0, 1.0, n, dtype=torch.float64).pow(1.7)[:, None]
        weights = rectilinear_nodal_weights(coords, (n,))
        value = coords.clone()
        zero = torch.zeros_like(value)
        # With the reference at zero, W2^2 is the quadrature of x^2.
        result = weighted_w2_columns(value, zero, weights, weights).item()
        errors.append(abs(result - 1 / 3))
        torch.testing.assert_close(weights.sum(), torch.tensor(1.0, dtype=torch.float64))
        torch.testing.assert_close(
            (weights * coords[:, 0]).sum(), torch.tensor(0.5, dtype=torch.float64)
        )
        torch.testing.assert_close(
            weighted_w2_columns(value + 2, value, weights, weights),
            torch.tensor([4.0], dtype=torch.float64),
        )
    assert errors[1] < errors[0] / 8
    f, restored = family(), family()
    native_weights = torch.tensor([0.1, 0.2, 0.7])
    receipt = measure_fingerprint(native_weights, metadata={"provider": "fixture"})
    f.bind_measure_artifact(receipt, weights=native_weights)
    restored.load_state_artifact(f.state_artifact())
    assert restored.state_artifact()["measure"] == receipt
    torch.testing.assert_close(
        restored.state_artifact()["measure_support"]["weights"], native_weights
    )
    damaged = f.state_artifact()
    damaged["measure_support"]["weights"][0] += 0.01
    with pytest.raises(ValueError, match="fingerprint"):
        family().load_state_artifact(damaged)
    with pytest.raises(ValueError, match="requires its frozen native measure"):
        family().load_state_artifact(family().state_artifact())
    other = measure_fingerprint(torch.tensor([0.2, 0.1, 0.7]), metadata={"provider": "fixture"})
    with pytest.raises(ValueError, match="different receipt"):
        restored.bind_measure_artifact(other)
    with pytest.raises(ValueError, match="differs from the frozen"):
        restored(
            torch.zeros(1, 3, 3),
            torch.zeros(1, 3, 3),
            context=context(
                torch.ones(3), measure_metadata={"native_measure_sha256": other["sha256"]}
            ),
        )
