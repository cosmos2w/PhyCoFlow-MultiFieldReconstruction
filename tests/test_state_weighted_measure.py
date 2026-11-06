"""Weighted State is an explicit occupation law, with legacy replay intact."""

import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from phycoflow_reconstruction.coherence import build_coherence_family
from phycoflow_reconstruction.coherence.base import empirical_w2_columns, weighted_w2_columns
from phycoflow_reconstruction.coherence.measures import (
    measure_fingerprint,
    rectilinear_nodal_weights,
    subset_inclusion_weights,
)
from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.evaluation.coherence_set import GlobalDistributionAccumulator


def family(*, weighted=True, fields=("a", "b", "c"), units="model_units", normalizer=None):
    path = "configs/coherence/state_mixed_weighted.yaml" if weighted else "configs/coherence/state_mixed_mean.yaml"
    cfg = load_config(path)["coherence"]["families"]["global_distribution"]
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
    loss = weighted_w2_columns(x, y, torch.tensor([.25, .75]), torch.tensor([.75, .25]))
    torch.testing.assert_close(loss, torch.tensor([9.0], dtype=torch.float64))
    dx, dy = torch.autograd.grad(loss.sum(), (x, y))
    torch.testing.assert_close(dx, torch.tensor([[-1.0], [-4.0]], dtype=torch.float64))
    torch.testing.assert_close(dy, torch.tensor([[5.0], [0.0]], dtype=torch.float64))


def test_separate_support_sizes_and_uniform_mass_exact_legacy_parity():
    x = torch.tensor([[0.0], [3.0]], dtype=torch.float64)
    y = torch.tensor([[0.0], [1.0], [3.0]], dtype=torch.float64)
    torch.testing.assert_close(
        weighted_w2_columns(x, y, torch.tensor([.5, .5]), torch.tensor([.5, .25, .25])),
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
        weighted_w2_columns(x, y, wx, wy), rtol=1e-14, atol=1e-14,
    )


def test_value_gradcheck_covers_all_columns_and_both_endpoints():
    torch.manual_seed(61)
    x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    wx = torch.tensor([.1, .15, .3, .17, .28], dtype=torch.float64)
    wy = torch.tensor([.21, .1, .15, .17, .14, .23], dtype=torch.float64)
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
    plus = torch.tensor(list(product((-1., 1.), repeat=3)), dtype=torch.float64)
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
    wx = torch.tensor([.4, .2, .1, .3, .8, .7, .2, .1], dtype=torch.float64)
    wy = torch.tensor([.3, .5, .2, .4, .9, .1, .3, .6], dtype=torch.float64)
    f = family()
    assert torch.autograd.gradcheck(lambda a, b: f(a, b, context=context(wx, wy)).scalar_loss, (x, y))
    offset = torch.tensor([20., -3., 101.], dtype=torch.float64)
    scale = torch.tensor([.01, 70., 2.], dtype=torch.float64)
    torch.testing.assert_close(
        f((x * scale + offset - offset) / scale, (y * scale + offset - offset) / scale,
          context=context(wx, wy)).scalar_loss,
        f(x, y, context=context(wx, wy)).scalar_loss,
    )
    normalizer = FieldNormalizer(offset, scale)
    physical = family(units="physical_units", normalizer=normalizer)
    physical_identity = family(units="physical_units")
    torch.testing.assert_close(
        physical(x, y, context=context(wx, wy)).scalar_loss,
        physical_identity(normalizer.decode(x), normalizer.decode(y), context=context(wx, wy)).scalar_loss,
    )
    one = family(fields=("a",))
    assert set(one.components_by_key) == {"self_marginal_w2"}
    assert one(x[..., :1], x[..., :1], context=context(wx)).scalar_loss == 0


def test_zero_mass_masks_invalid_measure_and_bank_reference_rejected():
    f = family()
    x = torch.tensor([[[1., 2., 3.], [2., 5., 1.], [50., -5., 3.], [6., 1., 2.]]], requires_grad=True)
    y = x.detach().clone()
    y[:, 2] = 999.
    mask = torch.tensor([True, True, False, True])
    w = torch.ones(4)
    masked = f(x, y, context=context(w, generated_mask=mask, reference_mask=mask)).scalar_loss
    assert masked == 0
    grad = torch.autograd.grad(masked, x)[0]
    assert torch.equal(grad[:, 2], torch.zeros_like(grad[:, 2]))
    weighted_zero = w.clone(); weighted_zero[2] = 0
    assert f(x, y, context=context(weighted_zero)).scalar_loss == 0
    with pytest.raises(ValueError, match="generated_weights and reference_weights"):
        f(x, y)
    with pytest.raises(ValueError, match="nonnegative"):
        f(x, y, context=context(torch.tensor([1., -1., 1., 1.])))
    with pytest.raises(ValueError, match="positive finite mass"):
        f(x, y, context=context(torch.zeros(4)))
    with pytest.raises(TypeError, match="boolean"):
        f(x, y, context=context(w, generated_mask=torch.ones(4)))
    cfg = dict(f.config); cfg["target_use"] = "training_reference"
    with pytest.raises(ValueError, match="reference banks"):
        build_coherence_family("global_distribution", cfg, DataSpec(("a", "b", "c"), ("1",)*3, 2, (4,)), FieldNormalizer.identity(3))
    cfg = {**f.config, "components": {**f.config["components"], "cross": {**f.config["components"]["cross"], "enabled": False}}}
    with pytest.raises(ValueError, match="requires the mixed projection bank"):
        build_coherence_family("global_distribution", cfg, DataSpec(("a", "b", "c"), ("1",)*3, 2, (4,)), FieldNormalizer.identity(3))


def test_pair_diagnostics_available_without_pair_training_and_handle_ties():
    f = family()
    x = torch.tensor([[[0., 0., 1.], [0., 1., 2.], [2., 2., 3.], [3., 3., 4.]]])
    y = x.clone(); y[..., 1] = y[..., 1].flip(1)
    weights = torch.tensor([1., 2., 3., 4.])
    result = f(x, y, context=context(weights, include_pair_occupations=True, diagnostic_bins=4))
    assert "mutual_pairwise_swd" not in f.components_by_key
    assert set(result.diagnostics["pair_occupations"]) == {"a::b", "a::c", "b::c"}
    pair = result.diagnostics["pair_occupations"]["a::b"]
    assert pair["rank_definition"] == "endpoint_weighted_midrank_ties_share_rank"
    torch.testing.assert_close(pair["generated_own_rank_occupation"].sum(), torch.tensor(1.))
    torch.testing.assert_close(pair["reference_occupation"].sum(), torch.tensor(1.))
    assert not torch.equal(pair["generated_own_rank_occupation"], pair["reference_own_rank_occupation"])
    assert set(f.pair_diagnostics(x, y, context=context(weights), pairs=[("b", "c")])) == {"b::c"}


def test_rectilinear_area_fingerprint_and_sparse_inclusion_estimator():
    y = torch.tensor([0., 2., 5.], dtype=torch.float64)
    x = torch.tensor([0., 1., 4., 10.], dtype=torch.float64)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack((xx, yy), -1).reshape(-1, 2)
    weights = rectilinear_nodal_weights(coords, (3, 4))
    torch.testing.assert_close(weights.sum(), torch.tensor(50., dtype=torch.float64))
    torch.testing.assert_close((weights * coords[:, 0]).sum() / weights.sum(), torch.tensor(5., dtype=torch.float64))
    torch.testing.assert_close((weights * coords[:, 1]).sum() / weights.sum(), torch.tensor(2.5, dtype=torch.float64))
    selected = torch.tensor([[1, 4, 6], [0, 8, 11]])
    torch.testing.assert_close(subset_inclusion_weights(weights, selected, .25), 4 * weights[selected])
    receipt = measure_fingerprint(weights, metadata={"coordinate_units": "m", "provider": "rectilinear"})
    assert receipt["total_mass"] == 50
    assert receipt["sha256"] == measure_fingerprint(weights.clone(), metadata={"coordinate_units": "m", "provider": "rectilinear"})["sha256"]
    assert receipt["sha256"] != measure_fingerprint(weights.flip(0), metadata={"coordinate_units": "m", "provider": "rectilinear"})["sha256"]
    with pytest.raises(ValueError, match="unique"):
        subset_inclusion_weights(weights, torch.tensor([1, 1]), .25)
    with pytest.raises(ValueError, match="rectilinear"):
        rectilinear_nodal_weights(coords + torch.randn_like(coords) * .01, (3, 4))


def test_nonuniform_two_resolutions_and_frozen_measure_receipt():
    errors = []
    for n in (9, 33):
        coords = torch.linspace(0., 1., n, dtype=torch.float64).pow(1.7)[:, None]
        weights = rectilinear_nodal_weights(coords, (n,))
        value = coords.clone()
        zero = torch.zeros_like(value)
        # With the reference at zero, W2^2 is the quadrature of x^2.
        result = weighted_w2_columns(value, zero, weights, weights).item()
        errors.append(abs(result - 1 / 3))
        torch.testing.assert_close(weights.sum(), torch.tensor(1., dtype=torch.float64))
        torch.testing.assert_close((weights * coords[:, 0]).sum(), torch.tensor(.5, dtype=torch.float64))
        torch.testing.assert_close(weighted_w2_columns(value + 2, value, weights, weights), torch.tensor([4.], dtype=torch.float64))
    assert errors[1] < errors[0] / 8
    f, restored = family(), family()
    native_weights = torch.tensor([.1, .2, .7])
    receipt = measure_fingerprint(native_weights, metadata={"provider": "fixture"})
    f.bind_measure_artifact(receipt, weights=native_weights)
    restored.load_state_artifact(f.state_artifact())
    assert restored.state_artifact()["measure"] == receipt
    torch.testing.assert_close(restored.state_artifact()["measure_support"]["weights"], native_weights)
    damaged = f.state_artifact()
    damaged["measure_support"]["weights"][0] += .01
    with pytest.raises(ValueError, match="fingerprint"):
        family().load_state_artifact(damaged)
    with pytest.raises(ValueError, match="requires its frozen native measure"):
        family().load_state_artifact(family().state_artifact())
    other = measure_fingerprint(torch.tensor([.2, .1, .7]), metadata={"provider": "fixture"})
    with pytest.raises(ValueError, match="different receipt"):
        restored.bind_measure_artifact(other)
    with pytest.raises(ValueError, match="differs from the frozen"):
        restored(torch.zeros(1, 3, 3), torch.zeros(1, 3, 3),
                 context=context(torch.ones(3), measure_metadata={"native_measure_sha256": other["sha256"]}))


def test_normal_state_accumulator_records_weighted_named_pairs_and_mean_label(tmp_path):
    f = family(fields=("a", "b"))
    spec = DataSpec(("a", "b"), ("1", "1"), 2, (8,))
    acc = GlobalDistributionAccumulator(f, spec, f.config, extra_view=True, extra_query_point_count=8)
    x = torch.tensor([[[0., 0.], [1., 1.], [2., 2.], [3., 3.],
                       [4., 4.], [5., 5.], [6., 6.], [7., 7.]]])
    y = x.clone(); y[..., 1] = y[..., 1].flip(1)
    weights = torch.tensor([7., 3., 2., 2., 1., 1., 1., 0.])
    receipt = measure_fingerprint(weights, metadata={"provider": "fixture"})
    f.bind_measure_artifact(receipt)
    metadata = {"native_measure_sha256": receipt["sha256"], "query_policy": "full_native"}
    acc.update(x, y, torch.zeros(1, 8, 2), "validation:0", physical_weights=weights, measure_metadata=metadata)
    out = acc.finalize(tmp_path, split="validation", checkpoint_label="SOURCE", run_label="fixture", scale="linear")
    report = json.loads(Path(out["report"]).read_text())
    assert report["component_availability"]["pairwise"] is False
    assert report["joint_definition"] == "joint_mixed_mean"
    assert report["measure"]["native_artifact"]["sha256"] == receipt["sha256"]
    with np.load(Path(out["directory"]) / "metrics.npz", allow_pickle=False) as metrics:
        assert str(metrics["joint_definition"]) == "joint_mixed_mean"
        np.testing.assert_array_equal(metrics["joint_mixed_mean"], metrics["joint_top_tail"])
    extra = out["extra"]
    extra_report = json.loads(Path(extra["report"]).read_text())
    assert extra_report["estimand"] == "pooled_equal_sample_self_normalized_physical_occupation_not_per_snapshot_loss"
    with np.load(extra["pair_occupations_payload"], allow_pickle=False) as pairs:
        assert pairs["pair_names"].tolist() == ["a::b"]
        assert pairs["generated_own_rank_occupation"].shape == (1, 1, 16, 16)
        assert pairs["generated_own_rank_occupation"].sum() == pytest.approx(1.0)
        assert not np.array_equal(pairs["generated_own_rank_occupation"], pairs["reference_own_rank_occupation"])


def test_normal_state_accumulator_binds_provider_measure():
    f = family(fields=("a", "b"))
    weights = torch.tensor([1., 2., 3.])
    receipt = measure_fingerprint(weights, metadata={"provider": "fixture"})
    dataset = SimpleNamespace(
        field_names=("a", "b"),
        data_spec=DataSpec(("a", "b"), ("1", "1"), 2, (3,)),
        normalizer=FieldNormalizer.identity(2),
        physical_weights=weights,
        physical_measure_receipt=receipt,
    )
    runtime = SimpleNamespace(
        dataset=dataset, device=torch.device("cpu"),
        config={"coherence": {"families": {"global_distribution": f.config}}},
    )
    acc = GlobalDistributionAccumulator.build(runtime)
    assert acc.family.state_artifact()["measure"] == receipt
    torch.testing.assert_close(acc.family.state_artifact()["measure_support"]["weights"], weights)
    dataset.physical_weights = None
    with pytest.raises(ValueError, match="both physical weights and their receipt"):
        GlobalDistributionAccumulator.build(runtime)
