"""Geometric coverage, execution equivalence, and legacy CQ contracts."""

import copy
import math
from pathlib import Path

import pytest
import torch
from helpers.pointcloud import _small_core_config

from phycoflow_reconstruction.config import load_config, validate_config
from phycoflow_reconstruction.models.flows.pointcloud import build_pointcloud_model
from phycoflow_reconstruction.models.flows.pointcloud.adapters.gl_rbf_cq_adapter import GLRbfCQ


def _config(**overrides):
    return {
        **_small_core_config(2),
        "cq_measurement_support_mode": "rbf_value_coverage",
        "cq_support_bandwidth": 0.1,
        **overrides,
    }


def _model(n_fields=3, **overrides):
    return build_pointcloud_model(_config(**overrides), n_fields=n_fields, device="cpu")


def _inputs(n_fields=3):
    rng = torch.Generator().manual_seed(232)
    coords = torch.rand(2, 17, 2, generator=rng)
    indices = torch.tensor([[0, 3, 7, 11, 14, 16]]).expand(2, -1)
    return {
        "x1": torch.randn(2, 17, n_fields, generator=rng),
        "coords": coords,
        "obs_coords": coords[:, [0, 3, 7, 11, 14, 16]],
        "obs_values": torch.randn(2, 6, 1, generator=rng),
        "obs_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]).bool(),
        "obs_field_ids": torch.arange(6).remainder(n_fields).expand(2, -1),
        "obs_indices": indices,
    }


def _features(core, distances, fields, values, *, valid=None, available=None):
    fields = torch.tensor([fields], dtype=torch.int32)
    valid = torch.ones_like(fields, dtype=torch.bool) if valid is None else torch.tensor([valid])
    if available is None:
        available = torch.zeros(1, core.n_fields)
        available.scatter_add_(1, fields.masked_fill(~valid, 0).long(), valid.float())
        available = (available > 0).float()
    context = {
        "raw_obs_values": torch.tensor(values).reshape(1, -1, 1),
        "raw_obs_field_ids": fields,
        "field_availability": available,
    }
    return core._cq_measurement_support_from_geometry(
        torch.tensor(distances).reshape(1, 1, -1),
        torch.arange(fields.shape[1]).reshape(1, 1, -1),
        valid[:, None, :],
        context,
    )[0, 0]


def test_t_only_coverage_retains_absolute_distance_and_measurement_amplitude():
    core = _model(n_fields=5).model
    near = _features(core, [0.0004, 0.0064], [2, 2], [2.0, 4.0])
    far = _features(core, [0.0904, 0.0964], [2, 2], [2.0, 4.0])
    # Adding the same squared-distance offset preserves normalized RBF weights.
    torch.testing.assert_close(near[:5], far[:5], rtol=1e-5, atol=1e-6)
    sigma = float(core.log_rbf_sigma.detach().exp())
    ratio = math.exp(-(0.0064 - 0.0004) / (2 * sigma**2 + 1e-12))
    assert float(near[2].detach()) == pytest.approx((2 + 4 * ratio) / (1 + ratio))
    assert float(near[7].detach()) == pytest.approx(math.exp(-0.02))
    assert float(far[7].detach()) == pytest.approx(math.exp(-4.52))
    assert near[12] > far[12]
    assert torch.equal(near[15:], torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))


def test_fieldwise_interpolation_is_stable_with_distant_field_and_invalid_padding():
    core = _model().model
    raw = _features(
        core, [0.0, 9.0, float("inf")], [0, 2, -1], [0.0, 7.0, float("nan")],
        valid=[True, True, False],
    )
    assert torch.isfinite(raw).all()
    torch.testing.assert_close(raw[:3], torch.tensor([0.0, 0.0, 7.0]))
    torch.testing.assert_close(raw[3:6], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(raw[9:], torch.tensor([1.0, 0.0, 1.0]))
    # An empty candidate set is finite, even for fields available elsewhere.
    empty = _features(
        core, [float("inf")], [-1], [float("nan")], valid=[False],
        available=torch.tensor([[0.0, 0.0, 1.0]]),
    )
    assert torch.isfinite(empty).all()
    assert torch.count_nonzero(empty[:9]) == 0
    assert empty[-1] == 1


def test_kernel_mass_counts_valid_neighbors_and_padding_does_not_contribute():
    core = _model(n_fields=1).model
    one = _features(core, [0.002, 0.002], [0, -1], [3.0, 999.0], valid=[True, False])
    two = _features(core, [0.002, 0.002], [0, 0], [3.0, 3.0])
    assert one[0] == two[0] == 3
    assert one[1] == two[1]
    assert float(one[2].detach()) == pytest.approx(math.log1p(math.exp(-0.1)) / math.log1p(3))
    assert two[2] > one[2]


def test_interpolation_sigma_has_gradients_but_cannot_change_geometric_coverage():
    core = _model(n_fields=1).model
    raw = _features(core, [0.001, 0.006, 0.009], [0, 0, 0], [1.0, 2.0, 4.0])
    gradient = torch.autograd.grad(raw[0], core.log_rbf_sigma, retain_graph=True)[0]
    assert torch.isfinite(gradient) and gradient.abs() > 0
    geometry_gradient = torch.autograd.grad(raw[1:3].sum(), core.log_rbf_sigma)[0]
    assert geometry_gradient == 0
    with torch.no_grad():
        core.log_rbf_sigma.add_(0.7)
    changed = _features(core, [0.001, 0.006, 0.009], [0, 0, 0], [1.0, 2.0, 4.0])
    assert not torch.isclose(raw[0], changed[0])
    torch.testing.assert_close(raw[1:], changed[1:], rtol=0, atol=0)


def test_global_field_availability_distinguishes_no_local_hit_and_unobserved_field():
    core = _model().model.eval()
    obs_coords = torch.tensor([[[0.0, 0.0], [0.02, 0.0], [0.03, 0.0], [0.8, 0.8]]])
    with torch.no_grad():
        condition = core.prepare_condition_context(
            obs_coords, torch.zeros(1, 4, 1), torch.ones(1, 4).bool(),
            torch.tensor([[0, 0, 0, 2]], dtype=torch.int32),
        )
        query = core.prepare_query_context(
            torch.zeros(1, 1, 2), condition, cache_level="static_features"
        )
    raw = query["raw_measurement_support"][0, 0]
    assert raw[3] == 1  # Truly observed zero value, covered locally.
    assert raw[5] == raw[8] == 0  # Field 2 is not in the shared Top-3.
    torch.testing.assert_close(raw[9:], torch.tensor([1.0, 0.0, 1.0]))


@pytest.mark.parametrize("n_fields", [1, 3])
@pytest.mark.parametrize("fusion", ["additive", "structured_concat"])
@pytest.mark.parametrize("readout", ["lowrank", "full"])
def test_coverage_survives_head_fusion_and_all_inference_cache_paths(
    n_fields, fusion, readout, monkeypatch
):
    model = _model(n_fields, cq_fusion_mode=fusion, cq_readout_mode=readout).eval()
    core = model.model
    values = _inputs(n_fields)
    captured = []
    handle = core.cq_head.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
    with torch.no_grad():
        condition = core.prepare_condition_context(
            values["obs_coords"], values["obs_values"], values["obs_mask"],
            values["obs_field_ids"],
        )
        kwargs = {
            "t": torch.tensor([0.2, 0.8]), "x_t_chunk": values["x1"],
            "coords_chunk": values["coords"], "condition_context": condition,
        }
        reference = core.forward_query_chunk(**kwargs)
        for level in ["geometry", "static_features"]:
            context = core.prepare_query_context(
                values["coords"], condition, cache_level=level, chunk_size=5
            )
            output = core.forward_query_chunk(**kwargs, query_context=context)
            torch.testing.assert_close(output, reference, rtol=1e-5, atol=2e-6)
            if level == "static_features":
                assert context["raw_measurement_support"].shape[-1] == 4 * n_fields
                # The head receives the original numeric features without LayerNorm.
                torch.testing.assert_close(
                    captured[-1][..., -4 * n_fields:], context["raw_measurement_support"],
                    rtol=0, atol=0,
                )
        geometry = model.prepare_reconstruction_geometry_cache(
            coords=values["coords"], obs_coords=values["obs_coords"],
            obs_mask=values["obs_mask"], chunk_size=5,
        )

        def unexpected_search(*args, **kwargs):
            raise AssertionError("persistent geometry must not trigger another neighbor search")

        monkeypatch.setattr(core, "_get_topk_neighbors", unexpected_search)
        persistent = core.prepare_query_context(
            values["coords"], condition, cache_level="static_features", chunk_size=5,
            precomputed_geometry=geometry,
        )
        torch.testing.assert_close(
            core.forward_query_chunk(**kwargs, query_context=persistent), reference,
            rtol=1e-5, atol=2e-6,
        )
    handle.remove()
    summary = core.model_summary()
    assert summary["measurement_support_width"] == 4 * n_fields
    assert summary["measurement_support_normalize"] is False
    assert not hasattr(core, "cq_measurement_support_norm")


@pytest.mark.parametrize("cache_level", ["geometry", "static_features"])
def test_differentiable_rollout_caches_preserve_parameter_gradients(cache_level):
    model = _model().eval()
    values = _inputs()
    core = model.model
    gradients = []
    outputs = []
    for use_cache in [False, True]:
        model.zero_grad(set_to_none=True)
        condition = core.prepare_condition_context(
            values["obs_coords"], values["obs_values"], values["obs_mask"],
            values["obs_field_ids"],
        )
        query = core.prepare_rollout_query_context(
            values["coords"], condition, cache_level=cache_level
        ) if use_cache else None
        output = core.forward_query_chunk(
            torch.tensor([0.3, 0.7]), values["x1"], values["coords"], condition, query
        )
        output.square().mean().backward()
        outputs.append(output.detach())
        gradients.append([None if p.grad is None else p.grad.clone() for p in model.parameters()])
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-5, atol=2e-6)
    for first, second in zip(*gradients):
        assert (first is None) == (second is None)
        if first is not None:
            assert torch.isfinite(second).all()
            torch.testing.assert_close(first, second, rtol=5e-5, atol=2e-6)


def test_coverage_query_microbatch_loss_and_gradients_match_full_bridge():
    full = _model()
    micro = copy.deepcopy(full)
    values = _inputs()
    torch.manual_seed(221)
    loss, _ = full.training_loss(**values)
    loss.backward()
    torch.manual_seed(221)
    micro_loss, _ = micro.training_loss_microbatched(
        **values, query_microbatch_size=5, backward=True, reuse_condition_context=True
    )
    torch.testing.assert_close(loss.detach(), micro_loss, rtol=1e-6, atol=1e-7)
    for first, second in zip(full.parameters(), micro.parameters()):
        assert (first.grad is None) == (second.grad is None)
        if first.grad is not None:
            assert torch.isfinite(second.grad).all()
            torch.testing.assert_close(first.grad, second.grad, rtol=5e-5, atol=2e-6)


def test_new_recipe_default_and_historical_constructor_defaults_are_separate():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "cases/turbulent_combustion/configs/base/gl_rbf_cq_cached_kv_5000ep.yaml")
    validate_config(config)
    assert config["model"]["cq_measurement_support_mode"] == "rbf_value_coverage"
    assert config["model"]["cq_support_bandwidth"] == 0.05
    assert config["model"]["cq_measurement_support_normalize"] is False

    legacy = _small_core_config(2)
    legacy.pop("cq_measurement_support_mode")
    # Adapter missing-mode semantics retain rbf_value_support; the tensor
    # builder's historical missing-mode default remains none.
    torch.manual_seed(32)
    implicit = GLRbfCQ(coordinate_dim=2, num_fields=3, logical_shape=(17,), **legacy)
    torch.manual_seed(32)
    explicit = GLRbfCQ(
        coordinate_dim=2, num_fields=3, logical_shape=(17,),
        **{**legacy, "cq_measurement_support_mode": "rbf_value_support"},
    )
    assert implicit.model.cq_measurement_support_width == 6
    implicit.load_state_dict(explicit.state_dict(), strict=True)
    for key, value in implicit.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
    assert build_pointcloud_model(legacy, n_fields=3).model.cq_measurement_support_mode == "none"
    with pytest.raises(RuntimeError, match="size mismatch"):
        _model().load_state_dict(implicit.state_dict(), strict=True)


@pytest.mark.parametrize("bandwidth", [0, -0.1, float("nan"), float("inf")])
def test_invalid_support_bandwidth_rejected_by_builder_and_config(bandwidth):
    with pytest.raises(ValueError, match="cq_support_bandwidth"):
        _model(cq_support_bandwidth=bandwidth)
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "cases/turbulent_combustion/configs/base/gl_rbf_cq_cached_kv_5000ep.yaml")
    config["model"]["cq_support_bandwidth"] = bandwidth
    with pytest.raises(ValueError, match="cq_support_bandwidth"):
        validate_config(config)
