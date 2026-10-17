"""Regression contracts for amplitude-preserving single-field CQ shortcuts."""

from __future__ import annotations

import copy

import pytest
import torch
from helpers.pointcloud import _small_core_config

from phycoflow_reconstruction.models.flows.pointcloud import build_pointcloud_model


def _model(n_fields: int = 1, *, normalize: bool = True):
    config = _small_core_config(2)
    config["num_latent_blocks"] = 1
    config["cq_measurement_support_normalize"] = normalize
    torch.manual_seed(3107)
    return build_pointcloud_model(config, n_fields=n_fields, device="cpu").eval()


def _inputs(n_fields: int = 1):
    generator = torch.Generator().manual_seed(527)
    coords = torch.rand(2, 11, 2, generator=generator)
    indices = torch.tensor([[0, 2, 5, 8], [1, 3, 7, 9]])
    return {
        "x1": torch.randn(2, 11, n_fields, generator=generator),
        "coords": coords,
        "obs_coords": torch.stack([coords[i, indices[i]] for i in range(2)]),
        "obs_values": torch.randn(2, 4, 1, generator=generator),
        "obs_mask": torch.ones(2, 4, dtype=torch.bool),
        "obs_field_ids": torch.arange(4).remainder(n_fields).repeat(2, 1),
        "obs_indices": indices,
    }


def _forward(model, values):
    return model.model(
        torch.tensor([0.2, 0.8]),
        values["x1"],
        values["coords"],
        values["obs_coords"],
        values["obs_values"],
        values["obs_mask"],
        values["obs_field_ids"],
    )


@pytest.mark.parametrize("measurement", [-4.0, -1.0, 0.0, 0.5, 2.0, 4.0])
def test_single_field_head_retains_measurement_and_its_gradient(measurement: float):
    model = _model()
    values = _inputs()
    values["obs_values"] = torch.full((2, 4, 1), measurement, requires_grad=True)
    fusion_inputs = []
    handle = model.model.cq_fusion_norm.register_forward_pre_hook(
        lambda _module, args: fusion_inputs.append(args[0])
    )
    try:
        output = _forward(model, values)
    finally:
        handle.remove()
    assert torch.isfinite(output).all()
    raw = torch.cat(fusion_inputs, dim=1)[..., -2:]
    torch.testing.assert_close(raw[..., 0], torch.full_like(raw[..., 0], measurement))
    torch.testing.assert_close(raw[..., 1], torch.ones_like(raw[..., 1]))
    # A uniform change of every observed value must reach the explicit shortcut
    # with unit derivative, even at amplitudes where LayerNorm(2) saturates.
    derivative = torch.autograd.grad(raw[..., 0].sum(), values["obs_values"])[0]
    torch.testing.assert_close(
        derivative.sum(dim=(1, 2)),
        torch.full((2,), values["coords"].shape[1], dtype=derivative.dtype),
    )


def test_single_field_retains_checkpoint_schema_and_matches_disabled_normalization():
    normalized = _model(normalize=True)
    disabled = _model(normalize=False)
    normalized_state = normalized.state_dict()
    disabled_state = disabled.state_dict()
    norm_keys = {
        "model.cq_measurement_support_norm.weight",
        "model.cq_measurement_support_norm.bias",
    }
    assert set(normalized_state) - set(disabled_state) == norm_keys
    assert set(disabled_state) - set(normalized_state) == set()
    assert normalized_state["model.cq_measurement_support_norm.weight"].shape == (2,)
    assert normalized_state["model.cq_measurement_support_norm.bias"].shape == (2,)
    for name, tensor in disabled_state.items():
        torch.testing.assert_close(normalized_state[name], tensor, rtol=0, atol=0)
    reloaded = _model(normalize=True)
    reloaded.load_state_dict(normalized_state, strict=True)
    values = _inputs()
    torch.testing.assert_close(
        _forward(normalized, values), _forward(disabled, values), rtol=0, atol=0
    )
    torch.testing.assert_close(
        _forward(normalized, values), _forward(reloaded, values), rtol=0, atol=0
    )


@pytest.mark.parametrize("n_fields", [2, 5])
def test_multifield_head_still_uses_unchanged_layernorm(n_fields: int):
    model = _model(n_fields)
    values = _inputs(n_fields)
    norm_inputs, norm_outputs, fusion_inputs = [], [], []
    # Nontrivial affine parameters ensure the complete old LayerNorm operation
    # survives, rather than only its default weight=1, bias=0 special case.
    with torch.no_grad():
        model.model.cq_measurement_support_norm.weight.copy_(torch.linspace(0.5, 1.5, 2 * n_fields))
        model.model.cq_measurement_support_norm.bias.copy_(torch.linspace(-0.2, 0.2, 2 * n_fields))

    def capture_norm(_module, args, output):
        norm_inputs.append(args[0])
        norm_outputs.append(output)

    norm_handle = model.model.cq_measurement_support_norm.register_forward_hook(capture_norm)
    fusion_handle = model.model.cq_fusion_norm.register_forward_pre_hook(
        lambda _module, args: fusion_inputs.append(args[0])
    )
    try:
        output = _forward(model, values)
    finally:
        norm_handle.remove()
        fusion_handle.remove()
    assert torch.isfinite(output).all()
    assert norm_inputs
    for raw, normalized, fusion in zip(norm_inputs, norm_outputs, fusion_inputs):
        expected = torch.nn.functional.layer_norm(
            raw,
            (2 * n_fields,),
            model.model.cq_measurement_support_norm.weight,
            model.model.cq_measurement_support_norm.bias,
            model.model.cq_measurement_support_norm.eps,
        )
        torch.testing.assert_close(normalized, expected, rtol=0, atol=0)
        torch.testing.assert_close(fusion[..., -2 * n_fields :], expected, rtol=0, atol=0)


def test_single_field_full_and_cached_reconstruction_agree():
    model = _model()
    values = _inputs()
    sample_inputs = {
        name: values[name]
        for name in ("coords", "obs_coords", "obs_values", "obs_mask", "obs_field_ids")
    }
    torch.manual_seed(811)
    full = model.sample(**sample_inputs, n_steps=2, obs_consistency_mode="none")
    torch.manual_seed(811)
    cached = model.sample(
        **sample_inputs,
        n_steps=2,
        obs_consistency_mode="none",
        reconstruction_execution_mode="cached_streamed",
        reconstruction_query_chunk_size=5,
        reconstruction_cache_level="static_features",
    )
    torch.testing.assert_close(cached, full, rtol=1e-5, atol=2e-6)


def test_single_field_microbatch_loss_and_gradients_agree():
    full = _model()
    micro = copy.deepcopy(full)
    values = _inputs()
    torch.manual_seed(914)
    full_loss, _ = full.training_loss(**values)
    full_loss.backward()
    torch.manual_seed(914)
    micro_loss, metrics = micro.training_loss_microbatched(
        **values,
        query_microbatch_size=5,
        backward=True,
        reuse_condition_context=True,
    )
    torch.testing.assert_close(micro_loss, full_loss.detach(), rtol=3e-6, atol=3e-7)
    assert metrics["query_microbatches"] == 3.0
    for (full_name, full_parameter), (micro_name, micro_parameter) in zip(
        full.named_parameters(), micro.named_parameters()
    ):
        assert full_name == micro_name
        if full_parameter.grad is None or micro_parameter.grad is None:
            assert full_parameter.grad is None and micro_parameter.grad is None
        else:
            torch.testing.assert_close(
                micro_parameter.grad, full_parameter.grad, rtol=5e-5, atol=5e-7
            )
