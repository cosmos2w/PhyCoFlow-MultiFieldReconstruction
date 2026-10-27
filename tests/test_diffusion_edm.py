"""Numerical contracts for conditional EDM and resumable microbatch training."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch import nn

from phycoflow_reconstruction.contracts import DataSpec, FieldSample
from phycoflow_reconstruction.data.observations import rasterize_observations
from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol, build_observation_batch
from phycoflow_reconstruction.models import build_model
from phycoflow_reconstruction.training.model_lifecycle import (
    add_training_aux_state,
    load_training_aux_state,
)


def make_batch(count=1, protocol=None):
    y, x = torch.meshgrid(torch.linspace(0, 1, 5), torch.linspace(0, 1, 7), indexing="ij")
    coords = torch.stack((x, y), -1).flatten(0, 1)
    samples = [
        FieldSample(
            values=torch.stack((x + i / 10, y - i / 10), -1).flatten(0, 1),
            coordinates=coords,
            coordinates_raw=coords,
            time=torch.tensor(0.0),
            trajectory_id=f"sample_{i}",
            time_index=0,
            conditions=torch.empty(0),
            field_names=("a", "b"),
            logical_shape=(5, 7),
        )
        for i in range(count)
    ]
    return build_observation_batch(
        samples, protocol or SensorProtocol(field_counts={"a": 3}, seed=17)
    )


def make_model(**overrides):
    config = {
        "name": "diffusion_pde",
        "formulation": "edm",
        "backbone": "conditional_unet",
        "base_channels": 8,
        "channel_multipliers": [1, 2],
        "num_res_blocks": 1,
        "time_embed_dim": 16,
        "attention_levels": [1],
        "attention_heads": 2,
        "dropout": 0.0,
        "high_noise_fraction": 0.2,
        "train_microbatch_size": 2,
    }
    config.update(overrides)
    return build_model(config, DataSpec(("a", "b"), ("1", "1"), 2, (5, 7)))


def test_edm_preconditioning_matches_equations_and_observed_x0_projection():
    model = make_model(sigma_data=1.5)

    class Ones(nn.Module):
        def forward(self, noisy, values, mask, time):
            return torch.ones_like(noisy)

    model.denoiser = Ones()
    noisy = torch.randn(2, 2, 5, 7)
    sigma = torch.tensor([0.01, 80.0]).reshape(2, 1, 1, 1)
    values = torch.zeros_like(noisy)
    mask = torch.zeros_like(noisy)
    values[:, 0, 0, 0] = 3
    mask[:, 0, 0, 0] = 1
    sd = 1.5
    expected = sd**2 / (sigma**2 + sd**2) * noisy + sigma * sd / (sigma**2 + sd**2).sqrt()
    expected = expected * (1 - mask) + values * mask
    torch.testing.assert_close(model.denoise(noisy, sigma, values, mask), expected)


def test_edm_microbatch_matches_full_batch_gradient_with_uneven_last_chunk():
    batch = make_batch(3)
    full = make_model()
    # Exercise every layer, not just the initially zero output projection.
    nn.init.normal_(full.denoiser.output_projection.weight, std=0.01)
    micro = deepcopy(full)
    torch.manual_seed(123)
    expected = full.training_loss(batch)
    expected.total.backward()
    torch.manual_seed(123)
    actual = micro.training_backward(batch)
    torch.testing.assert_close(actual.total, expected.total)
    for (name, a), (_, b) in zip(full.named_parameters(), micro.named_parameters()):
        assert a.grad is not None, name
        torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=1e-4, msg=name)


def test_edm_odd_grid_seed_target_independence_and_differentiable_sampling():
    batch = make_batch()
    model = make_model().eval()
    a = model.reconstruct(batch, steps=3, generator=torch.Generator().manual_seed(7))
    b = model.reconstruct(
        replace(batch, target_fields=None), steps=3, generator=torch.Generator().manual_seed(7)
    )
    assert torch.equal(a.prediction, b.prediction)
    assert a.prediction.shape == batch.target_fields.shape
    assert a.diagnostics["denoiser_calls"] == 5
    values, mask = rasterize_observations(batch, 2)
    prediction = a.prediction.transpose(1, 2).reshape_as(values)
    assert torch.equal(prediction[mask.bool()], values[mask.bool()])
    prediction = model.differentiable_reconstruct(replace(batch, target_fields=None), steps=3)
    prediction.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_edm_heun_constant_clean_oracle_reaches_clean_endpoint():
    model = make_model()
    batch = make_batch()
    clean = torch.full((1, 2, 5, 7), 0.3)
    model.denoise = lambda noisy, sigma, values, mask: clean
    result = model.reconstruct(batch, steps=4, generator=torch.Generator().manual_seed(9))
    values, mask = rasterize_observations(batch, 2)
    expected = clean * (1 - mask) + values * mask
    torch.testing.assert_close(result.prediction, expected.flatten(2).transpose(1, 2))


def test_edm_ema_auxiliary_resume_and_live_weight_restore():
    model = make_model()
    model.after_optimizer_step()
    with torch.no_grad():
        next(model.parameters()).add_(0.5)
    model.after_optimizer_step()
    payload = add_training_aux_state({"model": deepcopy(model.state_dict())}, model)
    restored = make_model()
    restored.load_state_dict(payload["model"], strict=True)
    load_training_aux_state(restored, payload)
    assert restored._ema.num_updates == model._ema.num_updates == 2
    before = deepcopy(restored.state_dict())
    with restored.evaluation_weight_context():
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, model._ema.shadow[name])
    for name, value in restored.state_dict().items():
        assert torch.equal(value, before[name])
    with pytest.raises(KeyError, match="training_aux_state"):
        load_training_aux_state(restored, {"model": payload["model"]})


def test_edm_variable_multifield_sensors_ignore_padding_and_influence_unknown_entries():
    batch = make_batch(3, SensorProtocol(field_count_ranges={"a": (1, 4), "b": (1, 6)}, seed=17))
    model = make_model().eval()
    nn.init.normal_(model.denoiser.output_projection.weight, std=0.01)
    noisy = torch.randn(3, 2, 5, 7)
    values, mask = rasterize_observations(batch, 2)
    conditioned = model.denoise(noisy, torch.tensor(80.0), values, mask)
    unconditioned = model.denoise(noisy, torch.tensor(80.0), torch.zeros_like(values), mask)
    assert (conditioned - unconditioned)[~mask.bool()].square().mean() > 0
    dirty_values = batch.obs_values.clone()
    dirty_values[~batch.obs_valid_mask] = 1e6
    dirty = replace(batch, obs_values=dirty_values)
    dirty_raster, dirty_mask = model._conditions(dirty)
    assert torch.equal(dirty_raster, values)
    assert torch.equal(dirty_mask, mask)


def test_edm_rejects_nonpoint_observation_operator():
    batch = make_batch()
    batch.metadata["protocol"]["name"] = "structured_block_mean"
    with pytest.raises(ValueError, match="block means"):
        make_model().reconstruct(batch, steps=3)


@pytest.mark.parametrize(
    "settings",
    [
        {"sigma_data": 0},
        {"sigma_min": 80},
        {"high_noise_fraction": 1.1},
        {"precision": "fp16"},
        {"backbone": "plain_cnn"},
        {"train_microbatch_size": 1.5},
        {"train_microbatch_size": True},
    ],
)
def test_edm_rejects_invalid_direct_construction(settings):
    with pytest.raises(ValueError):
        make_model(**settings)
