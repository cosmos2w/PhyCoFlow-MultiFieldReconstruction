from dataclasses import replace

import pytest
import torch

from phycoflow_reconstruction.contracts import ObservationBatch
from phycoflow_reconstruction.models.deterministic.senseiver import SenseiverRegressor
from phycoflow_reconstruction.models.flows.pointcloud.adapters.pointcloud_ffm_adapter import (
    PointCloudFFM,
)
from phycoflow_reconstruction.models.source_residual import FrozenSourceResidual
from phycoflow_reconstruction.training.rollout import differentiable_reconstruction


def batch():
    return ObservationBatch(
        obs_coords=torch.tensor([[[0.0], [0.5], [9.0]]]),
        obs_values=torch.tensor([[[2.0], [3.0], [float("nan")]]]),
        obs_field_ids=torch.tensor([[0, 1, -1]]),
        obs_valid_mask=torch.tensor([[True, True, False]]),
        query_coords=torch.tensor([[[0.0], [0.25], [0.5], [0.75]]]),
        query_valid_mask=torch.ones(1, 4, dtype=torch.bool),
        target_fields=torch.randn(1, 4, 2), sample_ids=["sample"],
        obs_indices=torch.tensor([[0, 2, -1]]),
        metadata={"query_indices": torch.arange(4)[None]},
    )


def rollout(model, data, seed, mode="none"):
    return differentiable_reconstruction(
        model, data, steps=2, solver="euler", generator=torch.Generator().manual_seed(seed),
        observation_config={"mode": mode, "final_clamp": mode != "none"},
    )


@pytest.mark.parametrize("kind", ["senseiver", "ffm"])
def test_real_source_parity_members_and_frozen_state(kind):
    torch.manual_seed(3)
    if kind == "senseiver":
        source = SenseiverRegressor(1, 2, width=8, num_latents=4, heads=2, depth=1)
        mode = "none"
    else:
        source = PointCloudFFM(1, 2, (4,), hidden_dim=8, latent_dim=8, num_latents=4,
                               heads=2, latent_blocks=1, gather_topk=2, fourier_bands=2,
                               field_embedding_dim=4, rff_features=4)
        mode = "hard"
    wrapper = FrozenSourceResidual(source, coordinate_dim=1, field_scales=[2, 3], width=8)
    data = batch()
    # Real model paths cannot consume invalid NaN sensor padding even when masked.
    data = replace(data, obs_values=torch.nan_to_num(data.obs_values),
                   obs_field_ids=data.obs_field_ids.clamp_min(0))
    original = {k: v.clone() for k, v in source.state_dict().items()}
    wrapper.train()
    for seed in (12, 21):
        expected = rollout(source, data, seed, mode)
        actual = rollout(wrapper, data, seed, mode)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        gradient = torch.autograd.grad(actual.sum(), wrapper.head[-1].bias)[0]
        assert gradient.abs().sum() > 0
    assert wrapper.capabilities.generative == (kind == "ffm")
    if kind == "ffm":
        assert not torch.equal(rollout(wrapper, data, 12, mode), rollout(wrapper, data, 21, mode))
    assert not source.training and all(not p.requires_grad for p in source.parameters())
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_nonzero_correction_sensor_null_gradient_and_chunk_parity():
    torch.manual_seed(4)
    source = SenseiverRegressor(1, 2, width=8, num_latents=4, heads=2, depth=1)
    wrapper = FrozenSourceResidual(source, coordinate_dim=1, field_scales=[2, 3], width=8)
    with torch.no_grad():
        wrapper.head[-1].weight.normal_()
        wrapper.head[-1].bias.fill_(0.2)
    data = batch()
    pred = torch.randn(1, 4, 2)
    delta = wrapper.correction(data, pred)
    assert torch.isfinite(delta).all()
    changed_padding = replace(data, obs_coords=data.obs_coords.clone(), obs_values=data.obs_values.clone(),
                              obs_field_ids=data.obs_field_ids.clone())
    changed_padding.obs_coords[0, 2] = -100
    changed_padding.obs_values[0, 2] = 500
    changed_padding.obs_field_ids[0, 2] = 999
    torch.testing.assert_close(wrapper.correction(changed_padding, pred), delta, rtol=0, atol=0)
    assert delta[0, 0, 0] == 0 and delta[0, 2, 1] == 0
    assert delta[0, 0, 1] != 0 and delta[0, 2, 0] != 0
    assert (delta.abs() <= torch.tensor([0.5, 0.75])).all()
    sensor_gradient = torch.autograd.grad(delta[0, 0, 0] + delta[0, 2, 1], wrapper.head[-1].bias)[0]
    assert torch.count_nonzero(sensor_gradient) == 0
    chunks = []
    for index in range(4):
        part = replace(data, query_coords=data.query_coords[:, index:index+1],
                       query_valid_mask=data.query_valid_mask[:, index:index+1], target_fields=None,
                       metadata={"query_indices": data.metadata["query_indices"][:, index:index+1]})
        chunks.append(wrapper.correction(part, pred[:, index:index+1]))
    torch.testing.assert_close(torch.cat(chunks, dim=1), delta, rtol=1e-6, atol=1e-7)


def test_rejects_missing_ids_and_bad_scales():
    source = SenseiverRegressor(1, 2, width=8, num_latents=4, heads=2, depth=1)
    with pytest.raises(ValueError, match="field_scales"):
        FrozenSourceResidual(source, coordinate_dim=1, field_scales=[0, 1])
    wrapper = FrozenSourceResidual(source, coordinate_dim=1, field_scales=[1, 1], width=8)
    with pytest.raises(ValueError, match="point IDs"):
        wrapper.correction(replace(batch(), obs_indices=None), torch.zeros(1, 4, 2))
