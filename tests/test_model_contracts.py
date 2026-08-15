"""One parametrized contract test replaces separate per-model smoke scripts."""

import pytest
import torch

from phycoflow_reconstruction.contracts import DataSpec, FieldSample
from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol, build_observation_batch
from phycoflow_reconstruction.models import build_model
from phycoflow_reconstruction.training.rollout import differentiable_reconstruction


def _batch():
    y, x = torch.meshgrid(torch.linspace(0, 1, 4), torch.linspace(0, 1, 4), indexing="ij")
    coords = torch.stack((x, y), -1).reshape(-1, 2)
    sample = FieldSample(
        values=torch.stack((torch.sin(x * 3.14), torch.cos(y * 3.14)), -1).reshape(-1, 2),
        coordinates=coords,
        coordinates_raw=coords,
        time=torch.tensor(0.0),
        trajectory_id="tiny",
        time_index=0,
        conditions=torch.empty(0),
        field_names=("u", "v"),
        logical_shape=(4, 4),
    )
    protocol = SensorProtocol(field_counts={"u": 4}, seed=3)
    return build_observation_batch([sample], protocol)


@pytest.mark.parametrize(
    "model_config",
    [
        {"name": "coordinate_mlp", "hidden_dim": 16, "fourier_bands": 4},
        {"name": "mlp_rbf", "hidden_dim": 16, "fourier_bands": 4},
        {"name": "deeponet", "width": 16, "basis_dim": 8},
        {"name": "senseiver", "width": 16, "num_latents": 4, "heads": 2, "depth": 1},
        {"name": "geofno", "hidden_channels": 8, "layers": 1},
        {"name": "diffusion_pde", "hidden_channels": 8},
        {"name": "latent_fm", "latent_channels": 4, "stage": 1},
        {
            "name": "pointcloud_ffm",
            "backbone": "gl_rbf_enh",
            "hidden_dim": 16,
            "latent_dim": 16,
            "num_latents": 4,
            "heads": 2,
            "latent_blocks": 1,
            "gather_topk": 2,
        },
        {"name": "pointcloud_ffm", "backbone": "fno", "fno_hidden_channels": 8},
    ],
)
def test_registered_model_loss_and_reconstruction(model_config):
    batch = _batch()
    spec = DataSpec(("u", "v"), ("1", "1"), 2, (4, 4), mesh_type="structured")
    model = build_model(model_config, spec)
    loss = model.training_loss(batch)
    assert torch.isfinite(loss.total)
    loss.total.backward()
    with torch.no_grad():
        result = model.reconstruct(batch, steps=1)
    assert result.prediction.shape == batch.target_fields.shape
    assert torch.isfinite(result.prediction).all()
    if model_config["name"] == "latent_fm" and model_config.get("stage") == 1:
        assert "post_training" not in model.capabilities.stages
    else:
        assert "post_training" in model.capabilities.stages


def test_pinn_requires_physics_provider():
    spec = DataSpec(("u",), ("1",), 1, (4,))
    with pytest.raises(ValueError, match="physics_provider"):
        build_model({"name": "pinn"}, spec)


def test_model_builder_rejects_parameters_from_another_adapter():
    spec = DataSpec(("u",), ("1",), 1, (4,))
    with pytest.raises(ValueError, match="unsupported coordinate_mlp model keys"):
        build_model({"name": "coordinate_mlp", "latent_dim": 32}, spec)


@pytest.mark.parametrize(
    "model_config",
    [
        {"name": "diffusion_pde", "hidden_channels": 8},
        {
            "name": "pointcloud_ffm",
            "backbone": "gl_rbf_enh",
            "hidden_dim": 16,
            "latent_dim": 16,
            "num_latents": 4,
            "heads": 2,
            "latent_blocks": 1,
            "gather_topk": 2,
            "query_chunk_size": 3,
        },
    ],
)
def test_generative_sampling_seed_and_sensor_clamp(model_config):
    batch = _batch()
    spec = DataSpec(("u", "v"), ("1", "1"), 2, (4, 4), mesh_type="structured")
    model = build_model(model_config, spec).eval()
    first = model.reconstruct(
        batch, steps=2, generator=torch.Generator().manual_seed(91)
    ).prediction
    second = model.reconstruct(
        batch, steps=2, generator=torch.Generator().manual_seed(91)
    ).prediction
    assert torch.equal(first, second)

    query_indices = batch.metadata["query_indices"][0]
    query_lookup = {int(point): index for index, point in enumerate(query_indices.tolist())}
    valid = batch.obs_valid_mask[0]
    for point, field, value in zip(
        batch.obs_indices[0, valid].tolist(),
        batch.obs_field_ids[0, valid].tolist(),
        batch.obs_values[0, valid, 0],
    ):
        assert first[0, query_lookup[int(point)], int(field)] == value


@pytest.mark.parametrize(
    "model_config",
    [
        {"name": "diffusion_pde", "hidden_channels": 8},
        {
            "name": "pointcloud_ffm",
            "backbone": "gl_rbf_enh",
            "hidden_dim": 16,
            "latent_dim": 16,
            "num_latents": 4,
            "heads": 2,
            "latent_blocks": 1,
            "gather_topk": 2,
        },
    ],
)
def test_generative_reconstruction_rejects_zero_steps(model_config):
    spec = DataSpec(("u", "v"), ("1", "1"), 2, (4, 4), mesh_type="structured")
    model = build_model(model_config, spec)
    with pytest.raises(ValueError, match="steps must be at least one"):
        model.reconstruct(_batch(), steps=0)


def test_diffusion_group_width_is_validated_before_torch_raises():
    spec = DataSpec(("u", "v"), ("1", "1"), 2, (4, 4), mesh_type="structured")
    with pytest.raises(ValueError, match="multiple of four"):
        build_model({"name": "diffusion_pde", "hidden_channels": 6}, spec)


@pytest.mark.parametrize(
    "model_config",
    [
        {"name": "coordinate_mlp", "hidden_dim": 16, "fourier_bands": 4},
        {"name": "mlp_rbf", "hidden_dim": 16, "fourier_bands": 4},
        {"name": "deeponet", "width": 16, "basis_dim": 8},
        {"name": "senseiver", "width": 16, "num_latents": 4, "heads": 2, "depth": 1},
        {"name": "geofno", "hidden_channels": 8, "layers": 1},
        {"name": "diffusion_pde", "hidden_channels": 8},
    ],
)
def test_native_differentiable_reconstruction_has_parameter_gradient(model_config):
    batch = _batch()
    spec = DataSpec(("u", "v"), ("1", "1"), 2, (4, 4), mesh_type="structured")
    model = build_model(model_config, spec)
    prediction = differentiable_reconstruction(
        model,
        batch,
        steps=2,
        solver="euler",
        generator=torch.Generator().manual_seed(17),
        observation_config={"mode": "none"},
    )
    prediction.square().mean().backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)


def test_latent_stage2_differentiable_reconstruction(tmp_path):
    batch = _batch()
    spec = DataSpec(("u", "v"), ("1", "1"), 2, (4, 4), mesh_type="structured")
    stage1 = build_model({"name": "latent_fm", "latent_channels": 4, "stage": 1}, spec)
    checkpoint = tmp_path / "stage1.pt"
    torch.save(
        {
            "model_name": "latent_fm",
            "model_config": {"stage": 1},
            "model": stage1.state_dict(),
        },
        checkpoint,
    )
    stage2 = build_model(
        {
            "name": "latent_fm",
            "latent_channels": 4,
            "stage": 2,
            "stage1_checkpoint": str(checkpoint),
        },
        spec,
    )
    prediction = differentiable_reconstruction(
        stage2,
        batch,
        steps=2,
        solver="euler",
        generator=torch.Generator().manual_seed(19),
        observation_config={"mode": "none"},
    )
    prediction.square().mean().backward()
    gradients = [
        parameter.grad
        for parameter in stage2.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert any(torch.count_nonzero(gradient) for gradient in gradients)
