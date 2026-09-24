"""Value, gradient, update, and recovery parity of shared rollout contexts."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from helpers.physical import model_config, post_config, sample

from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol, build_observation_batch
from phycoflow_reconstruction.models import build_model


@pytest.mark.parametrize("cache", ["condition", "geometry", "static_features"])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_shared_rollout_values_gradients_updates_and_reload(cache, checkpointed):
    torch.manual_seed(216)
    field = sample()
    batch = build_observation_batch(
        [field],
        SensorProtocol(name="structured_block_mean", field_counts={"phi": 16, "vx": 4, "vy": 4}),
    )
    batch = replace(
        batch,
        **{
            key: getattr(batch, key).double()
            for key in ("obs_coords", "obs_values", "query_coords", "target_fields")
        },
    )
    config = model_config()
    config["rollout_checkpointing"] = checkpointed
    baseline = (
        build_model(config, DataSpec(field.field_names, ("dimensionless",) * 3, 3, (8, 8)))
        .double()
        .eval()
    )
    optimized = deepcopy(baseline)
    optimized.rollout_context_cache = cache
    optimizers = [torch.optim.Adam(m.parameters(), lr=1e-5) for m in (baseline, optimized)]
    for step in range(2):
        predictions, gradients = [], []
        for model, optimizer in zip((baseline, optimized), optimizers):
            optimizer.zero_grad(set_to_none=True)
            prediction = model.differentiable_reconstruct(
                batch, steps=3, generator=torch.Generator().manual_seed(234 + step)
            )
            (prediction - batch.target_fields).square().mean().backward()
            predictions.append(prediction.detach())
            gradients.append(
                [
                    p.grad.detach().clone() if p.grad is not None else None
                    for p in model.parameters()
                ]
            )
            optimizer.step()
        torch.testing.assert_close(*predictions, rtol=2e-5, atol=3e-6)
        for a, b in zip(*gradients):
            assert (a is None) == (b is None)
            if a is not None:
                torch.testing.assert_close(a, b, rtol=2e-4, atol=3e-6)
        for a, b in zip(baseline.parameters(), optimized.parameters()):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-7)
        # Simulate recovery: cached autograd state must not survive reconstruction.
        restored = deepcopy(optimized)
        restored.load_state_dict(optimized.state_dict())
        optimizer_state = deepcopy(optimizers[1].state_dict())
        optimizers[1] = torch.optim.Adam(restored.parameters(), lr=1e-5)
        optimizers[1].load_state_dict(optimizer_state)
        optimized = restored


def test_shared_rollout_rejects_training_dropout_mode():
    field = sample()
    model = build_model(
        {**model_config(), "rollout_context_cache": "condition"},
        DataSpec(field.field_names, ("dimensionless",) * 3, 3, (8, 8)),
    )
    batch = build_observation_batch(
        [field],
        SensorProtocol(name="structured_block_mean", field_counts={"phi": 16, "vx": 4, "vy": 4}),
    )
    with pytest.raises(ValueError, match="eval mode"):
        model.differentiable_reconstruct(batch, steps=2)


def test_rollout_execution_survives_source_inheritance(tmp_path):
    import yaml

    from phycoflow_reconstruction.cli import _load_case_config
    from phycoflow_reconstruction.config import validate_config

    source = tmp_path / "source"
    source.mkdir()
    config = post_config(tmp_path / "synthetic.h5", source)
    (source / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    config["model"] = {"name": "gl_rbf_cq"}
    validate_config(config)
    path = tmp_path / "posttrain.yaml"
    path.write_text(yaml.safe_dump(config))
    resolved = _load_case_config(path, tmp_path, "fixture", [])
    assert (
        resolved["model"]["physical_field_transform"] == model_config()["physical_field_transform"]
    )
    assert resolved["optimization"]["rollout_execution"] == {
        "context_cache": "static_features",
        "checkpointing": True,
    }
    assert resolved["source"]["inherited_base_keys"] == ["dataset", "model", "observations"]


@pytest.mark.parametrize("inherit", [False, True])
def test_rollout_execution_rejects_resolved_model_without_physical_transform(tmp_path, inherit):
    from phycoflow_reconstruction.config import validate_config

    config = post_config(tmp_path / "synthetic.h5", tmp_path / "source")
    config["inherit_base_config"] = inherit
    config["model"].pop("physical_field_transform")
    with pytest.raises(ValueError, match="requires a physical CQ model"):
        validate_config(config)
