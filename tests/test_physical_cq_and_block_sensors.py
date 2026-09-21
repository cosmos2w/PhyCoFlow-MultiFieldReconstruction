"""Guard nonlinear coordinates, pooled observations, and training gradients."""

from dataclasses import replace

import pytest
import torch
from helpers.physical import model_config, post_config, sample

from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol, build_observation_batch
from phycoflow_reconstruction.models import build_model
from phycoflow_reconstruction.training.rollout import (
    differentiable_reconstruction,
    differentiable_rf_rollout,
)


def test_block_means_centroids_and_manifest_replay():
    field = sample()
    protocol = SensorProtocol(
        name="structured_block_mean", field_counts={"phi": 16, "vx": 4, "vy": 4}
    )
    batch = build_observation_batch([field], protocol)
    torch.testing.assert_close(batch.obs_coords[0, 0], field.coordinates[[0, 1, 8, 9]].mean(0))
    torch.testing.assert_close(batch.obs_values[0, 0, 0], field.values[[0, 1, 8, 9], 0].mean())
    torch.testing.assert_close(
        batch.obs_values[0, 16, 0], field.values.reshape(8, 8, 3)[:4, :4, 1].mean()
    )
    assert not torch.allclose(batch.obs_values[0, 16, 0], field.values[9, 1])
    pairs = list(zip(batch.obs_indices[0].tolist(), batch.obs_field_ids[0].tolist()))
    replay = build_observation_batch([field], protocol, manifest_indices={"run1:0": pairs[::-1]})
    torch.testing.assert_close(replay.obs_coords, batch.obs_coords.flip(1), atol=0, rtol=0)
    torch.testing.assert_close(replay.obs_values, batch.obs_values.flip(1), atol=0, rtol=0)
    with pytest.raises(ValueError, match="invalid block identifier"):
        build_observation_batch([field], protocol, manifest_indices={"run1:0": [(63, 1)]})


@pytest.mark.parametrize("checkpointed", [False, True])
def test_physical_adapter_keeps_rf_loss_and_rollout_in_source_coordinates(checkpointed):
    field = sample()
    batch = build_observation_batch(
        [field],
        SensorProtocol(name="structured_block_mean", field_counts={"phi": 16, "vx": 4, "vy": 4}),
    )
    spec = DataSpec(field.field_names, ("dimensionless",) * 3, 3, (8, 8))
    config = model_config()
    config["rollout_checkpointing"] = checkpointed
    model = build_model(config, spec).eval()
    encoded = model._encoded_batch(batch)
    physical = batch.target_fields
    expected = torch.stack(
        (
            physical[..., 0],
            torch.asinh(physical[..., 1] / 0.3),
            torch.asinh(physical[..., 2] / 0.2),
        ),
        -1,
    )
    expected = (expected - torch.tensor([0.1, -0.2, 0.3])) / torch.tensor([0.5, 2.0, 0.4])
    torch.testing.assert_close(encoded.target_fields, expected)
    torch.testing.assert_close(model.decode(encoded.target_fields), physical)
    # Encoding a physical block mean differs from averaging encoded pixels.
    assert (
        abs(float(encoded.obs_values[0, 16, 0] - expected.reshape(8, 8, 3)[:4, :4, 1].mean()))
        > 0.01
    )
    torch.manual_seed(99)
    direct = model.core.training_loss(encoded).total
    torch.manual_seed(99)
    wrapped = model.training_loss(batch).total
    torch.testing.assert_close(wrapped, direct, atol=0, rtol=0)
    direct_grads = torch.autograd.grad(direct, tuple(model.parameters()), allow_unused=True)
    wrapped_grads = torch.autograd.grad(wrapped, tuple(model.parameters()), allow_unused=True)
    for left, right in zip(direct_grads, wrapped_grads):
        if left is not None:
            torch.testing.assert_close(left, right, atol=0, rtol=0)
    settings = {"mode": "none", "final_clamp": False}
    actual = differentiable_reconstruction(
        model,
        replace(batch, target_fields=None),
        steps=2,
        solver="euler",
        generator=torch.Generator().manual_seed(35),
        observation_config=settings,
    )
    source = differentiable_rf_rollout(
        model.core,
        replace(encoded, target_fields=None),
        steps=2,
        solver="euler",
        generator=torch.Generator().manual_seed(35),
        observation_config=settings,
    )
    torch.testing.assert_close(actual, model.decode(source), atol=0, rtol=0)
    expected_grads = torch.autograd.grad(
        model.decode(source).square().mean(), tuple(model.parameters()), allow_unused=True
    )
    actual_grads = torch.autograd.grad(
        actual.square().mean(), tuple(model.parameters()), allow_unused=True
    )
    assert sum(float(g.square().sum()) for g in actual_grads if g is not None) > 0
    for left, right in zip(expected_grads, actual_grads):
        if left is not None:
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
    prediction = model.reconstruct(
        batch, steps=2, generator=torch.Generator().manual_seed(35)
    ).prediction
    torch.testing.assert_close(actual, prediction, rtol=1e-5, atol=2e-6)
    rebuilt = build_model(config, spec)
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    assert model.state_dict().keys() == model.core.state_dict().keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, rebuilt.state_dict()[name], rtol=0, atol=0)


def test_physical_model_rejects_pointwise_clamping():
    config = model_config()
    config["obs_consistency_final_clamp"] = True
    with pytest.raises(ValueError, match="without clamping"):
        build_model(config, DataSpec(("phi", "vx", "vy"), ("1",) * 3, 3, (8, 8)))


@pytest.mark.parametrize("rollout_cache", ["none", "static_features"])
def test_native_posttraining_roundtrip_with_physical_fields_and_block_means(
    tmp_path, rollout_cache
):
    pytest.importorskip("gudhi")
    import json

    import h5py
    import numpy as np
    from helpers.coherence import _base_config

    from phycoflow_reconstruction.config import validate_config
    from phycoflow_reconstruction.training.base_training import run_base_training
    from phycoflow_reconstruction.training.post_training import run_post_training
    from phycoflow_reconstruction.training.run_store import file_sha256, load_project_checkpoint

    field = sample()
    dataset_path = tmp_path / "physical.h5"
    with h5py.File(dataset_path, "w") as handle:
        values = field.values.numpy().reshape(8, 8, 3)
        handle.create_dataset(
            "fields",
            data=np.stack([values, values * 0.9, values * 1.1, values * 1.2])[
                :, None, :, :, None, :
            ],
        )
        handle.create_dataset("coordinates", data=field.coordinates.numpy().reshape(8, 8, 1, 3))
        handle.create_dataset("time", data=[0.0])
        handle.create_dataset("conditions", data=np.empty((4, 0), dtype=np.float32))
        handle.attrs["field_names"] = json.dumps(field.field_names)
        handle.attrs["grid_shape"] = "[8, 8]"
        for split, indices in {"train": [0], "validation": [1, 2], "test": [3]}.items():
            handle.create_dataset(f"splits/{split}", data=indices)
    base = _base_config(dataset_path)
    base["dataset"].update(
        field_names=list(field.field_names),
        field_units=["1"] * 3,
        normalization="none",
        coordinate_dim=3,
        grid_shape=[8, 8],
        augmentation={"kind": "periodic_translate_rot90", "vector_fields": ["vx", "vy"]},
    )
    base["model"] = {
        **model_config(),
        "query_points": 64,
        "rollout_checkpointing": True,
        "data_query_points": 16,
    }
    base["observations"] = {
        "protocol": "structured_block_mean",
        "fields": {"phi": {"count": 16}, "vx": {"count": 4}, "vy": {"count": 4}},
    }
    base["optimization"]["batch_size"] = 1
    source = run_base_training(base, case_dir=tmp_path / "case")
    source_hash = file_sha256(source / "checkpoints/last.pt")
    post = post_config(dataset_path, source, context_cache=rollout_cache)
    post.update(dataset=base["dataset"], model=base["model"], observations=base["observations"])
    validate_config(post)
    child = run_post_training(post, case_dir=tmp_path / "case", max_steps=1)
    assert json.loads((child / "status.json").read_text())["status"] == "integration_truncated"
    child = run_post_training(post, case_dir=tmp_path / "case", resume=child)
    assert file_sha256(source / "checkpoints/last.pt") == source_hash
    assert json.loads((child / "status.json").read_text())["status"] == "completed"
    before = json.loads((child / "evaluation/before.json").read_text())
    after = json.loads((child / "evaluation/after.json").read_text())
    assert before["sensor_manifest_sha256"] == after["sensor_manifest_sha256"]
    assert before["generation_seed"] == after["generation_seed"]
    history = (child / "metrics/history.jsonl").read_text()
    for component in ("self.persistence", "mutual.persistence"):
        assert f"topology.{component}" in history
    assert "topology.mutual.persistence" in after["coherence"]["components"]
    original = load_project_checkpoint(source / "checkpoints/last.pt")
    trained = load_project_checkpoint(child / "checkpoints/last.pt")
    assert any(
        not torch.equal(value, trained["model"][key]) for key, value in original["model"].items()
    )
    if rollout_cache != "none":
        uninterrupted = run_post_training(post, case_dir=tmp_path / "uninterrupted")
        complete = load_project_checkpoint(uninterrupted / "checkpoints/last.pt")
        for key, value in trained["model"].items():
            torch.testing.assert_close(value, complete["model"][key], rtol=0, atol=0)
        for key, state in trained["optimizer"]["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(
                    value, complete["optimizer"]["state"][key][name], rtol=0, atol=0
                )


@pytest.mark.parametrize("rotations", range(4))
def test_periodic_augmentation_matches_source_vector_convention(rotations):
    import numpy as np

    from phycoflow_reconstruction.data.augmentation import periodic_translate_rot90

    field = sample()
    expected = np.rot90(
        np.roll(field.values.numpy().reshape(8, 8, 3), (2, 5), axis=(0, 1)), rotations, axes=(0, 1)
    ).copy()
    vx, vy = expected[..., 1].copy(), expected[..., 2].copy()
    expected[..., 1], expected[..., 2] = ((vx, vy), (vy, -vx), (-vx, -vy), (-vy, vx))[rotations]
    actual = periodic_translate_rot90(
        field.values, (8, 8), (1, 2), shifts=(2, 5), rotations=rotations
    )
    torch.testing.assert_close(actual, torch.from_numpy(expected.reshape(-1, 3)), atol=0, rtol=0)


def test_data_query_subsampling_retains_full_rollout_and_native_loss():
    field = sample()
    batch = build_observation_batch(
        [field],
        SensorProtocol(name="structured_block_mean", field_counts={"phi": 16, "vx": 4, "vy": 4}),
    )
    config = {**model_config(), "data_query_points": 16}
    model = build_model(config, DataSpec(field.field_names, ("1",) * 3, 3, (8, 8))).eval()
    torch.manual_seed(53)
    selected = model._data_batch(batch)
    expected = model.core.training_loss(model._encoded_batch(selected)).total
    torch.manual_seed(53)
    actual = model.training_loss(batch).total
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert selected.query_coords.shape[1] == 16
    ids = selected.metadata["query_indices"][0]
    torch.testing.assert_close(selected.target_fields, batch.target_fields[:, ids], rtol=0, atol=0)
    result = model.differentiable_reconstruct(
        batch, steps=1, generator=torch.Generator().manual_seed(8)
    )
    assert result.shape == batch.target_fields.shape == (1, 64, 3)
