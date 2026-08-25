"""Downstream adapter and B/C data/RNG identity gates."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

from phycoflow_reconstruction.contracts import DataSpec, FieldSample, ObservationBatch
from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol, build_observation_batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks" / "gl_rbf_cq_migration_200ep"
CONFIG_B = BENCHMARK_ROOT / "configs" / "B_gl_rbf_cq_legacy_mha_200ep.yaml"
CONFIG_C = BENCHMARK_ROOT / "configs" / "C_gl_rbf_cq_cached_kv_200ep.yaml"


def _load(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    assert isinstance(config, dict)
    return config


def _field_sample(index: int, *, n_points: int = 512) -> FieldSample:
    generator = torch.Generator(device="cpu").manual_seed(7100 + index)
    coords = torch.rand(n_points, 2, generator=generator)
    values = torch.randn(n_points, 5, generator=generator)
    return FieldSample(
        values=values,
        coordinates=coords,
        coordinates_raw=coords.clone(),
        time=torch.tensor(float(index)),
        trajectory_id="gate-trajectory",
        time_index=index,
        conditions=torch.empty(0),
        field_names=("CH4", "CO", "T", "U_1", "p"),
        logical_shape=(n_points,),
    )


def _batch_from_config(config: dict, samples: list[FieldSample]) -> ObservationBatch:
    settings = config["observations"]
    protocol = SensorProtocol(
        name=settings["protocol"],
        field_count_ranges={
            name: (int(value["count_min"]), int(value["count_max"]))
            for name, value in settings["fields"].items()
        },
        seed=int(settings["seed"]),
    )
    return build_observation_batch(samples, protocol, query_points=128)


def test_b_c_sensor_batches_and_rng_draws_are_identical():
    samples = [_field_sample(index) for index in range(3)]
    first = _batch_from_config(_load(CONFIG_B), samples)
    second = _batch_from_config(_load(CONFIG_C), samples)
    assert first.sample_ids == second.sample_ids

    pairs = (
        (first.obs_coords, second.obs_coords),
        (first.obs_values, second.obs_values),
        (first.obs_field_ids, second.obs_field_ids),
        (first.obs_valid_mask, second.obs_valid_mask),
        (first.query_coords, second.query_coords),
        (first.query_valid_mask, second.query_valid_mask),
        (first.target_fields, second.target_fields),
        (first.obs_indices, second.obs_indices),
    )
    for left, right in pairs:
        assert left is not None and right is not None
        assert torch.equal(left, right)

    valid_counts = first.obs_valid_mask.sum(dim=1)
    assert torch.all((valid_counts >= 192) & (valid_counts <= 384))


def _portable_core(model):
    for name in ("portable_core", "core", "velocity_model", "model"):
        candidate = getattr(model, name, None)
        if candidate is None:
            continue
        if hasattr(candidate, "model") and hasattr(candidate.model, "prepare_condition_context"):
            return candidate.model
        if hasattr(candidate, "prepare_condition_context"):
            return candidate
    raise AssertionError(
        "GL_rbf_CQ adapter must expose its portable core through portable_core, core, "
        "velocity_model, or model"
    )


def _adapter_batch() -> ObservationBatch:
    generator = torch.Generator(device="cpu").manual_seed(9901)
    coords = torch.rand(2, 17, 2, generator=generator)
    indices = torch.tensor([[0, 2, 5, 8, 12, 16], [1, 3, 6, 9, 13, 15]])
    values = torch.randn(2, 17, 5, generator=generator)
    return ObservationBatch(
        obs_coords=torch.stack([coords[i, indices[i]] for i in range(2)]),
        obs_values=torch.randn(2, 6, 1, generator=generator),
        obs_field_ids=torch.arange(6).remainder(5).repeat(2, 1),
        obs_valid_mask=torch.tensor(
            [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.bool
        ),
        query_coords=coords,
        query_valid_mask=torch.ones(2, 17, dtype=torch.bool),
        target_fields=values,
        sample_ids=("adapter-a", "adapter-b"),
        obs_indices=indices,
        logical_shapes=((17,), (17,)),
    )


def test_adapter_and_portable_core_forward_loss_gradient_equality():
    pytest.importorskip("phycoflow_reconstruction.models.flows.gl_rbf_cq")
    portable = pytest.importorskip("phycoflow_pointcloud")
    from phycoflow_reconstruction.models import build_model
    from phycoflow_reconstruction.registry import MODEL_REGISTRY

    if "gl_rbf_cq" not in MODEL_REGISTRY.names():
        pytest.skip("GL_rbf_CQ adapter is present but not registered yet")

    config = _load(CONFIG_B)
    data_spec = DataSpec(
        field_names=("CH4", "CO", "T", "U_1", "p"),
        field_units=("unknown",) * 5,
        coordinate_dim=2,
        logical_shape=(17,),
        mesh_type="point",
    )
    adapter = build_model(config["model"], data_spec)
    core = _portable_core(adapter)
    portable_config = copy.deepcopy(config["model"])
    portable_config.pop("name", None)
    portable_config["model_name"] = "GL_rbf_CQ"
    portable_config["coord_dim"] = 2
    portable_model = portable.build_pointcloud_model(
        portable_config, n_fields=5, device="cpu"
    )
    portable_model.model.load_state_dict(core.state_dict(), strict=True)

    batch = _adapter_batch()
    state = batch.target_fields.detach().clone()
    time = torch.tensor([0.2, 0.8])
    adapter_output = adapter.velocity(batch, state, time)
    core_output = portable_model.model(
        time,
        state,
        batch.query_coords,
        batch.obs_coords,
        batch.obs_values,
        batch.obs_valid_mask,
        batch.obs_field_ids,
    )
    torch.testing.assert_close(adapter_output, core_output, rtol=0, atol=0)

    adapter.zero_grad(set_to_none=True)
    portable_model.zero_grad(set_to_none=True)
    adapter_output = adapter.velocity(batch, state, time)
    core_output = portable_model.model(
        time,
        state,
        batch.query_coords,
        batch.obs_coords,
        batch.obs_values,
        batch.obs_valid_mask,
        batch.obs_field_ids,
    )
    adapter_output.square().mean().backward()
    core_output.square().mean().backward()
    adapter_parameters = dict(core.named_parameters())
    for name, portable_parameter in portable_model.model.named_parameters():
        assert name in adapter_parameters
        adapter_gradient = adapter_parameters[name].grad
        if portable_parameter.grad is None:
            assert adapter_gradient is None
        else:
            assert adapter_gradient is not None
            torch.testing.assert_close(
                adapter_gradient, portable_parameter.grad, rtol=0, atol=0
            )


def test_b_c_actual_configs_have_identical_seeded_initial_state():
    pytest.importorskip("phycoflow_reconstruction.models.flows.gl_rbf_cq")
    from phycoflow_reconstruction.models import build_model
    from phycoflow_reconstruction.registry import MODEL_REGISTRY

    if "gl_rbf_cq" not in MODEL_REGISTRY.names():
        pytest.skip("GL_rbf_CQ adapter is present but not registered yet")

    data_spec = DataSpec(
        field_names=("CH4", "CO", "T", "U_1", "p"),
        field_units=("unknown",) * 5,
        coordinate_dim=2,
        logical_shape=(17,),
        mesh_type="point",
    )
    config_b = _load(CONFIG_B)["model"]
    config_c = _load(CONFIG_C)["model"]
    torch.manual_seed(42)
    arm_b = build_model(config_b, data_spec)
    torch.manual_seed(42)
    arm_c = build_model(config_c, data_spec)
    assert arm_b.state_dict().keys() == arm_c.state_dict().keys()
    for key in arm_b.state_dict():
        torch.testing.assert_close(arm_b.state_dict()[key], arm_c.state_dict()[key], rtol=0, atol=0)


def test_adapter_ema_lifecycle_is_generic_and_resumable():
    pytest.importorskip("phycoflow_reconstruction.models.flows.gl_rbf_cq")
    from phycoflow_reconstruction.models import build_model
    from phycoflow_reconstruction.registry import MODEL_REGISTRY

    if "gl_rbf_cq" not in MODEL_REGISTRY.names():
        pytest.skip("GL_rbf_CQ adapter is present but not registered yet")

    config = _load(CONFIG_B)
    data_spec = DataSpec(
        field_names=("CH4", "CO", "T", "U_1", "p"),
        field_units=("unknown",) * 5,
        coordinate_dim=2,
        logical_shape=(17,),
        mesh_type="point",
    )
    model = build_model(config["model"], data_spec)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    loss = model.training_loss(_adapter_batch()).total
    loss.backward()
    optimizer.step()
    model.after_optimizer_step()
    aux = model.training_aux_state_dict()
    assert aux["model_ema"]["num_updates"] == 1

    resumed = build_model(config["model"], data_spec)
    resumed.load_training_aux_state_dict(aux)
    assert resumed.training_aux_state_dict()["model_ema"]["num_updates"] == 1
    with resumed.evaluation_weight_context():
        assert resumed.training_aux_state_dict()["model_ema"]["num_updates"] == 1
