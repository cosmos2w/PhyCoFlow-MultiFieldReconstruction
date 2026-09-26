"""Focused contract checks for the MIMONet operator adapter."""

import pytest
import torch

from phycoflow_reconstruction.config.validate import validate_config
from phycoflow_reconstruction.contracts import DataSpec, ObservationBatch
from phycoflow_reconstruction.models import build_model
from phycoflow_reconstruction.training.rollout import differentiable_reconstruction

FIELD_NAMES = ("a", "b", "c")


def _model(*, conditioning_fields=("b", "a"), sensor_capacities=(2, 2)):
    spec = DataSpec(
        field_names=FIELD_NAMES,
        field_units=("1", "1", "1"),
        coordinate_dim=2,
        logical_shape=(2, 3),
        mesh_type="point",
    )
    return build_model(
        {
            "name": "mimonet",
            "conditioning_fields": list(conditioning_fields),
            "sensor_capacities": list(sensor_capacities),
            "basis_dim": 4,
            "branch_hidden_dim": 8,
            "trunk_hidden_dim": 7,
            "merge_type": "mul",
        },
        spec,
    )


def _batch(
    *,
    obs_coords=None,
    obs_values=None,
    obs_field_ids=None,
    obs_valid_mask=None,
    query_coords=None,
    target_fields="default",
):
    if obs_coords is None:
        obs_coords = torch.tensor(
            [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]],
            dtype=torch.float32,
        )
    batch_size, obs_count, _ = obs_coords.shape
    if obs_values is None:
        obs_values = torch.arange(1, obs_count + 1, dtype=torch.float32).reshape(
            1, obs_count, 1
        ).expand(batch_size, -1, -1).clone()
    if obs_field_ids is None:
        obs_field_ids = torch.tensor([[0, 1, 0, 1]], dtype=torch.long).expand(
            batch_size, -1
        )
    if obs_valid_mask is None:
        obs_valid_mask = torch.ones(batch_size, obs_count, dtype=torch.bool)
    if query_coords is None:
        query_coords = torch.tensor(
            [[[0.15, 0.25], [0.45, 0.55], [0.75, 0.85]]], dtype=torch.float32
        ).expand(batch_size, -1, -1)
    if isinstance(target_fields, str) and target_fields == "default":
        target_fields = torch.randn(batch_size, query_coords.shape[1], len(FIELD_NAMES))
    return ObservationBatch(
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_field_ids=obs_field_ids,
        obs_valid_mask=obs_valid_mask,
        query_coords=query_coords,
        query_valid_mask=torch.ones(query_coords.shape[:2], dtype=torch.bool),
        target_fields=target_fields,
        sample_ids=tuple(f"sample-{index}" for index in range(batch_size)),
    )


def _training_config():
    return {
        "stage": "base_training",
        "case": "mimonet-fixture",
        "dataset": {
            "path": "unused.h5",
            "field_names": list(FIELD_NAMES),
            "field_units": ["1"] * len(FIELD_NAMES),
        },
        "model": {
            "name": "mimonet",
            "conditioning_fields": ["b", "a"],
            "sensor_capacities": [2, 1],
            "basis_dim": 4,
            "branch_hidden_dim": 8,
            "trunk_hidden_dim": 7,
            "merge_type": "mul",
        },
        "observations": {
            "protocol": "random_uniform",
            "fields": {"b": {"count": 2}, "a": {"count": 1}},
        },
        "optimization": {"lr": 1.0e-3},
        "runtime": {"seed": 1},
        "output": {"experiment_name": "mimonet-contract"},
    }


def test_two_branch_multiplicative_output_matches_independent_reference():
    model = _model().eval()
    batch = _batch()
    value_input, location_input = model._packed_branches(batch)

    value_branch = model.branch_nets[0](value_input)
    location_branch = model.branch_nets[1](location_input)
    combined_branch = value_branch * location_branch
    trunk = model.trunk_net(batch.query_coords).reshape(
        batch.query_coords.shape[0],
        batch.query_coords.shape[1],
        model.basis_dim,
        len(FIELD_NAMES),
    )
    expected = (combined_branch[:, None, :, None] * trunk).sum(dim=2) + model.bias

    torch.testing.assert_close(model.forward_batch(batch), expected)


def test_sensor_slots_follow_field_order_pad_and_ignore_invalid_entries():
    model = _model(conditioning_fields=("b", "a"), sensor_capacities=(2, 2))
    batch = _batch(
        obs_coords=torch.tensor(
            [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]]
        ),
        obs_values=torch.tensor([[[10.0], [20.0], [30.0], [9.0e5]]]),
        obs_field_ids=torch.tensor([[0, 1, 0, 2]]),
        obs_valid_mask=torch.tensor([[True, True, True, False]]),
    )

    value_input, location_input = model._packed_branches(batch)
    expected_values = torch.tensor([[20.0, 1.0, 0.0, 0.0, 10.0, 1.0, 30.0, 1.0]])
    expected_locations = torch.tensor(
        [[0.3, 0.4, 1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 1.0, 0.5, 0.6, 1.0]]
    )
    torch.testing.assert_close(value_input, expected_values)
    torch.testing.assert_close(location_input, expected_locations)

    # The masked, undeclared field may contain arbitrary data without reaching either branch.
    batch.obs_values[0, 3, 0] = -8.0e8
    batch.obs_coords[0, 3] = torch.tensor([7.0e7, -6.0e7])
    changed_value_input, changed_location_input = model._packed_branches(batch)
    torch.testing.assert_close(changed_value_input, expected_values)
    torch.testing.assert_close(changed_location_input, expected_locations)


def test_undeclared_valid_field_and_sensor_capacity_overflow_are_rejected():
    model = _model(conditioning_fields=("a",), sensor_capacities=(1,))
    undeclared = _batch(
        obs_coords=torch.tensor([[[0.1, 0.2], [0.3, 0.4]]]),
        obs_values=torch.tensor([[[1.0], [2.0]]]),
        obs_field_ids=torch.tensor([[0, 1]]),
        obs_valid_mask=torch.tensor([[True, True]]),
    )
    with pytest.raises(ValueError, match="undeclared conditioning field"):
        model._packed_branches(undeclared)

    overflow = _batch(
        obs_coords=torch.tensor([[[0.1, 0.2], [0.3, 0.4]]]),
        obs_values=torch.tensor([[[1.0], [2.0]]]),
        obs_field_ids=torch.tensor([[0, 0]]),
        obs_valid_mask=torch.tensor([[True, True]]),
    )
    with pytest.raises(ValueError, match="exceeds its sensor capacity"):
        model._packed_branches(overflow)


def test_config_requires_matching_condition_fields_and_sufficient_capacities():
    config = _training_config()
    validate_config(config)

    config["observations"]["fields"]["c"] = {"count": 1}
    with pytest.raises(ValueError, match="observations must match conditioning_fields exactly"):
        validate_config(config)

    del config["observations"]["fields"]["c"]
    config["model"]["sensor_capacities"][0] = 1
    with pytest.raises(ValueError, match="sensor capacity for b is below observation count"):
        validate_config(config)


def test_prediction_and_backward_are_finite_with_shared_field_outputs():
    model = _model()
    batch = _batch()
    prediction = model.forward_batch(batch)

    assert prediction.shape == (1, 3, len(FIELD_NAMES))
    assert torch.isfinite(prediction).all()
    loss = model.training_loss(batch).total
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_post_training_reconstruction_does_not_require_or_read_targets():
    model = _model().eval()
    targetless = _batch(target_fields=None)
    prediction = differentiable_reconstruction(
        model,
        targetless,
        steps=1,
        solver="euler",
        generator=torch.Generator().manual_seed(5),
        observation_config={"mode": "none"},
    )

    assert "post_training" in model.capabilities.stages
    assert prediction.shape == (1, 3, len(FIELD_NAMES))
    assert torch.isfinite(prediction).all()

    targetless.target_fields = torch.full_like(prediction, 1.0e6)
    with_targets = differentiable_reconstruction(
        model,
        targetless,
        steps=1,
        solver="euler",
        generator=torch.Generator().manual_seed(5),
        observation_config={"mode": "none"},
    )
    torch.testing.assert_close(prediction, with_targets)
