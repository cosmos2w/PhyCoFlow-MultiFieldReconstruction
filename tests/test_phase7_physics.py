"""Verified case physics must be differentiable on its declared full-grid layout."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from phycoflow_reconstruction.contracts import DataSpec, ObservationBatch
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def _physics_class():
    path = Path(__file__).resolve().parents[1] / "Cases" / "brusselator" / "physics.py"
    spec = importlib.util.spec_from_file_location("phase7_brusselator_physics", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.BrusselatorPhysics


def test_brusselator_periodic_residual_has_finite_parameter_gradient():
    data_spec = DataSpec(
        field_names=("u", "v"),
        field_units=("1", "1"),
        coordinate_dim=2,
        logical_shape=(4, 4),
        mesh_type="structured_periodic",
    )
    batch = ObservationBatch(
        obs_coords=torch.zeros(1, 2, 2),
        obs_values=torch.ones(1, 2, 1),
        obs_field_ids=torch.zeros(1, 2, dtype=torch.long),
        obs_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        query_coords=torch.zeros(1, 16, 2),
        query_valid_mask=torch.ones(1, 16, dtype=torch.bool),
        target_fields=torch.ones(1, 16, 2),
        sample_ids=("fixture:1",),
        logical_shapes=((4, 4),),
        metadata={
            "sample_context": {
                "conditions": torch.tensor([[1.0, 3.0, 0.1, 0.1]]),
                "physics": {"temporal_derivative": torch.zeros(1, 16, 2)},
            }
        },
    )
    prediction = torch.full((1, 16, 2), 0.5, requires_grad=True)
    provider = _physics_class()(
        {"temporal_derivative_source": "paired_finite_difference"},
        data_spec,
        FieldNormalizer.identity(2),
    )
    loss = provider.loss(prediction, batch)
    loss.total.backward()
    assert torch.isfinite(loss.total)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0
