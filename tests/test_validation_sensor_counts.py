"""Fixed validation sensing must not alter ranged training or legacy draws."""

import copy
from pathlib import Path

import pytest
import torch

from phycoflow_reconstruction.config import load_config, validate_config
from phycoflow_reconstruction.contracts import FieldSample
from phycoflow_reconstruction.data.sensor_protocols import build_observation_batch
from phycoflow_reconstruction.training.common import sensor_protocol_from_config
from phycoflow_reconstruction.training.preview import _preview_sensor_protocol

CONFIG = Path(__file__).resolve().parents[1] / (
    "cases/kolmogorov_shu/configs/base/"
    "gl_rbf_enh_w256_c128_fixedsigma005_k64_b64_val1pct.yaml"
)


def sample():
    axis = torch.linspace(0, 1, 256)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return FieldSample(
        values=(coords[:, :1] + coords[:, 1:]), coordinates=coords,
        coordinates_raw=coords.clone(), time=torch.tensor(0.0),
        trajectory_id="validation", time_index=0, conditions=torch.empty(0),
        field_names=("vorticity",), logical_shape=(256, 256),
    )


def test_fixed_validation_count_is_repeatable_and_leaves_training_range_intact():
    cfg = load_config(CONFIG)
    before = copy.deepcopy(cfg)
    validate_config(cfg)
    protocol = _preview_sensor_protocol(cfg)
    first = build_observation_batch([sample()], protocol, query_points=32)
    second = build_observation_batch([sample()], _preview_sensor_protocol(cfg), query_points=32)
    assert first.obs_valid_mask.sum().item() == 655
    assert torch.equal(first.obs_indices, second.obs_indices)
    assert torch.equal(first.obs_values, second.obs_values)
    assert torch.equal(first.query_coords, second.query_coords)
    assert sensor_protocol_from_config(cfg).field_count_ranges == {"vorticity": (65, 6554)}
    assert cfg == before


def test_absent_validation_override_preserves_legacy_sensor_draw():
    cfg = load_config(CONFIG)
    cfg["evaluation"]["preview"].pop("sensor_counts")
    legacy = sensor_protocol_from_config(cfg, seed_offset=2027)
    actual = _preview_sensor_protocol(cfg)
    assert actual == legacy
    expected_batch = build_observation_batch([sample()], legacy, query_points=32)
    actual_batch = build_observation_batch([sample()], actual, query_points=32)
    assert torch.equal(actual_batch.obs_indices, expected_batch.obs_indices)
    assert actual_batch.obs_valid_mask.sum().item() == 3165


@pytest.mark.parametrize("counts", [
    {}, {"vorticity": 0}, {"vorticity": 655.5}, {"vorticity": True}, {"other": 655},
])
def test_invalid_validation_count_fails_config_validation(counts):
    cfg = load_config(CONFIG)
    cfg["evaluation"]["preview"]["sensor_counts"] = counts
    with pytest.raises(ValueError, match="sensor_counts"):
        validate_config(cfg)
