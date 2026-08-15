"""Focused checks for Kolmogorov configs and its multi-field sensor protocols."""

from pathlib import Path

import torch

from phycoflow_reconstruction.config import load_config, validate_config
from phycoflow_reconstruction.contracts import FieldSample
from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol, build_observation_batch

PROJECT = Path(__file__).resolve().parents[1]
CASE = PROJECT / "Cases" / "kolmogorov"


def _grid_sample() -> FieldSample:
    y, x = torch.meshgrid(torch.arange(256), torch.arange(256), indexing="ij")
    coordinates = torch.stack((x, y), dim=-1).reshape(-1, 2).float() / 255.0
    values = torch.stack(
        (
            torch.sin(2.0 * torch.pi * coordinates[:, 0]),
            torch.cos(2.0 * torch.pi * coordinates[:, 1]),
            torch.zeros(coordinates.shape[0]),
        ),
        dim=-1,
    )
    return FieldSample(
        values=values,
        coordinates=coordinates,
        coordinates_raw=coordinates,
        time=torch.tensor(0.0),
        trajectory_id="fixture",
        time_index=0,
        conditions=torch.tensor([40.0, 1.0, 4.0]),
        field_names=("u", "v", "p"),
        logical_shape=(256, 256),
    )


def test_all_kolmogorov_base_configs_resolve_and_validate():
    configs = sorted((CASE / "configs" / "base").glob("*.yaml"))
    assert len(configs) == 11
    for path in configs:
        if path.name == "plain_defaults.yaml":
            continue
        overrides = (
            ["model.stage1_checkpoint=/tmp/immutable-stage1.pt"] if "stage2" in path.name else []
        )
        validate_config(load_config(path, overrides))


def test_velocity_protocols_preserve_shared_and_structured_locations():
    sample = _grid_sample()
    shared = build_observation_batch(
        [sample],
        SensorProtocol(field_counts={"u": 512, "v": 512}, shared_locations=True, seed=42),
    )
    assert torch.equal(shared.obs_indices[0, :512], shared.obs_indices[0, 512:])

    structured = build_observation_batch(
        [sample],
        SensorProtocol(
            name="structured_stride",
            field_counts={"u": 1024, "v": 1024},
            spatial_downsample_ratio=8,
            phase=0,
        ),
    )
    assert structured.obs_valid_mask.sum().item() == 2048
    assert torch.equal(structured.obs_indices[0, :1024], structured.obs_indices[0, 1024:])
    assert structured.obs_indices[0, :4].tolist() == [0, 8, 16, 24]
