"""Observation tests prove deterministic manifests and independent KS strides."""

import pytest
import torch

from phycoflow_reconstruction.contracts import DataSpec, FieldSample
from phycoflow_reconstruction.data.manifest import (
    SensorManifest,
    build_batch_from_manifest,
    manifest_from_batch,
)
from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol, build_observation_batch
from phycoflow_reconstruction.models import build_model


def _sample(space_time: bool = False) -> FieldSample:
    count = 24 if space_time else 12
    coords = torch.stack([torch.linspace(0, 1, count), torch.zeros(count)], dim=-1)
    return FieldSample(
        values=torch.arange(count * 2, dtype=torch.float32).reshape(count, 2),
        coordinates=coords,
        coordinates_raw=coords,
        time=torch.arange(4) if space_time else torch.tensor(0.0),
        trajectory_id="t0",
        time_index=None if space_time else 0,
        conditions=torch.empty(0),
        field_names=("a", "b"),
        logical_shape=(4, 6) if space_time else (12,),
        reconstruction_unit="space_time_trajectory" if space_time else "snapshot",
    )


def test_random_protocol_is_adapter_independent(tmp_path):
    protocol = SensorProtocol(field_counts={"a": 4, "b": 3}, seed=7)
    first = build_observation_batch([_sample()], protocol)
    second = build_observation_batch([_sample()], protocol)
    assert torch.equal(first.obs_indices, second.obs_indices)
    dataset_path = tmp_path / "data.h5"
    dataset_path.write_bytes(b"fixture")
    manifest = manifest_from_batch(first, dataset_path, "test")
    assert manifest.indices[first.sample_ids[0]]
    manifest_path = tmp_path / "sensors.json"
    manifest.save(manifest_path)

    # Loading the same persisted evidence before two different adapters must
    # not permit either model family to resample or reorder sensors.
    loaded = SensorManifest.load(manifest_path)
    point_batch = build_batch_from_manifest([_sample()], loaded, dataset_path)
    grid_batch = build_batch_from_manifest([_sample()], loaded, dataset_path)
    spec = DataSpec(("a", "b"), ("1", "1"), 2, (12,))
    build_model({"name": "coordinate_mlp", "hidden_dim": 8}, spec)
    build_model({"name": "deeponet", "width": 8, "basis_dim": 4}, spec)
    assert point_batch.obs_indices.numpy().tobytes() == grid_batch.obs_indices.numpy().tobytes()


def test_space_time_ratios_are_independent():
    protocol = SensorProtocol(
        name="uniform_spacetime_stride",
        field_counts={"a": 1},
        temporal_downsample_ratio=2,
        spatial_downsample_ratio=3,
    )
    batch = build_observation_batch([_sample(space_time=True)], protocol)
    assert batch.obs_valid_mask.sum().item() == 4
    assert batch.query_coords.shape[1] == 24


def test_shared_fields_reuse_locations_and_grid_stride_is_two_dimensional():
    sample = _sample()
    shared = build_observation_batch(
        [sample],
        SensorProtocol(field_counts={"a": 4, "b": 4}, shared_locations=True, seed=11),
    )
    assert torch.equal(shared.obs_indices[0, :4], shared.obs_indices[0, 4:8])

    grid_sample = _sample(space_time=True)
    grid_sample.reconstruction_unit = "snapshot"
    grid_sample.logical_shape = (4, 6)
    structured = build_observation_batch(
        [grid_sample],
        SensorProtocol(
            name="structured_stride",
            field_counts={"a": 6},
            spatial_downsample_ratio=2,
        ),
    )
    assert structured.obs_indices[0, :6].tolist() == [0, 2, 4, 12, 14, 16]


def test_variable_sensor_counts_are_seeded_and_bounded():
    protocol = SensorProtocol(field_count_ranges={"a": (2, 5), "b": (3, 6)}, seed=19)
    first = build_observation_batch([_sample()], protocol)
    second = build_observation_batch([_sample()], protocol)
    assert torch.equal(first.obs_indices, second.obs_indices)
    count = int(first.obs_valid_mask.sum())
    assert 5 <= count <= 11


def test_manifest_indices_reject_duplicates_and_out_of_range_points():
    sample = _sample()
    protocol = SensorProtocol(field_counts={"a": 2})
    with pytest.raises(ValueError, match="duplicate"):
        build_observation_batch([sample], protocol, manifest_indices={"t0:0": [[1, 0], [1, 0]]})
    with pytest.raises(ValueError, match="out of range"):
        build_observation_batch([sample], protocol, manifest_indices={"t0:0": [[99, 0]]})
