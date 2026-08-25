"""Dataset tests cover trajectory/frame splits and reversible KS layout."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from phycoflow_reconstruction.data.factory import open_field_dataset
from phycoflow_reconstruction.data.h5_dataset import H5FieldDataset
from phycoflow_reconstruction.data.manifest import dataset_fingerprint
from phycoflow_reconstruction.data.pt_dataset import PTFieldDataset
from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol, build_observation_batch
from phycoflow_reconstruction.data.splits import chronological_frame_indices
from phycoflow_reconstruction.data.validation import validate_h5_dataset, validate_pt_dataset


def _first_sample(batch):
    return batch[0]


def _write_fixture(path: Path, trajectories: int = 3, times: int = 5) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "fields",
            data=np.arange(trajectories * times * 4 * 2, dtype=np.float32).reshape(
                trajectories, times, 4, 1, 1, 2
            ),
        )
        handle.create_dataset(
            "coordinates",
            data=np.stack([np.arange(4), np.zeros(4), np.zeros(4)], axis=-1).reshape(4, 1, 1, 3),
        )
        handle.create_dataset("time", data=np.arange(times, dtype=np.float64))
        handle.create_dataset("conditions", data=np.zeros((trajectories, 0), dtype=np.float32))
        handle.create_dataset(
            "trajectory_id",
            data=np.asarray([f"t{i}" for i in range(trajectories)], dtype=h5py.string_dtype()),
        )
        splits = handle.create_group("splits")
        splits.create_dataset("train", data=np.asarray([0], dtype=np.int64))
        splits.create_dataset("validation", data=np.asarray([1], dtype=np.int64))
        splits.create_dataset("test", data=np.asarray([2], dtype=np.int64))
        stats = handle.create_group("statistics")
        stats.create_dataset("train_mean", data=np.zeros(2, dtype=np.float32))
        stats.create_dataset("train_std", data=np.ones(2, dtype=np.float32))
        auxiliary = handle.create_group("auxiliary")
        auxiliary.create_dataset(
            "pressure",
            data=np.zeros((trajectories, times, 4, 1, 1, 1), dtype=np.float32),
        )
        handle.attrs["field_names"] = '["a", "b"]'
        handle.attrs["field_units"] = '["1", "1"]'
        handle.attrs["grid_shape"] = "[4]"
        handle.attrs["schema_version"] = "1.0"


def _write_normalization_artifact(path: Path, dataset_path: Path) -> None:
    payload = {
        "version": "1",
        "method": "mean_std",
        "field_names": ["a", "b"],
        "offset": [2.0, 4.0],
        "scale": [2.0, 4.0],
        "dataset_fingerprint": dataset_fingerprint(dataset_path),
        "statistics_split": "train",
        "split_strategy": "stored_trajectory",
        "sample_value_count_per_field": 20,
    }
    payload["artifact_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_trajectory_split_and_snapshot(tmp_path):
    path = tmp_path / "fixture.h5"
    _write_fixture(path)
    dataset = H5FieldDataset(path, split="validation")
    assert len(dataset) == 5
    assert dataset[0].trajectory_id == "t1"
    assert dataset[0].metadata["auxiliary"]["pressure"].shape == (4, 1, 1, 1)
    assert validate_h5_dataset(path)["valid"]


def test_space_time_layout(tmp_path):
    path = tmp_path / "fixture.h5"
    _write_fixture(path)
    dataset = H5FieldDataset(path, split="train", reconstruction_unit="space_time_trajectory")
    sample = dataset[0]
    assert sample.values.shape == (20, 2)
    assert sample.coordinates.shape == (20, 2)
    assert sample.logical_shape == (5, 4)


def test_explicit_coordinate_dimension_preserves_constant_legacy_axis(tmp_path):
    path = tmp_path / "fixture.h5"
    _write_fixture(path)
    dataset = H5FieldDataset(path, split="train", coordinate_dim=3)
    assert dataset.data_spec.coordinate_dim == 3
    assert dataset[0].coordinates.shape[1] == 3


def test_h5_loader_reopens_safely_in_workers(tmp_path):
    path = tmp_path / "fixture.h5"
    _write_fixture(path)
    dataset = H5FieldDataset(path, split="train")
    loader = DataLoader(dataset, batch_size=1, num_workers=2, collate_fn=_first_sample)
    assert next(iter(loader)).trajectory_id == "t0"


def test_external_training_normalizer_is_verified_and_reused_across_splits(tmp_path):
    dataset_path = tmp_path / "fixture.h5"
    artifact_path = tmp_path / "normalization.json"
    _write_fixture(dataset_path)
    _write_normalization_artifact(artifact_path, dataset_path)
    config = {
        "path": dataset_path,
        "field_names": ["a", "b"],
        "normalization": "mean_std",
        "normalization_stats_path": artifact_path,
    }

    train = open_field_dataset(config, split="train")
    validation = open_field_dataset(config, split="validation")

    assert torch.equal(train.normalizer.offset, torch.tensor([2.0, 4.0]))
    assert torch.equal(train.normalizer.scale, torch.tensor([2.0, 4.0]))
    assert train.normalizer.digest() == validation.normalizer.digest()
    values = torch.tensor([4.0, 12.0])
    assert torch.equal(train.normalizer.decode(train.normalizer.encode(values)), values)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("field_names", ["b", "a"], "field order"),
        ("statistics_split", "validation", "training split"),
        ("dataset_fingerprint", "wrong", "dataset fingerprint"),
    ],
)
def test_external_normalizer_rejects_provenance_mismatch(
    tmp_path, key, value, match
):
    dataset_path = tmp_path / "fixture.h5"
    artifact_path = tmp_path / "normalization.json"
    _write_fixture(dataset_path)
    _write_normalization_artifact(artifact_path, dataset_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload.pop("artifact_sha256")
    payload[key] = value
    payload["artifact_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        open_field_dataset(
            {
                "path": dataset_path,
                "field_names": ["a", "b"],
                "normalization": "mean_std",
                "normalization_stats_path": artifact_path,
            }
        )


def test_coordinate_reorder_and_physics_context_follow_query_indices(tmp_path):
    path = tmp_path / "permuted.h5"
    coordinates = np.asarray([[1, 1, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    fields = np.zeros((3, 3, 4, 1, 1, 2), dtype=np.float32)
    for time_index in range(3):
        fields[:, time_index, :, 0, 0, :] = time_index
    with h5py.File(path, "w") as handle:
        handle.create_dataset("fields", data=fields)
        handle.create_dataset("coordinates", data=coordinates.reshape(4, 1, 1, 3))
        handle.create_dataset("time", data=np.arange(3, dtype=np.float32))
        handle.create_dataset("conditions", data=np.ones((3, 4), dtype=np.float32))
        splits = handle.create_group("splits")
        splits.create_dataset("train", data=np.asarray([0]))
        splits.create_dataset("validation", data=np.asarray([1]))
        splits.create_dataset("test", data=np.asarray([2]))
        statistics = handle.create_group("statistics")
        statistics.create_dataset("train_mean", data=np.zeros(2))
        statistics.create_dataset("train_std", data=np.ones(2))
        handle.attrs["field_names"] = '["u", "v"]'
        handle.attrs["grid_shape"] = "[2, 2]"
    dataset = H5FieldDataset(
        path,
        split="train",
        grid_shape=(2, 2),
        coordinate_reorder="lexicographic_yx",
        include_temporal_derivative=True,
    )
    sample = dataset[1]
    assert sample.coordinates_raw[:, :2].tolist() == [[0, 0], [1, 0], [0, 1], [1, 1]]
    assert torch.equal(sample.metadata["physics"]["temporal_derivative"], torch.ones(4, 2))
    batch = build_observation_batch(
        [sample], SensorProtocol(field_counts={"u": 2}, seed=3), query_points=2
    ).to("cpu")
    context = batch.metadata["sample_context"]
    assert context["physics"]["temporal_derivative"].shape == (1, 2, 2)
    assert context["conditions"].shape == (1, 4)


def test_chronological_derivative_never_crosses_split_boundaries(tmp_path):
    path = tmp_path / "single_trajectory.h5"
    fields = np.zeros((1, 20, 2, 1, 1, 1), dtype=np.float32)
    fields[0, 15] = 0.0
    fields[0, 16] = 100.0
    fields[0, 17] = 102.0
    fields[0, 18] = 500.0
    with h5py.File(path, "w") as handle:
        handle.create_dataset("fields", data=fields)
        handle.create_dataset(
            "coordinates",
            data=np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32).reshape(2, 1, 1, 3),
        )
        handle.create_dataset("time", data=np.arange(20, dtype=np.float32))
        handle.create_dataset("conditions", data=np.empty((1, 0), dtype=np.float32))
        statistics = handle.create_group("statistics")
        statistics.create_dataset("train_mean", data=np.zeros(1))
        statistics.create_dataset("train_std", data=np.ones(1))
        handle.attrs["field_names"] = '["u"]'
        handle.attrs["grid_shape"] = "[2]"
    validation = H5FieldDataset(path, split="validation", include_temporal_derivative=True)
    assert [sample.time_index for sample in validation] == [16, 17]
    for sample in validation:
        derivative = sample.metadata["physics"]["temporal_derivative"]
        assert torch.equal(derivative, torch.full((2, 1), 2.0))


def test_ordered_frame_split_boundaries():
    assert chronological_frame_indices(10, "train").tolist() == list(range(8))
    assert chronological_frame_indices(10, "validation").tolist() == [8]
    assert chronological_frame_indices(10, "test").tolist() == [9]


def test_trusted_pt_uses_the_same_trajectory_contract(tmp_path):
    path = tmp_path / "fixture.pt"
    fields = torch.arange(3 * 5 * 4 * 2, dtype=torch.float32).reshape(3, 5, 4, 1, 1, 2)
    torch.save(
        {
            "fields": fields,
            "coordinates": torch.stack(
                (torch.arange(4), torch.zeros(4), torch.zeros(4)), dim=-1
            ).reshape(4, 1, 1, 3),
            "time": torch.arange(5),
            "conditions": torch.empty(3, 0),
            "field_names": ["a", "b"],
            "field_units": ["1", "1"],
            "grid_shape": [4],
            "trajectory_id": ["t0", "t1", "t2"],
            "splits": {
                "train": torch.tensor([0]),
                "validation": torch.tensor([1]),
                "test": torch.tensor([2]),
            },
            "statistics": {"train_mean": torch.zeros(2), "train_std": torch.ones(2)},
        },
        path,
    )
    assert validate_pt_dataset(path)["valid"]
    dataset = PTFieldDataset(path, split="validation")
    assert len(dataset) == 5
    assert dataset[0].trajectory_id == "t1"
    through_factory = open_field_dataset(
        {
            "path": path,
            "split": "validation",
            "normalization": "mean_std",
        }
    )
    assert through_factory.selection.strategy == "stored_trajectory"
    assert through_factory[0].trajectory_id == "t1"


def test_trusted_pt_rejects_overlapping_trajectory_splits(tmp_path):
    path = tmp_path / "overlap.pt"
    torch.save(
        {
            "fields": torch.zeros(3, 2, 2, 1, 1, 1),
            "coordinates": torch.zeros(2, 1, 1, 3),
            "time": torch.arange(2),
            "conditions": torch.empty(3, 0),
            "field_names": ["u"],
            "trajectory_id": ["a", "b", "c"],
            "splits": {
                "train": torch.tensor([0, 1]),
                "validation": torch.tensor([1]),
                "test": torch.tensor([2]),
            },
            "statistics": {"train_mean": torch.zeros(1), "train_std": torch.ones(1)},
        },
        path,
    )
    report = validate_pt_dataset(path)
    assert not report["valid"]
    assert any("overlap" in error for error in report["errors"])
    with pytest.raises(ValueError, match="duplicates or overlap"):
        PTFieldDataset(path)
