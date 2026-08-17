"""GPU integration checks for resident and compact asynchronous batch paths."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from phycoflow_reconstruction.data.h5_dataset import H5FieldDataset
from phycoflow_reconstruction.data.training_batches import (
    AsyncCompactH5BatchSource,
    ResidentH5BatchSource,
    build_training_batch_source,
    dataset_field_bytes,
)
from phycoflow_reconstruction.training.base_training import run_base_training
from phycoflow_reconstruction.training.common import iter_unique_batch_indices


def _require_gpu_one() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("this integration check requires physical GPU 1")
    return torch.device("cuda:1")


def _write_fixture(path: Path) -> None:
    trajectories, times, rows, columns, channels = 3, 5, 8, 4, 2
    values = np.arange(
        trajectories * times * rows * columns * channels, dtype=np.float32
    ).reshape(trajectories, times, rows, columns, 1, channels)
    yy, xx = np.meshgrid(
        np.arange(rows, dtype=np.float32),
        np.arange(columns, dtype=np.float32),
        indexing="ij",
    )
    coordinates = np.stack((xx, yy, np.zeros_like(xx)), axis=-1)[..., None, :]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("fields", data=values, compression="gzip")
        handle.create_dataset("coordinates", data=coordinates)
        handle.create_dataset("time", data=np.arange(times, dtype=np.float32))
        handle.create_dataset("conditions", data=np.zeros((trajectories, 1), dtype=np.float32))
        handle.create_dataset(
            "trajectory_id",
            data=np.asarray(["train", "validation", "test"], dtype=h5py.string_dtype()),
        )
        splits = handle.create_group("splits")
        splits.create_dataset("train", data=np.asarray([0], dtype=np.int64))
        splits.create_dataset("validation", data=np.asarray([1], dtype=np.int64))
        splits.create_dataset("test", data=np.asarray([2], dtype=np.int64))
        statistics = handle.create_group("statistics")
        statistics.create_dataset("train_mean", data=np.zeros(channels, dtype=np.float32))
        statistics.create_dataset("train_std", data=np.ones(channels, dtype=np.float32))
        handle.attrs["field_names"] = '["u", "v"]'
        handle.attrs["field_units"] = '["1", "1"]'
        handle.attrs["grid_shape"] = f"[{rows}, {columns}]"


def _config(strategy: str, *, workers: int = 2) -> dict:
    return {
        "observations": {
            "protocol": "random_uniform",
            "fields": {"u": {"count": 4}, "v": {"count": 4}},
            "shared_locations": True,
            "seed": 19,
        },
        "runtime": {
            "data_strategy": strategy,
            "vram_dataset_threshold_gb": 20,
            "num_workers": workers,
        },
    }


def test_dataset_reuses_static_coordinates_and_drops_full_query_indices(tmp_path):
    path = tmp_path / "compact.h5"
    _write_fixture(path)
    dataset = H5FieldDataset(path, split="train")

    first = dataset[0]
    second = dataset[1]
    assert first.coordinates.data_ptr() == second.coordinates.data_ptr()
    assert "query_indices" not in first.metadata
    assert dataset_field_bytes(dataset) == 3 * 5 * 8 * 4 * 2 * 4


def test_resident_source_samples_and_gathers_on_physical_gpu_one(tmp_path):
    device = _require_gpu_one()
    path = tmp_path / "resident.h5"
    _write_fixture(path)
    dataset = H5FieldDataset(path, split="train")
    batches = iter_unique_batch_indices(
        len(dataset), 2, 3, generator=torch.Generator().manual_seed(7)
    )
    source = build_training_batch_source(
        dataset,
        batches,
        _config("auto"),
        query_points=8,
        device=device,
        start_step=0,
    )
    assert isinstance(source, ResidentH5BatchSource)
    assert source.fields.device == device
    assert source.coordinates.device == device
    assert {"fields", "coordinates", "conditions", "times"} <= set(
        dict(source.named_buffers())
    )
    assert not source.state_dict()

    first, second = list(source)
    assert first.obs_values.device == device
    assert first.target_fields.device == device
    assert first.obs_coords.shape == (3, 8, 2)
    assert first.query_coords.shape == (3, 8, 2)
    assert not torch.equal(first.obs_indices, second.obs_indices)
    assert not torch.equal(
        first.metadata["query_indices"], second.metadata["query_indices"]
    )
    source.close()

    complete_source = build_training_batch_source(
        dataset,
        [[0, 1, 2]],
        _config("auto"),
        query_points=None,
        device=device,
        start_step=2,
    )
    complete = next(iter(complete_source))
    assert complete.query_coords.untyped_storage().nbytes() == 32 * 2 * 4
    assert complete.metadata["query_indices"].untyped_storage().nbytes() == 32 * 8
    assert complete.query_valid_mask.untyped_storage().nbytes() == 32
    complete_source.close()


def test_compact_async_source_pins_then_transfers_to_physical_gpu_one(tmp_path):
    device = _require_gpu_one()
    path = tmp_path / "async.h5"
    _write_fixture(path)
    dataset = H5FieldDataset(path, split="train")
    batches = iter_unique_batch_indices(
        len(dataset), 3, 3, generator=torch.Generator().manual_seed(11)
    )
    source = build_training_batch_source(
        dataset,
        batches,
        _config("async_cpu", workers=2),
        query_points=8,
        device=device,
        start_step=3,
    )
    assert isinstance(source, AsyncCompactH5BatchSource)
    received = list(source)
    assert len(received) == 3
    assert all(batch.obs_values.device == device for batch in received)
    assert all(batch.target_fields.shape == (3, 8, 2) for batch in received)
    assert all(batch.metadata["sample_context"]["time"].shape == (3,) for batch in received)
    assert not torch.equal(received[0].obs_indices, received[1].obs_indices)
    source.close()


def test_base_training_uses_resident_pipeline_on_physical_gpu_one(tmp_path):
    _require_gpu_one()
    path = tmp_path / "base.h5"
    _write_fixture(path)
    config = {
        "stage": "base_training",
        "case": "fixture",
        "dataset": {
            "path": str(path),
            "split": "train",
            "field_names": ["u", "v"],
            "field_units": ["1", "1"],
            "normalization": "mean_std",
        },
        "model": {
            "name": "mlp_rbf",
            "hidden_dim": 16,
            "fourier_bands": 2,
            "query_points": 8,
        },
        "observations": _config("auto")["observations"],
        "optimization": {
            "epochs": 1,
            "batch_size": 3,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
        },
        "runtime": {
            **_config("auto")["runtime"],
            "seed": 31,
            "device": "cuda:1",
            "deterministic": True,
            "progress": False,
        },
        "evaluation": {
            "generation_steps": 1,
            "preview": {
                "enabled": True,
                "every_epochs": 1,
                "split": "validation",
                "sample_index": 0,
                "query_points": None,
                "generation_steps": 1,
                "seed": 2027,
                "keep_history": False,
            },
        },
        "checkpointing": {
            "enabled": True,
            "every_epochs": 1,
            "save_epoch_one": True,
        },
        "output": {"experiment_name": "resident_gpu_one"},
    }
    run_dir = run_base_training(config, case_dir=tmp_path / "case")
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    history = (run_dir / "metrics" / "history.jsonl").read_text().splitlines()
    assert manifest["training_data_strategy"] == "vram_resident"
    assert len(history) == 2
    preview = run_dir / "evaluation" / "training_preview"
    assert (run_dir / "checkpoints" / "last.pt").is_file()
    assert (run_dir / "checkpoints" / "latest.pt").is_symlink()
    assert (run_dir / "checkpoints" / "best.pt").is_file()
    checkpoint_report = json.loads(
        (run_dir / "evaluation" / "latest_checkpoint.json").read_text()
    )
    assert checkpoint_report["global_step"] == 2
    assert checkpoint_report["latest"] == "checkpoints/latest.pt"
    assert (preview / "latest_reconstruction.png").is_file()
    assert (preview / "latest_reconstruction.svg").is_file()
    assert (preview / "latest_reconstruction.pdf").is_file()
    assert (preview / "latest_reconstruction.npz").is_file()
    assert (preview / "figure_contract.md").is_file()
    preview_metrics = json.loads((preview / "latest_metrics.json").read_text())
    assert preview_metrics["checkpoint"] == "checkpoints/last.pt"
    assert preview_metrics["training_epoch"] == 1.0
