"""CoastalBench preparation must preserve holdouts and exclude them from scaling."""

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from phycoflow_reconstruction.data.factory import open_field_dataset

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "coastal_prepare", PROJECT / "cases/coastal_bench/prepare.py"
)
coastal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coastal)


def _fixture(path):
    values = np.arange(3 * 2 * 8 * 4, dtype=np.float32).reshape(3, 2, 8, 1, 1, 4)
    values[1:] += 10000  # Detect even small holdout leakage into training statistics.
    with h5py.File(path, "w") as handle:
        handle.create_dataset("fields", data=values)
        coordinates = np.zeros((8, 1, 1, 3), dtype=np.float32)
        coordinates[:, 0, 0, 0] = np.arange(8)
        handle.create_dataset("coordinates", data=coordinates)
        handle.create_dataset("time", data=[0.0, 30.0])
        handle.create_dataset("conditions", data=np.zeros((3, 1)))
        handle.create_dataset("trajectory_id", data=["a", "b", "c"], dtype=h5py.string_dtype())
        for index, split in enumerate(("train", "validation", "test")):
            handle.create_dataset(f"splits/{split}", data=[index])
        handle.attrs["field_names"] = json.dumps(coastal.FIELDS)
    return values


def test_coastal_statistics_exclude_holdouts_and_load_for_each_split(tmp_path):
    source = tmp_path / "coastal.h5"
    values = _fixture(source)
    before = source.read_bytes()
    output = tmp_path / "artifacts"
    audit = coastal.prepare(source, output)
    stats = json.loads((output / "train_mean_std.json").read_text())
    np.testing.assert_allclose(stats["offset"], values[0].reshape(-1, 4).mean(0))
    np.testing.assert_allclose(stats["scale"], values[0].reshape(-1, 4).std(0), rtol=1e-6)
    assert audit["snapshot_counts"] == {"train": 2, "validation": 2, "test": 2}
    assert source.read_bytes() == before
    config = {
        "path": str(source),
        "field_names": coastal.FIELDS,
        "normalization": "mean_std",
        "normalization_stats_path": str(output / "train_mean_std.json"),
    }
    for split, expected_id in (("train", "a"), ("validation", "b"), ("test", "c")):
        dataset = open_field_dataset(config, split=split)
        assert len(dataset) == 2
        assert dataset[0].trajectory_id == expected_id
        dataset.close()
    assert coastal.prepare(source, output)["trajectory_indices"] == audit["trajectory_indices"]


def test_coastal_rejects_overlapping_splits_before_writing_statistics(tmp_path):
    source = tmp_path / "coastal.h5"
    _fixture(source)
    with h5py.File(source, "r+") as handle:
        handle["splits/test"][:] = [0]
    with pytest.raises(ValueError, match="overlap"):
        coastal.prepare(source, tmp_path / "artifacts")
    assert not (tmp_path / "artifacts/train_mean_std.json").exists()


def test_ocean_point_preview_preserves_both_spatial_axes(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from phycoflow_reconstruction.training.preview import _plot_preview

    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    captured = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *args, **kwargs: captured.append(self))
    _plot_preview(
        tmp_path / "ocean",
        prediction=np.arange(4.0)[:, None],
        target=np.arange(4.0)[:, None],
        query_coords=coords,
        obs_coords=coords[:1],
        obs_values=np.array([0.0]),
        obs_fields=np.array([0]),
        obs_valid=np.array([True]),
        field_names=("temp",),
        logical_shape=(4,),
        epoch=1,
    )
    assert captured
    np.testing.assert_allclose(captured[0].axes[0].collections[0].get_offsets(), coords)
    assert not captured[0].axes[0].lines


def test_long_training_segments_resume_one_run_with_frozen_config(tmp_path, monkeypatch):
    import yaml

    from phycoflow_reconstruction.config import load_config

    module_spec = importlib.util.spec_from_file_location(
        "coastal_segments", PROJECT / "cases/coastal_bench/train_segment.py"
    )
    segments = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(segments)
    config_path = tmp_path / "config.yaml"
    config = load_config(PROJECT / "cases/coastal_bench/configs/base/gl_rbf_cq.yaml")
    config_path.write_text(yaml.safe_dump(config))

    class Dataset:
        def __len__(self):
            return 27744

        def close(self):
            pass

    monkeypatch.setattr(segments, "open_field_dataset", lambda _: Dataset())
    run = tmp_path / "run"
    calls = []

    def train(config, *, case_dir, max_steps, resume):
        previous = (
            0 if resume is None else json.loads((resume / "status.json").read_text())["global_step"]
        )
        run.mkdir(exist_ok=True)
        (run / "status.json").write_text(json.dumps({"global_step": previous + max_steps}))
        calls.append((max_steps, resume, config))
        return run

    monkeypatch.setattr(segments, "run_base_training", train)
    monkeypatch.setattr(
        segments,
        "load_project_checkpoint",
        lambda path: json.loads((path.parents[1] / "status.json").read_text()),
    )
    state = tmp_path / "state"
    segments.train_segment(state, config_path=config_path)
    assert json.loads((state / "progress.json").read_text())["completed_epochs"] == 2500
    config["optimization"]["epochs"] = 1
    config_path.write_text(yaml.safe_dump(config))  # Later edits cannot change a running chain.
    segments.train_segment(state, config_path=config_path)
    progress = json.loads((state / "progress.json").read_text())
    assert progress["global_step"] == 4_335_000
    assert progress["complete"]
    assert [(steps, resume) for steps, resume, _ in calls] == [(2_167_500, None), (2_167_500, run)]
    assert calls[0][2] == calls[1][2]
    segments.train_segment(state, config_path=config_path)
    assert len(calls) == 2  # Completed chains are idempotent.
