"""Checks for streaming multi-sample reconstruction evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from phycoflow_reconstruction.cli import _evaluation_sample_limit
from phycoflow_reconstruction.contracts import DataSpec, FieldSample
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.evaluation import reconstruction_set
from phycoflow_reconstruction.evaluation.checkpoint import EvaluationRuntime


def test_sample_selection_is_evenly_spaced_and_supports_full_split():
    selected = reconstruction_set._select_sample_indices(1_000, 200)

    assert selected.shape == (200,)
    assert selected[0] == 0
    assert selected[-1] == 999
    assert np.unique(selected).size == 200
    np.testing.assert_array_equal(
        reconstruction_set._select_sample_indices(4, None),
        np.arange(4),
    )
    assert _evaluation_sample_limit("500") == 500
    assert _evaluation_sample_limit("all") is None


def test_distribution_renderer_overlays_violin_and_scatter(tmp_path, monkeypatch):
    payload = tmp_path / "relative_l2.npz"
    output = tmp_path / "relative_l2_violin.png"
    np.savez_compressed(
        payload,
        per_field_relative_l2_physical=np.asarray(
            [[0.1, 0.5], [0.2, 0.4], [0.3, 0.6]], dtype=np.float64
        ),
        field_names=np.asarray(["u", "v"]),
        sample_ids=np.asarray(["a", "b", "c"]),
        split=np.asarray("test"),
    )
    calls = {"violin": 0, "scatter": 0}
    scatter_styles: list[dict[str, object]] = []
    import matplotlib.pyplot as plt

    original_violin = plt.Axes.violinplot
    original_scatter = plt.Axes.scatter

    def capture_violin(self, *args, **kwargs):
        calls["violin"] += 1
        return original_violin(self, *args, **kwargs)

    def capture_scatter(self, *args, **kwargs):
        calls["scatter"] += 1
        scatter_styles.append(kwargs)
        return original_scatter(self, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "violinplot", capture_violin)
    monkeypatch.setattr(plt.Axes, "scatter", capture_scatter)

    result = reconstruction_set.render_reconstruction_set_distribution(payload, output)

    assert result == output
    assert output.is_file()
    assert calls == {"violin": 1, "scatter": 4}
    assert scatter_styles[0]["s"] == 11
    assert scatter_styles[0]["alpha"] == 0.48
    assert scatter_styles[1]["marker"] == "D"
    assert scatter_styles[1]["s"] == 31
    assert reconstruction_set._publication_field_label("CH4") == r"CH$_{4}$"
    assert reconstruction_set._publication_field_label("U_1") == r"$U_{1}$"
    assert reconstruction_set._publication_field_label("concentration") == "concentration"
    assert reconstruction_set._publication_field_label("u_x") == "u_x"


@pytest.mark.parametrize(
    "field_names",
    [("u",), ("u_x", "u_y", "concentration")],
)
def test_distribution_renderer_supports_general_field_counts(tmp_path, field_names):
    payload = tmp_path / f"fields_{len(field_names)}.npz"
    output = tmp_path / f"fields_{len(field_names)}.png"
    errors = np.linspace(0.05, 0.5, 4 * len(field_names)).reshape(4, len(field_names))
    np.savez_compressed(
        payload,
        per_field_relative_l2_physical=errors,
        field_names=np.asarray(field_names),
        sample_ids=np.asarray(["a", "b", "c", "d"]),
        split=np.asarray("validation"),
    )

    reconstruction_set.render_reconstruction_set_distribution(payload, output)

    assert output.is_file()


class _FixtureDataset:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.field_names = ("u", "v")
        self.normalizer = FieldNormalizer.identity(2)
        self.data_spec = DataSpec(
            field_names=self.field_names,
            field_units=("unknown", "unknown"),
            coordinate_dim=2,
            logical_shape=(2, 2),
            reconstruction_unit="space_time_trajectory",
            mesh_type="structured",
        )
        self.closed = False

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> FieldSample:
        coordinates = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        )
        values = torch.tensor(
            [
                [1.0 + index, 2.0],
                [2.0 + index, 3.0],
                [3.0 + index, 4.0],
                [4.0 + index, 5.0],
            ]
        )
        return FieldSample(
            values=values,
            coordinates=coordinates,
            coordinates_raw=coordinates,
            time=torch.tensor([0.0, 1.0]),
            trajectory_id="fixture",
            time_index=index,
            conditions=torch.zeros(1),
            field_names=self.field_names,
            logical_shape=(2, 2),
            reconstruction_unit="space_time_trajectory",
        )

    def close(self) -> None:
        self.closed = True


class _FixtureModel(torch.nn.Module):
    def reconstruct(self, batch, *, steps, generator):
        del steps, generator
        return SimpleNamespace(prediction=batch.target_fields * 0.9, samples=None)


def test_set_evaluation_streams_selected_samples_and_writes_outputs(tmp_path, monkeypatch):
    case_dir = tmp_path / "case"
    run_dir = case_dir / "runs" / "experiment" / "run-id"
    checkpoint = run_dir / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (run_dir / "resolved_config.yaml").write_text("fixture: true\n", encoding="utf-8")
    dataset_path = tmp_path / "dataset.bin"
    dataset_path.write_bytes(b"dataset")
    dataset = _FixtureDataset(dataset_path)
    runtime = EvaluationRuntime(
        config={
            "observations": {
                "protocol": "random_uniform",
                "seed": 42,
                "fields": {"u": {"count": 1}},
            },
            "runtime": {"seed": 42},
        },
        device=torch.device("cpu"),
        dataset=dataset,
        model=_FixtureModel(),
        checkpoint_path=checkpoint,
        generation_steps=2,
        seed=2027,
    )
    monkeypatch.setattr(reconstruction_set, "load_evaluation_runtime", lambda *a, **k: runtime)
    monkeypatch.setattr(
        reconstruction_set,
        "warn_if_cuda_memory_tight",
        lambda *args, **kwargs: False,
    )

    figure = reconstruction_set.evaluate_reconstruction_set(
        run_dir,
        case_dir=case_dir,
        split="test",
        max_samples=2,
    )

    output_dir = run_dir / "evaluation" / "reconstruction_set_test_best"
    assert figure == output_dir / "relative_l2_violin.png"
    assert figure.is_file()
    assert (output_dir / "relative_l2.csv").is_file()
    assert (output_dir / "relative_l2.npz").is_file()
    assert (output_dir / "sensor_manifest.jsonl").is_file()
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["sample_count"] == 2
    assert report["available_sample_count"] == 3
    assert report["selection"]["split_relative_indices"] == [0, 2]
    assert report["selection"]["policy"] == "evenly_spaced_split_subset"
    assert report["per_field_statistics"]["u"]["count"] == 2
    assert dataset.closed
