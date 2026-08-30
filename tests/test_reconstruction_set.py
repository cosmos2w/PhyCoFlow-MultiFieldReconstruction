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
from phycoflow_reconstruction.evaluation import coherence_set, reconstruction_set
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

    reconstruction_set.render_reconstruction_set_distribution(
        payload,
        output,
        scale="linear" if len(field_names) > 1 else "log",
    )

    assert output.is_file()


def test_coherence_publication_labels_format_compound_field_pairs():
    assert coherence_set._publication_label("CH4–U_1") == r"CH$_{4}$–$U_{1}$"


def test_jensen_shannon_divergence_is_bounded_and_calibrated():
    reference = np.asarray([[0.5, 0.5], [0.0, 0.0]])
    disjoint = np.asarray([[0.0, 0.0], [0.5, 0.5]])

    assert coherence_set._jensen_shannon_divergence_bits(reference, reference) == 0.0
    assert coherence_set._jensen_shannon_divergence_bits(reference, disjoint) == 1.0


class _FixtureDataset:
    def __init__(self, path: Path, sample_count: int = 4) -> None:
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
        self.sample_count = int(sample_count)

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> FieldSample:
        coordinates = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
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
    def __init__(self, scale: float = 0.88) -> None:
        super().__init__()
        self.scale = float(scale)

    def reconstruct(self, batch, *, steps, generator):
        del steps, generator
        target = batch.target_fields
        prediction = target * self.scale + 0.02 * target.square()
        prediction[..., 1] += 0.03 * target[..., 0]
        return SimpleNamespace(prediction=prediction, samples=None)


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
        coherence_families=["global_distribution"],
        extra_coherence_views=True,
    )

    output_dir = run_dir / "evaluation" / "reconstruction_set_test_best"
    assert figure == output_dir / "relative_l2_violin.png"
    assert figure.is_file()
    assert (output_dir / "relative_l2.csv").is_file()
    assert (output_dir / "relative_l2.npz").is_file()
    assert (output_dir / "sensor_manifest.jsonl").is_file()
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["sample_count"] == 2
    assert report["available_sample_count"] == 4
    assert report["selection"]["split_relative_indices"] == [0, 3]
    assert report["selection"]["policy"] == "evenly_spaced_split_subset"
    assert report["statistic_scale"] == "log"
    assert report["per_field_statistics"]["u"]["count"] == 2
    coherence_dir = output_dir / "coherence" / "global_distribution"
    assert (coherence_dir / "marginal_field_distributions.png").is_file()
    assert (coherence_dir / "pairwise_field_distributions.png").is_file()
    assert (coherence_dir / "joint_top_tail_distributions.png").is_file()
    assert (coherence_dir / "metrics.csv").is_file()
    assert (coherence_dir / "report.json").is_file()
    with np.load(coherence_dir / "metrics.npz", allow_pickle=False) as payload:
        assert payload["marginal_per_field"].shape == (2, 2)
        assert payload["pairwise_per_field_pair"].shape == (2, 1)
        assert payload["joint_top_tail"].shape == (2, 1)
        assert payload["family_total"].shape == (2, 1)
        assert payload["pair_labels"].tolist() == ["u–v"]
        np.testing.assert_allclose(
            payload["family_total"][:, 0],
            payload["weighted_component_totals"].sum(axis=1),
        )
    coherence_report = json.loads((coherence_dir / "report.json").read_text(encoding="utf-8"))
    assert coherence_report["target_use"] == "paired_supervised"
    assert coherence_report["sample_count"] == 2
    assert coherence_report["statistic_scale"] == "log"
    assert coherence_report["statistics"]["family_total"]["count"] == 2
    assert report["coherence"]["global_distribution"]["family"] == "global_distribution"
    extra_dir = coherence_dir / "global_distribution_extra"
    assert (extra_dir / "joint_pdf_u-v.png").is_file()
    assert (extra_dir / "joint_pdf_metrics.csv").is_file()
    assert (extra_dir / "joint_pdf_metrics.npz").is_file()
    extra_report = json.loads((extra_dir / "report.json").read_text(encoding="utf-8"))
    assert extra_report["sample_count"] == 2
    assert extra_report["pooled_point_count"] == 8
    assert 0.0 <= extra_report["pairs"][0]["jensen_shannon_divergence_bits"] <= 1.0
    assert dataset.closed


def test_set_evaluation_writes_training_aligned_cross_spectrum_statistics(tmp_path, monkeypatch):
    case_dir = tmp_path / "case"
    run_dir = case_dir / "runs" / "experiment" / "cross-run"
    checkpoint = run_dir / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (run_dir / "resolved_config.yaml").write_text("fixture: true\n", encoding="utf-8")
    dataset_path = tmp_path / "cross-dataset.bin"
    dataset_path.write_bytes(b"dataset")
    dataset = _FixtureDataset(dataset_path, sample_count=7)
    runtime = EvaluationRuntime(
        config={
            "observations": {
                "protocol": "random_uniform",
                "seed": 42,
                "fields": {"u": {"count": 1}},
            },
            "runtime": {"seed": 42},
            "coherence": {"compute_budget": {"batch_size": 3, "point_count": 4, "query_seed": 17}},
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

    reconstruction_set.evaluate_reconstruction_set(
        run_dir,
        case_dir=case_dir,
        split="test",
        max_samples=7,
        coherence_families=["cross_spectrum"],
    )

    output_dir = run_dir / "evaluation" / "reconstruction_set_test_best"
    cross_dir = output_dir / "coherence" / "cross_spectrum"
    assert (cross_dir / "self_spectrum_coherence.png").is_file()
    assert (cross_dir / "same_frequency_coherence.png").is_file()
    assert (cross_dir / "cross_frequency_coherence.png").is_file()
    assert not (cross_dir / "band_energy_coherence.png").exists()
    assert not (cross_dir / "same_frequency_pair_distributions.png").exists()
    assert (cross_dir / "metrics.csv").is_file()
    with np.load(cross_dir / "metrics.npz", allow_pickle=False) as payload:
        assert payload["self_spectrum_absolute_discrepancy"].shape == (2,)
        assert payload["self_spectrum_coherence_score"].shape == (2,)
        assert payload["same_frequency_absolute_discrepancy"].shape == (1,)
        assert payload["same_frequency_coherence_score"].shape == (1,)
        assert payload["cross_frequency_absolute_discrepancy"].shape == (1,)
        assert payload["cross_frequency_coherence_score"].shape == (1,)
        assert payload["self_spectrum_coherence_score_by_ensemble"].shape == (2, 2)
        assert payload["same_frequency_coherence_score_by_ensemble"].shape == (2, 1)
        assert payload["cross_frequency_coherence_score_by_ensemble"].shape == (2, 1)
        assert payload["sample_ids"].shape == (6,)
        assert payload["selected_sample_ids"].shape == (7,)
        assert payload["dropped_sample_ids"].shape == (1,)
        assert payload["ensemble_sample_ids"].shape == (2, 3)
        assert payload["aggregation"].item() == "training_aligned"
        assert payload["ensemble_size"].item() == 3
        assert payload["ensemble_count"].item() == 2
        np.testing.assert_allclose(
            payload["same_frequency_coherence_score"],
            payload["same_frequency_coherence_score_by_ensemble"].mean(axis=0),
        )
        np.testing.assert_allclose(
            payload["same_frequency_coherence_score_std"],
            payload["same_frequency_coherence_score_by_ensemble"].std(axis=0, ddof=1),
        )
        assert 0.0 <= payload["family_coherence_score"].item() <= 1.0
        assert np.all(payload["component_coherence_scores"] >= 0.0)
        assert np.all(payload["component_coherence_scores"] <= 1.0)
    cross_report = json.loads((cross_dir / "report.json").read_text(encoding="utf-8"))
    assert cross_report["selected_sample_count"] == 7
    assert cross_report["used_sample_count"] == 6
    assert cross_report["dropped_sample_count"] == 1
    assert len(cross_report["dropped_sample_ids"]) == 1
    assert cross_report["ensemble"]["policy"] == "training_aligned_fixed_size_nonoverlapping"
    assert cross_report["ensemble"]["aggregation"] == "training_aligned"
    assert cross_report["ensemble"]["minimum_size"] == 3
    assert cross_report["ensemble"]["ensemble_size"] == 3
    assert cross_report["ensemble"]["ensemble_count"] == 2
    assert cross_report["ensemble"]["sample_count"] == 6
    assert cross_report["statistic_scale"] == "bounded_linear_0_to_1"
    assert cross_report["coherence_score"]["perfect_agreement"] == 1.0
    assert cross_report["graph"]["query_policy"] == "fixed_shared"
    assert cross_report["graph"]["query_point_count"] == 4
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["coherence"]["cross_spectrum"]["family"] == "cross_spectrum"
    assert dataset.closed


def test_posttraining_set_evaluation_builds_matched_source_comparison(tmp_path, monkeypatch):
    case_dir = tmp_path / "case"
    source_run = case_dir / "runs" / "base_experiment" / "base-run"
    child_run = case_dir / "runs" / "post_experiment" / "child-run"
    source_checkpoint = source_run / "checkpoints" / "best.pt"
    child_checkpoint = child_run / "checkpoints" / "best.pt"
    for path, payload in (
        (source_checkpoint, b"source-checkpoint"),
        (child_checkpoint, b"child-checkpoint"),
    ):
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)
    (source_run / "resolved_config.yaml").write_text(
        json.dumps({"stage": "base_training"}), encoding="utf-8"
    )
    child_config = {
        "stage": "post_training",
        "source_run": str(source_run),
        "source_checkpoint": str(source_checkpoint),
        "coherence": {"compute_budget": {"batch_size": 3, "point_count": 4, "query_seed": 17}},
    }
    (child_run / "resolved_config.yaml").write_text(json.dumps(child_config), encoding="utf-8")
    dataset_path = tmp_path / "comparison-dataset.bin"
    dataset_path.write_bytes(b"dataset")

    def runtime_for(run_dir, **kwargs):
        del kwargs
        resolved = Path(run_dir).resolve()
        is_child = resolved == child_run.resolve()
        return EvaluationRuntime(
            config={
                "observations": {
                    "protocol": "random_uniform",
                    "seed": 42,
                    "fields": {"u": {"count": 1}},
                },
                "runtime": {"seed": 42},
                "coherence": child_config["coherence"],
            },
            device=torch.device("cpu"),
            dataset=_FixtureDataset(dataset_path),
            model=_FixtureModel(scale=0.94 if is_child else 0.84),
            checkpoint_path=child_checkpoint if is_child else source_checkpoint,
            generation_steps=2,
            seed=2027,
        )

    monkeypatch.setattr(reconstruction_set, "load_evaluation_runtime", runtime_for)
    monkeypatch.setattr(
        reconstruction_set,
        "warn_if_cuda_memory_tight",
        lambda *args, **kwargs: False,
    )
    reconstruction_limits = []
    coherence_limits = []
    original_reconstruction_renderer = reconstruction_set.render_reconstruction_set_distribution
    original_coherence_renderer = reconstruction_set.render_coherence_distribution

    def capture_reconstruction(*args, **kwargs):
        if kwargs.get("value_limits") is not None:
            reconstruction_limits.append(kwargs["value_limits"])
        return original_reconstruction_renderer(*args, **kwargs)

    def capture_coherence(*args, **kwargs):
        if kwargs.get("value_limits") is not None:
            coherence_limits.append(kwargs["value_limits"])
        return original_coherence_renderer(*args, **kwargs)

    monkeypatch.setattr(
        reconstruction_set,
        "render_reconstruction_set_distribution",
        capture_reconstruction,
    )
    monkeypatch.setattr(
        reconstruction_set,
        "render_coherence_distribution",
        capture_coherence,
    )

    reconstruction_set.evaluate_reconstruction_set(
        child_run,
        case_dir=case_dir,
        split="test",
        max_samples=4,
        coherence_families=["global_distribution", "cross_spectrum"],
        extra_coherence_views=True,
    )

    output_dir = child_run / "evaluation" / "reconstruction_set_test_best"
    assert (output_dir / "relative_l2_violin.png").is_file()
    assert (output_dir / "relative_l2_violin-base.png").is_file()
    global_dir = output_dir / "coherence" / "global_distribution"
    for stem in (
        "marginal_field_distributions",
        "pairwise_field_distributions",
        "joint_top_tail_distributions",
    ):
        assert (global_dir / f"{stem}.png").is_file()
        assert (global_dir / f"{stem}-base.png").is_file()
    cross_dir = output_dir / "coherence" / "cross_spectrum"
    for stem in (
        "self_spectrum_coherence",
        "same_frequency_coherence",
        "cross_frequency_coherence",
    ):
        assert (cross_dir / f"{stem}.png").is_file()
        assert (cross_dir / f"{stem}-base.png").is_file()
    assert (cross_dir / "metrics-base.csv").is_file()
    assert (cross_dir / "metrics-base.npz").is_file()
    assert (cross_dir / "report-base.json").is_file()
    extra_dir = global_dir / "global_distribution_extra"
    assert (extra_dir / "joint_pdf_u-v.png").is_file()
    assert (extra_dir / "joint_pdf_u-v-base.png").is_file()
    assert (extra_dir / "report.json").is_file()
    assert (extra_dir / "report-base.json").is_file()
    with (
        np.load(extra_dir / "joint_pdf_metrics.npz", allow_pickle=False) as current_extra,
        np.load(extra_dir / "joint_pdf_metrics-base.npz", allow_pickle=False) as base_extra,
    ):
        np.testing.assert_array_equal(current_extra["x_edges"], base_extra["x_edges"])
        np.testing.assert_array_equal(current_extra["y_edges"], base_extra["y_edges"])
    assert not (output_dir / "comparison").exists()
    assert len(reconstruction_limits) == 2
    assert reconstruction_limits[0] == reconstruction_limits[1]
    assert len(coherence_limits) == 6
    assert coherence_limits[0] == coherence_limits[1]
    assert coherence_limits[2] == coherence_limits[3]
    assert coherence_limits[4] == coherence_limits[5]
    report = json.loads((output_dir / "comparison_report.json").read_text(encoding="utf-8"))
    assert report["kind"] == "post_training_source_comparison"
    assert report["sample_count"] == 4
    assert report["runs"]["base"]["checkpoint"] == str(source_checkpoint)
    assert report["matched_inputs"]["evaluation_seed"] == 2027
    assert report["shared_axis_limits"]["cross_spectrum"]["coherence_score"] == [
        0.0,
        1.0,
    ]
    assert "density" in report["shared_axis_limits"]["global_distribution_extra"]["u–v"]
    child_report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert child_report["comparison"]["enabled"] is True
    assert not (source_run / "evaluation").exists()
