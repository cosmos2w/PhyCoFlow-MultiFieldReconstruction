"""Fixed-panel fidelity history shows gates and the selected eligible point."""

from __future__ import annotations

import json

import pytest

from phycoflow_reconstruction.training.fidelity_history import (
    build_fidelity_history_figure,
    render_fidelity_history,
)


def _rows():
    return [
        {
            "step": step,
            "metric": metric,
            "eligible": eligible,
            "failed_fidelity_fields": failed,
            "relative_mse_increase": relative,
        }
        for step, metric, eligible, failed, relative in (
            (0, 0.9, True, [], {"total": 0.0, "CO": 0.0, "T": 0.0}),
            (10, 0.7, True, [], {"total": -0.01, "CO": 0.02, "T": -0.03}),
            (20, 0.65, False, ["CO"], {"total": 0.03, "CO": 0.08, "T": 0.01}),
            (30, 0.55, True, [], {"total": 0.0, "CO": -0.01, "T": -0.02}),
        )
    ]


def test_fidelity_figure_separates_candidates_field_gates_and_selection() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    training_rows = [
        {"step": 10, "epoch": 1},
        {"step": 20, "epoch": 2},
        {"step": 30, "epoch": 3},
    ]
    figure = build_fidelity_history_figure(
        _rows(),
        training_rows,
        plt,
        total_budget=0.05,
        field_budget=0.05,
        selected_step=30,
    )
    assert figure is not None
    score_axis, full_range_axis, gate_axis, legend_axis = figure.axes
    assert score_axis.get_title(loc="left") == "Fixed-panel topology score · lower is better"
    assert len(score_axis.collections) == 3  # eligible, rejected, selected checkpoint
    assert score_axis.collections[1].get_paths()[0].vertices.shape[0] > 0
    assert full_range_axis.get_title(loc="left") == "Source-relative normalized MSE · full range"
    assert full_range_axis.get_yscale() == "linear"
    assert full_range_axis.get_ylim()[1] > 0.08
    assert gate_axis.get_title(loc="left") == "Fidelity gates · focused range"
    assert gate_axis.get_yscale() == "linear"
    assert gate_axis.get_ylim()[1] < full_range_axis.get_ylim()[1]
    assert gate_axis.get_xlabel() == "Training epoch"
    assert len(full_range_axis.lines) == 6  # fields, zero and gate lines, selected-step guide
    assert len(gate_axis.collections) > len(full_range_axis.collections)  # out-of-range triangle
    assert len(legend_axis.get_legend().get_texts()) == 3
    figure.canvas.draw()
    plt.close(figure)


def test_renderer_reads_saved_jsonl_and_writes_only_requested_output(tmp_path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    metrics = tmp_path / "run" / "metrics"
    metrics.mkdir(parents=True)
    run_dir = metrics.parent
    (run_dir / "resolved_config.yaml").write_text(
        "posttrain_fidelity:\n  max_relative_mse_increase: 0.05\n  max_relative_field_mse_increase: 0.05\n",
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"steps_per_epoch": 10}), encoding="utf-8"
    )
    (run_dir / "evaluation").mkdir()
    (run_dir / "evaluation" / "selected.json").write_text(
        json.dumps({"global_step": 30}), encoding="utf-8"
    )
    (metrics / "history.jsonl").write_text(
        "\n".join(json.dumps(row) for row in ({"step": 10, "epoch": 1}, {"step": 20, "epoch": 2}, {"step": 30, "epoch": 3})) + "\n",
        encoding="utf-8",
    )
    (metrics / "topology_validation.jsonl").write_text(
        "\n".join(json.dumps(row) for row in _rows()) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "examples" / "checkpoint_fidelity.png"

    path = render_fidelity_history(run_dir, output_path=output, pyplot=plt)

    assert path == output
    assert path.stat().st_size > 0
    assert not (run_dir / "checkpoint_fidelity.png").exists()
