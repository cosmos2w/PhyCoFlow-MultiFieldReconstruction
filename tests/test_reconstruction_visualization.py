"""Focused checks for one-command checkpoint reconstruction figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pytest

from phycoflow_reconstruction.evaluation import reconstruction_visualization as visualization


def _payload(path: Path, *, query_count: int = 6) -> Path:
    target = np.arange(query_count * 2, dtype=np.float32).reshape(1, query_count, 2)
    prediction = target + np.linspace(0.25, 1.0, query_count * 2, dtype=np.float32).reshape(
        1, query_count, 2
    )
    query_coords_physical = np.asarray(
        [[10.0, 100.0], [20.0, 100.0], [30.0, 100.0], [10.0, 110.0], [20.0, 110.0], [30.0, 110.0]],
        dtype=np.float32,
    )[:query_count]
    np.savez_compressed(
        path,
        prediction_physical=prediction,
        target_physical=target,
        query_coords=np.zeros((1, query_count, 2), dtype=np.float32),
        query_coords_physical=query_coords_physical[None, ...],
        obs_indices=np.asarray([[0, 3]], dtype=np.int64),
        obs_field_ids=np.asarray([[0, 1]], dtype=np.int64),
        obs_valid_mask=np.asarray([[True, True]]),
        logical_shape=np.asarray([2, 3], dtype=np.int64),
        field_names=np.asarray(["u", "v"]),
        sample_id=np.asarray("trajectory_0:9000"),
    )
    return path


def test_render_reconstruction_payload_writes_300_dpi_png(tmp_path, monkeypatch):
    payload = _payload(tmp_path / "reconstruction.npz")
    output = tmp_path / "reconstruction.png"
    recorded: dict[str, object] = {}
    contourf_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    contour_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    colorbar_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    scatter_axes: list[object] = []
    recorded_figure: dict[str, matplotlib.figure.Figure] = {}
    original = matplotlib.figure.Figure.savefig
    original_contourf = plt.Axes.contourf
    original_contour = plt.Axes.contour
    original_colorbar = matplotlib.figure.Figure.colorbar
    original_scatter = plt.Axes.scatter
    original_figure = plt.figure

    def capture_savefig(self, *args, **kwargs):
        recorded["dpi"] = kwargs.get("dpi")
        return original(self, *args, **kwargs)

    def capture_contourf(self, *args, **kwargs):
        contourf_calls.append((args, kwargs))
        return original_contourf(self, *args, **kwargs)

    def capture_contour(self, *args, **kwargs):
        contour_calls.append((args, kwargs))
        return original_contour(self, *args, **kwargs)

    def capture_colorbar(self, *args, **kwargs):
        colorbar_calls.append((args, kwargs))
        return original_colorbar(self, *args, **kwargs)

    def capture_scatter(self, *args, **kwargs):
        scatter_axes.append(self)
        return original_scatter(self, *args, **kwargs)

    def capture_figure(*args, **kwargs):
        figure = original_figure(*args, **kwargs)
        recorded_figure["figure"] = figure
        return figure

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture_savefig)
    monkeypatch.setattr(plt.Axes, "contourf", capture_contourf)
    monkeypatch.setattr(plt.Axes, "contour", capture_contour)
    monkeypatch.setattr(matplotlib.figure.Figure, "colorbar", capture_colorbar)
    monkeypatch.setattr(plt.Axes, "scatter", capture_scatter)
    monkeypatch.setattr(plt, "figure", capture_figure)

    result = visualization.render_reconstruction_payload(payload, output, contour_levels=17)

    assert result == output
    assert output.is_file()
    assert output.stat().st_size > 0
    assert recorded["dpi"] == 300
    assert all(len(call[1]["levels"]) == 17 for call in contourf_calls)
    assert all(len(call[1]["levels"]) == 17 for call in contour_calls)
    np.testing.assert_array_equal(contourf_calls[0][0][0][0], [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(contourf_calls[0][0][1][:, 0], [100.0, 110.0])
    figure = recorded_figure["figure"]
    plot_axes = [axis for axis in figure.axes if axis.get_title()]
    axes = np.asarray(plot_axes, dtype=object).reshape(2, 3)
    assert scatter_axes == [axes[0, 0], axes[1, 0]]
    assert len(colorbar_calls) == 4
    assert all(len(call[1]["ticks"]) == 4 for call in colorbar_calls)
    colorbar_axes = [call[1]["cax"] for call in colorbar_calls]
    expected_parents = [axes[0, 1], axes[0, 2], axes[1, 1], axes[1, 2]]
    for colorbar_axis, parent_axis in zip(colorbar_axes, expected_parents):
        assert colorbar_axis.get_position().height == pytest.approx(
            parent_axis.get_position().height, rel=1.0e-3
        )
    assert all(axis.get_aspect() == 1.0 for axis in axes.flat)
    row_gap = axes[0, 0].get_position().y0 - axes[1, 0].get_position().y1
    assert row_gap < axes[0, 0].get_position().height
    assert all(axis.title.get_fontsize() >= 8.0 for axis in axes.flat)


def test_render_reconstruction_payload_requires_full_grid(tmp_path):
    payload = _payload(tmp_path / "partial.npz", query_count=5)

    with pytest.raises(ValueError, match="requires full-grid query points"):
        visualization.render_reconstruction_payload(payload, tmp_path / "unused.png")


def test_visualize_run_defaults_to_best_first_test_snapshot(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "example" / "run-id"
    output_dir = run_dir / "evaluation" / "reconstruction_test_0000_best"
    output_dir.mkdir(parents=True)
    payload = _payload(output_dir / "reconstruction.npz")
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "trace": {
                    "portable_plot_payload": str(payload.relative_to(run_dir)),
                }
            }
        ),
        encoding="utf-8",
    )
    evaluated: dict[str, object] = {}
    rendered: dict[str, object] = {}

    def fake_evaluate(run, **kwargs):
        evaluated["run"] = run
        evaluated.update(kwargs)
        return report_path

    def fake_render(payload_path, output_path, **kwargs):
        rendered.update(payload=payload_path, output=output_path, **kwargs)
        Path(output_path).write_bytes(b"png")
        return Path(output_path)

    monkeypatch.setattr(visualization, "evaluate_run", fake_evaluate)
    monkeypatch.setattr(visualization, "render_reconstruction_payload", fake_render)
    monkeypatch.setattr(visualization, "warn_if_cuda_memory_tight", lambda *args, **kwargs: False)

    result = visualization.visualize_run(run_dir, case_dir=tmp_path)

    assert evaluated["checkpoint"] == "best"
    assert evaluated["split"] == "test"
    assert evaluated["sample_index"] == 0
    assert evaluated["max_samples"] == 1
    assert evaluated["query_points"] is None
    assert rendered["dpi"] == 300
    assert rendered["contour_levels"] == 20
    assert result == output_dir / "reconstruction.png"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["visualization"]["dpi"] == 300
    assert report["visualization"]["filled_contour_levels"] == 20
    assert report["visualization"]["colorbar_ticks"] == 4
    assert report["visualization"]["line_contour_levels"] == 20


def test_cuda_memory_preflight_warns_with_device_alternative(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "resolved_config.yaml").write_text("runtime: {device: cuda:2}\n")
    (run_dir / "status.json").write_text(
        json.dumps({"peak_cuda_memory_bytes": 8 * 1024**3}),
        encoding="utf-8",
    )
    (run_dir / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
    monkeypatch.setattr(visualization.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        visualization.torch.cuda,
        "mem_get_info",
        lambda device: (4 * 1024**3, 24 * 1024**3),
    )

    warned = visualization.warn_if_cuda_memory_tight(
        run_dir,
        checkpoint="best",
        device_name=None,
    )

    assert warned
    warning = capsys.readouterr().err
    assert "CUDA memory may be tight" in warning
    assert "4.00 GiB free of 24.00 GiB on cuda:2" in warning
    assert "--device cuda:<index>" in warning


def test_cuda_memory_preflight_skips_cpu(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "resolved_config.yaml").write_text("runtime: {device: cpu}\n")

    def unexpected_cuda_check():
        raise AssertionError("CPU visualization should not query CUDA")

    monkeypatch.setattr(visualization.torch.cuda, "is_available", unexpected_cuda_check)

    assert not visualization.warn_if_cuda_memory_tight(
        run_dir,
        checkpoint="best",
        device_name=None,
    )
