"""Training monitoring keeps progress artifacts current and restart-safe."""

import json

import pytest

from phycoflow_reconstruction.training.monitoring import TrainingMonitor


def test_monitor_loads_history_and_updates_loss_figure(tmp_path):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    monitor = TrainingMonitor(
        tmp_path,
        start_step=0,
        final_step=2,
        configured_steps=2,
        steps_per_epoch=2,
        description="test:model",
        enabled=False,
        plot_every_steps=1,
    )
    monitor.record({"step": 1, "total": 3.0}, lr=1.0e-4)
    assert not (metrics / "history.jsonl").exists()
    monitor.record({"step": 2, "total": 1.0, "coherence_loss": 4.0}, lr=1.0e-4)
    assert monitor._epoch_coordinates([1, 2, 3, 4]) == [0.5, 1.0, 1.5, 2.0]
    monitor.record_validation(
        {
            "global_step": 2,
            "training_epoch": 1.0,
            "loss": 2.5,
            "components": {"data_mse": 2.5},
        }
    )
    monitor.finish_step(checkpoint_checked=True, best_checkpoint_saved=True)
    monitor.close(checkpoint_checked=True, best_checkpoint_saved=True)

    history = json.loads((metrics / "history.jsonl").read_text())
    assert history == {
        "batches": 2,
        "epoch": 1,
        "epoch_complete": True,
        "step": 2,
        "total": 2.0,
        "coherence_loss": 4.0,
    }
    assert (tmp_path / "loss_history.png").stat().st_size > 0
    assert monitor.last_epoch_report is not None
    assert monitor.last_epoch_report["total"] == 2.0
    assert monitor.last_epoch_report["coherence_loss"] == 4.0
    assert monitor.last_epoch_report["validation_loss"] == 2.5
    assert monitor.last_epoch_report["best"] == "saved"
    assert monitor.last_epoch_report["train_seconds"] >= 0.0
    assert monitor.last_epoch_report["wall_seconds"] >= monitor.last_epoch_report["train_seconds"]
    validation_history = json.loads((metrics / "validation_history.jsonl").read_text())
    assert validation_history["validation_loss"] == 2.5


def test_loss_figure_combines_terms_and_gives_each_an_independent_panel(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (tmp_path / "metrics").mkdir()
    monitor = TrainingMonitor(
        tmp_path,
        start_step=0,
        final_step=2,
        configured_steps=2,
        steps_per_epoch=1,
        description="test:post-training",
        enabled=False,
        plot_every_steps=10,
    )
    monitor._capture(
        {
            "step": 1,
            "data_loss": 1.0,
            "coherence_loss": 1.0e-3,
            "validation_loss": 0.1,
        }
    )
    monitor._capture(
        {
            "step": 2,
            "data_loss": 0.8,
            "coherence_loss": 2.0e-4,
            "validation_loss": 0.09,
        }
    )

    figure = monitor._build_loss_figure(plt)
    assert len(figure.axes) == 4
    combined, data_axis, coherence_axis, validation_axis = figure.axes
    assert len(combined.lines) == 3
    assert combined.get_title() == "test · post-training objective history"
    assert [axis.get_title(loc="left") for axis in figure.axes[1:]] == [
        "Training data objective",
        "Coherence objective",
        "Fixed validation objective",
    ]
    assert all(len(axis.lines) == 1 for axis in figure.axes[1:])
    assert all(axis.get_yscale() == "log" for axis in figure.axes)
    assert combined.lines[0].get_color() == data_axis.lines[0].get_color()
    assert combined.lines[1].get_color() == coherence_axis.lines[0].get_color()
    assert combined.lines[2].get_color() == validation_axis.lines[0].get_color()
    assert validation_axis.lines[0].get_marker() == "o"
    assert [text.get_text() for text in validation_axis.texts] == [
        "Native model objective · one fixed validation sample"
    ]
    figure.canvas.draw()
    assert validation_axis.get_position().width > 1.5 * data_axis.get_position().width
    plt.close(figure)

    monitor._plot()
    assert [path.name for path in tmp_path.glob("*.png")] == ["loss_history.png"]
    monitor.close()


def test_coherence_panel_shows_multiple_family_contributions(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (tmp_path / "metrics").mkdir()
    monitor = TrainingMonitor(
        tmp_path,
        start_step=0,
        final_step=2,
        configured_steps=2,
        steps_per_epoch=1,
        description="test:multi-family",
        enabled=False,
        plot_every_steps=10,
    )
    for step, total, global_distribution, cross_spectrum in (
        (1, 0.7, 0.4, 0.3),
        (2, 0.5, 0.3, 0.2),
    ):
        monitor._capture(
            {
                "step": step,
                "coherence_loss": total,
                "coherence_family/global_distribution/weighted_contribution": global_distribution,
                "coherence_family/cross_spectrum/weighted_contribution": cross_spectrum,
            }
        )

    figure = monitor._build_loss_figure(plt)
    coherence_axis = figure.axes[1]
    assert coherence_axis.get_title(loc="left") == "Coherence objective"
    assert [line.get_label() for line in coherence_axis.lines] == [
        "Total coherence",
        "Cross spectrum",
        "Global distribution",
    ]
    assert len({line.get_linestyle() for line in coherence_axis.lines}) == 3
    assert coherence_axis.get_yscale() == "log"
    plt.close(figure)
    monitor.close()


def test_monitor_resets_progress_at_each_epoch(tmp_path):
    (tmp_path / "metrics").mkdir()
    monitor = TrainingMonitor(
        tmp_path,
        start_step=0,
        final_step=6,
        configured_steps=6,
        steps_per_epoch=3,
        description="test:model",
        enabled=True,
        plot_every_steps=100,
    )

    monitor.record({"step": 1, "total": 2.0})
    assert monitor.active_epoch == 1
    assert monitor.progress.total == 3
    assert monitor.progress.n == 1

    monitor.record({"step": 2, "total": 1.5})
    monitor.record({"step": 3, "total": 1.0})
    monitor.finish_step(checkpoint_checked=True, best_checkpoint_saved=False)
    assert monitor.last_epoch_report is not None
    assert monitor.last_epoch_report["best"] == "unchanged"
    assert monitor.last_epoch_report["total"] == 1.5
    monitor.record({"step": 4, "total": 0.8})
    assert monitor.active_epoch == 2
    assert monitor.progress.total == 3
    assert monitor.progress.n == 1
    monitor.close()


def test_plot_interval_is_measured_in_epochs(tmp_path):
    (tmp_path / "metrics").mkdir()
    monitor = TrainingMonitor(
        tmp_path,
        start_step=0,
        final_step=12,
        configured_steps=12,
        steps_per_epoch=3,
        description="test:model",
        enabled=False,
        plot_every_steps=3,
    )
    plotted_at: list[int] = []
    monitor._plot = lambda: plotted_at.append(monitor._steps["total"][-1])

    for step in range(1, 13):
        monitor.record({"step": step, "total": float(step)})
        monitor.finish_step()

    assert plotted_at == [3, 9, 12]
    history = [
        json.loads(line)
        for line in (tmp_path / "metrics" / "history.jsonl").read_text().splitlines()
    ]
    assert [row["epoch"] for row in history] == [1, 2, 3, 4]
    assert [row["total"] for row in history] == [2.0, 5.0, 8.0, 11.0]
    monitor.close()


def test_truncated_epoch_writes_one_explicit_partial_history_row(tmp_path):
    (tmp_path / "metrics").mkdir()
    monitor = TrainingMonitor(
        tmp_path,
        start_step=0,
        final_step=2,
        configured_steps=6,
        steps_per_epoch=3,
        description="test:model",
        enabled=False,
        plot_every_steps=10,
    )
    monitor.record({"step": 1, "total": 4.0, "update_mode": "weighted_sum"})
    monitor.record({"step": 2, "total": 2.0, "update_mode": "weighted_sum"})

    history = json.loads((tmp_path / "metrics" / "history.jsonl").read_text())
    assert history["epoch"] == 1
    assert history["batches"] == 2
    assert history["epoch_complete"] is False
    assert history["total"] == 3.0
    assert history["update_mode"] == "weighted_sum"
