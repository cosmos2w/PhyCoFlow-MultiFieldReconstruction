"""Training monitoring keeps progress artifacts current and restart-safe."""

import json

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
    monitor.close()

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
