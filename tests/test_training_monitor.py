"""Training monitoring keeps progress artifacts current and restart-safe."""

import json

from phycoflow_reconstruction.training.monitoring import TrainingMonitor


def test_monitor_loads_history_and_updates_loss_figure(tmp_path):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "history.jsonl").write_text(
        json.dumps({"step": 1, "total": 2.0}) + "\n",
        encoding="utf-8",
    )
    monitor = TrainingMonitor(
        tmp_path,
        start_step=1,
        final_step=2,
        configured_steps=2,
        steps_per_epoch=2,
        description="test:model",
        enabled=False,
        plot_every_steps=1,
    )
    monitor.record({"step": 2, "total": 1.0}, lr=1.0e-4)
    assert monitor._epoch_coordinates([1, 2, 3, 4]) == [0.5, 1.0, 1.5, 2.0]
    monitor.close()

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
