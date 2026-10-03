"""R2 reporting creates a PDF directly and preserves epoch coordinates."""
from phycoflow_reconstruction.training.coherence_history import extract_adaptive_coherence_history
from phycoflow_reconstruction.training.monitoring import TrainingMonitor


def test_pdf_monitor_exports_no_raster_or_svg(tmp_path):
    (tmp_path / "metrics").mkdir()
    monitor = TrainingMonitor(
        tmp_path, start_step=0, final_step=2, configured_steps=2, steps_per_epoch=2,
        description="post_training:r2", enabled=False, plot_every_steps=1, plot_format="pdf",
    )
    monitor.record({"step": 1, "total": 3., "data_loss": 1.}, lr=1e-5)
    monitor.record({"step": 2, "total": 2., "data_loss": .9}, lr=1e-5)
    monitor.close()
    assert monitor._epoch_coordinates([2, 4]) == [1., 2.]
    assert (tmp_path / "loss_history.pdf").read_bytes().startswith(b"%PDF")
    assert not list(tmp_path.rglob("*.png"))
    assert not list(tmp_path.rglob("*.svg"))


def test_adaptive_pdf_reporting_converts_saved_child_age_to_epochs():
    data = extract_adaptive_coherence_history(
        [{"step": 3800, "epoch": 100, "gradient/A/B/cosine": .5}],
        [{"step": 5700, "eligible": True}], selected={"global_step": 3800},
        config={"runtime": {"plot_format": "pdf"}}, steps_per_epoch=38,
    )
    assert data["x_label"] == "Epoch"
    assert data["training_records"][0]["x"] == 100
    assert data["validation_records"][0]["x"] == 150
    assert data["selected"]["x"] == 100
