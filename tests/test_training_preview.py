"""Unit checks for quantitative training-preview annotations."""

from types import SimpleNamespace

import numpy as np
import torch

from phycoflow_reconstruction.contracts import LossBundle
from phycoflow_reconstruction.training.preview import (
    TrainingReconstructionPreview,
    _absolute_error_title,
    _coordinates_in_dataset_units,
    _relative_l2_error,
    render_preview_payload,
)
from phycoflow_reconstruction.training.run_store import RunStore


def test_disabled_preview_is_safe_between_sparse_checkpoints(tmp_path):
    from phycoflow_reconstruction.training.checkpointing import PeriodicCheckpointManager
    config = {"evaluation": {"preview": {"enabled": False}},
              "checkpointing": {"every_epochs": 2, "save_epoch_one": True}}
    store = SimpleNamespace(run_dir=tmp_path)
    preview = TrainingReconstructionPreview(config, store=store, steps_per_epoch=9,
                                            device=torch.device("cpu"))
    manager = PeriodicCheckpointManager(config, store=store, steps_per_epoch=9)
    for step in (1, 9, 10, 18, 27):
        assert not preview.due_loss(step)
        assert not preview.due_reconstruction(step)
        assert not preview.due(step)
        assert manager.due_for_preview_or_checkpoint(step, preview) == (step in (9, 18))
    assert preview.update(torch.nn.Linear(1, 1), global_step=18, force=True) is None
    assert preview.last_validation_report is None
    assert preview.dataset is None and preview.batch is None
    assert not preview.output_dir.exists()
    preview.close()


def test_relative_l2_error_uses_field_reference_norm():
    truth = np.asarray([3.0, 4.0])
    estimate = np.asarray([0.0, 0.0])

    value = _relative_l2_error(estimate, truth)

    assert value == 1.0
    assert _absolute_error_title(value) == "Absolute error\nRelative $L_2$ = 1.000e+00"


def test_relative_l2_error_marks_zero_reference_as_unavailable():
    value = _relative_l2_error(np.ones(4), np.zeros(4))

    assert value is None
    assert _absolute_error_title(value) == "Absolute error\nRelative $L_2$ = N/A"


def test_coordinate_display_inverse_uses_raw_dataset_ranges():
    normalized_reference = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    physical_reference = np.asarray([[10.0, -4.0], [30.0, 8.0]])
    normalized_queries = np.asarray([[0.25, 0.5], [0.75, 0.0]])

    physical_queries = _coordinates_in_dataset_units(
        normalized_queries, normalized_reference, physical_reference
    )

    np.testing.assert_allclose(physical_queries, [[15.0, 2.0], [25.0, -4.0]])


def test_preview_renderer_uses_physical_axes_and_exports_vector_formats(tmp_path):
    payload_path = tmp_path / "preview.npz"
    output_stem = tmp_path / "external" / "abc_preview"
    rows, columns = 3, 4
    normalized_x, normalized_y = np.meshgrid(
        np.linspace(0.0, 1.0, columns),
        np.linspace(0.0, 1.0, rows),
    )
    query_coords = np.column_stack((normalized_x.ravel(), normalized_y.ravel()))
    physical_coords = np.column_stack(
        (5.0 + 2.0 * query_coords[:, 0], -3.0 + 4.0 * query_coords[:, 1])
    )
    target = np.column_stack(
        (np.linspace(280.0, 320.0, rows * columns), np.linspace(90_000.0, 110_000.0, rows * columns))
    )
    prediction = target + np.column_stack(
        (np.linspace(-2.0, 2.0, rows * columns), np.linspace(100.0, 200.0, rows * columns))
    )
    np.savez_compressed(
        payload_path,
        prediction_physical=prediction,
        target_physical=target,
        query_coords=query_coords,
        query_coords_physical=physical_coords,
        obs_coords=np.asarray([[0.0, 0.0], [0.5, 0.5]]),
        obs_coords_physical=np.asarray([[5.0, -3.0], [6.0, -1.0]]),
        obs_values_physical=np.asarray([280.0, 300.0]),
        obs_field_ids=np.asarray([0, 0]),
        obs_valid_mask=np.asarray([True, True]),
        logical_shape=np.asarray([rows, columns]),
        field_names=np.asarray(["T", "p"]),
        field_units=np.asarray(["K", "Pa"]),
        sample_id=np.asarray("fixture:12"),
    )

    outputs = render_preview_payload(payload_path, output_stem=output_stem, epoch=12.0)

    assert outputs == tuple(output_stem.with_suffix(suffix) for suffix in (".png", ".svg", ".pdf"))
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)


def test_sparse_preview_uses_light_zero_error_palette(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    payload_path = tmp_path / "sparse_preview.npz"
    output_stem = tmp_path / "sparse_preview"
    recorded_cmaps = []
    original_scatter = plt.Axes.scatter

    def capture_scatter(self, *args, **kwargs):
        recorded_cmaps.append(kwargs.get("cmap"))
        return original_scatter(self, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "scatter", capture_scatter)
    coords = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    np.savez_compressed(
        payload_path,
        prediction_physical=np.asarray([[1.0], [2.0]]),
        target_physical=np.asarray([[1.0], [3.0]]),
        query_coords=coords,
        query_coords_physical=coords,
        obs_coords=np.empty((0, 2)),
        obs_coords_physical=np.empty((0, 2)),
        obs_values_physical=np.empty((0,)),
        obs_field_ids=np.empty((0,), dtype=np.int64),
        obs_valid_mask=np.empty((0,), dtype=bool),
        logical_shape=np.asarray([2, 2]),
        field_names=np.asarray(["T"]),
        field_units=np.asarray(["K"]),
        sample_id=np.asarray("fixture:sparse"),
    )

    render_preview_payload(payload_path, output_stem=output_stem, epoch=3.0)

    assert recorded_cmaps == ["viridis", "viridis", "YlOrRd"]


def test_validation_loss_and_reconstruction_have_independent_cadences():
    preview = TrainingReconstructionPreview.__new__(TrainingReconstructionPreview)
    preview.enabled = True
    preview.steps_per_epoch = 2
    preview.loss_every_epochs = 10
    preview.reconstruct_every_epochs = 500

    assert preview.due_loss(20)
    assert not preview.due_reconstruction(20)
    assert preview.due_reconstruction(1000)
    assert preview.due(20)
    assert not preview.due(18)


def test_validation_loss_is_seeded_and_does_not_advance_training_rng(tmp_path):
    class RandomLossModel(torch.nn.Module):
        def training_loss(self, _batch):
            value = torch.rand(())
            return LossBundle(value, {"random_component": value})

    config = {"stage": "base_training", "case": "fixture", "output": {}}
    preview = TrainingReconstructionPreview.__new__(TrainingReconstructionPreview)
    preview.batch = SimpleNamespace(sample_ids=("validation:0",))
    preview.device = torch.device("cpu")
    preview.settings = {"seed": 123}
    preview.steps_per_epoch = 4
    preview.store = RunStore.create(tmp_path, "validation", config)
    model = RandomLossModel()

    torch.manual_seed(999)
    expected_next = torch.rand(())
    torch.manual_seed(999)
    first = preview._validation_loss(model, global_step=40)
    actual_next = torch.rand(())
    second = preview._validation_loss(model, global_step=80)

    assert first["loss"] == second["loss"]
    assert first["components"] == second["components"]
    assert torch.equal(actual_next, expected_next)
