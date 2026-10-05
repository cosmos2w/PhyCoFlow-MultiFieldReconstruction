"""R4 reporting is typed, modestly scheduled, and scientifically inert."""

import json
import copy
from types import SimpleNamespace

import pytest
import torch

from phycoflow_reconstruction.contracts import LossBundle
from phycoflow_reconstruction.training.checkpointing import PeriodicCheckpointManager
from phycoflow_reconstruction.training.coherence_history import (
    extract_coherence_history, recover_validation_metric_identity,
)
from phycoflow_reconstruction.training.monitoring import TrainingMonitor
from phycoflow_reconstruction.training.preview import TrainingReconstructionPreview
from phycoflow_reconstruction.training.run_store import RunStore


class _Preview:
    enabled = True

    def due(self, step):
        return step % 25 == 0

    def update(self, model, *, global_step, force=False):
        if not self.due(global_step):
            return None
        return {"validation": {"global_step": global_step,
            "training_epoch": float(global_step), "sample_id": "validation:0",
            "metric_name": "validation/native_preview_single_sample",
            "scope": "single_fixed_sample", "normalization": "native",
            "generation_convention": "native_loss", "native_noise_convention": "fixed_seed",
            "loss": .125, "value": .125}, "reconstruction": None}


def test_native_selector_and_coincident_epochs_never_mix(tmp_path):
    config = {"stage": "post_training", "checkpointing": {
        "every_epochs": 10, "validation_every_epochs": 10,
        "selection_metric": "coherence_with_fidelity"}}
    store = RunStore.create(tmp_path, "typed", config)
    manager = PeriodicCheckpointManager(config, store=store, steps_per_epoch=1)
    manager.panel_evaluator = lambda step: {"metric": 1.25, "mse": .5,
        "family_source_normalized_scores": {"A": 1.25, "B": 1.25, "C": 1.25},
        "eligible": True, "metrics": {"sample_ids": ["validation:1", "validation:2"],
        "sensor_manifest_sha256": "panel-sha", "generation_steps": 2,
        "generation_seed": 42, "per_field_mse_normalized": {"T": .4}}}
    monitor = TrainingMonitor(store.run_dir, start_step=0, final_step=100,
        configured_steps=100, steps_per_epoch=1, description="post_training:test",
        enabled=False, plot_every_epochs=25, plot_format="pdf")
    rendered = []
    monitor._plot = lambda: rendered.append(True)
    model = torch.nn.Linear(1, 1)
    for epoch in (20, 25, 50):
        manager.save({"model": model.state_dict()}, model=model, preview=_Preview(),
                     global_step=epoch, fallback_metric=0.)
        monitor.record_validation(manager.last_validation_report)
    assert monitor._steps["validation/native_preview_single_sample"] == [25, 50]
    assert monitor._steps["validation/coherence_selection_score"] == [20, 50]
    assert monitor._values["validation/native_preview_single_sample"] == [.125, .125]
    assert monitor._values["validation/coherence_selection_score"] == [1.25, 1.25]
    assert not monitor._values["validation_loss"]
    assert not rendered  # Validation never triggers full growing-history render.
    rows = [json.loads(line) for line in monitor.validation_history_path.read_text().splitlines()]
    assert all({"epoch", "metric_name", "scope", "normalization",
                "generation_convention", "native_noise_convention"} <= row.keys() for row in rows)
    assert sum(row["epoch"] == 50 for row in rows) == 4
    monitor.close()


def test_legacy_split_requires_recoverable_origin_value():
    rows = [{"step": epoch, "validation_loss": value}
            for epoch, value in ((20, 1.25), (25, .125), (50, 1.25), (75, .2))]
    split = recover_validation_metric_identity(rows,
        native_reports=[{"global_step": 25, "loss": .125}, {"global_step": 50, "loss": .125}],
        selector_reports=[{"step": 20, "metric": 1.25}, {"step": 50, "metric": 1.25}])
    assert [row["metric_name"] for row in split] == [
        "validation/coherence_selection_score", "validation/native_preview_single_sample",
        "validation/coherence_selection_score", "validation/unknown_legacy"]
    ambiguous = recover_validation_metric_identity([{"step": 50, "loss": 1.}],
        native_reports=[{"step": 50, "loss": 1.}], selector_reports=[{"step": 50, "metric": 1.}])
    assert ambiguous[0]["metric_name"] == "validation/unknown_legacy"


@pytest.mark.parametrize("fail", [False, True])
def test_native_preview_restores_rng_parameters_gradients_and_mixed_modes(tmp_path, fail):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(1, 1)

        def training_loss(self, batch):
            value = self.layer(torch.rand(1, 1)).square().sum()
            if fail:
                raise RuntimeError("fixture failure")
            return LossBundle(value, {"native": value})

    model = Model()
    model.train()
    model.layer.eval()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    rng = torch.random.get_rng_state().clone()
    preview = TrainingReconstructionPreview.__new__(TrainingReconstructionPreview)
    preview.batch = SimpleNamespace(sample_ids=("validation:0",))
    preview.device = torch.device("cpu")
    preview.settings = {"seed": 123}
    preview.steps_per_epoch = 1
    preview.store = RunStore.create(tmp_path, "native", {"stage": "base_training"})
    if fail:
        with pytest.raises(RuntimeError, match="fixture failure"):
            preview._validation_loss(model, global_step=25)
    else:
        report = preview._validation_loss(model, global_step=25)
        assert report["metric_name"] == "validation/native_preview_single_sample"
    assert torch.equal(rng, torch.random.get_rng_state())
    assert model.training and not model.layer.training
    assert all(torch.equal(state[key], value) for key, value in model.state_dict().items())
    assert all(torch.equal(before, parameter.grad) for before, parameter in zip(gradients, model.parameters()))


def test_reporting_toggle_preserves_actual_adamw_update_and_rng(tmp_path):
    class Model(torch.nn.Linear):
        def training_loss(self, batch):
            value = self(torch.rand(3, 1)).square().mean()
            return LossBundle(value, {"native": value})

    initial = Model(1, 1)
    outputs = []
    for enabled in (False, True):
        model = copy.deepcopy(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        torch.manual_seed(321)
        objective = model(torch.rand(3, 1)).square().mean()
        objective.backward()
        preview = TrainingReconstructionPreview.__new__(TrainingReconstructionPreview)
        preview.batch = SimpleNamespace(sample_ids=("validation:0",))
        preview.device = torch.device("cpu")
        preview.settings = {"seed": 123}
        preview.steps_per_epoch = 1
        store = RunStore.create(tmp_path, f"update_{enabled}", {"stage": "base_training"})
        preview.store = store
        monitor = TrainingMonitor(store.run_dir, start_step=0, final_step=1,
            configured_steps=1, steps_per_epoch=1, description="base_training:test",
            enabled=False, plot_format="pdf", plot_every_epochs=25)
        if enabled:
            monitor.record_validation(preview._validation_loss(model, global_step=25))
        optimizer.step()
        monitor.record({"step": 1, "total": float(objective.detach())})
        monitor.finish_step()
        monitor.close()
        outputs.append((model.state_dict(), torch.random.get_rng_state().clone(),
                        optimizer.state_dict()))
    assert torch.equal(outputs[0][1], outputs[1][1])
    assert all(torch.equal(outputs[0][0][key], value) for key, value in outputs[1][0].items())
    for parameter_id, state in outputs[0][2]["state"].items():
        for key, value in state.items():
            assert torch.equal(value, outputs[1][2]["state"][parameter_id][key])


def test_topology_leaf_role_is_explicit_independent_of_zero_weights():
    prefix = "coherence_component/topology/"
    data = extract_coherence_history([{"epoch": 1,
        f"{prefix}finite_primary/raw": 1.,
        f"{prefix}finite_primary/weighted_contribution": 1.,
        f"{prefix}finite_primary/role": "training_component",
        f"{prefix}self.T.h0/raw": 2.,
        f"{prefix}self.T.h0/weighted_contribution": 0.,
        f"{prefix}self.T.h0/role": "evaluation_only"}], {})
    assert {component.component: component.role for component in data.components} == {
        "finite_primary": "training_component", "self.T.h0": "evaluation_only"}
