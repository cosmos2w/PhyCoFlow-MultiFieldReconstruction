"""Prospective R2 data roles, estimator, and monitor invariants on CPU."""

import json
import random
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from phycoflow_reconstruction.evaluation.upgrade_1002_r2_protocol import (
    declare_panels,
    epoch_window_summary,
    error_decomposition,
    matched_native_monitor,
    mature_checkpoint_summary,
    render_pressure_comparison,
    save_declaration,
    selection_indices,
    slice_panel,
)


def test_declared_strata_groups_and_fresh_data_roles(tmp_path):
    used = list(range(96))
    panels = declare_panels(1000, previously_used_indices=used)
    assert panels["test_locked"]
    selection, extended, audit = [
        panels[role]["dataset_indices"] for role in ("selection", "extended", "audit")
    ]
    assert (len(selection), len(extended), len(audit)) == (64, 128, 64)
    assert set(selection) <= set(extended)
    assert not set(audit) & (set(extended) | set(used))
    for role in ("selection", "extended", "audit"):
        panel = panels[role]
        assert all(len(group) == 32 for group in panel["groups"])
        assert [index for group in panel["groups"] for index in group] == panel["dataset_indices"]
    boundaries = np.linspace(0, 1000, 17, dtype=int)
    assert [sum(start <= i < stop for i in selection) for start, stop in pairwise(boundaries)] == [
        4
    ] * 16
    assert panels["audit"]["generation_seed"] != panels["selection"]["generation_seed"]
    assert panels["audit"]["sensor_seed_offset"] != panels["selection"]["sensor_seed_offset"]
    path = save_declaration(tmp_path / "manifest.json", panels)
    save_declaration(path, panels)
    changed = json.loads(path.read_text())
    changed["test_locked"] = False
    with pytest.raises(ValueError, match="previously declared"):
        save_declaration(path, changed)


def test_insufficient_fresh_stratum_rejected():
    with pytest.raises(ValueError, match="insufficient"):
        selection_indices(1000, excluded_indices=list(range(63)))


def test_pressure_offset_identity_is_per_snapshot_and_masked():
    target = torch.zeros(2, 3, 1)
    prediction = torch.tensor([[[1.0], [3.0], [99.0]], [[-3.0], [-1.0], [99.0]]])
    result = error_decomposition(prediction, target, torch.tensor([[True, True, False]] * 2))
    assert result["mse"] == [5.0]
    assert result["mean_offset_squared"] == [4.0]
    assert result["centered_mse"] == [1.0]
    assert result["identity_max_abs_error"] == 0.0
    assert not result["production_metric_gauge_subtracted"]


def test_native_monitor_live_parity_rng_and_modes():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self._ema_eval = True

        def training_loss(self, _batch):
            return SimpleNamespace(
                total=self.weight.square() + torch.rand(()) + random.random() + np.random.random()
            )

    live, source = Model(), Model()
    source.eval().requires_grad_(False)
    live.train()
    batch = SimpleNamespace(
        query_coords=torch.zeros(32, 4, 2), sample_ids=tuple(str(i) for i in range(32))
    )
    torch_state = torch.get_rng_state().clone()
    py_state, np_state = random.getstate(), np.random.get_state()
    result = matched_native_monitor(live, source, {"train": batch, "selection": batch})
    assert result["train"]["ratio"] == 1.0
    assert result["selection"]["ratio"] == 1.0
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert live.training and not source.training
    assert live._ema_eval and source._ema_eval
    assert live.weight.grad is None and source.weight.grad is None


def test_mature_selector_does_not_call_source_or_early_winner_trained():
    records = [
        {"epoch": 0, "is_source_baseline": True, "eligible": True, "metric": 0.0},
        {"epoch": 5, "eligible": True, "metric": 0.2},
        {"epoch": 100, "eligible": True, "metric": 0.8},
        {"epoch": 110, "eligible": False, "metric": 0.4},
        {"epoch": 120, "eligible": True, "metric": 0.7},
    ]
    result = mature_checkpoint_summary(records)
    assert result["best_eligible_mature"]["epoch"] == 120
    assert len(result["nearby_mature"]) == 3
    assert result["eligible_fraction_last_50_epochs"] == pytest.approx(2 / 3)
    assert mature_checkpoint_summary(records[:2])["best_eligible_mature"] is None


def test_slice_preserves_nested_sample_context():
    from phycoflow_reconstruction.contracts import ObservationBatch

    batch = ObservationBatch(
        obs_coords=torch.zeros(4, 1, 2),
        obs_values=torch.zeros(4, 1, 1),
        obs_field_ids=torch.zeros(4, 1, dtype=torch.long),
        obs_valid_mask=torch.ones(4, 1, dtype=torch.bool),
        query_coords=torch.zeros(4, 3, 2),
        query_valid_mask=torch.ones(4, 3, dtype=torch.bool),
        target_fields=torch.zeros(4, 3, 1),
        sample_ids=("a", "b", "c", "d"),
        metadata={
            "query_indices": torch.arange(3).repeat(4, 1),
            "sample_context": {"ids": torch.arange(4)},
        },
    )
    sliced = slice_panel(batch, 2, 4)
    assert sliced.sample_ids == ("c", "d")
    assert sliced.metadata["sample_context"]["ids"].tolist() == [2, 3]
    assert sliced.metadata["query_indices"].shape == (2, 3)
    sliced.validate()


def test_negative_mature_arm_retains_latest_ineligible_neighbor_evidence():
    records = [
        {"epoch": epoch, "eligible": False, "metric": 1.2} for epoch in (5, 100, 110, 140, 150)
    ]
    result = mature_checkpoint_summary(records)
    assert result["best_eligible_mature"] is None
    assert result["center_role"] == "latest_mature"
    assert result["center_eligible"] is False
    assert result["neighbor_center"]["epoch"] == 150
    assert [item["epoch"] for item in result["nearby_mature"]] == [150, 140, 110]
    missing = mature_checkpoint_summary(records[:1])
    assert missing["neighbor_center"] is None and missing["nearby_mature"] == []
    assert missing["center_role"] is None and missing["center_eligible"] is None


def test_native_broad_audit_splits_exact32_and_replays_seeds():
    from phycoflow_reconstruction.contracts import ObservationBatch
    from phycoflow_reconstruction.evaluation.upgrade_1002_r2_protocol import grouped_native_monitor

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def training_loss(self, batch):
            assert len(batch.sample_ids) == 32
            return SimpleNamespace(
                total=self.weight.square() + batch.target_fields.mean() + torch.rand(())
            )

    batch = ObservationBatch(
        obs_coords=torch.zeros(128, 1, 2),
        obs_values=torch.zeros(128, 1, 1),
        obs_field_ids=torch.zeros(128, 1, dtype=torch.long),
        obs_valid_mask=torch.ones(128, 1, dtype=torch.bool),
        query_coords=torch.zeros(128, 3, 2),
        query_valid_mask=torch.ones(128, 3, dtype=torch.bool),
        target_fields=torch.arange(128).float().reshape(128, 1, 1).expand(-1, 3, 1),
        sample_ids=tuple(str(i) for i in range(128)),
        metadata={},
    )
    live, source = Model().train(), Model().eval().requires_grad_(False)
    state = torch.get_rng_state().clone()
    baseline = grouped_native_monitor(live, source, {"extended": batch}, (2027, 3027))["extended"]
    check = grouped_native_monitor(live, source, {"extended": batch}, (6027, 2027, 3027))[
        "extended"
    ]
    assert len(baseline["groups"]) == 4 and baseline["ratio"] == 1.0
    assert baseline["draws"] == check["draws"][1:]
    assert baseline["candidate"] == pytest.approx(
        np.mean([group["candidate"] for group in baseline["groups"]])
    )
    assert torch.equal(state, torch.get_rng_state()) and live.training and not source.training
    assert live.weight.grad is None and source.weight.grad is None
    with pytest.raises(ValueError, match="32"):
        grouped_native_monitor(live, source, {"invalid": slice_panel(batch, 0, 33)})


def test_legacy_grouped_metadata_does_not_claim_pooled_B(monkeypatch):
    import phycoflow_reconstruction.training.post_training as training
    from phycoflow_reconstruction.contracts import ObservationBatch
    from phycoflow_reconstruction.evaluation.upgrade_1002_r2_protocol import grouped_evaluate

    batch = ObservationBatch(
        obs_coords=torch.zeros(64, 1, 2),
        obs_values=torch.zeros(64, 1, 1),
        obs_field_ids=torch.zeros(64, 1, dtype=torch.long),
        obs_valid_mask=torch.ones(64, 1, dtype=torch.bool),
        query_coords=torch.zeros(64, 3, 2),
        query_valid_mask=torch.ones(64, 3, dtype=torch.bool),
        target_fields=torch.zeros(64, 3, 1),
        sample_ids=tuple(str(i) for i in range(64)),
        metadata={"query_indices": torch.arange(3).repeat(64, 1)},
    )

    class Model(torch.nn.Module):
        capabilities = SimpleNamespace(structured_grid_required=False)

        def reconstruct(self, batch):
            return SimpleNamespace(prediction=batch.target_fields)

    def saved_group(model, complete, comparison, *args, **kwargs):
        model.reconstruct(comparison)
        total = 1.0 if complete.sample_ids[0] == "0" else 3.0
        return {
            "sample_ids": list(complete.sample_ids),
            "query_ids": comparison.metadata["query_indices"].tolist(),
            "inference": {"samples": 32, "seconds": 1.0},
            "coherence": {"families": {"cross_spectrum": {"total": total, "reference_ids": []}}},
        }

    monkeypatch.setattr(training, "_evaluate", saved_group)
    result = grouped_evaluate(
        Model(),
        batch,
        batch,
        {"cross_spectrum": SimpleNamespace(definition="legacy_v3")},
        {},
        ("p",),
        {},
    )
    assert result["aggregation"] == "equal_sample_group32_mean" and "pooled_B" not in result
    assert result["coherence"]["families"]["cross_spectrum"]["total"] == 2.0
    assert len(result["group_reports"]) == 2


def test_grouped_evaluator_reuses_groups_and_pools_covariance(tmp_path, monkeypatch):
    from test_upgrade_1002_spectrum_evaluation import _field_snapshots, _runtime_with_artifact

    import phycoflow_reconstruction.training.post_training as training
    from phycoflow_reconstruction.coherence.families.cross_spectrum.covariance_blocks import (
        second_order_covariance_block_losses,
    )
    from phycoflow_reconstruction.coherence.families.cross_spectrum.statistics import graph_fourier
    from phycoflow_reconstruction.contracts import ObservationBatch
    from phycoflow_reconstruction.evaluation.upgrade_1002_r2_protocol import grouped_evaluate

    _runtime, family, coords, basis = _runtime_with_artifact(tmp_path)
    generated, target = _field_snapshots(basis, coords, 64)
    batch = ObservationBatch(
        obs_coords=torch.zeros(64, 1, 2),
        obs_values=torch.zeros(64, 1, 1),
        obs_field_ids=torch.zeros(64, 1, dtype=torch.long),
        obs_valid_mask=torch.ones(64, 1, dtype=torch.bool),
        query_coords=coords.expand(64, -1, -1),
        query_valid_mask=torch.ones(64, coords.shape[0], dtype=torch.bool),
        target_fields=target,
        sample_ids=tuple(str(i) for i in range(64)),
        metadata={"query_indices": torch.arange(coords.shape[0]).repeat(64, 1)},
    )

    class Model(torch.nn.Module):
        capabilities = SimpleNamespace(structured_grid_required=False)

        def __init__(self):
            super().__init__()
            self._ema_eval = True

        def reconstruct(self, batch, **_kwargs):
            assert self._ema_eval is False
            ids = [int(value) for value in batch.sample_ids]
            return SimpleNamespace(prediction=generated[ids])

    calls = []

    def fake_evaluate(model, complete, comparison, families, banks, fields, config, **kwargs):
        calls.append(complete.sample_ids)
        prediction = model.reconstruct(comparison).prediction
        result = families["cross_spectrum"](
            prediction, comparison.target_fields, coordinates=comparison.query_coords
        )
        return {
            "mse_normalized": float((prediction - comparison.target_fields).square().mean()),
            "sample_ids": list(comparison.sample_ids),
            "query_ids": comparison.metadata["query_indices"].tolist(),
            "inference": {"samples": 32, "seconds": 1.0},
            "coherence": {
                "families": {
                    "cross_spectrum": {
                        "total": float(result.scalar_loss),
                        "reference_ids": list(comparison.sample_ids),
                        "diagnostics": result.diagnostics,
                    }
                }
            },
        }

    monkeypatch.setattr(training, "_evaluate", fake_evaluate)
    model = Model().train()
    result = grouped_evaluate(model, batch, batch, {"cross_spectrum": family}, {}, ("u", "v"), {})
    assert result["aggregation"] == "equal_sample_group32_mean_with_separate_pooled_B"
    assert model._ema_eval is True and model.training
    assert result["evaluation_weight_source"] == "live_checkpoint_model_weights"
    assert [len(call) for call in calls] == [32, 32]
    assert result["sample_ids"] == list(batch.sample_ids)
    assert result["pooled_B"]["samples"] == 64
    direct = second_order_covariance_block_losses(
        graph_fourier(generated, basis),
        graph_fourier(target, basis),
        family.band_ids,
        family.pairs,
        relative_floor=family.relative_floor,
        absolute_floor=family.absolute_floor,
        minimum_reference_band_fraction=family.minimum_reference_band_fraction,
        energy_floor_policy=family.energy_floor_policy,
        calibration_reference_energies=family.calibrated_band_energies,
        calibration_ensemble_size=family.calibration_ensemble_size,
    )
    assert result["pooled_B"]["same_frequency"] == pytest.approx(
        float(direct.same_frequency), rel=2e-5
    )
    assert result["pooled_B"]["cross_frequency"] == pytest.approx(
        float(direct.cross_frequency), rel=2e-5
    )
    assert result["error_decomposition"]["mse"] == pytest.approx(
        (generated - target).square().mean(dim=(0, 1)).tolist()
    )


def test_epoch_windows_keep_missing_evidence_and_resume_axis():
    records = [{"epoch": epoch, "ratio": 1.0 + epoch * 0.001} for epoch in range(100, 151, 5)]
    result = epoch_window_summary(records, "ratio")
    assert result["windows"]["epochs_1_25"]["median"] is None
    assert result["latest_epoch"] == 150
    assert result["robust_slope_last_50_epochs"] == pytest.approx(0.001)


def test_pressure_review_pdf_only(tmp_path):
    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    target = np.asarray([0.0, 1.0, 2.0, 3.0])
    path = render_pressure_comparison(
        coords,
        target,
        target + 0.1,
        target - 0.2,
        tmp_path / "p.pdf",
        sample_id="validation:8032",
        epoch=100,
    )
    assert path.read_bytes().startswith(b"%PDF")
    assert sorted(item.suffix for item in tmp_path.iterdir()) == [".pdf"]
    with pytest.raises(ValueError, match="PDF"):
        render_pressure_comparison(
            coords,
            target,
            target,
            target,
            tmp_path / "p.png",
            sample_id="validation:8032",
            epoch=100,
        )
