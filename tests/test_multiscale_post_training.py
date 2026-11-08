"""Independent conditional draws, frozen calibration and checkpoint recovery."""

from __future__ import annotations

import copy
import json
import math

import pytest
import torch
from helpers.coherence import _base_config, _post_config, _write_fixture
from helpers.spectral import (
    _fitted,
    _geometry,
    _grid,
    _smooth_states,
    _third,
)

from phycoflow_reconstruction.config.validate import validate_config
from phycoflow_reconstruction.contracts import ModelCapabilities, ObservationBatch
from phycoflow_reconstruction.training.base_training import run_base_training
from phycoflow_reconstruction.training.post_training import (
    _coherence_objective,
    _conditional_draw_seed,
    run_post_training,
)
from phycoflow_reconstruction.training.rollout import differentiable_reconstruction


def test_shared_rollout_supplies_target_free_independent_differentiable_draws():
    class NoisyToyFlow(torch.nn.Module):
        capabilities = ModelCapabilities("point", True, True, False, True)

        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.4, dtype=torch.float64))
            self.rollout_calls = 0

        def sample_source(self, batch, *, generator=None):
            assert batch.target_fields is None
            self.rollout_calls += 1
            shape = (*batch.query_coords.shape[:2], 2)
            return 0.3 * torch.randn(shape, generator=generator, dtype=torch.float64)

        def velocity(self, batch, state, time):
            assert batch.target_fields is None
            x, y = batch.query_coords.unbind(dim=-1)
            pattern = torch.stack(
                (torch.sin(math.pi * x) * torch.cos(math.pi * y), torch.cos(2 * math.pi * x) * y),
                dim=-1,
            )
            return self.scale * pattern - 0.2 * state

    coordinates = _grid(6)
    states = _smooth_states(coordinates, 8, fields=2)
    family = _fitted(
        {
            "fields": ["u", "v"],
            "geometry": _geometry(),
            "bands": {"count": 2},
            "numerics": {"dtype": "float64"},
            "objective": {"kind": "conditional_energy_score", "conditional_draws": 2},
        },
        coordinates,
        states,
        fields=("u", "v"),
    )
    grid = coordinates.unsqueeze(0).expand(4, -1, -1)
    batch = ObservationBatch(
        obs_coords=grid[:, :2],
        obs_values=torch.zeros(4, 2, 1, dtype=torch.float64),
        obs_field_ids=torch.tensor([[0, 1]]).expand(4, -1),
        obs_valid_mask=torch.ones(4, 2, dtype=torch.bool),
        query_coords=grid,
        query_valid_mask=torch.ones(4, 36, dtype=torch.bool),
        target_fields=states[:4],
        sample_ids=("a", "b", "c", "d"),
        obs_indices=torch.tensor([[0, 1]]).expand(4, -1),
        logical_shapes=((36,),) * 4,
        metadata={"query_indices": torch.arange(36).expand(4, -1)},
    )
    config = {
        "coherence": {"compute_budget": {"batch_size": 4, "point_count": 36}},
        "rollout": {"steps": 2, "solver": "euler"},
        "observation_consistency": {"mode": "none", "final_clamp": False},
    }
    model = NoisyToyFlow()
    result, _ = _coherence_objective(
        model,
        batch,
        {"multiscale_spectrum": family},
        {"multiscale_spectrum": None},
        config,
        step=0,
        generator=torch.Generator().manual_seed(2),
    )
    assert model.rollout_calls == 2
    path = "multiscale_spectrum.descriptor_energy_score.conditional"
    # Band power and co-spectra are on by default; the cross-scale block is opt-in.
    assert set(result.component_results) == {
        path,
        "multiscale_spectrum.log_band_power.squared_error",
        "multiscale_spectrum.signed_band_cospectrum.squared_error",
    }
    assert result.component_results[path].diagnostics["repulsion"] > 0
    (gradient,) = torch.autograd.grad(result.scalar_loss, model.scale)

    target_free = copy.copy(batch)
    target_free.target_fields = None
    rollout = {
        "steps": 2,
        "solver": "euler",
        "observation_config": {"mode": "none", "final_clamp": False},
    }
    first = differentiable_reconstruction(
        model, target_free, generator=torch.Generator().manual_seed(2), **rollout
    )
    second = differentiable_reconstruction(
        model,
        target_free,
        generator=torch.Generator().manual_seed(_conditional_draw_seed(2, 1)),
        **rollout,
    )
    assert not torch.allclose(first, second)

    def objective(draw):
        return family(
            first, states[:4], coordinates=grid, context={"conditional_draws": (draw,)}
        ).scalar_loss

    (live,) = torch.autograd.grad(objective(second), model.scale, retain_graph=True)
    (detached,) = torch.autograd.grad(objective(second.detach()), model.scale)
    torch.testing.assert_close(gradient, live)
    assert not torch.allclose(live, detached)


def test_multiscale_posttraining_fits_references_scores_draws_and_resumes(tmp_path, monkeypatch):
    dataset_path = tmp_path / "fixture.h5"
    _write_fixture(dataset_path)
    case_dir = tmp_path / "case"
    source_run = run_base_training(_base_config(dataset_path), case_dir=case_dir)
    post = _post_config(dataset_path, source_run)
    post["optimization"]["epochs"] = 3
    post["runtime"].update(progress=False, plot_every_steps=1000)
    post["evaluation"]["preview"] = {"enabled": False}
    post["coherence"]["family_balance"] = {
        "mode": "none",
        "gradient_diagnostics_every_epochs": 1,
    }
    import phycoflow_reconstruction.training.post_training as trainer

    original_diagnostics = trainer._sparse_family_gradient_diagnostics

    def diagnostics(*args, **kwargs):
        assert kwargs["raw_family_losses"] is not None
        with monkeypatch.context() as context:

            def forbidden(*args, **kwargs):
                raise AssertionError("gradient diagnostics must reuse the existing draws")

            context.setattr(trainer, "_coherence_objective", forbidden)
            return original_diagnostics(*args, **kwargs)

    monkeypatch.setattr(trainer, "_sparse_family_gradient_diagnostics", diagnostics)
    post["evaluation"]["query_points"] = 16
    post["coherence"]["compute_budget"] = {
        "batch_size": 1,
        "point_count": 16,
        "query_policy": "fixed_shared",
    }
    post["coherence"]["families"] = {
        "multiscale_spectrum": {
            "enabled": True,
            "weight": 1.0,
            "target_use": "paired_supervised",
            "units": "model_units",
            "fields": ["u", "v"],
            "reference_bank": {"enabled": False},
            # Default raw_extent scaling: the fixture's raw grid spacing is one unit.
            "geometry": {"operator": "fem_p1"},
            "bands": {"count": 2, "min_wavelength": 1.5},
            "calibration": {"samples": 8, "seed": 5},
            "objective": {"kind": "conditional_energy_score", "conditional_draws": 2},
            "components": {
                "log_band_power": {"enabled": True, "weight": 1.0},
                "signed_band_cospectrum": {"enabled": True, "weight": 1.0},
                "third_order_product_moments": _third([[["u", 0], ["v", 1], ["v", 1]]]),
            },
        }
    }
    validate_config(post)
    child = run_post_training(post, case_dir=case_dir, max_steps=1)
    artifact = torch.load(child / "artifacts" / "multiscale_spectrum_family.pt", weights_only=True)
    assert artifact["calibration"]["split"] == "train"
    assert artifact["calibration"]["states"] == 2
    assert artifact["geometry"]["points"] == 16
    assert artifact["geometry"]["active_axis_lengths"] == [3.0, 3.0]
    before = json.loads((child / "evaluation" / "before.json").read_text())
    payload = before["coherence"]["families"]["multiscale_spectrum"]
    energy_path = "multiscale_spectrum.descriptor_energy_score.conditional"
    assert payload["components"][energy_path]["inner_weight"] == 1.0
    assert (
        payload["components"]["multiscale_spectrum.log_band_power.squared_error"]["inner_weight"]
        == 0.0
    )
    assert payload["diagnostics"]["report"]["reference"]["band_power_mean"]
    history = json.loads((child / "logs" / "history.jsonl").read_text().splitlines()[0])
    assert math.isfinite(history[energy_path])
    assert "family_gradient_diagnostics" in history
    resumed = run_post_training(post, case_dir=case_dir, resume=child, max_steps=1)
    assert resumed == child
    restored = torch.load(child / "artifacts" / "multiscale_spectrum_family.pt", weights_only=True)
    assert restored["calibration"]["sample_ids"] == artifact["calibration"]["sample_ids"]

    from phycoflow_reconstruction.training.posttrain_rescore import restore_evaluation_panel

    panel = restore_evaluation_panel(child, evaluation_overrides={}, device_name="cpu")
    panel.load_checkpoint("last")
    restored_metrics = panel.score()
    assert energy_path in restored_metrics["coherence"]["components"]
    calibration_path = child / "artifacts/coherence_calibration.json"
    original_calibration = calibration_path.read_bytes()
    calibration_path.write_text("{}")
    try:
        with pytest.raises(ValueError, match="calibration differs"):
            restore_evaluation_panel(child, evaluation_overrides={}, device_name="cpu")
    finally:
        calibration_path.write_bytes(original_calibration)

    mismatch = copy.deepcopy(post)
    mismatch["evaluation"]["query_points"] = 8
    with pytest.raises(ValueError, match="evaluation.query_points"):
        validate_config(mismatch)


def test_reference_calibration_preserves_random_streams():
    import random
    from types import SimpleNamespace

    import numpy as np

    from phycoflow_reconstruction.training.coherence_sampling import _fit_training_references

    class ReferenceConsumer:
        def fit_training_reference(self, dataset, *, query_indices, device):
            self.indices = query_indices.clone()
            random.random()
            np.random.random()
            torch.rand(3)

    config = {
        "coherence": {"compute_budget": {"point_count": 8, "query_seed": 9}},
        "runtime": {},
        "observations": {},
    }
    class Dataset:
        data_spec = SimpleNamespace(logical_shape=(4, 4))

        def __getitem__(self, index):
            return SimpleNamespace(valid_points=None)

    dataset = Dataset()
    family = ReferenceConsumer()
    before = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    _fit_training_references(
        {"test": family},
        dataset,
        config,
        device=torch.device("cpu"),
        structured_grid_required=False,
    )
    assert torch.equal(torch.get_rng_state(), before[0])
    assert random.getstate() == before[1]
    actual = np.random.get_state()
    assert actual[0] == before[2][0] and actual[2:] == before[2][2:]
    np.testing.assert_array_equal(actual[1], before[2][1])
    assert family.indices.numel() == 8


def test_ordinary_topology_keeps_structured_grid_query_subsets():
    from types import SimpleNamespace

    from phycoflow_reconstruction.coherence.families.topology.family import TopologyFamily
    from phycoflow_reconstruction.contracts import DataSpec
    from phycoflow_reconstruction.data.normalization import FieldNormalizer
    from phycoflow_reconstruction.training.coherence_sampling import _fit_training_references

    family = TopologyFamily(
        {"target_use": "paired_supervised", "fields": ["u", "v"]},
        DataSpec(("u", "v"), ("1", "1"), 2, (4, 4)), FieldNormalizer.identity(2),
    )
    config = {"coherence": {"compute_budget": {"point_count": 8}},
              "runtime": {}, "observations": {}}
    class Dataset:
        data_spec = DataSpec(("u", "v"), ("1", "1"), 2, (4, 4))

        def __getitem__(self, index):
            return SimpleNamespace(valid_points=None)

    dataset = Dataset()
    _fit_training_references(
        {"topology": family}, dataset, config, device=torch.device("cpu"),
        structured_grid_required=True,
    )
    assert family.neighbor_indices.numel() == 0  # Ordinary raster fitting stays lazy.
