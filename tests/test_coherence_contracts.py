"""Phase-5 tests cover taxonomy, gradients, leakage, and child-run lineage."""

from __future__ import annotations

import json
import math

import pytest
import torch
from helpers.coherence import _base_config, _family_config, _post_config, _write_fixture

from phycoflow_reconstruction.coherence import (
    ReferenceBank,
    build_coherence_family,
    fit_reference_bank,
)
from phycoflow_reconstruction.config.validate import validate_config
from phycoflow_reconstruction.contracts import DataSpec, ModelCapabilities, ObservationBatch
from phycoflow_reconstruction.data.h5_dataset import H5FieldDataset
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.training.base_training import run_base_training
from phycoflow_reconstruction.training.gradient_balance import two_objective_update
from phycoflow_reconstruction.training.post_training import (
    _coherence_objective,
    _coherence_weight,
    run_post_training,
)
from phycoflow_reconstruction.training.rollout import differentiable_rf_rollout
from phycoflow_reconstruction.training.run_store import file_sha256


def test_global_distribution_components_are_nested_deterministic_and_differentiable():
    spec = DataSpec(("u", "v"), ("1", "1"), 2, (4, 4), mesh_type="structured")
    normalizer = FieldNormalizer.identity(2)
    first = build_coherence_family("global_distribution", _family_config(), spec, normalizer)
    second = build_coherence_family("global_distribution", _family_config(), spec, normalizer)
    assert torch.equal(
        first.state_dict()["components_by_key.cross_joint_topk_swd.directions"],
        second.state_dict()["components_by_key.cross_joint_topk_swd.directions"],
    )

    generated = torch.randn(2, 16, 2, requires_grad=True)
    reference = torch.randn(2, 16, 2)
    result = first(generated, reference)
    assert set(result.component_results) == {
        "global_distribution.self.marginal_w2",
        "global_distribution.mutual.pairwise_swd",
        "global_distribution.cross.joint_topk_swd",
    }
    for component in result.component_results.values():
        gradient = torch.autograd.grad(component.scalar_loss, generated, retain_graph=True)[0]
        assert torch.isfinite(gradient).all()
        assert torch.linalg.vector_norm(gradient) > 0
    result.scalar_loss.backward()
    assert generated.grad is not None
    assert torch.isfinite(generated.grad).all()
    assert torch.linalg.vector_norm(generated.grad) > 0


def test_reference_bank_refuses_nontraining_data_and_serializes(tmp_path):
    dataset_path = tmp_path / "fixture.h5"
    _write_fixture(dataset_path)
    validation = H5FieldDataset(dataset_path, split="validation")
    with pytest.raises(ValueError, match="training split"):
        fit_reference_bank(validation, max_samples=1, points_per_sample=8, seed=1)
    validation.close()

    training = H5FieldDataset(dataset_path, split="train")
    bank = fit_reference_bank(training, max_samples=2, points_per_sample=8, seed=1)
    path = bank.save(tmp_path / "bank.pt")
    loaded = type(bank).load(path)
    assert loaded.metadata["split"] == "train"
    assert loaded.digest() == bank.digest()
    training.close()


def test_optional_config_gradient_update_is_callable():
    pytest.importorskip("conflictfree")
    model = torch.nn.Linear(1, 2, bias=False)
    parameter = next(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    data_loss = parameter[0, 0]
    coherence_loss = -0.5 * parameter[0, 0] + parameter[1, 0]
    diagnostics = two_objective_update(
        model,
        optimizer,
        data_loss,
        coherence_loss,
        mode="config",
        data_weight=1.0,
        coherence_weight=1.0,
        grad_clip=None,
    )
    assert diagnostics["gradient_conflict"] is True
    assert diagnostics["update_mode"] in {"config", "weighted_sum_nondescent_config"}
    assert math.isfinite(diagnostics["combined_grad_norm"])


def test_gradient_update_supports_mixed_real_and_complex_parameters():
    class SpectralToy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.real = torch.nn.Parameter(torch.tensor([0.5]))
            self.spectral = torch.nn.Parameter(torch.tensor([0.25 + 0.5j]))

    model = SpectralToy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    data_loss = model.real.square().sum() + model.spectral.abs().square().sum()
    coherence_loss = (model.real - model.spectral.real).square().sum()
    diagnostics = two_objective_update(
        model,
        optimizer,
        data_loss,
        coherence_loss,
        mode="weighted_sum",
        data_weight=1.0,
        coherence_weight=1.0,
        grad_clip=None,
    )
    assert model.spectral.grad is not None
    assert torch.is_complex(model.spectral.grad)
    assert math.isfinite(diagnostics["combined_grad_norm"])


def test_training_reference_rollout_cannot_receive_paired_target():
    class GuardedFlow(torch.nn.Module):
        capabilities = ModelCapabilities("point", True, True, False, True)

        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.2))

        def sample_source(self, batch, *, generator=None):
            assert batch.target_fields is None
            return torch.zeros(1, 8, 2)

        def velocity(self, batch, state, time):
            assert batch.target_fields is None
            return self.scale * torch.ones_like(state)

    spec = DataSpec(("u", "v"), ("1", "1"), 2, (8,))
    family = build_coherence_family(
        "global_distribution", _family_config(), spec, FieldNormalizer.identity(2)
    )
    batch = ObservationBatch(
        obs_coords=torch.zeros(1, 2, 2),
        obs_values=torch.zeros(1, 2, 1),
        obs_field_ids=torch.tensor([[0, 1]]),
        obs_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        query_coords=torch.rand(1, 8, 2),
        query_valid_mask=torch.ones(1, 8, dtype=torch.bool),
        target_fields=torch.full((1, 8, 2), float("nan")),
        sample_ids=("paired-target-must-not-be-read",),
        obs_indices=torch.tensor([[0, 1]]),
        logical_shapes=((8,),),
        metadata={"query_indices": torch.arange(8).view(1, 8)},
    )
    bank = ReferenceBank(
        values=torch.randn(1, 8, 2),
        sample_ids=("training-reference",),
        point_indices=torch.arange(8).view(1, 8),
        metadata={"split": "train"},
    )
    result, reference_ids = _coherence_objective(
        GuardedFlow(),
        batch,
        family,
        bank,
        {
            "coherence": {"compute_budget": {"batch_size": 1, "point_count": 8}},
            "rollout": {"steps": 1, "solver": "euler"},
            "observation_consistency": {"mode": "none", "final_clamp": False},
        },
        step=0,
        generator=torch.Generator().manual_seed(2),
    )
    assert reference_ids == ("training-reference",)
    assert torch.isfinite(result.scalar_loss)


def test_rollout_solvers_and_coherence_warmup_are_explicit():
    class LinearFlow(torch.nn.Module):
        capabilities = ModelCapabilities("point", True, True, False, True)

        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.25))

        def sample_source(self, batch, *, generator=None):
            return torch.zeros(1, 4, 1)

        def velocity(self, batch, state, time):
            return self.scale * torch.ones_like(state)

    batch = ObservationBatch(
        obs_coords=torch.zeros(1, 1, 1),
        obs_values=torch.zeros(1, 1, 1),
        obs_field_ids=torch.zeros(1, 1, dtype=torch.long),
        obs_valid_mask=torch.ones(1, 1, dtype=torch.bool),
        query_coords=torch.arange(4).view(1, 4, 1).float(),
        query_valid_mask=torch.ones(1, 4, dtype=torch.bool),
        target_fields=None,
        sample_ids=("x",),
    )
    for solver in ("euler", "heun"):
        model = LinearFlow()
        endpoint = differentiable_rf_rollout(
            model,
            batch,
            steps=2,
            solver=solver,
            generator=torch.Generator().manual_seed(1),
            observation_config={"mode": "none", "final_clamp": False},
        )
        endpoint.sum().backward()
        assert model.scale.grad is not None and torch.isfinite(model.scale.grad)

    config = {
        "objectives": {"coherence": {"enabled": True, "weight": 2.0}},
        "coherence": {"schedule": {"start_epoch": 3, "weight_warmup_epochs": 4}},
    }
    assert _coherence_weight(config, 2) == 0.0
    assert _coherence_weight(config, 3) == 0.5
    assert _coherence_weight(config, 4) == 1.0
    assert _coherence_weight(config, 6) == 2.0


def test_target_free_posttraining_writes_child_and_preserves_source(tmp_path):
    dataset_path = tmp_path / "fixture.h5"
    _write_fixture(dataset_path)
    case_dir = tmp_path / "case"
    base = _base_config(dataset_path)
    source_run = run_base_training(base, case_dir=case_dir)
    source_checkpoint = source_run / "checkpoints" / "last.pt"
    source_hash = file_sha256(source_checkpoint)

    post = _post_config(dataset_path, source_run)
    validate_config(post)
    child = run_post_training(post, case_dir=case_dir)
    manifest = json.loads((child / "run_manifest.json").read_text())
    before = json.loads((child / "evaluation" / "before.json").read_text())
    after = json.loads((child / "evaluation" / "after.json").read_text())
    history = json.loads((child / "metrics" / "history.jsonl").read_text())
    assert manifest["parent_run"] == str(source_run)
    assert manifest["source_immutable_verified"] is True
    assert file_sha256(source_checkpoint) == source_hash
    assert before["sensor_manifest_sha256"] == after["sensor_manifest_sha256"]
    assert before["coherence"]["target_use"] == "training_reference"
    assert history["coherence_reference_ids"]
    assert (child / "artifacts" / "coherence_reference.pt").is_file()
    evaluation_manifest = json.loads(
        (child / "artifacts" / "evaluation_sensor_manifest.json").read_text()
    )
    assert len(next(iter(evaluation_manifest["query_indices"].values()))) == 8


def test_initial_family_calibration_is_fixed_hashed_and_restored_on_resume(tmp_path):
    dataset_path = tmp_path / "fixture.h5"
    _write_fixture(dataset_path)
    case_dir = tmp_path / "case"
    source_run = run_base_training(_base_config(dataset_path), case_dir=case_dir)
    post = _post_config(dataset_path, source_run)
    post["optimization"]["epochs"] = 3
    post["coherence"]["family_balance"] = {
        "mode": "initial_grad_norm",
        "calibration_batches": 2,
        "reference": "median",
        "epsilon": 1.0e-12,
        "scale_min": 0.01,
        "scale_max": 100.0,
        "max_batch_ratio": 100.0,
        "seed": 811,
    }
    validate_config(post)

    child = run_post_training(post, case_dir=case_dir, max_steps=1)
    calibration_path = child / "artifacts" / "coherence_calibration.json"
    calibration_hash = file_sha256(calibration_path)
    calibration = json.loads(calibration_path.read_text())
    first_checkpoint = torch.load(
        child / "checkpoints" / "last.pt", map_location="cpu", weights_only=True
    )

    assert calibration["mode"] == "initial_grad_norm"
    assert calibration["resolved_scales"] == {"global_distribution": 1.0}
    assert len(calibration["calibration_batches"]) == 2
    assert first_checkpoint["coherence_calibration_sha256"] == calibration_hash
    assert first_checkpoint["family_scales"] == calibration["resolved_scales"]

    none_post = _post_config(dataset_path, source_run)
    none_post["optimization"]["epochs"] = 3
    none_post["output"]["experiment_name"] = "child_none"
    none_child = run_post_training(none_post, case_dir=case_dir, max_steps=1)
    calibrated_history = json.loads((child / "metrics" / "history.jsonl").read_text())
    none_history = json.loads((none_child / "metrics" / "history.jsonl").read_text())
    assert calibrated_history["native_data_loss"] == none_history["native_data_loss"]

    resumed = run_post_training(post, case_dir=case_dir, resume=child, max_steps=1)
    resumed_checkpoint = torch.load(
        resumed / "checkpoints" / "last.pt", map_location="cpu", weights_only=True
    )
    assert resumed == child
    assert file_sha256(calibration_path) == calibration_hash
    assert resumed_checkpoint["global_step"] == 2
    assert resumed_checkpoint["family_scales"] == calibration["resolved_scales"]
