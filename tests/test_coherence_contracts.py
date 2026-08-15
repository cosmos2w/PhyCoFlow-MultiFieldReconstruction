"""Phase-5 tests cover taxonomy, gradients, leakage, and child-run lineage."""

from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

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


def _write_fixture(path: Path) -> None:
    generator = np.random.default_rng(7)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "fields", data=generator.normal(size=(3, 2, 16, 1, 1, 2)).astype("float32")
        )
        y, x = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
        coordinates = np.stack((x, y, np.zeros_like(x)), axis=-1).reshape(16, 1, 1, 3)
        handle.create_dataset("coordinates", data=coordinates.astype("float32"))
        handle.create_dataset("time", data=np.asarray([0.0, 1.0]))
        handle.create_dataset("conditions", data=np.empty((3, 0), dtype="float32"))
        handle.create_dataset(
            "trajectory_id",
            data=np.asarray(["train", "validation", "test"], dtype=h5py.string_dtype()),
        )
        splits = handle.create_group("splits")
        splits.create_dataset("train", data=np.asarray([0]))
        splits.create_dataset("validation", data=np.asarray([1]))
        splits.create_dataset("test", data=np.asarray([2]))
        statistics = handle.create_group("statistics")
        statistics.create_dataset("train_mean", data=np.zeros(2))
        statistics.create_dataset("train_std", data=np.ones(2))
        handle.attrs["field_names"] = '["u", "v"]'
        handle.attrs["field_units"] = '["1", "1"]'
        handle.attrs["grid_shape"] = "[4, 4]"
        handle.attrs["schema_version"] = "1.0"


def _family_config(target_use: str = "training_reference") -> dict:
    return {
        "enabled": True,
        "target_use": target_use,
        "units": "model_units",
        "fields": ["u", "v"],
        "reference_bank": {
            "enabled": target_use == "training_reference",
            "max_samples": 2,
            "points_per_sample": 8,
            "seed": 13,
        },
        "components": {
            "self": {"enabled": True, "weight": 1.0},
            "mutual": {
                "enabled": True,
                "weight": 1.0,
                "pairs": [["u", "v"]],
                "directions": 4,
                "seed": 3,
            },
            "cross": {
                "enabled": True,
                "weight": 1.0,
                "directions": 6,
                "top_fraction": 0.5,
                "seed": 5,
                "include_axes": True,
                "qmc": True,
            },
        },
    }


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


def _base_config(dataset_path: Path) -> dict:
    return {
        "stage": "base_training",
        "case": "fixture",
        "dataset": {
            "path": str(dataset_path),
            "split": "train",
            "field_names": ["u", "v"],
            "field_units": ["1", "1"],
            "normalization": "mean_std",
        },
        "model": {
            "name": "pointcloud_ffm",
            "backbone": "gl_rbf_enh",
            "gather_mode": "topk_rbf",
            "prior": "iid",
            "hidden_dim": 16,
            "latent_dim": 16,
            "num_latents": 4,
            "heads": 2,
            "latent_blocks": 1,
            "gather_topk": 2,
            "query_chunk_size": 16,
            "query_points": 8,
        },
        "observations": {"protocol": "random_uniform", "seed": 4, "fields": {"u": {"count": 4}}},
        "optimization": {"epochs": 1, "batch_size": 2, "lr": 1e-3, "grad_clip": 1.0},
        "runtime": {"seed": 9, "device": "cpu", "deterministic": True, "num_workers": 0},
        "evaluation": {"generation_steps": 1},
        "output": {"experiment_name": "source"},
    }


def _post_config(dataset_path: Path, source_run: Path) -> dict:
    config = _base_config(dataset_path)
    config.update(
        stage="post_training",
        source_run=str(source_run),
        source_checkpoint="last.pt",
        inherit_base_config=True,
        source={
            "kind": "native_run",
            "allow_integration_source": False,
            "inherited_base_keys": ["dataset", "model", "observations"],
            "config_origins": {"post_training": "test"},
        },
        objectives={
            "data_retention": {"enabled": True, "weight": 0.1},
            "coherence": {"enabled": True, "weight": 1.0},
        },
        coherence={
            "schedule": {
                "start_epoch": 1,
                "every_n_steps": 1,
                "weight_warmup_epochs": 0,
                "interval_rescale": False,
            },
            "compute_budget": {"batch_size": 1, "point_count": 8},
            "families": {"global_distribution": _family_config()},
        },
        rollout={"steps": 1, "solver": "euler"},
        observation_consistency={
            "mode": "endpoint_smooth",
            "strength": 1.0,
            "sigma": 0.2,
            "schedule_power": 2.0,
            "final_clamp": True,
        },
        trainable={"scope": "full_model"},
        evaluation={
            "split": "validation",
            "max_samples": 1,
            "query_points": 8,
            "generation_steps": 1,
            "seed": 77,
        },
        output={"experiment_name": "child"},
    )
    config["optimization"].update(
        train_fraction=1.0,
        weight_decay=0.0,
        gradient_balance="weighted_sum",
        config_missing_behavior="error",
    )
    return config


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
