import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from phycoflow_reconstruction.contracts import DataSpec, FieldSample, ReconstructionBatch
from phycoflow_reconstruction.data.manifest import SensorManifest
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.training.reconstruction_panel import TrainingReconstructionPanel
from phycoflow_reconstruction.training.run_store import RunStore


class _Dataset:
    def __init__(self, path: Path, split: str, normalizer: FieldNormalizer):
        self.path = path
        self.split_name = split
        self.normalizer = normalizer
        self.field_names = ("u", "v")
        self.data_spec = DataSpec(
            self.field_names,
            ("unit", "unit"),
            2,
            (4, 4),
            mesh_type="structured",
        )
        y, x = torch.meshgrid(
            torch.linspace(0, 1, 4), torch.linspace(0, 1, 4), indexing="ij"
        )
        coords = torch.stack((x, y), -1).reshape(-1, 2)
        self.samples = [
            FieldSample(
                values=torch.stack((x + index, y - index), -1).reshape(-1, 2),
                coordinates=coords,
                coordinates_raw=coords,
                time=torch.tensor(float(index)),
                trajectory_id=f"trajectory_{index}",
                time_index=index,
                conditions=torch.empty(0),
                field_names=self.field_names,
                logical_shape=(4, 4),
            )
            for index in range(3)
        ]
        self.closed = False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def close(self):
        self.closed = True


class _RandomReconstructionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.target_fields_seen = []

    def reconstruct(self, batch, *, steps, generator):
        assert steps == 3
        self.target_fields_seen.append(batch.target_fields)
        prediction = torch.rand(
            batch.query_coords.shape[0],
            batch.query_coords.shape[1],
            2,
            device=batch.query_coords.device,
            generator=generator,
        )
        return ReconstructionBatch(prediction)


def _fixture(monkeypatch, tmp_path, *, sample_indices=(0, 2)):
    data_path = tmp_path / "validation.pt"
    data_path.write_bytes(b"small fixed dataset identity")
    train_normalizer = FieldNormalizer(
        torch.tensor([1.0, -2.0]), torch.tensor([2.0, 0.5])
    )
    opened = []

    def open_dataset(_config, *, split, normalizer=None):
        opened.append((split, normalizer))
        return _Dataset(data_path, split, normalizer or train_normalizer)

    # Keep the train statistics object explicit so the validation-open assertion
    # detects accidental use of validation-derived statistics.
    monkeypatch.setattr(
        "phycoflow_reconstruction.training.reconstruction_panel.open_field_dataset",
        open_dataset,
    )
    config = {
        "stage": "base_training",
        "case": "fixture",
        "dataset": {"path": str(data_path), "normalization": "mean_std"},
        "observations": {
            "protocol": "random_uniform",
            "seed": 31,
            "fields": {"u": {"count": 3}, "v": {"count": 4}},
        },
        "runtime": {"seed": 13},
        "model": {"model_ema_enabled": True, "model_ema_eval": True},
        "evaluation": {
            "reconstruction_panel": {
                "sample_indices": list(sample_indices),
                "seeds": [5, 11],
                "generation_steps": 3,
            }
        },
        "output": {"experiment_name": "panel_fixture"},
    }
    store = RunStore.create(tmp_path, "panel_fixture", config)
    return config, store, train_normalizer, opened


def test_fixed_panel_scores_multiple_frames_and_seeds_without_target_leakage(
    monkeypatch, tmp_path
):
    config, store, normalizer, opened = _fixture(monkeypatch, tmp_path)
    model = _RandomReconstructionModel()
    model.train()
    panel = TrainingReconstructionPanel(
        config,
        model=model,
        store=store,
        steps_per_epoch=4,
        device=torch.device("cpu"),
    )

    torch.manual_seed(919)
    expected_next_rng = torch.rand(())
    torch.manual_seed(919)
    report = panel(global_step=8)
    actual_next_rng = torch.rand(())

    assert report is not None
    assert report["split"] == "validation"
    assert report["sample_count"] == 2
    assert report["seed_count"] == 2
    assert report["example_seed_count"] == 4
    assert [row["sample_index"] for row in report["examples"]] == [0, 0, 2, 2]
    assert [row["seed"] for row in report["examples"]] == [5, 11, 5, 11]
    assert all(value is None for value in model.target_fields_seen)
    assert model.training
    assert torch.equal(actual_next_rng, expected_next_rng)
    assert opened[0] == ("train", None)
    assert opened[1][0] == "validation" and opened[1][1] is normalizer
    assert opened[2][0] == "validation" and opened[2][1] is normalizer
    assert report["metrics"]["mse_normalized"] == pytest.approx(
        sum(row["metrics"]["mse_normalized"] for row in report["examples"]) / 4
    )
    assert "mse_physical" in report["metrics"]

    output_dir = store.run_dir / "evaluation" / "reconstruction_panel"
    history = [
        json.loads(line)
        for line in (output_dir / "history.jsonl").read_text().splitlines()
    ]
    example_history = [
        json.loads(line)
        for line in (output_dir / "examples.jsonl").read_text().splitlines()
    ]
    assert len(history) == 1 and history[0]["global_step"] == 8
    assert len(example_history) == 4
    assert (output_dir / "reconstruction_mse.png").read_bytes().startswith(b"\x89PNG")
    assert (output_dir / "reconstruction_mse.pdf").read_bytes().startswith(b"%PDF")
    manifest = json.loads((output_dir / "panel.json").read_text())
    assert manifest["sensor_seed_policy"].endswith("sample_index")
    assert len(manifest["sensor_manifests"]) == 2
    assert manifest["weight_policy"] == {
        "context": "configured_evaluation_weight_context",
        "model_ema_enabled": True,
        "model_ema_eval": True,
    }
    assert "weight_source" not in manifest


def test_fixed_panel_history_upserts_repeated_step_and_keeps_prior_steps(
    monkeypatch, tmp_path
):
    config, store, _, _ = _fixture(monkeypatch, tmp_path, sample_indices=(1,))
    panel = TrainingReconstructionPanel(
        config,
        model=_RandomReconstructionModel(),
        store=store,
        steps_per_epoch=2,
        device=torch.device("cpu"),
    )
    panel(global_step=4)
    panel(global_step=4)
    panel(global_step=8)

    output_dir = store.run_dir / "evaluation" / "reconstruction_panel"
    history = [
        json.loads(line)
        for line in (output_dir / "history.jsonl").read_text().splitlines()
    ]
    examples = [
        json.loads(line)
        for line in (output_dir / "examples.jsonl").read_text().splitlines()
    ]
    assert [row["global_step"] for row in history] == [4, 8]
    assert len(examples) == 4
    assert {row["global_step"] for row in examples} == {4, 8}


def test_fixed_panel_rejects_indices_outside_validation_split(monkeypatch, tmp_path):
    config, store, _, _ = _fixture(monkeypatch, tmp_path, sample_indices=(4,))
    with pytest.raises(IndexError, match="outside validation split"):
        TrainingReconstructionPanel(
            config,
            model=_RandomReconstructionModel(),
            store=store,
            steps_per_epoch=2,
            device=torch.device("cpu"),
        )


def test_weight_source_tracks_lazily_initialized_ema(monkeypatch, tmp_path):
    config, store, _, _ = _fixture(monkeypatch, tmp_path, sample_indices=(1,))
    model = _RandomReconstructionModel()
    model._ema = None
    model._ema_eval = True
    panel = TrainingReconstructionPanel(
        config,
        model=model,
        store=store,
        steps_per_epoch=2,
        device=torch.device("cpu"),
    )

    live_report = panel(global_step=2)
    model._ema = object()
    ema_report = panel(global_step=4)

    assert live_report["weight_source"] == "live"
    assert ema_report["weight_source"] == "ema"
    assert {row["weight_source"] for row in ema_report["examples"]} == {"ema"}
    history = [
        json.loads(line)
        for line in (
            store.run_dir
            / "evaluation"
            / "reconstruction_panel"
            / "history.jsonl"
        ).read_text().splitlines()
    ]
    assert [row["weight_source"] for row in history] == ["live", "ema"]
    manifest = json.loads(
        (
            store.run_dir
            / "evaluation"
            / "reconstruction_panel"
            / "panel.json"
        ).read_text()
    )
    assert manifest["weight_policy"]["model_ema_eval"] is True


def test_fixed_panel_rejects_changed_existing_sensor_manifest(monkeypatch, tmp_path):
    config, store, _, _ = _fixture(monkeypatch, tmp_path, sample_indices=(1,))
    panel = TrainingReconstructionPanel(
        config,
        model=_RandomReconstructionModel(),
        store=store,
        steps_per_epoch=2,
        device=torch.device("cpu"),
    )
    manifest_path = panel.manifest_paths[1]
    existing = SensorManifest.load(manifest_path)
    changed_protocol = dict(existing.protocol)
    changed_protocol["seed"] += 1
    replace(existing, protocol=changed_protocol).save(manifest_path)

    with pytest.raises(ValueError, match="sensor manifest differs"):
        TrainingReconstructionPanel(
            config,
            model=_RandomReconstructionModel(),
            store=store,
            steps_per_epoch=2,
            device=torch.device("cpu"),
        )
