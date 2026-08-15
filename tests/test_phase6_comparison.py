"""Phase-6 reports reject unmatched protocols and summarize valid child pairs."""

import hashlib
import json

import pytest
import torch
import yaml

from phycoflow_reconstruction.data.manifest import SensorManifest
from phycoflow_reconstruction.evaluation.posttrain_comparison import (
    build_phase6_comparison,
    write_phase6_comparison,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture_matrix(tmp_path):
    source = tmp_path / "base"
    child = tmp_path / "child"
    source.mkdir()
    source_checkpoint = source / "last.pt"
    source_checkpoint.write_bytes(b"source")
    source_hash = hashlib.sha256(b"source").hexdigest()
    child_checkpoint = child / "checkpoints" / "last.pt"
    child_checkpoint.parent.mkdir(parents=True)
    child_checkpoint.write_bytes(b"child")
    child_hash = hashlib.sha256(b"child").hexdigest()
    sensor_manifest = SensorManifest(
        dataset_path="fixture.h5",
        dataset_fingerprint="dataset",
        split="validation",
        protocol={"seed": 42},
        indices={"sample": [[0, 0]]},
    )
    sensor_hash = sensor_manifest.digest()
    manifest = {
        "parent_run": str(source),
        "source_immutable_verified": True,
        "source_checkpoint": str(source_checkpoint),
        "source_hashes": {"checkpoint": source_hash},
        "source_hashes_after": {"checkpoint": source_hash},
        "checkpoint_hashes": {"last": child_hash},
        "dataset_fingerprint": "dataset",
        "reference_bank_sha256": "bank",
        "evaluation_sensor_manifest_sha256": sensor_hash,
        "parameter_count": 12,
        "trainable_parameter_count": 12,
        "differentiable_adapter": "native_reconstruction",
    }
    status = {
        "status": "integration_truncated",
        "post_training_seconds": 0.2,
        "seconds_per_step": 0.2,
        "peak_cuda_memory_bytes": 1024,
    }
    config = {
        "model": {"name": "coordinate_mlp"},
        "runtime": {"seed": 42},
        "evaluation": {"seed": 2027, "split": "validation"},
    }
    evaluation = {
        "mse_normalized": 2.0,
        "sensor_manifest_sha256": sensor_hash,
        "coherence": {"total": 3.0, "target_use": "training_reference"},
        "inference": {
            "seconds": 0.01,
            "samples": 1,
            "points_per_sample": 8,
            "generation_steps": 1,
        },
    }
    _write_json(child / "run_manifest.json", manifest)
    _write_json(child / "status.json", status)
    _write_json(
        child / "environment.json",
        {"cuda_visible_devices": None, "cuda_devices": []},
    )
    (child / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    _write_json(child / "evaluation" / "before.json", evaluation)
    after = {
        **evaluation,
        "mse_normalized": 1.5,
        "coherence": {**evaluation["coherence"], "total": 2.0},
    }
    _write_json(child / "evaluation" / "after.json", after)
    history = {
        "data_grad_norm": 1.0,
        "coherence_grad_norm": 2.0,
        "update_mode": "weighted_sum",
    }
    history_path = child / "metrics" / "history.jsonl"
    history_path.parent.mkdir(parents=True)
    history_path.write_text(json.dumps(history) + "\n", encoding="utf-8")
    artifact = child / "artifacts" / "evaluation_query_indices.pt"
    artifact.parent.mkdir(parents=True)
    torch.save({"query_indices": torch.arange(8).unsqueeze(0)}, artifact)
    sensor_manifest.save(child / "artifacts" / "evaluation_sensor_manifest.json")
    matrix = {
        "case": "tiny",
        "comparison_id": "phase6-test",
        "entries": [
            {
                "id": "coordinate_mlp",
                "family": "deterministic_point",
                "source_run": str(source),
                "post_run": str(child),
            }
        ],
        "unsupported": [
            {
                "id": "pinn",
                "reason_code": "no_plain_base_run",
                "reason": "PINN belongs to direct physics training.",
            }
        ],
    }
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(yaml.safe_dump(matrix), encoding="utf-8")
    return matrix_path, child


def test_phase6_comparison_validates_and_writes(tmp_path):
    matrix, _ = _fixture_matrix(tmp_path)
    payload = write_phase6_comparison(matrix, tmp_path / "summary.json", tmp_path / "summary.md")
    assert payload["matched_pairs"][0]["mse_normalized"]["absolute_reduction"] == 0.5
    assert payload["shared_protocol"]["sensor_manifest_sha256"]
    assert (tmp_path / "summary.md").read_text().startswith("# Phase 6 comparison")


def test_phase6_comparison_rejects_source_mutation(tmp_path):
    matrix, child = _fixture_matrix(tmp_path)
    manifest = json.loads((child / "run_manifest.json").read_text())
    manifest["source_hashes_after"] = {"checkpoint": "changed"}
    _write_json(child / "run_manifest.json", manifest)
    with pytest.raises(ValueError, match="source hashes changed"):
        build_phase6_comparison(matrix)
