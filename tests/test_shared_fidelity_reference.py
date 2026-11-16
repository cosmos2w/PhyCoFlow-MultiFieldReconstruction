import json
from types import SimpleNamespace

import pytest

from phycoflow_reconstruction.training import post_training as trainer
from phycoflow_reconstruction.training.fidelity_budget import FidelityBudget
from phycoflow_reconstruction.training.run_store import file_sha256


def test_shared_reference_binds_source_support_solver_and_core(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "dataset_fingerprint", lambda _: "dataset")
    dataset = SimpleNamespace(path="data", field_names=("a",), normalizer=SimpleNamespace(digest=lambda: "normalizer"))
    artifact = {"provenance": {"support_sha256": "support", "sample_ids": ["x"]}, "state_bank": {"direction": [1]}, "shape_artifact": {"provenance": {"config_sha256": "control"}, "scales": [1]}, "priorities": {}, "units": {}, "active": {}}
    runtime = SimpleNamespace(artifacts=artifact)
    cfg = {"rollout": {"steps": 4, "solver": "euler"}, "observations": {}, "observation_consistency": {}, "coherence": {"compute_budget": {"point_count": 4}, "families": {}}, "optimization": {"multitask": {"fidelity_budget": {"metric": "physical_member_mse_model_units"}}}}
    source_hashes = {"checkpoint": "checkpoint"}
    metadata = {"weight_view": "ema_materialized", "materialized_view_sha256": "view"}
    def binding():
        return trainer._fidelity_reference_binding(cfg, dataset, source_hashes, metadata, runtime, {})
    expected = binding()
    budget = FidelityBudget.from_reference(["a"], [1.], 2.)
    path = tmp_path / "reference.json"
    path.write_text(json.dumps({"binding": expected, "reference": budget.state_dict()}))
    declaration = {"path": str(path), "sha256": file_sha256(path)}
    loaded, _ = trainer._load_shared_fidelity_reference(declaration, expected)
    assert loaded == budget
    # Candidate config identity may change; actual common core values may not.
    artifact["shape_artifact"]["provenance"]["config_sha256"] = "candidate"
    assert binding() == expected
    cfg["rollout"]["steps"] = 2
    with pytest.raises(ValueError, match="binding mismatch"):
        trainer._load_shared_fidelity_reference(declaration, binding())
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        trainer._load_shared_fidelity_reference(declaration, expected)
