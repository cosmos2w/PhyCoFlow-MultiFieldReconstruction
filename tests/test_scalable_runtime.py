from types import SimpleNamespace

import pytest
import torch

from phycoflow_reconstruction.training.internal_calibration import BLOCKS
from phycoflow_reconstruction.training.scalable_runtime import ScalableRuntime


def test_runtime_acceptance_calibration_and_resume(monkeypatch):
    import phycoflow_reconstruction.training.scalable_runtime as module

    monkeypatch.setattr(module, "ScalableQueryAdapter", lambda *args: None)
    monkeypatch.setattr(module, "validate_scalable_artifacts", lambda *args: None)
    fields = ("u", "v")
    families = {
        k: SimpleNamespace(field_names=fields)
        for k in ("global_distribution", "multiscale_spectrum", "topology")
    }
    keys = [f"{f}.{b}" for f, blocks in BLOCKS.items() for b in blocks]
    provenance = {
        "split": "train",
        "field_order": fields,
        "sample_ids": ("train:1",),
        "source_checkpoint_sha256": "a" * 64,
        "normalizer_sha256": "b" * 64,
        "support_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "definition_version": "v1",
    }
    artifacts = {
        "sha256": "f" * 64,
        "provenance": provenance,
        "state_bank": {"field_order": fields},
        "shape_artifact": {"provenance": {"split": "train", "source_checkpoint_sha256": "a" * 64}},
        "priorities": {k: 1.0 for k in keys},
        "units": {k: 1.0 for k in keys},
        "active": {k: True for k in keys},
    }
    runtime = ScalableRuntime(
        artifacts, families, steps_per_epoch=1, expected_provenance=provenance
    )
    model = torch.nn.Linear(1, 1, bias=False)

    def fake_score(name, family, prediction, reference, coords, context, batch):
        prefix = {
            "global_distribution": "state",
            "multiscale_spectrum": "scale",
            "topology": "shape",
        }[name]
        blocks = {f"{prefix}.{b}": model.weight.square().sum() for b in BLOCKS[prefix]}
        difficulties = (
            None
            if prefix == "state"
            else {
                "self": {q.index: 1.0 for q in batch.self_queries},
                "pair": {q.index: 1.0 for q in batch.pair_queries},
            }
        )
        return None, blocks, difficulties

    monkeypatch.setattr(runtime, "_score", fake_score)
    runtime.begin_step(0)
    original = runtime.batch
    runtime.finish_step(False)
    runtime.begin_step(0)
    assert runtime.batch == original
    for step in range(10):
        if step:
            runtime.begin_step(step)
        for name, family in families.items():
            runtime.score_family(name, family, None, None, coordinates=None, context={})
        runtime.prepare_probe(
            model,
            None,
            None,
            None,
            {"sample_ids": ("train:1",), "query_indices": torch.tensor([[0, 1]])},
        )
        runtime.finish_step(True)
    assert runtime.calibration.coefficients is not None
    assert [r["epoch"] for r in runtime.calibration.records] == [1, 3, 6, 8, 10]
    state = runtime.state_dict()
    restored = ScalableRuntime(
        artifacts, families, steps_per_epoch=1, expected_provenance=provenance
    )
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    from copy import deepcopy

    broken = deepcopy(state)
    broken["sampler"]["counter"] -= 1
    with pytest.raises(ValueError, match="cursor"):
        restored.load_state_dict(broken)
    broken = deepcopy(state)
    broken["sampler"]["epsilon"] = 0.1
    with pytest.raises(ValueError, match="settings"):
        restored.load_state_dict(broken)
    with pytest.raises(ValueError, match="provenance"):
        ScalableRuntime(
            artifacts,
            families,
            steps_per_epoch=1,
            expected_provenance={**provenance, "support_sha256": "other"},
        )
