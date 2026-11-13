from types import SimpleNamespace

import pytest
import torch

from phycoflow_reconstruction.training.internal_calibration import BLOCKS
from phycoflow_reconstruction.training.scalable_runtime import ScalableRuntime


def test_runtime_deploys_and_restores_fixed_profile(monkeypatch):
    import phycoflow_reconstruction.training.scalable_runtime as module
    monkeypatch.setattr(module, "ScalableQueryAdapter", lambda *args: None)
    monkeypatch.setattr(module, "validate_scalable_artifacts", lambda *args: None)
    monkeypatch.setattr(module, "validate_bank", lambda *args, **kwargs: None)
    fields = ("a", "b", "c")
    families = {k: SimpleNamespace(field_names=fields) for k in
                ("global_distribution", "multiscale_spectrum", "topology")}
    keys = [f"{f}.{b}" for f, blocks in BLOCKS.items() for b in blocks]
    provenance = {"split": "train", "field_order": fields, "sample_ids": ("id",),
                  "source_checkpoint_sha256": "a" * 64, "normalizer_sha256": "b" * 64,
                  "support_sha256": "c" * 64, "config_sha256": "d" * 64, "definition_version": "v1"}
    bank = {"sha256": "e" * 64, "population_ids": ["id"], "directions": [], "triples": [],
            "block_units": {"state.group_projection": 1., "scale.third_order": 1.},
            "policy_qualified": True, "policy_mode": "fixed_source_profile",
            "policy_pair_probabilities": [.2, .3, .5], "policy_profile_sha256": "f" * 64}
    artifact = {"sha256": "9" * 64, "provenance": provenance, "adaptive_bank": bank,
                "state_bank": {"field_order": fields},
                "shape_artifact": {"provenance": {"split": "train", "source_checkpoint_sha256": "a" * 64}},
                "priorities": {k: 1. for k in keys}, "units": {k: 1. for k in keys},
                "active": {k: True for k in keys}}
    runtime = ScalableRuntime(artifact, families, steps_per_epoch=1, expected_provenance=provenance)
    restored = ScalableRuntime(artifact, families, steps_per_epoch=1, expected_provenance=provenance)
    restored.load_state_dict(runtime.state_dict())
    assert not runtime.policy_enabled
    assert runtime.sampler.fixed_probabilities == (.2, .3, .5)
    assert runtime.sampler.draw() == restored.sampler.draw()


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


def test_expanded_bank_runtime_calibration_rollback_and_resume(monkeypatch):
    import phycoflow_reconstruction.training.scalable_runtime as module
    from phycoflow_reconstruction.coherence.adaptive_fitting import _digest
    from phycoflow_reconstruction.contracts import FamilyResult

    fields = ("a", "b", "c")
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
    bank = {
        "version": "adaptive_measured_train_bank_v1",
        "policy_qualified": True,
        "policy_mode": "fixed_source_profile",
        "policy_pair_probabilities": [0.2, 0.3, 0.5],
        "policy_profile_sha256": "e" * 64,
        "policy_incremental_pair_costs": [1.0, 1.0, 1.0],
        "split": "TRAIN",
        "field_order": list(fields),
        "source_identity": provenance["source_checkpoint_sha256"],
        "support_identity": provenance["support_sha256"],
        "population_ids": ["train:1"],
        "physical_masses": [0.25] * 4,
        "directions": [[1.0, -1.0, 0.0]],
        "triples": [[0, [[0, 0], [0, 1], [0, 2]]]],
        "region_masses": [[0.25] * 4],
        "triple_scales": [[[1.0, 1.0, 1.0]]],
        "target_weights": {"state.group_projection": [1.0], "scale.third_order": [1.0]},
        "block_units": {"state.group_projection": 1.0, "scale.third_order": 1.0},
    }
    bank["sha256"] = _digest(bank)
    keys = [f"{f}.{b}" for f, blocks in BLOCKS.items() for b in blocks]
    artifacts = {
        "sha256": "f" * 64,
        "provenance": provenance,
        "state_bank": {"field_order": fields},
        "shape_artifact": {"provenance": {"split": "train", "source_checkpoint_sha256": "a" * 64}},
        "priorities": {k: 1.0 for k in keys},
        "units": {k: 1.0 for k in keys},
        "active": {k: True for k in keys},
        "adaptive_bank": bank,
    }

    class State:
        field_names = fields

        def _endpoint_measures(self, prediction, reference, context, required):
            return torch.ones(1, 4), torch.ones(1, 4)

    class FakeAdapter:
        def __init__(self, *args):
            pass

        def score(self, name, prediction, reference, *, coordinates, context, batch, coefficients):
            prefix = "scale" if name == "multiscale_spectrum" else "shape"
            core = prediction.square().mean()
            result = FamilyResult({}, core.expand(len(reference)), core, {})
            blocks = {f"{prefix}.{b}": core for b in BLOCKS[prefix]}
            difficulties = {
                "self": {q.index: 1.0 for q in batch.self_queries},
                "pair": {q.index: 1.0 for q in batch.pair_queries},
            }
            if context.get("_adaptive_capture_leaves"):
                difficulties = {
                    kind: {i: core for i in rows} for kind, rows in difficulties.items()
                }
            return result, blocks, difficulties

    monkeypatch.setattr(module, "ScalableQueryAdapter", FakeAdapter)
    monkeypatch.setattr(module, "validate_scalable_artifacts", lambda *args: None)
    monkeypatch.setattr(module, "spectral_filter", lambda family, x: x.unsqueeze(-3))
    families = {
        "global_distribution": State(),
        "multiscale_spectrum": SimpleNamespace(field_names=fields),
        "topology": SimpleNamespace(field_names=fields),
    }
    model = torch.nn.Linear(3, 3, bias=False)
    inputs = torch.tensor([[[1.0, 2.0, 4.0], [2.0, -1.0, 3.0], [-2.0, 1.0, 1.0], [1.0, 3.0, -1.0]]])
    reference = inputs * 0.7

    def fake_state(family, prediction, reference, context, coefficients):
        core = prediction.square().mean()
        return FamilyResult({}, core.expand(len(reference)), core, {}), {
            f"state.{b}": core for b in BLOCKS["state"]
        }

    def make(core_only=False):
        r = ScalableRuntime(
            artifacts,
            families,
            steps_per_epoch=1,
            expected_provenance=provenance,
            core_only=core_only,
        )
        r._state = fake_state
        return r

    r = make()
    r.record_occupied_time(1000.0)
    r.begin_step(0)
    first = r.batch
    r.finish_step(False)
    r.begin_step(0)
    assert r.batch == first
    for step in range(10):
        if step:
            r.begin_step(step)
        prediction, member = model(inputs), model(inputs * 0.8)
        context = {
            "conditional_draws": (member,),
            "sample_ids": ("train:1",),
            "query_indices": torch.tensor([[0, 1, 2, 3]]),
        }
        results = [
            r.score_family(name, family, prediction, reference, coordinates=None, context=context)
            for name, family in families.items()
        ]
        assert "state.group_projection" in results[0].component_results
        assert "scale.third_order" in results[1].component_results
        r.prepare_probe(model, prediction, reference, None, context)
        r.finish_step(True)
    assert r.calibration.coefficients is not None
    assert not r.sampler.cache["pair"] and not r.sampler.cache["self"] and r.policy_seconds == 0
    assert r.sampler.fixed_probabilities == (0.2, 0.3, 0.5)
    assert [p["epoch"] for p in r.calibration.records] == [1, 3, 6, 8, 10]
    restored = make()
    restored.load_state_dict(r.state_dict())
    assert restored.state_dict() == r.state_dict()
    core = make(core_only=True)
    assert core.bank is None and "state.group_projection" not in core.calibration.active
    with pytest.raises(ValueError, match="identity"):
        core.load_state_dict(r.state_dict())
