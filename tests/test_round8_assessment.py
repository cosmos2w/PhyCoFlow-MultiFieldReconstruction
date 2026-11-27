"""Focused checks for the Round 8 cached endpoint adapter."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.posttraining import assess_car_round8 as adapter


def test_prepare_refuses_an_incomplete_endpoint(tmp_path):
    result = tmp_path / "training_result.json"
    result.write_text(json.dumps({"schema": "car_round8_training_result_v1", "arm": "MemberMSE_BaseShape",
                                  "target_epoch": 100, "global_step": 1199, "target_reached": False}))
    args = SimpleNamespace(campaign=tmp_path, training_result=result, attempt=tmp_path / "assessment/attempt")
    with pytest.raises(ValueError, match="completed exact"):
        adapter.prepare(args)
    assert not args.attempt.exists()


def test_assessed_epoch_survives_a_continuation_overwriting_last(tmp_path, monkeypatch):
    import torch
    import yaml
    from scripts.posttraining import prepare_car_round7_assessment as preparation

    campaign = tmp_path / "round_8"
    (campaign / "configs").mkdir(parents=True)
    run = campaign / "runs/formal/MemberMSE_BaseShape/run"
    (run / "checkpoints").mkdir(parents=True)
    config = campaign / "configs/MemberMSE_BaseShape.yaml"
    config.write_text(yaml.safe_dump({"runtime": {"device": "cpu"}}))
    saved = run / "resolved_config.yaml"
    saved.write_bytes(config.read_bytes())
    checkpoint = run / "checkpoints/last.pt"
    torch.save({"global_step": 1200, "model": {"weight": torch.tensor([1.])}}, checkpoint)
    reference = campaign / "configs/reference.json"
    reference.write_text("{}")
    manifest = campaign / "configs/conditional_learning_prepared.json"
    manifest.write_text(json.dumps({"arms": {"MemberMSE_BaseShape": adapter.ref(config)},
                                    "reference": adapter.ref(reference)}))
    result = campaign / "training_result.json"
    result.write_text(json.dumps({"schema": "car_round8_training_result_v1", "arm": "MemberMSE_BaseShape",
        "target_epoch": 100, "global_step": 1200, "reporting_epoch": 100, "target_reached": True,
        "disposition": "target_reached", "test_locked": True, "prepared_config": adapter.ref(config),
        "saved_config": adapter.ref(saved), "checkpoint": adapter.ref(checkpoint), "run_dir": str(run)}))
    selected, native = tmp_path / "source_selected.json", tmp_path / "source_native.json"
    selected.write_text("{}")
    native.write_text("{}")
    ids = [f"condition_{index:03d}" for index in range(100)]
    baseline = tmp_path / "round_7/assessment/SOURCE_Baseline_VALIDATION100_v1/baseline.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text(json.dumps({"status": "completed", "binding": {}, "sample_ids": ids,
        "source_selected": adapter.ref(selected), "source_native": adapter.ref(native), "files_sha256": ["row"] * 100}))
    def prepare_spec(args):
        args.output.write_text(json.dumps({"split": "validation", "checkpoint": adapter.ref(args.checkpoint),
            "reporting_epoch": args.reporting_epoch, "source_native_receipt": adapter.ref(native)}))
    monkeypatch.setattr(preparation, "prepare", prepare_spec)
    monkeypatch.setattr(adapter, "validate_spec", lambda _spec: ({"sample_ids": ids}, {}))
    monkeypatch.setattr(adapter, "baseline_binding", lambda _spec: {})
    monkeypatch.setattr(adapter, "verify_bank", lambda *_args: ({}, {}))
    attempt = campaign / "assessment/MemberMSE_BaseShape_Epoch100_v1"
    adapter.prepare(SimpleNamespace(campaign=campaign, training_result=result, attempt=attempt))
    plan = json.loads((attempt / "plan.json").read_text())
    assert plan["checkpoint"]["sha256"] == plan["original_training_checkpoint"]["sha256"]
    assert Path(plan["checkpoint"]["path"]) != checkpoint
    torch.save({"global_step": 2400, "model": {"weight": torch.tensor([2.])}}, checkpoint)
    preserved, *_ = adapter.context(attempt)
    assert torch.load(adapter.checked(preserved["checkpoint"]), weights_only=False)["global_step"] == 1200
    Path(preserved["checkpoint"]["path"]).write_bytes(b"corrupt snapshot")
    with pytest.raises(ValueError, match="hash"):
        adapter.context(attempt)


def test_generation_uses_round8_lease_for_exactly_two_supports(tmp_path, monkeypatch):
    (tmp_path / "plan.json").write_text("{}")
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    plan = {"supports": ["native", "selected"], "checkpoint": {"sha256": "a" * 64},
            "spec": adapter.ref(spec)}
    calls = []
    monkeypatch.setattr(adapter, "context", lambda _attempt: (plan, {"run_config": {"sha256": "b" * 64}}, {}, {}))
    def fake_generate(args, _spec, _panel, _source):
        calls.append((args.support, adapter.os.environ.get("CAR_ROUND7_LEASE_TOKEN")))
        args.output.mkdir()
        (args.output / "receipt.json").write_text(json.dumps({"checkpoint_sha256": "a" * 64,
            "spec_sha256": plan["spec"]["sha256"], "run_config_sha256": "b" * 64}))
    monkeypatch.setattr(adapter, "generate_round7", fake_generate)
    monkeypatch.setattr(adapter, "verify_bank", lambda path, *_: (path, json.loads(path.read_text())))
    args = SimpleNamespace(attempt=tmp_path, device="cuda:0")
    monkeypatch.delenv("CAR_ROUND8_LEASE_TOKEN", raising=False)
    with pytest.raises(ValueError, match="Round 8 GPU lease"):
        adapter.generate(args)
    monkeypatch.setenv("CAR_ROUND8_LEASE_TOKEN", "lease-identity")
    monkeypatch.delenv("CAR_ROUND7_LEASE_TOKEN", raising=False)
    adapter.generate(args)
    assert calls == [("native", "lease-identity"), ("selected", "lease-identity")]
    assert "CAR_ROUND7_LEASE_TOKEN" not in adapter.os.environ
    receipt = json.loads((tmp_path / "generation_result.json").read_text())
    assert receipt["status"] == "completed" and set(receipt["banks"]) == {"native", "selected"}
    with pytest.raises(FileExistsError):
        adapter.generate(args)


def test_failed_generation_attempt_is_preserved_and_cannot_rerun(tmp_path, monkeypatch):
    (tmp_path / "plan.json").write_text("{}")
    (tmp_path / "spec.json").write_text("{}")
    plan = {"supports": ["native", "selected"], "spec": adapter.ref(tmp_path / "spec.json")}
    monkeypatch.setattr(adapter, "context", lambda _attempt: (plan, {}, {}, {}))
    monkeypatch.setenv("CAR_ROUND8_LEASE_TOKEN", "lease-identity")
    monkeypatch.delenv("CAR_ROUND7_LEASE_TOKEN", raising=False)
    def fail(*_args):
        raise RuntimeError("generation failed")
    monkeypatch.setattr(adapter, "generate_round7", fail)
    args = SimpleNamespace(attempt=tmp_path, device="cuda:0")
    with pytest.raises(RuntimeError, match="generation failed"):
        adapter.generate(args)
    assert json.loads((tmp_path / "generation_result.json").read_text())["status"] == "failed"
    with pytest.raises(FileExistsError):
        adapter.generate(args)


def test_score_retry_reuses_completed_scores_after_position_failure(tmp_path, monkeypatch):
    import scripts.analysis.car_round8_cached_summary as summary_module
    import scripts.analysis.car_round8_child_relations as relation_module

    monkeypatch.delenv("CAR_ROUND8_LEASE_TOKEN", raising=False)
    monkeypatch.delenv("CAR_ROUND7_LEASE_TOKEN", raising=False)
    for name in ("plan.json", "spec.json", "baseline.json", "source.json", "cached.json"):
        (tmp_path / name).write_text("{}")
    spec_ref = adapter.ref(tmp_path / "spec.json")
    bank_refs = {}
    for name in ("native", "selected"):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps({"checkpoint_sha256": "a" * 64, "spec_sha256": spec_ref["sha256"]}))
        bank_refs[name] = adapter.ref(path)
    plan = {"arm": "MemberMSE_BaseShape", "epoch": 100, "spec": spec_ref,
        "checkpoint": {"sha256": "a" * 64}, "source_baseline": adapter.ref(tmp_path / "baseline.json"),
        "source_selected": adapter.ref(tmp_path / "source.json"), "position_fit": {"sha256": "b" * 64}}
    monkeypatch.setattr(adapter, "context", lambda _attempt: (plan, {}, {"sample_ids": ["condition"]}, {}))
    monkeypatch.setattr(adapter, "verify_bank", lambda path, *_: (path, json.loads(path.read_text())))
    (tmp_path / "generation_result.json").write_text(json.dumps({"status": "completed",
        "plan": adapter.ref(tmp_path / "plan.json"), "banks": bank_refs}))
    calls = []
    def fake_score(args, *_):
        calls.append("score")
        args.output.mkdir()
        (args.output / "receipt.json").write_text(json.dumps({"status": "completed"}))
        (args.output / "assessment.json").write_text(json.dumps({"status": "completed", "spec": spec_ref,
            "sample_ids": ["condition"], "child_native": bank_refs["native"], "child_selected": bank_refs["selected"],
            "source_selected": plan["source_selected"], "source_baseline": plan["source_baseline"]}))
        for name in ("decomposition.json", "scale_summary.json"):
            (args.output / name).write_text("{}")
    monkeypatch.setattr(adapter, "score_round7", fake_score)
    monkeypatch.setattr(summary_module, "summarize", lambda *_: {"fields": []})
    position_calls = []
    def fake_position(_assessment, _plan, cached, *, threads):
        position_calls.append(threads)
        if len(position_calls) == 1:
            raise RuntimeError("position scorer interrupted")
        return {"status": "completed", "fit": plan["position_fit"], "cached_source": adapter.ref(cached)}
    monkeypatch.setattr(adapter, "position_readout", fake_position)
    relation_calls = []
    def fake_relations(assessment, cached, *, label, threads):
        relation_calls.append(threads)
        if len(relation_calls) == 1:
            raise RuntimeError("relation scorer interrupted")
        return {"status": "completed", "endpoint_label": label,
                "inputs": {"assessment": adapter.ref(assessment),
                           "decomposition": adapter.ref(assessment.parent / "decomposition.json"),
                           "source_readout": adapter.ref(cached)}}
    monkeypatch.setattr(relation_module, "build_readout", fake_relations)
    args = SimpleNamespace(attempt=tmp_path, threads=2, position_source_readout=tmp_path / "cached.json",
                           relation_source_readout=tmp_path / "cached.json")
    with pytest.raises(RuntimeError, match="position scorer interrupted"):
        adapter.score(args)
    with pytest.raises(RuntimeError, match="relation scorer interrupted"):
        adapter.score(args)
    adapter.score(args)
    assert calls == ["score"] and position_calls == [2, 2] and relation_calls == [2, 2]
    result = json.loads((tmp_path / "assessment_result.json").read_text())
    assert result["reused_completed_cpu_scores"] and result["position_readout"] and result["relation_context"]
    with pytest.raises(FileExistsError, match="already complete"):
        adapter.score(args)


def test_score_forwards_and_revalidates_explicit_direct_source_companion(tmp_path, monkeypatch):
    import scripts.analysis.car_round8_cached_summary as summary_module
    import scripts.analysis.car_round8_child_relations as relation_module

    monkeypatch.delenv("CAR_ROUND8_LEASE_TOKEN", raising=False)
    monkeypatch.delenv("CAR_ROUND7_LEASE_TOKEN", raising=False)
    for name in ("plan.json", "spec.json", "baseline.json", "source.json", "cached.json", "companion.json"):
        (tmp_path / name).write_text("{}")
    spec_ref = adapter.ref(tmp_path / "spec.json")
    banks = {}
    for name in ("native", "selected"):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps({"checkpoint_sha256": "a" * 64, "spec_sha256": spec_ref["sha256"]}))
        banks[name] = adapter.ref(path)
    plan = {"arm": "MemberMSE_BaseShape", "epoch": 100, "spec": spec_ref,
            "checkpoint": {"sha256": "a" * 64}, "source_baseline": adapter.ref(tmp_path / "baseline.json"),
            "source_selected": adapter.ref(tmp_path / "source.json")}
    monkeypatch.setattr(adapter, "context", lambda _attempt: (plan, {}, {"sample_ids": ["condition"]}, {}))
    monkeypatch.setattr(adapter, "verify_bank", lambda path, *_: (path, json.loads(path.read_text())))
    (tmp_path / "generation_result.json").write_text(json.dumps({"status": "completed",
        "plan": adapter.ref(tmp_path / "plan.json"), "banks": banks}))
    def fake_score(args, *_):
        args.output.mkdir()
        (args.output / "receipt.json").write_text(json.dumps({"status": "completed"}))
        (args.output / "assessment.json").write_text(json.dumps({"status": "completed", "spec": spec_ref,
            "sample_ids": ["condition"], "child_native": banks["native"], "child_selected": banks["selected"],
            "source_selected": plan["source_selected"], "source_baseline": plan["source_baseline"]}))
        for name in ("decomposition.json", "scale_summary.json"):
            (args.output / name).write_text("{}")
    monkeypatch.setattr(adapter, "score_round7", fake_score)
    monkeypatch.setattr(summary_module, "summarize", lambda *_: {"fields": []})
    seen = []
    def fake_relation(assessment, cached, *, label, threads, source_companion_path):
        seen.append(source_companion_path)
        return {"status": "completed", "endpoint_label": label,
                "inputs": {"assessment": adapter.ref(assessment),
                           "decomposition": adapter.ref(assessment.parent / "decomposition.json"),
                           "source_readout": adapter.ref(cached),
                           "source_companion": adapter.ref(source_companion_path)}}
    monkeypatch.setattr(relation_module, "build_readout", fake_relation)
    args = SimpleNamespace(attempt=tmp_path, threads=2, position_source_readout=None,
                           relation_source_readout=tmp_path / "cached.json",
                           direct_source_companion=tmp_path / "companion.json")
    adapter.score(args)
    assert seen == [tmp_path / "companion.json"]
    (tmp_path / "assessment_result.json").unlink()
    (tmp_path / "companion.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="saved relation readout differs"):
        adapter.score(args)
    args.direct_source_companion = None
    with pytest.raises(ValueError, match="must be supplied explicitly"):
        adapter.score(args)


def test_tasks_require_sealed_release_and_write_round8_receipt(tmp_path, monkeypatch):
    monkeypatch.delenv("CAR_ROUND8_LEASE_TOKEN", raising=False)
    monkeypatch.delenv("CAR_ROUND7_LEASE_TOKEN", raising=False)
    for name in ("plan.json", "spec.json", "source.json", "selected.json"):
        (tmp_path / name).write_text("{}")
    spec_ref = adapter.ref(tmp_path / "spec.json")
    source_ref = adapter.ref(tmp_path / "source.json")
    child = tmp_path / "native.json"
    child.write_text(json.dumps({"status": "completed", "split": "validation", "sample_ids": ["condition"],
        "members": 4, "panel_sha256": "panel", "source_sha256": "source", "source_bank_sha256": source_ref["sha256"],
        "support": "native", "files_sha256": ["row"], "checkpoint_sha256": "checkpoint", "spec_sha256": spec_ref["sha256"]}))
    banks = {"native": adapter.ref(child), "selected": adapter.ref(tmp_path / "selected.json")}
    plan = {"arm": "MemberMSE_BaseShape", "epoch": 100, "checkpoint": {"sha256": "checkpoint"}, "spec": spec_ref}
    spec = {"split": "validation", "panel": {"sha256": "panel"}, "source_checkpoint": {"sha256": "source"},
            "source_native_receipt": source_ref}
    panel = {"sample_ids": ["condition"]}
    monkeypatch.setattr(adapter, "context", lambda _attempt: (plan, spec, panel, {}))
    (tmp_path / "generation_result.json").write_text(json.dumps({"status": "completed",
        "plan": adapter.ref(tmp_path / "plan.json"), "banks": banks}))
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({"status": "draft"}))
    args = SimpleNamespace(attempt=tmp_path, decision=decision, threads=2)
    with pytest.raises(ValueError, match="sealed"):
        adapter.tasks(args)
    assert not (tmp_path / "tasks").exists()
    decision.write_text(json.dumps({"schema": "car_round7_task_release_v1", "status": "sealed",
        "child_checkpoint_sha256": "other", "child_native_receipt_sha256": banks["native"]["sha256"],
        "split": "validation", "reason": "endpoint evaluation"}))
    with pytest.raises(ValueError, match="sealed"):
        adapter.tasks(args)

    decision.write_text(json.dumps({"schema": "car_round7_task_release_v1", "status": "sealed",
        "child_checkpoint_sha256": "checkpoint", "child_native_receipt_sha256": banks["native"]["sha256"],
        "split": "validation", "reason": "endpoint evaluation"}))
    def fake_tasks(call, *_):
        assert call.child_native == child and call.decision == decision and call.threads == 2
        call.output.mkdir()
        (call.output / "receipt.json").write_text(json.dumps({"status": "completed"}))
        (call.output / "tasks.json").write_text(json.dumps({"status": "completed", "split": "validation",
            "members": 4, "sample_ids": ["condition"], "test_locked": True, "spec": spec_ref,
            "decision": adapter.ref(decision), "child_native": banks["native"], "source_native": source_ref}))
    monkeypatch.setattr(adapter, "tasks_round7", fake_tasks)
    adapter.tasks(args)
    result = json.loads((tmp_path / "task_result.json").read_text())
    assert result["schema"] == "car_round8_validation_task_assessment_v1"
    assert result["release"] == adapter.ref(decision) and result["additional_inference"] is False
    assert "outside training PI and continuation nomination" in result["scope"]
    with pytest.raises(FileExistsError, match="already complete"):
        adapter.tasks(args)


def test_position_adapter_matches_one_saved_real_cached_row(tmp_path):
    """Re-score one existing stored condition; no model inference or new panel."""
    import numpy as np
    path = Path("cases/turbulent_combustion/runs/coherence_adaptive_repair/round_8/assessment/SpatialShape_Common_VALIDATION100_v1/readout.json")
    if not path.exists():
        pytest.skip("optional sealed campaign fixtures are unavailable")
    cached = json.loads(path.read_text())
    assessment = json.loads(Path(cached["inputs"]["joint500"]["path"]).read_text())
    row = cached["records"][0]
    fixture = deepcopy(cached)
    fixture["sample_ids"] = fixture["sample_ids"][:1]
    fixture["records"] = fixture["records"][:1]
    copied = tmp_path / "source_one.json"
    copied.write_text(json.dumps(fixture))
    assessment["sample_ids"] = assessment["sample_ids"][:1]
    result = adapter.position_readout(assessment, {"position_fit": cached["inputs"]["fit"]}, copied, threads=2)
    actual = result["records"][0]["models"]["candidate"]
    for kind in ("self", "mutual"):
        for metric in ("ph_ARS", "location_ARS", "location_unit_matched_ARS", "combined_ARS", "native_location_unit_matched_ARS"):
            np.testing.assert_allclose(actual[kind][metric], row["models"]["JointAnchor500"][kind][metric], rtol=1e-5, atol=1e-8)
    assert result["records"][0]["models"]["SOURCE"] == row["models"]["SOURCE"]
