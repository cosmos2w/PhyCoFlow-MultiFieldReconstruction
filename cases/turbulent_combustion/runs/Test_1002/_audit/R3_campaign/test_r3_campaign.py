"""R3 opt-in selector and lineage accounting regression tests."""
import importlib.util
from pathlib import Path

import pytest
import torch

from phycoflow_reconstruction.training.checkpointing import PeriodicCheckpointManager
from phycoflow_reconstruction.training.run_store import RunStore

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").exists())
spec = importlib.util.spec_from_file_location("r3_pilot", ROOT / "scripts/training/run_upgrade_1002_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)

class Preview:
    enabled = False
    def due(self, step): return False
    def update(self, *a, **kw): return None

def test_r3_epoch_envelope_and_replay():
    config = {"stage":"post_training", "case":"turbulent_combustion",
        "output":{"experiment_name":"Test_1002/R3_10_corridor_control"},
        "optimization":{"epochs":200,"batch_size":32,"train_fraction":.15},
        "runtime":{"device":"cuda:0"}}
    planned = {"maximum_initial_epochs":{"R3_10":210}, "every_lineage_epoch_cap":249}
    envelope=pilot.validate_test_envelope(config,planned_runs=planned,train_count=8000)
    assert envelope["lineage_attempted_update_cap"] == 210*38
    assert pilot.epoch_segment_limit(until_epoch=200, additional_epochs=None,
        start_step=150*38,configured_epochs=200)==50*38
    config["optimization"]["epochs"]=250
    with pytest.raises(pilot.PilotContractError,match=">= 250"):
        pilot.validate_test_envelope(config,planned_runs=planned,train_count=8000)

def test_exploratory_preserves_strict_and_recovers(tmp_path):
    policy={"version":"r3_endpoint_corridor_v1", "minimum_epoch":100,
        "max_relative_mse_increase":.05,"max_relative_field_mse_increase":.10}
    config={"stage":"post_training", "checkpointing":{"every_epochs":10,
        "validation_every_epochs":10,"selection_metric":"coherence_with_fidelity",
        "exploratory_policy":policy}}
    store=RunStore.create(tmp_path,"r3_policy",config)
    model=torch.nn.Linear(1,1)
    manager=PeriodicCheckpointManager(config,store=store,steps_per_epoch=38)
    def report(step):
        epoch=step//38
        return {"eligible":epoch==0,"metric":1.0 if epoch==0 else .9+epoch*.00001,
            "mse":1., "metrics":{},"family_source_normalized_scores":{"A":.9},
            "relative_mse_increase":{"total":.01,"p":.087,"T":.02}}
    manager.panel_evaluator=report
    for epoch in (0,90,100,110,120,130):
        manager.save({"model":model.state_dict()},model=model,preview=Preview(),
            global_step=epoch*38,fallback_metric=0.,force=epoch==0)
    assert store.load_checkpoint("best")["global_step"]==0
    assert store.load_checkpoint("best_exploratory")["global_step"]==100*38
    assert (store.run_dir/"checkpoints/best_exploratory.pt").is_symlink()
    assert len(list((store.run_dir/"checkpoints").glob("exploratory_epoch_*.pt")))==3
    committed=store.load_checkpoint("last")
    recovered=PeriodicCheckpointManager(config,store=store,steps_per_epoch=38)
    recovered.restore_coherence_selector(committed)
    assert recovered.exploratory_archive==manager.exploratory_archive
    assert recovered._exploratory_report(report(100*38),100)["eligible"]
    fail=report(100*38);fail["relative_mse_increase"]["p"]=.101
    assert not recovered._exploratory_report(fail,100)["eligible"]
    assert not recovered._exploratory_report(report(90*38),90)["eligible"]


def test_emergency_requires_sustained_fixed_panel_catastrophe(tmp_path):
    import json
    metrics=tmp_path/"metrics";metrics.mkdir()
    rows=[]
    path=metrics/"coherence_validation.jsonl"
    def write(): path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    for epoch in (10,20,30,40):
        rows.append({"step":epoch*38,"relative_mse_increase":{"total":.3,"p":.4}})
    write();assert pilot.r3_emergency_evidence(tmp_path) is None
    rows=[]
    for epoch in (50,60,70,80):
        rows.append({"step":epoch*38,"relative_mse_increase":{"total":.04,"p":.087}})
    write();assert pilot.r3_emergency_evidence(tmp_path) is None
    rows=[]
    for epoch in (50,60):
        rows.append({"step":epoch*38,"relative_mse_increase":{"total":.21,"p":.4}})
    write();assert pilot.r3_emergency_evidence(tmp_path) is None
    rows.append({"step":70*38,"relative_mse_increase":{"total":.21,"p":.4}})
    write();assert pilot.r3_emergency_evidence(tmp_path)["reason"]=="sustained_catastrophic_fixed_panel_risk"


def test_r3_selection_uses_new_C_for_control_without_relabeling_legacy():
    from phycoflow_reconstruction.training.fidelity_controller import (
        coherence_selection_report,
        r3_coherence_selection_report,
    )
    source={"mse_normalized":1.,"per_field_mse_normalized":{"p":1.},"coherence":{"families":{
        "global_distribution":{"total":2.},"cross_spectrum":{"total":4.},
        "topology":{"total":100.,"component_scalars":{"topology.legacy":100.,"topology.finite_primary":1.}}}}}
    child={"mse_normalized":1.,"per_field_mse_normalized":{"p":1.},"coherence":{"families":{
        "global_distribution":{"total":1.8},"cross_spectrum":{"total":3.6},
        "topology":{"total":50.,"component_scalars":{"topology.legacy":50.,"topology.finite_primary":1.2}}}}}
    settings={"max_relative_mse_increase":.05,"max_relative_field_mse_increase":.05}
    old=coherence_selection_report(source,child,settings)
    new=r3_coherence_selection_report(source,child,settings)
    assert old["family_source_normalized_scores"]["topology"]==.5
    assert new["family_source_normalized_scores"]["topology"]==1.2
    assert new["metric"]==pytest.approx(1.)
    assert source["coherence"]["families"]["topology"]["total"]==100.
    assert new["metrics"]["coherence"]["families"]["topology"]["total"]==50.
    del child["coherence"]["families"]["topology"]["component_scalars"]["topology.finite_primary"]
    with pytest.raises(ValueError,match="TRAIN-calibrated"):
        r3_coherence_selection_report(source,child,settings)


def test_native_002_is_frozen_train_scale_allowance_with_live_graph():
    from phycoflow_reconstruction.training.fidelity_controller import FidelityController
    calibration={"version":"endpoint_native_source_calibration_v2","split":"train",
        "field_names":["p"],"source_risks":[1.,1.],"native":{
            "version":"native_source_calibration_v2","split":"train","source_scale":2.5}}
    options={"version":"endpoint_native_primal_dual_v2","native_loss_role":"constraint","native_budget":.02}
    controller=FidelityController(["p"],options,calibration)
    endpoint=torch.ones(2,3,1,dtype=torch.float64,requires_grad=True)
    reference=torch.zeros_like(endpoint)
    for source_value in (.2,1.,9.):
        live=torch.tensor(source_value+.075,dtype=torch.float64,requires_grad=True)
        source=torch.tensor(source_value,dtype=torch.float64,requires_grad=True)
        violations,_,_=controller.violations(endpoint,endpoint.detach(),reference,
            native_loss=live,source_native_loss=source)
        assert violations[-1].item()==pytest.approx(.01,abs=1e-14)
        penalty,_,pressure=controller.primal(violations)
        assert pressure[-1]>0
        penalty.backward()
        assert live.grad.item()==pytest.approx(.01/2.5,abs=1e-14)
        assert source.grad is None
    # .02*2.5=.05 native-loss units is allowed independently of each source draw.
    live=torch.tensor(.2+.04,dtype=torch.float64,requires_grad=True)
    violations,_,_=controller.violations(endpoint,endpoint.detach(),reference,
        native_loss=live,source_native_loss=torch.tensor(.2,dtype=torch.float64))
    assert violations[-1]<0
    assert controller.primal(violations)[2][-1]==0


def test_native_002_multiplier_recovers_and_incompatible_resume_rejected():
    from phycoflow_reconstruction.training.fidelity_controller import FidelityController
    calibration={"version":"endpoint_native_source_calibration_v2","split":"train",
        "field_names":["p"],"source_risks":[1.,1.],"native":{
            "version":"native_source_calibration_v2","split":"train","source_scale":2.5}}
    options={"version":"endpoint_native_primal_dual_v2","native_loss_role":"constraint","native_budget":.02}
    controller=FidelityController(["p"],options,calibration)
    for _ in range(20): controller.advance(torch.tensor([-.05,-.05,.01]))
    assert controller.multipliers[-1]>0
    recovered=FidelityController(["p"],options,calibration,state=controller.state_dict())
    for _ in range(100):
        feasible=torch.tensor([-.05,-.05,-.05])
        controller.advance(feasible);recovered.advance(feasible)
    assert controller.multipliers[-1]==0
    assert torch.equal(controller.multipliers,recovered.multipliers)
    assert torch.equal(controller.ema,recovered.ema)
    with pytest.raises(ValueError,match="settings differ"):
        FidelityController(["p"],{**options,"native_budget":0.},calibration,state=controller.state_dict())
