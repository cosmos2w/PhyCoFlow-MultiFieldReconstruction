import copy
import json
import subprocess
import sys

import pytest
import torch
import yaml

from phycoflow_reconstruction.training.fidelity_budget import MeanRiskDifferenceBudget
from phycoflow_reconstruction.training.internal_calibration import (
    BLOCKS,
    InternalCalibration,
    fit_common_prefit,
)
from phycoflow_reconstruction.training.run_store import RunStore, config_digest


def test_prefit_uses_weighted_batch_rms_and_roundtrips_without_online_probes():
    names = [f"{family}.{block}" for family, blocks in BLOCKS.items() for block in blocks]
    calibration = InternalCalibration({k: 1.0 for k in names}, {k: 1.0 for k in names}, {k: True for k in names})
    p = torch.nn.Parameter(torch.tensor(1.0))
    batches = [(["one"], {k: p * (i + 1) for i, k in enumerate(names)}), (["two", "three"], {k: p * 2 * (i + 1) for i, k in enumerate(names)})]
    artifact = fit_common_prefit(calibration, batches, [p], binding={"split": "train"}, scale_joint_ratio=2)
    assert artifact["rms_gradient_norms"][names[0]] == pytest.approx(3 ** 0.5)
    restored = InternalCalibration.from_state_dict(calibration.state_dict())
    assert restored.coefficients == calibration.coefficients
    assert restored.records == []
    broken = copy.deepcopy(artifact)
    broken["coefficients"]["scale.joint"] += 1
    with pytest.raises(ValueError, match="coefficients"):
        restored.load_prefit(broken)


def test_signed_fieldwise_budget_does_not_floor_allowance_and_keeps_gradients():
    budget = MeanRiskDifferenceBudget.from_reference(["a", "b"], [-1.0, 2.0], [0.0, 0.3], [2.0, 4.0])
    risks = torch.tensor([-0.5, 2.5], requires_grad=True)
    violation = budget.violations(risks)
    assert violation.tolist() == pytest.approx([0.25, 0.05])
    violation.sum().backward()
    assert risks.grad.tolist() == pytest.approx([0.5, 0.25])


def test_resume_accepts_device_change_and_reports_scientific_differences(tmp_path):
    saved = {"runtime": {"device": "cuda:0"}, "optimization": {"epochs": 12}, "coherence": {"order": ["state", "scale"]}}
    (tmp_path / "resolved_config.yaml").write_text(yaml.safe_dump(saved))
    (tmp_path / "run_manifest.json").write_text(json.dumps({"config_sha256": config_digest(saved)}))
    incoming = copy.deepcopy(saved)
    incoming["runtime"]["device"] = "cuda:1"
    assert RunStore.resume(tmp_path, incoming).config_hash == config_digest(saved)
    incoming["coherence"]["order"].reverse()
    with pytest.raises(ValueError, match="changed coherence.order"):
        RunStore.resume(tmp_path, incoming)


def test_resume_saved_mapping_order_owns_component_order(tmp_path):
    saved = {"coherence": {"families": {"state": {"weight": 1}, "scale": {"weight": 2}}},
             "runtime": {"device": "cuda:0"}}
    (tmp_path / "resolved_config.yaml").write_text(yaml.safe_dump(saved, sort_keys=False))
    (tmp_path / "run_manifest.json").write_text(json.dumps({"config_sha256": config_digest(saved)}))
    incoming = {"runtime": {"device": "cuda:1"}, "coherence": {"families": {"scale": {"weight": 2}, "state": {"weight": 1}}}}
    store = RunStore.resume(tmp_path, incoming)
    assert list(store.config['coherence']['families']) == ['state', 'scale']
    assert store.config['runtime']['device'] == 'cuda:1'


def test_saved_recording_column_order_ignores_incoming_mapping_order(tmp_path, monkeypatch):
    import phycoflow_reconstruction.config as config_module
    from phycoflow_reconstruction.training.monitoring import TrainingMonitor
    monkeypatch.setattr(config_module, 'load_config', lambda _: {'coherence': {'scalable_anchored_coherence': {'common_prefit': {'path': 'unused'}}}})
    monkeypatch.setattr(TrainingMonitor, '_plot', lambda self: None)
    (tmp_path / 'logs').mkdir()
    (tmp_path / 'resolved_config.yaml').touch()
    monitor = TrainingMonitor(tmp_path, start_step=0, final_step=2, configured_steps=2,
                              steps_per_epoch=1, description='order', enabled=False)
    monitor.record({'step': 1, 'coherence_loss': 1., 'update_seconds': .2, 'update_accepted': True})
    monitor.record({'step': 2, 'update_accepted': True, 'update_seconds': .3, 'coherence_loss': 2.})
    monitor.close()
    rows = [json.loads(line) for line in (tmp_path / 'logs/history.jsonl').read_text().splitlines()]
    assert list(rows[0]) == list(rows[1])
    columns = {row['name']: row for row in json.loads((tmp_path / 'logs/recording_definitions.json').read_text())['columns']}
    assert columns['update_seconds']['units'] == 'seconds'


def test_monitor_restores_partial_epoch_accumulation(tmp_path, monkeypatch):
    import phycoflow_reconstruction.config as config_module
    from phycoflow_reconstruction.training.monitoring import TrainingMonitor
    monkeypatch.setattr(config_module, "load_config", lambda _: {"coherence": {"scalable_anchored_coherence": {"common_prefit": {"path": "unused"}}}})
    monkeypatch.setattr(TrainingMonitor, "_plot", lambda self: None)
    def create(root, start, state=None):
        (root / "logs").mkdir(parents=True, exist_ok=True)
        (root / "resolved_config.yaml").touch()
        return TrainingMonitor(root, start_step=start, final_step=4, configured_steps=4,
                               steps_per_epoch=4, description="test", enabled=False, resume_state=state)
    full = create(tmp_path / "full", 0)
    partial = create(tmp_path / "partial", 0)
    for step in (1, 2):
        row = {"step": step, "coherence_loss": float(step), "update_accepted": True}
        full.record(row); partial.record(row)
    state = partial.checkpoint_state()
    partial.close()
    resumed = create(tmp_path / "partial", 2, state)
    for step in (3, 4):
        row = {"step": step, "coherence_loss": float(step), "update_accepted": True}
        full.record(row); resumed.record(row)
    full.close(); resumed.close()
    assert (tmp_path / "full/logs/history.jsonl").read_text() == (tmp_path / "partial/logs/history.jsonl").read_text()
    assert json.loads((tmp_path / "partial/logs/history.jsonl").read_text())["coherence_loss"] == 2.5


def test_fresh_process_signed_policy_resume_matches_uninterrupted(tmp_path):
    pytest.importorskip("cooper")
    script = tmp_path / "worker.py"
    script.write_text('''
import json, sys, torch
from phycoflow_reconstruction.training.fidelity_budget import ExternalPolicy, MeanRiskDifferenceBudget
from phycoflow_reconstruction.coherence.physical_risk import physical_ensemble_risks
path, mode = sys.argv[1:]
torch.manual_seed(71)
model = torch.nn.Linear(1, 2, dtype=torch.float64)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
budget = MeanRiskDifferenceBudget.from_reference(["a", "b"], [-0.2, 0.1], [0.01, 0.02], [1.0, 2.0])
policy = ExternalPolicy({"method": "lagrangian", "lagrangian": {"dual_lr": 0.1}}, budget, "cpu")
history = []
start = 0
if mode == "resume":
    saved = torch.load(path, weights_only=False)
    model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); policy.load_state_dict(saved["policy"])
    torch.set_rng_state(saved["rng"]); history = saved["history"]; start = saved["step"]
end = 3 if mode == "pause" else 6
for step in range(start, end):
    x = torch.randn(3, 1, dtype=torch.float64)
    value = model(x).unsqueeze(1)
    members = torch.stack((value + 0.2, value - 0.3), dim=1)
    _, risk = physical_ensemble_risks(members, torch.zeros_like(value), torch.ones(3, 1))
    primary = (value - 0.4).square().mean()
    result = policy.update(model, optimizer, primary, risk.mean(0), None)
    history.append([step + 1, float(primary.detach()), *result["field_violation"].values()])
torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "policy": policy.state_dict(), "rng": torch.get_rng_state(), "history": history, "step": end}, path)
''')
    uninterrupted, resumed = tmp_path / "full.pt", tmp_path / "resume.pt"
    for path, mode in ((uninterrupted, "full"), (resumed, "pause"), (resumed, "resume")):
        subprocess.run([sys.executable, str(script), str(path), mode], check=True, capture_output=True, text=True)
    full = torch.load(uninterrupted, weights_only=False)
    split = torch.load(resumed, weights_only=False)
    assert full["history"] == split["history"]
    assert full["policy"]["completed_updates"] == split["policy"]["completed_updates"] == 6
    assert full["policy"]["multipliers"] == split["policy"]["multipliers"]
    for key in full["model"]:
        torch.testing.assert_close(full["model"][key], split["model"][key], rtol=0, atol=0)
    assert torch.equal(full["rng"], split["rng"])
