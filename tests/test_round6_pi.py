import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from phycoflow_reconstruction.training.fidelity_budget import ExternalPolicy, MeanRiskDifferenceBudget


def make_policy():
    ref = MeanRiskDifferenceBudget.from_reference(['a', 'b'], [0.2, -0.1], [0.05, 0.1], [0.5, 0.8])
    return ExternalPolicy({'method': 'lagrangian', 'lagrangian': {'epoch_calibrated_pi': {'gamma': 3., 'tau_epochs': 1., 'horizon_epochs': 10., 'steps_per_epoch': 12}}}, ref, 'cpu')


def test_pi_pre_action_projection_and_failed_action():
    policy = make_policy()
    model = torch.nn.Linear(1, 2, bias=False, dtype=torch.float64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    initial = policy.state_dict()
    assert initial['epoch_calibrated_pi']['z'] == pytest.approx([-.1, -.125])
    x = model(torch.ones(1, 1, dtype=torch.float64)).reshape(-1)
    risks = x + torch.tensor([2., -2.])
    error = policy.reference.violations(risks).detach()
    report = policy.update(model, optimizer, x.square().sum(), risks, None)
    nu = math.exp(-1/12)
    old_z = torch.tensor([-.1, -.125], dtype=torch.float64)
    z = nu*old_z + (1-nu)*error
    expected = torch.clamp(.0036/12*error + .036*(z-old_z), min=0)
    assert report['multipliers_before'] == [0., 0.]
    assert report['multipliers_after'] == pytest.approx(expected.tolist())
    assert policy.state_dict()['epoch_calibrated_pi']['z'] == pytest.approx(z.tolist())
    saved = copy.deepcopy(policy.state_dict())
    def fail(*args, **kwargs):
        raise RuntimeError('failed action')
    optimizer.step = fail
    x = model(torch.ones(1, 1, dtype=torch.float64)).reshape(-1)
    with pytest.raises(RuntimeError, match='failed action'):
        policy.update(model, optimizer, x.square().sum(), x, None)
    assert policy.state_dict() == saved


def equal(a, b):
    if isinstance(a, torch.Tensor): return torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def test_pi_config_records_executed_action_products():
    from phycoflow_reconstruction.training.multitask import OuterAggregator
    torch.manual_seed(81)
    model = torch.nn.Linear(2, 3, bias=False, dtype=torch.float64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.02)
    policy = make_policy()
    before = model.weight.detach().clone()
    output = model(torch.tensor([[1.0, -0.4]], dtype=torch.float64)).reshape(-1)
    losses = {name: (output[i] - (i+1)).square() for i, name in enumerate(('state', 'scale', 'shape'))}
    gradients = torch.stack([torch.autograd.grad(loss, model.weight, retain_graph=True)[0].reshape(-1) for loss in losses.values()])
    report = policy.update_with_coherence_aggregation(model, optimizer, losses,
        output[:2].square(), 0.1, OuterAggregator({'method': 'torchjd_config'}))
    displacement = (model.weight.detach() - before).reshape(-1)
    expected = gradients @ displacement
    assert list(report['action_family_products']['actual_displacement']) == list(losses)
    assert list(report['action_family_products']['actual_displacement'].values()) == pytest.approx(expected.tolist())
    assert report['action_displacement_norm'] == pytest.approx(float(displacement.norm()))
    assert report['action_constraint_product'] == 0.0


def test_fresh_process_resume_and_append(tmp_path):
    pytest.importorskip('cooper'); pytest.importorskip('torchjd')
    worker = Path(__file__).parent / 'helpers/round6_pi_resume_worker.py'
    for name, mode in [('continuous','continuous'), ('resumed','partial'), ('resumed','resume')]:
        subprocess.run([sys.executable, str(worker), str(tmp_path/name), mode], check=True, capture_output=True, text=True)
    a, b = [torch.load(tmp_path/name/'checkpoint.pt', weights_only=False) for name in ('continuous', 'resumed')]
    for key in ('model', 'optimizer', 'policy', 'monitor', 'coefficients', 'python', 'torch'):
        assert equal(a[key], b[key]), key
    assert (a['numpy'][1] == b['numpy'][1]).all() and a['numpy'][2:] == b['numpy'][2:]
    for file in ('logs/recording_definitions.json', 'logs/history.jsonl'):
        left, right = [(tmp_path/name/file).read_text() for name in ('continuous', 'resumed')]
        assert json.loads(left) == json.loads(right) if file.endswith('definitions.json') else left == right
    definition = json.loads((tmp_path/'resumed/logs/recording_definitions.json').read_text())
    assert definition['loss_component_accumulation_order'] == ['state', 'scale', 'shape']
    assert all(set(c) == {'name', 'type', 'units', 'meaning'} for c in definition['columns'])


@pytest.mark.parametrize('change', ['units', 'meaning', 'key_order'])
def test_saved_recording_semantics_own_resume_append(tmp_path, change):
    from phycoflow_reconstruction.training.monitoring import TrainingMonitor
    import yaml

    (tmp_path / 'resolved_config.yaml').write_text(yaml.safe_dump({
        'coherence': {'scalable_anchored_coherence': {'common_prefit': True},
                      'families': {'state': {}, 'scale': {}, 'shape': {}}}}))
    def make_monitor(start):
        monitor = TrainingMonitor(tmp_path, start_step=start, final_step=2,
            configured_steps=2, steps_per_epoch=1, description='recording_fixture', enabled=False)
        monitor._plot = lambda: None
        return monitor
    monitor = make_monitor(0)
    monitor.record({'step': 1, 'coherence_loss': 1.0})
    monitor.close()
    definitions = tmp_path / 'logs/recording_definitions.json'
    saved = json.loads(definitions.read_text())
    if change != 'key_order':
        next(c for c in saved['columns'] if c['name'] == 'coherence_loss')[change] = 'different scientific definition'
    definitions.write_text(json.dumps(saved, sort_keys=True))
    before = (tmp_path / 'logs/history.jsonl').read_bytes()
    resumed = make_monitor(1)
    if change == 'key_order':
        resumed.record({'coherence_loss': 0.5, 'step': 2})
        resumed.close()
        assert len((tmp_path / 'logs/history.jsonl').read_text().splitlines()) == 2
    else:
        with pytest.raises(ValueError, match="semantic=\\['coherence_loss'\\]"):
            resumed.record({'step': 2, 'coherence_loss': 0.5})
        assert (tmp_path / 'logs/history.jsonl').read_bytes() == before
        resumed.progress.close()
