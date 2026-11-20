"""Exercise saved-config recovery in fresh interpreters across a warmup boundary."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from phycoflow_reconstruction.training.run_store import load_project_checkpoint


def _equal(left, right, path='root'):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right), path
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            _equal(left[key], right[key], f'{path}.{key}')
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            _equal(a, b, f'{path}[{index}]')
    else:
        assert left == right, path


def test_epoch_warmup_fresh_process_saved_config_resume(tmp_path):
    pytest.importorskip('cooper')
    worker = Path(__file__).parent / 'helpers/epoch_warmup_resume_worker.py'
    device = os.environ.get('PHYCOFLOW_QUALIFICATION_DEVICE', 'cpu')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(Path(__file__).parent), os.environ.get('PYTHONPATH', ''))))
    for mode in ('prepare', 'continuous', 'partial', 'resume'):
        subprocess.run([sys.executable, str(worker), str(tmp_path), mode, device], env=env, check=True)
    continuous = Path((tmp_path / 'continuous.path').read_text())
    partial = Path((tmp_path / 'partial.path').read_text())
    baseline = load_project_checkpoint(continuous / 'checkpoints/last.pt')
    resumed = load_project_checkpoint(partial / 'checkpoints/last.pt')
    for key in ('global_step', 'model', 'optimizer', 'external_policy', 'rng_state', 'family_scalarizer', 'training_monitor'):
        _equal(baseline[key], resumed[key], key)
    for run in (continuous, partial):
        rows = [json.loads(line) for line in (run / 'metrics/lr_warmup_updates.jsonl').read_text().splitlines()]
        assert [r['optimizer_lr'] for r in rows] == [pytest.approx(1e-4)]
    if os.environ.get('PHYCOFLOW_EXTERNAL_ASSESSMENT') == '1':
        for run in (continuous, partial):
            assert not (run / 'evaluation/before.json').exists()
            assert not (run / 'evaluation/after.json').exists()
            assert not (run / 'checkpoints/best.pt').exists()
            manifest = json.loads((run / 'run_manifest.json').read_text())
            assert manifest['before_metric'] is None and manifest['after_metric'] is None
            assert manifest['best_metric'] == {'name': None, 'value': None}
    for name in ('metrics/history.jsonl', 'logs/history.jsonl'):
        left = [json.loads(line) for line in (continuous / name).read_text().splitlines()]
        right = [json.loads(line) for line in (partial / name).read_text().splitlines()]
        assert len(left) == len(right) == 2
        for a, b in zip(left, right):
            assert list(a) == list(b)
            for key in a:
                if 'second' not in key:
                    _equal(a[key], b[key], key)


def test_ordered_recording_fresh_process_partial_epoch(tmp_path):
    worker = tmp_path / 'monitor_worker.py'
    worker.write_text('''
import json, sys
from pathlib import Path
from phycoflow_reconstruction.training.monitoring import TrainingMonitor
root, mode = Path(sys.argv[1]), sys.argv[2]
(root / "logs").mkdir(parents=True, exist_ok=True)
config = root / "resolved_config.yaml"
if not config.exists():
    config.write_text("coherence:\\n  families:\\n    state: {enabled: true}\\n    scale: {enabled: true}\\n  scalable_anchored_coherence:\\n    common_prefit: {path: unused}\\n")
TrainingMonitor._plot = lambda self: None
state_path = root / "state.json"
state = json.loads(state_path.read_text()) if mode == "resume" else None
start = 2 if mode == "resume" else 0
monitor = TrainingMonitor(root, start_step=start, final_step=4, configured_steps=4,
    steps_per_epoch=4, description="qualification", enabled=False, resume_state=state)
steps = (1, 2, 3, 4) if mode == "full" else ((1, 2) if mode == "partial" else (3, 4))
for step in steps:
    monitor.record({"step": step, "coherence_loss": float(step), "update_accepted": True})
if mode == "partial":
    state_path.write_text(json.dumps(monitor.checkpoint_state()))
monitor.close()
''')
    env = dict(os.environ)
    for run, mode in (('full', 'full'), ('resumed', 'partial'), ('resumed', 'resume')):
        subprocess.run([sys.executable, str(worker), str(tmp_path / run), mode], env=env, check=True)
    full, resumed = tmp_path / 'full/logs', tmp_path / 'resumed/logs'
    assert json.loads((resumed / 'history.jsonl').read_text())['coherence_loss'] == 2.5
    assert (full / 'history.jsonl').read_text() == (resumed / 'history.jsonl').read_text()
    assert json.loads((full / 'recording_definitions.json').read_text()) == json.loads((resumed / 'recording_definitions.json').read_text())
