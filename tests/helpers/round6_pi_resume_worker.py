"""Fresh-process model, AdamW, PI and ordered-monitor resume fixture."""
import random
import sys
from pathlib import Path
import numpy as np
import torch
import yaml
from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.training.fidelity_budget import ExternalPolicy, MeanRiskDifferenceBudget
from phycoflow_reconstruction.training.monitoring import TrainingMonitor
from phycoflow_reconstruction.training.multitask import OuterAggregator

root, mode = Path(sys.argv[1]), sys.argv[2]
root.mkdir(parents=True, exist_ok=True)
(root / 'logs').mkdir(exist_ok=True)
(root / 'metrics').mkdir(exist_ok=True)
config_path = root / 'resolved_config.yaml'
if mode != 'resume':
    config_path.write_text(yaml.safe_dump({'coherence': {'scalable_anchored_coherence': {'common_prefit': True}, 'families': {'state': {}, 'scale': {}, 'shape': {}}}, 'policy': {'method': 'lagrangian', 'lagrangian': {'epoch_calibrated_pi': {'gamma': 3.0, 'tau_epochs': 1.0, 'horizon_epochs': 10.0, 'steps_per_epoch': 12}}}, 'coefficients': [0.7, 1.1, 1.4]}, sort_keys=False))
    random.seed(27); np.random.seed(27); torch.manual_seed(27)
config = load_config(config_path)
model = torch.nn.Linear(3, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
reference = MeanRiskDifferenceBudget.from_reference(['one', 'two'], [0.2, -0.1], [0.05, 0.1], [0.5, 0.8], 1.3)
policy = ExternalPolicy(config['policy'], reference, 'cpu')
state = None
start = 0
if mode == 'resume':
    state = torch.load(root / 'checkpoint.pt', weights_only=False)
    assert state['coefficients'] == config['coefficients']
    model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); policy.load_state_dict(state['policy'])
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    start = state['step']
monitor = TrainingMonitor(root, start_step=start, final_step=24, configured_steps=24, steps_per_epoch=12, description='fixture', enabled=False, resume_state=None if state is None else state['monitor'])
end = 17 if mode == 'partial' else 24
for step in range(start, end):
    x = torch.randn(5, 3) + random.random() * 0.1 + float(np.random.random()) * 0.1
    y = model(x)
    families = {name: (y[:, i % 2] - (i + 1) * .2).square().mean() * config['coefficients'][i] for i, name in enumerate(('state', 'scale', 'shape'))}
    risks = torch.stack((y[:, 0].mean() + .2, y[:, 1].mean() - .1))
    diagnostics = policy.update_with_coherence_aggregation(model, optimizer, families, risks, None, OuterAggregator({'method': 'torchjd_config'}))
    monitor.record({'step': step + 1, 'coherence_loss': diagnostics['coherence_value'], 'field_violation': diagnostics['field_violation'], 'coherence_family/state_loss': float(families['state'].detach()), 'coherence_family/scale_loss': float(families['scale'].detach()), 'coherence_family/shape_loss': float(families['shape'].detach())})
checkpoint = {'step': end, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'policy': policy.state_dict(), 'monitor': monitor.checkpoint_state(), 'coefficients': config['coefficients'], 'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
torch.save(checkpoint, root / 'checkpoint.pt')
monitor.close()
