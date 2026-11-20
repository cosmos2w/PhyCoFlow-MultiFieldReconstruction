"""Fresh-process fixture worker for epoch-warmup resume qualification."""
import sys
import os
from pathlib import Path
import yaml

from helpers.coherence import _base_config, _post_config, _write_fixture
from helpers.multifamily import _global_config
from phycoflow_reconstruction.training.base_training import run_base_training
from phycoflow_reconstruction.training.post_training import run_post_training

root, mode, device = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
case = root / 'case'
if mode == 'prepare':
    root.mkdir(parents=True, exist_ok=True)
    data = root / 'fixture.h5'
    _write_fixture(data)
    base = _base_config(data)
    base['runtime']['device'] = device
    source = run_base_training(base, case_dir=case)
    config = _post_config(data, source)
    config['runtime'].update(device=device, progress=False)
    config['optimization'].update(epochs=2, steps_per_epoch=1, sampling='full_pass',
        lr=1e-3, lr_warmup={'enabled': True, 'start_lr': 1e-4, 'epochs': 1},
        multitask={'family_scalarization': {'method': 'dwa'}, 'outer_aggregation': None,
                   'external_policy': {'method': 'lagrangian', 'vector_reduction': 'sum'},
                   'fidelity_budget': {'calibration_batches': 2, 'calibration_draws': 2}})
    config['objectives']['data_retention'] = {'enabled': False, 'weight': 0.0}
    config['coherence']['compute_budget']['batch_size'] = 2
    config['coherence']['families'] = {'global_distribution': _global_config()}
    config['coherence']['family_balance'] = {'mode': 'initial_grad_norm', 'calibration_batches': 2}
    config['checkpointing'] = {'every_epochs': 1, 'selection_metric': 'reconstruction_mse'}
    config['evaluation']['preview'] = {'enabled': False}
    config['evaluation']['external_assessment'] = os.environ.get('PHYCOFLOW_EXTERNAL_ASSESSMENT') == '1'
    (root / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
elif mode in ('continuous', 'partial'):
    config = yaml.safe_load((root / 'config.yaml').read_text())
    config['output']['experiment_name'] = mode
    child = run_post_training(config, case_dir=case, max_steps=1 if mode == 'partial' else None)
    (root / (mode + '.path')).write_text(str(child))
elif mode == 'resume':
    child = Path((root / 'partial.path').read_text())
    config = yaml.safe_load((child / 'resolved_config.yaml').read_text())
    run_post_training(config, case_dir=case, resume=child, max_steps=1)
else:
    raise ValueError(mode)
