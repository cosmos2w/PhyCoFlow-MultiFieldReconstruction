"""Create an independently recorded identical-science continuation fork."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import torch
import yaml
from .run_store import RunStore, file_sha256


def _science(config):
    config = deepcopy(config)
    config.pop('output', None)
    config.get('runtime', {}).pop('device', None)
    return config


def create_control_fork(case_dir, config, parent_run, *, parent_checkpoint_sha256):
    """Create a new empty-log fork at a completed parent epoch; no updates run."""
    parent = Path(parent_run).resolve(strict=True)
    old = yaml.safe_load((parent/'resolved_config.yaml').read_text())
    if _science(old) != _science(config):
        raise ValueError('fork control requires identical science; candidate transition needs explicit implementation')
    checkpoint_path = parent/'checkpoints/epoch_500.pt'
    if file_sha256(checkpoint_path) != parent_checkpoint_sha256:
        raise ValueError('parent checkpoint changed')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    origin = int(checkpoint['global_step'])
    if origin != 6000 or checkpoint['external_policy']['completed_updates'] != origin or checkpoint['scalable_runtime']['step'] != origin:
        raise ValueError('parent checkpoint is not the expected action-6000 state')
    if checkpoint['training_monitor']['step'] != origin or checkpoint['training_monitor']['batch_count'] != 0:
        raise ValueError('parent is not on an ordered recording epoch boundary')
    parent_manifest = json.loads((parent/'run_manifest.json').read_text())
    store = RunStore.create(case_dir, config['output']['experiment_name'], config, parent_run=str(parent))
    shutil.copytree(parent/'artifacts', store.run_dir/'artifacts', dirs_exist_ok=True)
    for name in ('coherence_calibration_sha256', 'coherence_family_state_sha256s', 'common_core_receipt_sha256'):
        if name in parent_manifest:
            store.update_manifest(**{name:parent_manifest[name]})
    checkpoint['config_sha256'] = store.config_hash
    checkpoint['fork_origin'] = {'parent_run':str(parent), 'parent_checkpoint_sha256':parent_checkpoint_sha256, 'absolute_step':origin}
    store.save_checkpoint('last', checkpoint)
    store.update_manifest(fork_origin=checkpoint['fork_origin'])
    store.set_status('fork_initialized', global_step=origin, reporting_epoch=0, absolute_reporting_epoch=origin/12)
    return store.run_dir
