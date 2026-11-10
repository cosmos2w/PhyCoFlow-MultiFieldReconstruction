"""Round 7 fixed-panel, field-only full-native endpoint and matched comparisons.

Inference is delegated unchanged to native_coherence; this wrapper neither
chunks native queries nor qualifies a new inference route.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch

from .native_coherence import _manifest, infer_native_panel, native_field_risk, require_matching_native_panels
from ..training.run_store import file_sha256

_NATIVE_SITES = 40300
_PANEL_SIZE = 100


@contextmanager
def _preserve_rng_and_mode(model, device):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    modes = [(module, module.training) for module in model.modules()]
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for module, mode in modes:
            module.training = mode


def _summary(risk_path, field_names, panel_size, expected_ids):
    with np.load(risk_path, allow_pickle=False) as risk:
        mse = np.asarray(risk['member_mse_physical'], dtype=np.float64)
        power = np.asarray(risk['reference_power_physical'], dtype=np.float64)
        names = risk['field_names'].tolist()
        ids = risk['sample_ids'].tolist()
    if (names != list(field_names) or len(set(ids)) != panel_size or len(ids) != panel_size
            or ids != list(expected_ids)
            or mse.shape != (panel_size, 2, len(names)) or power.shape != (panel_size, len(names))
            or not np.isfinite(mse).all() or not np.isfinite(power).all()
            or (mse < 0).any() or (power < 0).any()):
        raise ValueError('native risk arrays do not match the fixed J2 panel')
    # Equal condition weight; spatial area weights were normalized by native_field_risk.
    primary = mse[:, 0, :].mean(axis=0)
    pooled_power = power.mean(axis=0)
    relative = [math.sqrt(float(m / p)) if p > 0 else None for m, p in zip(primary, pooled_power)]
    return {'primary_mse_physical': primary.tolist(), 'primary_rmse_physical': np.sqrt(primary).tolist(),
            'reference_power_physical': pooled_power.tolist(), 'pooled_relative_l2_physical': relative,
            'primary_mse_per_condition_physical': mse[:, 0, :].tolist()}


def evaluate_round7_native(runtime, dataset, indices, output_dir, endpoint_config, *, panel_seed_base):
    """Evaluate exactly 100 fixed conditions over the native 40,300-site support.

    The endpoint requires the frozen observation manifest and live weights. A
    staging directory is published only after successful inference and receipt
    creation; incomplete staging retains condition records and a failure receipt.
    """
    positions = [int(i) for i in indices]
    if len(positions) != _PANEL_SIZE or len(set(positions)) != _PANEL_SIZE:
        raise ValueError('Round 7 native evaluation requires exactly 100 unique condition indices')
    if any(i < 0 or i >= len(dataset) for i in positions):
        raise ValueError('native panel index outside dataset')
    if dataset.split_name not in {'validation', 'test'}:
        raise ValueError('Round 7 native field evaluation requires validation or test')
    if endpoint_config.get('sensor_manifest') is None or endpoint_config.get('weight_selection') != 'live':
        raise ValueError('Round 7 native endpoint requires frozen sensor_manifest and live weights')
    frozen_manifest = _manifest(endpoint_config, dataset)
    if frozen_manifest is None:
        raise ValueError('Round 7 native evaluation requires a frozen sensor manifest')
    checkpoint_sha256 = file_sha256(runtime.checkpoint_path)
    endpoint_identity = {'sensor_manifest_sha256': frozen_manifest.digest(),
                         'checkpoint_sha256': checkpoint_sha256,
                         'steps': int(endpoint_config['steps']), 'solver': str(endpoint_config['solver']),
                         'observation_config': dict(endpoint_config['observation_config']),
                         'weight_selection': 'live', 'panel_seed_base': int(panel_seed_base), 'draws': 2}
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Round 7 native output already exists')
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}.', dir=output.parent))
    start = time.perf_counter()
    seen = []
    def check_support(arrays, row):
        if (row['point_count'] != _NATIVE_SITES or row['valid_count'] <= 0
                or row['valid_count'] > _NATIVE_SITES or not row['physical_measure']
                or len(row['draw_seeds']) != 2):
            raise ValueError('native condition does not cover the declared 40,300-site J2 support')
        if arrays['physical_mask'].shape != (_NATIVE_SITES,) or arrays['physical_weights'].shape != (_NATIVE_SITES,):
            raise ValueError('native mask/physical area shape mismatch')
        risk = native_field_risk(arrays['draws_physical'], arrays['target_physical'],
                                 arrays['physical_weights'], arrays['physical_mask'])
        record = {'sample_id': row['sample_id'], 'dataset_index': row['dataset_index'],
                  'contract_sha256': row['contract_sha256'], 'point_count': row['point_count'],
                  'valid_count': row['valid_count'], 'physical_measure': row['physical_measure'],
                  'draw_seeds': row['draw_seeds'], 'endpoint_identity': endpoint_identity,
                  'elapsed_seconds': time.perf_counter() - start,
                  'primary_mse_physical': risk['member_mse_physical'][0].tolist(),
                  'reference_power_physical': risk['reference_power_physical'].tolist()}
        with (staging / 'partial_conditions.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        seen.append(row['sample_id'])
    try:
        with _preserve_rng_and_mode(runtime.model, runtime.device):
            path = infer_native_panel(runtime, dataset, positions, staging, endpoint_config,
                                      panel_seed_base=int(panel_seed_base), draws=2,
                                      retain_payloads=False, consumer=check_support)
        manifest = json.loads(path.read_text(encoding='utf-8'))
        rows = manifest['conditions']
        if (len(rows) != _PANEL_SIZE or len(set(seen)) != _PANEL_SIZE
                or manifest['query_policy'] != 'full_native'
                or manifest['sampler']['path'] != 'full_query_differentiable_reconstruction'
                or manifest['sampler']['effective_weights'] != 'live'
                or manifest['sensor_manifest_sha256'] != endpoint_identity['sensor_manifest_sha256']
                or manifest['checkpoint_sha256'] != checkpoint_sha256
                or [r['dataset_index'] for r in rows] != positions):
            raise ValueError('native inference manifest does not match Round 7 endpoint')
        risk_path = staging / 'field_risk.npz'
        summary = _summary(risk_path, manifest['field_names'], _PANEL_SIZE, [r['sample_id'] for r in rows])
        receipt = {'schema': 'round7_native_field_only_v1', 'status': 'completed',
                   'qualification': 'existing_native_route; synthetic wrapper tests do not establish model qualification',
                   'dataset_split': manifest['dataset_split'], 'conditions': _PANEL_SIZE,
                   'native_sites_per_condition': _NATIVE_SITES, 'draws': 2,
                   'condition_weighting': 'equal', 'spatial_measure': 'per-condition normalized physical area on valid mask',
                   'field_names': manifest['field_names'], 'field_units': manifest['field_units'],
                   'checkpoint_sha256': manifest['checkpoint_sha256'],
                   'sensor_manifest_sha256': manifest['sensor_manifest_sha256'],
                   'manifest_sha256': file_sha256(path), 'risk_sha256': file_sha256(risk_path),
                   'inference_wall_seconds': manifest['wall_seconds'],
                   'wrapper_wall_seconds': time.perf_counter() - start,
                   'coverage': [{'sample_id': r['sample_id'], 'dataset_index': r['dataset_index'],
                                 'valid_count': r['valid_count'], 'physical_measure': r['physical_measure'],
                                 'contract_sha256': r['contract_sha256'], 'draw_seeds': r['draw_seeds']}
                                for r in rows],
                   **summary}
        receipt_path = staging / 'round7_native_receipt.json'
        temporary = staging / '.round7_native_receipt.tmp'
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
        os.replace(temporary, receipt_path)
        if output.exists():
            raise FileExistsError('Round 7 native output appeared during evaluation')
        os.rename(staging, output)
        return output / receipt_path.name
    except BaseException as exc:
        failure = {'schema': 'round7_native_field_only_v1', 'status': 'incomplete',
                   'exception_type': type(exc).__name__, 'exception_message': str(exc),
                   'elapsed_seconds': time.perf_counter() - start, 'completed_sample_ids': seen,
                   'planned_indices': positions, 'dataset_split': dataset.split_name,
                   'endpoint_identity': endpoint_identity,
                   'partial_conditions': 'partial_conditions.jsonl',
                   'note': 'Incomplete staging; never treat as a completed panel.'}
        temporary = staging / '.failure_receipt.tmp'
        temporary.write_text(json.dumps(failure, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
        os.replace(temporary, staging / 'failure_receipt.json')
        raise


def _load_receipt(directory):
    directory = Path(directory)
    receipt = json.loads((directory / 'round7_native_receipt.json').read_text(encoding='utf-8'))
    manifest_path, risk_path = directory / 'manifest.json', directory / 'field_risk.npz'
    if (receipt.get('schema') != 'round7_native_field_only_v1' or receipt.get('status') != 'completed'
            or receipt.get('conditions') != _PANEL_SIZE or receipt.get('native_sites_per_condition') != _NATIVE_SITES
            or receipt.get('draws') != 2 or file_sha256(manifest_path) != receipt.get('manifest_sha256')
            or file_sha256(risk_path) != receipt.get('risk_sha256')):
        raise ValueError('native receipt or payload identity mismatch')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest['checkpoint_sha256'] != receipt['checkpoint_sha256']
            or manifest['sensor_manifest_sha256'] != receipt['sensor_manifest_sha256']
            or manifest['dataset_split'] != receipt['dataset_split']
            or manifest['field_names'] != receipt['field_names']
            or manifest['query_policy'] != 'full_native'
            or manifest['sampler']['path'] != 'full_query_differentiable_reconstruction'
            or manifest['sampler']['effective_weights'] != 'live' or manifest['sampler']['draws'] != 2
            or len(manifest['conditions']) != _PANEL_SIZE
            or any(r['point_count'] != _NATIVE_SITES or not 0 < r['valid_count'] <= _NATIVE_SITES
                   or len(r['draw_seeds']) != 2 or not r['physical_measure'] for r in manifest['conditions'])
            or _summary(risk_path, receipt['field_names'], _PANEL_SIZE, [r['sample_id'] for r in manifest['conditions']]) !=
            {k: receipt[k] for k in ('primary_mse_physical', 'primary_rmse_physical',
                                      'reference_power_physical', 'pooled_relative_l2_physical',
                                      'primary_mse_per_condition_physical')}):
        raise ValueError('native receipt contents differ from inference artifacts')
    return receipt, manifest


def compare_round7_native(source_dir, candidate_dir):
    """Return per-field primary MSE ratios only after strict matched-panel checks."""
    source, source_manifest = _load_receipt(source_dir)
    candidate, candidate_manifest = _load_receipt(candidate_dir)
    require_matching_native_panels(source_manifest, candidate_manifest)
    a = np.asarray(source['primary_mse_physical'])
    b = np.asarray(candidate['primary_mse_physical'])
    if (a <= 0).any():
        raise ValueError('SOURCE primary MSE must be positive to form a ratio')
    return {'field_names': source['field_names'], 'source_mse_physical': a.tolist(),
            'candidate_mse_physical': b.tolist(), 'primary_mse_ratio': (b / a).tolist(),
            'source_checkpoint_sha256': source['checkpoint_sha256'],
            'candidate_checkpoint_sha256': candidate['checkpoint_sha256'],
            'dataset_split': source['dataset_split']}
