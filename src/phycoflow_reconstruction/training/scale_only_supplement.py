"""Narrow provenance contract for the fixed Round-6 Scale supplement."""
import hashlib
import json
from copy import deepcopy

import torch

from ..evaluation.loss_aligned import model_state_digest
from ..models.flows.pointcloud.runtime.ema import ModelEMA
from .fidelity_reference import load_shared_reference, shared_reference_binding
from .run_store import file_sha256, load_model_state_strict

SOURCE_SHA = '2d1da713538f6e9613560f3a7eb4a9df2a95d1ad1f56dd11a1a8a4f9ef65050c'
REFERENCE_SHA = 'faf360b45dc9cda79c868ea3eebed339ba14129acf6798ce32eaf1e53c0922c2'
INIT_SHA = '6316aca32cb3214acced9b0f7afe51e76c5ba1cafb58e12f640a03bfc083ea54'
MODEL_SHA = '78bad910fc601c4d7c77a274e1839a0aac98f50d72b0ede944b6b7d9a3f549b1'
SCALE_SHA = '01d3b9575d98558fa6f0fe808fe28f646d5c53f93dc9fd28bc0449e15b660e83'
C0 = 1.9418972484616779
SOURCE_VALIDATION_SHA = '146e5763e42780418186522b49f77407b614e9795b12a2f4bcfe8810a2e7a463'
EVALUATION_MANIFEST_ID = '77ae5b743ae500800f0a1c26c99ccdd433b4a6360d5be8dea8cd92c752704a10'
SOURCE_MODEL_SHA = '3336fbeb7c17ec792b597e1784b2cb55edf83a58559aa62422c0e7c94be61fb5'


def load_source_validation_metrics(path, field_names):
    """Load frozen matched SOURCE validation reductions for reporting only."""
    from pathlib import Path

    from ..evaluation.loss_aligned import legacy_metrics

    if list(field_names) != ['CH4', 'CO', 'T', 'U_1', 'p']:
        raise ValueError('supplement SOURCE validation field order mismatch')
    if file_sha256(path) != SOURCE_VALIDATION_SHA:
        raise ValueError('supplement SOURCE validation reductions hash mismatch')
    payload = json.loads(Path(path).read_text())
    if (payload.get('manifest_sha256') != EVALUATION_MANIFEST_ID
        or payload.get('weight_identity_kind') != 'model_state_sha256'
        or payload.get('weight_sha256') != SOURCE_MODEL_SHA
        or payload.get('split') != 'validation' or len(payload.get('rows', [])) != 100):
        raise ValueError('supplement SOURCE validation identity mismatch')
    metrics = legacy_metrics(path, field_names)
    metrics['weight_provenance'] = 'original_SOURCE_live'
    metrics['source_validation_reductions_sha256'] = SOURCE_VALIDATION_SHA
    metrics['evaluation_manifest_sha256'] = EVALUATION_MANIFEST_ID
    return metrics


def validate_recipe(config, baseline):
    """Reject all changes to the bound family recipe except the authorized subset."""
    from ..coherence.relation_queries import resolve_coverage_config
    base = deepcopy(baseline)
    base['coherence'] = resolve_coverage_config(base['coherence'], base['dataset']['field_names'])
    active = deepcopy(config)
    bf, af = base['coherence']['families'], active['coherence']['families']
    for name in ('global_distribution', 'topology'):
        if bf[name].get('enabled') is not True or af[name].get('enabled') is not False:
            raise ValueError('supplement must disable exactly State and Shape')
        af[name]['enabled'] = True
    if af != bf or active['coherence'].get('query_coverage', {}).get('enabled', False):
        raise ValueError('supplement changed the frozen family recipe or retained sampler')
    frozen = active['coherence'].get('frozen_family_artifacts', {})
    if set(frozen) != {'multiscale_spectrum'} or frozen['multiscale_spectrum'].get('sha256') != SCALE_SHA:
        raise ValueError('supplement requires the pinned Scale artifact only')
    opt = active['optimization']; mt = opt['multitask']
    expected_opt = deepcopy(base['optimization'])
    expected_opt['epochs'] = 100
    expected_opt['multitask'].pop('coherence_aggregation')
    expected_opt['multitask']['family_scalarization'] = {'method': 'fixed_sum'}
    if opt != expected_opt:
        raise ValueError('supplement optimization differs from the fixed recipe')
    if ('coherence_aggregation' in mt or mt.get('family_scalarization') != {'method': 'fixed_sum'}
        or mt.get('external_policy') != base['optimization']['multitask']['external_policy']
        or mt.get('fidelity_budget') != base['optimization']['multitask']['fidelity_budget']
        or opt.get('lr') != 5e-5 or opt.get('weight_decay') != 1e-6
        or opt.get('grad_clip') != 1.0 or opt.get('model_mode') != 'eval'
        or opt.get('lr_warmup') != {'enabled': True, 'start_lr': 5e-6, 'updates': 76}
        or opt.get('batch_size') != 32 or opt.get('steps_per_epoch') != 13
        or opt.get('epochs') != 100 or active['runtime'].get('seed') != 43
        or active['trainable'].get('scope') != 'full_model'
        or active['model'].get('model_ema_enabled') is not True
        or active['model'].get('model_ema_eval') is not False
        or active['model'].get('model_ema_decay') != 0.999):
        raise ValueError('supplement optimization or lifecycle contract changed')
    for key in ('source_run', 'source_checkpoint', 'objectives', 'rollout', 'observation_consistency'):
        if active.get(key) != base.get(key):
            raise ValueError(f'supplement changed {key}')
    for key in ('compute_budget', 'family_balance', 'schedule'):
        if active['coherence'].get(key) != base['coherence'].get(key):
            raise ValueError(f'supplement changed coherence {key}')
    return base


def bind_reference(config, baseline, *, reference_path, source_hashes, binding_arguments):
    if source_hashes.get('checkpoint') != SOURCE_SHA:
        raise ValueError('supplement requires original SOURCE')
    parent = shared_reference_binding(baseline, source_hashes=source_hashes,
        family_scales={'global_distribution': 1.0, 'multiscale_spectrum': 1.0, 'topology': 1.0}, **binding_arguments)
    reference, raw = load_shared_reference(reference_path, REFERENCE_SHA, parent)
    if reference.objective_scale != C0:
        raise ValueError('supplement requires original C0')
    # Reconstruct the complete parent anchor with only the two authorized flags changed.
    candidate = deepcopy(config)
    candidate['coherence']['families']['global_distribution']['enabled'] = True
    candidate['coherence']['families']['topology']['enabled'] = True
    candidate_binding = shared_reference_binding(candidate, source_hashes=source_hashes,
        family_scales={'global_distribution': 1.0, 'multiscale_spectrum': 1.0, 'topology': 1.0}, **binding_arguments)
    if candidate_binding != parent:
        raise ValueError('supplement changed SOURCE calibration binding')
    binding = {'schema': 'round6_scale_only_subset_v1', 'parent_shared_reference_sha256': REFERENCE_SHA,
        'source_checkpoint_sha256': SOURCE_SHA, 'initializer_checkpoint_sha256': INIT_SHA,
        'source_validation_reductions_sha256': SOURCE_VALIDATION_SHA,
        'evaluation_manifest_sha256': EVALUATION_MANIFEST_ID,
        'initializer_model_sha256': MODEL_SHA, 'scale_artifact_sha256': SCALE_SHA,
        'active_families': ['multiscale_spectrum'], 'disabled_families': ['global_distribution', 'topology'],
        'sampler': 'disabled', 'objective_scale': C0,
        'optimizer': {'name': 'AdamW', 'lr': 5e-5, 'weight_decay': 1e-6, 'betas': [0.9, 0.999], 'eps': 1e-8, 'amsgrad': False},
        'warmup': {'start_lr': 5e-6, 'updates': 76}, 'duals': {'initial': 0, 'lr': 0.01},
        'ema': {'decay': 0.999, 'updates': 0, 'eval': False}}
    return reference, raw, binding


def initialize_live_model(model, path):
    if file_sha256(path) != INIT_SHA:
        raise ValueError('supplement initializer checkpoint hash mismatch')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    class SavedState:
        def state_dict(self):
            return checkpoint['model']
    if model_state_digest(SavedState()) != MODEL_SHA:
        raise ValueError('supplement initializer model hash mismatch')
    load_model_state_strict(model, checkpoint['model'])
    if getattr(model, '_ema', None) is None or getattr(model, '_ema_eval', True):
        raise ValueError('supplement requires enabled live-eval EMA')
    model._ema = ModelEMA(model, decay=0.999)


def binding_digest(binding):
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
