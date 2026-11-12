"""Frozen Round 8 populations, coupling identity and budget accounting.

No function in this module starts a job or reads an active run. Callers must
persist these records before consuming them and enforce the training watchdog.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def freeze_manifest(train, heldout, *, seed=47, exposed=None):
    """Select 800/200 from explicitly supplied 8000/2000 records.

    Each record has a unique ``id`` and optional ``regime``; heldout records
    also have ``source`` (validation/test). IDs are unique within their source.
    Exposure is a set of (source, id); None records unknown provenance.
    """
    train, heldout = list(train), list(heldout)
    if len(train) != 8000 or len(heldout) != 2000:
        raise ValueError('expected exactly 8000 TRAIN and 2000 held-out records')
    if any('id' not in x for x in train + heldout):
        raise ValueError('every record requires an id')
    if len({str(x['id']) for x in train}) != len(train):
        raise ValueError('duplicate TRAIN id')
    if any(x.get('source') not in ('validation', 'test') for x in heldout):
        raise ValueError('held-out source must be validation or test')
    if len({(x['source'], str(x['id'])) for x in heldout}) != len(heldout):
        raise ValueError('duplicate held-out identity')
    if {x['source'] for x in heldout} != {'validation', 'test'}:
        raise ValueError('both legacy sources required')
    rng = random.Random(seed)

    def sample(records, count, key, prefer):
        groups = defaultdict(list)
        for row in records:
            groups[key(row)].append(dict(row))
        keys = sorted(groups)
        quotas = {k: count * len(groups[k]) // len(records) for k in keys}
        remainder = count - sum(quotas.values())
        order = sorted(keys, key=lambda k: (-(count * len(groups[k]) % len(records)), k))
        for k in order[:remainder]:
            quotas[k] += 1
        result = []
        for k in keys:
            rows = sorted(groups[k], key=lambda x: (str(x.get('source', '')), str(x['id'])))
            rng.shuffle(rows)
            if prefer:
                rows.sort(key=lambda x: (x['source'], str(x['id'])) in exposed)
            result.extend(rows[:quotas[k]])
        return sorted(result, key=lambda x: (str(x.get('source', '')), str(x['id'])))

    tr = sample(train, 800, lambda x: (str(x.get('trajectory', 'unknown')), str(x.get('regime', 'unknown'))), False)
    ass = sample(heldout, 200, lambda x: (x['source'], str(x.get('trajectory', 'unknown')), str(x.get('regime', 'unknown'))), exposed is not None)
    return {'schema': 'round8_manifest_v1', 'seed': seed, 'TRAIN': tr, 'ASSESS': ass,
            'population_sizes': {'TRAIN': 8000, 'ASSESS': 2000},
            'exposure_provenance': 'prior exposure unknown' if exposed is None else 'supplied inventory; unexposed preferred',
            'train_source_sha256': canonical_hash(train), 'heldout_source_sha256': canonical_hash(heldout)}


def validate_t2000_receipt(receipt_path):
    """Validate a completed R7 continuation receipt and its pinned endpoint."""
    path = Path(receipt_path)
    receipt = json.loads(path.read_text())
    if receipt.get('status') != 'completed' or receipt.get('epoch') != 2000:
        raise ValueError('a completed epoch-2000 receipt is required')
    if not receipt.get('user_continuation') or not receipt.get('user_continuation_sha256'):
        raise ValueError('continuation authorization binding missing')
    if file_hash(receipt['user_continuation']) != receipt['user_continuation_sha256']:
        raise ValueError('continuation authorization changed')
    endpoint = Path(receipt['run']) / 'checkpoints/last.pt'
    if file_hash(endpoint) != receipt.get('last_checkpoint_sha256'):
        raise ValueError('completed endpoint differs from receipt')
    return {'receipt_sha256': file_hash(path), 'checkpoint': str(endpoint),
            'checkpoint_sha256': receipt['last_checkpoint_sha256'], 'epoch': 2000}


COUPLING_KEYS = ('manifest_sha256', 'population', 'condition_ids_sha256',
                 'observations_sha256', 'latent_inputs_sha256', 'prior_buffers_sha256',
                 'solver_sha256', 'normalizer_sha256', 'spatial_support_sha256',
                 'sensor_clamp_sha256', 'field_order_sha256', 'member_count')


def cache_identity(coupling, checkpoint_sha256):
    if set(coupling) != set(COUPLING_KEYS) or any(coupling[k] in ('', None) for k in COUPLING_KEYS):
        raise ValueError('complete coupling identity required')
    for key in COUPLING_KEYS:
        if key.endswith('_sha256') and (not isinstance(coupling[key], str) or len(coupling[key]) != 64 or any(c not in '0123456789abcdef' for c in coupling[key])):
            raise ValueError(f'{key} must be a SHA256 digest')
    if coupling['member_count'] != 2 or coupling['population'] not in ('TRAIN', 'ASSESS'):
        raise ValueError('Round 8 uses TRAIN/ASSESS and J=2')
    if (not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64
            or any(c not in '0123456789abcdef' for c in checkpoint_sha256)):
        raise ValueError('checkpoint hash required')
    return {'schema': 'round8_prediction_cache_v1', 'coupling': dict(coupling),
            'coupling_sha256': canonical_hash(coupling), 'checkpoint_sha256': checkpoint_sha256}


def assert_matched(left, right):
    for record in (left, right):
        if record != cache_identity(record['coupling'], record['checkpoint_sha256']):
            raise ValueError('invalid cache identity')
    if left['coupling'] != right['coupling']:
        raise ValueError('endpoint coupling differs')


def occupied_seconds(intervals, now):
    total = 0.0
    by_device = defaultdict(list)
    for item in intervals:
        start, end = float(item['start']), float(item.get('end', now))
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError('invalid occupied interval')
        by_device[item['device_uuid']].append((start, end))
    for spans in by_device.values():
        spans.sort()
        for i, (start, end) in enumerate(spans):
            if i and start < spans[i - 1][1]:
                raise ValueError('overlapping reservations on same GPU')
            total += end - start
    return total


def forecast(*, spent_seconds, update_seconds, callback_seconds, stage0_seconds,
             qualification_seconds, final_seconds, storage_bytes):
    values = (spent_seconds, update_seconds, callback_seconds, stage0_seconds,
              qualification_seconds, final_seconds, storage_bytes)
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('forecast requires finite nonnegative measurements')
    if update_seconds == 0 or final_seconds == 0 or storage_bytes == 0:
        raise ValueError('measured update, closeout and storage costs required')
    # Costs supplied here are remaining totals, including all scheduled events.
    training = 25870 * update_seconds + callback_seconds
    reserved = max(4 * 3600, final_seconds)
    projected = spent_seconds + stage0_seconds + qualification_seconds + training + reserved
    return {'schema': 'round8_forecast_v1', 'formal_unique_updates': 25870,
            'spent_seconds': spent_seconds, 'training_seconds': training,
            'stage0_remaining_seconds': stage0_seconds, 'qualification_remaining_seconds': qualification_seconds,
            'closeout_reserved_seconds': reserved, 'projected_seconds': projected,
            'cap_seconds': 32 * 3600, 'shortfall_seconds': max(0, projected - 32 * 3600),
            'fits': projected <= 32 * 3600, 'endpoint_cache_storage_bytes': storage_bytes}


def write_prediction_cache(path, *, identity, predictions, targets, weights):
    """Persist matched fields with a content-bound identity for reuse."""
    import numpy as np
    path = Path(path)
    identity = cache_identity(identity['coupling'], identity['checkpoint_sha256'])
    prediction = np.asarray(predictions)
    target = np.asarray(targets)
    weight = np.asarray(weights)
    if prediction.ndim != 4 or prediction.shape[1] != 2:
        raise ValueError('predictions must have [condition,member=2,field,site] shape')
    if target.shape != (prediction.shape[0], *prediction.shape[2:]):
        raise ValueError('target shape does not match predictions')
    if weight.shape != (prediction.shape[0], prediction.shape[-1]):
        raise ValueError('weights must have [condition,site] shape')
    if not all(np.isfinite(x).all() for x in (prediction, target, weight)):
        raise ValueError('cache must be finite')
    if (weight < 0).any():
        raise ValueError('physical weights must be nonnegative')
    if identity['coupling']['population'] == 'ASSESS' and not np.allclose(weight.sum(axis=-1), 1):
        raise ValueError('native ASSESS weights must sum to one')
    # TRAIN uses Horvitz-Thompson weights and must not be self-normalized.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, predictions=prediction, targets=target, weights=weight,
                            identity=json.dumps(identity, sort_keys=True))
    return {'schema': 'round8_cache_receipt_v1', 'path': str(path),
            'sha256': file_hash(path), 'identity': identity, 'shape': list(prediction.shape)}


def load_prediction_cache(receipt):
    import numpy as np
    if file_hash(receipt['path']) != receipt['sha256']:
        raise ValueError('prediction cache changed')
    with np.load(receipt['path'], allow_pickle=False) as data:
        identity = json.loads(str(data['identity']))
        if identity != receipt['identity']:
            raise ValueError('prediction identity changed')
        cache_identity(identity['coupling'], identity['checkpoint_sha256'])
        return {k: data[k].copy() for k in ('predictions', 'targets', 'weights')}, identity


def load_matched_prediction_caches(source_receipt, repair_receipt):
    import numpy as np
    source, left = load_prediction_cache(source_receipt)
    repair, right = load_prediction_cache(repair_receipt)
    assert_matched(left, right)
    for key in ('targets', 'weights'):
        if not np.array_equal(source[key], repair[key]):
            raise ValueError(f'matched cache {key} differ')
    return source, repair
