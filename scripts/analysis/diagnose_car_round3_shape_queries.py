#!/usr/bin/env python3
"""Replay shared and per-condition Shape queries on fixed cached TRAIN endpoints."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import signal
from pathlib import Path
import statistics
import time

import torch
import yaml
from phycoflow_reconstruction.coherence import build_enabled_families
from phycoflow_reconstruction.coherence.query_runtime import _shape_grid
from phycoflow_reconstruction.coherence.query_sampling import SharedQuerySampler
from phycoflow_reconstruction.coherence.shape_compute import LazyPooledShape
from phycoflow_reconstruction.data.factory import open_field_dataset
from phycoflow_reconstruction.training.run_store import file_sha256
from phycoflow_reconstruction.training.scalable_fitting import validate_scalable_artifacts


def query_plan(fields, seed, n, batch_size):
    """Both arms draw from the exact production law; first condition is coupled."""
    shared, diverse = [], []
    # Distinct fixed streams per batch and condition; the first condition shares
    # the query to provide a meaningful paired common-random-numbers anchor.
    for start in range(0, n, batch_size):
        count = min(batch_size, n - start)
        batch_seed = seed * 1000003 + start
        q = SharedQuerySampler(fields, seed=batch_seed, epsilon=1.0).draw()
        shared.extend([q] * count)
        diverse.append(q)
        diverse.extend(SharedQuerySampler(fields, seed=batch_seed + i * 7919, epsilon=1.0).draw()
                       for i in range(1, count))
    return shared, diverse


_save_terminal = None


class BudgetExpired(Exception):
    pass


def replay(shape, grids, mask, queries, batch_size, deadline, progress):
    # Group identical query batches to keep persistence packing bounded. Grouping
    # does not alter the estimator or coalesced inner draws.
    groups = defaultdict(list)
    for index, query in enumerate(queries):
        groups[query].append(index)
    sums = {'self': 0., 'mutual': 0.}
    started = time.perf_counter()
    progress.update(conditions_scored=0, panel_conditions=len(queries))
    for query, indexes in groups.items():
        for start in range(0, len(indexes), batch_size):
            if time.monotonic() >= deadline:
                raise BudgetExpired('worker time exhausted')
            ids = indexes[start:start + batch_size]
            values = shape.score(grids[0][ids], grids[1][ids], grids[2][ids], query,
                                 valid_mask=mask, mode='conditional')
            for key in sums:
                sums[key] += float(values[key].double().sum())
            progress['conditions_scored'] += len(ids)
            progress['scorer_seconds'] = time.perf_counter() - started
    return {key: value / len(queries) for key, value in sums.items()}, time.perf_counter() - started


def main():
    global _save_terminal
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-seeds', type=int, default=8)
    parser.add_argument('--worker-seconds-limit', type=float, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_seeds <= 8 or args.threads != 1 or args.batch_size != 64 or not 0 < args.worker_seconds_limit <= 7200:
        parser.error('max seeds 1..8, threads 1..4 and training batch size 64 required')
    started = time.monotonic()
    deadline = started + args.worker_seconds_limit
    args.output.mkdir(parents=True, exist_ok=False)
    record = {'status': 'initializing', 'worker_seconds_limit': args.worker_seconds_limit, 'seeds': []}
    def save():
        (args.output / 'results.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    def save_terminal():
        signal.setitimer(signal.ITIMER_REAL, 0)
        record['status'] = 'deadline_expired'
        record['elapsed_worker_seconds'] = time.monotonic() - started
        save()
    _save_terminal = save_terminal
    def expired(signum, frame):
        raise BudgetExpired('worker time deadline reached')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, args.worker_seconds_limit)
    save()
    torch.set_num_threads(args.threads)
    source = json.loads(args.setup.read_text())['sources']['ffm']['source_fit']
    run = Path(source['run']); panel_path = Path(source['source_panel']); core_path = Path(source['core'])
    for path, expected in ((panel_path, source['source_panel_sha256']), (core_path, source['core_sha256'])):
        if file_sha256(path) != expected:
            raise ValueError(f'input hash mismatch: {path}')
    panel = torch.load(panel_path, map_location='cpu', weights_only=True)
    core = torch.load(core_path, map_location='cpu', weights_only=True)
    validate_scalable_artifacts(core, panel['provenance'])
    if panel['schema'] != 'car_source_panel_v1' or len(panel['sample_ids']) != 800 or panel['sample_ids'] != panel['provenance']['sample_ids'] or panel['members'].shape[1] < 2:
        raise ValueError('expected matched TRAIN800 SOURCE J2 panel')
    config = yaml.safe_load((run / 'resolved_config.yaml').read_text())
    dataset = open_field_dataset(config['dataset'], split='train')
    family = build_enabled_families(config['coherence'], dataset.data_spec, dataset.normalizer)['topology']
    manifest = json.loads((run / 'run_manifest.json').read_text())
    family_path = run / 'artifacts/topology_family.pt'
    if file_sha256(family_path) != manifest['coherence_family_state_sha256s']['topology']:
        raise ValueError('topology family hash mismatch')
    family.load_state_artifact(torch.load(family_path, map_location='cpu', weights_only=True))
    shape = LazyPooledShape.from_state_dict(core['shape_artifact'])
    family._raster_map(panel['coordinates'])
    mask = family.domain_mask if family.native_masked else None
    # Reuse only immutable fixed endpoint transforms across the two replay laws.
    grids = [[], [], []]
    with torch.no_grad():
        for start in range(0, 800, args.batch_size):
            sl = slice(start, start + args.batch_size); ref = panel['reference'][sl]
            for out, data in zip(grids, (panel['members'][sl, 0], panel['members'][sl, 1], ref)):
                out.append(_shape_grid(family, data, ref))
        grids = [torch.cat(v) for v in grids]
        weights = {k: float(family.spatial_objective.weights[f'{k}.persistence']) for k in ('self', 'mutual')}
        # Report physical-weighted raw family scores as well as raw blocks. Frozen
        # learned coefficients are not assumed to be present in the source core.
        record.update({'schema': 'car_round3_shape_query_diagnostic_v1', 'status': 'running', 'panel': str(panel_path),
                  'panel_sha256': source['source_panel_sha256'], 'core_sha256': source['core_sha256'],
                  'family_sha256': file_sha256(family_path), 'sample_count': 800,
                  'sample_ids_sha256': hashlib.sha256(json.dumps(panel['sample_ids']).encode()).hexdigest(),
                  'members': [0, 1], 'batch_size': args.batch_size, 'threads': args.threads,
                  'physical_weights': weights, 'sampler': 'uniform outer, production inner law; duplicate outer pairs share inner draw',
                  'scale_shape_coupling': 'Shape-only replay; training shares outer queries with Scale',
                  'worker_seconds_budget': args.worker_seconds_limit})
        save()
        used = 0.
        for seed in range(1, args.max_seeds + 1):
            shared, diverse = query_plan(tuple(family.field_names), seed, 800, args.batch_size)
            entry = {'seed': seed}
            for name, plan in (('shared', shared), ('condition_specific', diverse)):
                record['partial_pair'] = {'seed': seed, 'arm': name, 'completed_arms': dict(entry)}
                save()
                try:
                    result, seconds = replay(shape, grids, mask, plan, args.batch_size, deadline, record['partial_pair'])
                except BudgetExpired:
                    record['status'] = 'deadline_expired_partial_pair'
                    record['elapsed_worker_seconds'] = time.monotonic() - started
                    save()
                    return
                result['physical_weighted_total'] = sum(weights[k] * result[k] for k in weights)
                entry[name] = {'scores': result, 'scorer_seconds': seconds}
                used += seconds * args.threads
            record['seeds'].append(entry)
            record.pop('partial_pair', None)
            # Forecast from the full paired replay; never start a replay whose
            # measured expected worker cost would exceed the allowance.
            cost = (entry['shared']['scorer_seconds'] + entry['condition_specific']['scorer_seconds']) * args.threads
            record['scorer_worker_seconds'] = used
            record['next_pair_forecast_worker_seconds'] = cost
            (args.output / 'results.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
            if time.monotonic() - started + cost > args.worker_seconds_limit:
                break
    signal.setitimer(signal.ITIMER_REAL, 0)
    record['status'] = 'completed'
    record['elapsed_worker_seconds'] = time.monotonic() - started
    for arm in ('shared', 'condition_specific'):
        record[arm + '_dispersion'] = {k: {'mean': statistics.mean(v), 'sample_sd': statistics.stdev(v) if len(v) > 1 else None}
            for k in ('self', 'mutual', 'physical_weighted_total')
            for v in [[x[arm]['scores'][k] for x in record['seeds']]]}
    (args.output / 'results.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except BudgetExpired:
        if _save_terminal is not None:
            _save_terminal()
