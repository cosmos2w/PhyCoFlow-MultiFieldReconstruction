#!/usr/bin/env python3
"""Qualify an identical-science phase fork and its append in fresh processes.

The existing longer source/data-stream qualification remains applicable. This
small real-model check specifically covers PI state and mid-epoch recording.
It trains two reporting epochs twice, never performs a model metric evaluation,
and must run inside one enclosing preparation lease.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(os.environ.get('CAR_EXECUTION_ROOT', os.environ.get('CAR_REPO_ROOT', Path(__file__).resolve().parents[2]))).resolve()
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from phycoflow_reconstruction.evaluation.adaptive_campaign import atomic_json, digest
from scripts.posttraining.qualify_car_round2_resume import (
    differences, rows, scientific_checkpoint_view, scientific_history_view,
)


def worker(args):
    import torch
    import yaml
    from phycoflow_reconstruction.training.post_training import run_post_training
    torch.set_num_threads(4)
    cfg = yaml.safe_load((args.resume / 'resolved_config.yaml' if args.resume else args.config).read_text())
    if args.resume is None:
        from phycoflow_reconstruction.training.fork_control import create_control_fork
        cfg['output']['experiment_name'] = 'qualification/IdenticalScaleFork_ProductionResume'
        args.resume = create_control_fork(args.case_dir, cfg, args.parent, parent_checkpoint_sha256=args.parent_sha256)
    pi = cfg['optimization']['multitask']['external_policy']['lagrangian'].get('epoch_calibrated_pi')
    if pi is None:
        raise ValueError('the affected PI configuration is required')
    started = time.time()
    run = run_post_training(cfg, case_dir=args.case_dir,
        resume=args.resume, budget_stop_requested=lambda step, _run: step >= args.stop_action)
    atomic_json(args.output, {'run_dir': str(run), 'pid': os.getpid(),
        'resume': str(args.resume) if args.resume else None, 'stop_action': args.stop_action,
        'start_unix': started, 'end_unix': time.time()})


def compare(uninterrupted, resumed):
    import torch
    a, b = [torch.load(path / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
            for path in (uninterrupted, resumed)]
    if a['global_step'] != 6024 or b['global_step'] != 6024:
        raise ValueError('both real-model trajectories must end after two reporting epochs')
    diff = differences(scientific_checkpoint_view(a), scientific_checkpoint_view(b))
    definitions = [json.loads((p / 'logs/recording_definitions.json').read_text())
                   for p in (uninterrupted, resumed)]
    diff += differences(*definitions, path='recording_definitions')
    histories = [scientific_history_view(rows(p / 'logs/history.jsonl'), d)
                 for p, d in zip((uninterrupted, resumed), definitions, strict=True)]
    diff += differences(*histories, path='history')
    for relative in ('logs/epoch_calibrated_pi_updates.jsonl', 'metrics/training_coupling.jsonl'):
        values = [rows(p / relative) for p in (uninterrupted, resumed)]
        if any([r['step'] for r in value] != list(range(6001, 6025)) for value in values):
            raise ValueError(f'contiguous per-action append history required: {relative}')
        diff += differences(*values, path=relative)
    for path in (uninterrupted, resumed):
        for name in ('loss_history.png', 'loss_history.pdf'):
            if not (path / 'logs' / name).is_file():
                raise ValueError(f'missing required pause/end curve: {path / "logs" / name}')
    if diff:
        raise ValueError('production fresh-process resume differs: ' + '; '.join(diff[:20]))
    return {'uninterrupted': str(uninterrupted), 'resumed': str(resumed),
            'checkpoint_sha256': [digest(p / 'checkpoints/last.pt') for p in (uninterrupted, resumed)],
            'recording_definitions_sha256': [digest(p / 'logs/recording_definitions.json') for p in (uninterrupted, resumed)]}


def parent_and_source_identity(parent, config):
    """Bound checkpoint, parent manifest/artifacts, and source before and after."""
    import yaml
    from phycoflow_reconstruction.training.source import source_hashes
    cfg = yaml.safe_load(Path(config).read_text())
    parent = Path(parent).resolve(strict=True)
    required = [parent/'resolved_config.yaml', parent/'run_manifest.json', parent/'status.json', parent/'checkpoints/epoch_500.pt']
    required += sorted(p for p in (parent/'artifacts').rglob('*') if p.is_file())
    if len(required) == 4: raise ValueError('parent artifacts missing')
    source = Path(cfg['source_run']).resolve(strict=True)
    additional = {name:digest(source/name) for name in ('status.json','snapshot_receipt.json') if (source/name).is_file()}
    return {'parent_run':str(parent), 'parent_files':{str(p.relative_to(parent)):digest(p) for p in required},
            'source_run':str(source), 'source_hashes':source_hashes(cfg), 'source_extra':additional}


def orchestrate(args):
    if not os.environ.get('CAR_ROUND6_LEASE_TOKEN'):
        raise ValueError('run under the Round 6 preparation lease')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    receipt = {'schema': 'car_scale_fork_resume_v1', 'status': 'failed',
               'config_sha256': digest(args.config), 'start_unix': started}
    paths = []
    before = None
    try:
        before = parent_and_source_identity(args.parent, args.config)
        receipt['parent_source_before'] = before
        if before['parent_files']['checkpoints/epoch_500.pt'] != args.parent_sha256:
            raise ValueError('qualification parent checkpoint differs')
        for name, stop, resume in [('continuous', 6024, None), ('pause', 6017, None),
                                    ('resume', 6024, 'previous')]:
            output = args.output / f'{name}.json'
            command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                '--config', str(args.config), '--output', str(output), '--stop-action', str(stop), '--parent', str(args.parent), '--parent-sha256', args.parent_sha256, '--case-dir', str(args.case_dir)]
            if resume:
                command += ['--resume', paths[-1]['run_dir']]
            with (args.output / f'{name}.log').open('x') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            paths.append(json.loads(output.read_text()))
        if len({p['pid'] for p in paths}) != 3 or paths[2]['run_dir'] != paths[1]['run_dir']:
            raise ValueError('three distinct processes and a real saved-run resume required')
        receipt.update(compare(Path(paths[0]['run_dir']), Path(paths[2]['run_dir'])))
        after = parent_and_source_identity(args.parent, args.config)
        receipt['parent_source_after'] = after
        if before != after: raise ValueError('parent or SOURCE bytes changed during qualification')
        receipt['parent_source_immutable_verified'] = True
        receipt.update(status='completed', qualified=True, pause_action=17,
                       final_reporting_epoch=2, action_visits=48*64,
                       saved_config_reloaded=True, native_loss=False,
                       reused_qualification='Existing SOURCE/data-stream crossing qualification; affected PI+journal+mid-epoch production append checked here.')
    except BaseException as exc:
        receipt.update(error_type=type(exc).__name__, error=str(exc), qualified=False)
        raise
    finally:
        if before is not None and 'parent_source_after' not in receipt:
            try:
                after = parent_and_source_identity(args.parent, args.config)
                receipt['parent_source_after'] = after
                receipt['parent_source_immutable_verified'] = before == after
            except Exception as exc:
                receipt['parent_source_verification_error'] = str(exc)
        receipt.update(end_unix=time.time(), wall_seconds=time.time()-started)
        atomic_json(args.output / 'receipt.json', receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--stop-action', type=int, choices=(6017, 6024))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--parent-sha256', required=True)
    parser.add_argument('--case-dir', type=Path, required=True)
    args = parser.parse_args()
    (worker if args.worker else orchestrate)(args)


if __name__ == '__main__':
    main()
