#!/usr/bin/env python3
"""Run at most two reporting epochs of a resolved Round 5 startup recipe.

GPU reservation, campaign lease, and timeout belong to the invoking launcher.
This driver does not alter or retune the supplied scientific configuration.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='resolved frozen startup YAML')
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--historical-timing', action='store_true', help='one complete old-route epoch')
    parser.add_argument('--epochs', type=int, choices=(1, 2), default=2,
                        help='absolute reporting-epoch endpoint of this startup trace, including any resumed prefix')
    args = parser.parse_args()
    from phycoflow_reconstruction.config import load_config
    from phycoflow_reconstruction.training.post_training import run_post_training
    config = load_config(args.config)
    direct = config['optimization'].get('direct_supervision')
    if args.historical_timing:
        if direct or 'external_policy' not in config['optimization'].get('multitask', {}) or args.epochs != 1 or args.resume is not None:
            raise ValueError('historical timing requires the old external policy, one epoch, and no resume')
    elif not direct or direct['mode'] not in ('native', 'safeguarded'):
        raise ValueError('startup trace requires an explicit Round 5 direct policy')
    steps_per_epoch = int(config['optimization']['steps_per_epoch'])
    target_step = steps_per_epoch * args.epochs
    start_step = 0
    saved_config_sha256 = None
    if args.resume is not None:
        from phycoflow_reconstruction.training.run_store import RunStore, load_project_checkpoint
        from phycoflow_reconstruction.evaluation.adaptive_campaign import digest
        # Validate the prepared config against the run's immutable saved config,
        # then run from the saved definitions and ordering in this fresh process.
        config = RunStore.resume(args.resume, config).config
        saved_config_sha256 = digest(args.resume / 'resolved_config.yaml')
        start_step = int(load_project_checkpoint(args.resume / 'checkpoints/last.pt')['global_step'])
    steps = target_step - start_step
    if steps <= 0:
        raise ValueError('startup trace already reached the requested absolute epoch endpoint')
    if int(config['runtime'].get('plot_every_steps', 10)) != 10:
        raise ValueError('startup trace must retain the 10-epoch plot cadence')
    token = os.environ.get('CAR_ROUND5_LEASE_TOKEN')
    if not token:
        raise ValueError('startup traces require a measured Round 5 GPU lease')
    start = time.time()
    receipt = {'schema': 'car_round5_training_trace_v1', 'status': 'failed', 'config': str(args.config),
               'resume': None if args.resume is None else str(args.resume), 'attempted_steps': steps,
               'reporting_epochs': args.epochs, 'historical_timing': args.historical_timing, 'start_step': start_step,
               'target_step': target_step, 'lease_token': token, 'saved_config_reloaded': args.resume is not None,
               'saved_config_sha256': saved_config_sha256, 'started_unix': start}
    try:
        run = run_post_training(config, case_dir=args.case_dir, resume=args.resume, max_steps=steps)
        receipt.update(status='completed', run_dir=str(run))
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt['wall_seconds'] = time.time() - start
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
