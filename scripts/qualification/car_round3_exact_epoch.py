#!/usr/bin/env python3
"""Run one actual trainer epoch for the Round 3 exact-compute comparison."""
import argparse
import hashlib
import json
import subprocess
import time
import types
from pathlib import Path

import torch
import yaml

from phycoflow_reconstruction.training.post_training import run_post_training
import phycoflow_reconstruction.training.scalable_runtime as scalable_runtime
import phycoflow_reconstruction.coherence.query_runtime as query_runtime

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--case-dir', type=Path, required=True)
parser.add_argument('--receipt', type=Path, required=True)
parser.add_argument('--implementation', choices=('historical', 'reused'), required=True)
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--baseline-ref', default='e1366b9')
args = parser.parse_args()
config = yaml.safe_load(args.config.read_text())
config['runtime']['device'] = args.device
config['output']['experiment_name'] = ('coherence_adaptive_repair/round_3/runs/qualification/'
                                      f'FFM_ExactCompute_{args.implementation}')
steps = int(config['optimization']['steps_per_epoch'])
old_source = subprocess.check_output(['git', 'show', f'{args.baseline_ref}:src/phycoflow_reconstruction/coherence/state_compute.py'], text=True)
old = types.ModuleType('phycoflow_reconstruction.coherence._historical_state')
old.__package__ = 'phycoflow_reconstruction.coherence'
exec(compile(old_source, 'historical_state.py', 'exec'), old.__dict__)
original_score = query_runtime.score
if args.implementation == 'historical':
    scalable_runtime.group_projection_scores = old.group_projection_scores
    def historical_score(*positional, **keyword):
        positional[5].pop('_reuse_scale_for_expanded', None)
        return original_score(*positional, **keyword)
    query_runtime.score = historical_score
start = time.time()
receipt = {'status': 'failed', 'implementation': args.implementation,
           'baseline_ref': args.baseline_ref,
           'baseline_state_sha256': hashlib.sha256(old_source.encode()).hexdigest(),
           'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
           'device': args.device, 'device_name': torch.cuda.get_device_name(torch.device(args.device)),
           'start_unix': start, 'steps_per_epoch': steps,
           'scope': 'actual trainer one-epoch qualification, including loader, logging, curves and checkpoint callbacks'}
try:
    run = run_post_training(config, case_dir=args.case_dir, max_steps=steps)
    receipt.update(status='completed', run_dir=str(run))
except BaseException as exc:
    receipt.update(error_type=type(exc).__name__, error=str(exc))
    raise
finally:
    receipt['end_unix'] = time.time()
    receipt['wall_seconds'] = receipt['end_unix'] - start
    args.receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
