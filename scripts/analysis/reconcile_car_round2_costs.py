#!/usr/bin/env python3
"""Index completed checkpoint clocks or reconcile explicitly attributed campaign costs.

Allocation input schema car_round2_cost_allocation_v1 has gpu_jobs, cpu_jobs,
checkpoint_runs, checkpoint_extras and allocator_disposition. Each job has an
id, scope, arm allocations and hash-bound receipt; GPU jobs also name a closed
ledger token. CPU jobs must declare independent work or its containing parent.
Checkpoint extras name the measured assessment jobs for each arm and epoch.
The result remains incomplete until every scope and all six clocks are present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get('CAR_REPO_ROOT', Path(__file__).resolve().parents[2])).resolve()
CODE = Path(os.environ.get('CAR_CODE_ROOT', ROOT)).resolve()
sys.path.insert(0, str(CODE / 'src'))

from phycoflow_reconstruction.evaluation.round2_costs import (
    index_checkpoint_clock,
    reconcile,
)


def _read(path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    index = sub.add_parser('index-checkpoints')
    index.add_argument('--run', type=Path, required=True)
    index.add_argument('--arm', required=True)
    index.add_argument('--output', type=Path, required=True)
    costs = sub.add_parser('reconcile')
    costs.add_argument('--family', choices=('ffm', 'senseiver'), required=True)
    costs.add_argument('--manifest', type=Path, required=True)
    costs.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.time()
    campaign = args.campaign.resolve()
    root = Path('/data2/wanglz/PhyCoFlow-MultiFieldReconstruction').resolve()
    output = args.output.resolve()
    if root not in campaign.parents or campaign not in output.parents:
        parser.error('campaign must be under DATA2 and output must be campaign-owned')
    if args.command == 'index-checkpoints':
        result = index_checkpoint_clock(campaign=campaign, run=args.run, arm=args.arm)
    else:
        manifest_path = args.manifest.resolve()
        if campaign not in manifest_path.parents:
            parser.error('allocation manifest must be campaign-owned')
        ledger_path = campaign / 'operations/gpu_ledger.json'
        manifest, manifest_hash = _read(manifest_path)
        ledger, ledger_hash = _read(ledger_path)
        result = reconcile(campaign=campaign, ledger=ledger, manifest=manifest, family=args.family)
        if hashlib.sha256(ledger_path.read_bytes()).hexdigest() != ledger_hash:
            result['complete'] = False
            result['within_budget'] = False
            result['missing'].append('GPU ledger changed during reconciliation')
            for arm in result['arms'].values():
                arm['serial_cost_seconds'] = None
                arm['checkpoint_cost_seconds'] = {}
        result['source_receipts'].extend([
            {'path': str(manifest_path), 'sha256': manifest_hash},
            {'path': str(ledger_path), 'sha256': ledger_hash},
        ])
    result['reconciliation_timer'] = {'started_unix': start, 'finished_unix': time.time()}
    _write(output, result)
    print(json.dumps({'output': str(output), 'complete': result.get('complete', True),
                      'missing': result.get('missing', [])}))


if __name__ == '__main__':
    main()
