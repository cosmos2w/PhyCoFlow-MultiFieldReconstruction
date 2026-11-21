#!/usr/bin/env python3
"""One complete current-route epoch on the Round 4 runtime device."""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import uuid
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('CAR_REPO_ROOT', CODE)).resolve()
sys.path[:0] = [str(CODE / 'src'), str(CODE)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--max-seconds', type=int, default=900)
    args = p.parse_args()
    if not 30 < args.max_seconds <= 7200:
        p.error('qualification must fit the two-hour allowance')
    start, cpu = time.time(), time.process_time()
    from scripts.posttraining.prepare_car_round4 import device_identity, resolve_config, sha
    device = device_identity()
    import torch
    from phycoflow_reconstruction.evaluation.adaptive_campaign import atomic_json, locked_ledger, spent
    from phycoflow_reconstruction.training.post_training import run_post_training

    campaign = args.campaign.resolve(strict=True)
    cfg, binding = resolve_config(campaign.parent / 'round_3')
    token = uuid.uuid4().hex
    cfg['output']['experiment_name'] = f'coherence_adaptive_repair/round_4/runs/qualification/FFM_Current_SourceScaledRisk_Timing/{token}'
    cfg['runtime'].update(device='cuda:0', plot_every_steps=10)
    ledger = campaign / 'operations/gpu_ledger.json'
    with locked_ledger(ledger) as state:
        reserved = sum(max(0., row.get('max_seconds', 0.) - (start-row['start_unix'])) for row in state['active'].values())
        if spent(state) + reserved + args.max_seconds > 86400:
            raise ValueError('qualification exceeds the campaign ceiling')
        if any(row['gpu_uuid'] == device['uuid'] for row in state['active'].values()):
            raise ValueError('selected GPU has an active campaign lease')
        state['active'][token] = dict(gpu_uuid=device['uuid'], phase='current_route_timing', start_unix=start, pid=os.getpid(), max_seconds=args.max_seconds)
    receipt = dict(schema='car_round4_current_route_timing_v1', status='failed', token=token,
                   start_unix=start, device=device, inherited_config=binding, script_sha256=sha(__file__),
                   scope='one complete existing SourceScaledRisk reporting epoch, with the unchanged historical conditioning protocol; timing qualification only')
    output = campaign / 'operations/current_route_timing' / f'{token}.json'
    def timeout(_signal, _frame):
        raise TimeoutError('current-route qualification reached its declared limit')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_seconds - 10)
    try:
        torch.set_num_threads(4)
        run = run_post_training(cfg, case_dir=ROOT / 'cases/turbulent_combustion', max_steps=12)
        receipt.update(status='completed', run_dir=str(run))
    except BaseException as exc:
        receipt.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.alarm(0)
        end = time.time()
        receipt.update(end_unix=end, wall_seconds=end-start, process_cpu_seconds=time.process_time()-cpu)
        atomic_json(output, receipt)
        with locked_ledger(ledger) as state:
            row = state['active'].pop(token)
            state['closed'].append({**row, 'token':token, 'end_unix':end, 'seconds':end-start})
        print(output, flush=True)


if __name__ == '__main__':
    main()
