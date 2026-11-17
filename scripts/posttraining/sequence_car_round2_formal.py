#!/usr/bin/env python3
"""Run one source's frozen full and residual formal arms serially."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / 'src'))
from phycoflow_reconstruction.evaluation.adaptive_campaign import digest, immutable_json


def sequence(args):
    campaign = args.campaign.resolve(strict=True)
    output = args.output.resolve()
    work = output.parent / f'{output.stem}_files'
    if campaign not in output.parents or output.exists() or work.exists():
        raise ValueError('formal group output and work directory must be new and campaign-owned')
    selector = getattr(args, 'arm', 'both')
    selected_indices = {'both': (0, 1), 'full_model': (0,), 'residual_repair': (1,)}[selector]
    bounds = [[args.full_max_seconds, args.residual_max_seconds][i] for i in selected_indices]
    epochs = [[args.full_epoch_seconds, args.residual_epoch_seconds][i] for i in selected_indices]
    values = [*bounds, *epochs, args.save_margin_seconds, args.initial_stage_bound_seconds]
    if any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('positive finite measured timing and bounds required')
    if any(args.initial_stage_bound_seconds > bound - args.save_margin_seconds or
           1000 * epoch + args.save_margin_seconds > bound for bound, epoch in zip(bounds, epochs, strict=True)):
        raise ValueError('each child bound must cover forecast epoch time, startup and safe-save margin')
    spec = importlib.util.spec_from_file_location('round2_formal_driver', CODE_ROOT / 'scripts/posttraining/run_coherence_adaptive_repair_round2.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    capsule = driver.check_code_capsule(campaign)
    family_admission = getattr(args, 'family_admission', None)
    freeze = Path(family_admission).resolve(strict=True) if family_admission else campaign / 'configs/formal_freeze.json'
    if campaign not in freeze.parents:
        raise ValueError('admission must be campaign-owned')
    freeze_receipt = json.loads(freeze.read_text())
    if family_admission:
        manifest = driver.check_freeze(freeze_receipt, campaign, args.family)
    else:
        if freeze_receipt.get('schema') != 'car_round2_freeze_receipt_v1':
            raise ValueError('formal freeze receipt required')
        manifest = driver.check_freeze(freeze_receipt['manifest'], campaign)
    code_inventory = json.loads(driver.check_binding(manifest['artifacts']['code_inventory'], campaign).read_text())
    if code_inventory.get('schema') != 'car_execution_code_v1' or code_inventory.get('sha256') != capsule:
        raise ValueError('running capsule must equal the code inventory in the formal freeze')
    forecast_path = driver.check_binding(manifest['artifacts']['forecast'], campaign)
    forecast = json.loads(forecast_path.read_text())
    if forecast.get('schema') != ('car_round2_family_forecast_v1' if family_admission else 'car_round2_forecast_v2') or forecast.get('fits') is not True:
        raise ValueError('passing measured formal forecast required')
    update_seconds = forecast.get('per_arm_update_seconds')
    callback_seconds = forecast.get('per_arm_callback_seconds_per_epoch')
    outside_seconds = forecast.get('per_arm_outside_epoch_seconds')
    cold_seconds = forecast.get('per_arm_cold_train_excess_seconds')
    expected_count = 2 if family_admission else len(driver.ARMS)
    if (not isinstance(update_seconds, list) or len(update_seconds) != expected_count
            or not isinstance(callback_seconds, list) or len(callback_seconds) != expected_count
            or not isinstance(outside_seconds, list) or len(outside_seconds) != expected_count
            or not isinstance(cold_seconds, list) or len(cold_seconds) != expected_count):
        raise ValueError('forecast must identify all four per-arm update measurements')
    prefix = 'FFM' if args.family == 'ffm' else 'Senseiver'
    pair_arms = [f'{prefix}_Selected_Bank_Uniform_FullModel', f'{prefix}_Selected_Bank_Residual_Repair']
    arms = [pair_arms[i] for i in selected_indices]
    overheads = []
    for arm, epoch, bound in zip(arms, epochs, bounds, strict=True):
        index = pair_arms.index(arm) if family_admission else driver.ARMS.index(arm)
        measured = update_seconds[index]
        callback = callback_seconds[index]
        overhead = outside_seconds[index]
        cold = cold_seconds[index]
        if (not isinstance(measured, (int, float)) or not math.isfinite(measured) or measured <= 0
                or not isinstance(callback, (int, float)) or not math.isfinite(callback) or callback < 0
                or not isinstance(overhead, (int, float)) or not math.isfinite(overhead) or overhead < 0
                or not isinstance(cold, (int, float)) or not math.isfinite(cold) or cold < 0
                or epoch < 12 * measured + callback):
            raise ValueError('epoch estimates must cover the bound forecast update times and reporting')
        overheads.append(overhead + cold)
        if 1000 * epoch + overhead + cold + args.save_margin_seconds > bound:
            raise ValueError('child bound must include measured startup, closure and cold-update excess')
    live_budget = driver.check_family_live_budget(manifest, campaign) if family_admission else None
    claim_dir = campaign / 'operations/formal_arm_claims'
    claim_dir.mkdir(parents=True, exist_ok=True)
    claimed = []
    claim_records = []
    try:
        for arm in arms:
            claim = claim_dir / f'{arm}.json'
            with claim.open('x') as stream:
                json.dump({'arm': arm, 'group_output': str(output), 'admission_sha256': digest(freeze)}, stream)
                stream.write('\n')
            claimed.append(claim)
            claim_records.append({'path': str(claim), 'sha256': digest(claim), 'contents': json.loads(claim.read_text())})
    except BaseException:
        for claim in claimed:
            claim.unlink()
        raise
    try:
        work.mkdir(parents=True)
        start = time.time()
        # The second arm's absolute estimates assume the first starts now and
        # takes the supplied elapsed epoch estimate. They are scheduling hints.
        schedule = []
        offset = 0.0
        for arm, epoch, overhead in zip(arms, epochs, overheads, strict=True):
            for milestone in (500, 1000):
                schedule.append({'arm': arm, 'completed_epochs': milestone,
                                 'anticipated_unix': start + offset + overhead + milestone * epoch})
            offset += overhead + 1000 * epoch
        immutable_json(work / 'group_launch.json', {'schema': 'car_round2_formal_group_launch_v1',
                       'family': args.family, 'device_request': args.device, 'code_capsule_sha256': capsule,
                       'formal_freeze_sha256': None if family_admission else digest(freeze),
                       'family_admission_sha256': digest(freeze) if family_admission else None,
                       'family_live_budget': live_budget, 'arm_claims': claim_records,
                       'forecast_sha256': digest(forecast_path),
                       'per_arm_max_seconds': dict(zip(arms, bounds, strict=True)),
                       'per_arm_epoch_seconds_including_reporting': dict(zip(arms, epochs, strict=True)),
                       'per_arm_measured_one_time_overhead_seconds': dict(zip(arms, overheads, strict=True)),
                       'initial_stage_bound_seconds': args.initial_stage_bound_seconds,
                       'save_margin_seconds': args.save_margin_seconds, 'started_unix': start,
                       'luna_sparse_schedule': schedule, 'anticipated_group_completion_unix': start + offset,
                       'schedule_scope': 'estimates for read-only Luna checks, not progress polling; revise after observations',
                       'wall_accounting': 'group wall encloses child leases and is not an additional GPU cost'})
    except BaseException:
        for claim in claimed:
            claim.unlink()
        raise
    children, failure, status, run_dirs = [], None, 'failed', set()
    try:
        for arm, bound in zip(arms, bounds, strict=True):
            config = driver.check_binding(manifest['arms'][arm]['config'], campaign)
            launch = work / f'{arm}_launch.json'
            stdout, stderr = work / f'{arm}.stdout', work / f'{arm}.stderr'
            command = [sys.executable, str(CODE_ROOT / 'scripts/posttraining/run_coherence_adaptive_repair_round2.py'),
                       '--campaign', str(campaign), 'run', '--arm', arm, '--config', str(config),
                       '--device', args.device, '--max-seconds', str(bound),
                       '--save-margin-seconds', str(args.save_margin_seconds),
                       '--initial-stage-bound-seconds', str(args.initial_stage_bound_seconds),
                       '--launch-receipt-output', str(launch)]
            if family_admission:
                command.extend(['--family-admission', str(freeze)])
            began, pid, launch_error = time.time(), None, None
            with stdout.open('x') as out, stderr.open('x') as err:
                try:
                    process = subprocess.Popen(command, stdout=out, stderr=err)
                    pid = process.pid
                    code = process.wait()
                except OSError as error:
                    code, launch_error = None, str(error)
            done_path, done = None, None
            if launch.exists():
                launch_doc = json.loads(launch.read_text())
                done_path = campaign / 'operations/completions' / f"{launch_doc['token']}.json"
                if done_path.exists():
                    done = json.loads(done_path.read_text())
            ended = time.time()
            record = {'schema': 'car_round2_formal_child_v1', 'arm': arm, 'command': command,
                      'pid': pid, 'returncode': code, 'launch_error': launch_error,
                      'launch_receipt': str(launch) if launch.exists() else None,
                      'launch_sha256': digest(launch) if launch.exists() else None,
                      'completion_receipt': str(done_path) if done is not None else None,
                      'completion_sha256': digest(done_path) if done is not None else None,
                      'stdout': str(stdout), 'stderr': str(stderr), 'start_unix': began,
                      'end_unix': ended, 'wall_seconds': ended - began,
                      'wall_accounting': 'encloses the child GPU lease; do not add both'}
            record_path = work / f'{arm}_child.json'
            immutable_json(record_path, record)
            children.append({'path': str(record_path), 'sha256': digest(record_path)})
            run_dir = Path(done.get('run_dir', '')).resolve() if done is not None else None
            expected_parent = (campaign / 'runs/formal' / arm).resolve()
            if (code != 0 or done is None or done.get('token') != launch_doc.get('token')
                    or done.get('status') != 'completed' or done.get('global_step') != 12000
                    or expected_parent not in run_dir.parents or run_dir in run_dirs):
                raise RuntimeError(f'{arm} did not complete 1000 epochs: exit={code}, completion={done}; see {stderr}')
            run_dirs.add(run_dir)
        status = 'completed'
    except Exception as error:  # noqa: BLE001 - retain group failure details and child receipts.
        failure = {'type': type(error).__name__, 'message': str(error)}
    finally:
        ended = time.time()
        immutable_json(output, {'schema': 'car_round2_formal_group_v1', 'status': status, 'family': args.family,
                       'launch_receipt': str(work / 'group_launch.json'), 'launch_sha256': digest(work / 'group_launch.json'),
                       'children': children, 'failure': failure, 'end_unix': ended, 'wall_seconds': ended - start,
                       'wall_accounting': 'encloses child leases; do not add as independent GPU cost'})
        for claim, record in zip(claimed, claim_records, strict=True):
            if claim.is_file() and digest(claim) == record['sha256']:
                claim.unlink()
    if failure:
        raise RuntimeError(failure['message'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--family', choices=['ffm', 'senseiver'], required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--family-admission', type=Path)
    parser.add_argument('--arm', choices=('both', 'full_model', 'residual_repair'), default='both')
    parser.add_argument('--full-max-seconds', type=float, required=True)
    parser.add_argument('--residual-max-seconds', type=float, required=True)
    parser.add_argument('--full-epoch-seconds', type=float, required=True, help='measured forecast per epoch including reporting')
    parser.add_argument('--residual-epoch-seconds', type=float, required=True, help='measured forecast per epoch including reporting')
    parser.add_argument('--save-margin-seconds', type=float, required=True)
    parser.add_argument('--initial-stage-bound-seconds', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    sequence(parser.parse_args())


if __name__ == '__main__':
    main()
