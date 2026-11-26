#!/usr/bin/env python3
"""Run the fixed matched-100 review package; no scoring or selection is done."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

CODE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CODE / 'src'), str(CODE)]
from phycoflow_reconstruction.evaluation.adaptive_campaign import atomic_json, digest

ARMS = ('FFM_Round7_Composite_Scale_Control', 'FFM_Round7_Joint_Anchor_Scale')
SPLITS = ('train', 'validation')
SUPPORTS = ('native', 'selected')


def ref(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': digest(path)}


def commands(code, python, campaign, runtime, attempt, arm, device, training_seconds, bank_seconds):
    """Build the fixed command plan; checkpoint/config paths resolve after training."""
    code, campaign, runtime, attempt = map(Path, (code, campaign, runtime, attempt))
    lease = [python, str(code/'scripts/posttraining/lease_car_round7.py'),
             '--campaign', str(campaign), '--runtime', str(runtime)]
    result = attempt / arm / 'training_result.json'
    train = lease + ['--phase', 'learning', '--label', arm+'_Matched100',
        '--max-seconds', str(training_seconds), '--', python,
        str(code/'scripts/posttraining/run_car_round7.py'), '--campaign', str(campaign),
        'train', '--arm', arm, '--stop-epoch', '100', '--device', device, '--result', str(result)]
    banks = lease + ['--phase', 'assessment', '--label', arm+'_Matched100_Banks',
        '--max-seconds', str(bank_seconds), '--', python, str(code/'scripts/posttraining/run_car_round7_review.py'),
        '_generate-four', '--campaign', str(campaign), '--attempt', str(attempt), '--arm', arm, '--device', device]
    return {'training': train, 'bank_generation': banks, 'training_result': str(result)}


def run_process(command, log, *, env):
    start = time.time()
    with Path(log).open('x') as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
    result = {'command': command, 'start_unix': start, 'end_unix': time.time(),
              'exit_code': completed.returncode, 'log': str(log)}
    return result


def generate_four(args):
    if not os.environ.get('CAR_ROUND7_LEASE_TOKEN'):
        raise ValueError('bank generation requires an enclosing assessment lease')
    root = args.attempt / args.arm
    progress = root / 'bank_progress.json'
    plan = json.loads((root/'bank_commands.json').read_text())
    if plan['arm'] != args.arm or len(plan['commands']) != 4:
        raise ValueError('four declared matched banks required')
    receipt = {'schema': 'car_round7_review_banks_v1', 'arm': args.arm, 'status': 'running',
               'lease_token': os.environ['CAR_ROUND7_LEASE_TOKEN'], 'steps': []}
    atomic_json(progress, receipt)
    try:
        for item in plan['commands']:
            log = root / (item['split']+'_'+item['support']+'_generate.log')
            step = run_process(item['command'], log, env=os.environ.copy())
            step.update(split=item['split'], support=item['support'])
            receipt['steps'].append(step)
            atomic_json(progress, receipt)
            if step['exit_code']:
                raise RuntimeError(f"{item['split']} {item['support']} generation failed; see {log}")
            path = Path(item['output'])/'receipt.json'
            bank = json.loads(path.read_text())
            if (bank.get('status') != 'completed' or bank.get('split') != item['split']
                    or bank.get('support') != item['support']
                    or bank.get('checkpoint_sha256') != plan['checkpoint']['sha256']):
                raise ValueError('completed bank identity differs from declared checkpoint/split/support')
            step['bank_receipt'] = ref(path)
            atomic_json(progress, receipt)
        receipt['status'] = 'completed'
    except BaseException as exc:
        receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt['end_unix'] = time.time()
        atomic_json(progress, receipt)


def review(args):
    if os.environ.get('CAR_ROUND7_LEASE_TOKEN'):
        raise ValueError('review coordinator must run outside a GPU lease')
    campaign = args.campaign.resolve(strict=True)
    runtime = args.runtime.resolve(strict=True)
    freeze_path = campaign/'configs/formal_freeze.json'
    freeze = json.loads(freeze_path.read_text())
    prepared_path = campaign/'configs/prepared.json'
    if Path(freeze['code_root']).resolve(strict=True) != CODE.resolve() or freeze['prepared_sha256'] != digest(prepared_path):
        raise ValueError('run the frozen code copy with its hash-bound prepared manifest')
    prepared = json.loads(prepared_path.read_text())
    repo = Path(os.environ.get('CAR_REPO_ROOT', '')).resolve(strict=True)
    if not (repo/'cases/turbulent_combustion').is_dir():
        raise ValueError('CAR_REPO_ROOT must name the real checkout containing the case')
    if not 60 <= args.training_lease_seconds <= 144000 or not 60 <= args.bank_lease_seconds <= 144000:
        raise ValueError('lease durations must be between 60 seconds and 40 hours')
    attempt = campaign/'assessment'/('Matched100_Review_'+time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())+'_'+uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CAR_REPO_ROOT=str(repo))
    py = sys.executable
    plans = {arm: commands(CODE, py, campaign, runtime, attempt, arm, args.device,
                           args.training_lease_seconds, args.bank_lease_seconds) for arm in ARMS}
    result = {'schema': 'car_round7_matched100_review_v1', 'status': 'running',
        'campaign': str(campaign), 'attempt': str(attempt), 'formal_freeze': ref(freeze_path),
        'prepared': ref(prepared_path), 'runtime': ref(runtime), 'device': args.device,
        'code_root': str(CODE), 'repo_root': str(repo), 'commands': plans,
        'start_unix': time.time(), 'arms': {}, 'test_locked': True}
    progress = attempt/'review_progress.json'
    atomic_json(progress, result)
    print(str(progress), flush=True)
    try:
        for arm in ARMS:
            root = attempt/arm
            root.mkdir()
            arm_result = {'status': 'training', 'steps': []}
            result['arms'][arm] = arm_result
            atomic_json(progress, result)
            step = run_process(plans[arm]['training'], root/'training.log', env=env)
            arm_result['steps'].append(step)
            atomic_json(progress, result)
            if step['exit_code']:
                raise RuntimeError(f'{arm} training failed; see {root / "training.log"}')
            receipt_path = Path(plans[arm]['training_result'])
            training = json.loads(receipt_path.read_text())
            if training.get('target_reached') is not True or training.get('global_step') != 1200:
                raise RuntimeError(f'{arm} stopped before epoch 100; inspect {receipt_path}')
            run = Path(training['run_dir']).resolve(strict=True)
            checkpoint = run/'checkpoints/epoch_100.pt'
            checkpoint_ref = ref(checkpoint)
            config = Path(training['saved_config']['path']).resolve(strict=True)
            if digest(config) != training['saved_config']['sha256']:
                raise ValueError('saved run configuration changed')
            arm_result.update(status='preparing_banks', training_result=ref(receipt_path),
                              checkpoint=checkpoint_ref, saved_config=ref(config))
            atomic_json(progress, result)
            reference = Path(prepared['reference']['path']).resolve(strict=True)
            if digest(reference) != prepared['reference']['sha256']:
                raise ValueError('prepared reference changed')
            specs = {}
            for split in SPLITS:
                path = root/(split+'_spec.json')
                command = [py, str(CODE/'scripts/posttraining/prepare_car_round7_assessment.py'),
                    '--campaign', str(campaign), '--config', str(config), '--reference', str(reference),
                    '--split', split, '--checkpoint', str(checkpoint), '--reporting-epoch', '100', '--output', str(path)]
                step = run_process(command, root/(split+'_spec.log'), env=env)
                arm_result['steps'].append(step)
                atomic_json(progress, result)
                if step['exit_code']:
                    raise RuntimeError(f'{arm} {split} assessment preparation failed; see {step["log"]}')
                specs[split] = ref(path)
            bank_plan = {'schema': 'car_round7_four_banks_v1', 'arm': arm, 'checkpoint': checkpoint_ref,
                         'specs': specs, 'commands': []}
            for split in SPLITS:
                for support in SUPPORTS:
                    output = root/(split+'_'+support+'_bank')
                    cmd = [py, str(CODE/'scripts/posttraining/assess_car_round7.py'), 'generate',
                        '--spec', specs[split]['path'], '--output', str(output), '--support', support, '--device', args.device]
                    bank_plan['commands'].append({'split': split, 'support': support,
                                                  'output': str(output), 'command': cmd})
            bank_commands = root/'bank_commands.json'
            with bank_commands.open('x') as stream:
                json.dump(bank_plan, stream, indent=2, allow_nan=False); stream.write('\n')
            arm_result.update(status='generating_banks', bank_commands=ref(bank_commands))
            atomic_json(progress, result)
            step = run_process(plans[arm]['bank_generation'], root/'banks_lease.log', env=env)
            arm_result['steps'].append(step)
            atomic_json(progress, result)
            bank_progress = root/'bank_progress.json'
            if step['exit_code']:
                raise RuntimeError(f'{arm} bank generation failed; see {root / "banks_lease.log"} and {bank_progress}')
            banks = json.loads(bank_progress.read_text())
            if banks.get('status') != 'completed' or len(banks['steps']) != 4:
                raise ValueError('incomplete four-bank receipt')
            arm_result.update(status='completed', bank_progress=ref(bank_progress),
                              banks=[item['bank_receipt'] for item in banks['steps']])
            ready = root/'ready.json'
            with ready.open('x') as stream:
                json.dump({'schema':'car_round7_review_arm_ready_v1', 'arm':arm,
                           'checkpoint':checkpoint_ref, 'specs':specs, 'banks':arm_result['banks']}, stream, indent=2)
                stream.write('\n')
            arm_result['ready'] = ref(ready)
            atomic_json(progress, result)
            print(str(ready), flush=True)
        result['status'] = 'completed'
    except BaseException as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result['end_unix'] = time.time()
        atomic_json(progress, result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--campaign', type=Path, required=True)
    r.add_argument('--runtime', type=Path, required=True)
    r.add_argument('--training-lease-seconds', type=float, required=True)
    r.add_argument('--bank-lease-seconds', type=float, required=True)
    r.add_argument('--device', default='cuda:0')
    g = sub.add_parser('_generate-four')
    g.add_argument('--campaign', type=Path, required=True)
    g.add_argument('--attempt', type=Path, required=True)
    g.add_argument('--arm', choices=ARMS, required=True)
    g.add_argument('--device', required=True)
    args = p.parse_args()
    (review if args.command == 'run' else generate_four)(args)

if __name__ == '__main__':
    main()
