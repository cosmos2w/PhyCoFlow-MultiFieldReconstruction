#!/usr/bin/env python3
"""Bound GPU leases and freeze the measured Round 5 execution contract.

Use lease to charge every GPU job, including failed attempts and assessments. The
child must write its own scientific result; a lease receipt is not qualification.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
import platform
import resource
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from phycoflow_reconstruction.evaluation.adaptive_campaign import atomic_json, digest, immutable_json, locked_ledger, spent
from scripts.posttraining.prepare_car_round4 import device_identity

CAP = 45 * 3600
INITIAL = 2 * 3600
CORRECTION = 2 * 3600
ARMS = ('Native_GT_Continuation', 'Native_GT_SSS_Safeguarded_Continuation')


def reserved(state, now):
    total = 0.
    for row in state['active'].values():
        if 'max_seconds' not in row:
            raise ValueError('active lease has no bounded reservation')
        total += max(0., float(row['max_seconds']) - (now - float(row['start_unix'])))
    return total


def acquire(state, *, token, gpu_uuid, phase, now, duration):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('positive finite lease required')
    if spent(state, now) + reserved(state, now) + duration > CAP:
        raise ValueError('lease exceeds additive 45 GPU-hour ceiling')
    if any(row['gpu_uuid'] == gpu_uuid for row in state['active'].values()):
        raise ValueError('selected device is already leased')
    if phase in ('startup_initial', 'startup_correction'):
        limit = INITIAL if phase == 'startup_initial' else CORRECTION
        rows = [x for x in state['closed'] if x['phase'] == phase]
        active = [x for x in state['active'].values() if x['phase'] == phase]
        used = sum(float(x['seconds']) for x in rows)
        used += sum(float(x['max_seconds']) for x in active)
        if used + duration > limit:
            raise ValueError(f'{phase} exceeds its two GPU-hour allocation')
    state['active'][token] = dict(gpu_uuid=gpu_uuid, phase=phase, start_unix=now,
                                  pid=os.getpid(), max_seconds=duration)


def lease(args, campaign):
    if not args.command:
        raise ValueError('a GPU command is required after --')
    if not os.environ.get('KEOPS_CACHE_FOLDER'):
        raise ValueError('KEOPS_CACHE_FOLDER must be selected at runtime')
    # Installed KeOps probes CUDA headers before creating its device-specific
    # compiler directory. Create that runtime path before any model import.
    cache = Path(os.environ['KEOPS_CACHE_FOLDER'])
    build = '_'.join(platform.uname()[:3]) + '_p' + sys.version.split(' ')[0]
    selected = os.environ.get('CUDA_VISIBLE_DEVICES')
    if selected:
        build += '_CUDA_VISIBLE_DEVICES_' + selected.replace(',', '_')
    (cache / build).mkdir(parents=True, exist_ok=True)
    if args.phase == 'startup_correction':
        if args.correction_evidence is None:
            raise ValueError('correction requires a recorded demonstrated failure')
        evidence = json.loads(args.correction_evidence.read_text())
        if evidence.get('demonstrated_failure') is not True or not evidence.get('correction'):
            raise ValueError('correction evidence must identify the failure and repair')
    if args.phase in ('formal_training', 'assessment'):
        frozen = json.loads((campaign / 'configs/formal_freeze.json').read_text())
        if frozen.get('schema') != 'car_round5_freeze_v1':
            raise ValueError('formal execution requires the Round 5 scientific freeze')
        if not 100 <= frozen['final_epoch'] <= 500:
            raise ValueError('invalid frozen common endpoint')
    device = device_identity()
    token, started = uuid.uuid4().hex, time.time()
    ledger = campaign / 'operations/gpu_ledger.json'
    with locked_ledger(ledger) as state:
        if args.phase == 'formal_training':
            reserve = float(frozen['forecast']['assessment_gpu_seconds']) + float(frozen['forecast']['reserve_gpu_seconds'])
            if spent(state, started) + reserved(state, started) + args.max_seconds + reserve > CAP:
                raise ValueError('formal lease would consume the frozen assessment and contingency reserve')
        acquire(state, token=token, gpu_uuid=device['uuid'], phase=args.phase,
                now=started, duration=args.max_seconds)
    receipt = dict(schema='car_round5_lease_v1', token=token, phase=args.phase,
        start_unix=started, max_seconds=args.max_seconds, device=device,
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], command=args.command,
        correction_evidence=str(args.correction_evidence) if args.correction_evidence else None,
        status='failed')
    atomic_json(campaign / 'operations/launches' / f'{token}.json', receipt)
    child = None
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        child_env = dict(os.environ, CAR_ROUND5_LEASE_TOKEN=token)
        child = subprocess.Popen(args.command, start_new_session=True, env=child_env)
        try:
            code = child.wait(timeout=max(1., args.max_seconds - 12.))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise TimeoutError('GPU lease expired')
        receipt.update(status='completed' if code == 0 else 'failed', exit_code=code)
        if code:
            raise subprocess.CalledProcessError(code, args.command)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        receipt.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        end = time.time()
        usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        receipt.update(child_user_cpu_seconds=usage_after.ru_utime - usage_before.ru_utime,
                       child_system_cpu_seconds=usage_after.ru_stime - usage_before.ru_stime,
                       cpu_accounting='child CPU inside occupied GPU lease; do not add to GPU wall time')
        with locked_ledger(ledger) as state:
            row = state['active'].pop(token)
            state['closed'].append({**row, 'token':token, 'end_unix':end, 'seconds':end-started})
        atomic_json(campaign / 'operations/completions' / f'{token}.json',
                    {**receipt, 'end_unix':end, 'seconds':end-started})


def forecast(*, already, native_epoch_seconds, direct_epoch_seconds=None,
             assessment_seconds=3600., reserve_seconds=10800.):
    vals = [already, native_epoch_seconds, assessment_seconds, reserve_seconds]
    if direct_epoch_seconds is not None: vals.append(direct_epoch_seconds)
    if not all(math.isfinite(x) and x >= 0 for x in vals) or native_epoch_seconds <= 0 or (direct_epoch_seconds is not None and direct_epoch_seconds <= 0):
        raise ValueError('measured positive epoch durations and nonnegative costs required')
    total_epoch = native_epoch_seconds + (direct_epoch_seconds if direct_epoch_seconds is not None else 0.)
    remaining = CAP - already - assessment_seconds - reserve_seconds
    endpoint = min(500, math.floor(remaining / total_epoch))
    if endpoint < 100:
        raise ValueError('100 matched epochs do not fit the ceiling and reserves')
    return {'final_epoch': endpoint, 'spent_gpu_seconds': already,
            'native_epoch_seconds': native_epoch_seconds, 'direct_epoch_seconds': direct_epoch_seconds,
            'assessment_gpu_seconds': assessment_seconds, 'reserve_gpu_seconds': reserve_seconds,
            'forecast_gpu_seconds': already + endpoint * total_epoch + assessment_seconds + reserve_seconds}


def qualification(startup, units_hash):
    if startup.get('schema') != 'car_round5_startup_qualification_v1' or startup.get('units_sha256') != units_hash:
        raise ValueError('unit-bound startup qualification required')
    if startup.get('qualified') is True:
        return list(ARMS)
    if (startup.get('qualified') is False and startup.get('native_qualified') is True
            and startup.get('candidate_rejection_terminal') is True
            and isinstance(startup.get('candidate_rejection_reason'), str)
            and startup['candidate_rejection_reason'].strip()):
        return [ARMS[0]]
    raise ValueError('candidate must qualify or have a documented terminal rejection and qualified native control')


def cpu_forecast(startup):
    info = startup.get('cpu_assessment_forecast')
    if not isinstance(info, dict):
        raise ValueError('whole-assessment CPU forecast is required')
    for key in ('worker_hours', 'wall_seconds', 'tail_seconds'):
        value = info.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'CPU forecast requires finite nonnegative {key}')
    if not info.get('basis') or not isinstance(info.get('evidence'), dict):
        raise ValueError('CPU forecast requires a stated basis and bound timing evidence')
    checked(info['evidence'])
    return info


def measured_forecast(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def reference(path):
    path = path.resolve(strict=True)
    return {'path': str(path), 'sha256': digest(path)}


def checked(ref):
    path = Path(ref['path']).resolve(strict=True)
    if digest(path) != ref['sha256']:
        raise ValueError(f'frozen artifact changed: {path}')
    return path


def prepare(args, campaign):
    import yaml
    from scripts.posttraining.prepare_car_round4 import resolve_config
    units_doc = json.loads(args.units.read_text())
    startup = json.loads(args.startup.read_text())
    disposition = json.loads(args.disposition.read_text())
    if (units_doc.get('schema') != 'car_round5_source_units_v1' or units_doc.get('status') != 'completed'
            or units_doc.get('sample_count') != 800):
        raise ValueError('SOURCE TRAIN800 unit fit required')
    if any(units_doc.get('floor_active', {}).values()):
        raise ValueError('active unit floor requires observable diagnosis')
    active_arms = qualification(startup, digest(args.units))
    cpu_forecast(startup)
    if (disposition.get('schema') != 'car_round5_context_estimator_v1'
            or disposition.get('qualified') is not True
            or disposition.get('query_law') not in ('shared', 'condition_groups_8')
            or disposition.get('context_cache') not in ('none', 'static_features')):
        raise ValueError('qualified implemented context and query law required')
    cfg, prior = resolve_config(campaign.parent / 'round_3')
    old_ref = cfg['optimization']['multitask']['fidelity_budget']['reference_artifact']
    ref_path = checked(old_ref)
    original = json.loads(ref_path.read_text())
    mse = [float(x) for x in original['risk_reference_panel']['source_member_mse']]
    tolerance = [float(x) for x in startup.get('field_tolerance', [])]
    if len(mse) != 5 or any(not math.isfinite(x) or x <= 0 for x in mse):
        raise ValueError('five positive SOURCE member MSE values required')
    if ARMS[1] in active_arms and (len(tolerance) != len(mse) or any(not math.isfinite(x) or x < 0 or x >= .001*.1*y for x,y in zip(tolerance,mse,strict=True))):
        raise ValueError('field tolerances must be nonnegative and negligible relative to allowances')
    unit = units_doc['units']
    if set(unit) != {'native','state','scale','shape'} or any(not math.isfinite(float(x)) or x <= 0 for x in unit.values()):
        raise ValueError('four finite positive units required')
    start, target = float(startup['lr_start']), float(startup['lr_target'])
    startup_preparation = None
    prep_ref = startup.get('evidence', {}).get('startup_preparation')
    if prep_ref is not None:
        startup_preparation = json.loads(checked(prep_ref).read_text())
        if (startup_preparation.get('schema') != 'car_round5_startup_preparation_v1'
                or startup_preparation.get('units_sha256') != digest(args.units)
                or startup_preparation.get('lr_start') != start
                or startup_preparation.get('lr_target') != target
                or startup_preparation.get('beta1') != float(startup['beta1'])
                or startup_preparation.get('context_cache') != disposition['context_cache']
                or startup_preparation.get('query_law') != disposition['query_law']):
            raise ValueError('formal settings differ from the qualified startup preparation')
    corrected = (startup_preparation is not None and startup_preparation.get('kind') == 'correction'
                 and bool(startup_preparation.get('correction_evidence_sha256')))
    if (start,target) not in ((5e-6,5e-5), (2.5e-6,2.5e-5)):
        raise ValueError('startup schedule must be one of the two declared schedules')
    if start == 2.5e-6 and not (corrected or startup.get('finite_step_overshoot_demonstrated')):
        raise ValueError('lower schedule requires demonstrated finite-step overshoot')
    beta1, beta2 = float(startup['beta1']), float(startup['beta2'])
    if beta1 not in (0., .9) or not 0 <= beta2 < 1:
        raise ValueError('invalid bounded optimizer choice')
    if beta1 == 0 and not (corrected or startup.get('ascending_proposal_fraction', 0) > .1):
        raise ValueError('beta1=0 requires more than 10% ascending proposals')
    with locked_ledger(campaign/'operations/gpu_ledger.json') as state:
        if state['active']: raise ValueError('finish startup GPU leases before preparation')
        costs = forecast(already=spent(state), native_epoch_seconds=float(startup['native_epoch_seconds']),
                         direct_epoch_seconds=(float(startup['direct_epoch_seconds']) if ARMS[1] in active_arms else None))
    endpoint = costs['final_epoch']
    paths = {}
    for arm in active_arms:
        current = copy.deepcopy(cfg)
        opt = current['optimization']
        for key in ('multitask', 'fast_lagrangian', 'lpc_ordered', 'execution_strategy'):
            opt.pop(key, None)
        opt.update(epochs=endpoint, gradient_balance='weighted_sum', model_mode='eval', lr=target,
                   lr_warmup={'enabled': True, 'start_lr':start, 'epochs':50}, adam_betas=[beta1,beta2])
        opt['rollout_execution'] = {'checkpointing':True, 'context_cache':disposition['context_cache']}
        direct = {'mode':'native' if arm == ARMS[0] else 'safeguarded', 'units':unit}
        if arm == ARMS[1]:
            direct.update(source_member_mse=mse, field_tolerance=tolerance,
                          dot_tolerance=float(startup['dot_tolerance']))
            current['coherence']['scalable_anchored_coherence']['condition_query_groups'] = (
                8 if disposition['query_law'] == 'condition_groups_8' else 1)
        opt['direct_supervision'] = direct
        if arm == ARMS[0]:
            current['coherence']['scalable_anchored_coherence'] = {'enabled': False}
            current['evaluation'].pop('batch_size', None)
        current['runtime'].update(device='cuda:0', plot_every_steps=10)
        current['evaluation'].update(split='validation', max_samples=100, external_assessment=True)
        current['output']['experiment_name'] = f'coherence_adaptive_repair/round_5/runs/formal/{arm}'
        current['checkpointing'].update(every_epochs=25, epochs=sorted(set([100, endpoint])), panel_milestones_only=True, selection_metric='native_validation_loss')
        from phycoflow_reconstruction.config.validate import validate_config
        validate_config(current)
        dest = campaign / 'configs' / f'{arm}.yaml'
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = yaml.safe_dump(current, sort_keys=False)
        if dest.exists() and dest.read_text() != content: raise ValueError(f'configuration changed: {dest}')
        if not dest.exists(): dest.write_text(content)
        paths[arm] = reference(dest)
    prepared = {'schema':'car_round5_prepared_v1', 'arms':paths,
        'source_reference':reference(ref_path), 'source_snapshot':cfg['source_run'],
        'units':reference(args.units), 'startup':reference(args.startup),
        'disposition':reference(args.disposition), 'source_config':prior,
        'test_locked':True, 'selection_use':'development VALIDATION100', 'forecast':costs, 'final_epoch':endpoint, 'active_arms':active_arms}
    immutable_json(campaign/'configs/prepared.json', prepared)


def freeze(campaign):
    from phycoflow_reconstruction.evaluation.adaptive_campaign import freeze_execution_code
    from scripts.posttraining.assess_car_round5 import context
    prepared = json.loads((campaign/'configs/prepared.json').read_text())
    for ref in [*prepared['arms'].values(), prepared['units'],prepared['startup'],prepared['disposition'],prepared['source_reference']]: checked(ref)
    startup = json.loads(checked(prepared['startup']).read_text())
    active_arms = qualification(startup, prepared['units']['sha256'])
    if active_arms != prepared['active_arms'] or set(prepared['arms']) != set(active_arms):
        raise ValueError('active arms do not match bounded startup disposition')
    cpu = cpu_forecast(startup)
    with locked_ledger(campaign/'operations/gpu_ledger.json') as state:
        if state['active']: raise ValueError('finish startup GPU leases before freeze')
        current = spent(state)
    costs = forecast(already=current, native_epoch_seconds=float(startup['native_epoch_seconds']),
        direct_epoch_seconds=(float(startup['direct_epoch_seconds']) if ARMS[1] in active_arms else None))
    if costs['final_epoch'] < prepared['final_epoch']:
        raise ValueError('remaining budget no longer fits prepared endpoint')
    costs['final_epoch'] = prepared['final_epoch']
    costs['forecast_gpu_seconds'] = (current + costs['final_epoch'] *
        (costs['native_epoch_seconds'] + (costs['direct_epoch_seconds'] or 0.)) +
        costs['assessment_gpu_seconds'] + costs['reserve_gpu_seconds'])
    generation = cpu.get('historical_generation_gpu_seconds')
    estimated_evaluation_wall = (cpu['wall_seconds'] + 2 * generation
                                 if measured_forecast(generation) else None)
    training_wall_serial = costs['final_epoch'] * (costs['native_epoch_seconds'] + (costs['direct_epoch_seconds'] or 0.))
    costs['evaluation_wall_forecast'] = {'seconds': estimated_evaluation_wall,
        'fraction_of_serial_training_wall': (estimated_evaluation_wall / training_wall_serial
                                             if estimated_evaluation_wall is not None else None),
        'basis': 'CPU assessment wall forecast plus twice historical generation GPU seconds; serial comparison, not an adoption gate',
        'target_fraction': .20}
    assessment = context(campaign)
    code = freeze_execution_code(ROOT, campaign/'operations/code', [
        'scripts/posttraining/run_car_round5.py', 'scripts/posttraining/assess_car_round5.py',
        'scripts/posttraining/assess_car_round2_checkpoints.py', 'scripts/posttraining/freeze_car_event.py',
        'scripts/posttraining/prepare_car_round4.py'])
    sources = {key:reference(assessment[key]) for key in ('panel_path','source_receipt','baseline_path','freeze_path')}
    immutable_json(campaign/'configs/formal_freeze.json', {**prepared, 'schema':'car_round5_freeze_v1',
        'final_epoch':costs['final_epoch'], 'forecast':costs, 'code_root':str(code),
        'code_sha256':code.name, 'assessment_sources':sources, 'created_unix':time.time(),
        'cpu_assessment_forecast':cpu, 'test_locked':True})
    print(json.dumps(costs, indent=2))


def train_internal(args, campaign):
    import yaml
    from phycoflow_reconstruction.training.post_training import run_post_training
    manifest = json.loads((campaign/'configs/formal_freeze.json').read_text())
    if manifest.get('schema') != 'car_round5_freeze_v1' or ROOT.resolve() != Path(manifest['code_root']).resolve():
        raise ValueError('training must run from the frozen code snapshot')
    if manifest.get('test_locked') is not True or args.arm not in manifest['active_arms']:
        raise ValueError('TEST must remain locked and the arm must be active')
    if args.stop_epoch not in (100, manifest['final_epoch']): raise ValueError('invalid frozen stop epoch')
    token = os.environ.get('CAR_ROUND5_LEASE_TOKEN')
    with locked_ledger(campaign/'operations/gpu_ledger.json') as state:
        entry = state['active'].get(token)
        if not entry or entry['phase'] != 'formal_training': raise ValueError('formal GPU lease required')
    cfg = yaml.safe_load(checked(manifest['arms'][args.arm]).read_text())
    if args.stop_epoch > 100:
        decision = json.loads((campaign/'operations/learning_review_epoch_100.json').read_text())
        if args.resume is None or args.arm not in decision['continued_arms']:
            raise ValueError('continuation requires a recorded review and resume')
        review = json.loads(checked(decision['assessment']).read_text())
        if (review.get('epoch') != 100 or review.get('split') != 'validation'
                or review.get('task_scope') != 'both tasks excluded'
                or args.arm not in review.get('candidates', {})):
            raise ValueError('continuation requires this arm in the task-free epoch-100 review')
    if args.resume:
        saved = yaml.safe_load((args.resume/'resolved_config.yaml').read_text())
        if saved != cfg: raise ValueError('resume must preserve the saved scientific config')
    started, limit = float(entry['start_unix']), float(entry['max_seconds'])
    requested = False
    def request_stop(_signal, _frame):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    def stop(step, _run):
        return step >= args.stop_epoch * 12 or requested or time.time() >= started + limit - 300
    run = run_post_training(cfg, case_dir=Path(os.environ.get('CAR_REPO_ROOT', ROOT))/'cases/turbulent_combustion',
                            resume=args.resume, budget_stop_requested=stop)
    print(run, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    subs = p.add_subparsers(dest='action', required=True)
    job = subs.add_parser('lease')
    job.add_argument('--phase', choices=('startup_initial','startup_correction','formal_training','assessment'), required=True)
    job.add_argument('--max-seconds', type=float, required=True)
    job.add_argument('--correction-evidence', type=Path)
    job.add_argument('command', nargs=argparse.REMAINDER)
    prep = subs.add_parser('prepare')
    prep.add_argument('--units', type=Path, required=True)
    prep.add_argument('--startup', type=Path, required=True)
    prep.add_argument('--disposition', type=Path, required=True)
    subs.add_parser('freeze')
    public_train = subs.add_parser('train')
    public_train.add_argument('--arm', choices=ARMS, required=True)
    public_train.add_argument('--stop-epoch', type=int, required=True)
    public_train.add_argument('--max-seconds', type=float, required=True)
    public_train.add_argument('--resume', type=Path)
    train = subs.add_parser('train-internal')
    train.add_argument('--arm', choices=ARMS, required=True)
    train.add_argument('--stop-epoch', type=int, required=True)
    train.add_argument('--resume', type=Path)
    plan = subs.add_parser('forecast')
    plan.add_argument('--native-epoch-seconds', type=float, required=True)
    plan.add_argument('--direct-epoch-seconds', type=float)
    plan.add_argument('--assessment-seconds', type=float, default=3600.)
    plan.add_argument('--reserve-seconds', type=float, default=10800.)
    args = p.parse_args()
    campaign = args.campaign.resolve(strict=True)
    if args.action == 'lease':
        if args.command[:1] == ['--']: args.command = args.command[1:]
        lease(args, campaign)
    elif args.action == 'prepare':
        prepare(args, campaign)
    elif args.action == 'freeze':
        freeze(campaign)
    elif args.action == 'train':
        if args.max_seconds <= 300: raise ValueError('training lease must include 300 seconds for recovery')
        frozen = json.loads((campaign/'configs/formal_freeze.json').read_text())
        driver = Path(frozen['code_root'])/'scripts/posttraining/run_car_round5.py'
        args.command = [sys.executable, str(driver), '--campaign', str(campaign), 'train-internal',
                        '--arm', args.arm, '--stop-epoch', str(args.stop_epoch)]
        if args.resume: args.command += ['--resume', str(args.resume.resolve(strict=True))]
        args.phase, args.correction_evidence = 'formal_training', None
        os.environ.setdefault('CAR_REPO_ROOT', str(ROOT))
        lease(args, campaign)
    elif args.action == 'train-internal':
        train_internal(args, campaign)
    else:
        with locked_ledger(campaign / 'operations/gpu_ledger.json') as state:
            if state['active']: raise ValueError('forecast requires all preparation leases closed')
            current = spent(state)
        print(json.dumps(forecast(already=current, native_epoch_seconds=args.native_epoch_seconds,
              direct_epoch_seconds=args.direct_epoch_seconds, assessment_seconds=args.assessment_seconds,
              reserve_seconds=args.reserve_seconds), indent=2))

if __name__ == '__main__':
    main()
