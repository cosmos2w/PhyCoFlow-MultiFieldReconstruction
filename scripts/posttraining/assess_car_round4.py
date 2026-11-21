#!/usr/bin/env python3
"""Assess Round 4 raw students against the frozen Round 3 VALIDATION100 observer."""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import platform
import signal
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from scripts.posttraining import assess_car_round3 as prior
from scripts.posttraining.prepare_car_round4 import device_identity

TEACHER = 'FFM_SSS_Target_Amortization'
NATIVE = 'FFM_Native_Supervised_Continuation'
ARMS = (TEACHER, NATIVE)
_check_prior_bank = prior.check_bank


def check_bank(bank, ctx, epoch, arm):
    receipt = _check_prior_bank(bank, ctx, epoch, arm)
    for prefix in ('native_config', 'native_checkpoint'):
        prior.verify({'path':receipt[prefix+'_path'], 'sha256':receipt[prefix+'_sha256']})
    for name, path in (('train_manifest', ctx['campaign']/'configs/train_manifest.json'),
                       ('train_sensor_manifest', ctx['campaign']/'configs/train_sensor_manifest.json'),
                       ('teacher_qualification', ctx['campaign']/'operations/teacher_qualification.json')):
        if receipt[name+'_sha256'] != prior.hash_file(path):
            raise ValueError(f'{name} differs from assessment bank')
    gate = prior.load_json(ctx['campaign']/'operations/teacher_qualification.json')
    if receipt['teacher_checkpoint_sha256'] != gate['teacher_checkpoint_sha256']:
        raise ValueError('teacher checkpoint differs from assessment bank')
    return receipt


def context(campaign):
    campaign = campaign.resolve(strict=True)
    ctx = prior.context(campaign.parent/'round_3')
    ctx['round3_campaign'] = ctx['campaign']
    selection_path = campaign/'configs/visual_selection.json'
    selection = prior.load_json(selection_path)
    if (selection.get('schema') != 'car_round4_source_defined_visual_cases_v1'
            or selection.get('candidates_used') is not False or selection.get('test_locked') is not True
            or selection['panel']['sha256'] != prior.hash_file(ctx['panel_path'])
            or selection['source_bank']['sha256'] != prior.hash_file(ctx['source_receipt'])
            or selection['baseline']['sha256'] != prior.hash_file(ctx['baseline_path'])):
        raise ValueError('SOURCE-defined visualization selection or frozen panel changed')
    for row in selection['cases'].values():
        if ctx['source']['sample_ids'][row['ordinal']] != row['sample_id']:
            raise ValueError('visual selection sample identity changed')
    # Both Round 4 arms inherit the identical Round 3 frozen SOURCE config.
    inherited = ctx['prepared']['arms']['FFM_SourceScaledRisk_Uniform_Repair']
    ctx['prepared'] = copy.deepcopy(ctx['prepared'])
    ctx['prepared']['arms'] = {arm:inherited for arm in ARMS}
    ctx['campaign'] = campaign
    ctx['visual_selection_sha256'] = prior.hash_file(selection_path)
    prior.ARMS = ARMS
    prior.check_bank = check_bank
    # Assessment produces evidence; adoption follows only after inclusive cost.
    prior.finalize_decision = lambda result: None
    return ctx


def generate(args, ctx):
    started = time.time()
    cpu_started = time.process_time()
    device = device_identity()
    from phycoflow_reconstruction.evaluation.adaptive_campaign import atomic_json, locked_ledger, spent
    import torch
    from phycoflow_reconstruction.data.manifest import SensorManifest
    from phycoflow_reconstruction.training.run_store import load_project_checkpoint
    from phycoflow_reconstruction.training.scalable_fitting import _digest
    expected = ctx['campaign']/'assessment'/'banks'/args.arm/args.run.name/f'epoch_{args.epoch}'
    if args.output.resolve() != expected.resolve():
        raise ValueError('Output must be the single campaign bank for this named attempt and checkpoint')
    if not os.environ.get('KEOPS_CACHE_FOLDER') or args.device != 'cuda:0':
        raise ValueError('Select KeOps cache and logical cuda:0')
    if not (math.isfinite(args.max_seconds) and math.isfinite(args.save_margin_seconds)
            and 0 < args.save_margin_seconds < args.max_seconds):
        raise ValueError('A bounded lease and positive save margin are required')
    run = args.run.resolve(strict=True)
    native_config_path = run/'resolved_config.json'
    native_config = prior.load_json(native_config_path)
    native_checkpoint_path = run/f'checkpoint_epoch_{args.epoch}.pt'
    native_checkpoint = torch.load(native_checkpoint_path, map_location='cpu', weights_only=False)
    scientific = native_checkpoint['scientific']
    manifest_path = ctx['campaign']/'configs/train_manifest.json'
    sensor_path = ctx['campaign']/'configs/train_sensor_manifest.json'
    teacher_gate_path = ctx['campaign']/'operations/teacher_qualification.json'
    teacher_gate = prior.load_json(teacher_gate_path)
    teacher_path = (ctx['campaign']/teacher_gate['checkpoint']).resolve(strict=True)
    exported = load_project_checkpoint(run/'checkpoints'/f'epoch_{args.epoch}.pt')
    kind = 'teacher' if args.arm == TEACHER else 'truth'
    identity = scientific['identity']
    if (not teacher_path.is_relative_to(ctx['campaign'])
            or native_checkpoint['epoch'] != args.epoch or native_config['identity'] != identity
            or _digest(native_checkpoint['model']) != _digest(exported['model'])
            or identity['target_kind'] != kind
            or identity['source_sha256'] != ctx['source']['source_sha256']
            or identity['manifest_sha256'] != SensorManifest.load(sensor_path).digest()
            or teacher_gate.get('admitted') is not True
            or teacher_gate.get('train_manifest_sha256') != prior.hash_file(manifest_path)
            or teacher_gate.get('teacher_checkpoint_sha256') != prior.hash_file(teacher_path)
            or teacher_gate.get('completed_epochs') != 100
            or (kind == 'teacher' and identity.get('bank_sha256') != teacher_gate['teacher_checkpoint_sha256'])):
        raise ValueError('Native checkpoint, training configuration, SOURCE, manifest or teacher binding differs')
    token = uuid.uuid4().hex
    ledger = ctx['campaign']/'operations/gpu_ledger.json'
    with locked_ledger(ledger) as state:
        if any('max_seconds' not in entry for entry in state['active'].values()):
            raise ValueError('Concurrent lease reservation unknown')
        reserved = sum(max(0., float(v['max_seconds']) - (started-float(v['start_unix']))) for v in state['active'].values())
        if spent(state) + reserved + args.max_seconds > 86400 or any(v['gpu_uuid'] == device['uuid'] for v in state['active'].values()):
            raise ValueError('Round 4 reserved GPU budget or device unavailable')
        state['active'][token] = {'gpu_uuid':device['uuid'], 'phase':'round4_assessment',
            'start_unix':started, 'pid':os.getpid(), 'max_seconds':args.max_seconds}
    args.runtime_device = {'gpu_uuid':device['uuid'], 'gpu_index':device['index'], 'gpu_name':device['name'],
        'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'], 'platform':platform.platform(),
        'python':sys.version, 'keops_cache':os.environ['KEOPS_CACHE_FOLDER'], 'lease_token':token,
        'max_seconds':args.max_seconds, 'save_margin_seconds':args.save_margin_seconds,
        'visual_selection_sha256':ctx['visual_selection_sha256'], 'round4_code_sha256':prior.hash_file(Path(__file__)),
        'native_config_path':str(native_config_path), 'native_config_sha256':prior.hash_file(native_config_path),
        'native_checkpoint_path':str(native_checkpoint_path), 'native_checkpoint_sha256':prior.hash_file(native_checkpoint_path),
        'train_manifest_sha256':prior.hash_file(manifest_path), 'train_sensor_manifest_sha256':prior.hash_file(sensor_path),
        'teacher_qualification_sha256':prior.hash_file(teacher_gate_path),
        'teacher_checkpoint_sha256':teacher_gate['teacher_checkpoint_sha256'],
        'native_scientific_definitions':scientific,
        'inference_config_role':'inherited inference configuration; native scientific definitions recorded separately'}
    previous = signal.getsignal(signal.SIGALRM)
    def expire(_signal, _frame): raise TimeoutError('Assessment reached its save margin')
    outcome = {'status':'failed'}
    try:
        signal.signal(signal.SIGALRM, expire)
        signal.alarm(max(1, math.ceil(args.max_seconds-args.save_margin_seconds)))
        prior.generate(args, ctx, config_name='inherited_inference_config.yaml')
        outcome['status'] = 'completed'
    except BaseException as exc:
        outcome.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
        end = time.time()
        with locked_ledger(ledger) as state:
            entry = state['active'].pop(token)
            state['closed'].append({**entry, 'token':token, 'end_unix':end, 'seconds':end-started})
        atomic_json(ctx['campaign']/'operations/assessment_jobs'/f'{token}.json',
                    {**args.runtime_device, **outcome, 'arm':args.arm, 'epoch':args.epoch,
                     'start_unix':started, 'end_unix':end, 'wall_seconds':end-started,
                     'process_cpu_seconds':time.process_time()-cpu_started})


def score(args, ctx):
    # Prior scorer's endpoint switch excludes both task observables at epoch 100.
    prior.score(args, ctx)
    annotate(args, ctx)


def annotate(args, ctx):
    from phycoflow_reconstruction.evaluation.adaptive_campaign import atomic_json
    path = args.output/'assessment.json'
    try:
        result = prior.load_json(path)
        result['round4_code_sha256'] = prior.hash_file(Path(__file__))
        result['visual_selection_sha256'] = ctx['visual_selection_sha256']
        histories = {}
        for arm, bank in ((TEACHER, args.raw_bank), (NATIVE, args.scaled_bank)):
            receipt = prior.load_json(bank/'receipt.json')
            history_path = Path(receipt['checkpoint']).parent.parent/'logs/history.jsonl'
            rows = [json.loads(line) for line in history_path.read_text().splitlines() if line]
            rows = [row for row in rows if row['epoch'] <= result['epoch']]
            if [row['epoch'] for row in rows] != list(range(1, result['epoch'] + 1)):
                raise ValueError('native training history missing completed epochs')
            histories[arm] = {'history_sha256':prior.hash_file(history_path),
                'native_rf_loss': [row['native_rf_loss'] for row in rows]}
        result['native_learning_histories'] = histories
        atomic_json(path, result)
    except BaseException as exc:
        receipt_path = args.output/'receipt.json'
        receipt = prior.load_json(receipt_path)
        receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        atomic_json(receipt_path, receipt)
        raise


def terminal_100(args, ctx):
    from phycoflow_reconstruction.evaluation.adaptive_campaign import immutable_json
    decision = prior.load_json(args.decision)
    banks = {TEACHER:args.raw_bank.resolve(strict=True), NATIVE:args.scaled_bank.resolve(strict=True)}
    bank_hashes = {arm:prior.hash_file(bank/'receipt.json') for arm,bank in banks.items()}
    if (decision.get('schema') != 'car_round4_terminal_decision_v1'
            or decision.get('decision') != 'stop_at_epoch_100' or decision.get('authorize_endpoint_tasks') is not True
            or decision.get('review_sha256') != prior.hash_file(args.review)
            or decision.get('bank_receipt_sha256') != bank_hashes):
        raise ValueError('Terminal decision must bind the task-free review and both cached banks')
    if ctx['campaign'] not in args.decision.resolve().parents:
        raise ValueError('terminal decision must be campaign-owned')
    # Round 3 scorer validates this translated immutable decision and reuses the
    # existing epoch-100 prediction banks; it does not generate predictions.
    translated = args.output.parent/(args.output.name + '_task_authorization.json')
    immutable_json(translated, {'schema':'car_round3_terminal_decision_v1', 'decision':'stop_at_epoch_100',
        'epoch':100, 'authorize_endpoint_tasks':True, 'review_sha256':prior.hash_file(args.review),
        'bank_receipt_sha256':bank_hashes, 'round4_decision_sha256':prior.hash_file(args.decision)})
    forwarded = copy.copy(args); forwarded.decision = translated
    prior.score_early_endpoint_tasks(forwarded, ctx)
    annotate(args, ctx)


def decision(result, candidate_gpu_seconds, reference_gpu_seconds):
    """Apply the declared field, fair-core, task and inclusive-cost gates."""
    if result.get('epoch') not in (100, 500) or not result.get('task_scope', '').startswith('reused development tasks'):
        raise ValueError('terminal task-bearing assessment required')
    if not all(math.isfinite(x) and x > 0 for x in (candidate_gpu_seconds, reference_gpu_seconds)):
        raise ValueError('positive candidate and matched-age reference GPU costs required')
    out = copy.deepcopy(result)
    for arm in ARMS:
        row = out['candidates'][arm]
        diffs, fidelity, task = row['normalized_core_differences'], row['fidelity'], row['reused_tasks']
        row['scientific_eligible'] = bool(fidelity['primary_pass'] and fidelity['legacy_member_pass']
            and all(value <= .02 for value in diffs.values())
            and any(value <= -.05 for value in diffs.values())
            and task['regional_normalized_improvement'] >= .05
            and task['event_normalized_difference'] <= .05)
    teacher, native = (out['candidates'][arm] for arm in ARMS)
    regional_gain = teacher['reused_tasks']['regional_normalized_improvement'] - native['reused_tasks']['regional_normalized_improvement']
    useful = teacher['scientific_eligible'] and regional_gain >= .02
    ratio = reference_gpu_seconds / candidate_gpu_seconds
    if useful and ratio >= 2:
        choice = TEACHER
    elif native['scientific_eligible']:
        choice = NATIVE
    else:
        choice = 'retain SOURCE/C8'
    out['round4_decision'] = {'recommendation':choice, 'teacher_scientifically_useful':useful,
        'teacher_regional_improvement_vs_native':regional_gain,
        'candidate_inclusive_gpu_seconds':candidate_gpu_seconds, 'round3_reference_gpu_seconds':reference_gpu_seconds,
        'inclusive_speed_ratio':ratio, 'acceleration_pass':ratio >= 2,
        'scientifically_useful_but_cost_failed':bool(useful and ratio < 2)}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    sub = p.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('generate'); gen.add_argument('--arm', choices=ARMS, required=True)
    gen.add_argument('--epoch', type=int, choices=(100,500), required=True); gen.add_argument('--run', type=Path, required=True)
    gen.add_argument('--device', required=True); gen.add_argument('--output', type=Path, required=True)
    gen.add_argument('--max-seconds', type=float, required=True); gen.add_argument('--save-margin-seconds', type=float, required=True)
    for name in ('score', 'terminal-100'):
        s = sub.add_parser(name)
        if name == 'score': s.add_argument('--epoch', type=int, choices=(100,500), required=True)
        else:
            s.add_argument('--decision', type=Path, required=True); s.add_argument('--review', type=Path, required=True)
        s.add_argument('--raw-bank', type=Path, required=True); s.add_argument('--scaled-bank', type=Path, required=True)
        s.add_argument('--output', type=Path, required=True)
    d = sub.add_parser('decide'); d.add_argument('--assessment', type=Path, required=True)
    d.add_argument('--cost-receipt', type=Path, required=True); d.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); ctx = context(args.campaign)
    if args.command == 'generate': generate(args, ctx)
    elif args.command == 'score': score(args, ctx)
    elif args.command == 'terminal-100': terminal_100(args, ctx)
    else:
        from phycoflow_reconstruction.evaluation.adaptive_campaign import immutable_json
        if ctx['campaign'] not in args.assessment.resolve().parents or ctx['campaign'] not in args.cost_receipt.resolve().parents or ctx['campaign']/'assessment' not in args.output.resolve().parents:
            raise ValueError('Assessment, cost receipt and decision must be campaign-owned')
        cost = prior.load_json(args.cost_receipt)
        assessment = prior.load_json(args.assessment)
        if (cost.get('schema') != 'car_round4_inclusive_cost_v1' or cost.get('complete') is not True
                or cost.get('assessment_sha256') != prior.hash_file(args.assessment)
                or cost.get('reference_epoch') != assessment.get('epoch')
                or cost.get('cost_basis') not in ('measured', 'conservative_upper_bound')):
            raise ValueError('Complete assessment-bound inclusive candidate cost receipt required')
        pieces = cost['candidate_gpu_seconds']
        if set(pieces) != {'source_generation','teacher','student','assessment'} or any(not math.isfinite(v) or v < 0 for v in pieces.values()):
            raise ValueError('All four inclusive GPU cost components are required')
        overhead = cost.get('candidate_overhead_gpu_seconds')
        if not isinstance(overhead, (int, float)) or isinstance(overhead, bool) or not math.isfinite(overhead) or overhead < 0:
            raise ValueError('Measured or bounded candidate qualification/failure overhead is required')
        reference_path = prior.verify(cost['reference_receipt'])
        frozen_cost = (ctx['round3_campaign']/'assessment/epoch_500/costs.json').resolve(strict=True)
        if reference_path != frozen_cost:
            raise ValueError('Reference cost must be the retained Round 3 cost record')
        historical = prior.load_json(reference_path)['training']['FFM_SourceScaledRisk_Uniform_Repair']['segments']
        reference_seconds = sum(segment['occupied_gpu_seconds'] for segment in historical
                                if segment['through_epoch'] <= assessment['epoch'])
        if not math.isclose(float(cost['reference_gpu_seconds']), reference_seconds, rel_tol=1e-12):
            raise ValueError('Reference cost does not match the declared learning age')
        result = decision(assessment, sum(pieces.values()) + overhead, reference_seconds)
        result['round4_decision'].update(cost_basis=cost.get('cost_basis'),
            bounded_unknown_components=cost.get('bounded_unknown_components', []),
            candidate_overhead_gpu_seconds=overhead)
        result['round4_decision'].update(cost_receipt_sha256=prior.hash_file(args.cost_receipt),
            assessment_sha256=prior.hash_file(args.assessment), visual_selection_sha256=ctx['visual_selection_sha256'])
        immutable_json(args.output, result)

if __name__ == '__main__': main()
