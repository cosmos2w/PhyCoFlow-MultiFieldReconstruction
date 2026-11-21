#!/usr/bin/env python3
"""Assemble measured or conservative Round 4 costs from existing job receipts.

No job, device inspection or model evaluation is performed. Qualification and
failed-startup charges are included conservatively in the candidate components;
control and unrelated work are separately identified.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

TEACHER = 'FFM_SSS_Target_Amortization'
NATIVE = 'FFM_Native_Supervised_Continuation'


def load(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve(strict=True)
    return {'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def enclosing_elapsed(jobs, failed, extra=()):
    """Wall span through observed completions, with failed-end upper bounds."""
    starts, ends, bounded = [], [], False
    for _, job in [*jobs, *extra]:
        if not finite(job.get('start_unix')) or not finite(job.get('end_unix')):
            return None, 'unavailable: an enclosing clock is missing'
        starts.append(job['start_unix']); ends.append(job['end_unix'])
    for _, row in failed:
        try:
            start = datetime.fromisoformat(row['launch_utc'].replace('Z', '+00:00')).timestamp()
            end = datetime.fromisoformat(row['failure_discovered_before_utc'].replace('Z', '+00:00')).timestamp()
        except (KeyError, ValueError, TypeError):
            return None, 'unavailable: failed-start enclosing clock is missing'
        if end < start: return None, 'unavailable: failed-start clocks are inconsistent'
        starts.append(start); ends.append(end); bounded = True
    if not starts: return None, 'unavailable: no enclosing clocks'
    return max(ends)-min(starts), ('enclosing wall including gaps; failed ends are upper bounds' if bounded
                                    else 'enclosing wall including gaps')


def add_cpu_only_auxiliary(auxiliary, clocks, unknown, *, scope, ref, doc, cpu_key):
    """Record disjoint final and prior-attempt clocks without inventing values."""
    if doc.get('status') != 'completed' or doc.get('gpu_used') is not False:
        raise ValueError(f'{scope} must be a completed CPU-only receipt')
    attempts = [(scope, doc.get(cpu_key), doc.get('wall_seconds'))]
    attempts.extend((f'{scope}: prior attempt {i}', attempt.get('cpu_seconds'), attempt.get('wall_seconds'))
                    for i, attempt in enumerate(doc.get('previous_failed_attempts', []), 1))
    for name, cpu, wall in attempts:
        auxiliary.append({'scope':name, 'receipt':ref,
                          'process_cpu_seconds':cpu if finite(cpu) else None,
                          'wall_seconds':wall if finite(wall) else None})
        for label, value in (('process CPU', cpu), ('wall', wall)):
            if not finite(value): unknown.append({'scope':name, 'timer':label, 'receipt':ref})
    if finite(doc.get('start_unix')) and finite(doc.get('end_unix')):
        clocks.append((ref, doc))
    else:
        unknown.append({'scope':scope, 'timer':'absolute start/end', 'receipt':ref,
                        'effect':'excluded from enclosing wall boundary'})


def add_cpu_resume_auxiliary(auxiliary, clocks, unknown, *, ref, doc):
    """Count the parent's CPU plus waited children once; child receipts are nested."""
    scope = 'CPU-only real FFM fresh-process resume qualification'
    if doc.get('schema') != 'car_round4_native_real_ffm_resume_cpu_v1' or doc.get('gpu_used') is not False:
        raise ValueError('real FFM resume qualification must be a CPU-only parent receipt')
    parent = doc.get('parent_process_cpu_seconds')
    children = doc.get('waited_children_cpu_seconds')
    cpu = parent + children if finite(parent) and finite(children) else None
    wall = doc.get('wall_seconds')
    auxiliary.append({'scope':scope, 'receipt':ref, 'status':doc.get('status'),
        'process_cpu_seconds':cpu, 'wall_seconds':wall if finite(wall) else None,
        'cpu_scope':'parent plus waited children; child receipt CPU timers are nested diagnostics'})
    for label, value in (('parent process CPU', parent), ('waited children CPU', children), ('wall', wall)):
        if not finite(value): unknown.append({'scope':scope, 'timer':label, 'receipt':ref})
    if finite(doc.get('start_unix')) and finite(doc.get('end_unix')):
        clocks.append((ref, doc))
    else:
        unknown.append({'scope':scope, 'timer':'absolute start/end', 'receipt':ref,
                        'effect':'excluded from enclosing wall boundary'})


def add_standalone_review_auxiliary(auxiliary, clocks, unknown, *, ref, doc, kind, manifest_sha256):
    """Account only the standalone review timer, excluding nested teacher costs."""
    if kind == 'warm_forecast':
        scope = 'CPU-only warm teacher cost review'
        if (doc.get('schema') != 'car_round4_teacher_warm_cache_forecast_v1'
                or doc.get('train_manifest_sha256') != manifest_sha256):
            unknown.append({'scope':scope, 'timer':'receipt identity', 'receipt':ref})
            return
        timer = doc.get('review_timer', {})
        start, end = timer.get('start_unix'), timer.get('end_unix')
        cpu = timer.get('process_cpu_seconds')
        wall = end-start if finite(start) and finite(end) and end >= start else None
        attempts = []
    elif kind == 'visual_recovery':
        scope = 'CPU-only epoch-1 history recovery visual check'
        if doc.get('schema') != 'car_round4_epoch1_history_visual_v1' or not str(doc.get('scope', '')).startswith('CPU-only'):
            unknown.append({'scope':scope, 'timer':'receipt identity', 'receipt':ref})
            return
        start, end = doc.get('wall_start_unix'), doc.get('wall_end_unix')
        cpu, wall = doc.get('process_cpu_seconds'), doc.get('elapsed_wall_seconds')
        attempts = doc.get('earlier_setup_attempts', [])
    else:
        raise ValueError('unknown auxiliary review kind')
    auxiliary.append({'scope':scope, 'receipt':ref, 'process_cpu_seconds':cpu if finite(cpu) else None,
                      'wall_seconds':wall if finite(wall) else None,
                      'cpu_scope':'standalone review only; nested teacher clocks excluded'})
    for label, value in (('process CPU',cpu), ('wall',wall)):
        if not finite(value): unknown.append({'scope':scope, 'timer':label, 'receipt':ref})
    if finite(start) and finite(end) and end >= start:
        clocks.append((ref, {'start_unix':start, 'end_unix':end}))
    else:
        unknown.append({'scope':scope, 'timer':'absolute start/end', 'receipt':ref,
                        'effect':'excluded from enclosing wall boundary'})
    for i, attempt in enumerate(attempts, 1):
        name = f'{scope}: prior attempt {i}'
        prior_cpu, prior_wall = attempt.get('process_cpu_seconds'), attempt.get('enclosing_wall_seconds_rounded')
        auxiliary.append({'scope':name, 'receipt':ref,
                          'process_cpu_seconds':prior_cpu if finite(prior_cpu) else None,
                          'wall_seconds':prior_wall if finite(prior_wall) else None,
                          'wall_precision':'rounded' if finite(prior_wall) else 'unknown'})
        if not finite(prior_cpu): unknown.append({'scope':name, 'timer':'process CPU', 'receipt':ref})
        if not finite(prior_wall): unknown.append({'scope':name, 'timer':'wall', 'receipt':ref})
        unknown.append({'scope':name, 'timer':'absolute start/end', 'receipt':ref,
                        'effect':'excluded from enclosing wall boundary'})


def aggregate(ledger, jobs, failed, *, attempt, epoch, teacher_only=False, partial_teacher=False):
    """Return disjoint inclusive GPU components and explicitly unknown clocks.

    Jobs are pairs (receipt binding, receipt). GPU time comes from the ledger,
    matched by token; its elapsed occupied lease is not added again as job wall.
    Formal teacher CPU is the sum of per-lease ledger deltas, never cumulative
    ``cpu_worker_seconds`` from resumed receipts. Unknown CPU is reported apart
    from GPU eligibility. Failed starts enter only through documented bounds.
    """
    components = {name:0. for name in ('source_generation','teacher','student','assessment','candidate_overhead')}
    cpu = {name:0. for name in components}
    cpu_unknown = []
    missing = []
    bounded = []
    included = []
    excluded = []
    by_token = {}
    for ref, job in jobs:
        token = job.get('token', job.get('lease_token'))
        if not token or token in by_token:
            missing.append('missing or duplicate job token')
        else:
            by_token[token] = (ref, job)
    if ledger.get('active'):
        missing.append('active leases have no final measured occupation')
    tokens = set()
    teacher_terminal = False
    native_terminal = False
    assess_epochs = set()
    generated = False
    for entry in ledger.get('closed', []):
        token = entry.get('token'); phase = entry.get('phase')
        if not token or token in tokens:
            missing.append('missing or duplicate closed lease token'); continue
        tokens.add(token)
        pair = by_token.get(token)
        if phase in ('qualify','generate'):
            component = 'source_generation'
        elif phase in ('current_route_timing','teacher_numerical_qualification'):
            component = 'candidate_overhead'
        elif phase == 'endpoint_teacher':
            component = 'teacher'
        elif phase == TEACHER:
            component = None if teacher_only else 'student'
        elif phase == NATIVE:
            component = None
        elif phase == 'round4_assessment':
            component = None if teacher_only else 'assessment'
        else:
            missing.append(f'unclassified lease phase {phase!r}'); continue
        if pair is None:
            # A known excluded control lease cannot affect candidate occupation.
            if component is not None: missing.append(f'job receipt missing for lease {token}')
            else: excluded.append({'token':token,'phase':phase,'reason':'control or excluded work; receipt missing'})
            continue
        ref, job = pair
        if phase == 'round4_assessment' and job.get('arm') != TEACHER:
            component = None
        if component is None:
            excluded.append({'token':token,'phase':phase,'receipt':ref, 'reason':'control or not needed for teacher closeout'})
            continue
        wall = job.get('wall_seconds'); elapsed = entry.get('seconds')
        if not finite(wall) or not finite(elapsed) or not math.isclose(wall, elapsed, rel_tol=1e-9, abs_tol=1e-4):
            missing.append(f'lease and receipt GPU timer absent or disagree: {token}'); continue
        components[component] += elapsed
        included.append({'token':token,'phase':phase,'status':job.get('status'),'gpu_seconds':elapsed,
                         'receipt':ref,'component':component})
        if phase == 'generate' and job.get('status') == 'completed': generated = True
        if phase == 'endpoint_teacher' and job.get('status') == 'completed' and job.get('completed_epochs') == 100:
            teacher_terminal = True
        if partial_teacher and phase == 'endpoint_teacher' and job.get('status') == 'paused' and job.get('completed_epochs') == epoch:
            teacher_terminal = True
        if phase == TEACHER:
            # Unknown/other attempts are charged conservatively, never dropped.
            if job.get('attempt') == attempt and job.get('status') in ('completed','paused') and job.get('completed_epochs') == epoch:
                native_terminal = True
        if phase == 'round4_assessment' and job.get('status') == 'completed':
            assess_epochs.add(job.get('epoch'))
        seconds = entry.get('cpu_seconds') if phase == 'endpoint_teacher' else job.get('process_cpu_seconds')
        if finite(seconds): cpu[component] += seconds
        else:
            cpu_unknown.append({'token':token,'phase':phase,'reason':'complete per-job CPU timer unavailable'})
            if phase in ('generate', 'endpoint_teacher', TEACHER, 'round4_assessment'):
                missing.append(f'required formal process CPU timer missing: {token}')
    for token in by_token.keys() - tokens:
        ref, job = by_token[token]
        if job.get('status') == 'running':
            missing.append(f'unclosed running job {token}')
        else:
            missing.append(f'job receipt missing closed ledger lease {token}')
    for ref, doc in failed:
        if doc.get('token') and doc['token'] in tokens:
            missing.append(f'failed startup bound overlaps measured lease {doc["token"]}')
            continue
        amount = doc.get('conservative_gpu_seconds_charge')
        if not finite(amount):
            missing.append(f'unbounded failed startup {ref["path"]}')
        else:
            components['candidate_overhead'] += amount
            bounded.append({'receipt':ref,'component':'candidate_overhead','gpu_seconds_upper_bound':amount,
                            'reason':doc.get('charge_interpretation','documented conservative bound')})
    if not generated: missing.append('completed full SOURCE generation job missing')
    if not teacher_terminal:
        missing.append('matching paused partial teacher job missing' if partial_teacher else 'completed 100-epoch teacher job missing')
    if not teacher_only:
        if not native_terminal: missing.append('named learner attempt endpoint job missing')
        if 100 not in assess_epochs or (epoch == 500 and 500 not in assess_epochs):
            missing.append('candidate prediction jobs missing for required review or endpoint')
    return {'components':components, 'cpu_by_component':cpu, 'cpu_unknown':cpu_unknown,
            'included_jobs':included, 'excluded_jobs':excluded, 'bounded_unknown_components':bounded,
            'missing':missing, 'cost_basis':'conservative_upper_bound' if bounded else 'measured'}


def validate_budget_stop(decision, forecast, manifest_sha256):
    """Require an explicit early stop whose bound forecast exceeds a fixed cap."""
    epoch = decision.get('completed_epochs')
    resource_keys = ('reserve_committed_elsewhere_gpu_seconds', 'candidate_other_gpu_seconds',
                     'whole_other_gpu_seconds')
    if (forecast.get('base_teacher_gpu_allowance_seconds') != 14400
            or forecast.get('variation_reserve_gpu_seconds') != 18000
            or forecast.get('candidate_gpu_cap_seconds') != 35317
            or forecast.get('whole_gpu_cap_seconds') != 86400
            or any(not finite(forecast.get(key)) for key in resource_keys)
            or forecast['reserve_committed_elsewhere_gpu_seconds'] > 18000):
        raise ValueError('invalid teacher reserve or remaining campaign budget')
    effective_gpu_cap = min(14400 + 18000 - forecast[resource_keys[0]],
                            35317 - forecast[resource_keys[1]],
                            86400 - forecast[resource_keys[2]])
    if (decision.get('schema') != 'car_round4_teacher_budget_stop_v1'
            or decision.get('decision') != 'stop_cost_infeasible'
            or decision.get('train_manifest_sha256') != manifest_sha256
            or type(epoch) is not int or not 1 <= epoch < 100
            or decision.get('teacher_gpu_cap_seconds') != effective_gpu_cap
            or decision.get('teacher_cpu_cap_seconds') != 21600
            or forecast.get('schema') != 'car_round4_teacher_warm_cache_forecast_v1'
            or forecast.get('train_manifest_sha256') != manifest_sha256
            or forecast.get('completed_epochs') != epoch
            or forecast.get('teacher_checkpoint_sha256') != decision.get('checkpoint', {}).get('sha256')
            or forecast.get('teacher_history_sha256') != decision.get('history', {}).get('sha256')
            or forecast.get('teacher_gpu_cap_seconds') != effective_gpu_cap
            or forecast.get('teacher_cpu_cap_seconds') != 21600
            or forecast.get('safety_factor') != 1.2
            or forecast.get('training_window_epochs') != list(range(max(1, epoch-4), epoch+1))
            or forecast.get('remaining_epochs') != 100-epoch):
        raise ValueError('invalid manifest-bound partial teacher budget-stop decision')
    keys = ('projected_teacher_gpu_seconds','projected_teacher_cpu_worker_seconds')
    if any(not finite(decision.get(k)) or decision[k] != forecast.get(k) for k in keys):
        raise ValueError('budget projections must equal the bound forecast')
    if decision[keys[0]] <= effective_gpu_cap and decision[keys[1]] <= 21600:
        raise ValueError('neither teacher budget cap is forecast to be exceeded')
    return epoch


def campaign_binding(campaign, ref):
    path = (campaign / ref['path']).resolve(strict=True)
    if not path.is_relative_to(campaign) or binding(path)['sha256'] != ref['sha256']:
        raise ValueError('budget-stop artifact is outside the campaign or changed')
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--teacher-only',action='store_true')
    mode.add_argument('--teacher-budget-stop',type=Path)
    p.add_argument('--attempt')
    p.add_argument('--assessment',type=Path)
    p.add_argument('--score-receipt',type=Path,action='append',default=[])
    args=p.parse_args()
    campaign=args.campaign.resolve(strict=True)
    if campaign/'assessment' not in args.output.resolve().parents:
        p.error('output must be under campaign assessment')
    teacher_mode = args.teacher_only or args.teacher_budget_stop is not None
    if not teacher_mode and (not args.attempt or not args.assessment):
        p.error('student cost requires --attempt and --assessment')
    ledger_path=campaign/'operations/gpu_ledger.json'
    manifest_path=campaign/'configs/train_manifest.json'
    ledger=load(ledger_path); manifest=load(manifest_path)
    jobs=[]
    for directory in ('preparation_jobs','teacher_jobs','current_route_timing','native_jobs','assessment_jobs'):
        for path in sorted((campaign/'operations'/directory).glob('*.json')):
            jobs.append((binding(path),load(path)))
    failed=[(binding(path),load(path)) for path in sorted((campaign/'operations/failed_launches').glob('*.json'))]
    assessment=None if teacher_mode else load(args.assessment)
    partial = args.teacher_budget_stop is not None
    stop = None
    if partial:
        stop_path = args.teacher_budget_stop.resolve(strict=True)
        if not stop_path.is_relative_to(campaign/'operations'):
            raise ValueError('budget stop decision must be in campaign operations')
        stop = load(stop_path)
        forecast_path = campaign_binding(campaign, stop['forecast'])
        checkpoint_path = campaign_binding(campaign, stop['checkpoint'])
        history_path = campaign_binding(campaign, stop['history'])
        scientific_path = campaign_binding(campaign, stop['scientific_config'])
        forecast_doc = load(forecast_path)
        epoch = validate_budget_stop(stop, forecast_doc, binding(manifest_path)['sha256'])
        teacher_root = (campaign/'runs/formal/SSS_Empirical_Endpoint_Teacher').resolve()
        if not scientific_path.is_relative_to(teacher_root) or scientific_path.name != 'resolved_scientific_config.json':
            raise ValueError('scientific configuration must belong to a named formal teacher attempt')
        import torch
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True, mmap=True)
        history = [json.loads(line) for line in history_path.read_text().splitlines() if line]
        scientific = load(scientific_path)
        if (saved.get('schema') != 'car_round4_teacher_checkpoint_v1'
                or saved.get('completed_epochs') != epoch or saved.get('history') != history
                or scientific.get('schema') != 'car_round4_teacher_scientific_config_v1'
                or not isinstance(scientific.get('identity'), dict)
                or scientific['identity'].get('train_manifest_sha256') != binding(manifest_path)['sha256']
                or any(saved.get(key) != value for key,value in scientific.get('identity',{}).items())
                or saved.get('normalizers') != scientific.get('normalizers')
                or [row.get('epoch') for row in history] != list(range(1, epoch+1))):
            raise ValueError('partial checkpoint and complete recorded history do not match')
        if ledger.get('active') or any(entry.get('phase') in (TEACHER,NATIVE,'round4_assessment') for entry in ledger.get('closed', [])):
            raise ValueError('partial teacher closeout requires closed leases and no student assessments or training')
        if any((campaign/'runs/formal'/arm).exists() for arm in (TEACHER,NATIVE)):
            raise ValueError('partial teacher closeout requires no student output')
        for path in (campaign/'assessment').rglob('assessment.json'):
            if 'reused development tasks' in load(path).get('task_scope',''):
                raise ValueError('partial teacher closeout requires no terminal task output')
        matching = [job for _,job in jobs if job.get('schema') == 'car_round4_teacher_job_v1'
                    and job.get('status') == 'paused' and job.get('completed_epochs') == epoch
                    and Path(job.get('checkpoint','')).is_file()
                    and binding(Path(job['checkpoint']))['sha256'] == stop['checkpoint']['sha256']]
        if not matching:
            raise ValueError('partial checkpoint must match a closed paused teacher receipt')
    else:
        epoch=100 if teacher_mode else assessment['epoch']
        if epoch not in (100,500): raise ValueError('student endpoint must be epoch 100 or 500')
    result=aggregate(ledger,jobs,failed,attempt=args.attempt,epoch=epoch,teacher_only=teacher_mode,partial_teacher=partial)
    whole_unknown=[]
    if ledger.get('active'): whole_unknown.append('active GPU leases have no final measured time')
    if any(not finite(entry.get('seconds')) for entry in ledger.get('closed', [])):
        whole_unknown.append('one or more closed GPU leases lack a measured duration')
    if any(not finite(doc.get('conservative_gpu_seconds_charge')) or
           (doc.get('token') and any(entry.get('token') == doc['token'] for entry in ledger.get('closed', [])))
           for _,doc in failed):
        whole_unknown.append('failed-start charge is missing or overlaps a closed lease')
    whole_gpu = None if whole_unknown else (sum(entry['seconds'] for entry in ledger.get('closed', [])) +
                 sum(doc['conservative_gpu_seconds_charge'] for _,doc in failed))
    bank_path=campaign/'artifacts/source_bank/receipt.json'
    if not bank_path.is_file():
        result['missing'].append('completed full TRAIN SOURCE bank receipt missing')
        bank_ref=None
    else:
        bank=load(bank_path); bank_ref=binding(bank_path)
        if (bank.get('status') != 'completed' or bank.get('manifest_sha256') != binding(manifest_path)['sha256']
                or bank.get('population') != manifest['population']
                or len(bank.get('rows_sha256', [])) != manifest['population']):
            result['missing'].append('SOURCE bank receipt does not bind the full TRAIN manifest')
        qualified = [job for _,job in jobs if job.get('schema') == 'car_round4_preparation_job_v1'
                     and job.get('command') == 'qualify' and job.get('status') == 'completed'
                     and job.get('rows_generated') == 64 and job.get('manifest_sha256') == binding(manifest_path)['sha256']]
        if len(qualified) != 1:
            result['missing'].append('one matched 64-row SOURCE qualification is required')
        else:
            origin = Path(qualified[0]['output'])
            if len(bank.get('rows_sha256', [])) < 64 or not all((origin/f'{i:04d}.sha256').is_file() and
                       (origin/f'{i:04d}.sha256').read_text().strip() == bank['rows_sha256'][i]
                       for i in range(64)):
                result['missing'].append('qualified SOURCE rows do not match the full bank receipt')
    components=result['components']
    refs={'ledger':binding(ledger_path),'train_manifest':binding(manifest_path),'source_bank':bank_ref}
    auxiliary=[]
    prep_timer=manifest.get('preparation_timer', {})
    if all(finite(prep_timer.get(k)) for k in ('start_unix','end_unix','cpu_seconds')):
        auxiliary.append({'scope':'TRAIN manifest preparation', 'receipt':refs['train_manifest'],
                          'process_cpu_seconds':prep_timer['cpu_seconds'],
                          'wall_seconds':prep_timer['end_unix']-prep_timer['start_unix']})
        auxiliary_clocks=[(None,prep_timer)]
    else:
        auxiliary_clocks=[]
    io_path=campaign/'operations/teacher_recovery_io.json'
    if io_path.is_file():
        io=load(io_path)
        if io.get('status') == 'completed' and all(finite(io.get(k)) for k in ('start_unix','end_unix','process_cpu_seconds','wall_seconds')):
            auxiliary.append({'scope':'CPU-only teacher recovery I/O qualification', 'receipt':binding(io_path),
                              'process_cpu_seconds':io['process_cpu_seconds'], 'wall_seconds':io['wall_seconds']})
            auxiliary_clocks.append((None,io))
    auxiliary_unknown=[]
    for path, scope, cpu_key, schema in (
            (campaign/'operations/student_target_io_timing.json', 'CPU-only student target I/O qualification',
             'process_cpu_seconds', 'car_round4_student_target_io_timing_v1'),
            (campaign/'operations/source_seed_copy.json', 'CPU-only qualified SOURCE row copy',
             'process_cpu_seconds', 'car_round4_source_seed_copy_v1'),
            (campaign/'runs/qualification/SOURCE_Fixed_Case_Report_Validation/receipt.json',
             'CPU-only fixed-case report validation', 'cpu_process_seconds',
             'car_round4_relationship_cpu_numerical_v1'),
            (campaign/'operations/teacher_history_recovery.json',
             'CPU-only teacher history recovery', 'process_cpu_seconds',
             'car_round4_teacher_history_recovery_v1'),
            (campaign/'operations/teacher_epoch020_pause_curve_refresh.json',
             'CPU-only epoch-20 teacher pause curve refresh', 'process_cpu_seconds',
             'car_round4_teacher_pause_curve_refresh_v1')):
        if path.is_file():
            doc=load(path)
            if doc.get('schema') != schema:
                raise ValueError(f'unexpected auxiliary receipt schema: {path}')
            if schema == 'car_round4_source_seed_copy_v1' and doc.get('manifest_sha256') != refs['train_manifest']['sha256']:
                raise ValueError('SOURCE seed copy belongs to another TRAIN manifest')
            add_cpu_only_auxiliary(auxiliary, auxiliary_clocks, auxiliary_unknown,
                                   scope=scope, ref=binding(path), doc=doc, cpu_key=cpu_key)
    resource_code_path = campaign/'configs/execution_code_teacher_resource.json'
    if resource_code_path.is_file():
        resource_code = load(resource_code_path)
        ref = binding(resource_code_path)
        scope = 'CPU-only teacher resource-code freeze and verification'
        cost = resource_code.get('cost', {})
        if resource_code.get('schema') != 'car_round4_teacher_resource_code_v1' or cost.get('gpu_used') is not False:
            auxiliary_unknown.append({'scope':scope, 'timer':'receipt identity', 'receipt':ref})
        else:
            auxiliary.append({'scope':scope, 'receipt':ref,
                'process_cpu_seconds':cost.get('process_cpu_seconds') if finite(cost.get('process_cpu_seconds')) else None,
                'wall_seconds':cost.get('wall_seconds') if finite(cost.get('wall_seconds')) else None})
            for label, key in (('process CPU','process_cpu_seconds'), ('wall','wall_seconds')):
                if not finite(cost.get(key)): auxiliary_unknown.append({'scope':scope, 'timer':label, 'receipt':ref})
            auxiliary_unknown.append({'scope':scope, 'timer':'absolute start/end', 'receipt':ref,
                                      'effect':'excluded from enclosing wall boundary'})
    resume_cpu_path = (campaign/'runs/qualification/'
                       'FFM_Native_Truth_Fresh_Process_Resume_Qualification_v1/receipt.json')
    if resume_cpu_path.is_file():
        add_cpu_resume_auxiliary(auxiliary, auxiliary_clocks, auxiliary_unknown,
                                 ref=binding(resume_cpu_path), doc=load(resume_cpu_path))
    visual_path = campaign/'runs/qualification/Teacher_Epoch_1_History_Recovery_Visual/receipt.json'
    if visual_path.is_file():
        add_standalone_review_auxiliary(auxiliary, auxiliary_clocks, auxiliary_unknown,
            ref=binding(visual_path), doc=load(visual_path), kind='visual_recovery',
            manifest_sha256=refs['train_manifest']['sha256'])
    for path in sorted((campaign/'assessment/teacher_cost_reviews').glob('*/warm_forecast.json')):
        add_standalone_review_auxiliary(auxiliary, auxiliary_clocks, auxiliary_unknown,
            ref=binding(path), doc=load(path), kind='warm_forecast',
            manifest_sha256=refs['train_manifest']['sha256'])
    cpu_teacher=result['cpu_by_component']['teacher']
    if any(x['phase'] == 'endpoint_teacher' for x in result['cpu_unknown']): cpu_teacher=None
    elapsed, elapsed_basis = enclosing_elapsed(jobs, failed, auxiliary_clocks)
    base={'train_manifest_sha256':refs['train_manifest']['sha256'], 'complete':False,
          'cost_basis':result['cost_basis'], 'bounded_unknown_components':result['bounded_unknown_components'],
          'whole_campaign_gpu_seconds':whole_gpu,
          'whole_campaign_cost_basis':'unavailable' if whole_unknown else result['cost_basis'],
          'whole_campaign_unknown':whole_unknown,
          'missing_required_timers_or_artifacts':result['missing'], 'cpu_unknown':result['cpu_unknown'],
          'included_jobs':result['included_jobs'], 'excluded_jobs':result['excluded_jobs'], 'bindings':refs,
          'cpu_worker_seconds':cpu_teacher, 'cpu_worker_seconds_scope':'formal teacher jobs only; summed per-lease CPU deltas',
          'teacher_cpu_worker_seconds':cpu_teacher,
          'auxiliary_cpu_work':auxiliary,
          'auxiliary_cpu_known_seconds':sum(item['process_cpu_seconds'] for item in auxiliary if finite(item['process_cpu_seconds'])),
          'auxiliary_cpu_unknown':auxiliary_unknown,
          'elapsed_wall_seconds':elapsed, 'elapsed_wall_basis':elapsed_basis,
          'interpretation':'The SOURCE 64-row qualification timer includes RF smoke work and is charged once to preparation. Current-route and teacher numerical qualifications and failed-start upper bounds are separate candidate overhead. CPU totals exclude unmeasured qualification CPU; lease times are not added to nested CPU or scoring wall.'}
    if teacher_mode:
        base.update(schema='car_round4_teacher_cost_rejected_closeout_v1' if partial else 'car_round4_teacher_closeout_cost_v1',gpu_seconds={k:components[k] for k in ('source_generation','teacher')},
                    candidate_overhead_gpu_seconds=components['candidate_overhead'])
        if partial:
            base.update(partial_epoch=epoch, checkpoint_sha256=stop['checkpoint']['sha256'],
                        history_sha256=stop['history']['sha256'], forecast_sha256=stop['forecast']['sha256'],
                        budget_stop_decision_sha256=binding(stop_path)['sha256'],
                        budget_stop_decision=binding(stop_path), checkpoint=stop['checkpoint'],
                        history=stop['history'], scientific_config=stop['scientific_config'],
                        scientific_config_sha256=stop['scientific_config']['sha256'], forecast=stop['forecast'],
                        projected_teacher_gpu_seconds=stop['projected_teacher_gpu_seconds'],
                        projected_teacher_cpu_worker_seconds=stop['projected_teacher_cpu_worker_seconds'],
                        teacher_gpu_cap_seconds=stop['teacher_gpu_cap_seconds'], teacher_cpu_cap_seconds=21600,
                        teacher_gpu_budget_inputs={k:forecast_doc[k] for k in (
                            'base_teacher_gpu_allowance_seconds', 'variation_reserve_gpu_seconds',
                            'reserve_committed_elsewhere_gpu_seconds', 'candidate_other_gpu_seconds',
                            'whole_other_gpu_seconds', 'candidate_gpu_cap_seconds', 'whole_gpu_cap_seconds')})
    else:
        if campaign not in args.assessment.resolve(strict=True).parents:
            raise ValueError('assessment must belong to campaign')
        if args.assessment.resolve() != (args.assessment.parent/'assessment.json').resolve():
            raise ValueError('assessment must be the scored assessment.json')
        score_cpu=score_wall=0.; score_schemas=[]; score_refs=[]
        if len({path.resolve() for path in args.score_receipt}) != len(args.score_receipt):
            raise ValueError('duplicate score receipt would double count CPU work')
        for path in args.score_receipt:
            if campaign not in path.resolve(strict=True).parents: raise ValueError('score receipt outside campaign')
            row=load(path); score_refs.append(binding(path)); score_schemas.append((row.get('schema'),row.get('epoch')))
            scored_path=path.parent/'assessment.json'
            if not scored_path.is_file():
                result['missing'].append(f'score assessment missing for {path}')
            else:
                scored=load(scored_path)
                bank_epoch=scored.get('epoch')
                selected_bank=campaign/'assessment/banks'/TEACHER/args.attempt/f'epoch_{bank_epoch}'/'receipt.json'
                if (not selected_bank.is_file() or scored.get('bank_receipt_sha256',{}).get(TEACHER) != binding(selected_bank)['sha256']):
                    result['missing'].append(f'score does not bind candidate attempt bank: {path}')
                else:
                    bank_receipt=load(selected_bank)
                    expected_checkpoint=campaign/'runs/formal'/TEACHER/args.attempt/'checkpoints'/f'epoch_{bank_epoch}.pt'
                    token=bank_receipt.get('lease_token')
                    if (not expected_checkpoint.is_file() or Path(bank_receipt.get('checkpoint','')).resolve() != expected_checkpoint.resolve()
                            or bank_receipt.get('checkpoint_sha256') != binding(expected_checkpoint)['sha256']
                            or not any(j['token']==token and j['phase']=='round4_assessment' and j['status']=='completed'
                                       for j in result['included_jobs'])):
                        result['missing'].append(f'candidate bank does not bind completed attempt and assessment lease: {path}')
            if row.get('status') != 'completed' or not finite(row.get('cpu_seconds')) or not finite(row.get('wall_seconds')):
                result['missing'].append(f'incomplete CPU scoring receipt {path}')
            else:
                score_cpu += row['cpu_seconds']; score_wall += row['wall_seconds']
        elapsed, elapsed_basis = enclosing_elapsed(jobs, failed, auxiliary_clocks + [(None,load(path)) for path in args.score_receipt])
        base['elapsed_wall_seconds'], base['elapsed_wall_basis'] = elapsed, elapsed_basis
        if ('car_round3_cpu_score_v1',100) not in score_schemas:
            result['missing'].append('epoch-100 task-free score receipt missing')
        expected=('car_round3_cpu_score_v1',500) if epoch == 500 else ('car_round3_early_endpoint_task_job_v1',None)
        if expected not in score_schemas: result['missing'].append('terminal scoring receipt missing')
        if not any(path.parent.resolve() == args.assessment.parent.resolve() for path in args.score_receipt):
            result['missing'].append('terminal assessment scoring receipt missing')
        historical_path=(campaign.parent/'round_3/assessment/epoch_500/costs.json').resolve(strict=True)
        segments=load(historical_path)['training']['FFM_SourceScaledRisk_Uniform_Repair']['segments']
        reference=sum(v['occupied_gpu_seconds'] for v in segments if v['through_epoch'] <= epoch)
        if not finite(reference) or reference == 0: result['missing'].append('matched-age reference GPU cost missing')
        base.update(schema='car_round4_inclusive_cost_v1',candidate_gpu_seconds={k:components[k] for k in ('source_generation','teacher','student','assessment')},
            candidate_overhead_gpu_seconds=components['candidate_overhead'],
            assessment_sha256=binding(args.assessment)['sha256'],reference_epoch=epoch,
            reference_gpu_seconds=reference,reference_receipt=binding(historical_path),
            score_receipts=score_refs,assessment_cpu_wall_seconds=score_wall,assessment_cpu_seconds=score_cpu,
            assessment_cpu_worker_seconds=score_cpu,
            native_cpu_worker_seconds=None if any(x['phase']==TEACHER for x in result['cpu_unknown']) else result['cpu_by_component']['student'])
    if any(item['timer'] == 'absolute start/end' for item in auxiliary_unknown):
        base['elapsed_wall_basis'] += '; known-clock span excludes auxiliary work with unknown absolute timestamps'
    if base['elapsed_wall_seconds'] is None:
        result['missing'].append('enclosing elapsed wall clock unavailable')
    base['complete']=not result['missing']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as out: json.dump(base,out,indent=2,allow_nan=False); out.write('\n')
    print(args.output)

if __name__=='__main__': main()
