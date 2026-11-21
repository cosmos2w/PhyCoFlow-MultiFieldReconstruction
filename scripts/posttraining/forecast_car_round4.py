#!/usr/bin/env python3
"""Resolve the bounded Round 4 launch forecast from measured qualification."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def bound(path):
    return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--source-qualification',type=Path,required=True)
    p.add_argument('--teacher-qualification',type=Path,required=True)
    p.add_argument('--current-route',type=Path,required=True)
    args=p.parse_args()
    campaign=args.campaign.resolve(strict=True)
    manifest_path=campaign/'configs/train_manifest.json'
    storage_path=campaign/'operations/teacher_recovery_io.json'
    historical_path=campaign.parent/'round_3/assessment/epoch_500/costs.json'
    manifest,source,teacher,route,storage,historical=map(read,[manifest_path,args.source_qualification,args.teacher_qualification,args.current_route,storage_path,historical_path])
    manifest_hash=bound(manifest_path)['sha256']
    if (any(item.get('status')!='completed' for item in (source,teacher,route,storage))
            or source['manifest_sha256']!=manifest_hash or teacher['manifest_sha256']!=manifest_hash
            or source['source_parameter_buffer_sha256']!=teacher['source_parameter_buffer_sha256']
            or source['native_batch']!=64 or source['native_query_points']!=4096
            or source['rows_generated']!=64 or manifest['population']!=8000):
        raise ValueError('Completed matched qualification and resolved TRAIN8000 are required')
    check=next(row for row in teacher['checks'] if row['batch_size']==64)
    if not check['all_finite'] or not check['all_native_constraints_checked_fp64']:
        raise ValueError('Teacher native numerical qualification failed')
    safety=1.2
    # Use the measured slower update, include target reads twice to cover hashing,
    # and reserve two seconds per epoch for learner persistence and plotting.
    target_read=source['source_row_read_seconds']*64/source['source_rows_read']
    rf=max(source['native_update_seconds'])
    truth_epoch=safety*(12*(rf+source['native_batch_input_seconds'])+2.)
    teacher_target_epoch=safety*(12*(rf+source['native_batch_input_seconds']+2*target_read)+2.)
    source_generation=safety*((8000-source['rows_generated'])*statistics.mean(source['row_end_to_end_seconds'])+30.)
    # All forward-only calibration/replay and the full native audit are bounded
    # by the measured complete differentiated update, deliberately conservatively.
    diagnostic_batches=math.ceil(800/64)+2*math.ceil(800/64)
    feasibility_batches=math.ceil(8000/64)
    teacher_batches=1200+diagnostic_batches+feasibility_batches
    teacher_seconds=safety*(teacher_batches*check['total_wall_seconds']+100*storage['write_and_fsync_seconds']+60.)
    teacher_cpu=safety*(teacher_batches*check['cpu_process_seconds']+100*storage['process_cpu_seconds']+60.)
    native_candidate=500*teacher_target_epoch
    native_control=500*truth_epoch
    assessment_gpu=900.
    assessment_cpu=1.5*historical['assessment']['cpu_seconds']
    cpu_tail=1.5*historical['assessment']['cpu_scoring_wall_seconds']
    assessment_elapsed=assessment_gpu+cpu_tail+300.
    failed=list((campaign/'operations/failed_launches').glob('*.json'))
    failed_upper=sum(read(path)['conservative_gpu_seconds_charge'] for path in failed)
    qualification=source['wall_seconds']+teacher['wall_seconds']+route['wall_seconds']+failed_upper+60.
    # The final 60-second qualification allowance covers uninstrumented early
    # startup intervals; it is a conservative allowance, never a measured zero.
    candidate=qualification+source_generation+teacher_seconds+native_candidate+assessment_gpu
    whole=candidate+native_control
    baseline=historical['training']['FFM_SourceScaledRisk_Uniform_Repair']['occupied_gpu_seconds']
    checks={'whole_round_24_gpu_hours':whole<=86400,
            'inclusive_candidate_twofold_target':candidate<=baseline/2,
            'target_preparation_2_gpu_hours':source_generation<=7200,
            'teacher_4_gpu_hours':teacher_seconds<=14400,
            'teacher_6_cpu_worker_hours':teacher_cpu<=21600,
            'students_10_gpu_hours':native_candidate+native_control<=36000,
            'assessment_1_gpu_hour':assessment_gpu<=3600,
            'assessment_2_cpu_worker_hours':assessment_cpu<=7200}
    output=campaign/'operations/feasibility_gate.json'
    if output.exists(): raise FileExistsError(output)
    result={'schema':'car_round4_feasibility_gate_v1','admitted':all(checks.values()),
            'manifest_sha256':manifest_hash,'candidate_inclusive_gpu_seconds':candidate,'whole_round_gpu_seconds':whole,
            'checks':checks,'safety_factor':safety,'reference_epoch':500,'reference_gpu_seconds':baseline,
            'inclusive_candidate_ratio_forecast':baseline/candidate,
            'forecast_gpu_seconds':{'qualification_measured_plus_upper_bound':qualification,'full_train_target_preparation_remaining':source_generation,
              'teacher_including_calibration_replay_audit_and_persistence':teacher_seconds,'native_amortization_500_epochs':native_candidate,
              'native_supervision_500_epochs':native_control,'assessment':assessment_gpu},
            'teacher_cpu_worker_seconds':teacher_cpu,'assessment_cpu_worker_seconds':assessment_cpu,
            'assessment_added_wall_seconds':assessment_elapsed,'assessment_cpu_queue_and_delivery_allowance_seconds':300.,
            'assessment_wall_fraction_of_teacher_plus_learners':assessment_elapsed/(teacher_seconds+native_candidate+native_control),
            'epochs':{'teacher_initial_and_maximum':100,'student_review':100,'student_conditional_maximum':500},
            'native_epoch_forecast_seconds':{'teacher_targets':teacher_target_epoch,'dense_truth':truth_epoch},
            'teacher_epoch_reserves':{'gpu_wall_seconds':safety*(12*check['total_wall_seconds']+storage['write_and_fsync_seconds']),
               'cpu_worker_seconds':safety*(12*check['cpu_process_seconds']+storage['process_cpu_seconds'])},
            'teacher_closeout_reserves':{'wall_seconds':safety*((diagnostic_batches+feasibility_batches)*check['total_wall_seconds']+60.),
               'cpu_worker_seconds':safety*((diagnostic_batches+feasibility_batches)*check['cpu_process_seconds']+60.)},
            'interpretation':'Forecast, not measured efficacy or a speedup result. Full TRAIN target construction is charged once; qualified identical 64 rows are reused. Teacher timing is conservative for forward-only work; native target I/O is approximate and must be refreshed before students. Preserve matched VALIDATION100 and independent J4; TEST locked.',
            'bindings':{'source_qualification':bound(args.source_qualification),'teacher_qualification':bound(args.teacher_qualification),
              'current_route':bound(args.current_route),'storage_qualification':bound(storage_path),'historical_costs':bound(historical_path),
              'failed_attempts':[bound(path) for path in failed]}}
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:result[key] for key in ('admitted','candidate_inclusive_gpu_seconds','whole_round_gpu_seconds','teacher_cpu_worker_seconds','checks')},indent=2))


def nonnegative(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


def refresh_calculation(gate, qualification_seconds, generation_seconds, failed_seconds, *, source_qualification_seconds=0.):
    """Replace preparation forecasts with observed leases and explicit upper bounds."""
    costs = gate['forecast_gpu_seconds']
    qualification_seconds = nonnegative(qualification_seconds, 'qualification seconds')
    generation_seconds = nonnegative(generation_seconds, 'generation seconds')
    failed_seconds = nonnegative(failed_seconds, 'failed startup bound')
    source_qualification_seconds = nonnegative(source_qualification_seconds, 'source qualification seconds')
    if source_qualification_seconds > qualification_seconds:
        raise ValueError('SOURCE qualification cannot exceed total qualification')
    # The original 60-second startup allowance remains conservative and is
    # separate from the individually charged failed startup receipts.
    prior = qualification_seconds + generation_seconds + failed_seconds + 60.
    teacher = nonnegative(costs['teacher_including_calibration_replay_audit_and_persistence'], 'teacher') + 120.
    teacher_cpu = nonnegative(gate['teacher_cpu_worker_seconds'], 'teacher CPU') + 120.
    learner = nonnegative(costs['native_amortization_500_epochs'], 'candidate learner')
    control = nonnegative(costs['native_supervision_500_epochs'], 'control learner')
    assessment = nonnegative(costs['assessment'], 'assessment')
    candidate = prior + teacher + learner + assessment
    whole = candidate + control
    reference = nonnegative(gate['reference_gpu_seconds'], 'reference')
    cpu_assessment = nonnegative(gate['assessment_cpu_worker_seconds'], 'assessment CPU')
    ew = max(30, math.ceil(nonnegative(gate['teacher_epoch_reserves']['gpu_wall_seconds'], 'epoch wall')))
    ec = max(1, math.ceil(nonnegative(gate['teacher_epoch_reserves']['cpu_worker_seconds'], 'epoch CPU')))
    cw = max(1, math.ceil(nonnegative(gate['teacher_closeout_reserves']['wall_seconds'], 'closeout wall') + 120.))
    cc = max(1, math.ceil(nonnegative(gate['teacher_closeout_reserves']['cpu_worker_seconds'], 'closeout CPU') + 120.))
    lease = math.ceil(teacher)
    checks = {'initial_gate_admitted':gate.get('admitted') is True,
              'whole_round_24_gpu_hours':whole <= 86400,
              'inclusive_candidate_twofold_target':candidate <= reference / 2 and candidate <= 35317,
              # Conservatively include the whole original SOURCE qualification,
              # which also contained the 64 reused rows and native timing.
              'target_preparation_2_gpu_hours':generation_seconds + source_qualification_seconds <= 7200,
              'teacher_4_gpu_hours':lease <= 14400,
              'teacher_6_cpu_worker_hours':teacher_cpu <= 21600,
              'teacher_lease_covers_reserves':lease > ew + cw,
              'students_10_gpu_hours':learner + control <= 36000,
              'assessment_1_gpu_hour':assessment <= 3600,
              'assessment_2_cpu_worker_hours':cpu_assessment <= 7200}
    return {'admitted':all(checks.values()), 'checks':checks,
            'candidate_inclusive_gpu_seconds':candidate, 'whole_round_gpu_seconds':whole,
            'measured_qualification_gpu_seconds':qualification_seconds,
            'measured_generation_gpu_seconds':generation_seconds,
            'target_preparation_upper_gpu_seconds':generation_seconds + source_qualification_seconds,
            'failed_startup_conservative_gpu_seconds':failed_seconds,
            'unmeasured_startup_allowance_gpu_seconds':60.,
            'teacher_lease_seconds':lease, 'teacher_cpu_worker_forecast_seconds':teacher_cpu,
            'teacher_epoch_reserve_seconds':ew, 'teacher_epoch_reserve_cpu_seconds':ec,
            'teacher_closing_reserve_seconds':cw, 'teacher_closing_reserve_cpu_seconds':cc,
            'teacher_field_reduction_planning_allowance_seconds':120.,
            'remaining_gpu_forecast_seconds':{'teacher':teacher, 'candidate_student':learner,
                                               'control_student':control, 'assessment':assessment}}


def checked_binding(ref):
    path = Path(ref['path'])
    if bound(path)['sha256'] != ref['sha256']:
        raise ValueError(f'bound receipt changed: {path}')
    return path


def validate_bank(bank, manifest, source, manifest_hash):
    if (bank.get('schema') != 'car_round4_source_bank_v1' or bank.get('status') != 'completed'
            or bank.get('split') != 'train' or bank.get('population') != manifest.get('population')
            or manifest.get('population') != 8000 or bank.get('members') != 4
            or bank.get('native_points') != 40300 or bank.get('units') != 'model field units'
            or bank.get('manifest_sha256') != manifest_hash
            or bank.get('source_checkpoint_sha256') != manifest.get('source_checkpoint_sha256')
            or bank.get('source_parameter_buffer_sha256') != source.get('source_parameter_buffer_sha256')
            or len(bank.get('rows_sha256', [])) != 8000):
        raise ValueError('incomplete or mismatched full TRAIN SOURCE bank')


def refresh_main(argv):
    p = argparse.ArgumentParser(description='Refresh teacher budget from terminal preparation receipts')
    p.add_argument('--campaign', type=Path, required=True)
    args = p.parse_args(argv)
    c = args.campaign.resolve(strict=True)
    # Share the campaign ledger's lock with launchers for the whole check and
    # immutable receipt write, so a new lease cannot race the snapshot.
    with (c/'operations/gpu_ledger.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        refresh_locked(c)


def refresh_locked(c):
    output = c/'operations/teacher_launch_forecast.json'
    if output.exists():
        raise FileExistsError(output)
    ledger_path = c/'operations/gpu_ledger.json'
    ledger = read(ledger_path)
    if ledger.get('schema') != 'car_gpu_ledger_v1' or ledger.get('active'):
        raise ValueError('all GPU leases must be closed before refreshing the forecast')
    gate_path, manifest_path = c/'operations/feasibility_gate.json', c/'configs/train_manifest.json'
    gate, manifest = read(gate_path), read(manifest_path)
    mh = bound(manifest_path)['sha256']
    if gate.get('schema') != 'car_round4_feasibility_gate_v1' or gate.get('manifest_sha256') != mh:
        raise ValueError('initial gate or manifest differs')
    bindings = gate['bindings']
    source_path = checked_binding(bindings['source_qualification'])
    teacher_path = checked_binding(bindings['teacher_qualification'])
    route_path = checked_binding(bindings['current_route'])
    for name in ('storage_qualification', 'historical_costs'):
        checked_binding(bindings[name])
    source, teacher, route = map(read, (source_path, teacher_path, route_path))
    if (any(x.get('status') != 'completed' for x in (source, teacher, route))
            or source.get('manifest_sha256') != mh or teacher.get('manifest_sha256') != mh
            or source.get('source_parameter_buffer_sha256') != teacher.get('source_parameter_buffer_sha256')):
        raise ValueError('matched source and teacher qualification required')
    bank_path = c/'artifacts/source_bank/receipt.json'
    bank = read(bank_path)
    validate_bank(bank, manifest, source, mh)
    # Construction has already hashed each row. Teacher reads verify the bytes.
    for i, expected in enumerate(bank['rows_sha256']):
        path = bank_path.parent/f'{i:04d}.pt'
        side = path.with_suffix('.sha256')
        if (not isinstance(expected, str) or len(expected) != 64 or not path.is_file()
                or not side.is_file() or side.read_text().strip() != expected):
            raise ValueError(f'SOURCE bank row {i} missing or hash mismatch')
    code_path = c/'configs/execution_code_learning.json'
    code = read(code_path)
    root = Path(code['code_root']).resolve(strict=True)
    inventory = read(root/'execution_receipt.json')
    files = inventory.get('files')
    if (code.get('schema') != 'car_round4_execution_code_v1' or root.parent != (c/'operations/code').resolve()
            or root.name != code.get('code_sha256')
            or not isinstance(files, dict) or not files
            or inventory.get('schema') != 'car_execution_code_v1'
            or inventory.get('sha256') != root.name
            or hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest() != root.name):
        raise ValueError('learning capsule receipt identity differs')
    for name, expected in files.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or bound(path)['sha256'] != expected:
            raise ValueError(f'learning capsule file changed: {name}')
    tokens = set()
    qualification = generation = 0.
    cpu_clocks, preparation_refs = [], []
    qualified_tokens = {source['token'], teacher['token'], route['token']}
    found = set()
    for entry in ledger['closed']:
        token, phase = entry['token'], entry['phase']
        if token in tokens:
            raise ValueError('duplicate ledger token')
        tokens.add(token)
        seconds = nonnegative(entry['seconds'], 'closed lease seconds')
        if phase == 'generate':
            generation += seconds
            path = c/'operations/preparation_jobs'/f'{token}.json'
            doc = read(path)
            if (doc.get('token') != token or doc.get('manifest_sha256') != mh
                    or doc.get('command') != 'generate' or doc.get('status') not in ('completed','failed')
                    or abs(nonnegative(doc['wall_seconds'], 'generation receipt seconds') - seconds) > .01):
                raise ValueError('generation lease and receipt differ')
            if doc.get('status') == 'completed' and (doc.get('source_parameter_buffer_sha256') != bank['source_parameter_buffer_sha256']
                    or Path(doc.get('output', '')).resolve() != bank_path.parent):
                raise ValueError('completed generation receipt does not bind SOURCE bank')
            preparation_refs.append((path, doc))
        elif phase in ('qualify', 'teacher_numerical_qualification', 'current_route_timing'):
            if token not in qualified_tokens:
                raise ValueError('unexpected unbound qualification lease')
            doc = {source['token']:source, teacher['token']:teacher, route['token']:route}[token]
            if abs(nonnegative(doc['wall_seconds'], 'qualification receipt seconds') - seconds) > .01:
                raise ValueError('qualification lease and receipt differ')
            qualification += seconds
            found.add(token)
        else:
            raise ValueError(f'unexpected closed phase before teacher launch: {phase}')
        cpu = doc.get('process_cpu_seconds', doc.get('cpu_seconds'))
        cpu_clocks.append({'token':token, 'phase':phase, 'cpu_seconds':None if cpu is None else nonnegative(cpu, 'CPU clock')})
    if found != qualified_tokens or not preparation_refs or sum(x.get('status') == 'completed' for _, x in preparation_refs) != 1:
        raise ValueError('missing matched qualifications or terminal generation receipt')
    failed_paths = sorted((c/'operations/failed_launches').glob('*.json'))
    original = {Path(x['path']).resolve():x for x in bindings['failed_attempts']}
    if not set(original).issubset(set(failed_paths)):
        raise ValueError('original failed startup receipts missing')
    failed = 0.
    for path in failed_paths:
        if path.resolve() in original:
            checked_binding(original[path.resolve()])
        doc = read(path)
        if doc.get('token') in tokens:
            raise ValueError('failed startup overlaps an already charged lease')
        failed += nonnegative(doc['conservative_gpu_seconds_charge'], 'failed startup charge')
    result = refresh_calculation(gate, qualification, generation, failed,
                                 source_qualification_seconds=source['wall_seconds'])
    result.update(schema='car_round4_teacher_launch_forecast_v1', manifest_sha256=mh,
                  refresh_tool_sha256=bound(Path(__file__))['sha256'],
                  interpretation='Measured closed GPU leases plus separate failed-start upper bounds and 60-second startup allowance; teacher field reductions add a 120-second wall and CPU planning allowance, not a measured bound.',
                  recorded_cpu_clocks=cpu_clocks,
                  bindings={'initial_gate':bound(gate_path), 'manifest':bound(manifest_path),
                            'source_bank':bound(bank_path), 'learning_code':bound(code_path),
                            'learning_capsule_receipt':bound(root/'execution_receipt.json'),
                            'ledger':bound(ledger_path), 'preparation_receipts':[bound(path) for path, _ in preparation_refs],
                            'failed_startups':[bound(path) for path in failed_paths]})
    with output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k:result[k] for k in ('admitted','candidate_inclusive_gpu_seconds','whole_round_gpu_seconds','teacher_lease_seconds','checks')}, indent=2))


if __name__=='__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'refresh-teacher':
        refresh_main(sys.argv[2:])
    else:
        main()
