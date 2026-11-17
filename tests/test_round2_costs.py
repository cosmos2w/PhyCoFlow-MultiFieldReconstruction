import hashlib
import json

import pytest

from phycoflow_reconstruction.evaluation.round2_costs import ARMS, reconcile


def _receipt(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def test_missing_evidence_is_never_reported_as_zero(tmp_path):
    ledger = {'schema': 'car_gpu_ledger_v1', 'active': {}, 'closed': [
        {'token': 'lease', 'gpu_uuid': 'gpu', 'start_unix': 10, 'end_unix': 20, 'seconds': 10},
    ]}
    manifest = {'schema': 'car_round2_cost_allocation_v1'}
    result = reconcile(campaign=tmp_path, ledger=ledger, manifest=manifest, family='ffm')
    assert not result['complete']
    assert result['total_occupied_gpu_seconds'] == 10
    assert any('lease' in message for message in result['missing'])
    assert all(row['serial_cost_seconds'] is None for row in result['arms'].values())


def test_nested_cpu_is_excluded_but_independent_and_failed_work_are_charged(tmp_path):
    full, residual = ARMS[:2]
    weights = {full: 0.5, residual: 0.5}
    ledger = {'schema': 'car_gpu_ledger_v1', 'active': {}, 'closed': [
        {'token': 'lease', 'gpu_uuid': 'gpu', 'start_unix': 10, 'end_unix': 20, 'seconds': 10},
    ]}
    gpu_receipt = _receipt(tmp_path, 'gpu.json', {'token': 'lease', 'gpu_uuid': 'gpu', 'start_unix': 10,
                                                 'end_unix': 20, 'status': 'failed'})
    cpu_receipt = _receipt(tmp_path, 'cpu.json', {'start_unix': 12, 'end_unix': 18, 'status': 'completed'})
    manifest = {'schema': 'car_round2_cost_allocation_v1', 'gpu_jobs': [
        {'id': 'gpu', 'token': 'lease', 'scope': 'source_preparation', 'allocations': weights,
         'receipt': gpu_receipt}], 'cpu_jobs': [
        {'id': 'nested', 'scope': 'source_preparation', 'allocations': weights,
         'receipt': cpu_receipt, 'parent_gpu_token': 'lease'},
        {'id': 'independent', 'scope': 'bank_selection', 'allocations': weights,
         'receipt': cpu_receipt, 'independent': True}]}
    result = reconcile(campaign=tmp_path, ledger=ledger, manifest=manifest, family='ffm')
    assert next(job for job in result['allocated_jobs'] if job['id'] == 'gpu')['gpu_uuid'] == 'gpu'
    assert result['arms'][full]['observed_serial_cost_seconds'] == 8
    assert result['arms'][full]['failed_attempt_seconds'] == 5
    assert next(job for job in result['allocated_jobs'] if job['id'] == 'nested')['seconds'] == 0


def test_bad_common_allocation_rejected(tmp_path):
    ledger = {'schema': 'car_gpu_ledger_v1', 'active': {}, 'closed': [
        {'token': 'lease', 'gpu_uuid': 'gpu', 'start_unix': 10, 'end_unix': 20, 'seconds': 10}]}
    receipt = _receipt(tmp_path, 'gpu.json', {'token': 'lease', 'gpu_uuid': 'gpu', 'start_unix': 10, 'end_unix': 20,
                                              'status': 'completed'})
    manifest = {'schema': 'car_round2_cost_allocation_v1', 'gpu_jobs': [
        {'id': 'gpu', 'token': 'lease', 'scope': 'source_preparation',
         'allocations': {ARMS[0]: 1}, 'receipt': receipt}]}
    with pytest.raises(ValueError, match='shared equally'):
        reconcile(campaign=tmp_path, ledger=ledger, manifest=manifest, family='ffm')


def test_source_residual_microcase_uses_its_saved_nested_timer(tmp_path):
    receipt = _receipt(tmp_path, 'microcase.json', {
        'schema':'car_round2_source_residual_microcase_v1',
        'timer':{'start_unix':1., 'end_unix':3., 'wall_seconds':2., 'gpu_lease':None}})
    manifest = {'schema':'car_round2_cost_allocation_v1', 'cpu_jobs':[
        {'id':'microcase', 'scope':'residual_probe', 'allocations':{ARMS[1]:1.},
         'receipt':receipt, 'independent':True}]}
    ledger = {'schema':'car_gpu_ledger_v1', 'closed':[], 'active':{}}
    result = reconcile(campaign=tmp_path, ledger=ledger, manifest=manifest, family='ffm')
    assert result['arms'][ARMS[1]]['observed_serial_cost_seconds'] == 2.


def test_paused_segments_remain_charged_separately_from_failures(tmp_path):
    start = 1_794_880_000.
    statuses = ('completed', 'budget_paused', 'failed')
    leases, jobs = [], []
    for index, status in enumerate(statuses):
        begin = start + 20 * index
        token = f'lease-{index}'
        lease = {'token': token, 'gpu_uuid': 'gpu', 'start_unix': begin, 'end_unix': begin + 10}
        leases.append(lease)
        receipt = _receipt(tmp_path, f'{token}.json', {**lease, 'status': status})
        jobs.append({'id': token, 'token': token, 'scope': 'qualification',
                     'allocations': {ARMS[0]: 1.}, 'receipt': receipt})
    result = reconcile(campaign=tmp_path, ledger={'schema': 'car_gpu_ledger_v1', 'closed': leases},
                       manifest={'schema': 'car_round2_cost_allocation_v1', 'gpu_jobs': jobs}, family='ffm')
    arm = result['arms'][ARMS[0]]
    assert arm['observed_serial_cost_seconds'] == 30.
    assert arm['gpu_seconds'] == 30.
    assert arm['paused_attempt_seconds'] == 10.
    assert arm['failed_attempt_seconds'] == 10.


@pytest.mark.parametrize('binding', ['receipt', 'launch_completion'])
@pytest.mark.parametrize('endpoint', ['start', 'end'])
@pytest.mark.parametrize('offset,accepted', [(0.0005, True), (0.01, False), (1., False)])
def test_gpu_receipt_lease_tolerance_is_absolute_at_current_unix_time(tmp_path, binding, endpoint, offset, accepted):
    start, end = 1_794_880_000., 1_794_880_010.
    lease = {'token': 'lease', 'gpu_uuid': 'gpu', 'start_unix': start, 'end_unix': end}
    recorded_start = start + (offset if endpoint == 'start' else 0.)
    recorded_end = end + (offset if endpoint == 'end' else 0.)
    job = {'id': 'gpu', 'token': 'lease', 'scope': 'qualification', 'allocations': {ARMS[0]: 1.}}
    if binding == 'receipt':
        job['receipt'] = _receipt(tmp_path, 'receipt.json', {'token': 'lease', 'gpu_uuid': 'gpu',
            'start_unix': recorded_start, 'end_unix': recorded_end, 'status': 'completed'})
    else:
        job['launch'] = _receipt(tmp_path, 'launch.json', {'token': 'lease', 'gpu_uuid': 'gpu',
                                                          'started_unix': recorded_start})
        job['completion'] = _receipt(tmp_path, 'completion.json', {'token': 'lease',
                                             'finished_unix': recorded_end, 'status': 'completed'})
    ledger = {'schema': 'car_gpu_ledger_v1', 'closed': [lease]}
    manifest = {'schema': 'car_round2_cost_allocation_v1', 'gpu_jobs': [job]}
    if accepted:
        result = reconcile(campaign=tmp_path, ledger=ledger, manifest=manifest, family='ffm')
        assert result['arms'][ARMS[0]]['gpu_seconds'] == 10.
    else:
        with pytest.raises(ValueError, match='(does not|do not) match closed lease'):
            reconcile(campaign=tmp_path, ledger=ledger, manifest=manifest, family='ffm')


def test_controller_active_wall_is_charged_once_without_inventing_prefix(tmp_path):
    from phycoflow_reconstruction.evaluation.round2_costs import _prefix_cost, _UnknownActivePrefix

    receipt = _receipt(tmp_path, 'controller.json', {
        'schema': 'car_round2_independent_cpu_job_v1', 'status': 'completed',
        'start_unix': 10., 'end_unix': 110., 'wall_seconds': 100.,
        'blocked_wait_seconds': 20., 'gpu_child_wall_seconds': 50., 'score_wait_seconds': 15.,
        'active_wall_seconds': 15., 'independent_cpu': True,
    })
    job = {'id': 'controller', 'scope': 'qualification', 'allocations': {ARMS[0]: 1.},
           'receipt': receipt, 'independent': True, 'active_wall_field': 'active_wall_seconds'}
    result = reconcile(campaign=tmp_path, ledger={'schema':'car_gpu_ledger_v1','closed':[],'active':{}},
                       manifest={'schema':'car_round2_cost_allocation_v1','cpu_jobs':[job]}, family='ffm')
    row = next(row for row in result['allocated_jobs'] if row['id'] == 'controller')
    assert row['seconds'] == 15.
    assert result['arms'][ARMS[0]]['observed_serial_cost_seconds'] == 15.
    assert _prefix_cost(row, 10.) == 0.
    assert _prefix_cost(row, 110.) == 15.
    with pytest.raises(_UnknownActivePrefix, match='no checkpoint-time placement'):
        _prefix_cost(row, 60.)
    with pytest.raises(ValueError, match='explicitly select'):
        reconcile(campaign=tmp_path, ledger={'schema':'car_gpu_ledger_v1','closed':[],'active':{}},
                  manifest={'schema':'car_round2_cost_allocation_v1','cpu_jobs':[{k:v for k,v in job.items() if k != 'active_wall_field'}]}, family='ffm')


def test_active_timer_requires_valid_independent_disjoint_accounting(tmp_path):
    from phycoflow_reconstruction.evaluation.round2_costs import _active_wall

    base = {'schema':'car_round2_cpu_coordination_v1','start_unix':10.,'end_unix':110.,'wall_seconds':100.,
            'blocked_wait_seconds':20.,'nomination_child_wall_seconds':10.,'parallel_family_enclosure_seconds':60.,
            'controller_active_wall_seconds':10.,'controller_active_wall_independent_cpu':True,'independent_cpu':False}
    assert _active_wall(base, 'controller_active_wall_seconds') == 10.
    with pytest.raises(ValueError, match='disjoint waits'):
        _active_wall({**base, 'controller_active_wall_seconds':11.}, 'controller_active_wall_seconds')
    with pytest.raises(ValueError, match='independent CPU declaration'):
        _active_wall({**base, 'controller_active_wall_independent_cpu':False}, 'controller_active_wall_seconds')


def test_aggregate_active_prefix_bounds_are_honest():
    from phycoflow_reconstruction.evaluation.round2_costs import _prefix_bounds

    job = {'interval': (10., 110.), 'active_wall_seconds': 15.}
    assert _prefix_bounds(job, 9.) == (0., 0.)
    assert _prefix_bounds(job, 60.) == (0., 15.)
    assert _prefix_bounds(job, 110.) == (15., 15.)


def test_checkpoint_bounds_count_assessment_extra_once():
    from phycoflow_reconstruction.evaluation.round2_costs import _checkpoint_cost_bounds

    arm = ARMS[0]
    controller = {'id':'controller', 'interval':(10.,110.), 'active_wall_seconds':15.,
                  'seconds':15., 'weights':{arm:1.}}
    assessment = {'id':'assessment', 'interval':(20.,120.), 'active_wall_seconds':7.,
                  'seconds':7., 'weights':{arm:1.}}
    assert _checkpoint_cost_bounds([controller, assessment], ['assessment'], arm, 60.) == (7.,22.)
    assert _checkpoint_cost_bounds([controller, assessment], ['assessment'], arm, 150.) == (22.,22.)
