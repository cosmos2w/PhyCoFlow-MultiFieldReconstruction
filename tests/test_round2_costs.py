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
