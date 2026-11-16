import importlib.util
import json
from pathlib import Path

import pytest

from phycoflow_reconstruction.evaluation.round2_costs import ARMS, reconcile

spec = importlib.util.spec_from_file_location(
    'cost_registry', Path(__file__).resolve().parents[1] / 'scripts/analysis/assemble_car_round2_cost_manifest.py')
registry_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(registry_tool)


def test_registry_expands_receipts_and_reports_unattributed_jobs(tmp_path):
    receipt = tmp_path/'source.receipt.json'
    receipt.write_text(json.dumps({'token':'first', 'gpu_uuid':'gpu', 'start_unix':1, 'end_unix':5, 'status':'failed'}))
    unregistered = tmp_path/'unknown.receipt.json'
    unregistered.write_text('{}')
    ledger = {'schema':'car_gpu_ledger_v1','active':{}, 'closed':[
        {'token':'first', 'gpu_uuid':'gpu', 'start_unix':1, 'end_unix':5},
        {'token':'second', 'gpu_uuid':'gpu', 'start_unix':6, 'end_unix':7}]}
    registry = {'schema':'car_round2_cost_registry_v1', 'registrations':[
        {'kind':'gpu','scope':'source_preparation','owner':'ffm_shared','receipt':str(receipt)}],
        'receipt_inventory_globs':[str(tmp_path/'*.receipt.json')]}
    manifest = registry_tool.assemble(tmp_path, registry, ledger)
    assert manifest['gpu_jobs'][0]['allocations'] == {ARMS[0]:.5, ARMS[1]:.5}
    assert any('unknown.receipt.json' in item for item in manifest['assembly_missing'])
    assert any('second' in item for item in manifest['assembly_missing'])
    result = reconcile(campaign=tmp_path, ledger=ledger, manifest=manifest, family='ffm')
    assert not result['complete']
    assert result['arms'][ARMS[0]]['failed_attempt_seconds'] == 2.


def test_checkpoint_extras_are_constructed_from_explicit_job_epochs(tmp_path):
    receipt = tmp_path/'assessment.json'
    receipt.write_text(json.dumps({'start_unix':2, 'end_unix':3, 'status':'completed'}))
    registry = {'schema':'car_round2_cost_registry_v1', 'registrations':[
        {'kind':'cpu','scope':'assessment','owner':ARMS[0], 'receipt':str(receipt),
         'independent':True, 'checkpoint_epochs':[0, 1000]}],
        'receipt_inventory_globs':[str(receipt)]}
    manifest = registry_tool.assemble(tmp_path, registry, {'closed':[], 'active':{}})
    job_id = manifest['cpu_jobs'][0]['id']
    assert manifest['checkpoint_extras'][ARMS[0]]['0'] == [job_id]
    assert manifest['checkpoint_extras'][ARMS[0]]['1000'] == [job_id]
    assert manifest['checkpoint_extras'][ARMS[0]]['100'] == []
    with pytest.raises(ValueError, match='shared owner'):
        registry['registrations'][0]['scope'] = 'source_preparation'
        registry_tool.assemble(tmp_path, registry, {'closed':[], 'active':{}})


def test_launch_glob_pairs_completions_without_guessing_missing(tmp_path):
    launches = tmp_path/'launches'
    completions = tmp_path/'completions'
    launches.mkdir()
    completions.mkdir()
    for token in ('first', 'second'):
        (launches/f'{token}.json').write_text(json.dumps({'token':token, 'gpu_uuid':'gpu', 'started_unix':1}))
    (completions/'first.json').write_text(json.dumps({'token':'first', 'finished_unix':2, 'status':'completed'}))
    registry = {'schema':'car_round2_cost_registry_v1', 'receipt_inventory_globs':[str(launches/'*.json')],
                'registrations':[{'kind':'gpu', 'scope':'training', 'owner':ARMS[0],
                                  'launch_glob':str(launches/'*.json'), 'completion_dir':str(completions)}]}
    manifest = registry_tool.assemble(tmp_path, registry, {'closed':[], 'active':{}})
    assert [job['token'] for job in manifest['gpu_jobs']] == ['first']
    assert any('missing receipt' in item and 'second.json' in item for item in manifest['assembly_missing'])


def test_assessment_gpu_enclosure_is_excluded_but_cpu_child_must_be_counted(tmp_path):
    group = tmp_path/'group.json'
    gpu_child = tmp_path/'gpu_child.json'
    cpu_child = tmp_path/'cpu_child.json'
    group.write_text(json.dumps({'schema':'car_round2_assessment_sequence_v1',
                                 'wall_accounting':'group wall encloses child costs; do not add both'}))
    gpu_child.write_text(json.dumps({'schema':'car_round2_assessment_sequence_child_v1','kind':'gpu',
                                     'wall_accounting':'encloses the GPU lease; do not add both'}))
    cpu_child.write_text(json.dumps({'schema':'car_round2_assessment_sequence_child_v1','kind':'cpu',
                                     'wall_accounting':'CPU child wall includes its scoring receipt; do not add both',
                                     'start_unix':1, 'end_unix':3, 'returncode':1}))
    registry = {'schema':'car_round2_cost_registry_v1', 'registrations':[],
                'receipt_inventory_globs':[str(tmp_path/'*.json')],
                'enclosing_receipts':[str(group), str(gpu_child), str(cpu_child)]}
    with pytest.raises(ValueError, match='sequence GPU/group'):
        registry_tool.assemble(tmp_path, registry, {'closed':[], 'active':{}})
    registry['enclosing_receipts'].remove(str(cpu_child))
    registry['registrations'].append({'kind':'cpu','scope':'assessment','owner':ARMS[0],
                                      'receipt':str(cpu_child),'independent':True,'checkpoint_epochs':[0]})
    manifest = registry_tool.assemble(tmp_path, registry, {'closed':[], 'active':{}})
    result = reconcile(campaign=tmp_path, ledger={'schema':'car_gpu_ledger_v1','closed':[], 'active':{}},
                       manifest=manifest, family='ffm')
    assert result['arms'][ARMS[0]]['failed_attempt_seconds'] == 2


def test_explicit_registry_open_items_block_completion(tmp_path):
    registry = {'schema':'car_round2_cost_registry_v1', 'registrations':[],
                'receipt_inventory_globs':[str(tmp_path/'*.json')],
                'open_items':['pending terminal qualification receipts']}
    manifest = registry_tool.assemble(tmp_path, registry, {'closed':[], 'active':{}})
    assert 'pending terminal qualification receipts' in manifest['assembly_missing']
