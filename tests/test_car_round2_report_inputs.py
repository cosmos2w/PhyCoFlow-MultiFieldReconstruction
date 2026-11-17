"""Terminal identity and coverage checks for report input assembly."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('report_inputs', ROOT / 'scripts/analysis/build_car_round2_report_inputs.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
report_spec = importlib.util.spec_from_file_location('report', ROOT / 'scripts/analysis/report_car_round2.py')
report_module = importlib.util.module_from_spec(report_spec)
report_spec.loader.exec_module(report_module)


def report_stub(validate):
    return SimpleNamespace(validate_inputs=validate, cost_sensitivity_lines=report_module.cost_sensitivity_lines)


def fixture(tmp_path):
    def put(name, data):
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data)); return path
    def binding(path):
        return {'path': str(path), 'sha256': mod.sha(path)}
    alloc = put('configs/allocation.json', {'mode': 'uniform'})
    freeze = put('configs/formal_freeze.json', {'schema': 'car_round2_freeze_receipt_v1',
        'manifest': {'status': 'frozen', 'artifacts': {'allocation_disposition': binding(alloc)}}})
    groups = []
    for index, arm in enumerate(mod.ARMS):
        run = tmp_path / 'runs/formal' / arm / 'terminal'; put(run / 'status.json', {'status':'completed','global_step':12000})
        launch = put(f'operations/formal_{index}_launch.json', {'token': f'token-{index}'})
        completion = put(f'operations/formal_{index}_done.json', {'token': f'token-{index}', 'status':'completed', 'global_step':12000, 'run_dir':str(run)})
        child = put(f'operations/formal_{index}_child.json', {'schema':'car_round2_formal_child_v1','arm':arm,'returncode':0,
             'launch_receipt':str(launch),'launch_sha256':mod.sha(launch), 'completion_receipt':str(completion),'completion_sha256':mod.sha(completion)})
        group_launch = put(f'operations/formal_{index}_group_launch.json', {})
        groups.append(put(f'operations/formal_{index}_group.json', {'schema':'car_round2_formal_group_v1','status':'completed',
            'launch_receipt':str(group_launch),'launch_sha256':mod.sha(group_launch),'children':[binding(child)]}))
    train_groups = []
    for index, (arm, epoch) in enumerate((arm, epoch) for arm in mod.ARMS for epoch in mod.EPOCHS):
        children = []
        for i, kind in enumerate(('gpu', 'gpu', 'cpu')):
            children.append(binding(put(f'operations/train_{index}_child_{i}.json', {'schema':'car_round2_assessment_sequence_child_v1','kind':kind,'returncode':0})))
        launch = put(f'operations/train_{index}_launch.json', {'train_batch_binding':{'arm':arm,'checkpoints':{str(epoch):{}}}})
        group = put(f'operations/train_{index}_group.json', {'schema':'car_round2_assessment_sequence_v1','status':'completed','phase':'train-batch',
            'launch_receipt':str(launch),'launch_sha256':mod.sha(launch),'children':children})
        train_groups.append({'arm':arm,'epoch':epoch,'group':str(group),'group_sha256':mod.sha(group)})
        for split, name in [('train','nomination'),('validation','assessment')]:
            put(f'assessment/{split}/{arm}/epoch_{epoch:03d}/{name}.json', {})
    train = put('operations/train_result.json', {'schema':'car_round2_train_checkpoint_successor_result_v1',
        'status':'completed','freeze_sha256':mod.sha(freeze),'groups':train_groups})
    barrier = put('operations/barrier.json', {})
    family_refs = []
    for family in ('ffm','senseiver'):
        launch = put(f'operations/{family}_val_launch.json', {})
        children = [binding(put(f'operations/{family}_val_child_{i}.json', {'schema':'car_round2_assessment_sequence_child_v1','kind':'cpu' if i < 13 else 'gpu','returncode':0})) for i in range(26)]
        val_group = put(f'operations/{family}_val_group.json', {'schema':'car_round2_validation_successor_v11','family':family,
             'status':'completed','launch_receipt':str(launch),'launch_sha256':mod.sha(launch),'children':children})
        family_refs.append({'family':family,**binding(val_group)})
        for name in [f'assessment/train/method_priority_{family}.json',f'assessment/{family}_final_decision_v1.json',
                     f'assessment/{family}_final_costs_v1.json',f'artifacts/selected_banks/{family}_topology_stable_v1/decision.json',
                     f'assessment/source_validation_{family}_v1/receipt.json',f'artifacts/allocation/{family}_residual_allocator_topology_stable_v1/audit.json']:
            put(name, {})
        cost = tmp_path / f'assessment/{family}_final_costs_v1.json'
        put(f'assessment/{family}_cost_sensitivity_v1.json', {'schema':'car_round2_cost_sensitivity_v1',
            'family':family,'conditional':True,'adoption_unchanged':True,
            'source_bindings':{'cost_reconciliation':binding(cost)},'gate_statuses':{},'endpoint_scenarios':[]})
    val = put('operations/validation_result.json', {'schema':'car_round2_validation_successor_result_v1','status':'completed',
        'formal_freeze_sha256':mod.sha(freeze),'barrier':str(barrier),'barrier_sha256':mod.sha(barrier),'families':family_refs})
    put('assessment/fidelity_qualification_v1/ledger.json', {})
    return groups, train, val


def test_terminal_roots_assemble_four_runs_and_all_matched_trajectories(tmp_path):
    groups, train, val = fixture(tmp_path)
    calls = []
    report = report_stub(lambda inputs, campaign: calls.append((inputs, campaign)))
    inputs = mod.build(tmp_path, groups, train, val, report)
    assert set(inputs['runs']) == set(mod.ARMS)
    assert sum(len(epochs) for arm in inputs['trajectories'].values() for epochs in arm.values()) == 48
    assert len(calls) == 1 and inputs['source_bindings']
    missing = Path(inputs['trajectories'][mod.ARMS[0]]['validation']['250'])
    saved = missing.read_text(); missing.unlink()
    with pytest.raises(FileNotFoundError):
        mod.build(tmp_path, groups, train, val, report)
    missing.write_text(saved)
    train_doc = json.loads(train.read_text()); train_doc['groups'].pop(); train.write_text(json.dumps(train_doc))
    with pytest.raises(ValueError, match='24 TRAIN'):
        mod.build(tmp_path, groups, train, val, report)


def test_stale_terminal_evidence_is_rejected(tmp_path):
    groups, train, val = fixture(tmp_path)
    group = json.loads(groups[0].read_text())
    child = Path(group['children'][0]['path']); child.write_text('{}')
    with pytest.raises(ValueError, match='hash changed'):
        mod.build(tmp_path, groups, train, val, report_stub(lambda *_: None))


def test_incomplete_validation_and_wrong_train_checkpoint_identity_are_rejected(tmp_path):
    groups, train, val = fixture(tmp_path)
    report = report_stub(lambda *_: None)
    document = json.loads(val.read_text()); document['families'].pop(); val.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='both VALIDATION'):
        mod.build(tmp_path, groups, train, val, report)
    train_doc = json.loads(train.read_text())
    item = train_doc['groups'][0]
    group = json.loads(Path(item['group']).read_text())
    launch = Path(group['launch_receipt']); data = json.loads(launch.read_text())
    data['train_batch_binding']['arm'] = mod.ARMS[1]; launch.write_text(json.dumps(data))
    group['launch_sha256'] = mod.sha(launch); Path(item['group']).write_text(json.dumps(group))
    item['group_sha256'] = mod.sha(item['group']); train.write_text(json.dumps(train_doc))
    with pytest.raises(ValueError, match='TRAIN checkpoint launch identity'):
        mod.build(tmp_path, groups, train, val, report)


def test_builder_reuses_real_sensitivity_validation_and_rejects_stale_cost(tmp_path):
    groups, train, val = fixture(tmp_path)
    report = report_stub(lambda *_: None)
    result = mod.build(tmp_path, groups, train, val, report)
    assert set(result['cost_sensitivity']) == {'ffm', 'senseiver'}
    (tmp_path/'assessment/ffm_final_costs_v1.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='cost identity differs'):
        mod.build(tmp_path, groups, train, val, report)


@pytest.mark.parametrize('bad_token', ['', 'token-0'])
def test_empty_or_duplicate_formal_tokens_are_rejected(tmp_path, bad_token):
    groups, train, val = fixture(tmp_path)
    group = json.loads(groups[1].read_text()); child_path = Path(group['children'][0]['path'])
    child = json.loads(child_path.read_text())
    for field, digest_field in [('launch_receipt', 'launch_sha256'), ('completion_receipt', 'completion_sha256')]:
        path = Path(child[field]); data = json.loads(path.read_text()); data['token'] = bad_token
        path.write_text(json.dumps(data)); child[digest_field] = mod.sha(path)
    child_path.write_text(json.dumps(child)); group['children'][0]['sha256'] = mod.sha(child_path)
    groups[1].write_text(json.dumps(group))
    with pytest.raises(ValueError, match='completion token'):
        mod.build(tmp_path, groups, train, val, report_stub(lambda *_: None))


@pytest.mark.parametrize('stage', ['train', 'validation'])
def test_repeated_child_receipt_is_rejected(tmp_path, stage):
    groups, train, val = fixture(tmp_path)
    if stage == 'train':
        root_path = train; root = json.loads(train.read_text()); item = root['groups'][0]
        child_group = Path(item['group']); group = json.loads(child_group.read_text())
        group['children'][1] = group['children'][0]; child_group.write_text(json.dumps(group))
        item['group_sha256'] = mod.sha(child_group)
    else:
        root_path = val; root = json.loads(val.read_text()); item = root['families'][0]
        child_group = Path(item['path']); group = json.loads(child_group.read_text())
        group['children'][1] = group['children'][0]; child_group.write_text(json.dumps(group))
        item['sha256'] = mod.sha(child_group)
    root_path.write_text(json.dumps(root))
    with pytest.raises(ValueError, match='receipts duplicated|unique child receipts'):
        mod.build(tmp_path, groups, train, val, report_stub(lambda *_: None))


@pytest.mark.parametrize('invalid', ['schema', 'kind'])
def test_validation_child_requires_expected_schema_and_kinds(tmp_path, invalid):
    groups, train, val = fixture(tmp_path)
    root = json.loads(val.read_text()); item = root['families'][0]
    group_path = Path(item['path']); group = json.loads(group_path.read_text())
    child_path = Path(group['children'][0]['path']); child = json.loads(child_path.read_text())
    child[invalid] = 'incorrect'; child_path.write_text(json.dumps(child))
    group['children'][0]['sha256'] = mod.sha(child_path); group_path.write_text(json.dumps(group))
    item['sha256'] = mod.sha(group_path); val.write_text(json.dumps(root))
    with pytest.raises(ValueError, match='VALIDATION child'):
        mod.build(tmp_path, groups, train, val, report_stub(lambda *_: None))
