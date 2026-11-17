"""Build final Round 2 report inputs from explicit completed terminal receipts."""
import time

START = time.time()

import argparse
import hashlib
import importlib.util
import json
import traceback
from pathlib import Path

ARMS = ('FFM_Selected_Bank_Uniform_FullModel', 'FFM_Selected_Bank_Residual_Repair',
        'Senseiver_Selected_Bank_Uniform_FullModel', 'Senseiver_Selected_Bank_Residual_Repair')
EPOCHS = (0, 100, 250, 500, 750, 1000)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def build(campaign, formal_groups, train_result, validation_result, report):
    campaign = Path(campaign).resolve(strict=True)
    bindings = {}
    def path(value):
        candidate = Path(value)
        candidate = (candidate if candidate.is_absolute() or candidate.exists() else campaign / candidate).resolve(strict=True)
        if campaign not in candidate.parents:
            raise ValueError(f'evidence outside campaign: {candidate}')
        return candidate
    def read(value, expected=None):
        p = path(value); digest = sha(p)
        if expected is not None and digest != expected:
            raise ValueError(f'evidence hash changed: {p}')
        bindings[str(p)] = digest
        return json.loads(p.read_text())
    def ref(record, key, hash_key):
        return read(record[key], record[hash_key])

    freeze_path = path('configs/formal_freeze.json')
    freeze = read(freeze_path)
    if freeze.get('schema') != 'car_round2_freeze_receipt_v1' or freeze.get('manifest', {}).get('status') != 'frozen':
        raise ValueError('completed formal freeze required')
    runs, tokens = {}, set()
    for group_path in formal_groups:
        group = read(group_path)
        if group.get('schema') != 'car_round2_formal_group_v1' or group.get('status') != 'completed' or group.get('failure'):
            raise ValueError('completed formal group required')
        ref(group, 'launch_receipt', 'launch_sha256')
        for item in group.get('children', []):
            child = ref(item, 'path', 'sha256')
            arm = child.get('arm')
            if child.get('schema') != 'car_round2_formal_child_v1' or child.get('returncode') != 0 or arm not in ARMS or arm in runs:
                raise ValueError('formal child missing, failed, or duplicated')
            launch = ref(child, 'launch_receipt', 'launch_sha256')
            completion = ref(child, 'completion_receipt', 'completion_sha256')
            token = launch.get('token')
            if (not isinstance(token, str) or not token.strip() or token in tokens
                    or completion.get('token') != token or completion.get('status') != 'completed'
                    or completion.get('global_step') != 12000):
                raise ValueError('formal completion token or horizon differs')
            tokens.add(token)
            run = path(completion['run_dir'])
            if campaign / 'runs/formal' / arm not in run.parents:
                raise ValueError('formal run identity differs from arm')
            status = read(run / 'status.json')
            if status.get('status') != 'completed' or status.get('global_step') != 12000:
                raise ValueError('formal run not terminal at 1000 epochs')
            runs[arm] = str(run)
    if set(runs) != set(ARMS):
        raise ValueError('exactly four terminal formal arms required')

    train = read(train_result)
    if (train.get('schema') != 'car_round2_train_checkpoint_successor_result_v1' or train.get('status') != 'completed'
            or train.get('freeze_sha256') != sha(freeze_path)):
        raise ValueError('TRAIN terminal result or freeze differs')
    expected = {(arm, epoch) for arm in ARMS for epoch in EPOCHS}
    seen = set()
    for item in train.get('groups', []):
        key = (item.get('arm'), item.get('epoch'))
        if key not in expected or key in seen:
            raise ValueError('TRAIN checkpoint identity duplicated or unexpected')
        group = ref(item, 'group', 'group_sha256')
        if (group.get('schema') != 'car_round2_assessment_sequence_v1' or group.get('status') != 'completed'
                or group.get('phase') != 'train-batch' or len(group.get('children', [])) != 3):
            raise ValueError('incomplete TRAIN checkpoint group')
        launch = ref(group, 'launch_receipt', 'launch_sha256')
        binding = launch.get('train_batch_binding', {})
        if binding.get('arm') != key[0] or set(binding.get('checkpoints', {})) != {str(key[1])}:
            raise ValueError('TRAIN checkpoint launch identity differs')
        if len({str(path(child['path'])) for child in group['children']}) != 3:
            raise ValueError('TRAIN checkpoint child receipts duplicated')
        children = [ref(child, 'path', 'sha256') for child in group['children']]
        if any(child.get('schema') != 'car_round2_assessment_sequence_child_v1' or child.get('returncode') != 0 for child in children) or sorted(child.get('kind') for child in children) != ['cpu', 'gpu', 'gpu']:
            raise ValueError('TRAIN checkpoint child failed')
        seen.add(key)
    if seen != expected:
        raise ValueError('all 24 TRAIN checkpoint groups required')

    val = read(validation_result)
    if (val.get('schema') != 'car_round2_validation_successor_result_v1' or val.get('status') != 'completed'
            or val.get('formal_freeze_sha256') != sha(freeze_path)):
        raise ValueError('VALIDATION terminal result or freeze differs')
    ref(val, 'barrier', 'barrier_sha256')
    seen_families = set()
    for entry in val.get('families', []):
        family = entry.get('family')
        if family not in ('ffm', 'senseiver') or family in seen_families:
            raise ValueError('VALIDATION family duplicated or unknown')
        group = ref(entry, 'path', 'sha256')
        if group.get('schema') != 'car_round2_validation_successor_v11' or group.get('status') != 'completed' or group.get('family') != family or group.get('failure'):
            raise ValueError('VALIDATION family incomplete')
        ref(group, 'launch_receipt', 'launch_sha256')
        child_refs = group.get('children', [])
        if len(child_refs) != 26 or len({str(path(child['path'])) for child in child_refs}) != 26:
            raise ValueError('VALIDATION family requires 26 unique child receipts')
        kinds = []
        for child_ref in child_refs:
            child = ref(child_ref, 'path', 'sha256')
            if child.get('schema') != 'car_round2_assessment_sequence_child_v1' or child.get('returncode') != 0:
                raise ValueError('VALIDATION child failed')
            kinds.append(child.get('kind'))
        if sorted(kinds) != ['cpu'] * 13 + ['gpu'] * 13:
            raise ValueError('VALIDATION child kinds differ')
        seen_families.add(family)
    if seen_families != {'ffm', 'senseiver'}:
        raise ValueError('both VALIDATION family completions required')

    def canonical(value):
        p = path(value); bindings[str(p)] = sha(p); return str(p)
    trajectories = {arm: {split: {str(epoch): canonical(f'assessment/{split}/{arm}/epoch_{epoch:03d}/' +
                    ('nomination.json' if split == 'train' else 'assessment.json')) for epoch in EPOCHS}
                    for split in ('train', 'validation')} for arm in ARMS}
    for family in ('ffm', 'senseiver'):
        canonical(f'assessment/train/method_priority_{family}.json')
    allocation = freeze['manifest']['artifacts']['allocation_disposition']
    allocation_doc = ref(allocation, 'path', 'sha256')
    profiles = allocation_doc.get('profiles', {})
    if set(profiles) != {'ffm', 'senseiver'}:
        raise ValueError('final allocation disposition must bind both actual source audits')
    for profile in profiles.values():
        ref(profile, 'path', 'sha256')
    inputs = {'schema': 'car_round2_report_inputs_v1', 'runs': runs, 'trajectories': trajectories,
              'decisions': {f: canonical(f'assessment/{f}_final_decision_v1.json') for f in ('ffm', 'senseiver')},
              'costs': {f: canonical(f'assessment/{f}_final_costs_v1.json') for f in ('ffm', 'senseiver')},
              'cost_sensitivity': {f: canonical(f'assessment/{f}_cost_sensitivity_v1.json') for f in ('ffm', 'senseiver')},
              'bank_decisions': {f: canonical(f'artifacts/selected_banks/{f}_topology_stable_v1/decision.json') for f in ('ffm', 'senseiver')},
              'source_banks': {f: str(path(f'assessment/source_validation_{f}_v1')) for f in ('ffm', 'senseiver')},
              'allocation_disposition': str(path(allocation['path'])),
              'allocator_audits': {f: [str(path(profiles[f]['path']))] for f in ('ffm', 'senseiver')},
              'definition_qualification_ledger': canonical('assessment/fidelity_qualification_v1/ledger.json')}
    for bank in inputs['source_banks'].values():
        canonical(Path(bank) / 'receipt.json')
    report.validate_inputs(inputs, campaign)
    report.cost_sensitivity_lines(inputs, campaign, None)
    inputs['source_bindings'] = [{'path': p, 'sha256': h} for p, h in sorted(bindings.items())]
    return inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--formal-group', type=Path, action='append', required=True)
    parser.add_argument('--train-result', type=Path, required=True)
    parser.add_argument('--validation-result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True); output = args.output.resolve()
    if campaign not in output.parents or output.exists():
        raise ValueError('new campaign-owned output required')
    receipt = Path(str(output) + '.receipt.json')
    if receipt.exists():
        raise ValueError('new output receipt required')
    result = {'schema': 'car_round2_independent_cpu_job_v1', 'status': 'failed', 'start_unix': START,
              'independent_cpu': True, 'scope': 'Terminal receipt validation and final report input assembly; no inference'}
    try:
        module_path = Path(__file__).with_name('report_car_round2.py')
        spec = importlib.util.spec_from_file_location('car_round2_report', module_path)
        report = importlib.util.module_from_spec(spec); spec.loader.exec_module(report)
        inputs = build(campaign, args.formal_group, args.train_result, args.validation_result, report)
        with output.open('x') as f:
            json.dump(inputs, f, indent=2); f.write('\n')
        result.update(status='completed', output=str(output), output_sha256=sha(output),
                      source_count=len(inputs['source_bindings']), report_source_sha256=sha(module_path),
                      builder_sha256=sha(Path(__file__)))
    except BaseException:
        result['failure'] = traceback.format_exc()
        raise
    finally:
        end = time.time(); result.update(end_unix=end, wall_seconds=end-START)
        with receipt.open('x') as f:
            json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
