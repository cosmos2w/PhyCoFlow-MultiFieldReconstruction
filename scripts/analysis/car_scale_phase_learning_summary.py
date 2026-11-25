#!/usr/bin/env python3
"""Summarize completed phase-100 online TRAIN records; no model evaluation."""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

FAMILIES = {'state': 'global_distribution', 'scale': 'multiscale_spectrum', 'shape': 'topology'}


def file_ref(path):
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return {'path': str(path), 'sha256': h.hexdigest()}


def verify(ref):
    actual = file_ref(ref['path'])
    if actual['sha256'] != ref['sha256']:
        raise ValueError(f"Changed bound input: {ref['path']}")
    return actual


def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_definitions(definitions):
    columns = {c['name']: c for c in definitions['columns']}
    expected = {'epoch': 'number', 'step': 'number', 'epoch_complete': 'bool'}
    expected.update({f'coherence_family/{x}/raw': 'number' for x in FAMILIES.values()})
    for name, kind in expected.items():
        column = columns.get(name, {})
        if column.get('type') != kind or not column.get('units') or not column.get('meaning'):
            raise ValueError(f'Missing or invalid typed recording definition: {name}')


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected finite numerical record')
    return float(value)


def stats(values):
    values = [number(x) for x in values]
    return {'mean': sum(values) / len(values), 'negative': sum(x < 0 for x in values),
            'zero': sum(x == 0 for x in values), 'positive': sum(x > 0 for x in values)}


def summarize(history, journal, reference, origin):
    fields = reference['field_names']
    scales, allowances = reference['risk_scales'], reference['absolute_allowance']
    if len(fields) != 5 or len(set(fields)) != 5 or len(scales) != 5 or len(allowances) != 5:
        raise ValueError('Invalid reference fields')
    if any(number(x) <= 0 for x in scales + allowances):
        raise ValueError('Reference scales and allowances must be positive')
    if len(history) < 100 or [r['epoch'] for r in history[:100]] != list(range(1, 101)):
        raise ValueError('Missing or disordered phase history')
    if any(r.get('epoch_complete') is not True or r['step'] != origin + 12 * (i+1) for i,r in enumerate(history[:100])):
        raise ValueError('Phase history has incomplete epochs or wrong action clock')
    if len(journal) < 1200 or [r['step'] for r in journal[:1200]] != list(range(origin+1,origin+1201)):
        raise ValueError('Missing or disordered phase action journal')
    if any(r['epoch'] != i//12+1 for i,r in enumerate(journal[:1200])):
        raise ValueError('Journal epoch and action clock differ')
    result = {}
    for start,end in ((51,75),(76,100)):
        h = history[start-1:end]; actions = journal[(start-1)*12:end*12]
        signed = {f: sum(number(r['field_violation'][f]) for r in actions)/len(actions) for f in fields}
        excess = {f: max(signed[f], 0.0)*number(scales[i])/number(allowances[i]) for i,f in enumerate(fields)}
        family_mean = {label: sum(number(r[f'coherence_family/{key}/raw']) for r in h)/len(h) for label,key in FAMILIES.items()}
        result[f'epochs_{start}_{end}'] = {
            'epoch_count':len(h), 'action_count':len(actions), 'mean_signed_e':signed,
            'positive_pooled_excess_by_field':excess, 'positive_pooled_excess_J':math.sqrt(sum(v*v for v in excess.values())),
            'sampled_family_mean':family_mean,
            'mean_multipliers_before':{f:sum(number(r['multipliers_before'][i]) for r in actions)/len(actions) for i,f in enumerate(fields)},
            'constraint_actual_displacement_product':stats(r['action_constraint_product'] for r in actions),
            'family_actual_displacement_products':{label:stats(r['action_family_products']['actual_displacement'][key] for r in actions) for label,key in FAMILIES.items()}}
    result['family_mean_change_76_100_minus_51_75'] = {f:result['epochs_76_100']['sampled_family_mean'][f]-result['epochs_51_75']['sampled_family_mean'][f] for f in FAMILIES}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-result', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    output = a.output.resolve()
    snapshot = output.with_name(output.stem + '_recorded_inputs')
    if output.exists() or snapshot.exists():
        raise ValueError('Output and its recorded-input directory must both be new')
    receipt=json.loads(a.training_result.read_text())
    if (receipt.get('schema') != 'car_scale_followup_training_v1' or receipt.get('status') != 'completed_phase'
        or receipt.get('phase_origin_step') != 6000 or receipt.get('phase_target_epoch') != 100
        or receipt.get('global_step') != 7200 or receipt.get('original_parent_unchanged') is not True
        or receipt.get('test_locked') is not True):
        raise ValueError('Expected completed bound phase-100 fork result')
    if receipt['fork_origin']['absolute_step'] != 6000 or receipt['fork_origin']['parent_checkpoint_sha256'] != receipt['parent_checkpoint']['sha256']:
        raise ValueError('Fork parent or clock differs')
    verify(receipt['parent_checkpoint']);verify(receipt['parent_snapshot']);verify(receipt['selected_config'])
    evidence=receipt['complete_epoch_evidence']; run=Path(receipt['run']).resolve(strict=True)
    refs={name:verify(evidence[name]) for name in ('history','pi_journal','recording_definitions','checkpoint')}
    for name in ('history','pi_journal','recording_definitions'):
        if Path(refs[name]['path']).parent != run/'logs':raise ValueError('Log outside bound fork run')
    if Path(refs['checkpoint']['path']) != run/'checkpoints/epoch_100.pt':raise ValueError('Wrong phase checkpoint')
    source=verify(receipt['source_reference']); reference=json.loads(Path(source['path']).read_text())['reference']
    output.parent.mkdir(parents=True,exist_ok=True)
    snapshot.mkdir()
    copied = {}
    for name in ('history', 'pi_journal', 'recording_definitions'):
        destination = snapshot / Path(refs[name]['path']).name
        shutil.copyfile(refs[name]['path'], destination)
        copied[name] = file_ref(destination)
        if copied[name]['sha256'] != refs[name]['sha256']:
            raise ValueError(f'Recorded input changed while copying: {name}')
    validate_definitions(json.loads(Path(copied['recording_definitions']['path']).read_text()))
    result=summarize(rows(copied['history']['path']),rows(copied['pi_journal']['path']),reference,6000)
    document={'schema':'car_scale_phase100_learning_summary_v1','training_result':file_ref(a.training_result),
        'analysis_code':file_ref(Path(__file__)),
        'inputs':{**copied,'checkpoint':refs['checkpoint']},'original_recorded_inputs':{k:refs[k] for k in copied},
        'source_reference':source,'phase_origin_step':6000,'field_order':reference['field_names'],
        'definition':'J = norm((risk_scale / absolute_allowance) * max(mean signed e, 0)); clip after pooling.',
        'interpretation':'Online changing-model TRAIN windows, not fixed-model TRAIN-panel evaluation. Field quadratures differ between control and candidate; raw online e is not a matched native-field comparison. Negative actual displacement products indicate first-order local decrease. Sampled family losses differ from native held-out metrics.',
        'windows':result}
    with output.open('x') as f:json.dump(document,f,indent=2,allow_nan=False);f.write('\n')

if __name__=='__main__':main()
