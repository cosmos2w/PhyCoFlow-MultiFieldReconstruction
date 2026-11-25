"""The resumed report permits measured judgment while retaining field safeguards."""
from copy import deepcopy
import hashlib

import pytest

from scripts.analysis.report_car_round6 import validate_model_decision


def test_authorized_margin_miss_is_distinct_from_physical_failure(tmp_path):
    protocol = tmp_path / 'resumption.md'
    protocol.write_text('User-authorized resumption: judge measured progress.\n')
    flags = {key: True for key in ('primary_field_pass', 'sensor_pass',
                                   'primary_pass', 'legacy_member_pass')}
    candidate = {'fidelity': flags, 'replacement_eligible': False}
    decision = {'schema': 'car_round6_terminal_disposition_v1', 'status': 'completed',
        'test_locked': True, 'recommendation': 'measured_repair',
        'scientific_progress': 'Field retention and held-out Shape improved; regional gain is below the reference margin.',
        'metrics': {'physical_safeguards': flags.copy()},
        'evidence': {'resumption_protocol': {'path': str(protocol),
            'sha256': hashlib.sha256(protocol.read_bytes()).hexdigest()}}}
    assert validate_model_decision(decision, candidate, 'measured_repair') is True
    original = dict(decision, schema='original_decision')
    with pytest.raises(ValueError, match='original guards'):
        validate_model_decision(original, candidate, 'measured_repair')
    failed = deepcopy(candidate)
    failed['fidelity']['legacy_member_pass'] = False
    rejected = deepcopy(decision)
    rejected['metrics']['physical_safeguards'] = failed['fidelity'].copy()
    with pytest.raises(ValueError, match='field, sensor and member'):
        validate_model_decision(rejected, failed, 'measured_repair')
    rejected['recommendation'] = 'retain SOURCE/C8'
    assert validate_model_decision(rejected, failed, 'measured_repair') is True
    protocol.write_text('Different protocol\n')
    with pytest.raises(ValueError, match='evidence changed'):
        validate_model_decision(decision, candidate, 'measured_repair')


def test_completed_learning_binds_inputs_and_labels_distinct_measurements(tmp_path):
    import json
    from scripts.analysis.report_car_round6 import completed_learning_section
    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    history = write('history.json', {'data': 1})
    journal = write('journal.json', {'data': 2})
    identity = {'field_order': ['CH4', 'CO'], 'sample_ids': ['a']}
    current = write('current.json', dict(identity, epoch=500))
    old = write('old.json', dict(identity, epoch=100))
    historical = write('historical.json', dict(identity, epoch=500))
    windows = {k: {'action_count': 300, 'constraint_actual_negative_count': 300,
                   'positive_pooled_excess_J': j, 'mean_signed_e': {'CH4': -.1, 'CO': .2}}
               for k, j in [('epochs_76_100', 1.1), ('epochs_451_475', 0.), ('epochs_476_500', .03)]}
    training = write('training.json', {'schema': 'car_round6_completed500_training_readout_v1',
        'epoch_count': 500, 'accepted_action_count': 6000, 'field_order': identity['field_order'],
        'history': history, 'pi_journal': journal, 'windows': windows,
        'sampled_family_change_early100_to_late500': {'state': -.1, 'scale': -.06, 'shape': -.04}})
    core = {k: {'mean': .162561 if k == 'scale' else 0., 'interval_95': [.06396, .26323]}
            for k in ('state', 'scale', 'shape')}
    paired = write('paired.json', {'schema': 'car_round6_cached_paired_endpoint_comparisons_v1',
        'epoch500': current, 'epoch100': old, 'historical_epoch500': historical,
        'matched_identity_keys': ['field_order', 'sample_ids'], 'comparisons': {
            'epoch500_minus_epoch100': {'core': core}, 'epoch500_minus_historical_epoch500': {'core': core}}})
    assessment = dict(identity, epoch=500, review_sha256=current['sha256'])
    result = '\n'.join(completed_learning_section(training, paired, assessment, history, journal, {'assessment': old}))
    assert '+0.16256' in result and '[+0.06396, +0.26323]' in result
    assert '600 late TRAIN actions' in result and 'same reused development trajectory' in result
    assert 'does not isolate a PI effect' in result and '-6.000%' in result
    with pytest.raises(ValueError, match='history, journal'):
        completed_learning_section(training, paired, assessment, history, history, {'assessment': old})
    with pytest.raises(ValueError, match='endpoint, history'):
        completed_learning_section(training, paired, dict(assessment, review_sha256='wrong'), history, journal, {'assessment': old})


def test_report_evidence_binds_outputs_and_resource_timing(tmp_path):
    import json
    import resource
    import time
    from scripts.analysis.report_car_round6 import finish_report_evidence
    evidence=tmp_path/'assessment.json';evidence.write_text('{"complete": true}')
    ref={'path':str(evidence),'sha256':hashlib.sha256(evidence.read_bytes()).hexdigest()}
    inputs={'assessment':ref};manifest=tmp_path/'inputs.json';manifest.write_text(json.dumps(inputs))
    output=tmp_path/'report';output.mkdir();figures=output/'figures';figures.mkdir()
    for p in (output/'report.md',output/'report.pdf',figures/'example.png',figures/'example.pdf'):p.write_bytes(p.name.encode())
    receipt=finish_report_evidence(manifest,inputs,output,{'example':'caption'},1,time.time(),resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
    index=json.loads((output/'evidence_index.json').read_text())
    assert receipt['evidence_index_sha256']==hashlib.sha256((output/'evidence_index.json').read_bytes()).hexdigest()
    assert index['bindings']['assessment']==ref and index['input_manifest']['sha256']==hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert index['artifacts']['figures']['example']['pdf_sha256']==hashlib.sha256((figures/'example.pdf').read_bytes()).hexdigest()
    assert receipt['cpu_user_seconds']>=0 and receipt['cpu_system_seconds']>=0 and 'completed child processes' in receipt['cost_scope']
    evidence.write_text('changed')
    with pytest.raises(ValueError,match='evidence changed'):
        finish_report_evidence(manifest,inputs,output,{},1,time.time(),resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
