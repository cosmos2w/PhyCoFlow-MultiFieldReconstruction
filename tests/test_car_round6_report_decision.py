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
