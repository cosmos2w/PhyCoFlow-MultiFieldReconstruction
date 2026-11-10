import copy
import json
import math
import pytest
from phycoflow_reconstruction.evaluation.round7_summary import summarize_reductions, compare_summaries


def fixture(tmp_path):
    rows = []
    for sample_id, value in [('a', 1.), ('b', 3.)]:
        rows.append({'sample_id': sample_id, 'losses': {'global_distribution.total': value,
            'multiscale_spectrum.total': value * 2}, 'field_error_sums': [value * 4],
            'field_error_counts': [4], 'reference_power_sums': [16],
            'shape_observers': {'fair': {'losses': {'topology.total': -.1 * value},
                                'diagnostics': {'attraction_per_sample': [value], 'separation_per_sample': [1.1 * value]}}},
            'query_selection': {'frozen': sample_id}, 'family_diagnostics': {}, 'component_diagnostics': {},
            'j2_unweighted_model_units': {'variance_proxy': [value]},
            'j2_physical_measure_model_units': {'variance_proxy': [value * 2]}})
    payload = {'split': 'validation', 'rows': rows, 'manifest_sha256': 'a'*64,
               'observer_identities': {'fair': 'b'*64}, 'weight_sha256': 'c'*64}
    path = tmp_path / 'reductions.json'; path.write_text(json.dumps(payload))
    return path


def test_pooled_errors_signed_score_and_paired_tails(tmp_path):
    summary = summarize_reductions(fixture(tmp_path), ['T'], expected_count=2)
    assert summary['fields']['T']['mse_model_units'] == 2
    assert summary['fields']['T']['rmse_model_units'] == math.sqrt(2)
    assert summary['fields']['T']['relative_l2'] == math.sqrt(.5)
    assert summary['shape_observers']['fair']['diagnostics']['attraction_per_sample'] == [2]
    changed = copy.deepcopy(summary)
    for condition in changed['per_condition'].values():
        condition['values']['scale'] -= .5
    changed['scores']['scale'] -= .5
    result = compare_summaries(summary, changed, positive_scales={'scale': .25})
    assert result['changes']['scale']['source_units_change'] == -2
    assert result['changes']['scale']['paired_p95_change'] == -.5
    assert result['changes']['shape.fair']['fractional_change'] is None


def test_comparison_rejects_changed_observer_support_or_ids(tmp_path):
    original = summarize_reductions(fixture(tmp_path), ['T'], expected_count=2)
    for key, value in [('observer_identities', {'fair': 'd'*64}), ('manifest_sha256', 'd'*64)]:
        changed = copy.deepcopy(original); changed[key] = value
        with pytest.raises(ValueError): compare_summaries(original, changed)
    changed = copy.deepcopy(original)
    changed['per_condition']['a']['reference_power_sums'] = [17]
    with pytest.raises(ValueError): compare_summaries(original, changed)
    with pytest.raises(ValueError): summarize_reductions(fixture(tmp_path), ['T'])
