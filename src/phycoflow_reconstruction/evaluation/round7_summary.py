"""Pure saved-evidence reductions for matched Round 7 reports; no inference."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean

COUNTS = {'train': 800, 'validation': 100, 'test': 100}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _means(values):
    """Reduce aligned numeric diagnostic trees, preserving their array shapes."""
    first = values[0]
    if all(_finite(v) for v in values):
        return mean(values)
    if isinstance(first, dict) and all(isinstance(v, dict) for v in values):
        keys = set.intersection(*(set(v) for v in values))
        result = {key: _means([v[key] for v in values]) for key in sorted(keys)}
        return {k: v for k, v in result.items() if v is not None}
    if isinstance(first, list) and all(isinstance(v, list) and len(v) == len(first) for v in values):
        reduced = [_means([v[i] for v in values]) for i in range(len(first))]
        return reduced if all(v is not None for v in reduced) else None
    return None


def summarize_reductions(path, field_names, *, expected_count=None):
    """Summarize one complete frozen Q4096 panel, retaining condition statistics."""
    payload = json.loads(Path(path).read_text())
    split, rows = payload['split'], payload['rows']
    count = COUNTS[split] if expected_count is None else int(expected_count)
    if (len(rows) != count or len({r['sample_id'] for r in rows}) != count
            or not field_names or len(set(field_names)) != len(field_names)):
        raise ValueError('incomplete or duplicate matched panel')
    if not payload.get('observer_identities'):
        raise ValueError('explicit common observer identities required')
    conditions, errors, counts, powers = {}, [], [], []
    for row in rows:
        error, mass, power = (row[key] for key in
                              ('field_error_sums', 'field_error_counts', 'reference_power_sums'))
        if (any(len(v) != len(field_names) for v in (error, mass, power))
                or not all(_finite(x) and x >= 0 for x in (*error, *power))
                or not all(_finite(x) and x > 0 for x in mass)):
            raise ValueError('invalid field error, count or reference power')
        values = {'state': row['losses']['global_distribution.total'],
                  'scale': row['losses']['multiscale_spectrum.total']}
        for name, observer in row['shape_observers'].items():
            values['shape.' + name] = observer['losses']['topology.total']
        values.update({'field.' + name + '.mse': error[i] / mass[i]
                       for i, name in enumerate(field_names)})
        if not all(_finite(v) for v in values.values()):
            raise ValueError('nonfinite common score')
        conditions[row['sample_id']] = {
            'values': values, 'counts': mass, 'reference_power_sums': power,
            'query_selection': row['query_selection'],
        }
        errors.append(error); counts.append(mass); powers.append(power)
    fields = {}
    for i, name in enumerate(field_names):
        e, n, p = (sum(v[i] for v in collection) for collection in (errors, counts, powers))
        fields[name] = {'mse_model_units': e / n, 'rmse_model_units': math.sqrt(e / n),
                        'relative_l2': math.sqrt(e / p) if p > 0 else None,
                        'reference_power_model_units': p / n, 'point_visits': n}
    return {'schema': 'round7_saved_summary_v1', 'split': split, 'conditions': count,
            'manifest_sha256': payload['manifest_sha256'],
            'observer_identities': payload['observer_identities'],
            'weight_sha256': payload['weight_sha256'], 'field_names': list(field_names),
            'scores': _means([r['values'] for r in conditions.values()]), 'fields': fields,
            'loss_means': _means([r['losses'] for r in rows]),
            'family_diagnostics': _means([r.get('family_diagnostics', {}) for r in rows]),
            'component_diagnostics': _means([r.get('component_diagnostics', {}) for r in rows]),
            'shape_observers': _means([r['shape_observers'] for r in rows]),
            'j2_unweighted_model_units': _means([r['j2_unweighted_model_units'] for r in rows]),
            'j2_physical_measure_model_units': _means([r['j2_physical_measure_model_units'] for r in rows]),
            'per_condition': conditions, 'source_path': str(Path(path).resolve())}


def _quantile(sorted_values, fraction):
    position = (len(sorted_values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return sorted_values[lo] + (position - lo) * (sorted_values[hi] - sorted_values[lo])


def compare_summaries(reference, candidate, *, positive_scales=None):
    """Candidate-minus-reference paired changes on the identical frozen panel."""
    for key in ('schema', 'split', 'manifest_sha256', 'observer_identities', 'field_names'):
        if reference[key] != candidate[key]:
            raise ValueError('incompatible matched comparison: ' + key)
    left, right = reference['per_condition'], candidate['per_condition']
    if set(left) != set(right):
        raise ValueError('condition IDs differ')
    for sample_id in left:
        for key in ('counts', 'reference_power_sums', 'query_selection'):
            if left[sample_id][key] != right[sample_id][key]:
                raise ValueError('condition support, reference or query pack differs')
        if left[sample_id]['values'].keys() != right[sample_id]['values'].keys():
            raise ValueError('condition metric coverage differs')
    positive_scales = positive_scales or {}
    if not all(_finite(v) and v > 0 for v in positive_scales.values()):
        raise ValueError('report scales must be positive finite frozen SOURCE values')
    changes = {}
    for key in reference['scores']:
        deltas = sorted(right[s]['values'][key] - left[s]['values'][key] for s in left)
        baseline, value = reference['scores'][key], candidate['scores'][key]
        delta = value - baseline
        changes[key] = {'reference': baseline, 'candidate': value, 'raw_change': delta,
                        'fractional_change': delta / baseline if baseline > 0 else None,
                        'source_units_change': delta / positive_scales[key] if key in positive_scales else None,
                        'paired_mean_change': mean(deltas), 'paired_median_change': _quantile(deltas, .5),
                        'paired_p05_change': _quantile(deltas, .05), 'paired_p95_change': _quantile(deltas, .95),
                        'conditions_improved': sum(v < 0 for v in deltas), 'conditions': len(deltas)}
    return {'schema': 'round7_matched_saved_changes_v1', 'split': reference['split'],
            'reference_weight_sha256': reference['weight_sha256'],
            'candidate_weight_sha256': candidate['weight_sha256'], 'changes': changes}
