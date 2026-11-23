#!/usr/bin/env python3
"""Decide the Round 5 query law from completed paired replay receipts.

The optional action/cost JSON must have schema car_round5_estimator_action_cost_v1,
status completed, matched_inputs true, source_panel_sha256, units_sha256,
checkpoint_sha256, device, and for each of shared/grouped: device, proposals,
accepted, descending, median_same_query_decrease, median_independent_query_decrease,
full_attempted_epoch_seconds. All counts and medians must describe the same
paired panel, seeds, update rule, learning rates and numerical tolerances. The
receipt must identify its action and complete-epoch measurement artifacts with
action_artifact and timing_artifact objects containing path and sha256. Paths
are relative to that receipt or absolute. Absent evidence retains
shared queries. Mean agreement uses a declared two-standard-error paired-vector
rule; it is a finite-replay decision convention, not a confidence guarantee.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return None if path is None else json.loads(path.read_text())


def load_action_cost(path):
    data = load(path)
    if not isinstance(data, dict):
        return data
    verified = True
    for key in ('action_artifact', 'timing_artifact'):
        ref = data.get(key)
        if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'} or not isinstance(ref['path'], str):
            verified = False
            continue
        target = Path(ref['path'])
        if not target.is_absolute():
            target = path.parent / target
        if not target.is_file() or sha(target) != ref['sha256']:
            verified = False
    data['_artifacts_verified'] = verified
    return data


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def gradient_evidence(data, n):
    gram = data['gradient_gram']
    if len(gram) != 2*n or any(len(row) != 2*n for row in gram):
        raise ValueError('gradient Gram shape differs from paired seeds')
    if any(not finite(v) for row in gram for v in row):
        raise ValueError('nonfinite gradient Gram')
    scale = max(1., max(abs(v) for row in gram for v in row))
    if any(abs(gram[i][j] - gram[j][i]) > 1e-8*scale for i in range(2*n) for j in range(2*n)):
        raise ValueError('asymmetric gradient Gram')
    shared, grouped = list(range(0, 2*n, 2)), list(range(1, 2*n, 2))
    def mean_dot(a, b):
        return sum(gram[i][j] for i in a for j in b) / (n*n)
    ss, gg, sg = mean_dot(shared, shared), mean_dot(grouped, grouped), mean_dot(shared, grouped)
    if min(ss, gg) < -1e-8*scale:
        raise ValueError('negative mean gradient squared norm')
    ss, gg = max(0., ss), max(0., gg)
    var_s = sum(gram[i][i] for i in shared)/n - ss
    var_g = sum(gram[i][i] for i in grouped)/n - gg
    diff_sq = ss + gg - 2*sg
    if min(var_s, var_g, diff_sq) < -1e-8*scale:
        raise ValueError('invalid negative centered norm')
    var_s, var_g, diff_sq = max(0., var_s), max(0., var_g), max(0., diff_sq)
    diff_sum = sum(gram[g][g] + gram[s][s] - 2*gram[g][s] for s, g in zip(shared, grouped))
    noise = (diff_sum - n*diff_sq) / (n*(n-1))
    if noise < -1e-8*scale:
        raise ValueError('invalid paired gradient variance')
    mean_difference, standard_error = math.sqrt(diff_sq), math.sqrt(max(0., noise))
    return {'shared_centered_gradient_sd':math.sqrt(var_s), 'grouped_centered_gradient_sd':math.sqrt(var_g),
        'shared_centered_gradient_variance':var_s, 'grouped_centered_gradient_variance':var_g,
        'centered_variance_ratio':var_g/var_s if var_s > 0 else None,
        'centered_sd_ratio': math.sqrt(var_g/var_s) if var_s > 0 else None,
        'mean_difference_norm':mean_difference, 'paired_mean_standard_error_norm':standard_error,
        'mean_direction_cosine': max(-1., min(1., sg / math.sqrt(ss*gg))) if ss > 0 and gg > 0 else None,
        'agreement_rule':'mean difference norm <= 2 * paired standard error norm',
        'mean_agreement':mean_difference <= 2*standard_error + 1e-12*math.sqrt(scale)}


def scalar_evidence(scores, n):
    if len(scores) != 2*n or not all(finite(v) for v in scores):
        raise ValueError('endpoint score count differs from paired seeds')
    diff = [scores[2*i+1]-scores[2*i] for i in range(n)]
    mean = sum(diff)/n
    se = math.sqrt(sum((v-mean)**2 for v in diff)/(n*(n-1)))
    return {'paired_mean_difference':mean, 'paired_standard_error':se,
        'agreement_rule':'absolute paired mean difference <= 2 * paired standard error',
        'mean_agreement':abs(mean) <= 2*se + 1e-12}


def action_cost_evidence(data, param):
    if not isinstance(data, dict) or data.get('status') != 'completed':
        return {'available':False, 'reason':'completed paired finite-action and attempted-epoch evidence is absent'}
    if (data.get('schema') != 'car_round5_estimator_action_cost_v1' or data.get('matched_inputs') is not True
            or data.get('sample_count') != 800
            or data.get('source_panel_sha256') != param.get('source_panel_sha256')
            or data.get('checkpoint_sha256') != param.get('source_checkpoint_sha256')
            or data.get('units_sha256') != param.get('units_sha256')
            or data.get('_artifacts_verified') is not True):
        return {'available':False, 'reason':'action/cost receipt lacks matched SOURCE identity or measurement artifacts'}
    rows = [data.get(name) for name in ('shared', 'grouped')]
    if any(not isinstance(row, dict) or row.get('device') != data.get('device') for row in rows) or not data.get('device'):
        return {'available':False, 'reason':'matched-device action/cost records required'}
    for row in rows:
        counts = [row.get(k) for k in ('proposals', 'accepted', 'descending')]
        metrics = [row.get(k) for k in ('median_same_query_decrease','median_independent_query_decrease','full_attempted_epoch_seconds')]
        if (any(type(v) is not int for v in counts) or not 0 <= counts[2] <= counts[1] <= counts[0]
                or counts[0] < 1 or counts[1] < 1 or not all(finite(v) for v in metrics) or metrics[2] <= 0):
            return {'available':False, 'reason':'action/cost measures are incomplete or invalid'}
    a, b = rows
    if a['proposals'] != b['proposals']:
        return {'available':False, 'reason':'action proposal counts differ'}
    same = b['median_same_query_decrease'] >= a['median_same_query_decrease']
    independent = b['median_independent_query_decrease'] >= a['median_independent_query_decrease']
    action_pass = (b['accepted']/b['proposals'] >= a['accepted']/a['proposals']
        and b['descending']/b['accepted'] >= a['descending']/a['accepted'] and same and independent)
    cost_ratio = b['full_attempted_epoch_seconds']/a['full_attempted_epoch_seconds']
    return {'available':True, 'finite_action_no_worse':action_pass,
        'acceptance_fraction_shared':a['accepted']/a['proposals'], 'acceptance_fraction_grouped':b['accepted']/b['proposals'],
        'accepted_descent_fraction_shared':a['descending']/a['accepted'], 'accepted_descent_fraction_grouped':b['descending']/b['accepted'],
        'same_query_median_no_worse':same, 'independent_query_median_no_worse':independent,
        'full_attempted_epoch_cost_ratio':cost_ratio, 'cost_gate':cost_ratio <= 1.10,
        'device':data['device'], 'action_artifact':data['action_artifact'], 'timing_artifact':data['timing_artifact']}


def assess(endpoint, parameter, context, action_cost=None):
    result = {'schema':'car_round5_estimator_decision_v1', 'query_law':'shared',
        'context':'unqualified', 'context_speed':'not measured by synthetic parity', 'limitations':[]}
    if (isinstance(context, dict) and context.get('schema') == 'car_round5_context_parity_v1'
            and context.get('status') == 'passed' and str(context.get('device','')).startswith('cuda')
            and context.get('neighbor_backend') == 'keops' and context.get('steps') == 4
            and context.get('checkpointing') is True and context.get('duplicated_sensor_coordinates') is True
            and context.get('observation_mode') == 'endpoint_smooth' and context.get('final_clamp') is True
            and all(context.get(key, {}).get('passed') is True for key in ('endpoint','gradients','parameters_after_action'))):
        result['context'] = 'static_features_synthetic_cuda_parity_passed'
    else:
        result['limitations'].append(
            'CUDA context parity failed; retain the existing context route'
            if isinstance(context, dict) and context.get('status') == 'failed'
            else 'CUDA context parity is missing or incomplete')
    if (not isinstance(endpoint, dict) or endpoint.get('schema') != 'car_round5_endpoint_query_replay_v1'
            or endpoint.get('status') != 'completed' or endpoint.get('sample_count') != 800
            or endpoint.get('completed_conditions') != 800 or endpoint.get('seeds') != 8):
        result['limitations'].append('complete eight-seed cached TRAIN800 endpoint replay is unavailable')
        return result
    if (not isinstance(parameter, dict) or parameter.get('schema') != 'car_round5_parameter_query_replay_v1'
            or parameter.get('status') != 'completed' or parameter.get('sample_count') != 800
            or parameter.get('completed_conditions') != 800 or parameter.get('seeds') != 4):
        result['limitations'].append('complete four-seed TRAIN800 parameter replay is unavailable')
        return result
    if any(endpoint.get(a) != parameter.get(b) or not endpoint.get(a) for a,b in
           (('panel_sha256','source_panel_sha256'), ('units_sha256','units_sha256'), ('config_sha256','config_sha256'))):
        result['limitations'].append('endpoint and parameter replay identities do not match')
        return result
    try:
        result['endpoint_gradient'] = gradient_evidence(endpoint, 8)
        result['endpoint_scores'] = scalar_evidence(endpoint['scores'], 8)
        result['shape_parameter_gradient'] = gradient_evidence(parameter['shape'], 4)
        result['scalar_parameter_gradient'] = gradient_evidence(parameter['direct_scalar'], 4)
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as exc:
        result['limitations'].append('invalid replay evidence: ' + str(exc))
        return result
    action = action_cost_evidence(action_cost, parameter)
    result['action_and_cost'] = action
    ratio = result['shape_parameter_gradient']['centered_variance_ratio']
    checks = {'shape_parameter_variance_reduced_at_least_20_percent':ratio is not None and ratio <= .8,
        'endpoint_score_mean_agreement':result['endpoint_scores']['mean_agreement'],
        'endpoint_gradient_mean_agreement':result['endpoint_gradient']['mean_agreement'],
        'shape_gradient_mean_agreement':result['shape_parameter_gradient']['mean_agreement'],
        'scalar_gradient_mean_agreement':result['scalar_parameter_gradient']['mean_agreement'],
        'finite_action_no_worse':action.get('finite_action_no_worse', False), 'attempted_epoch_overhead_at_most_10_percent':action.get('cost_gate', False)}
    result['gates'] = checks
    if all(checks.values()):
        result['query_law'] = 'condition_groups_8'
    else:
        result['limitations'].append('grouped queries did not pass all stated gates')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', type=Path)
    parser.add_argument('--parameter', type=Path)
    parser.add_argument('--context', type=Path)
    parser.add_argument('--action-cost', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = assess(load(args.endpoint), load(args.parameter), load(args.context), load_action_cost(args.action_cost))
    result['inputs'] = {name:{'path':str(path), 'sha256':sha(path)} for name,path in
        (('endpoint', args.endpoint), ('parameter', args.parameter), ('context', args.context), ('action_cost', args.action_cost)) if path is not None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
