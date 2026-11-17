#!/usr/bin/env python3
"""Evaluate two source-backed Round 2 timing gaps without changing cost claims."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound(campaign, ref):
    path = Path(ref['path']).resolve(strict=True)
    if campaign not in path.parents or digest(path) != ref['sha256']:
        raise ValueError('source binding is external or changed')
    return json.loads(path.read_text())


def valid_number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def interval(value):
    return (isinstance(value, list) and len(value) == 2 and all(valid_number(x) for x in value)
            and value[0] <= value[1])


def gate_range(values, threshold):
    if not values:
        return {'status':'unavailable', 'difference_seconds':None}
    lo, hi = min(values), max(values)
    return {'status':'pass' if hi <= threshold else 'fail' if lo > threshold else 'unresolved',
            'difference_seconds':[lo, hi]}


def prefix_options(start, end, cutoff, amount):
    if cutoff <= start:
        return (0.,)
    if cutoff >= end:
        return (amount,)
    return (0., amount)


def correlated_shared_prefixes(start, end, full_time, residual_time, amount):
    """Vertices for one shared unknown interval, split equally between arms."""
    full = prefix_options(start, end, full_time, amount/2)
    residual = prefix_options(start, end, residual_time, amount/2)
    pairs = []
    for f, r in itertools.product(full, residual):
        if full_time < residual_time and f > r:
            continue
        if residual_time < full_time and r > f:
            continue
        if full_time == residual_time and f != r:
            continue
        pairs.append((f, r))
    return pairs


def sensitivity(*, family, costs, decision, clocks, sense_bound, sense_source, ffm_bound):
    if family not in ('ffm', 'senseiver') or costs.get('schema') != 'car_round2_cost_reconciliation_v1' or costs.get('family') != family:
        raise ValueError('family-matched cost reconciliation required')
    if (sense_bound.get('bound_not_measured') is not True or ffm_bound.get('bound_not_measured') is not True):
        raise ValueError('source bounds must explicitly be unmeasured')
    for item in (sense_bound, ffm_bound):
        if not valid_number(item.get('lower_seconds')) or not valid_number(item.get('upper_seconds')) or item['lower_seconds'] > item['upper_seconds']:
            raise ValueError('invalid source bounds')
    sense_window = (sense_source.get('external_start_unix'), sense_source.get('external_result_unix'))
    ffm_window = (ffm_bound.get('external_start_unix'), ffm_bound.get('external_result_unix'))
    if (any(not valid_number(x) for x in (*sense_window, *ffm_window)) or sense_window[0] > sense_window[1] or ffm_window[0] > ffm_window[1]
            or sense_bound['upper_seconds'] > sense_window[1]-sense_window[0]+1e-6
            or ffm_bound['upper_seconds'] > ffm_window[1]-ffm_window[0]+1e-6):
        raise ValueError('source placement windows missing')
    prefix = 'FFM' if family == 'ffm' else 'Senseiver'
    full = f'{prefix}_Selected_Bank_Uniform_FullModel'
    residual = f'{prefix}_Selected_Bank_Residual_Repair'
    arms = costs.get('arms', {})
    if decision is not None and (decision.get('schema') != 'car_round2_final_decision_v1' or decision.get('family') != family):
        raise ValueError('family-matched final decision required')
    missing = costs.get('missing', [])
    if not isinstance(missing, list):
        raise TypeError('reconciliation missing evidence must be explicit')
    known = ('Original Senseiver topology-stable allocator cost input', 'Original unwrapped FFM controller compile/test')
    unaddressed = [x for x in missing if not isinstance(x, str) or not x.startswith(known)]
    cost_ok = not unaddressed
    if costs.get('complete') is not True and not missing:
        cost_ok = False
        unaddressed.append('reconciliation incomplete without an explicit missing-evidence list')
    full_cost = arms.get(full, {}).get('observed_serial_cost_seconds')
    res_cost = arms.get(residual, {}).get('observed_serial_cost_seconds')
    if not all(valid_number(x) and x > 0 for x in (full_cost, res_cost)):
        cost_ok = False
    crossing = (decision or {}).get('crossings', {})
    cross = {}
    for arm in (full, residual):
        item = crossing.get(arm, {})
        epoch = item.get('crossing_epoch')
        bounds = item.get('crossing_cost_bounds_seconds')
        if epoch is not None and interval(bounds):
            try:
                when = clocks[arm]['epochs'][str(epoch)]['recorded_unix']
            except (KeyError, TypeError):
                continue
            if valid_number(when):
                cross[arm] = (when, bounds)
    allocator = costs.get('allocator_final', {})
    overhead = allocator.get('overhead_seconds')
    training = allocator.get('candidate_training_seconds')
    alloc_ok = (cost_ok and allocator.get('status') in {'measured', 'incomplete_campaign_costs'} and valid_number(overhead)
                and valid_number(training) and training > 0)
    scenarios = []
    for sense, ffm in itertools.product((sense_bound['lower_seconds'], sense_bound['upper_seconds']),
                                       (ffm_bound['lower_seconds'], ffm_bound['upper_seconds'])):
        shared = ffm/2 if family == 'ffm' else 0.
        own = sense if family == 'senseiver' else 0.
        quality = gate_range([res_cost+shared+own-1.1*(full_cost+shared)] if cost_ok else [], 0.)
        allocation = gate_range([overhead+own-.05*training] if alloc_ok else [], 0.)
        differences = []
        if cost_ok and len(cross) == 2:
            tf, fb = cross[full]; tr, rb = cross[residual]
            if family == 'ffm':
                additions = correlated_shared_prefixes(*ffm_window, tf, tr, ffm)
            else:
                additions = [(0., v) for v in prefix_options(*sense_window, tr, sense)]
            for f, r in additions:
                differences.extend((rb[0]+r-.8*(fb[1]+f), rb[1]+r-.8*(fb[0]+f)))
        scenarios.append({'senseiver_input_bound_seconds':sense, 'ffm_test_bound_seconds':ffm,
                          'quality_cost_le_1_10':quality, 'allocator_burden_le_0_05':allocation,
                          'efficiency_cost_le_0_80':gate_range(differences, 0.)})
    summaries = {}
    for name in ('quality_cost_le_1_10', 'allocator_burden_le_0_05', 'efficiency_cost_le_0_80'):
        states = {s[name]['status'] for s in scenarios}
        summaries[name] = next(iter(states)) if len(states) == 1 else ('unavailable' if 'unavailable' in states else 'sensitive')
    return {'schema':'car_round2_cost_sensitivity_v1', 'family':family, 'conditional':True,
            'adoption_unchanged':True, 'measured_cost_complete':costs.get('complete') is True,
            'unaddressed_missing':unaddressed, 'endpoint_scenarios':scenarios, 'gate_statuses':summaries,
            'scope':'Only cost inequalities; scientific eligibility, other gates and adoption remain unchanged. Bounds are not measured job clocks.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--inputs', required=True, type=Path, help='JSON document of path/SHA-256 bindings')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    output = args.output.resolve()
    if campaign not in output.parents or campaign not in args.inputs.resolve(strict=True).parents:
        raise ValueError('inputs and output must be campaign-owned')
    refs = json.loads(args.inputs.read_text())
    costs = bound(campaign, refs['cost_reconciliation'])
    decision = bound(campaign, refs['decision']) if refs.get('decision') else None
    if decision and decision.get('cost_reconciliation_sha256') != refs['cost_reconciliation']['sha256']:
        raise ValueError('decision does not bind cost reconciliation')
    clocks = {arm:bound(campaign, ref) for arm, ref in refs.get('checkpoint_indexes', {}).items()}
    cost_sources = {(str(Path(ref['path']).resolve()), ref['sha256']) for ref in costs.get('source_receipts', [])}
    for arm, clock in clocks.items():
        ref = refs['checkpoint_indexes'][arm]
        if (str(Path(ref['path']).resolve()), ref['sha256']) not in cost_sources:
            raise ValueError('checkpoint clock is not bound by cost reconciliation')
        if clock.get('schema') != 'car_round2_checkpoint_clock_index_v1' or clock.get('arm') != arm:
            raise ValueError('checkpoint clock arm mismatch')
    sense = bound(campaign, refs['senseiver_bound'])
    source_path = Path(sense['source_retrospective']).resolve()
    source_ref = {'path':str(source_path), 'sha256':sense['source_sha256']}
    sense_source = bound(campaign, source_ref)
    ffm = bound(campaign, refs['ffm_bound'])
    if (sense.get('schema') != 'car_round2_serial_cost_uncertainty_bound_v1'
            or ffm.get('schema') != 'car_round2_serial_cost_uncertainty_bound_v1'
            or sense_source.get('schema') != 'car_round2_unresolved_numerical_timer_v1'
            or ffm.get('source_event') != 'exec-f0ae9688-ca86-4ea4-b5ca-e9f94ad8b05f'
            or sense_source.get('event_id') != 'exec-ce6d3e4c-2b79-46c7-a9a4-1ddbcf89da07'):
        raise ValueError('two specific source-backed timing gaps required')
    result = sensitivity(family=refs['family'], costs=costs, decision=decision, clocks=clocks,
                         sense_bound=sense, sense_source=sense_source, ffm_bound=ffm)
    result['source_bindings'] = {**refs, 'senseiver_bound_retrospective':source_ref,
                                 'inputs':{'path':str(args.inputs.resolve()),'sha256':digest(args.inputs)}}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
