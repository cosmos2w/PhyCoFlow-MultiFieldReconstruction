"""Freeze measured fixed-stage cost from closed, hash-bound campaign evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def _union(intervals):
    by_device = {}
    for item in intervals:
        if 'end' not in item or item['end'] < item['start'] or not item.get('device_uuid'):
            raise ValueError('all physical GPU intervals must be closed and valid')
        by_device.setdefault(item['device_uuid'], []).append((item['start'], item['end']))
    total = 0.
    for spans in by_device.values():
        end = None
        for start, stop in sorted(spans):
            if end is None or start > end:
                total += stop - start
                end = stop
            elif stop > end:
                total += stop - end
                end = stop
    return total


def build_cost_receipt(artifacts, epoch, arms):
    """Only complete single-invocation stages qualify; recovery requires complete timing evidence."""
    artifacts = Path(artifacts).resolve()
    ledger_path = artifacts / 'budget_ledger.json'
    attempt_path = artifacts / 'attempts.jsonl'
    forecast_path = artifacts / 'resource_forecast.json'
    ledger = read(ledger_path)
    occupied = _union(ledger['intervals'])
    attempts = [json.loads(line) for line in attempt_path.read_text().splitlines() if line.strip()]
    if len(attempts) != ledger['attempt_count']:
        raise ValueError('attempt ledger does not match closed budget ledger')
    forecast = read(forecast_path)
    measure_path = Path(forecast['measurements_path']).resolve()
    if sha(measure_path) != forecast['measurement_sha256']:
        raise ValueError('prelearning measurements changed')
    measures = read(measure_path)
    files = [ledger_path, attempt_path, forecast_path, measure_path]
    result = {}
    if epoch not in (500, 1000, 1250, 1500, 1750, 2000) or not arms or len(set(arms)) != len(arms):
        raise ValueError('invalid fixed decision boundary or arms')
    for arm in arms:
        if arm not in ('B', 'P', 'T') or arm not in forecast['selected_arms']:
            raise ValueError('arm is outside frozen design')
        previous = 0 if epoch == 500 else 500 if epoch == 1000 else epoch - 250
        receipt_path = artifacts / f'{arm}_{epoch}_complete.json'
        receipt = read(receipt_path)
        stage = f'{arm}_{epoch}'
        intervals = [v for v in ledger['intervals'] if v['stage'] == stage]
        if receipt.get('status') != 'completed' or receipt.get('epoch') != epoch or len(intervals) != 1:
            raise ValueError('missing, interrupted, or multiply invoked fixed-stage timing')
        span = intervals[0]
        if any(other is not span and other['device_uuid'] == span['device_uuid']
               and other['start'] < span['end'] and span['start'] < other['end'] for other in ledger['intervals']):
            raise ValueError('overlapping same-device stage prevents comparable timing')
        rows = [v for v in attempts if v.get('phase') == 'formal' and v.get('arm') == arm
                and v.get('pid') == span['pid'] and span['start'] <= v['time'] <= span['end']]
        expected = list(range(previous * 13, epoch * 13))
        if [v['step'] for v in rows] != expected:
            raise ValueError('stage attempts are incomplete, failed, or replayed')
        seconds = receipt['elapsed_seconds']
        callback = receipt['timing']['totals_seconds']['callback']
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (seconds, callback)) or not 0 <= callback < seconds <= span['end'] - span['start'] + 1.:
            raise ValueError('incomplete or invalid timing')
        actual = seconds / len(rows)
        exclusive = (seconds - callback) / len(rows)
        qualification = measures['update_seconds'][arm]
        if not isinstance(qualification, (float, int)) or not math.isfinite(qualification) or qualification <= 0:
            raise ValueError('invalid prelearning qualification rate')
        result[arm] = {'stage': stage, 'device_uuid': span['device_uuid'], 'attempts': len(rows),
                       'qualification_seconds_per_attempt': qualification,
                       'elapsed_seconds': seconds, 'callback_seconds': callback,
                       'whole_stage_seconds_per_attempt': actual,
                       'callback_exclusive_seconds_per_attempt': exclusive,
                       'forecast_seconds_per_attempt': max(exclusive, qualification),
                       'completion_receipt': str(receipt_path)}
        files.append(receipt_path)
    mixed = len({value['device_uuid'] for value in result.values()}) != 1
    qualification_device = None
    if mixed:
        qualification_devices = {v['device_uuid'] for v in ledger['intervals']
                                 if v.get('stage') in ('qualify', 'qualify-repair')}
        if len(qualification_devices) != 1:
            raise ValueError('mixed-device tie requires a common qualification GPU')
        qualification_device = next(iter(qualification_devices))
    for value in result.values():
        value['decision_seconds_per_attempt'] = (value['qualification_seconds_per_attempt'] if mixed
                                                  else value['whole_stage_seconds_per_attempt'])
    return {'schema': 'round7_decision_cost_v1', 'artifacts': str(artifacts), 'epoch': epoch, 'arms': result,
            'gpu_seconds_used': occupied,
            'tie_cost_basis': 'common-device cold-start qualification' if mixed else 'formal whole-stage',
            'qualification_device_uuid': qualification_device, 'scope': 'whole-stage wall per attempted update includes callbacks; forecast subtracts recorded callback time and reserves panels separately',
            'files': {str(path): sha(path) for path in files}}


def verify_cost_receipt(path, epoch, arms):
    data = read(path)
    if data.get('schema') != 'round7_decision_cost_v1' or data.get('epoch') != epoch or set(data.get('arms', {})) != set(arms):
        raise ValueError('decision cost receipt does not match stage and arms')
    for name, digest in data['files'].items():
        if sha(name) != digest:
            raise ValueError('decision cost evidence changed: ' + name)
    if build_cost_receipt(data['artifacts'], epoch, arms) != data:
        raise ValueError('decision cost receipt does not reproduce from frozen evidence')
    for arm in arms:
        for key in ('decision_seconds_per_attempt', 'whole_stage_seconds_per_attempt', 'forecast_seconds_per_attempt'):
            value = data['arms'][arm][key]
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError('invalid decision cost rate')
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--arms', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    payload = build_cost_receipt(args.artifacts, args.epoch, args.arms)
    with Path(args.output).open('x') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
