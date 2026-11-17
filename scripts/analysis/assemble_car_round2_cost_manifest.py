#!/usr/bin/env python3
"""Expand explicit receipt registrations into a Round 2 cost allocation manifest.

Each registration declares kind, scope, owner (ffm_shared, senseiver_shared,
residual_shared, all_shared, or a literal arm), and either receipt/glob or launch+completion.
CPU registrations also declare independent or a parent token/job ID. The
optional checkpoint_epochs on assessment jobs associates them with all owned
arms and those epochs. Inventory globs report unregistered receipt files.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get('CAR_REPO_ROOT', Path(__file__).resolve().parents[2])).resolve()
CODE = Path(os.environ.get('CAR_CODE_ROOT', ROOT)).resolve()
sys.path.insert(0, str(CODE / 'src'))
from phycoflow_reconstruction.evaluation.round2_costs import (
    ARMS,
    EPOCHS,
    validate_excluded_enclosure,
)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file(campaign, value):
    path = Path(value).resolve()
    if campaign not in path.parents:
        raise ValueError(f'receipt must be campaign-owned: {path}')
    if not path.is_file():
        raise FileNotFoundError(path)
    return {'path': str(path), 'sha256': _hash(path)}


def _weights(owner):
    if owner in ARMS:
        return {owner: 1.}
    if owner == 'all_shared':
        return dict.fromkeys(ARMS, .25)
    if owner == 'residual_shared':
        return {arm: .5 for arm in ARMS if arm.endswith('Residual_Repair')}
    if owner in ('ffm_shared', 'senseiver_shared'):
        pair = [a for a in ARMS if a.lower().startswith(owner.split('_')[0])]
        return dict.fromkeys(pair, .5)
    raise ValueError(f'unknown cost owner: {owner}')


def assemble(campaign, registry, ledger):
    campaign = Path(campaign).resolve()
    if registry.get('schema') != 'car_round2_cost_registry_v1':
        raise ValueError('cost registry v1 required')
    missing = list(registry.get('open_items', []))
    jobs, referenced, identifiers = {'gpu': [], 'cpu': []}, set(), set()
    for ordinal, entry in enumerate(registry.get('registrations', [])):
        kind, scope, owner = entry['kind'], entry['scope'], entry['owner']
        if kind not in jobs or not scope:
            raise ValueError('each registration requires a GPU/CPU kind and named scope')
        weights = _weights(owner)
        if scope in {'source_preparation', 'bank_selection', 'common_observer'} and owner not in {'ffm_shared', 'senseiver_shared', 'all_shared'}:
            raise ValueError('common SOURCE/bank/observer work must use a shared owner')
        if scope in {'allocation', 'policy_qualification', 'policy_refresh', 'residual_probe'} and any(not arm.endswith('Residual_Repair') for arm in weights):
            raise ValueError('residual-only work requires a residual owner')
        if 'receipt_glob' in entry:
            pattern = Path(entry['receipt_glob'])
            if campaign not in pattern.resolve().parents:
                raise ValueError('receipt glob must be campaign-owned')
            receipts = sorted(glob.glob(str(pattern), recursive=True))
            if not receipts:
                missing.append(f'registration {ordinal}: no receipts match {pattern}')
            records = [{'receipt': path} for path in receipts]
        elif 'launch_glob' in entry and kind == 'gpu':
            pattern = Path(entry['launch_glob'])
            completion_dir = Path(entry['completion_dir']).resolve()
            if campaign not in pattern.resolve().parents or campaign not in completion_dir.parents:
                raise ValueError('launch glob and completion directory must be campaign-owned')
            launches = sorted(glob.glob(str(pattern), recursive=True))
            if not launches:
                missing.append(f'registration {ordinal}: no launches match {pattern}')
            records = []
            for launch in launches:
                token = json.loads(Path(launch).read_text()).get('token')
                if not token or Path(token).name != token:
                    missing.append(f'registration {ordinal}: invalid token in {launch}')
                    continue
                records.append({'launch':launch, 'completion':str(completion_dir / f'{token}.json')})
        elif 'receipt' in entry:
            records = [{'receipt': entry['receipt']}]
        elif kind == 'gpu' and 'launch' in entry and 'completion' in entry:
            records = [{'launch': entry['launch'], 'completion': entry['completion']}]
        else:
            raise ValueError('register a receipt, receipt_glob, or GPU launch/completion pair')
        for record in records:
            try:
                bindings = {name: _file(campaign, path) for name, path in record.items()}
            except FileNotFoundError as exc:
                missing.append(f'registration {ordinal}: missing receipt {exc}')
                continue
            paths = [value['path'] for value in bindings.values()]
            if referenced.intersection(paths):
                raise ValueError('receipt registered more than once')
            referenced.update(paths)
            first = next(iter(bindings.values()))
            identifier = entry.get('id') if len(records) == 1 and 'id' in entry else f"{kind}_{hashlib.sha256(first['path'].encode()).hexdigest()[:20]}"
            if identifier in identifiers:
                raise ValueError('job IDs must be unique')
            identifiers.add(identifier)
            job = {'id':identifier, 'scope':scope, 'allocations':weights, **bindings}
            if 'checkpoint_epochs' in entry:
                epochs = entry['checkpoint_epochs']
                if scope not in {'assessment', 'deployment'} or not isinstance(epochs, list) or not epochs or set(epochs) - set(EPOCHS):
                    raise ValueError('checkpoint epochs require assessment/deployment scope and valid epoch numbers')
                job['checkpoint_epochs'] = epochs
            if kind == 'gpu':
                data = json.loads(Path(first['path']).read_text())
                token = data.get('token')
                if not token:
                    missing.append(f'{identifier}: receipt has no lease token')
                    continue
                job['token'] = token
            else:
                for key in ('independent', 'parent_gpu_token', 'parent_cpu_job', 'active_wall_field'):
                    if key in entry:
                        job[key] = entry[key]
                if 'active_wall_field' in job and (job.get('independent') is not True
                        or job.get('parent_gpu_token') or job.get('parent_cpu_job')):
                    raise ValueError('active wall registration requires independent CPU without a parent')
                if job.get('independent') is not True and not job.get('parent_gpu_token') and not job.get('parent_cpu_job'):
                    raise ValueError('CPU registration must declare independent work or its parent')
            jobs[kind].append(job)
    excluded_enclosures = []
    for value in registry.get('enclosing_receipts', []):
        try:
            binding = _file(campaign, value)
        except FileNotFoundError:
            missing.append(f'enclosing group receipt missing: {value}')
            continue
        doc = json.loads(Path(binding['path']).read_text())
        validate_excluded_enclosure(doc)
        if binding['path'] in referenced:
            raise ValueError('enclosing receipt cannot also be allocated')
        referenced.add(binding['path'])
        excluded_enclosures.append(binding)
    if not registry.get('receipt_inventory_globs'):
        missing.append('receipt inventory globs are not declared')
    for pattern in registry.get('receipt_inventory_globs', []):
        if campaign not in Path(pattern).resolve().parents:
            raise ValueError('receipt inventory glob must be campaign-owned')
        for path in sorted(glob.glob(pattern, recursive=True)):
            resolved = str(Path(path).resolve())
            if Path(path).is_file() and resolved not in referenced:
                missing.append(f'unregistered receipt inventory item: {resolved}')
    allocated = {job['token'] for job in jobs['gpu']}
    for row in ledger.get('closed', []):
        if row.get('token') not in allocated:
            missing.append(f"unregistered closed GPU lease: {row.get('token')}")
    if ledger.get('active'):
        missing.append('GPU ledger contains active leases')
    clocks, extras = {}, {}
    for arm in ARMS:
        path = registry.get('checkpoint_indexes', {}).get(arm)
        if not path:
            missing.append(f'{arm}: checkpoint index is not registered')
        else:
            try:
                clocks[arm] = {'receipt':_file(campaign, path)}
            except FileNotFoundError:
                missing.append(f'{arm}: checkpoint index missing: {path}')
        extras[arm] = {}
        for epoch in EPOCHS:
            matching = [job['id'] for group in jobs.values() for job in group
                        if job['scope'] in {'assessment', 'deployment'} and arm in job['allocations']
                        and epoch in job.get('checkpoint_epochs', [])]
            extras[arm][str(epoch)] = matching
            if not matching:
                missing.append(f'{arm}: epoch {epoch} assessment receipt not registered')
    disposition = registry.get('allocator_disposition', {})
    for family in ('ffm', 'senseiver'):
        if disposition.get(family) not in ('uniform', 'qualified_policy'):
            missing.append(f'{family}: final allocator disposition missing')
    final = {}
    for arm, path in registry.get('final_checkpoints', {}).items():
        if arm not in ARMS:
            raise ValueError('unknown final checkpoint arm')
        try:
            final[arm] = _file(campaign, path)
        except FileNotFoundError:
            missing.append(f'{arm}: final checkpoint missing: {path}')
    return {'schema':'car_round2_cost_allocation_v1', 'gpu_jobs':jobs['gpu'], 'cpu_jobs':jobs['cpu'],
            'checkpoint_runs':clocks, 'checkpoint_extras':extras, 'allocator_disposition':disposition,
            'final_checkpoints':final, 'historical_sunk':registry.get('historical_sunk', []),
            'assembly_missing':missing, 'excluded_enclosures':excluded_enclosures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--registry', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    root = Path('/data2/wanglz/PhyCoFlow-MultiFieldReconstruction').resolve()
    registry, output = args.registry.resolve(), args.output.resolve()
    if root not in campaign.parents or campaign not in registry.parents or campaign not in output.parents:
        raise ValueError('campaign must be on DATA2; registry and output must be campaign-owned')
    ledger_path = campaign / 'operations/gpu_ledger.json'
    before = _hash(ledger_path)
    result = assemble(campaign, json.loads(registry.read_text()), json.loads(ledger_path.read_text()))
    if _hash(ledger_path) != before:
        result['assembly_missing'].append('GPU ledger changed during registry assembly')
    result['registry_binding'] = _file(campaign, registry)
    result['ledger_binding'] = {'path':str(ledger_path), 'sha256':before}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output':str(output), 'missing':result['assembly_missing']}))


if __name__ == '__main__':
    main()
