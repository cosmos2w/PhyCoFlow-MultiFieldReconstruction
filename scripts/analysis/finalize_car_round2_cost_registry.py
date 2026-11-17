#!/usr/bin/env python3
"""Derive a terminal-only registry from explicit, hash-bound campaign evidence.

The separate evidence document supplies GPU outputs and independent coordinators:
those are not hash-bound by the sequence child receipts themselves. No ledger is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get('CAR_REPO_ROOT', Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(ROOT / 'src'))
from phycoflow_reconstruction.evaluation.round2_costs import _active_wall

ARMS = tuple(f'{family}_Selected_Bank_{method}' for family in ('FFM', 'Senseiver')
             for method in ('Uniform_FullModel', 'Residual_Repair'))
EPOCHS = (0, 100, 250, 500, 750, 1000)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_bound(campaign, binding):
    if not isinstance(binding, dict) or set(binding) != {'path', 'sha256'}:
        raise ValueError('explicit path and SHA-256 binding required')
    path = Path(binding['path']).resolve(strict=True)
    if campaign not in path.parents or digest(path) != binding['sha256']:
        raise ValueError('external or changed evidence')
    return path, json.loads(path.read_text())


def binding(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': digest(path)}


def finalize(campaign, base, formal_roots, train_root, validation_root, checkpoints, evidence):
    campaign = Path(campaign).resolve(strict=True)
    _, registry = read_bound(campaign, base)
    if registry.get('schema') != 'car_round2_cost_registry_v1':
        raise ValueError('cost registry v1 required')
    _, extras = read_bound(campaign, evidence)
    _, freeze = read_bound(campaign, extras['formal_freeze'])
    if freeze.get('schema') != 'car_round2_freeze_receipt_v1' or freeze.get('manifest', {}).get('status') != 'frozen':
        raise ValueError('explicit frozen formal identity required')
    outputs = extras.get('output_receipts')
    if not isinstance(outputs, list):
        raise TypeError('explicit GPU output receipt bindings required')
    output_by_path = {}
    for ref in outputs:
        path, doc = read_bound(campaign, ref)
        if str(path) in output_by_path:
            raise ValueError('duplicate output receipt')
        output_by_path[str(path)] = (ref, doc)
    used_outputs = set()
    existing = set()
    for entry in registry['registrations']:
        for key in ('receipt', 'launch', 'completion'):
            if key in entry:
                path = str(Path(entry[key]).resolve())
                if path in existing:
                    raise ValueError('duplicate base registration')
                existing.add(path)
    seen_tokens = set()
    for entry in registry['registrations']:
        if entry.get('kind') == 'gpu':
            p = Path(entry.get('receipt', entry.get('launch', '')))
            if p.is_file():
                token = json.loads(p.read_text()).get('token')
                if token:
                    seen_tokens.add(token)
    graph_context = []
    enclosure_seen = {str(Path(path).resolve()) for path in registry.get('enclosing_receipts', [])}

    def add(entry):
        paths = [str(Path(entry[k]).resolve()) for k in ('receipt', 'launch', 'completion') if k in entry]
        if len(paths) != len(set(paths)) or existing.intersection(paths):
            raise ValueError('duplicate cost registration')
        existing.update(paths)
        registry['registrations'].append(entry)

    def lease(ref, owner, scope, epoch=None):
        path, doc = read_bound(campaign, ref)
        if doc.get('status') != 'completed' or not isinstance(doc.get('token'), str):
            raise ValueError('GPU output must be terminal and token-bound')
        if doc['token'] in seen_tokens:
            raise ValueError('duplicate GPU token')
        seen_tokens.add(doc['token'])
        entry = {'kind': 'gpu', 'scope': scope, 'owner': owner, 'receipt': str(path)}
        if epoch is not None and scope in ('assessment', 'deployment'):
            entry['checkpoint_epochs'] = [epoch]
        add(entry)

    def context(ref):
        path, doc = read_bound(campaign, ref)
        graph_context.append({'path': str(path), 'sha256': ref['sha256'],
                              'schema': doc.get('schema'), 'nonadditive': True})
        if doc.get('schema') in {'car_round2_formal_group_v1', 'car_round2_formal_child_v1',
                                 'car_round2_assessment_sequence_v1', 'car_round2_validation_successor_v11'} or (doc.get('schema') == 'car_round2_assessment_sequence_child_v1' and doc.get('kind') == 'gpu'):
            if str(path) in enclosure_seen or str(path) in existing:
                raise ValueError('duplicate enclosing receipt')
            enclosure_seen.add(str(path))
            registry.setdefault('enclosing_receipts', []).append(str(path))
        return path, doc

    if len(formal_roots) != 3:
        raise ValueError('three terminal formal groups required')
    mature = set()
    mature_runs = {}
    for ref in formal_roots:
        _, group = context(ref)
        if group.get('schema') != 'car_round2_formal_group_v1' or group.get('status') != 'completed' or group.get('failure') is not None:
            raise ValueError('formal group must be terminal successful')
        context({'path': group['launch_receipt'], 'sha256': group['launch_sha256']})
        for child_ref in group['children']:
            _, child = context(child_ref)
            arm = child.get('arm')
            if arm not in ARMS or arm in mature or child.get('returncode') != 0:
                raise ValueError('formal arm failed or duplicated')
            lp, launch = read_bound(campaign, {'path':child['launch_receipt'], 'sha256':child['launch_sha256']})
            cp, completion = read_bound(campaign, {'path':child['completion_receipt'], 'sha256':child['completion_sha256']})
            token = launch.get('token')
            if not token or token in seen_tokens or completion.get('token') != token or completion.get('status') != 'completed' or completion.get('global_step') != 12000:
                raise ValueError('formal completion identity or horizon mismatch')
            run = Path(completion.get('run_dir', '')).resolve(strict=True)
            if (campaign / 'runs' / 'formal' / arm) not in run.parents or run in mature_runs.values():
                raise ValueError('formal run ownership or identity differs')
            mature_runs[arm] = run
            seen_tokens.add(token)
            add({'kind':'gpu', 'scope':'training', 'owner':arm, 'launch':str(lp), 'completion':str(cp)})
            mature.add(arm)
    if mature != set(ARMS):
        raise ValueError('four distinct terminal mature arms required')

    def command_value(cmd, flag):
        if cmd.count(flag) != 1:
            raise ValueError(f'child missing unique {flag}')
        return cmd[cmd.index(flag)+1]

    def child_cost(ref, family, arm=None, epoch=None):
        path, child = read_bound(campaign, ref)
        if child.get('schema') != 'car_round2_assessment_sequence_child_v1' or child.get('returncode') != 0:
            raise ValueError('assessment child failed or invalid')
        cmd = child.get('command', [])
        if child.get('kind') == 'cpu':
            if child.get('independent_cpu') is not True:
                raise ValueError('CPU child is not independent')
            add({'kind':'cpu', 'scope':'assessment', 'owner':arm or family+'_shared',
                 'receipt':str(path), 'independent':True,
                 **({'checkpoint_epochs':[epoch]} if arm is not None else {})})
            return
        if child.get('kind') != 'gpu':
            raise ValueError('unknown assessment child kind')
        context(ref)
        output = Path(command_value(cmd, '--output'))
        if 'generate' in cmd:
            output = output / 'receipt.json'
        elif 'profile-deployment' not in cmd:
            raise ValueError('unknown GPU assessment stage')
        key = str(output.resolve())
        if key not in output_by_path or key in used_outputs:
            raise ValueError('GPU output receipt missing, unbound or duplicated')
        ref2, doc = output_by_path[key]
        requested = command_value(cmd, '--gpu-uuid')
        expected_schema = 'car_round2_native_bank_v1' if 'generate' in cmd else 'car_round2_deployment_timing_v1'
        if doc.get('schema') != expected_schema:
            raise ValueError('GPU output schema differs from stage')
        if doc.get('gpu_uuid') != requested:
            raise ValueError('GPU stage device differs from its command')
        if arm is not None and (doc.get('arm') != arm or doc.get('epoch') != epoch):
            raise ValueError('assessment output arm or epoch mismatch')
        scope = 'deployment' if 'profile-deployment' in cmd else 'assessment'
        lease(ref2, arm or family+'_shared', scope, epoch)
        used_outputs.add(key)

    context(train_root)
    _, train = read_bound(campaign, train_root)
    if train.get('schema') != 'car_round2_train_checkpoint_successor_result_v1' or train.get('status') != 'completed' or train.get('freeze_sha256') != extras['formal_freeze']['sha256']:
        raise ValueError('terminal successful TRAIN root required')
    expected = {(arm, epoch) for arm in ARMS for epoch in EPOCHS}
    seen = set()
    for item in train.get('groups', []):
        arm, epoch = item.get('arm'), item.get('epoch')
        if (arm, epoch) not in expected or (arm, epoch) in seen:
            raise ValueError('TRAIN checkpoint coverage duplicated or invalid')
        seen.add((arm, epoch))
        _, group = context({'path':item['group'], 'sha256':item['group_sha256']})
        if group.get('schema') != 'car_round2_assessment_sequence_v1' or group.get('status') != 'completed' or group.get('phase') != 'train-batch' or len(group.get('children', [])) != 3:
            raise ValueError('TRAIN group incomplete')
        _, launch = read_bound(campaign, {'path':group['launch_receipt'], 'sha256':group['launch_sha256']})
        b = launch.get('train_batch_binding', {})
        if b.get('arm') != arm or set(b.get('checkpoints', {})) != {str(epoch)}:
            raise ValueError('TRAIN launch binding mismatch')
        family = 'ffm' if arm.startswith('FFM_') else 'senseiver'
        children = [read_bound(campaign, ref)[1] for ref in group['children']]
        if {c.get('name') for c in children} != {f'train_{arm}_{epoch:03d}_{stage}' for stage in ('generate', 'deployment', 'score')}:
            raise ValueError('TRAIN child names do not match arm and epoch')
        if sorted(c.get('kind') for c in children) != ['cpu', 'gpu', 'gpu']:
            raise ValueError('TRAIN group requires two GPU stages and one CPU score')
        if {('generate' if 'generate' in c['command'] else 'profile-deployment') for c in children if c['kind'] == 'gpu'} != {'generate', 'profile-deployment'}:
            raise ValueError('TRAIN generation/profile stages invalid')
        profile = next(c for c in children if c['kind'] == 'gpu' and 'profile-deployment' in c['command'])
        if command_value(profile['command'], '--gpu-uuid') != train.get('gpu_devices', {}).get(family):
            raise ValueError('TRAIN profile device differs from family profiling device')
        for ref in group['children']:
            child_cost(ref, family, arm, epoch)
    if seen != expected:
        raise ValueError('all 24 TRAIN checkpoint groups required')

    context(validation_root)
    _, validation = read_bound(campaign, validation_root)
    if validation.get('schema') != 'car_round2_validation_successor_result_v1' or validation.get('status') != 'completed' or validation.get('formal_freeze_sha256') != extras['formal_freeze']['sha256']:
        raise ValueError('terminal successful VALIDATION root required')
    families = set()
    for ref in validation.get('families', []):
        family = ref.get('family')
        if family not in ('ffm', 'senseiver') or family in families:
            raise ValueError('VALIDATION family missing or duplicated')
        families.add(family)
        _, group = context({'path':ref['path'], 'sha256':ref['sha256']})
        if group.get('schema') != 'car_round2_validation_successor_v11' or group.get('status') != 'completed' or group.get('failure') is not None:
            raise ValueError('VALIDATION family incomplete')
        _, family_launch = context({'path':group['launch_receipt'], 'sha256':group['launch_sha256']})
        if family_launch.get('family') != family or not family_launch.get('gpu_uuid'):
            raise ValueError('VALIDATION launch family or device mismatch')
        children = group.get('children', [])
        if len(children) != 26:
            raise ValueError('VALIDATION requires normalizer, source and 24 arm children')
        expected_validation = {(a, e, stage) for a in ARMS if a.lower().startswith(family)
                               for e in EPOCHS for stage in ('generate', 'score')}
        seen_validation = set()
        specials = set()
        for cref in children:
            _, child = read_bound(campaign, cref)
            name = child.get('name', '')
            if child.get('kind') == 'gpu' and command_value(child.get('command', []), '--gpu-uuid') != family_launch['gpu_uuid']:
                raise ValueError('VALIDATION child device differs from family launch')
            arm = next((a for a in ARMS if a in name), None)
            epoch = next((e for e in EPOCHS if f'_{e:03d}_' in name), None)
            if arm is None and name != 'freeze_source_reserved_normalizer' and name != 'validation_source_generate':
                raise ValueError('unrecognized VALIDATION child')
            if arm is not None and (epoch is None or not arm.lower().startswith(family)):
                raise ValueError('invalid VALIDATION arm or epoch')
            if arm is None:
                if name in specials:
                    raise ValueError('duplicate VALIDATION source child')
                specials.add(name)
            else:
                stage = 'generate' if name.endswith('_generate') else 'score' if name.endswith('_score') else None
                key = (arm, epoch, stage)
                if key not in expected_validation or key in seen_validation:
                    raise ValueError('VALIDATION checkpoint stage invalid or duplicated')
                seen_validation.add(key)
            child_cost(cref, family, arm, epoch)
        if seen_validation != expected_validation or specials != {'freeze_source_reserved_normalizer', 'validation_source_generate'}:
            raise ValueError('VALIDATION family coverage incomplete')
    if families != {'ffm', 'senseiver'}:
        raise ValueError('both VALIDATION families required')
    if used_outputs != set(output_by_path):
        raise ValueError('unused output receipt binding')
    _, barrier = read_bound(campaign, {'path':validation['barrier'], 'sha256':validation['barrier_sha256']})
    if barrier.get('status') != 'passed' or barrier.get('train_result_sha256') != train_root['sha256'] or Path(barrier.get('train_result', '')).resolve() != Path(train_root['path']).resolve():
        raise ValueError('global TRAIN choice barrier missing or mismatched')
    nominations = barrier.get('nominations', [])
    if len(nominations) != 2:
        raise ValueError('two nomination CPU jobs required')
    nomination_families = set()
    for ref in nominations:
        path, doc = read_bound(campaign, ref)
        cmd = doc.get('command', [])
        family = command_value(cmd, '--family')
        if family not in families or family in nomination_families or doc.get('status') != 'completed' or doc.get('returncode') != 0:
            raise ValueError('nomination job failed or family invalid')
        nomination_families.add(family)
        add({'kind':'cpu', 'scope':'assessment', 'owner':family+'_shared', 'receipt':str(path), 'independent':True})
    controllers = extras.get('independent_cpu_jobs', [])
    required_controllers = {'train_checkpoint_successor_v13_job.json', 'train_checkpoint_successor_v13_ffm_cpu_job.json', 'train_checkpoint_successor_v13_senseiver_cpu_job.json', 'train_checkpoint_successor_v13_ffm_prefetch_cpu_job.json', 'validation_successor_v11_job.json', 'validation_ffm_successor_v11_coordinator.json', 'validation_senseiver_successor_v11_coordinator.json'}
    if {Path(rec['receipt']['path']).name for rec in controllers} != required_controllers or len(controllers) != len(required_controllers):
        raise ValueError('all seven independent controller receipts required exactly once')
    for rec in controllers:
        name = Path(rec['receipt']['path']).name
        field = 'controller_active_wall_seconds' if name in {'train_checkpoint_successor_v13_job.json', 'validation_successor_v11_job.json'} else 'active_wall_seconds'
        owner = 'ffm_shared' if '_ffm_' in name else 'senseiver_shared' if '_senseiver_' in name else 'all_shared'
        if rec.get('active_wall_field') != field or rec.get('owner') != owner or rec.get('scope') != 'campaign_setup':
            raise ValueError('controller ownership or active wall field differs')
        path, doc = read_bound(campaign, rec['receipt'])
        if doc.get('status', 'completed') != 'completed' or not rec.get('independent'):
            raise ValueError('independent controller evidence is not terminal')
        entry = {'kind':'cpu', 'scope':rec['scope'], 'owner':rec['owner'], 'receipt':str(path), 'independent':True}
        if 'active_wall_field' in rec:
            _active_wall(doc, rec['active_wall_field'])
            entry['active_wall_field'] = rec['active_wall_field']
        add(entry)
    if set(checkpoints) != set(ARMS):
        raise ValueError('four explicitly bound checkpoint indexes required')
    for arm, ref in checkpoints.items():
        path, clock = read_bound(campaign, ref)
        if clock.get('schema') != 'car_round2_checkpoint_clock_index_v1' or clock.get('arm') != arm or Path(clock.get('run', '')).resolve() != mature_runs[arm] or set(clock.get('epochs', {})) != {str(e) for e in EPOCHS}:
            raise ValueError('checkpoint clock identity or coverage mismatch')
        registry.setdefault('checkpoint_indexes', {})[arm] = str(path)
        final = Path(clock['epochs']['1000']['checkpoint_path']).resolve(strict=True)
        if campaign not in final.parents or digest(final) != clock['epochs']['1000']['checkpoint_sha256']:
            raise ValueError('final checkpoint binding changed')
        registry.setdefault('final_checkpoints', {})[arm] = str(final)
    registry.setdefault('terminal_graph_context', []).extend(graph_context)
    registry.setdefault('open_items', []).append('Terminal graph registrations require closed-ledger reconciliation, explicit inventory completeness and final disposition; no inventory or prior timing gap is automatically cleared.')
    return registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--base', required=True, type=Path, help='JSON path/SHA binding')
    parser.add_argument('--formal-root', required=True, action='append', type=Path, help='JSON path/SHA binding; three required')
    parser.add_argument('--train-root', required=True, type=Path, help='JSON path/SHA binding')
    parser.add_argument('--validation-root', required=True, type=Path, help='JSON path/SHA binding')
    parser.add_argument('--checkpoint-indexes', required=True, type=Path, help='JSON map of arm to path/SHA binding')
    parser.add_argument('--evidence', required=True, type=Path, help='JSON path/SHA binding to output and CPU evidence document')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    output = args.output.resolve()
    if campaign not in output.parents:
        raise ValueError('output must be campaign owned')
    read_json = lambda p: json.loads(p.read_text())
    result = finalize(campaign, read_json(args.base), [read_json(x) for x in args.formal_root],
                      read_json(args.train_root), read_json(args.validation_root),
                      read_json(args.checkpoint_indexes), read_json(args.evidence))
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
