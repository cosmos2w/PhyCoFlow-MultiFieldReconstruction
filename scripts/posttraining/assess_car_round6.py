#!/usr/bin/env python3
"""Assess one Round 6 raw checkpoint on the frozen VALIDATION100 J4 panel."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CODE / 'src'), str(CODE)]
from scripts.posttraining.assess_car_round5 import context as legacy_context, check_bank, generate, load_json, verify, paired_condition_intervals, source_task_values
from scripts.posttraining.assess_car_round2_checkpoints import hash_file, matched_rows, score_event, sensor_mse


def context(campaign):
    from scripts.posttraining.run_car_round6 import frozen_prepared
    ctx = legacy_context(campaign)
    freeze = load_json(ctx['campaign'] / 'configs/formal_freeze.json')
    path, prepared = frozen_prepared(ctx['campaign'], freeze)
    # selected_prepared proves source snapshot and all other non-execution
    # metadata match the SOURCE identities already verified by legacy_context.
    ctx['prepared_path'] = path
    ctx['prepared'] = prepared
    return ctx


def selected(ctx):
    arms = list(ctx['prepared']['arms'])
    if len(arms) != 1 or arms[0] not in ('FFM_SSS_EpochCalibrated_PI_Repair', 'FFM_SSS_ConstraintExecution_Repair'):
        raise ValueError('one descriptive Round 6 repair candidate required')
    return arms[0]


def bank_rows(ctx, bank, receipt):
    import torch
    for i, sample_id in enumerate(ctx['source']['sample_ids']):
        paths = (ctx['source_dir'] / f'{i:04d}.pt', bank / f'{i:04d}.pt')
        hashes = (ctx['source']['files_sha256'][i], receipt['files_sha256'][i])
        for path, expected in zip(paths, hashes, strict=True):
            if hash_file(path) != expected:
                raise ValueError(f'bank row changed: {path}')
        source, child = (torch.load(p, map_location='cpu', weights_only=True) for p in paths)
        if source['sample_id'] != sample_id:
            raise ValueError('SOURCE sample order differs')
        matched_rows(source, child)
        yield sample_id, source, child


def summarize_fields(records):
    import numpy as np
    keys = ('member_mse', 'ensemble_mean_mse', 'signed_mean_risk', 'unbiased_member_variance', 'sensor_mse')
    return {key: np.asarray([row[key] for row in records], dtype=float).mean(axis=0).tolist() for key in keys}


def intervals(left, right, *, scale=None, seed=6042026):
    import numpy as np
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape or a.ndim < 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('matched finite per-condition values required')
    diff = b-a
    if scale is not None:
        denom = np.asarray(scale, dtype=float)
        if not np.isfinite(denom).all() or (denom <= 0).any():
            raise ValueError('positive finite frozen normalization required')
        diff = diff / denom
    idx = np.random.default_rng(seed).integers(0, len(a), size=(2000, len(a)))
    means = diff[idx].mean(axis=1)
    return {'mean': diff.mean(axis=0).tolist(), 'interval_95': np.quantile(means, [.025, .975], axis=0).tolist(), 'per_condition': diff.tolist()}


def check_terminal(decision, review, receipt_hash):
    if (decision.get('epoch') != review.get('epoch') or decision.get('epoch') not in (100, 225, 500)
        or decision.get('decision') != {100:'stop_at_epoch_100', 225:'endpoint_at_epoch_225', 500:'endpoint_at_epoch_500'}.get(decision.get('epoch'))
        or decision.get('authorize_endpoint_tasks') is not True
        or decision.get('review_sha256') != review['_sha256']
        or decision.get('bank_receipt_sha256') != receipt_hash):
        raise ValueError('explicit terminal decision must bind the task-free review and its bank')
    if review.get('task_scope') != 'both tasks excluded' or review.get('bank_receipt_sha256') != receipt_hash:
        raise ValueError('endpoint must reuse the same task-free assessment and prediction bank')


def do_generate(args, ctx):
    if not os.environ.get('CAR_ROUND6_LEASE_TOKEN'):
        raise ValueError('prediction generation requires the Round 6 GPU lease')
    freeze = load_json(ctx['campaign'] / 'configs/formal_freeze.json')
    if CODE.resolve() != Path(freeze['code_root']).resolve():
        raise ValueError('prediction generation requires the frozen execution copy')
    if args.epoch == 500:
        decision = load_json(ctx['campaign'] / 'operations/learning_review_epoch_100.json')
        if decision.get('continue_to_500') is not True:
            raise ValueError('epoch 500 prediction requires explicit continuation')
    args.arm = selected(ctx)
    args.runtime_device = {'lease_token': os.environ['CAR_ROUND6_LEASE_TOKEN'], 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    generate(args, ctx)


def review(args, ctx):
    import torch
    import yaml
    from phycoflow_reconstruction.coherence import build_enabled_families
    from phycoflow_reconstruction.data.factory import open_field_dataset
    from phycoflow_reconstruction.evaluation.frozen_coherence import FrozenCoherenceObserver
    from phycoflow_reconstruction.evaluation.round2_assessment import native_risks, field_sensor_eligibility
    arm = selected(ctx); bank = args.bank.resolve(strict=True)
    receipt = check_bank(bank, ctx, args.epoch, arm)
    output = args.output.resolve()
    if ctx['campaign'] / 'assessment' not in output.parents:
        raise ValueError('assessment must be campaign-owned')
    output.mkdir(parents=True, exist_ok=False)
    started, cpu = time.time(), time.process_time()
    status = {'schema': 'car_round6_review_job_v1', 'status': 'failed', 'epoch': args.epoch}
    dataset = None
    try:
        baseline = next(row for row in ctx['baseline']['rows'] if row['alpha'] == 0.)
        contract = load_json(verify(ctx['artifact']['contract']))
        core_run = Path(ctx['artifact']['core_run'])
        cfg = yaml.safe_load((core_run / 'resolved_config.yaml').read_text())
        dataset = open_field_dataset(cfg['dataset'], split='train')
        if dataset.normalizer.digest() != ctx['source']['normalizer_sha256'] or list(dataset.field_names) != ctx['source']['field_order']:
            raise ValueError('TRAIN normalizer or field order differs')
        if contract['source_checkpoint_sha256'] != ctx['source']['source_sha256']:
            raise ValueError('observer contract SOURCE differs')
        ref = load_json(verify(ctx['prepared']['source_reference']))['reference']
        allowance = torch.as_tensor(ref['absolute_allowance'], dtype=torch.float64) * dataset.normalizer.scale.double().square()
        if not torch.allclose(allowance, torch.as_tensor(contract['mean_allowance_physical'], dtype=torch.float64), rtol=1e-9, atol=0):
            raise ValueError('TRAIN allowance conversion differs from frozen native field allowance')
        families = build_enabled_families(cfg['coherence'], dataset.data_spec, dataset.normalizer)
        manifest = load_json(core_run / 'run_manifest.json')
        for name, family in families.items():
            path = core_run / 'artifacts' / f'{name}_family.pt'
            if hash_file(path) != manifest['coherence_family_state_sha256s'][name]:
                raise ValueError('observer family artifact changed')
            family.load_state_artifact(torch.load(path, map_location='cpu', weights_only=True))
        core = torch.load(verify(ctx['artifact']['core']), map_location='cpu', weights_only=True)
        if core.get('adaptive_bank') is not None or core['provenance']['source_checkpoint_sha256'] != ctx['source']['source_sha256']:
            raise ValueError('SOURCE-bound core observer required')
        query = torch.tensor(ctx['panel']['query_indices'], dtype=torch.long)
        if query.numel() != 4096 or query.unique().numel() != 4096 or (query < 0).any() or (query >= 40300).any():
            raise ValueError('invalid frozen Q4096 support')
        rows = {'source': [], 'child': []}
        for sample_id, source, child in bank_rows(ctx, bank, receipt):
            truth, mass = source['truth_model'], source['native_masses'].double()
            scale, offset = source['normalizer_scale'], source['normalizer_offset']
            observer = FrozenCoherenceObserver(core, families, expected_provenance=core['provenance'],
                reference=truth[:,query], coordinates=source['coordinates_model'][:,query], sample_ids=[sample_id],
                member_weights=mass[query].unsqueeze(0), reference_weights=mass[query].unsqueeze(0),
                mode='conditional', query_seed=int(contract['shape_query_seed']), complete_scale=True)
            for name, batch in (('source', source), ('child', child)):
                x = batch['members_model']; physical = x*scale+offset
                risk = native_risks(physical, truth*scale+offset, mass, model_family='ffm')
                sensor, counts = sensor_mse(x, source, len(ctx['source']['field_order']))
                record = {key: risk[key].tolist() for key in ('member_mse','ensemble_mean_mse','signed_mean_risk','unbiased_member_variance')}
                record['sensor_mse'] = (sensor * scale.double().square()).tolist()
                record['sensor_counts'] = counts.tolist()
                if name == 'child':
                    values = observer.score_tensors(x[:,:,query])
                    record['scores'] = {key: float(values[key].mean()) for key in ('state','scale','shape')}
                rows[name].append(record)
        n = len(rows['source']); norm = contract['normalizers']
        source_risk = summarize_fields(rows['source']); child_risk = summarize_fields(rows['child'])
        for key in ('member_mse','ensemble_mean_mse','signed_mean_risk','unbiased_member_variance'):
            if not torch.allclose(torch.tensor(source_risk[key]), torch.tensor(baseline['risk'][key]), rtol=1e-5, atol=1e-8):
                raise ValueError(f'cached VALIDATION SOURCE {key} mismatch')
        counts = [sum(r['sensor_counts'][j] for r in rows['child']) for j in range(len(ctx['source']['field_order']))]
        fidelity = field_sensor_eligibility(source_risk, child_risk, model_family='ffm', absolute_mean_allowance=allowance.tolist(), source_sensor_mse=source_risk['sensor_mse'], child_sensor_mse=child_risk['sensor_mse'], sensor_equality_tolerance=contract['sensor_equality_tolerance'])
        scores = {key: sum(r['scores'][key] for r in rows['child']) / n for key in ('state','scale','shape')}
        differences = {key: (scores[key]-baseline['scores'][key])/norm[key] for key in scores}
        child_view = {'scores': scores, 'normalized_core_differences': differences, 'risk': child_risk,
                      'sensor_mse': [x if c else None for x,c in zip(child_risk['sensor_mse'], counts)], 'sensor_observation_counts': counts,
                      'fidelity': fidelity, 'per_condition_scores': [r['scores'] for r in rows['child']],
                      'per_condition_member_mse': [r['member_mse'] for r in rows['child']]}
        source_view = {'scores': {k:baseline['scores'][k] for k in scores}, 'risk': source_risk, 'sensor_mse': baseline['sensor_mse'],
                       'per_condition_scores': baseline['per_condition_scores'], 'per_condition_member_mse': [r['member_mse'] for r in rows['source']]}
        result = {'schema':'car_round6_matched_validation_v1', 'epoch':args.epoch, 'split':'validation', 'population':ctx['panel']['population'],
            'selection_use':'previously accessed same-trajectory development validation', 'sample_ids':ctx['source']['sample_ids'],
            'field_order':ctx['source']['field_order'], 'members':4, 'arm':arm, 'source':source_view, 'candidates':{arm:child_view},
            'normalizers':{k:norm[k] for k in scores}, 'task_scope':'both tasks excluded', 'bank_receipt_sha256':hash_file(bank/'receipt.json'),
            'source_bank_sha256':hash_file(ctx['source_receipt']), 'panel_sha256':hash_file(ctx['panel_path']),
            'checkpoint_sha256':receipt['checkpoint_sha256'], 'prepared_sha256':hash_file(ctx['prepared_path']),
            'source_checkpoint_sha256':ctx['source']['source_sha256'], 'contract_sha256':hash_file(verify(ctx['artifact']['contract'])),
            'core_sha256':hash_file(verify(ctx['artifact']['core'])), 'source_baseline_sha256':hash_file(ctx['baseline_path']),
            'per_condition_fields':rows, 'native_allowance':allowance.tolist()}
        result['paired_core'] = {k:intervals([r[k] for r in source_view['per_condition_scores']], [r[k] for r in child_view['per_condition_scores']], scale=norm[k]) for k in scores}
        result['paired_fields'] = {k:intervals([r[k] for r in rows['source']], [r[k] for r in rows['child']]) for k in ('signed_mean_risk','member_mse','ensemble_mean_mse','unbiased_member_variance','sensor_mse')}
        (output/'assessment.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        status['status'] = 'completed'
    except BaseException as exc:
        status.update(error_type=type(exc).__name__, error=str(exc)); raise
    finally:
        if dataset is not None: dataset.close()
        status.update(wall_seconds=time.time()-started, cpu_seconds=time.process_time()-cpu)
        (output/'receipt.json').write_text(json.dumps(status, indent=2, allow_nan=False)+'\n')


def endpoint(args, ctx):
    import torch
    from phycoflow_reconstruction.evaluation.round2_assessment import scalar_task_score
    from phycoflow_reconstruction.evaluation.round2_task import regional_covariance
    from scripts.posttraining.freeze_car_event import FrozenRedesignedEvent
    arm = selected(ctx); bank = args.bank.resolve(strict=True)
    review_path = args.review.resolve(strict=True)
    decision_path = args.decision.resolve(strict=True)
    if ctx['campaign'] not in review_path.parents or ctx['campaign'] not in decision_path.parents:
        raise ValueError('review and terminal decision must be campaign-owned')
    review_doc = load_json(review_path); review_doc['_sha256'] = hash_file(review_path)
    receipt = check_bank(bank, ctx, review_doc['epoch'], arm)
    bank_hash = hash_file(bank/'receipt.json')
    check_terminal(load_json(decision_path), review_doc, bank_hash)
    if review_doc['source_bank_sha256'] != hash_file(ctx['source_receipt']) or review_doc['panel_sha256'] != hash_file(ctx['panel_path']) or review_doc['arm'] != arm:
        raise ValueError('review SOURCE or panel identity differs')
    output = args.output.resolve()
    if ctx['campaign']/'assessment' not in output.parents: raise ValueError('assessment must be campaign-owned')
    output.mkdir(parents=True, exist_ok=False)
    start, cpu = time.time(), time.process_time()
    status = {'schema':'car_round6_endpoint_job_v1', 'status':'failed'}
    try:
        event = FrozenRedesignedEvent(load_json(verify(ctx['freeze']['event'])))
        task = load_json(verify(ctx['freeze']['reserved_task']))
        rows = {'source': [], 'child': []}
        for sample_id, source, child in bank_rows(ctx, bank, receipt):
            mass, scale, offset = source['native_masses'].double(), source['normalizer_scale'], source['normalizer_offset']
            truth = source['truth_model']*scale+offset
            for label, batch in (('source', source), ('child', child)):
                physical = batch['members_model']*scale+offset
                event_score, area, target = score_event(event, physical, truth, mass)
                regional_members = torch.stack([regional_covariance(m, mass, task, tuple(ctx['source']['field_order'])) for m in physical])
                regional_truth = regional_covariance(truth, mass, task, tuple(ctx['source']['field_order']))
                rows[label].append({'sample_id':sample_id, 'event_score':float(event_score.mean()),
                    'event_member_area':area.reshape(-1).tolist(), 'event_target_area':float(target.mean()),
                    'regional_score':float(scalar_task_score(regional_members, regional_truth, model_family='ffm').mean()),
                    'regional_members':regional_members.reshape(-1).tolist(), 'regional_truth':float(regional_truth.mean())})
        baseline = next(row for row in ctx['baseline']['rows'] if row['alpha'] == 0.)
        source_tasks = source_task_values(ctx, baseline, [r['event_score'] for r in rows['source']], len(rows['source']))
        source_mean = {k: sum(r[k] for r in rows['source'])/len(rows['source']) for k in ('event_score','regional_score')}
        child_mean = {k: sum(r[k] for r in rows['child'])/len(rows['child']) for k in source_mean}
        event_unit = load_json(verify(ctx['artifact']['contract']))['normalizers']['event']
        regional_unit = ctx['freeze']['task_units']['ffm']['normalizer']
        event_diff = (child_mean['event_score']-source_mean['event_score'])/event_unit
        regional_gain = (source_mean['regional_score']-child_mean['regional_score'])/regional_unit
        result = dict(review_doc); result.pop('_sha256')
        candidate = result['candidates'][arm]
        candidate['reused_tasks'] = {'event_score':child_mean['event_score'], 'event_normalized_difference':event_diff,
            'regional_score':child_mean['regional_score'], 'regional_source_score':source_mean['regional_score'],
            'regional_normalized_improvement':regional_gain, 'event_per_condition':[r['event_score'] for r in rows['child']],
            'regional_per_condition':[r['regional_score'] for r in rows['child']]}
        result['source']['reused_tasks'] = source_tasks
        result['task_scope'] = 'reused development tasks at selected endpoint'
        result['task_normalizers'] = {'event':event_unit, 'regional':regional_unit}
        result['per_condition_tasks'] = rows
        result['paired_tasks'] = {name:intervals([r[key] for r in rows['source']], [r[key] for r in rows['child']], scale=unit)
            for name,key,unit in (('event','event_score',event_unit),('regional','regional_score',regional_unit))}
        diffs = candidate['normalized_core_differences']; fidelity = candidate['fidelity']
        candidate['replacement_eligible'] = bool(fidelity['primary_pass'] and fidelity['legacy_member_pass'] and all(x <= .02 for x in diffs.values()) and any(x <= -.05 for x in diffs.values()) and regional_gain >= .05 and event_diff <= .05)
        result['task_decision_sha256'] = hash_file(decision_path)
        result['review_sha256'] = hash_file(review_path)
        (output/'assessment.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        status['status'] = 'completed'
    except BaseException as exc:
        status.update(error_type=type(exc).__name__, error=str(exc)); raise
    finally:
        status.update(wall_seconds=time.time()-start, cpu_seconds=time.process_time()-cpu)
        (output/'receipt.json').write_text(json.dumps(status, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('generate'); p.add_argument('--run',type=Path,required=True); p.add_argument('--epoch',type=int,choices=(100,500),required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--device',default='cuda:0')
    p = sub.add_parser('score'); p.add_argument('--bank',type=Path,required=True); p.add_argument('--epoch',type=int,choices=(100,500),required=True); p.add_argument('--output',type=Path,required=True)
    p = sub.add_parser('endpoint'); p.add_argument('--bank',type=Path,required=True); p.add_argument('--review',type=Path,required=True); p.add_argument('--decision',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); ctx = context(args.campaign)
    {'generate':do_generate, 'score':review, 'endpoint':endpoint}[args.command](args,ctx)

if __name__ == '__main__': main()
