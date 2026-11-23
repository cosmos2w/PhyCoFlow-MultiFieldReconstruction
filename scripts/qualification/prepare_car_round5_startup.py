#!/usr/bin/env python3
"""Prepare, but do not run, one bounded Round 5 startup recipe and old-route timing."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import yaml
from phycoflow_reconstruction.evaluation.adaptive_campaign import digest
from phycoflow_reconstruction.config.validate import validate_config
from phycoflow_reconstruction.training.source import source_hashes
from scripts.posttraining.run_car_round5 import checked

ARMS = ('Native_GT_Continuation', 'Native_GT_SSS_Safeguarded_Continuation')


def prepare(*, campaign, config_path, units_path, action_path, attempt_id,
            kind='initial', disposition_path=None, correction_path=None):
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{2,79}', attempt_id):
        raise ValueError('attempt ID must be a descriptive portable name')
    if kind not in ('initial', 'correction'):
        raise ValueError('unknown startup attempt kind')
    campaign = campaign.resolve(strict=True)
    startup = campaign / 'configs' / 'startup'
    prior = [] if not startup.exists() else [json.loads(p.read_text()) for p in startup.glob('*/preparation.json')]
    if any(p['kind'] == kind for p in prior):
        raise ValueError(f'{kind} startup already prepared; preserve the existing attempt')
    if kind == 'correction' and not any(p['kind'] == 'initial' for p in prior):
        raise ValueError('correction requires an initial startup')
    if kind == 'initial' and (prior or correction_path):
        raise ValueError('initial startup cannot use correction evidence')
    correction = json.loads(correction_path.read_text()) if correction_path else {}
    lower = correction.get('finite_step_overshoot_demonstrated') is True
    beta_zero = float(correction.get('ascending_proposal_fraction', 0)) > .1
    if kind == 'correction' and (correction.get('demonstrated_failure') is not True or not correction.get('correction') or not (lower or beta_zero)):
        raise ValueError('correction requires a demonstrated overshoot or >10% ascending proposals')
    cfg = yaml.safe_load(config_path.read_text())
    units_doc = json.loads(units_path.read_text())
    action = json.loads(action_path.read_text())
    if (units_doc.get('schema') != 'car_round5_source_units_v1' or units_doc.get('status') != 'completed'
            or units_doc.get('sample_count') != 800 or units_doc.get('split') != 'train'
            or units_doc.get('members') != 2 or any(units_doc.get('floor_active', {}).values())):
        raise ValueError('completed matched SOURCE TRAIN800 J2 units without active floors required')
    if (action.get('schema') != 'car_round5_action_replay_v1' or action.get('status') != 'completed'
            or action.get('sample_count') != 800 or action.get('units_sha256') != digest(units_path)):
        raise ValueError('completed TRAIN800 action replay bound to these units required')
    expected_source = source_hashes(cfg)['checkpoint']
    if (units_doc.get('source_checkpoint_sha256') != expected_source or
            units_doc.get('source_panel_sha256') != cfg['coherence']['scalable_anchored_coherence']['common_core']['source_panel_sha256']):
        raise ValueError('SOURCE checkpoint or TRAIN panel identity differs from the units replay')
    units = units_doc.get('units', {})
    if set(units) != {'native', 'state', 'scale', 'shape'} or any(not isinstance(v, (int,float)) or isinstance(v,bool) or not math.isfinite(v) or v <= 0 for v in units.values()):
        raise ValueError('four positive finite SOURCE units required')
    ref = cfg['optimization']['multitask']['fidelity_budget']['reference_artifact']
    ref_path = checked(ref)
    mse = json.loads(ref_path.read_text())['risk_reference_panel']['source_member_mse']
    tol = action.get('field_tolerance_max', [])
    if len(mse) != 5 or len(tol) != 5 or any(not math.isfinite(float(m)) or float(m) <= 0 or not math.isfinite(float(t)) or float(t) < 0 or float(t) >= .0001 * float(m) for m,t in zip(mse,tol,strict=True)):
        raise ValueError('five SOURCE MSEs and negligible replay field tolerances required')
    disposition = {'context_cache': 'none', 'query_law': 'shared'}
    if disposition_path:
        disposition = json.loads(disposition_path.read_text())
        if (disposition.get('schema') != 'car_round5_context_estimator_v1' or disposition.get('qualified') is not True
                or disposition.get('context_cache') not in ('none', 'static_features')
                or disposition.get('query_law') not in ('shared', 'condition_groups_8')):
            raise ValueError('only a qualified context and query disposition can change defaults')
    if cfg.get('source', {}).get('weight_view') != 'ema_materialized':
        raise ValueError('Round 5 must start from the EMA-materialized SOURCE')
    opt = cfg['optimization']
    if (opt.get('batch_size') != 64 or opt.get('steps_per_epoch') != 12 or opt.get('sampling') != 'full_pass'
            or cfg['coherence']['compute_budget']['batch_size'] != 64
            or cfg['coherence']['compute_budget']['point_count'] != 4096
            or cfg['rollout']['steps'] != 4 or cfg['rollout']['solver'] != 'euler'):
        raise ValueError('historical input must carry the matched B64/12/full-pass/Q4096/four-Euler contract')
    dest = startup / attempt_id
    if dest.exists(): raise FileExistsError(dest)
    start_lr, target_lr = (2.5e-6, 2.5e-5) if lower else (5e-6, 5e-5)
    beta1 = 0. if beta_zero else .9
    configs = {}
    for arm in ARMS:
        current = deepcopy(cfg)
        opt = current['optimization']
        for key in ('multitask', 'fast_lagrangian', 'lpc_ordered', 'execution_strategy'):
            opt.pop(key, None)
        opt.update(epochs=500, gradient_balance='weighted_sum', model_mode='eval', lr=target_lr,
                   lr_warmup={'enabled':True,'start_lr':start_lr,'epochs':50}, adam_betas=[beta1,.999],
                   rollout_execution={'checkpointing':True,'context_cache':disposition['context_cache']})
        direct = {'mode':'native' if arm == ARMS[0] else 'safeguarded', 'units':units}
        if arm == ARMS[1]:
            direct.update(source_member_mse=mse, field_tolerance=tol, dot_tolerance=0.)
            current['coherence']['scalable_anchored_coherence']['condition_query_groups'] = 8 if disposition['query_law'] == 'condition_groups_8' else 1
        else:
            current['coherence']['scalable_anchored_coherence'] = {'enabled':False}
            current['evaluation'].pop('batch_size', None)
        opt['direct_supervision'] = direct
        current['runtime']['plot_every_steps'] = 10
        current['evaluation'].update(split='validation', max_samples=100, external_assessment=True)
        current['output']['experiment_name'] = f'coherence_adaptive_repair/round_5/runs/pilot/{attempt_id}/{arm}'
        current['checkpointing'].update(every_epochs=25, epochs=[100,500], panel_milestones_only=True, selection_metric='native_validation_loss')
        validate_config(current)
        configs[arm] = current
    historical_reference = None
    if kind == 'initial':
        historical = deepcopy(cfg)
        historical['runtime']['plot_every_steps'] = 10
        historical['evaluation']['external_assessment'] = True
        historical['output']['experiment_name'] = f'coherence_adaptive_repair/round_5/runs/qualification/{attempt_id}/Historical_SourceScaledRisk_Timing'
        validate_config(historical)
        configs['Historical_SourceScaledRisk_Timing'] = historical
    else:
        initial = next(p for p in prior if p['kind'] == 'initial')
        if initial['historical_config_sha256'] != digest(config_path):
            raise ValueError('correction historical input differs from the initial preparation')
        historical_reference = initial['configs']['Historical_SourceScaledRisk_Timing']
        checked(historical_reference)
    dest.mkdir(parents=True)
    paths = {}
    for name, current in configs.items():
        path = dest / f'{name}.yaml'
        path.write_text(yaml.safe_dump(current, sort_keys=False))
        paths[name] = {'path':str(path), 'sha256':digest(path)}
    if historical_reference is not None:
        paths['Historical_SourceScaledRisk_Timing'] = historical_reference
    receipt = {'schema':'car_round5_startup_preparation_v1','kind':kind,'attempt_id':attempt_id,
               'configs':paths,'historical_config_sha256':digest(config_path),
               'units_sha256':digest(units_path),'action_replay_sha256':digest(action_path),
               'source_reference_sha256':digest(ref_path),'disposition_sha256':digest(disposition_path) if disposition_path else None,
               'correction_evidence_sha256':digest(correction_path) if correction_path else None,
               'context_cache':disposition['context_cache'],'query_law':disposition['query_law'],
               'lr_start':start_lr,'lr_target':target_lr,'beta1':beta1,
               'note':'Preparation only; use measured lease and complete-epoch trainer timing, including callback overhead.'}
    (dest/'preparation.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--historical-config', type=Path, required=True)
    p.add_argument('--units', type=Path, required=True)
    p.add_argument('--action-replay', type=Path, required=True)
    p.add_argument('--attempt-id', required=True)
    p.add_argument('--kind', choices=('initial','correction'), default='initial')
    p.add_argument('--disposition', type=Path)
    p.add_argument('--correction-evidence', type=Path)
    a=p.parse_args()
    result=prepare(campaign=a.campaign, config_path=a.historical_config, units_path=a.units,
                   action_path=a.action_replay, attempt_id=a.attempt_id, kind=a.kind,
                   disposition_path=a.disposition, correction_path=a.correction_evidence)
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
