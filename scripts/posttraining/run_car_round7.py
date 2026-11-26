#!/usr/bin/env python3
"""Prepare and execute the two fresh-SOURCE Round 7 trajectories under a lease."""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import uuid

CODE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('CAR_REPO_ROOT', CODE)).resolve()
sys.path[:0] = [str(CODE / 'src'), str(CODE)]
from phycoflow_reconstruction.evaluation.adaptive_campaign import atomic_json, digest

CONTROL = 'FFM_Round7_Composite_Scale_Control'
JOINT = 'FFM_Round7_Joint_Anchor_Scale'


def validate_common(cfg):
    """Check the frozen scientific recipe before making either fresh arm."""
    opt = cfg['optimization']
    budget = opt['multitask']['fidelity_budget']
    policy = opt['multitask']['external_policy']
    pi = policy.get('lagrangian', {}).get('epoch_calibrated_pi', {})
    warmup = opt.get('lr_warmup', {})
    data = cfg['objectives']['data_retention']
    families = cfg['coherence']['families']
    checks = (
        cfg['source'].get('weight_view') == 'ema_materialized',
        opt.get('sampling') == 'full_pass', opt.get('train_fraction') == 1.0,
        opt.get('batch_size') == 64, opt.get('steps_per_epoch') == 12,
        cfg['coherence']['compute_budget'].get('batch_size') == 64,
        cfg['coherence']['compute_budget'].get('point_count') == 4096,
        all(families[name].get('enabled') is True for name in
            ('global_distribution', 'multiscale_spectrum', 'topology')),
        families['global_distribution'].get('conditional_draws') == 2,
        families['multiscale_spectrum'].get('objective', {}).get('conditional_draws') == 2,
        cfg['rollout'] == {'steps': 4, 'solver': 'euler'},
        warmup.get('enabled') is True, warmup.get('epochs') == 50,
        policy.get('method') == 'lagrangian',
        pi.get('gamma') == 11.621005062848388,
        pi.get('tau_epochs') == 1.0, pi.get('horizon_epochs') == 10.0,
        pi.get('steps_per_epoch') == 12,
        opt['multitask']['coherence_aggregation'].get('method') == 'torchjd_config',
        not data.get('enabled') and data.get('weight') == 0,
        not opt.get('direct_supervision'),
        budget.get('reference_law') == 'training_selected_complement_v1',
        budget.get('rotating_complement') == {'points': 512, 'seed': 20261125},
        cfg['coherence']['scalable_anchored_coherence'].get('condition_query_groups') == 8,
        cfg['runtime'].get('plot_every_steps') == 10,
    )
    if not all(checks):
        raise ValueError('base configuration does not match the frozen Round 7 scientific recipe')


def validate_reference(ref, calibration, reference_path):
    if (calibration.get('schema') != 'car_round7_training_law_calibration_v1'
            or calibration.get('status') != 'completed' or calibration.get('sample_count') != 800):
        raise ValueError('completed TRAIN800 parameter-gradient calibration required')
    if calibration.get('reference', {}).get('sha256') != digest(reference_path):
        raise ValueError('calibration and reference identities differ')
    reference = ref['reference']
    panel = ref['risk_reference_panel']
    if (reference['kind'] != 'mean_risk_difference' or panel.get('sample_count') != 800
            or ref.get('binding', {}).get('fidelity_settings', {}).get('reference_law') != 'training_selected_complement_v1'):
        raise ValueError('training-law TRAIN800 mean risk reference required')
    mse = panel['source_member_mse']
    expected_scales = [max(float(value), 1e-8) for value in mse]
    expected_allowances = [0.1 * float(value) for value in mse]
    if (not mse or any(not math.isfinite(float(value)) or float(value) < 0 for value in mse)
            or reference['risk_scales'] != expected_scales
            or reference['absolute_allowance'] != expected_allowances
            or reference['reference_risks'] != panel['source_mean_risk']
            or ref['binding']['fidelity_settings'].get('mean_risk_scales') != expected_scales):
        raise ValueError('reference must use v2 training-law member-MSE scales and allowances')
    c0_source = ref['c0_preserved_from']
    c0_path = Path(c0_source['path'])
    if (digest(c0_path) != c0_source['sha256'] or
            reference['objective_scale'] != json.loads(c0_path.read_text())['reference']['objective_scale']):
        raise ValueError('objective scale must retain the full-precision admitted C0')
    return reference


def prepare_base(args):
    import yaml
    cfg = yaml.safe_load(args.base_config.read_text())
    budget = cfg['optimization']['multitask']['fidelity_budget']
    budget.pop('reference_artifact', None)
    budget['reference_law'] = 'training_selected_complement_v1'
    budget['rotating_complement'] = {'points': 512, 'seed': 20261125}
    cfg['coherence']['scalable_anchored_coherence']['condition_query_groups'] = 8
    cfg['runtime']['plot_every_steps'] = 10
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    print(args.output)


def prepare(args):
    import yaml
    base = yaml.safe_load(args.base_config.read_text())
    ref = json.loads(args.reference.read_text())
    calibration = json.loads(args.calibration.read_text())
    validate_common(base)
    reference = validate_reference(ref, calibration, args.reference)
    kappa = float(calibration['kappa'])
    if not math.isfinite(kappa) or kappa <= 0:
        raise ValueError('a positive finite TRAIN800 gradient-norm kappa is required')
    out = args.campaign / 'configs'
    out.mkdir(parents=True, exist_ok=True)
    configs = {}
    for arm, objective in ((CONTROL, 'composite'), (JOINT, 'joint_anchor')):
        cfg = copy.deepcopy(base)
        opt = cfg['optimization']
        opt['epochs'], opt['steps_per_epoch'], opt['batch_size'] = 500, 12, 64
        budget = opt['multitask']['fidelity_budget']
        budget['mean_risk_scales'] = reference['risk_scales']
        budget['reference_artifact'] = {'path': str(args.reference.resolve()), 'sha256': digest(args.reference)}
        budget['reference_law'] = 'training_selected_complement_v1'
        budget['rotating_complement'] = {'points': 512, 'seed': 20261125}
        opt['multitask']['coherence_aggregation']['family_vjp'] = args.family_vjp
        opt['multitask']['coherence_aggregation']['vjp_chunk_size'] = args.vjp_chunk_size
        opt.setdefault('rollout_execution', {})['checkpointing'] = args.checkpoint_policy == 'on'
        opt['rollout_execution']['context_cache'] = args.context_cache
        scalable = cfg['coherence']['scalable_anchored_coherence']
        scalable.update(condition_query_groups=8, scale_objective=objective,
                        scale_joint_kappa=kappa if objective == 'joint_anchor' else 1.0)
        cfg['runtime']['plot_every_steps'] = 10
        cfg['output']['experiment_name'] = 'coherence_adaptive_repair/round_7/runs/formal/' + arm
        cfg['checkpointing'].update(every_epochs=25, epochs=[100, 500], panel_milestones_only=True)
        path = out / (arm + '.yaml')
        with path.open('x') as stream:
            yaml.safe_dump(cfg, stream, sort_keys=False)
        configs[arm] = {'path': str(path.resolve()), 'sha256': digest(path)}
    manifest = {'schema': 'car_round7_prepared_v1', 'arms': configs,
                'reference': {'path': str(args.reference.resolve()), 'sha256': digest(args.reference)},
                'calibration': {'path': str(args.calibration.resolve()), 'sha256': digest(args.calibration)},
                'base_config': {'path': str(args.base_config.resolve()), 'sha256': digest(args.base_config)},
                'executor': {'family_vjp': args.family_vjp, 'vjp_chunk_size': args.vjp_chunk_size,
                             'rollout_checkpointing': args.checkpoint_policy == 'on',
                             'context_cache': args.context_cache},
                'test_locked': True, 'review_epoch': 100, 'maximum_epoch': 500}
    path = out / 'prepared.json'
    if path.exists():
        raise FileExistsError(path)
    atomic_json(path, manifest)
    print(path)


def train(args):
    import yaml
    from phycoflow_reconstruction.training.post_training import run_post_training
    from phycoflow_reconstruction.training.run_store import load_project_checkpoint
    if not os.environ.get('CAR_ROUND7_LEASE_TOKEN'):
        raise ValueError('training requires an enclosing Round 7 GPU lease')
    if args.result is not None and args.result.exists():
        raise FileExistsError(args.result)
    prepared = json.loads((args.campaign / 'configs/prepared.json').read_text())
    binding = prepared['arms'][args.arm]
    path = Path(binding['path'])
    if digest(path) != binding['sha256']:
        raise ValueError('prepared config changed')
    if not args.pilot:
        freeze = json.loads((args.campaign / 'configs/formal_freeze.json').read_text())
        if (CODE.resolve() != Path(freeze['code_root']).resolve() or
            freeze.get('prepared_sha256') != digest(args.campaign / 'configs/prepared.json')):
            raise ValueError('formal training requires its frozen execution copy and prepared configuration')
    cfg = yaml.safe_load(path.read_text())
    cfg['runtime']['device'] = args.device
    if args.pilot:
        if args.stop_epoch != 10:
            raise ValueError('pilot endpoint is 10 epochs')
        cfg['output']['experiment_name'] = 'coherence_adaptive_repair/round_7/runs/pilot/' + args.arm
    if args.resume:
        saved = yaml.safe_load((args.resume / 'resolved_config.yaml').read_text())
        saved['runtime'].pop('device', None)
        expected = copy.deepcopy(cfg)
        expected['runtime'].pop('device', None)
        if saved != expected:
            raise ValueError('saved scientific configuration differs from prepared configuration')
    elif args.stop_epoch == 500:
        raise ValueError('epoch 500 requires a resumed, selected epoch-100 lineage')
    if args.stop_epoch == 500:
        review = json.loads((args.campaign / 'operations/learning_review_epoch_100.json').read_text())
        if review.get('selected_arm') != args.arm or review.get('continue_to_500') is not True:
            raise ValueError('epoch-100 decision must select this arm for continuation')
    requested = False
    def stop_signal(_sig, _frame):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    def stop(step, _run):
        return requested or step >= 12 * args.stop_epoch or time.time() >= float(os.environ.get('CAR_ROUND7_LEASE_DEADLINE', 'inf')) - 180
    run = run_post_training(cfg, case_dir=ROOT / 'cases/turbulent_combustion', resume=args.resume, budget_stop_requested=stop)
    receipt = args.result or args.campaign / 'operations' / (('pilot_' if args.pilot else 'formal_') + args.arm + '_epoch_' + str(args.stop_epoch) + '_' + uuid.uuid4().hex + '.json')
    status = json.loads((run / 'status.json').read_text())
    checkpoint = run / 'checkpoints/last.pt'
    actual_step = int(load_project_checkpoint(checkpoint)['global_step'])
    if status.get('global_step') is not None and status['global_step'] != actual_step:
        raise ValueError('terminal status and saved checkpoint steps differ')
    target_reached = actual_step >= 12 * args.stop_epoch
    deadline = float(os.environ.get('CAR_ROUND7_LEASE_DEADLINE', 'inf'))
    disposition = ('target_reached' if target_reached else 'interrupted' if requested else
                   'budget_paused' if time.time() >= deadline - 180 else 'unexpected_stop')
    saved_config = run / 'resolved_config.yaml'
    result = {'schema': 'car_round7_training_result_v1', 'arm': args.arm,
        'run_dir': str(run), 'target_epoch': args.stop_epoch, 'global_step': actual_step,
        'reporting_epoch': actual_step / 12, 'target_reached': target_reached,
        'disposition': disposition, 'status': status, 'config': binding,
        'saved_config': {'path': str(saved_config.resolve()), 'sha256': digest(saved_config)},
        'checkpoint': {'path': str(checkpoint.resolve()), 'sha256': digest(checkpoint)},
        'lease_token': os.environ['CAR_ROUND7_LEASE_TOKEN']}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(run, flush=True)
    if disposition == 'unexpected_stop':
        raise RuntimeError('training stopped before the target without a signal or lease deadline')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    sub = p.add_subparsers(dest='command', required=True)
    base = sub.add_parser('prepare-base')
    base.add_argument('--base-config', type=Path, required=True)
    base.add_argument('--output', type=Path, required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--base-config', type=Path, required=True)
    prep.add_argument('--reference', type=Path, required=True)
    prep.add_argument('--calibration', type=Path, required=True)
    prep.add_argument('--family-vjp', choices=('separate', 'batched', 'factored'), required=True)
    prep.add_argument('--vjp-chunk-size', type=int, choices=(1, 2, 3), required=True)
    prep.add_argument('--checkpoint-policy', choices=('on', 'off'), required=True)
    prep.add_argument('--context-cache', choices=('none', 'condition', 'geometry', 'static_features'), required=True)
    run = sub.add_parser('train')
    run.add_argument('--arm', choices=(CONTROL, JOINT), required=True)
    run.add_argument('--stop-epoch', type=int, choices=(10, 100, 500), required=True)
    run.add_argument('--device', required=True)
    run.add_argument('--resume', type=Path)
    run.add_argument('--pilot', action='store_true')
    run.add_argument('--result', type=Path, help='exclusive path for the terminal training receipt')
    args = p.parse_args()
    {'prepare-base': prepare_base, 'prepare': prepare, 'train': train}[args.command](args)

if __name__ == '__main__':
    main()
