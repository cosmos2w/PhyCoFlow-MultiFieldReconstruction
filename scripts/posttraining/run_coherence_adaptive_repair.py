#!/usr/bin/env python3
"""Prepare and account for Coherence Adaptive Repair Round 1 (CPU only).

The preflight subcommand can run bounded GPU updates; formal training requires
separate qualification and forecast evidence. Preparation receipts alone do
not certify scientific qualification.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(os.environ.get('CAR_REPO_ROOT', Path(__file__).resolve().parents[2])).resolve()
CODE_ROOT = Path(os.environ.get('CAR_CODE_ROOT', Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(CODE_ROOT / 'src'))
from phycoflow_reconstruction.evaluation.adaptive_campaign import (
    CAP_SECONDS,
    EPOCHS,
    RESERVE_SECONDS,
    SOURCES,
    atomic_json,
    begin,
    budget_stop,
    digest,
    end,
    enforce_execution_constraints,
    forecast,
    immutable_json,
    locked_ledger,
    panel,
    snapshot_source,
    spent,
)


def validate_code_capsule(code_root):
    """Rehash the frozen execution inventory before admitting an arm."""
    root = Path(code_root).resolve()
    receipt = json.loads((root / 'execution_receipt.json').read_text())
    files = receipt.get('files')
    if receipt.get('schema') != 'car_execution_code_v1' or not isinstance(files, dict) or not files:
        raise ValueError('invalid frozen code inventory')
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if receipt.get('sha256') != identity or root.name != identity:
        raise ValueError('frozen code inventory identity differs')
    for name, expected in files.items():
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file() or digest(path) != expected:
            raise ValueError(f'frozen execution file changed: {name}')
    return identity


def validate_train_panel(config, campaign):
    """Bind TRAIN callbacks to the exact frozen T256/J4 condition panel."""
    from phycoflow_reconstruction.data.factory import open_field_dataset

    panel_path = campaign / 'panels' / 'train.json'
    manifest = json.loads(panel_path.read_text())
    evaluation = config['evaluation']
    if (manifest.get('split') != 'train' or manifest.get('members') != 4
            or manifest.get('sensor_count') != 256 or evaluation.get('split') != 'train'
            or evaluation.get('members') != 4 or evaluation.get('sensor_counts') != {'T': 256}
            or evaluation.get('sensor_seed') != 2047):
        raise ValueError('formal TRAIN panel requires frozen T256/J4 and sensor seed 2047')
    ids = manifest['sample_ids']
    indices = evaluation.get('sample_indices')
    if (not isinstance(indices, list) or len(indices) != len(ids)
            or len(ids) != math.ceil(manifest['population'] * .1)
            or evaluation.get('max_samples') != len(ids) or len(set(indices)) != len(indices)):
        raise ValueError('formal TRAIN callback indices must cover the frozen 10% panel')
    dataset = open_field_dataset(config['dataset'], split='train')
    try:
        if len(dataset) != manifest['population'] or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(dataset) for i in indices):
            raise ValueError('formal TRAIN split population or indices differ')
        actual = [f'{dataset.trajectory_ids[dataset._items[i][0]]}:{dataset._items[i][1]}' for i in indices]
        if actual != ids:
            raise ValueError('formal TRAIN callback IDs differ from the frozen panel')
    finally:
        dataset.close()
    return digest(panel_path)


def validate_formal_pair(configs):
    """Require identical comparison settings except expanded target and arm path."""
    if set(configs) != {'FFM_B1', 'FFM_A1'}:
        raise ValueError('both formal arm configurations are required')
    normalized = {}
    for arm, supplied in configs.items():
        config = copy.deepcopy(supplied)
        adaptive = config['coherence']['scalable_anchored_coherence'].pop('adaptive', None)
        if arm == 'FFM_B1' and adaptive and adaptive.get('enabled', False):
            raise ValueError('B1 must retain the core-only target')
        if arm == 'FFM_A1':
            if not isinstance(adaptive, dict) or adaptive.get('enabled') is not True or not adaptive.get('bank_artifact'):
                raise ValueError('A1 requires the frozen discovered bank')
            bank = adaptive['bank_artifact']
            path = Path(bank['path'])
            if digest(path) != bank['sha256']:
                raise ValueError('A1 frozen bank file hash differs')
            data = json.loads(path.read_text())
            payload = {k: v for k, v in data.items() if k != 'sha256'}
            internal_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            if data.get('sha256') != internal_hash or not data.get('source_fit_binding_sha256') or data.get('policy_qualified') or not (data.get('directions') or data.get('triples')):
                raise ValueError('A1 requires a bound nonempty uniform-fallback bank')
        config['output'].pop('experiment_name', None)
        normalized[arm] = config
    if normalized['FFM_B1'] != normalized['FFM_A1']:
        raise ValueError('formal arms differ outside the declared adaptive target and output path')


def validate_frozen_pair(manifest, campaign):
    import yaml

    if set(manifest.get('arms', {})) != {'FFM_B1', 'FFM_A1'}:
        raise ValueError('formal freeze must declare both arms')
    configs = {}
    for arm, declaration in manifest['arms'].items():
        path = Path(declaration['config_path'])
        if digest(path) != declaration['config_sha256']:
            raise ValueError(f'{arm} frozen configuration hash differs')
        configs[arm] = yaml.safe_load(path.read_text())
    validate_formal_pair(configs)
    return validate_train_panel(configs['FFM_B1'], campaign)


def modernize_c8(config, train_count=800):
    """Resolve archived case geometry through the current C8 runtime contract.

    The archived query_coverage block was consumed by its runner, not by the
    maintained framework. SharedQuerySampler implements that same C8 floor,
    unary/pair counts and ten-epoch refresh; scalable_anchored_coherence selects it.
    All source-dependent fitted artifacts and relation coefficients are refitted.
    """
    coherence = config['coherence']
    coherence.pop('query_coverage', None)
    coherence.pop('frozen_family_artifacts', None)
    coherence['scalable_anchored_coherence'] = {
        'enabled': True, 'max_samples': train_count, 'field_floor': .01,
        'shape_batches': 8, 'candidate_count': 32,
        'source_members': 4, 'source_sensor_count': 256, 'save_source_panel': True,
    }
    families = coherence['families']
    # Disabled legacy families do not participate in the retained C8 core.
    coherence['families'] = {k: v for k, v in families.items() if v.get('enabled', True)}
    scale = coherence['families']['multiscale_spectrum']
    scale['calibration']['samples'] = train_count
    scale['components'] = {'signed_band_cospectrum': {'enabled': False}}
    scale['relation_extension'].update(implicit_pairs=True, self_weight=1.0, rho_weight=1.0)
    shape = coherence['families']['topology']
    shape['persistence'] = {'definition': 'scalable_pooled_v1', 'projections': 32, 'distance': 'sliced_wasserstein'}
    shape['components'] = {'self': {'enabled': True, 'weight': 1.0}, 'mutual': {'enabled': True, 'weight': 1.0}}
    config['optimization']['multitask']['fidelity_budget']['metric'] = 'physical_member_mse_model_units'
    config['optimization']['multitask']['external_policy'].pop('round7_diagnostics', None)
    config['checkpointing'].pop('validation_epochs', None)
    config['checkpointing']['selection_metric'] = 'reconstruction_mse'
    config['runtime']['plot_every_steps'] = 10
    config['evaluation']['batch_size'] = 8
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, default=ROOT / 'cases/turbulent_combustion/runs/coherence_adaptive_repair/round_1')
    sub = p.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init')
    init.add_argument('--train-ids', type=Path, required=True, help='JSON array of split sample IDs')
    init.add_argument('--validation-ids', type=Path, required=True, help='JSON array of split sample IDs; no endpoint access')
    init.add_argument('--seed', type=int, default=2027)
    init.add_argument('--config-template', type=Path, required=True)
    snap = sub.add_parser('prepare-source')
    snap.add_argument('--source', choices=sorted(SOURCES), required=True)
    edm = sub.add_parser('draft-edm')
    edm.add_argument('--config-template', type=Path, required=True)
    f = sub.add_parser('forecast')
    f.add_argument('--update-seconds', type=float, required=True)
    f.add_argument('--callbacks-seconds', type=float, required=True)
    f.add_argument('--remaining-seconds', type=float, required=True)
    b = sub.add_parser('begin')
    b.add_argument('--token', required=True)
    b.add_argument('--gpu-uuid', required=True)
    b.add_argument('--phase', required=True)
    e = sub.add_parser('end')
    e.add_argument('--token', required=True)
    run = sub.add_parser('preflight')
    run.add_argument('--config', type=Path, required=True)
    run.add_argument('--gpu-uuid', choices=['GPU-233fcd85-5c6a-6212-3f44-655252afec70', 'GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e'], required=True)
    run.add_argument('--max-steps', type=int, required=True)
    run.add_argument('--save-margin-seconds', type=float, required=True)
    run.add_argument('--initial-stage-bound-seconds', type=float, required=True)
    run.add_argument('--resume', type=Path)
    run.add_argument('--pause-at-update', type=int)
    formal = sub.add_parser('formal', help='run one frozen 1000-reporting-epoch FFM arm')
    formal.add_argument('--arm', choices=['FFM_B1', 'FFM_A1'], required=True)
    formal.add_argument('--config', type=Path, required=True)
    formal.add_argument('--gpu-uuid', choices=['GPU-233fcd85-5c6a-6212-3f44-655252afec70', 'GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e'], required=True)
    formal.add_argument('--save-margin-seconds', type=float, required=True)
    formal.add_argument('--initial-stage-bound-seconds', type=float, required=True)
    formal.add_argument('--resume', type=Path)
    fr = sub.add_parser('freeze')
    fr.add_argument('--manifest', type=Path, required=True)
    rc = sub.add_parser('checkpoint-receipt')
    rc.add_argument('--arm', choices=['FFM_B1', 'FFM_A1'], required=True)
    rc.add_argument('--epoch', type=int, choices=EPOCHS, required=True)
    rc.add_argument('--checkpoint', type=Path, required=True)
    args = p.parse_args()
    campaign = args.campaign.resolve()
    ledger = campaign / 'gpu_ledger.json'
    if args.command == 'init':
        import yaml
        train = panel(json.loads(args.train_ids.read_text()), 'train', args.seed)
        validation = panel(json.loads(args.validation_ids.read_text()), 'validation', args.seed + 1)
        immutable_json(campaign / 'panels' / 'train.json', train)
        immutable_json(campaign / 'panels' / 'validation_sealed.json', validation)
        config = yaml.safe_load(args.config_template.read_text())
        source_config_path = campaign / 'sources' / 'ffm' / 'resolved_config.yaml'
        if not (campaign / 'sources' / 'ffm' / 'snapshot_receipt.json').is_file():
            raise ValueError('prepare FFM source before preparing configuration')
        source_config = yaml.safe_load(source_config_path.read_text())
        physical_measure = config['dataset'].get('physical_measure')
        for section in ('dataset', 'model', 'observations'):
            config[section] = source_config[section]
        config['dataset']['physical_measure'] = physical_measure
        config.pop('inherit_base_config', None)
        config['inherit_base_config'] = False
        config['source_run'] = str(campaign / 'sources' / 'ffm')
        config['source_checkpoint'] = 'last.pt'
        config['source'] = {'kind': 'native_run', 'admission': 'frozen_snapshot',
                            'snapshot_receipt': str(campaign / 'sources' / 'ffm' / 'snapshot_receipt.json'),
                            'weight_view': 'ema_materialized', 'allow_integration_source': False}
        config['model']['model_ema_enabled'] = True
        config['model']['model_ema_eval'] = True
        config['coherence'].pop('frozen_family_artifacts', None)
        relation = config['coherence']['families']['multiscale_spectrum'].get('relation_extension', {})
        relation.update(self_weight=1.0, rho_weight=1.0)
        config['rollout']['steps'] = 4
        config['optimization']['epochs'] = 1000
        config['optimization']['steps_per_epoch'] = 13
        config['runtime']['plot_every_steps'] = 10
        config['checkpointing']['epochs'] = list(EPOCHS[1:])
        config['checkpointing'].pop('validation_epochs', None)
        config['checkpointing'].pop('validation_every_epochs', None)
        config['checkpointing']['panel_milestones_only'] = True
        config['checkpointing']['selection_metric'] = 'reconstruction_mse'
        config['evaluation']['generation_steps'] = 4
        config['evaluation']['split'] = 'train'
        config['evaluation']['max_samples'] = max(1, len(train['sample_ids']))
        config['output']['experiment_name'] = 'coherence_adaptive_repair/round_1/FFM_B1_draft'
        config = modernize_c8(config, len(train['sample_ids']))
        # Formal admission additionally requires native and TRAIN qualification receipts.
        out = campaign / 'draft_c8_config.yaml'
        rendered = yaml.safe_dump(config, sort_keys=False)
        if out.exists() and out.read_text() != rendered:
            raise ValueError('draft config already exists with different content')
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(rendered)
        immutable_json(campaign / 'preparation.json', {
            'schema': 'car_preparation_v1', 'status': 'draft_not_qualified',
            'template_sha256': digest(args.config_template), 'draft_sha256': digest(out),
            'warning': 'DRAFT ONLY: refit historical relation coefficients and family artifacts from TRAIN/source, configure adaptive target, qualify EMA and native rollout, freeze full configs and forecast before launch.',
            'checkpoint_epochs': list(EPOCHS), 'loss_curve_epoch_interval': 10,
            'train_panel_sha256': digest(campaign / 'panels' / 'train.json'),
            'validation_panel_sha256': digest(campaign / 'panels' / 'validation_sealed.json'),
        })
        print(campaign / 'preparation.json')
    elif args.command == 'prepare-source':
        rel, expected, epoch, step = SOURCES[args.source]
        source = ROOT / 'cases' / 'turbulent_combustion' / 'runs' / rel
        result = snapshot_source(source, campaign / 'sources' / args.source, expected, epoch, step)
        print(json.dumps(result, indent=2))
    elif args.command == 'draft-edm':
        import yaml
        source = campaign / 'sources' / 'edm'
        if not (source / 'snapshot_receipt.json').is_file():
            raise ValueError('prepare EDM source first')
        config = yaml.safe_load(args.config_template.read_text())
        base = yaml.safe_load((source / 'resolved_config.yaml').read_text())
        physical_measure = config['dataset'].get('physical_measure')
        for section in ('dataset', 'model', 'observations'):
            config[section] = base[section]
        config['dataset']['physical_measure'] = physical_measure
        config['inherit_base_config'] = False
        config['source_run'] = str(source)
        config['source_checkpoint'] = 'last.pt'
        config['source'] = {'kind': 'native_run', 'admission': 'frozen_snapshot', 'snapshot_receipt': str(source / 'snapshot_receipt.json'), 'weight_view': 'ema_materialized', 'allow_integration_source': False}
        config['model']['model_ema_enabled'] = True
        config['model']['model_ema_eval'] = True
        config['coherence'].pop('frozen_family_artifacts', None)
        relation = config['coherence']['families']['multiscale_spectrum'].get('relation_extension', {})
        relation.update(self_weight=1.0, rho_weight=1.0)
        config['rollout'] = {'steps': 32, 'solver': 'heun'}
        config['observation_consistency'] = {'mode': 'native', 'final_clamp': True}
        config['optimization'].update({'epochs': 12, 'steps_per_epoch': 1, 'batch_size': 1})
        config['coherence']['compute_budget'].update(batch_size=1, point_count=40300)
        config['checkpointing']['epochs'] = [1, 9, 12]
        config['runtime']['plot_every_steps'] = 10
        config['evaluation'].update({'split': 'train', 'generation_steps': 32, 'query_points': 40300, 'batch_size': 1, 'max_samples': 800})
        config['checkpointing'].pop('validation_epochs', None)
        config['checkpointing'].pop('validation_every_epochs', None)
        config['checkpointing']['selection_metric'] = 'reconstruction_mse'
        config['output']['experiment_name'] = 'coherence_adaptive_repair/round_1/EDM_draft'
        config = modernize_c8(config)
        config['evaluation']['batch_size'] = 1
        out = campaign / 'draft_edm_config.yaml'
        rendered = yaml.safe_dump(config, sort_keys=False)
        if out.exists() and out.read_text() != rendered:
            raise ValueError('draft EDM config already differs')
        if not out.exists():
            out.write_text(rendered)
        print(out)
    elif args.command == 'forecast':
        with locked_ledger(ledger) as state:
            result = forecast(state, args.update_seconds, args.callbacks_seconds, args.remaining_seconds)
        atomic_json(campaign / 'forecast.json', result)
        print(json.dumps(result, indent=2))
        return 0 if result['fits'] else 2
    elif args.command == 'begin':
        enforce_execution_constraints(campaign, args.gpu_uuid)
        with locked_ledger(ledger) as state:
            begin(state, args.token, args.gpu_uuid, args.phase)
    elif args.command == 'end':
        with locked_ledger(ledger) as state:
            end(state, args.token)
    elif args.command in ('preflight', 'formal'):
        is_formal = args.command == 'formal'
        if not all(math.isfinite(x) and x > 0 for x in (args.save_margin_seconds, args.initial_stage_bound_seconds)):
            raise ValueError('positive finite save and initial-stage bounds required')
        if not is_formal and (args.max_steps < 1 or args.max_steps > 156):
            raise ValueError('preflight must be bounded to 1..156 accepted updates')
        import yaml
        declared_model = yaml.safe_load(args.config.read_text()).get('model', {}).get('name')
        source_family = {'gl_rbf_cq': 'ffm', 'diffusion_pde': 'edm'}.get(declared_model)
        enforce_execution_constraints(campaign, args.gpu_uuid, source_family)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
        cache = campaign / 'caches' / args.gpu_uuid / 'keops'
        cache.mkdir(parents=True, exist_ok=True)
        os.environ['KEOPS_CACHE_FOLDER'] = str(cache)
        build = '_'.join(platform.uname()[:3]) + f'_p{sys.version.split()[0]}_CUDA_VISIBLE_DEVICES_{args.gpu_uuid}'
        (cache / build).mkdir(parents=True, exist_ok=True)
        # Check ownership before scientific imports can initialize CUDA.
        listed = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader,nounits'], text=True).splitlines()
        if args.gpu_uuid not in [x.strip() for x in listed]:
            raise ValueError('requested physical GPU UUID not found')
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
        if any(line.split(',')[0].strip() == args.gpu_uuid and int(line.split(',')[1].strip()) != os.getpid() for line in apps.splitlines() if ',' in line):
            raise ValueError('requested GPU has active compute process')
        occupied_start = time.time()
        from phycoflow_reconstruction.cli import _load_case_config
        case_dir = ROOT / 'cases' / 'turbulent_combustion'
        config = _load_case_config(args.config, case_dir, 'turbulent_combustion', [])
        if config['evaluation']['split'] != 'train':
            raise ValueError('preflight evaluation must remain TRAIN only')
        if config.get('source', {}).get('admission') != 'frozen_snapshot' or config['source'].get('weight_view') != 'ema_materialized':
            raise ValueError('pinned EMA materialized source snapshot required')
        model_name = config['model']['name']
        if model_name == 'gl_rbf_cq':
            if config['rollout']['steps'] != 4 or config['rollout']['solver'] != 'euler':
                raise ValueError('FFM preflight requires native four-step Euler')
        elif model_name == 'diffusion_pde':
            if config['rollout']['steps'] != 32 or config['rollout']['solver'] != 'heun' or config['observation_consistency'].get('mode') != 'native':
                raise ValueError('EDM preflight requires native 32-step Heun and native clamping')
        else:
            raise ValueError('unexpected source model')
        source_kind = 'ffm' if model_name == 'gl_rbf_cq' else 'edm'
        expected_source = (campaign / 'sources' / source_kind).resolve()
        if Path(config['source_run']).resolve() != expected_source or digest(expected_source / 'checkpoints' / 'last.pt') != SOURCES[source_kind][1]:
            raise ValueError('source does not match pinned campaign snapshot')
        if config['runtime'].get('plot_every_steps') != 10:
            raise ValueError('ten-epoch curve cadence required')
        if is_formal:
            freeze_path = campaign / 'formal_freeze.json'
            frozen = json.loads(freeze_path.read_text())['manifest']
            declared = frozen['arms'][args.arm]
            validate_frozen_pair(frozen, campaign)
            if frozen.get('status') != 'frozen' or declared['config_sha256'] != digest(args.config):
                raise ValueError('arm config does not match formal freeze')
            if frozen.get('code_root') != str(CODE_ROOT) or not frozen.get('forecast', {}).get('fits') or frozen.get('driver_sha256') != digest(Path(__file__)):
                raise ValueError('qualified code capsule, frozen driver and feasible forecast required')
            validate_code_capsule(CODE_ROOT)
            if frozen.get('assessment_driver_sha256') != digest(ROOT / 'scripts/posttraining/assess_car_checkpoints.py'):
                raise ValueError('frozen assessment driver changed')
            if model_name != 'gl_rbf_cq' or config['optimization']['epochs'] != 1000 or config['optimization']['steps_per_epoch'] != 13 or config['optimization']['batch_size'] != 32:
                raise ValueError('formal arm requires native FFM, 1000 reporting epochs and 13 B32 updates per epoch')
            if config['coherence']['compute_budget']['point_count'] != 4096:
                raise ValueError('formal scoring support must remain Q4096')
            if config['checkpointing'].get('epochs') != list(EPOCHS[1:]) or not config['checkpointing'].get('save_epoch_zero'):
                raise ValueError('all prescribed checkpoints, including epoch zero, required')
            if config['checkpointing'].get('panel_milestones_only') is not True:
                raise ValueError('formal TRAIN panels must be limited to prescribed milestones')
            if config['evaluation'].get('members') != 4 or config['evaluation'].get('sensor_counts') != {'T': 256}:
                raise ValueError('fixed T256/J4 assessment required')
            if not config['coherence']['scalable_anchored_coherence'].get('common_core') or not config['optimization']['multitask']['fidelity_budget'].get('reference_artifact'):
                raise ValueError('formal arms require the common frozen C8 body and risk/C0 reference')
        config['runtime']['device'] = 'cuda:0'
        token = str(uuid.uuid4())
        with locked_ledger(ledger) as state:
            if spent(state) + args.initial_stage_bound_seconds + args.save_margin_seconds >= CAP_SECONDS - RESERVE_SECONDS:
                raise ValueError('insufficient budget for bounded initial stage and safe save')
            begin(state, token, args.gpu_uuid, 'formal_training' if is_formal else 'preflight', now=occupied_start)
        previous_alarm = signal.getsignal(signal.SIGALRM)
        def initial_stage_expired(_signum, _frame):
            raise TimeoutError('initial fitting/first-update stage exceeded its admitted time bound')
        def safe_boundary(completed_updates, run_dir):
            signal.alarm(0)
            return (
                (not is_formal and args.pause_at_update is not None and completed_updates >= args.pause_at_update)
                or budget_stop(ledger, safety_seconds=args.save_margin_seconds)
            )
        try:
            signal.signal(signal.SIGALRM, initial_stage_expired)
            signal.alarm(max(1, math.ceil(args.initial_stage_bound_seconds - (time.time() - occupied_start))))
            atomic_json(campaign / 'launches' / f'{token}.json', {'schema': 'car_formal_launch_v1' if is_formal else 'car_preflight_launch_v1', 'token': token, 'arm': args.arm if is_formal else None, 'gpu_uuid': args.gpu_uuid, 'config_sha256': digest(args.config), 'max_steps': None if is_formal else args.max_steps, 'started_unix': time.time(), 'pid': os.getpid(), 'code_root': str(CODE_ROOT), 'driver_sha256': digest(Path(__file__)), 'initial_stage_bound_seconds': args.initial_stage_bound_seconds, 'save_margin_seconds': args.save_margin_seconds})
            from phycoflow_reconstruction.training.post_training import run_post_training
            result = run_post_training(config, case_dir=case_dir, max_steps=None if is_formal else args.max_steps,
                                       resume=args.resume,
                                       budget_stop_requested=safe_boundary)
            status = json.loads((result / 'status.json').read_text())
            atomic_json(campaign / 'completions' / f'{token}.json', {'token': token, 'status': status.get('status'), 'run_dir': str(result), 'finished_unix': time.time(), 'global_step': status.get('global_step')})
            print(result)
        except BaseException as exc:
            atomic_json(campaign / 'completions' / f'{token}.json', {'token': token, 'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc), 'finished_unix': time.time()})
            raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_alarm)
            with locked_ledger(ledger) as state:
                end(state, token)
    elif args.command == 'freeze':
        doc = json.loads(args.manifest.read_text())
        if doc.get('status') != 'frozen' or not doc.get('train_qualification_sha256'):
            raise ValueError('manifest must declare frozen status and TRAIN qualification digest')
        if doc.get('code_root') != str(CODE_ROOT) or not doc.get('forecast', {}).get('fits') or doc.get('driver_sha256') != digest(Path(__file__)):
            raise ValueError('formal freeze requires the executed code capsule, driver hash and feasible forecast')
        validate_code_capsule(CODE_ROOT)
        if doc.get('assessment_driver_sha256') != digest(ROOT / 'scripts/posttraining/assess_car_checkpoints.py'):
            raise ValueError('formal freeze requires the frozen assessment driver hash')
        validate_frozen_pair(doc, campaign)
        immutable_json(campaign / 'formal_freeze.json', {'schema': 'car_freeze_receipt_v1',
                       'manifest_sha256': digest(args.manifest), 'manifest': doc})
    elif args.command == 'checkpoint-receipt':
        if not (campaign / 'formal_freeze.json').is_file():
            raise ValueError('formal freeze required')
        immutable_json(campaign / 'checkpoints' / args.arm / f'epoch_{args.epoch}.json', {
            'schema': 'car_checkpoint_receipt_v1', 'arm': args.arm, 'epoch': args.epoch,
            'checkpoint_path': str(args.checkpoint.resolve()), 'checkpoint_sha256': digest(args.checkpoint),
            'freeze_sha256': digest(campaign / 'formal_freeze.json')})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
