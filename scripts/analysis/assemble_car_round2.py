#!/usr/bin/env python3
"""Assemble immutable Round 2 configurations from completed SOURCE preparation."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / 'src'))


def write_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(payload)


def inspect_prepared(run, family, campaign):
    """Read one terminal SOURCE fit and bind the actual saved TRAIN panel."""
    import torch
    import yaml

    from phycoflow_reconstruction.data.manifest import dataset_fingerprint
    from phycoflow_reconstruction.data.normalization import FieldNormalizer
    from phycoflow_reconstruction.training.run_store import file_sha256
    from phycoflow_reconstruction.training.scalable_fitting import (
        _digest,
        validate_scalable_artifacts,
    )

    run = run.resolve(strict=True)
    if campaign not in run.parents:
        raise ValueError('prepared run must reside inside the campaign')
    status = json.loads((run / 'status.json').read_text())
    if status.get('status') != 'prepared' or status.get('training_updates') != 0:
        raise ValueError('only terminal zero-update SOURCE preparation is eligible')
    config = yaml.safe_load((run / 'resolved_config.yaml').read_text())
    if config['model']['name'] != ('gl_rbf_cq' if family == 'ffm' else 'senseiver'):
        raise ValueError('prepared SOURCE family differs')
    manifest = json.loads((run / 'run_manifest.json').read_text())
    checkpoint = Path(config['source_run']) / 'checkpoints' / 'last.pt'
    if file_sha256(checkpoint) != manifest.get('source_hashes', {}).get('checkpoint'):
        raise ValueError('prepared SOURCE checkpoint identity changed')
    for name, sha in manifest.get('coherence_family_state_sha256s', {}).items():
        if file_sha256(run / 'artifacts' / f'{name}_family.pt') != sha:
            raise ValueError(f'prepared family artifact changed: {name}')
    if set(manifest.get('coherence_family_state_sha256s', {})) != {'global_distribution', 'multiscale_spectrum', 'topology'}:
        raise ValueError('prepared family manifest must cover all three families')
    core_path, panel_path = run / 'artifacts/scalable_coherence.pt', run / 'artifacts/source_panel.pt'
    core = torch.load(core_path, map_location='cpu', weights_only=True)
    panel = torch.load(panel_path, map_location='cpu', weights_only=True)
    validate_scalable_artifacts(core, panel['provenance'])
    if (panel.get('schema') != 'car_source_panel_v1' or panel['provenance'] != core['provenance']
        or panel['provenance']['source_checkpoint_sha256'] != file_sha256(checkpoint)
        or panel['members_count'] != (4 if family == 'ffm' else 1)
        or panel['sensor_count'] != 256 or panel['reference'].shape[1] != 4096
        or tuple(config['dataset']['grid_shape']) != (100, 403)
        or len(panel['sample_ids']) != 800 or len(set(panel['sample_ids'])) != 800):
        raise ValueError('prepared panel must bind SOURCE and matched TRAIN800/T256/J4-or-J1/Q4096/native40300')
    norm = FieldNormalizer.from_artifact(config['dataset']['normalization_stats_path'],
             field_names=tuple(panel['provenance']['field_order']),
             dataset_fingerprint=dataset_fingerprint(config['dataset']['path']))
    if norm.digest() != panel['provenance']['normalizer_sha256']:
        raise ValueError('prepared panel normalization identity differs')
    ref, mass = panel['reference'].double(), panel['masses'].double()
    if not torch.isfinite(ref).all() or not torch.isfinite(mass).all() or (mass < 0).any() or (mass.sum(-1) <= 0).any():
        raise ValueError('finite TRAIN truth and positive physical mass required')
    weights = mass / mass.sum(-1, keepdim=True)
    scales = ((weights[..., None] * ref.square()).sum(1).mean(0)).sqrt()
    if not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('nonpositive TRAIN field scale requires explicit disposition')
    epsilon = torch.finfo(panel['reference'].dtype).eps
    model_tol = (scales * epsilon).square()
    physical_tol = model_tol * norm.scale.double().square()
    fidelity_path = run / 'artifacts/fidelity_budget_reference.json'
    fidelity = json.loads(fidelity_path.read_text())
    if fidelity.get('binding', {}).get('source_checkpoint_sha256') != file_sha256(checkpoint):
        raise ValueError('prepared fidelity reference SOURCE differs')
    if family == 'ffm':
        risk = torch.tensor(fidelity['risk_reference_panel']['source_member_mse'], dtype=torch.float64)
        allowed = torch.tensor(fidelity['reference']['absolute_allowance'], dtype=torch.float64)
        if not torch.allclose(allowed, risk * .1, rtol=1e-9, atol=0):
            raise ValueError('FFM mean-risk allowance must equal 0.1 times SOURCE member MSE')
    common_hash = _digest({key: panel[key] for key in ('sample_ids', 'sample_indices', 'query_indices',
                                                       'reference', 'coordinates', 'masses', 'observation_manifest')})
    return {'family': family, 'run': str(run), 'config': config, 'sample_ids': panel['sample_ids'],
            'source_checkpoint': str(checkpoint.resolve()), 'source_checkpoint_sha256': file_sha256(checkpoint),
            'source_panel': str(panel_path), 'source_panel_sha256': file_sha256(panel_path),
            'core': str(core_path), 'core_sha256': file_sha256(core_path),
            'fidelity': str(fidelity_path), 'fidelity_sha256': file_sha256(fidelity_path),
            'normalization_stats_sha256': file_sha256(Path(config['dataset']['normalization_stats_path'])),
            'normalizer_sha256': norm.digest(), 'matched_train_support_sha256': common_hash,
            'model_field_scales': scales.tolist(), 'model_sensor_mse_zero_tolerance': model_tol.tolist(),
            'physical_sensor_mse_zero_tolerance': physical_tol.tolist(),
            'reference_dtype': str(panel['reference'].dtype), 'epsilon': epsilon}


def prepare(args):
    import yaml

    campaign = args.campaign.resolve(strict=True)
    output = args.output_dir.resolve()
    if campaign not in output.parents:
        raise ValueError('setup output must be inside the campaign')
    if output.exists():
        raise FileExistsError(output)
    start = time.time()
    sources = {family: inspect_prepared(path, family, campaign) for family, path in
               (('ffm', args.ffm_prepared_run), ('senseiver', args.senseiver_prepared_run))}
    first, second = sources['ffm'], sources['senseiver']
    if (first['matched_train_support_sha256'] != second['matched_train_support_sha256']
        or first['normalizer_sha256'] != second['normalizer_sha256']):
        raise ValueError('SOURCE preparations must use identical TRAIN IDs, observations, targets and support')
    if first['model_field_scales'] != second['model_field_scales']:
        raise ValueError('matched TRAIN field scales differ')
    output.mkdir(parents=True)
    records = {}
    for family, source in sources.items():
        config = copy.deepcopy(source.pop('config'))
        if family == 'ffm':
            config['optimization'].setdefault('rollout_execution', {})['checkpointing'] = True
        config['residual'] = {'enabled': True, 'coordinate_dim': int(config['dataset']['coordinate_dim']),
                              'field_scales': source['model_field_scales'], 'width': 64, 'max_fraction': .25}
        # The initial residual proposal uses the same declared SOURCE recipe;
        # its actual virtual updates provide the TRAIN qualification evidence.
        path = output / f'provisional_residual_probe_{family}.yaml'
        write_new(path, yaml.safe_dump(config, sort_keys=False))
        records[family] = {'path': str(path), 'source_fit': source,
                           'schedule': 'same initial AdamW and warmup as source full-model preparation',
                           'architecture_status': 'provisional until TRAIN response qualification'}
    result = {'schema': 'car_round2_setup_references_v1', 'sources': records,
              'scale_definition': 'sqrt of mean over matched TRAIN frames of native-mass-normalized Q4096 mean squared model-unit truth',
              'sensor_tolerance_definition': '(float32 epsilon times fitted field RMS)^2; model units squared, then multiplied by frozen physical normalization scale squared',
              'sensor_tolerance_role': 'numerical equality tolerance only when matched SOURCE sensor error is exactly zero',
              'sample_count': 800, 'query_count': 4096, 'native_points': 40300,
              'no_model_scores_computed': True, 'start_unix': start, 'end_unix': time.time()}
    result['wall_seconds'] = result['end_unix'] - start
    write_new(output / 'setup_references.json', json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(output / 'setup_references.json')


def assessment_prerequisites(args):
    """Use the existing assessment runner for CPU-only panels and truth scales."""
    from phycoflow_reconstruction.evaluation.adaptive_campaign import digest

    start = time.time()
    campaign = args.campaign.resolve(strict=True)
    setup = json.loads(args.setup.read_text())
    if setup.get('schema') != 'car_round2_setup_references_v1':
        raise ValueError('completed metadata setup is required')
    train_panel = campaign / 'configs/panels/train.json'
    panel = json.loads(train_panel.read_text())
    expected_ids = setup['sources']['ffm']['source_fit']['sample_ids']
    if (panel.get('schema') != 'car_round2_panel_v1' or panel.get('split') != 'train'
        or panel.get('sample_ids') != expected_ids or len(expected_ids) != 800):
        raise ValueError('frozen assessment TRAIN panel differs from prepared SOURCE IDs')
    ffm_run = setup['sources']['ffm']['source_fit']['run']
    script = Path(__file__).resolve().parents[1] / 'posttraining/assess_car_round2_checkpoints.py'
    prefix = [sys.executable, str(script), '--campaign', str(campaign)]
    validation = campaign / 'configs/panels/validation.json'
    if not validation.exists():
        subprocess.run([*prefix, 'panel', '--split', 'validation', '--core-run', ffm_run], check=True)
    validation_doc = json.loads(validation.read_text())
    if (validation_doc.get('schema') != 'car_round2_panel_v1' or validation_doc.get('split') != 'validation'
        or validation_doc.get('normalizer_sha256') != panel.get('normalizer_sha256')
        or validation_doc.get('dataset_fingerprint') != panel.get('dataset_fingerprint')):
        raise ValueError('frozen VALIDATION panel differs from the matched TRAIN dataset')
    original_event = args.event.resolve(strict=True)
    event = campaign / 'artifacts/development_event_round1_reused_v1.json'
    data = original_event.read_bytes()
    with event.open('xb') as stream:
        stream.write(data)
    if digest(event) != digest(original_event):
        raise ValueError('copied development event does not match Round 1 bytes')
    role = {'schema': 'car_round2_event_role_v1', 'event_sha256': digest(event),
            'original_sha256': digest(original_event), 'original': str(original_event),
            'role': 'Round 1 reused continuity/development endpoint, not an untouched Round 2 task',
            'original_receipt_modified': False}
    write_new(campaign / 'artifacts/development_event_role_v1.json', json.dumps(role, indent=2) + '\n')
    task = args.reserved_task.resolve(strict=True)
    if campaign not in task.parents:
        raise ValueError('frozen reserved task must belong to this campaign')
    normalizer = campaign / 'artifacts/reserved_task_truth_normalizer_v1.json'
    subprocess.run([*prefix, 'reserved-normalizer', '--core-run', ffm_run, '--reserved-task', str(task),
                    '--output', str(normalizer)], check=True)
    evidence = json.loads(normalizer.read_text())
    if (evidence.get('schema') != 'car_round2_reserved_normalizer_v1'
        or evidence.get('reserved_model_scores_used') is not False
        or evidence.get('panel_sha256') != digest(train_panel) or evidence.get('task_sha256') != digest(task)
        or evidence.get('sample_ids') != expected_ids):
        raise ValueError('TRAIN truth-only reserved normalizer is not bound to the frozen panel and task')
    commands = {}
    for family in ('ffm', 'senseiver'):
        source = setup['sources'][family]['source_fit']
        commands[family] = [*prefix, 'prepare-contract',
                            '--source-bank', str(campaign / f'assessment/source_train_{family}_v1'),
                            '--core-run', source['run'], '--event', str(event), '--reserved-task', str(task),
                            '--reserved-task-normalizer', str(evidence['value']),
                            '--reserved-normalizer-evidence', str(normalizer), '--sensor-equality-tolerance',
                            *[str(x) for x in source['physical_sensor_mse_zero_tolerance']],
                            '--output', str(campaign / f'artifacts/assessment_contract_{family}_v2.json')]
    output = args.output.resolve()
    if campaign not in output.parents:
        raise ValueError('assessment prerequisite receipt must be campaign-owned')
    record = {'schema': 'car_round2_assessment_prerequisites_v1', 'train_panel_sha256': digest(train_panel),
              'validation_panel_sha256': digest(validation), 'event': str(event), 'event_sha256': digest(event),
              'event_role_sha256': digest(campaign / 'artifacts/development_event_role_v1.json'),
              'reserved_task': str(task), 'reserved_task_sha256': digest(task),
              'reserved_task_normalizer': str(normalizer), 'reserved_task_normalizer_sha256': digest(normalizer),
              'prepare_contract_commands': commands, 'no_model_inference': True,
              'start_unix': start, 'end_unix': time.time()}
    record['wall_seconds'] = record['end_unix'] - start
    write_new(output, json.dumps(record, indent=2) + '\n')
    print(output)


def bound_file(path, campaign):
    from phycoflow_reconstruction.evaluation.adaptive_campaign import digest

    path = Path(path).resolve(strict=True)
    if campaign not in path.parents or not path.is_file():
        raise ValueError('all assembled scientific artifacts must belong to the campaign')
    return {'path': str(path), 'sha256': digest(path)}


def checkpoint_probe(args):
    """Create a new immutable FFM probe config without changing its source."""
    import yaml

    campaign = args.campaign.resolve(strict=True)
    source = bound_file(args.config, campaign)
    config = yaml.safe_load(Path(source['path']).read_text())
    if config['model']['name'] != 'gl_rbf_cq' or not config.get('residual', {}).get('enabled'):
        raise ValueError('existing FFM residual probe config required')
    output = args.output.resolve()
    if campaign not in output.parents:
        raise ValueError('probe configuration must be campaign-owned')
    config['optimization'].setdefault('rollout_execution', {})['checkpointing'] = True
    write_new(output, yaml.safe_dump(config, sort_keys=False))
    print(output)


def validate_prefit_inputs(prefit, source, probe_ref):
    """Require the final common coefficients to use this exact probe setup."""
    inputs = prefit.get('inputs', {})
    if (prefit.get('version') != 'common_prefit_calibration_v1'
            or prefit.get('binding', {}).get('source_checkpoint_sha256') != source['source_checkpoint_sha256']
            or inputs.get('residual_config_sha256') != probe_ref['sha256']
            or inputs.get('source_panel_sha256') != source['source_panel_sha256']
            or inputs.get('scalable_artifact_sha256') != source['core_sha256']
            or inputs.get('fidelity_reference_sha256') != source['fidelity_sha256']
            or prefit.get('scale_probe', {}).get('schema') != 'car_round2_scale_probe_v1'
            or prefit.get('scale_joint_ratio') not in (1, 2, 4)
            or prefit.get('scale_probe', {}).get('reserved_task_used') is not False
            or prefit.get('scale_probe', {}).get('decision', {}).get('chosen_ratio') != prefit.get('scale_joint_ratio')):
        raise ValueError('final prefit requires matching SOURCE, residual config and completed TRAIN Scale probes')


def assemble_arms(args):
    """Write paired formal configs after bank, coefficient and allocation freeze."""
    import yaml

    from phycoflow_reconstruction.coherence.adaptive_fitting import validate_bank_target_binding
    from phycoflow_reconstruction.config.validate import validate_config

    campaign = args.campaign.resolve(strict=True)
    setup = json.loads(args.setup.read_text())
    inputs = json.loads(args.inputs.read_text())
    if setup.get('schema') != 'car_round2_setup_references_v1' or inputs.get('schema') != 'car_round2_arm_inputs_v1':
        raise ValueError('completed setup and named final arm inputs required')
    disposition = bound_file(inputs['allocation_disposition'], campaign)
    decision = json.loads(Path(disposition['path']).read_text())
    if decision.get('schema') != 'car_round2_allocation_disposition_v1' or decision.get('mode') not in {'uniform', 'qualified_fixed'}:
        raise ValueError('frozen allocation disposition required')
    selected_family = getattr(args, 'family', None)
    if selected_family is not None and decision['mode'] != 'uniform':
        raise ValueError('source-only initial qualification requires uniform allocation; qualified policy admission uses both families')
    families_to_assemble = [(family, prefix) for family, prefix in
                            (('ffm', 'FFM'), ('senseiver', 'Senseiver'))
                            if selected_family is None or family == selected_family]
    artifacts = {'allocation_disposition': disposition}
    configs = {}
    for family, prefix in families_to_assemble:
        source = setup['sources'][family]['source_fit']
        run = Path(source['run'])
        probe_path = Path(inputs.get('residual_configs', {}).get(family, setup['sources'][family]['path']))
        probe_ref = bound_file(probe_path, campaign)
        probe = yaml.safe_load(probe_path.read_text())
        if family == 'ffm':
            probe['optimization'].setdefault('rollout_execution', {})['checkpointing'] = True
        if not probe['residual']['enabled'] or probe['residual']['field_scales'] != source['model_field_scales']:
            raise ValueError('provisional residual differs from frozen TRAIN field scales')
        bank_ref = bound_file(inputs['banks'][family], campaign)
        prefit_ref = bound_file(inputs['prefits'][family], campaign)
        bank = json.loads(Path(bank_ref['path']).read_text())
        prefit = json.loads(Path(prefit_ref['path']).read_text())
        validate_bank_target_binding(prefit.get('binding', {}), bank)
        validate_prefit_inputs(prefit, source, probe_ref)
        core = bound_file(source['core'], campaign)
        panel = bound_file(source['source_panel'], campaign)
        fidelity = bound_file(source['fidelity'], campaign)
        if (core['sha256'] != source['core_sha256'] or panel['sha256'] != source['source_panel_sha256']
            or fidelity['sha256'] != source['fidelity_sha256']):
            raise ValueError('prepared source/core/fidelity artifacts changed')
        source_manifest = json.loads((run / 'run_manifest.json').read_text())
        families = {}
        for name, expected in source_manifest['coherence_family_state_sha256s'].items():
            record = bound_file(run / 'artifacts' / f'{name}_family.pt', campaign)
            if record['sha256'] != expected:
                raise ValueError(f'prepared {name} family changed')
            families[name] = record
        artifacts[f'bank_{family}'], artifacts[f'coefficients_{family}'], artifacts[f'observer_{family}'] = bank_ref, prefit_ref, core
        for residual in (False, True):
            arm = f'{prefix}_Selected_Bank_' + ('Residual_Repair' if residual else 'Uniform_FullModel')
            config = copy.deepcopy(probe)
            config['residual']['enabled'] = residual
            config['output']['experiment_name'] = f'coherence_adaptive_repair/round_2/runs/formal/{arm}'
            config['optimization'].update(epochs=1000, steps_per_epoch=12, batch_size=64,
                                          train_fraction=1.0, sampling='full_pass')
            config['checkpointing']['epochs'] = [100, 250, 500, 750, 1000]
            config['checkpointing']['save_epoch_zero'] = True
            settings = config['coherence']['scalable_anchored_coherence']
            settings['common_core'] = {**core, 'source_panel_path': panel['path'], 'source_panel_sha256': panel['sha256']}
            settings['common_prefit'] = prefit_ref
            settings['adaptive'] = {'enabled': True, 'bank_artifact': bank_ref}
            settings['allocation'] = {'mode': decision['mode'] if residual else 'uniform', 'uniform_epochs': 10}
            config['coherence']['frozen_family_artifacts'] = families
            config['optimization']['multitask']['fidelity_budget']['reference_artifact'] = fidelity
            config['runtime']['plot_every_steps'] = 10
            validate_config(config)
            configs[arm] = config
    # Reuse the same admission checks that will govern formal launch.
    import importlib.util
    driver_path = Path(__file__).resolve().parents[1] / 'posttraining/run_coherence_adaptive_repair_round2.py'
    spec = importlib.util.spec_from_file_location('car_round2_driver', driver_path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    for arm, config in configs.items():
        driver.check_config(config, arm, 1000)
    for family, _prefix in families_to_assemble:
        driver.check_source_pair(configs, family)
    if selected_family is None:
        driver.check_allocation_disposition(artifacts, campaign, configs)
    output = args.output_dir.resolve()
    if campaign not in output.parents or output.exists():
        raise ValueError('arm output must be a new directory inside the campaign')
    output.mkdir(parents=True)
    records = {}
    for arm, config in configs.items():
        path = output / f'{arm}.yaml'
        write_new(path, yaml.safe_dump(config, sort_keys=False))
        records[arm] = bound_file(path, campaign)
    result = {'schema': 'car_round2_assembled_arms_v1', 'arms': records, 'inputs': artifacts,
              'setup_sha256': bound_file(args.setup, campaign)['sha256'],
              'source_families': [family for family, _prefix in families_to_assemble],
              'note': 'formal candidate configs; fresh-process qualification and full formal freeze remain required'}
    write_new(output / 'arm_configs.json', json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(output / 'arm_configs.json')


def assemble_manifest(args):
    """Create a reviewable freeze manifest only from complete bound evidence."""
    import importlib.util

    campaign = args.campaign.resolve(strict=True)
    setup = json.loads(args.setup.read_text())
    prerequisites = json.loads(args.assessment_prerequisites.read_text())
    arms = json.loads(args.arms.read_text())
    inputs = json.loads(args.inputs.read_text())
    if (setup.get('schema') != 'car_round2_setup_references_v1'
        or prerequisites.get('schema') != 'car_round2_assessment_prerequisites_v1'
        or arms.get('schema') != 'car_round2_assembled_arms_v1'
        or inputs.get('schema') != 'car_round2_freeze_inputs_v1'):
        raise ValueError('complete setup, assessment prerequisites, arm configs and freeze inputs required')
    artifacts = dict(arms['inputs'])
    artifacts.update({
        'train_panel': bound_file(campaign / 'configs/panels/train.json', campaign),
        'validation_panel': bound_file(campaign / 'configs/panels/validation.json', campaign),
        'reserved_task': bound_file(prerequisites['reserved_task'], campaign),
        'reserved_task_normalizer': bound_file(prerequisites['reserved_task_normalizer'], campaign),
        'development_event': bound_file(prerequisites['event'], campaign),
        'forecast': bound_file(inputs['forecast'], campaign),
        'code_inventory': bound_file(inputs['code_inventory'], campaign),
    })
    for family in ('ffm', 'senseiver'):
        source = setup['sources'][family]['source_fit']
        artifacts[f'source_{family}'] = bound_file(source['source_checkpoint'], campaign)
        artifacts[f'fidelity_{family}'] = bound_file(source['fidelity'], campaign)
        artifacts[f'assessment_contract_{family}'] = bound_file(inputs['assessment_contracts'][family], campaign)
    arm_records = {arm: {'config': record, 'qualification': bound_file(inputs['qualifications'][arm], campaign)}
                   for arm, record in arms['arms'].items()}
    manifest = {'schema': 'car_round2_freeze_v1', 'status': 'frozen', 'arms': arm_records, 'artifacts': artifacts}
    driver_path = Path(__file__).resolve().parents[1] / 'posttraining/run_coherence_adaptive_repair_round2.py'
    spec = importlib.util.spec_from_file_location('car_round2_driver', driver_path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.check_freeze(manifest, campaign)
    output = args.output.resolve()
    if campaign not in output.parents:
        raise ValueError('freeze manifest must be campaign-owned')
    write_new(output, json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--campaign', type=Path, required=True)
    prep.add_argument('--ffm-prepared-run', type=Path, required=True)
    prep.add_argument('--senseiver-prepared-run', type=Path, required=True)
    prep.add_argument('--output-dir', type=Path, required=True)
    assessment = sub.add_parser('assessment-prerequisites')
    assessment.add_argument('--campaign', type=Path, required=True)
    assessment.add_argument('--setup', type=Path, required=True)
    assessment.add_argument('--event', type=Path, required=True)
    assessment.add_argument('--reserved-task', type=Path, required=True)
    assessment.add_argument('--output', type=Path, required=True)
    arm_parser = sub.add_parser('arms')
    arm_parser.add_argument('--campaign', type=Path, required=True)
    arm_parser.add_argument('--setup', type=Path, required=True)
    arm_parser.add_argument('--inputs', type=Path, required=True)
    arm_parser.add_argument('--output-dir', type=Path, required=True)
    arm_parser.add_argument('--family', choices=('ffm', 'senseiver'),
                            help='assemble one uniform pair for independent initial qualification')
    freeze = sub.add_parser('manifest')
    freeze.add_argument('--campaign', type=Path, required=True)
    freeze.add_argument('--setup', type=Path, required=True)
    freeze.add_argument('--assessment-prerequisites', type=Path, required=True)
    freeze.add_argument('--arms', type=Path, required=True)
    freeze.add_argument('--inputs', type=Path, required=True)
    freeze.add_argument('--output', type=Path, required=True)
    checkpoint = sub.add_parser('checkpoint-probe')
    checkpoint.add_argument('--campaign', type=Path, required=True)
    checkpoint.add_argument('--config', type=Path, required=True)
    checkpoint.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'assessment-prerequisites':
        assessment_prerequisites(args)
    elif args.command == 'arms':
        assemble_arms(args)
    elif args.command == 'manifest':
        assemble_manifest(args)
    elif args.command == 'checkpoint-probe':
        checkpoint_probe(args)


if __name__ == '__main__':
    main()
