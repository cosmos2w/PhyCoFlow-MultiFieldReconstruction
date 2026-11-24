#!/usr/bin/env python3
"""Review completed Round 6 pilot recording and exact PI recursion, without inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]


def main():
    import numpy as np
    import torch
    import yaml
    from PIL import Image
    from phycoflow_reconstruction.evaluation.adaptive_campaign import digest, immutable_json
    from phycoflow_reconstruction.training.fidelity_budget import ExternalPolicy, MeanRiskDifferenceBudget
    from phycoflow_reconstruction.training.scalable_fitting import _digest
    from scripts.posttraining.assess_car_round3 import context
    from scripts.posttraining.run_car_round6 import validate_pilot_epoch_timing
    from scripts.qualification.car_round6_joint_audit import coarse_profile
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--resume-qualification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    config = yaml.safe_load((run / 'resolved_config.yaml').read_text())
    resume = json.loads(args.resume_qualification.read_text())
    if resume.get('qualified') is not True:
        raise ValueError('the affected production fresh-process resume must qualify')
    campaign = run.parents[3]
    prepared = json.loads((campaign / 'configs/prepared.json').read_text())
    if len(prepared['arms']) != 1:
        raise ValueError('exactly one prepared Round 6 pilot required')
    arm, binding = next(iter(prepared['arms'].items()))
    prepared_path = Path(binding['path'])
    if digest(prepared_path) != binding['sha256'] or resume.get('config_sha256') != binding['sha256']:
        raise ValueError('pilot and production resume must bind the same prepared configuration')
    expected = yaml.safe_load(prepared_path.read_text())
    expected['output']['experiment_name'] = 'coherence_adaptive_repair/round_6/runs/pilot/' + arm
    if config != expected:
        raise ValueError('pilot saved configuration differs from its prepared scientific configuration')
    history = [json.loads(line) for line in (run / 'logs/history.jsonl').read_text().splitlines() if line.strip()]
    journal = [json.loads(line) for line in (run / 'logs/epoch_calibrated_pi_updates.jsonl').read_text().splitlines() if line.strip()]
    state = torch.load(run / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
    if state['global_step'] != 240 or [r.get('epoch') for r in history] != list(range(1, 21)):
        raise ValueError('a complete 20-reporting-epoch pilot is required')
    if any(not r.get('epoch_complete') for r in history) or [r['step'] for r in journal] != list(range(1, 241)):
        raise ValueError('complete ordered epoch and per-action histories are required')
    settings = config['optimization']['multitask']['external_policy']
    pi = settings['lagrangian']['epoch_calibrated_pi']
    if config['runtime']['plot_every_steps'] != 10 or pi['steps_per_epoch'] != 12:
        raise ValueError('retained reporting clock and ten-epoch curves are required')
    if settings['method'] != 'lagrangian' or 'direct_supervision' in config['optimization']:
        raise ValueError('the pilot must preserve the SSS-only Lagrangian route')
    reference = MeanRiskDifferenceBudget.from_state_dict(state['external_policy']['reference'])
    policy = ExternalPolicy(settings, reference, 'cpu')
    initial_path = run / 'checkpoints/epoch_000.pt'
    initial = torch.load(initial_path, map_location='cpu', weights_only=False)
    source = context(campaign.parent / 'round_3')['source']
    if initial['global_step'] != 0 or _digest(initial['model']) != source['source_parameter_buffer_sha256']:
        raise ValueError('pilot did not start at the admitted EMA-materialized SOURCE')
    if initial['optimizer']['state'] or initial['external_policy'] != policy.state_dict():
        raise ValueError('pilot must start with fresh child Adam and PI state')
    for row in journal:
        before = policy.state_dict()['multipliers']
        if not np.allclose(before, row['multipliers_before'], rtol=1e-12, atol=1e-12):
            raise ValueError('recorded action used a different pre-action multiplier')
        error = torch.tensor([row['field_violation'][field] for field in reference.field_names], dtype=torch.float64)
        if not bool(torch.isfinite(error).all()):
            raise ValueError('nonfinite pilot constraint')
        policy._commit_feedback(error)
        policy.completed_updates += 1
        reconstructed = policy.state_dict()
        if not np.allclose(reconstructed['multipliers'], row['multipliers_after'], rtol=1e-12, atol=1e-12):
            raise ValueError('recorded multiplier does not match exact PI recursion')
        if not np.allclose(reconstructed['epoch_calibrated_pi']['z'], row['z_after'], rtol=1e-12, atol=1e-12):
            raise ValueError('recorded smoother does not match exact PI recursion')
    final = policy.state_dict()
    if final != state['external_policy']:
        raise ValueError('checkpoint PI state differs from its complete exact action journal')
    keys = [k for k in history[0] if k.startswith('coherence_component/') and k.endswith(('/inner_weight', '/outer_weight', '/calibration_scale'))]
    if not keys or any(len({row[k] for row in history}) != 1 for k in keys):
        raise ValueError('SOURCE-prefit coefficient history is missing or changed')
    for name in ('loss_history.png', 'loss_history.pdf'):
        if not (run / 'logs' / name).is_file():
            raise ValueError(f'missing completed-pilot curve {name}')
    with Image.open(run / 'logs/loss_history.png') as image:
        image.verify()
    definitions = json.loads((run / 'logs/recording_definitions.json').read_text())
    if not definitions.get('loss_component_accumulation_order') or any(
        set(column) != {'name', 'type', 'units', 'meaning'} for column in definitions['columns']):
        raise ValueError('typed ordered recording definitions are incomplete')
    epoch_timing = [json.loads(line) for line in (run / 'logs/epoch_timing.jsonl').read_text().splitlines() if line.strip()]
    validate_pilot_epoch_timing(epoch_timing)
    coarse_lambda, coarse_z = coarse_profile(history, reference.field_names,
        reference.absolute_allowance, reference.risk_scales, epochs=20)
    coarse_lambda *= pi['gamma']
    immutable_json(args.output, {
        'schema': 'car_round6_pilot_qualification_v1', 'qualified': True,
        'run_dir': str(run), 'epochs': 20, 'actions': 240, 'objective_condition_visits': 240*64,
        'checkpoint_sha256': digest(run / 'checkpoints/last.pt'),
        'source_initial_checkpoint_sha256': digest(initial_path),
        'source_parameter_buffer_sha256': source['source_parameter_buffer_sha256'],
        'fresh_source_child_optimizer_and_controller': True,
        'history_sha256': digest(run / 'logs/history.jsonl'),
        'pi_journal_sha256': digest(run / 'logs/epoch_calibrated_pi_updates.jsonl'),
        'recording_definitions_sha256': digest(run / 'logs/recording_definitions.json'),
        'resume_qualification': {'path': str(args.resume_qualification.resolve()), 'sha256': digest(args.resume_qualification)},
        'exact_pi_recursion_matches': True,
        'fixed_coefficients': {key: history[0][key] for key in keys},
        'actual_multipliers': final['multipliers'], 'coarse_last_batch_multipliers': coarse_lambda.tolist(),
        'coarse_minus_actual_multipliers': (coarse_lambda-np.asarray(final['multipliers'])).tolist(),
        'actual_z': final['epoch_calibrated_pi']['z'], 'coarse_last_batch_z': coarse_z.tolist(),
        'coarse_interpretation': 'Pilot local approximation check; last batch repeated for all 12 actions per epoch, not a trained counterfactual or verification of the historical 100-epoch trajectory.',
        'epoch_timing': epoch_timing, 'test_locked': True, 'model_efficacy_assessed': False,
    })
    print(args.output)


if __name__ == '__main__':
    main()
