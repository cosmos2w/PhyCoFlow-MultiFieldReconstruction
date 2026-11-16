#!/usr/bin/env python3
"""Run two matched final-target qualification sequences serially on one GPU."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import yaml

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / 'src'))
from phycoflow_reconstruction.evaluation.adaptive_campaign import digest, immutable_json


def sequence(args):
    campaign = args.campaign.resolve(strict=True)
    output = args.output.resolve()
    if campaign not in output.parents or output.exists():
        raise ValueError('group receipt must be a new campaign-owned path')
    work = output.parent / f'{output.stem}_files'
    if work.exists():
        raise FileExistsError(work)
    if (not all(math.isfinite(x) and x > 0 for x in (args.max_seconds, args.save_margin_seconds, args.initial_stage_bound_seconds))
            or args.initial_stage_bound_seconds > args.max_seconds - args.save_margin_seconds):
        raise ValueError('each child bound must leave its startup and safe-save margin')
    spec = importlib.util.spec_from_file_location('round2_driver', CODE_ROOT / 'scripts/posttraining/run_coherence_adaptive_repair_round2.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    capsule = driver.check_code_capsule(campaign)
    if campaign not in args.arm_configs.resolve(strict=True).parents:
        raise ValueError('arm configuration manifest must be campaign-owned')
    arms_doc = json.loads(args.arm_configs.read_text())
    if arms_doc.get('schema') != 'car_round2_assembled_arms_v1':
        raise ValueError('assembled formal arm configuration manifest required')
    prefix = 'FFM' if args.family == 'ffm' else 'Senseiver'
    arms = [f'{prefix}_Selected_Bank_Uniform_FullModel', f'{prefix}_Selected_Bank_Residual_Repair']
    formal = {}
    for arm in arms:
        path = driver.check_binding(arms_doc['arms'][arm], campaign)
        formal[arm] = yaml.safe_load(path.read_text())
        driver.check_config(formal[arm], arm, 1000)
    driver.check_source_pair(formal, args.family)
    work.mkdir(parents=True)
    start = time.time()
    immutable_json(work / 'group_launch.json', {'schema': 'car_round2_qualification_group_launch_v1',
                   'family': args.family, 'device_request': args.device, 'arm_configs_sha256': digest(args.arm_configs),
                   'code_capsule_sha256': capsule, 'max_seconds_per_child': args.max_seconds,
                   'save_margin_seconds': args.save_margin_seconds,
                   'initial_stage_bound_seconds': args.initial_stage_bound_seconds, 'started_unix': start,
                   'wall_accounting': 'group wall encloses child work and is not an additional GPU cost'})
    children = []
    status = 'failed'
    failure = None

    def child(name, command):
        began = time.time()
        stdout, stderr = work / f'{name}.stdout', work / f'{name}.stderr'
        pid, launch_error = None, None
        with stdout.open('x') as out, stderr.open('x') as err:
            try:
                process = subprocess.Popen(command, stdout=out, stderr=err)
                pid = process.pid
                code = process.wait()
            except OSError as error:
                code, launch_error = None, str(error)
        record = {'schema': 'car_round2_qualification_child_v1', 'name': name, 'command': command,
                  'pid': pid, 'returncode': code, 'launch_error': launch_error, 'start_unix': began, 'end_unix': time.time(),
                  'stdout': str(stdout), 'stderr': str(stderr),
                  'wall_accounting': ('CPU comparison/timing child wall, separately accountable' if name.endswith(('_compare', '_timing'))
                                      else 'encloses the child GPU lease; do not add both')}
        record['wall_seconds'] = record['end_unix'] - began
        record_path = work / f'{name}.json'
        immutable_json(record_path, record)
        children.append({'path': str(record_path), 'sha256': digest(record_path)})
        if code != 0:
            raise RuntimeError(f'{name} failed with exit code {code}; see {stderr}')

    try:
        for arm in arms:
            path = driver.check_binding(arms_doc['arms'][arm], campaign)
            config = copy.deepcopy(formal[arm])
            config['optimization']['epochs'] = 12
            config['checkpointing']['epochs'] = [9, 10, 12]
            config['output']['experiment_name'] = f'coherence_adaptive_repair/round_2/runs/qualification/{arm}'
            driver.check_config(config, arm, 12)
            config_path = work / f'{arm}_qualification.yaml'
            with config_path.open('x') as stream:
                yaml.safe_dump(config, stream, sort_keys=False)
            runs = {}
            for phase in ('uninterrupted', 'pause', 'resume'):
                launch = work / f'{arm}_{phase}_launch.json'
                name = f'{arm}_{phase}'
                # Resume deliberately reloads the persisted run configuration in a fresh process.
                actual_config = runs['pause'] / 'resolved_config.yaml' if phase == 'resume' else config_path
                command = [sys.executable, str(CODE_ROOT / 'scripts/posttraining/run_coherence_adaptive_repair_round2.py'),
                           '--campaign', str(campaign), 'run', '--arm', arm, '--config', str(actual_config),
                           '--device', args.device, '--qualification', '--max-seconds', str(args.max_seconds),
                           '--save-margin-seconds', str(args.save_margin_seconds),
                           '--initial-stage-bound-seconds', str(args.initial_stage_bound_seconds),
                           '--launch-receipt-output', str(launch)]
                if phase == 'pause':
                    command.extend(['--pause-at-epoch', '9'])
                elif phase == 'resume':
                    command.extend(['--resume', str(runs['pause'])])
                child(name, command)
                launched = json.loads(launch.read_text())
                completion = campaign / 'operations/completions' / f"{launched['token']}.json"
                done = json.loads(completion.read_text())
                expected = ('budget_paused', 108) if phase == 'pause' else ('completed', 144)
                if (done.get('status'), done.get('global_step')) != expected:
                    raise RuntimeError(f'{name} finished without required state {expected}: {done}')
                runs[phase] = Path(done['run_dir']).resolve(strict=True)
                if phase == 'resume' and runs[phase] != runs['pause']:
                    raise ValueError('resume returned a different run')
                if phase == 'pause':
                    pause_completion = completion
            q = work / f'{arm}_qualification.json'
            qualifier = [sys.executable, str(CODE_ROOT / 'scripts/posttraining/qualify_car_round2_resume.py'),
                         '--arm', arm, '--formal-config', str(path), '--uninterrupted', str(runs['uninterrupted']),
                         '--resumed', str(runs['resume']), '--pause-launch', str(work / f'{arm}_pause_launch.json'),
                         '--pause-completion', str(pause_completion), '--resume-launch', str(work / f'{arm}_resume_launch.json'),
                         '--output', str(q)]
            child(f'{arm}_compare', qualifier)
            timing = [sys.executable, str(CODE_ROOT / 'scripts/analysis/summarize_car_round2_timing.py'),
                      '--formal-config', str(path), '--run', str(runs['resume']), '--qualification', str(q),
                      '--pause-launch', str(work / f'{arm}_pause_launch.json'),
                      '--resume-launch', str(work / f'{arm}_resume_launch.json'),
                      '--pause-completion', str(pause_completion), '--resume-completion', str(completion),
                      '--output', str(work / f'{arm}_complete_update_timing.json')]
            child(f'{arm}_timing', timing)
        status = 'completed'
    except Exception as error:  # noqa: BLE001 - preserve the group receipt on any child or validation failure.
        failure = {'type': type(error).__name__, 'message': str(error)}
    finally:
        ended = time.time()
        immutable_json(output, {'schema': 'car_round2_qualification_group_v1', 'status': status,
                       'family': args.family, 'launch_receipt': str(work / 'group_launch.json'),
                       'launch_sha256': digest(work / 'group_launch.json'), 'children': children,
                       'failure': failure, 'end_unix': ended, 'wall_seconds': ended - start,
                       'wall_accounting': 'group wall encloses child work and is not an additional GPU cost'})
    if failure:
        raise RuntimeError(failure['message'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--arm-configs', type=Path, required=True)
    parser.add_argument('--family', choices=['ffm', 'senseiver'], required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    parser.add_argument('--save-margin-seconds', type=float, required=True)
    parser.add_argument('--initial-stage-bound-seconds', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    sequence(parser.parse_args())


if __name__ == '__main__':
    main()
