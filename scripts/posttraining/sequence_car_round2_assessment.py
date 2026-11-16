"""Execute the frozen one-family Round 2 TRAIN then VALIDATION assessment."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / 'src'))
from phycoflow_reconstruction.evaluation.adaptive_campaign import digest, immutable_json

EPOCHS = (0, 100, 250, 500, 750, 1000)


def build_jobs(args, artifacts):
    """Build immutable output paths and the existing driver's ordered commands."""
    campaign = args.campaign
    prefix = 'FFM' if args.family == 'ffm' else 'Senseiver'
    arms = [(f'{prefix}_Selected_Bank_Uniform_FullModel', args.full_run),
            (f'{prefix}_Selected_Bank_Residual_Repair', args.residual_run)]
    executable = [sys.executable, str(CODE_ROOT / 'scripts/posttraining/assess_car_round2_checkpoints.py'),
                  '--campaign', str(campaign)]
    source_train = args.train_source_bank
    source_validation = campaign / 'assessment' / f'source_validation_{args.family}_v1'
    jobs = []
    closeout = ['--final-assessment'] if args.final_assessment else []

    def add(name, kind, command):
        jobs.append({'name': name, 'kind': kind, 'command': executable + [str(v) for v in command]})

    def generate(arm, run, epoch, split, source):
        bank = campaign / 'assessment' / 'banks' / split / arm / f'epoch_{epoch:03d}'
        add(f'{split}_{arm}_{epoch:03d}_generate', 'gpu',
            ['generate', *closeout, '--kind', 'child', '--split', split,
             '--arm', arm, '--run', run, '--epoch', epoch, '--source-bank', source,
             '--output', bank, '--gpu-uuid', args.gpu_uuid, '--max-seconds', args.generation_seconds,
             '--save-margin-seconds', args.save_margin_seconds])
        return bank

    def score(arm, epoch, split, source, bank):
        output = campaign / 'assessment' / split / arm / f'epoch_{epoch:03d}'
        timing = campaign / 'assessment' / 'deployment' / arm / f'epoch_{epoch:03d}.json'
        add(f'{split}_{arm}_{epoch:03d}_score', 'cpu',
            ['score', '--source-bank', source, '--child-bank', bank, '--core-run', args.core_run,
             '--contract', artifacts['contract'], '--event', artifacts['event'],
             '--reserved-task', artifacts['reserved'], '--deployment-timing', timing, '--output', output])

    if args.phase == 'train':
        for arm, run in arms:
            for epoch in EPOCHS:
                bank = generate(arm, run, epoch, 'train', source_train)
                timing = campaign / 'assessment' / 'deployment' / arm / f'epoch_{epoch:03d}.json'
                add(f'train_{arm}_{epoch:03d}_deployment', 'gpu',
                    ['profile-deployment', *closeout, '--arm', arm, '--run', run, '--epoch', epoch,
                     '--gpu-uuid', args.gpu_uuid, '--max-seconds', args.deployment_seconds,
                     '--save-margin-seconds', args.save_margin_seconds, '--warmup', args.warmup, '--output', timing])
                score(arm, epoch, 'train', source_train, bank)
        add('freeze_train_nominations', 'cpu',
            ['freeze-nominations', '--family', args.family, '--full-run', args.full_run,
             '--residual-run', args.residual_run])
        return jobs
    add('freeze_source_reserved_normalizer', 'cpu',
        ['freeze-reserved-normalizer', '--family', args.family, '--source-bank', source_train,
         '--reserved-task', artifacts['reserved']])
    add('validation_source_generate', 'gpu',
        ['generate', *closeout, '--kind', 'source', '--split', 'validation',
         '--arm', arms[0][0], '--run', args.full_run, '--output', source_validation,
         '--gpu-uuid', args.gpu_uuid, '--max-seconds', args.generation_seconds,
         '--save-margin-seconds', args.save_margin_seconds])
    for arm, run in arms:
        for epoch in EPOCHS:
            bank = generate(arm, run, epoch, 'validation', source_validation)
            score(arm, epoch, 'validation', source_validation, bank)
    return jobs


def execute_jobs(jobs, work, env, children, *, runner=subprocess.run):
    """Wait for each bounded child and stop immediately on failure."""
    for job in jobs:
        started = time.time()
        stdout, stderr = work / f"{job['name']}.stdout", work / f"{job['name']}.stderr"
        code, launch_error = None, None
        with stdout.open('x') as out, stderr.open('x') as err:
            try:
                result = runner(job['command'], stdout=out, stderr=err, env=env, check=False)
                code = result.returncode
            except OSError as exc:
                launch_error = str(exc)
        ended = time.time()
        record = {'schema': 'car_round2_assessment_sequence_child_v1', 'name': job['name'],
                  'command': job['command'], 'kind': job['kind'], 'returncode': code,
                  'launch_error': launch_error, 'start_unix': started, 'end_unix': ended,
                  'wall_seconds': ended-started, 'stdout': str(stdout), 'stderr': str(stderr),
                  'independent_cpu': job['kind'] == 'cpu',
                  'wall_accounting': ('CPU child wall includes its scoring receipt; do not add both'
                                      if job['kind'] == 'cpu' else 'encloses the GPU lease; do not add both')}
        path = work / f"{job['name']}.json"
        immutable_json(path, record)
        children.append({'path': str(path), 'sha256': digest(path)})
        if code != 0:
            raise RuntimeError(f"{job['name']} failed with exit code {code}; see {stderr}")


def sequence(args):
    campaign = args.campaign.resolve(strict=True)
    args.campaign = campaign
    output = args.output.resolve()
    work = output.parent / f'{output.stem}_files'
    if campaign not in output.parents or output.exists() or work.exists():
        raise ValueError('sequence receipt and work directory must be new and campaign-owned')
    if (not all(math.isfinite(v) and v > 0 for v in
                (args.generation_seconds, args.deployment_seconds, args.save_margin_seconds))
            or min(args.generation_seconds, args.deployment_seconds) <= args.save_margin_seconds
            or args.warmup < 1):
        raise ValueError('bounded generation/deployment jobs must include a positive save margin and warmup')
    driver_path = CODE_ROOT / 'scripts/posttraining/run_coherence_adaptive_repair_round2.py'
    spec = importlib.util.spec_from_file_location('round2_driver', driver_path)
    driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
    capsule = driver.check_code_capsule(campaign)
    assess_path = CODE_ROOT / 'scripts/posttraining/assess_car_round2_checkpoints.py'
    spec = importlib.util.spec_from_file_location('round2_assessment', assess_path)
    assess = importlib.util.module_from_spec(spec); spec.loader.exec_module(assess)
    if args.final_assessment:
        assess.assessment_budget_limit(campaign, True)
    if args.phase == 'validation':
        assess.global_train_choice_barrier(campaign)
    formal_path = campaign / 'configs/formal_freeze.json'
    formal = json.loads(formal_path.read_text())
    bindings = formal['manifest']['artifacts']
    artifacts = {name: driver.check_binding(bindings[key], campaign) for name, key in
                 [('contract', f'assessment_contract_{args.family}'), ('event', 'development_event'),
                  ('reserved', 'reserved_task')]}
    for key in ('full_run', 'residual_run', 'core_run', 'train_source_bank'):
        value = getattr(args, key).resolve(strict=True)
        if campaign not in value.parents:
            raise ValueError(f'{key} must be campaign-owned')
        setattr(args, key, value)
    if args.full_run == args.residual_run:
        raise ValueError('distinct mature arm runs required')
    completed = assess.completed_frozen_mature_runs(campaign)
    prefix = 'FFM' if args.family == 'ffm' else 'Senseiver'
    for suffix, run in (('Uniform_FullModel', args.full_run), ('Residual_Repair', args.residual_run)):
        arm = f'{prefix}_Selected_Bank_{suffix}'
        if run not in completed[arm]:
            raise ValueError(f'{arm} requires its supplied frozen mature run completed at 12000 updates')
    args.keops_cache.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid, KEOPS_CACHE_FOLDER=str(args.keops_cache.resolve()))
    jobs = build_jobs(args, artifacts)
    work.mkdir(parents=True)
    start, children, failure = time.time(), [], None
    immutable_json(work / 'group_launch.json', {'schema':'car_round2_assessment_sequence_launch_v1',
                   'family':args.family, 'phase':args.phase, 'final_assessment':args.final_assessment,
                   'gpu_uuid':args.gpu_uuid, 'formal_freeze_sha256':digest(formal_path),
                   'source_bank_receipt_sha256':digest(args.train_source_bank / 'receipt.json'),
                   'full_run':str(args.full_run), 'residual_run':str(args.residual_run),
                   'core_run':str(args.core_run), 'train_source_bank':str(args.train_source_bank),
                   'generation_seconds':args.generation_seconds, 'deployment_seconds':args.deployment_seconds,
                   'save_margin_seconds':args.save_margin_seconds, 'script_sha256':digest(Path(__file__)),
                   'capsule_sha256':capsule, 'jobs':len(jobs), 'start_unix':start,
                   'wall_accounting':'group wall encloses child costs; do not add both'})
    try:
        execute_jobs(jobs, work, env, children)
    except Exception as exc:  # noqa: BLE001 - preserve the failed immutable attempt and all child logs.
        failure = {'type':type(exc).__name__, 'message':str(exc)}
    ended = time.time()
    immutable_json(output, {'schema':'car_round2_assessment_sequence_v1',
                   'status':'failed' if failure else 'completed', 'family':args.family, 'phase':args.phase,
                   'start_unix':start, 'end_unix':ended, 'wall_seconds':ended-start,
                   'launch_receipt':str(work / 'group_launch.json'),
                   'launch_sha256':digest(work / 'group_launch.json'), 'children':children, 'failure':failure,
                   'wall_accounting':'group wall encloses child costs; do not add both'})
    if failure:
        raise RuntimeError(failure['message'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--family', choices=('ffm', 'senseiver'), required=True)
    p.add_argument('--phase', choices=('train', 'validation'), required=True)
    p.add_argument('--final-assessment', action='store_true', help='use closeout reserve only after all four mature arms complete')
    p.add_argument('--full-run', type=Path, required=True)
    p.add_argument('--residual-run', type=Path, required=True)
    p.add_argument('--core-run', type=Path, required=True)
    p.add_argument('--train-source-bank', type=Path, required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--keops-cache', type=Path, required=True)
    p.add_argument('--generation-seconds', type=float, required=True)
    p.add_argument('--deployment-seconds', type=float, required=True)
    p.add_argument('--save-margin-seconds', type=float, required=True)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    sequence(p.parse_args())


if __name__ == '__main__':
    main()
