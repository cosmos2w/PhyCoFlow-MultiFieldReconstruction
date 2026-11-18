#!/usr/bin/env python3
"""Continue the four-arm assessment in phases without new latency profiling."""
import time

START = time.time()
CPU_START = time.process_time()
import argparse
import concurrent.futures
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import threading
from pathlib import Path

EPOCHS = (0, 100, 250, 500, 750, 1000)
ARMS = tuple(f'{family}_Selected_Bank_{method}' for family in ('FFM', 'Senseiver')
             for method in ('Residual_Repair', 'Uniform_FullModel'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def completed(path, filename):
    receipt = path / filename
    if not receipt.is_file():
        return False
    return json.loads(receipt.read_text()).get('status') == 'completed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--assessment-script', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True, help='JSON mapping each of the four arm names to its terminal run path')
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--ffm-generation-seconds', type=float, required=True)
    parser.add_argument('--senseiver-generation-seconds', type=float, required=True)
    parser.add_argument('--save-margin-seconds', type=float, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    script = args.assessment_script.resolve(strict=True)
    if campaign not in script.parents or '/operations/code/' not in str(script):
        raise ValueError('assessment script must be in the campaign execution snapshot')
    if not args.gpu_uuid.startswith('GPU-') or any(not math.isfinite(x) or x <= args.save_margin_seconds
            for x in (args.ffm_generation_seconds, args.senseiver_generation_seconds)) or args.save_margin_seconds <= 0:
        raise ValueError('runtime GPU and bounded generation with save margin required')
    ops = campaign / 'operations'
    receipt_path = ops / 'gpu0_assessment_continuation_v1.json'
    work = ops / 'gpu0_assessment_continuation_v1_files'
    if receipt_path.exists() or work.exists():
        raise FileExistsError('continuation attempt already exists')
    work.mkdir()
    result = {'schema': 'car_round2_independent_cpu_job_v1', 'status': 'failed', 'start_unix': START,
              'scope': 'Phasewise GPU generation followed by at most two CPU scorers; no new deployment profiling',
              'assessment_script': str(script), 'assessment_script_sha256': digest(script),
              'gpu_uuid': args.gpu_uuid, 'children': [], 'banks': {}}
    child_wall = 0.
    child_lock = threading.Lock()
    child_index = itertools.count()
    try:
        spec = importlib.util.spec_from_file_location('continuation_assessment', script)
        assess = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assess)
        supplied = json.loads(args.runs.read_text())
        if set(supplied) != set(ARMS):
            raise ValueError('exactly the four named mature runs are required')
        terminal = assess.completed_frozen_mature_runs(campaign)
        runs = {arm: Path(supplied[arm]).resolve(strict=True) for arm in ARMS}
        if any(runs[arm] not in terminal[arm] for arm in ARMS):
            raise ValueError('all four runs must be receipt-bound terminal 12000-update frozen runs')
        formal = json.loads((campaign / 'configs/formal_freeze.json').read_text())['manifest']
        setup = json.loads((campaign / 'configs/topology_stable_setup_v1/setup_references.json').read_text())
        base = [sys.executable, str(script), '--campaign', str(campaign)]
        cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                       KEOPS_CACHE_FOLDER=str(campaign / 'cache/keops/cpu_gpu0_continuation'))
        build = '_'.join(platform.uname()[:3]) + f'_p{sys.version.split()[0]}'
        (Path(cpu_env['KEOPS_CACHE_FOLDER']) / f'{build}_CUDA_VISIBLE_DEVICES_').mkdir(parents=True, exist_ok=True)
        gpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       KEOPS_CACHE_FOLDER=str(campaign / 'cache/keops' / args.gpu_uuid))

        def run(kind, command, *, account_wait=True):
            nonlocal child_wall
            if 'profile-deployment' in command:
                raise ValueError('new deployment profiling is prohibited')
            with child_lock:
                index = next(child_index)
            stdout, stderr = work / f'{index:03d}.stdout', work / f'{index:03d}.stderr'
            begin = time.time()
            try:
                with stdout.open('x') as out, stderr.open('x') as err:
                    child = subprocess.run(base + [str(v) for v in command], env=gpu_env if kind == 'gpu' else cpu_env,
                                           stdout=out, stderr=err, check=False)
            finally:
                end = time.time()
                if account_wait:
                    child_wall += end - begin
            with child_lock:
                result['children'].append({'kind': kind, 'command': [str(v) for v in command], 'start_unix': begin,
                                           'end_unix': end, 'wall_seconds': end-begin, 'returncode': child.returncode,
                                           'stdout': str(stdout), 'stderr': str(stderr)})
            if child.returncode:
                raise RuntimeError(f'{kind} child failed; see {stderr}')

        def refs(family):
            return (Path(setup['sources'][family]['source_fit']['run']).resolve(strict=True),
                    Path(formal['artifacts'][f'assessment_contract_{family}']['path']).resolve(strict=True),
                    Path(formal['artifacts']['development_event']['path']).resolve(strict=True),
                    Path(formal['artifacts']['reserved_task']['path']).resolve(strict=True))

        def score(split, arm, epoch, bank, source, family):
            canonical = campaign / 'assessment' / split / arm / f'epoch_{epoch:03d}'
            name = 'nomination.json' if split == 'train' else 'assessment.json'
            if completed(canonical, 'cpu_receipt.json') and (canonical / name).is_file():
                return canonical / name
            if canonical.exists():
                raise FileExistsError(f'existing incomplete score output must be preserved: {canonical}')
            output = canonical
            core, contract, event, task = refs(family)
            command = ['score', '--source-bank', source, '--child-bank', bank, '--core-run', core,
                       '--contract', contract, '--event', event, '--reserved-task', task,
                       '--allow-missing-latency', '--output', output]
            timing = campaign / 'assessment/deployment' / arm / f'epoch_{epoch:03d}.json'
            if timing.is_file() and json.loads(timing.read_text()).get('status') == 'completed':
                command.extend(['--deployment-timing', timing])
            run('cpu', command, account_wait=False)
            return output / name

        def bank_for(split, arm, epoch, source, family):
            canonical = campaign / 'assessment/banks' / split / arm / f'epoch_{epoch:03d}'
            if completed(canonical, 'receipt.json'):
                return canonical
            if canonical.exists():
                raise FileExistsError(f'existing incomplete bank output must be preserved: {canonical}')
            output = canonical
            limit = getattr(args, f'{family}_generation_seconds')
            run('gpu', ['generate', '--final-assessment', '--kind', 'child', '--split', split, '--arm', arm,
                        '--run', runs[arm], '--epoch', epoch, '--source-bank', source, '--output', output,
                        '--gpu-uuid', args.gpu_uuid, '--max-seconds', limit, '--save-margin-seconds', args.save_margin_seconds])
            return output

        # Existing scorer drain and the specifically assigned epoch-100 smoke
        # must finish before this continuation; never duplicate their outputs.
        prior = [('FFM_Selected_Bank_Residual_Repair', epoch) for epoch in EPOCHS]
        prior += [('FFM_Selected_Bank_Uniform_FullModel', 0), ('FFM_Selected_Bank_Uniform_FullModel', 100)]
        for arm, epoch in prior:
            path = campaign / 'assessment/train' / arm / f'epoch_{epoch:03d}'
            if not completed(path, 'cpu_receipt.json') or not (path / 'nomination.json').is_file():
                raise ValueError(f'existing TRAIN score has not completed: {path}')
        for split in ('train', 'validation'):
            score_tasks = []
            if split == 'validation':
                panel = campaign / 'configs/panels/validation.json'
                if not panel.exists():
                    run('cpu', ['panel', '--split', 'validation', '--core-run', refs('ffm')[0]])
            for family, prefix in (('ffm', 'FFM'), ('senseiver', 'Senseiver')):
                full = f'{prefix}_Selected_Bank_Uniform_FullModel'
                source = campaign / 'assessment' / f'source_{split}_{family}_v1'
                if split == 'validation' and not completed(source, 'receipt.json'):
                    if source.exists():
                        raise FileExistsError(f'existing incomplete SOURCE bank must be preserved: {source}')
                    run('gpu', ['generate', '--final-assessment', '--kind', 'source', '--split', split,
                                '--arm', full, '--run', runs[full], '--output', source, '--gpu-uuid', args.gpu_uuid,
                                '--max-seconds', getattr(args, f'{family}_generation_seconds'), '--save-margin-seconds', args.save_margin_seconds])
                for arm in (f'{prefix}_Selected_Bank_Residual_Repair', full):
                    for epoch in EPOCHS:
                        bank = bank_for(split, arm, epoch, source, family)
                        result['banks'][f'{split}/{arm}/{epoch}'] = str(bank)
                        score_tasks.append((split, arm, epoch, bank, source, family))
            # All GPU generation in this phase has ended. Account the enclosing
            # two-worker interval once; each scorer owns its own CPU receipt.
            started = time.time()
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
            pending = set()
            try:
                tasks = iter(score_tasks)
                for _ in range(2):
                    task = next(tasks, None)
                    if task is not None:
                        pending.add(pool.submit(score, *task))
                while pending:
                    done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        future.result()
                    for _ in done:
                        task = next(tasks, None)
                        if task is not None:
                            pending.add(pool.submit(score, *task))
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
                child_wall += time.time() - started
            if split == 'train':
                for family, prefix in (('ffm', 'FFM'), ('senseiver', 'Senseiver')):
                    run('cpu', ['freeze-nominations', '--family', family,
                                '--full-run', runs[f'{prefix}_Selected_Bank_Uniform_FullModel'],
                                '--residual-run', runs[f'{prefix}_Selected_Bank_Residual_Repair'],
                                '--allow-missing-latency'])
                assess.global_train_choice_barrier(campaign)
                for family in ('ffm', 'senseiver'):
                    normalizer = campaign / 'assessment/train' / f'reserved_normalizer_{family}.json'
                    if not normalizer.exists():
                        run('cpu', ['freeze-reserved-normalizer', '--family', family,
                                    '--source-bank', campaign / 'assessment' / f'source_train_{family}_v1',
                                    '--reserved-task', refs(family)[3]])
                    assess.validate_reserved_freeze(campaign, family)
        result.update(status='completed')
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        end = time.time()
        result.update(end_unix=end, wall_seconds=end-START, blocked_wait_seconds=0.,
                      nested_child_wall_seconds=child_wall, independent_cpu=True,
                      active_wall_seconds=max(0., end-START-child_wall), process_cpu_seconds=time.process_time()-CPU_START,
                      accounting='Child subprocess waits are nonadditive; their GPU leases and CPU job receipts are charged separately.')
        publish(receipt_path, result)


if __name__ == '__main__':
    main()
