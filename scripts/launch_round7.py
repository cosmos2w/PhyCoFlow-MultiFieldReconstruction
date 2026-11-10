"""Launch one owned Round-7 stage after checking its physical GPU is idle."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=(0, 2), default=0)
    parser.add_argument('stage')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    campaign = root / 'cases/turbulent_combustion/runs/coherence_compute_round7'
    if not (campaign / 'artifacts/storage_receipt.json').is_file():
        raise RuntimeError('Round-7 case-owned storage has not been prepared')
    devices = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    applications = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True)
    device = next((line.split(',') for line in devices.splitlines()
                   if int(line.split(',')[0].strip()) == args.gpu), None)
    if device is None:
        raise RuntimeError(f'Physical GPU{args.gpu} is unavailable')
    uuid = device[1].strip()
    occupied = any(line.split(',')[0].strip() == uuid for line in applications.splitlines())
    if occupied or int(device[2].strip()) >= 512:
        raise RuntimeError(f'Physical GPU{args.gpu} is occupied; preserve existing jobs')
    source = root / 'scripts/run_round7.py'
    content = source.read_bytes()
    compile(content, str(source), 'exec')
    digest = hashlib.sha256(content).hexdigest()
    script = campaign / 'artifacts/execution_scripts' / (digest + '.py')
    script.parent.mkdir(parents=True, exist_ok=True)
    if script.exists():
        if script.read_bytes() != content:
            raise RuntimeError('Execution-script hash collision')
    else:
        script.write_bytes(content)
    cache = campaign / f'caches/gpu{args.gpu}'
    build = f'{platform.system()}_{platform.node()}_{platform.release()}_p{platform.python_version()}'
    for name in ('matplotlib', 'torch_extensions', 'keops/' + build,
                 'keops/' + build + '_CUDA_VISIBLE_DEVICES_' + uuid):
        (cache / name).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=uuid, ROUND7_REPO_ROOT=str(root),
               ROUND7_PHYSICAL_GPU_UUID=uuid, ROUND7_PHYSICAL_GPU_INDEX=str(args.gpu),
               ROUND7_GPU_UUID=uuid,
               KEOPS_CACHE_FOLDER=str(cache / 'keops'), MPLCONFIGDIR=str(cache / 'matplotlib'),
               TORCH_EXTENSIONS_DIR=str(cache / 'torch_extensions'),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONPATH=str(root) + ':' + str(root / 'src'),
               PYTHONUNBUFFERED='1')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    name = '_'.join((args.stage, *[x for x in args.arguments if not x.startswith('-')]))
    name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in name)[:100]
    log = campaign / 'logs' / f'{name}_{stamp}.log'
    with log.open('x') as output:
        process = subprocess.Popen([sys.executable, str(script), args.stage, *args.arguments],
                                   cwd=root, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(schema='round7_launch_receipt_v1', stage=args.stage, arguments=args.arguments,
                   pid=process.pid, launched_utc=stamp, gpu_index=args.gpu, gpu_uuid=uuid,
                   gpu_preflight=devices, compute_apps_preflight=applications, log=str(log),
                   execution_script=str(script), execution_script_sha256=digest,
                   source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                                         text=True).strip(),
                   initial_status=process.poll())
    path = campaign / 'artifacts' / f'{name}_launch_{stamp}.json'
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'receipt': str(path), **receipt}, indent=2))


if __name__ == '__main__':
    main()
