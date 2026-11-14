"""Round 1 campaign accounting and immutable, deterministic panel manifests.

All costs are additive device occupancy, not elapsed campaign wall time.  Active
leases are conservatively charged through the instant of each budget check.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import random
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

CAP_SECONDS = 80 * 3600
RESERVE_SECONDS = 6 * 3600
EPOCHS = (0, 100, 250, 500, 750, 1000)


def load_execution_runtime(campaign: Path):
    """Load current operational device choices, separate from frozen science."""
    path = os.environ.get('CAR_EXECUTION_RUNTIME')
    if not path:
        raise ValueError('CAR_EXECUTION_RUNTIME is required for GPU execution')
    runtime = json.loads(Path(path).read_text())
    if runtime.get('schema') != 'car_execution_runtime_v1':
        raise ValueError('invalid execution runtime schema')
    selected = runtime.get('gpu_uuid')
    excluded = runtime.get('excluded_gpu_uuids')
    paused = runtime.get('paused_source_families')
    if (not isinstance(selected, str) or not selected.startswith('GPU-')
            or not isinstance(excluded, list) or not isinstance(paused, list)):
        raise ValueError('runtime requires selected GPU and exclusion/pause lists')
    return runtime


def enforce_execution_constraints(campaign: Path, gpu_uuid: str, source_family=None):
    runtime = load_execution_runtime(campaign)
    if gpu_uuid != runtime['gpu_uuid'] or gpu_uuid in runtime['excluded_gpu_uuids']:
        raise ValueError('GPU differs from selection or is excluded by current runtime')
    if source_family in runtime['paused_source_families']:
        raise ValueError('backend paused by owner; explicit continuation required')


def _content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _capsule_inventory(root):
    root = Path(root).resolve()
    receipt = json.loads((root / 'execution_receipt.json').read_text())
    files = receipt.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('capsule inventory missing')
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if receipt.get('schema') != 'car_execution_code_v1' or receipt.get('sha256') != identity or root.name != identity:
        raise ValueError('capsule identity mismatch')
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != expected:
            raise ValueError(f'capsule file changed: {name}')
    return files


def load_assessment_formal(campaign: Path):
    """Verify an optional device-only amendment and derive effective metadata."""
    campaign = Path(campaign)
    freeze_path = campaign / 'formal_freeze.json'
    formal = json.loads(freeze_path.read_text())
    if formal.get('schema') != 'car_freeze_receipt_v1' or formal.get('manifest', {}).get('status') != 'frozen':
        raise ValueError('valid frozen formal metadata required')
    manifest = formal['manifest']
    amendment_path = os.environ.get('CAR_ASSESSMENT_AMENDMENT')
    if not amendment_path:
        formal['manifest'] = {**manifest, 'assessment_code_root': manifest['code_root']}
        return formal
    amendment = json.loads(Path(amendment_path).read_text())
    body = {k:v for k,v in amendment.items() if k != 'sha256'}
    if amendment.get('schema') != 'car_device_amendment_v1' or amendment.get('sha256') != _content_hash(body):
        raise ValueError('invalid assessment amendment identity')
    if amendment.get('formal_freeze_sha256') != digest(freeze_path):
        raise ValueError('amendment frozen formal identity differs')
    old_root = Path(manifest['code_root']).resolve()
    new_root = Path(amendment['assessment_code_root']).resolve()
    if (Path(amendment['training_code_root']).resolve() != old_root
            or amendment.get('original_assessment_driver_sha256') != manifest['assessment_driver_sha256']
            or amendment.get('artifact_namespace') != 'assessment/device_v1'):
        raise ValueError('amendment source or namespace differs')
    old, new = _capsule_inventory(old_root), _capsule_inventory(new_root)
    allowed = {'src/phycoflow_reconstruction/evaluation/adaptive_campaign.py', 'scripts/posttraining/assess_car_checkpoints.py'}
    if set(old) != set(new) or {k for k in old if old[k] != new[k]} != allowed:
        raise ValueError('amendment changes files outside permitted device-only scope')
    if (amendment.get('training_capsule_sha256') != old_root.name
            or amendment.get('assessment_capsule_sha256') != new_root.name
            or old['scripts/posttraining/assess_car_checkpoints.py'] != manifest['assessment_driver_sha256']
            or new['scripts/posttraining/assess_car_checkpoints.py'] != amendment['assessment_driver_sha256']
            or amendment.get('changed_files') != {k:{'old':old[k], 'new':new[k]} for k in sorted(allowed)}):
        raise ValueError('amendment capsule or driver identity differs')
    formal['manifest'] = {**manifest, 'assessment_code_root': str(new_root),
                          'assessment_driver_sha256': amendment['assessment_driver_sha256'],
                          'assessment_amendment_sha256': amendment['sha256']}
    return formal


def assessment_root(campaign: Path):
    formal = load_assessment_formal(campaign)
    suffix = 'assessment/device_v1' if formal['manifest'].get('assessment_amendment_sha256') else 'assessment'
    return Path(campaign) / suffix


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def panel(ids, split: str, seed: int):
    if split not in ('train', 'validation') or not ids or len(set(ids)) != len(ids):
        raise ValueError('unique nonempty train or validation IDs required')
    ids = sorted(ids)
    selected = random.Random(seed).sample(ids, max(1, math.ceil(len(ids) * .1)))
    return {'schema': 'car_panel_v1', 'split': split, 'population': len(ids),
            'seed': seed, 'sample_ids': selected, 'sensor_count': 256, 'members': 4}


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.campaign-')
    try:
        with os.fdopen(fd, 'w') as out:
            json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
            out.write('\n')
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def immutable_json(path: Path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'immutable identity differs: {path}')
    else:
        atomic_json(path, value)


def freeze_execution_code(repository: Path, output: Path, drivers):
    """Preserve the Python execution tree under a content-derived identity."""
    repository, output = repository.resolve(), output.resolve()
    paths = sorted({*repository.joinpath('src').rglob('*.py'),
                    *(repository / path for path in drivers)})
    inventory = {str(path.relative_to(repository)): digest(path) for path in paths}
    identity = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
    target = output / identity
    for path in paths:
        destination = target / path.relative_to(repository)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if digest(destination) != inventory[str(path.relative_to(repository))]:
                raise ValueError(f'frozen execution file changed: {destination}')
        else:
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != inventory[str(path.relative_to(repository))]:
                raise ValueError(f'execution source changed during snapshot: {path}')
            destination.write_bytes(content)
            destination.chmod(0o444)
    immutable_json(target / 'execution_receipt.json', {
        'schema': 'car_execution_code_v1', 'sha256': identity, 'files': inventory,
    })
    return target


@contextmanager
def locked_ledger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text()) if path.exists() else {
            'schema': 'car_gpu_ledger_v1', 'closed': [], 'active': {}}
        yield state
        atomic_json(path, state)


def spent(state, now=None):
    now = time.time() if now is None else now
    total = sum(float(entry['seconds']) for entry in state['closed'])
    total += sum(max(0., now - float(entry['start_unix'])) for entry in state['active'].values())
    return total


def forecast(state, update_seconds, callback_seconds, remaining_seconds, *, now=None):
    values = (update_seconds, callback_seconds, remaining_seconds)
    if any(not math.isfinite(x) or x < 0 for x in values) or update_seconds == 0:
        raise ValueError('positive measured update and finite nonnegative remaining costs required')
    already = spent(state, now)
    projected = already + 26000 * update_seconds + callback_seconds + remaining_seconds
    return {'spent_seconds': already, 'projected_seconds': projected,
            'reserve_seconds': RESERVE_SECONDS, 'cap_seconds': CAP_SECONDS,
            'fits': projected + RESERVE_SECONDS <= CAP_SECONDS}


def begin(state, token, uuid, phase, *, now=None):
    if not token or not uuid.startswith('GPU-') or not phase:
        raise ValueError('token, physical GPU UUID and phase required')
    if token in state['active'] or any(x['token'] == token for x in state['closed']):
        raise ValueError('lease token already used')
    if any(x['gpu_uuid'] == uuid for x in state['active'].values()):
        raise ValueError('device already leased')
    if spent(state, now) >= CAP_SECONDS - (0 if phase == 'closeout' else RESERVE_SECONDS):
        raise ValueError('budget exhausted')
    state['active'][token] = {'gpu_uuid': uuid, 'phase': phase,
                              'start_unix': time.time() if now is None else now}


def end(state, token, *, now=None):
    if token not in state['active']:
        raise ValueError('unknown active lease')
    entry = state['active'].pop(token)
    end_time = time.time() if now is None else now
    if end_time < entry['start_unix']:
        state['active'][token] = entry
        raise ValueError('clock moved backwards')
    state['closed'].append({**entry, 'token': token, 'end_unix': end_time,
                            'seconds': end_time - entry['start_unix']})


SOURCES = {
    'ffm': ('tc_gl_rbf_cq_coverage_5000ep/20261029T210910Z_cf9c2276',
            '2d1da713538f6e9613560f3a7eb4a9df2a95d1ad1f56dd11a1a8a4f9ef65050c', 5000, 315000),
    'edm': ('tc_diffusion_pde_5000ep/20261027T222333Z_187a0271',
            '0bcaa0cf7d1082536fea8fce698646018912d1e8b9f76004c9480ad4a9c3f753', 1400, 88200),
}


def source_ownership(run: Path):
    """Read process cmdline/cwd and writable descriptors referencing this run."""
    run = run.resolve()
    hits = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            cwd = (proc / 'cwd').resolve()
        except (OSError, PermissionError):
            continue
        matched = str(run) in cmd or cwd == run or run in cwd.parents
        write_fds = []
        try:
            for fd in (proc / 'fd').iterdir():
                try:
                    dest = fd.resolve()
                    if dest == run or run in dest.parents:
                        info = (proc / 'fdinfo' / fd.name).read_text()
                        flags = int(next(x.split()[1] for x in info.splitlines() if x.startswith('flags:')), 8)
                        if flags & os.O_ACCMODE in (os.O_WRONLY, os.O_RDWR):
                            write_fds.append(str(dest))
                except (OSError, StopIteration, ValueError):
                    continue
        except (OSError, PermissionError):
            pass
        if matched or write_fds:
            hits.append({'pid': int(proc.name), 'command': cmd, 'cwd': str(cwd), 'write_fds': write_fds})
    return hits


def snapshot_source(run: Path, output: Path, expected_hash: str, expected_epoch: int, expected_step: int):
    """Copy a pinned source after a bounded writer check; never mutate original."""
    import shutil

    import torch
    run = run.resolve()
    output = output.resolve()
    if run == output or run in output.parents:
        raise ValueError('snapshot must be outside original source')
    owners = source_ownership(run)
    if owners:
        raise ValueError(f'source ownership is not clear: {owners}')
    source = run / 'checkpoints' / 'last.pt'
    before = source.stat()
    if digest(source) != expected_hash:
        raise ValueError('pinned source checkpoint hash mismatch')
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    epoch = checkpoint.get('epoch', checkpoint.get('completed_epoch'))
    step = checkpoint.get('global_step', checkpoint.get('step'))
    if epoch is None:
        # Historical checkpoints record steps, not epochs; 63 steps/epoch is
        # fixed by the original run's dataset/batch convention.
        if expected_step % expected_epoch:
            raise ValueError('nonintegral historical steps per epoch')
        epoch = int(step) // (expected_step // expected_epoch)
    if int(epoch) != expected_epoch or int(step) != expected_step:
        raise ValueError(f'pinned checkpoint metadata mismatch: epoch={epoch}, step={step}')
    if 'model' not in checkpoint or 'model_ema' not in checkpoint.get('training_aux_state', {}):
        raise ValueError('checkpoint lacks model or EMA auxiliary state')
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'checkpoints' / 'last.pt'
    target.parent.mkdir(exist_ok=True)
    for item in ('resolved_config.yaml', 'run_manifest.json', 'status.json'):
        if not (run / item).is_file():
            raise FileNotFoundError(run / item)
    if target.exists():
        if digest(target) != expected_hash:
            raise ValueError('existing snapshot differs')
    else:
        tmp = target.with_name('.last.pt.copy')
        try:
            shutil.copyfile(source, tmp)
            if digest(tmp) != expected_hash:
                raise ValueError('snapshot copy hash mismatch')
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
    after = source.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns) or digest(source) != expected_hash:
        raise ValueError('source changed during snapshot')
    for item in ('resolved_config.yaml', 'run_manifest.json', 'status.json'):
        src, dst = run / item, output / item
        if dst.exists() and digest(src) != digest(dst):
            raise ValueError(f'existing snapshot metadata differs: {item}')
        if not dst.exists():
            shutil.copyfile(src, dst)
    receipt = {'schema': 'frozen_source_snapshot.v1', 'source_run': str(run),
               'checkpoint_sha256': expected_hash, 'epoch': expected_epoch, 'global_step': expected_step,
               'epoch_basis': 'global_step / 63; checkpoint has no epoch field',
               'model_name': checkpoint.get('model_name'), 'checkpoint_config_sha256': checkpoint.get('config_sha256'),
               'ownership_scan': {'method': 'proc cwd, command and writable open descriptors', 'matches': owners},
               'metadata_sha256': {item: digest(output / item) for item in ('resolved_config.yaml', 'run_manifest.json', 'status.json')}}
    immutable_json(output / 'snapshot_receipt.json', receipt)
    for item in (target, output / 'resolved_config.yaml', output / 'run_manifest.json', output / 'status.json', output / 'snapshot_receipt.json'):
        item.chmod(0o444)
    return receipt


def budget_stop(path: Path, *, safety_seconds: float):
    """Read charged cost at an update boundary; reserve time for safe save."""
    if not math.isfinite(safety_seconds) or safety_seconds <= 0:
        raise ValueError('positive measured save/overrun margin required')
    with path.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        state = json.loads(path.read_text())
        return spent(state) + safety_seconds >= CAP_SECONDS - RESERVE_SECONDS
