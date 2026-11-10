#!/usr/bin/env python3
"""Round 7 campaign stages. Requires the frozen case-owned contract and launcher."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import time
import uuid

ROOT = Path(os.environ.get('ROUND7_REPO_ROOT', Path(__file__).resolve().parents[1]))
CASE = ROOT / 'cases/turbulent_combustion'
CAMPAIGN = CASE / 'runs/coherence_compute_round7'
ART = CAMPAIGN / 'artifacts'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def same_checkpoint_state(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(same_checkpoint_state(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(same_checkpoint_state(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def exact_restoration(saved, live):
    """Check actual loaded state; independent CUDA trajectories may differ."""
    keys = ('global_step', 'model', 'optimizer', 'external_policy', 'query_coverage',
            'rng_state', 'training_aux_state', 'family_states', 'family_configs',
            'normalization', 'family_scales', 'source_hashes', 'config_sha256',
            'coherence_calibration_sha256')
    for key in keys:
        if key not in saved or key not in live or not same_checkpoint_state(saved[key], live[key]):
            raise ValueError('exact checkpoint restoration failed for ' + key)
    return {'global_step': saved['global_step'], 'exact_keys': list(keys),
            'exact_loaded_state': True, 'independent_trajectory_bitwise_equality_claimed': False}


def verify_endpoint_binding(path, *, model_state_sha256, manifest_sha256, observer_identities):
    """Validate an existing scientific endpoint without rewriting any of its files."""
    binding = read(path)
    if binding['model_state_sha256'] != model_state_sha256:
        raise ValueError('existing endpoint model state differs')
    for label in ('checkpoint', 'train_reductions', 'validation_reductions', 'native_receipt'):
        if digest(binding[label]) != binding[label + '_sha256']:
            raise ValueError('existing endpoint artifact bytes differ: ' + label)
    for label in ('train_reductions', 'validation_reductions'):
        payload = read(binding[label])
        if payload.get('manifest_sha256') != manifest_sha256 or payload.get('observer_identities') != observer_identities:
            raise ValueError('existing endpoint observer or cohort differs')
    return binding


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def ledger_transaction():
    ART.mkdir(parents=True, exist_ok=True)
    with (ART / 'budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = ART / 'budget_ledger.json'
        ledger = read(path) if path.exists() else {'version': 2, 'attempt_count': 0,
            'qualification_attempts': 0, 'formal_unique': 0, 'formal_replays': 0,
            'arm_last_step': {}, 'intervals': []}
        yield ledger
        write(path, ledger)


def total_seconds(intervals, now=None):
    from phycoflow_reconstruction.evaluation.round7_decision import gpu_seconds
    now = time.time() if now is None else now
    return gpu_seconds([(x['device_uuid'], x['start'], x.get('end', now)) for x in intervals])


def charge_attempt(phase, arm, step):
    with ledger_transaction() as ledger:
        if total_seconds(ledger['intervals']) >= 36 * 3600:
            raise RuntimeError('36 occupied GPU-hour limit reached')
        if ledger['attempt_count'] >= 46214:
            raise RuntimeError('absolute optimizer attempt cap reached')
        if phase == 'qualification':
            if ledger['qualification_attempts'] >= 64:
                raise RuntimeError('qualification attempt cap reached')
            ledger['qualification_attempts'] += 1
        elif phase == 'formal':
            last = ledger['arm_last_step'].get(arm, -1)
            if step > last + 1:
                raise ValueError('formal update positions must be contiguous')
            repeated = step <= last
            if repeated and ledger['formal_replays'] >= 650:
                raise RuntimeError('shared replay allowance exhausted')
            if not repeated and ledger['formal_unique'] >= 45500:
                raise RuntimeError('unique planned position cap reached')
            if repeated:
                ledger['formal_replays'] += 1
            else:
                ledger['formal_unique'] += 1
                ledger['arm_last_step'][arm] = step
        else:
            raise ValueError('unknown attempt phase')
        ledger['attempt_count'] += 1
        with (ART / 'attempts.jsonl').open('a') as stream:
            stream.write(json.dumps({'phase': phase, 'arm': arm, 'step': int(step),
                                     'time': time.time(), 'pid': os.getpid()}) + '\n')


@contextmanager
def full_update_recording():
    """Retain every trainer row, including replayed rows, without extra inference."""
    from phycoflow_reconstruction.training.monitoring import TrainingMonitor
    from phycoflow_reconstruction.training.post_training import _json_safe
    original = TrainingMonitor.record
    def record(monitor, row, **kwargs):
        if 'step' in row and 'coherence_loss' in row:
            safe = _json_safe(dict(row))
            directory = Path(monitor.run_dir) / 'logs'
            for name, payload in (
                ('full_updates.jsonl', safe),
                ('family_updates.jsonl', {k: v for k, v in safe.items() if k in {'epoch', 'step', 'total_semantics'}
                                         or k.startswith(('coherence_family/', 'coherence_component/'))}),
            ):
                with (directory / name).open('a') as stream:
                    stream.write(json.dumps(payload, allow_nan=False) + '\n')
        return original(monitor, row, **kwargs)
    TrainingMonitor.record = record
    try:
        yield
    finally:
        TrainingMonitor.record = original


@contextmanager
def occupied(stage):
    # The launcher supplies the verified physical UUID, never an inferred cuda index.
    device = os.environ.get('ROUND7_PHYSICAL_GPU_UUID')
    if not device or not os.environ.get('CUDA_VISIBLE_DEVICES') or not os.environ.get('KEOPS_CACHE_FOLDER'):
        raise RuntimeError('verified GPU UUID, CVD and KeOps cache required')
    token = uuid.uuid4().hex
    with ledger_transaction() as ledger:
        if total_seconds(ledger['intervals']) >= 36 * 3600:
            raise RuntimeError('GPU budget exhausted')
        ledger['intervals'].append({'id': token, 'device_uuid': device, 'start': time.time(), 'stage': stage, 'pid': os.getpid()})
    stop = threading.Event()
    previous_handler = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise RuntimeError('Round 7 GPU budget watchdog stopped the stage')
    signal.signal(signal.SIGTERM, interrupted)
    def guard():
        # Reserve 60 GPU-seconds for the check interval and device cleanup.
        while not stop.wait(5):
            try:
                with ledger_transaction() as ledger:
                    remaining = 36 * 3600 - total_seconds(ledger['intervals'])
                if remaining <= 60:
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
            except Exception:
                os.kill(os.getpid(), signal.SIGTERM)
                return
    watchdog = threading.Thread(target=guard, daemon=True)
    watchdog.start()
    try:
        yield
    finally:
        stop.set()
        watchdog.join()
        signal.signal(signal.SIGTERM, previous_handler)
        with ledger_transaction() as ledger:
            interval = next(x for x in ledger['intervals'] if x['id'] == token)
            interval['end'] = time.time()


def verify_contract(name='formal_contract.json'):
    contract = read(ART / name)
    for name, value in contract['files'].items():
        if digest(name) != value:
            raise ValueError(f'contract file changed: {name}')
    if contract['source_sha256'] != '2d1da713538f6e9613560f3a7eb4a9df2a95d1ad1f56dd11a1a8a4f9ef65050c':
        raise ValueError('SOURCE identity differs')
    return contract


def preparation_contract():
    """Pin existing Round 6 inputs without changing their bytes."""
    path = ART / 'preparation_contract.json'
    prior = CASE / 'runs/coherence_compute_round6'
    paths = {'calibration_config': prior / 'configs/U6.yaml',
             'manifest': prior / 'artifacts/evaluation_manifest.json',
             'positions': prior / 'artifacts/positions.json'}
    payload = {key: str(value) for key, value in paths.items()}
    payload.update({'files': {str(p): digest(p) for p in paths.values()},
                    'source_sha256': '2d1da713538f6e9613560f3a7eb4a9df2a95d1ad1f56dd11a1a8a4f9ef65050c',
                    'fair_artifact': str(ART / 'conditional_fair_topology.pt')})
    if path.exists():
        if read(path) != payload:
            raise ValueError('preparation input bytes changed')
    else:
        write(path, payload)
    return payload


def runtime_source_files():
    package = ROOT / 'src/phycoflow_reconstruction'
    paths = [package / '__init__.py', package / 'contracts.py', ROOT / 'scripts/run_round7.py']
    for directory in ('coherence', 'training', 'models', 'data', 'config'):
        paths.extend((package / directory).rglob('*.py'))
    paths.extend(package / 'evaluation' / name for name in (
        '__init__.py', 'loss_aligned.py', 'round7_reductions.py', 'round7_source.py',
        'round7_native.py', 'native_coherence.py', 'checkpoint.py', 'round7_decision.py',
        'round7_summary.py', 'round7_final_test.py'))
    return sorted(set(paths))


def freeze_qualification_inputs(preparation, *, legacy=False):
    """Bind arm settings and fitted observer before disposable qualifications."""
    import copy
    import yaml
    fair = Path(preparation['fair_artifact'])
    if not legacy and not fair.is_file():
        raise FileNotFoundError('fair Shape calibration artifact required')
    base = yaml.safe_load(Path(preparation['calibration_config']).read_text())
    base['runtime']['seed'] = 47
    base['runtime']['plot_every_steps'] = 100
    base['optimization']['epochs'] = 2000
    base['checkpointing'].update(every_epochs=25, validation_every_epochs=100,
                                 validation_epochs=[750, 1250, 1500, 1750, 2000],
                                 epochs=[1, 100, 300, 500, 750, 800, 900, 1000, 1250, 1500, 1750, 2000])
    base['coherence']['query_coverage']['mode'] = 'uniform'
    base['coherence']['query_coverage']['seed'] = 47
    base['optimization']['multitask']['external_policy']['round7_diagnostics'] = True
    configs = {}
    for arm in (('B', 'P') if legacy else ('B', 'P', 'T')):
        value = copy.deepcopy(base)
        value['output']['experiment_name'] = 'coherence_compute_round7/' + arm
        if arm == 'P':
            value['optimization']['multitask']['external_policy']['direction_preference'] = {
                'family_order': ['global_distribution', 'multiscale_spectrum', 'topology'],
                'vector': [1.0, 1.2, 1.0], 'norm_match': True}
        if arm == 'T':
            value['coherence']['families']['topology']['persistence']['definition'] = 'conditional_fixed_transform_fair_v1'
            value['coherence']['frozen_family_artifacts']['topology'] = {'path': str(fair), 'sha256': digest(fair)}
        path = CAMPAIGN / 'configs' / (arm + '.yaml')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if yaml.safe_load(path.read_text()) != value:
                raise ValueError('frozen arm config differs')
        else:
            path.write_text(yaml.safe_dump(value, sort_keys=False))
        configs[arm] = str(path)
    files = {**preparation['files'], **{p: digest(p) for p in configs.values()}}
    if not legacy:
        files[str(fair)] = digest(fair)
    for declaration in base['coherence']['frozen_family_artifacts'].values():
        if digest(declaration['path']) != declaration['sha256']:
            raise ValueError('historical family artifact differs')
        files[declaration['path']] = declaration['sha256']
    reference = base['optimization']['multitask']['fidelity_budget']
    if digest(reference['reference_artifact']) != reference['reference_sha256']:
        raise ValueError('SOURCE fidelity reference differs')
    files[reference['reference_artifact']] = reference['reference_sha256']
    s100 = CASE / 'runs/coherence_compute_round6/supplement_scale_only/artifacts/train_result.json'
    s100_receipt = read(s100)
    if digest(s100_receipt['endpoint']) != s100_receipt['endpoint_sha256']:
        raise ValueError('S100 endpoint differs from its saved receipt')
    files[str(s100)] = digest(s100)
    files[s100_receipt['endpoint']] = s100_receipt['endpoint_sha256']
    contract = {**preparation, 'files': files, 'configs': configs, 'arms': list(configs)}
    target = ART / 'formal_contract.json'
    if target.exists():
        if read(target) != contract:
            raise ValueError('frozen qualification input changed')
    else:
        write(target, contract)
    return contract


def qualify():
    import torch
    import yaml
    from phycoflow_reconstruction.training.post_training import run_post_training
    from phycoflow_reconstruction.training.coherence_timing import HostTiming, instrument_training_timing
    from phycoflow_reconstruction.training.run_store import load_project_checkpoint
    contract = verify_contract('qualification_contract.json')
    results, failures = [], []
    for arm in contract['arms']:
        config = yaml.safe_load(Path(contract['configs'][arm]).read_text())
        config['output']['experiment_name'] = f'coherence_compute_round7/qualification/{arm}'
        timer = HostTiming()
        before = time.time()
        try:
            with full_update_recording(), instrument_training_timing(timer):
                run = run_post_training(config, case_dir=CASE, max_steps=12, qualification_only=True,
                    evaluation_callback=timer.wrap_callback(lambda **_: None),
                    update_attempt_callback=lambda step, a=arm: charge_attempt('qualification', a, step))
            results.append({'arm': arm, 'run': str(run), 'attempts': 12,
                            'seconds': time.time() - before, 'timing': timer.report(),
                            'peak_bytes': torch.cuda.max_memory_allocated()})
        except Exception as exc:
            failure = {'arm': arm, 'error_type': type(exc).__name__, 'message': str(exc),
                       'seconds': time.time() - before, 'efficacy_ranking': False}
            failures.append(failure)
            write(ART / f'qualification_{arm}_failure_{time.time_ns()}.json', failure)
            if arm == 'B':
                raise
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    if len(results) < 2:
        raise RuntimeError('baseline has no qualified treatment for a matched comparison')
    config = yaml.safe_load(Path(contract['configs']['B']).read_text())
    config['output']['experiment_name'] = 'coherence_compute_round7/qualification/resume'
    callback = lambda **_: None
    charge = lambda step: charge_attempt('qualification', 'resume', step)
    restored = {}
    with full_update_recording():
        resumed = run_post_training(config, case_dir=CASE, max_steps=6, qualification_only=True,
                                   evaluation_callback=callback, update_attempt_callback=charge)
        run_post_training(config, case_dir=CASE, max_steps=6, resume=resumed, qualification_only=True,
                          evaluation_callback=callback, update_attempt_callback=charge,
                          resume_state_callback=lambda saved, live: restored.update(exact_restoration(saved, live)))
    continuous = Path(next(item['run'] for item in results if item['arm'] == 'B'))
    left = load_project_checkpoint(resumed / 'checkpoints/last.pt')
    right = load_project_checkpoint(continuous / 'checkpoints/last.pt')
    if left['global_step'] != 12 or right['global_step'] != 12:
        raise ValueError('resume endpoint cursor mismatch')
    if not restored.get('exact_loaded_state'):
        raise ValueError('actual loaded checkpoint state was not verified')
    results.append({'arm': 'B_resume', 'run': str(resumed), 'continuous': str(continuous),
                    'attempts': 12, 'restore_evidence': restored})
    complete_qualification(contract, results, failures)


def complete_qualification(contract, results, failures):
    """Publish only after arm telemetry and a direct exact-restore check pass."""
    stream_identity = None
    query_sequence = None
    for item in results:
        if item['arm'] not in contract['arms']:
            continue
        rows = [json.loads(line) for line in (Path(item['run']) / 'logs/round7_updates.jsonl').read_text().splitlines()]
        if [row['step'] for row in rows] != list(range(1, 13)):
            raise ValueError('qualification telemetry is missing attempted updates')
        for row in rows:
            gradient = row['gradient']
            if (not gradient.get('update_accepted') or gradient.get('coherence_scale') != 1.9418972484616779
                    or gradient.get('family_order') != ['global_distribution', 'multiscale_spectrum', 'topology']):
                raise ValueError('qualification policy or SOURCE C0 mismatch')
        if 'round7_actual_step' not in rows[0]['gradient']:
            raise ValueError('qualification lacks the first actual-step diagnostic')
        identity = rows[0]['gradient']['round7_actual_step']['identity']
        matched = {key: identity[key] for key in ('sample_ids', 'query_indices', 'draw_seeds', 'shape_queries')}
        queries = [{key: row[key] for key in ('step', 'queries', 'self_draws', 'mutual_draws')}
                   for row in (json.loads(line) for line in
                       (Path(item['run']) / 'logs/query_coverage.jsonl').read_text().splitlines()) if 'step' in row]
        if stream_identity is not None and (matched != stream_identity or queries != query_sequence):
            raise ValueError('qualification condition, observation, draw or query streams differ')
        stream_identity, query_sequence = matched, queries
        updates = [json.loads(line) for line in (Path(item['run']) / 'logs/full_updates.jsonl').read_text().splitlines()]
        if len(updates) != 12 or any(row['native_data_loss'] is not None or row['data_update_weight'] != 0 for row in updates):
            raise ValueError('qualification history or disabled native-loss replay differs')
        for suffix in ('png', 'pdf'):
            if not (Path(item['run']) / 'logs' / ('loss_history.' + suffix)).is_file():
                raise ValueError('qualification loss curve missing')
    source = read(ART / 'reference_SOURCE.json')
    s100 = read(ART / 'reference_S100.json')
    fair_observer = contract.get('observer_branch', 'fair') == 'fair'
    if (source.get('reference') != 'SOURCE' or s100.get('reference') != 'S100'
            or source['manifest_sha256'] != s100['manifest_sha256']
            or source['observer_identities'] != s100['observer_identities']
            or (fair_observer and source['train']['fair_shape_attraction_mean'] <= 0)):
        raise ValueError('common observer references are not qualified')
    admitted = [item['arm'] for item in results if item['arm'] in contract['arms']]
    branch_path = ART / 'observer_branch.json'
    if fair_observer and not branch_path.exists():
        write(branch_path, {'observer': 'fair', 'status': 'frozen_before_learning',
                            'source_receipt_sha256': digest(ART / 'reference_SOURCE.json'),
                            's100_receipt_sha256': digest(ART / 'reference_S100.json')})
    write(ART / 'qualification_timing.json', {'segments': results, 'failures': failures, 'efficacy_ranking': False,
                                               'qualification_complete': True})
    write(ART / 'qualified.json', {'status': 'completed', 'qualified_arms': admitted,
          'observer_qualified': fair_observer, 'observer_branch': contract.get('observer_branch', 'fair'),
          'evaluation_reference_receipts': {
              'SOURCE': digest(ART / 'reference_SOURCE.json'), 'S100': digest(ART / 'reference_S100.json')},
          'timing_sha256': digest(ART / 'qualification_timing.json'),
          'qualification_contract_sha256': digest(ART / 'qualification_contract.json'),
          'evidence_scope': 'observer source calibration, fixed-domain checks and real-model update/resume; not efficacy'})


def repair_qualification():
    """Reuse completed arm checks and perform one exact-state resumed update."""
    import shutil
    import yaml
    from phycoflow_reconstruction.training.post_training import run_post_training
    from phycoflow_reconstruction.training.run_store import load_project_checkpoint
    contract = verify_contract('qualification_contract.json')
    inputs_path = ART / 'qualification_repair_inputs.json'
    inputs = read(inputs_path)
    if (ART / 'qualified.json').exists() or inputs.get('schema') != 'round7_exact_restore_repair_v1':
        raise ValueError('repair requires the pinned failed qualification evidence')
    for path, expected in inputs['files'].items():
        if digest(path) != expected:
            raise ValueError('original qualification evidence changed: ' + path)
    with ledger_transaction() as ledger:
        if ledger['formal_unique'] or ledger['formal_replays'] or ledger['qualification_attempts'] != 48:
            raise ValueError('one-update repair requires exactly the original 48 qualification attempts')
    original = Path(inputs['baseline_run'])
    config = yaml.safe_load((original / 'resolved_config.yaml').read_text())
    baseline = load_project_checkpoint(original / 'checkpoints/last.pt')
    if baseline['global_step'] != 12 or not baseline.get('qualification_only'):
        raise ValueError('repair cannot substitute a formal or different checkpoint')
    probe = CAMPAIGN / 'qualification' / 'exact_restore_probe' / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    probe.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(original, probe)
    write(probe / 'artifacts/restore_probe_origin.json', {
        'original_run': str(original), 'original_checkpoint_sha256': digest(original / 'checkpoints/last.pt'),
        'purpose': 'disposable checkpoint copy; preserve original evidence; one resumed qualification update'})
    proof = {}
    def observe(saved, live):
        if proof:
            raise ValueError('resume observer invoked more than once')
        if not same_checkpoint_state(saved['model'], baseline['model']):
            raise ValueError('probe saved checkpoint differs from pinned B12')
        proof.update(exact_restoration(saved, live))
        write(ART / 'qualification_exact_restore.json', {
            **proof, 'probe_run': str(probe), 'baseline_run': str(original),
            'baseline_checkpoint_sha256': digest(original / 'checkpoints/last.pt'),
            'repair_inputs_sha256': digest(inputs_path),
            'criterion': 'bit-exact loaded state, not equality of independent CUDA trajectories'})
    with full_update_recording():
        run_post_training(config, case_dir=CASE, max_steps=1, resume=probe, qualification_only=True,
            evaluation_callback=lambda **_: None, resume_state_callback=observe,
            update_attempt_callback=lambda step: charge_attempt('qualification', 'exact_restore_probe', step))
    endpoint = load_project_checkpoint(probe / 'checkpoints/last.pt')
    if not proof.get('exact_loaded_state') or endpoint['global_step'] != 13:
        raise ValueError('repair did not restore exactly and complete the resumed update')
    rows = [json.loads(line) for line in (probe / 'logs/round7_updates.jsonl').read_text().splitlines()]
    if rows[-1]['step'] != 13 or not rows[-1]['gradient']['update_accepted']:
        raise ValueError('resumed qualification update was not accepted')
    if digest(original / 'checkpoints/last.pt') != inputs['files'][str(original / 'checkpoints/last.pt')]:
        raise ValueError('original B12 qualification checkpoint changed')
    results = [*inputs['segments'], {'arm': 'B_exact_restore_probe', 'run': str(probe), 'attempts': 1,
               'restore_evidence': proof, 'endpoint_step': 13}]
    complete_qualification(contract, results, [inputs['original_failure']])


def bind_projection_scales():
    """Freeze telemetry-only SOURCE TRAIN normalizers before any optimizer attempt."""
    import copy
    import math
    import yaml
    contract = verify_contract()
    receipt_path = ART / 'reference_SOURCE.json'
    receipt = read(receipt_path)
    if receipt.get('reference') != 'SOURCE' or receipt.get('parent_source_checkpoint_sha256') != contract['source_sha256']:
        raise ValueError('SOURCE evaluation receipt identity mismatch')
    train = receipt['train']
    if train['conditions'] != 800 or digest(train['reductions']) != train['sha256']:
        raise ValueError('SOURCE TRAIN800 reduction identity mismatch')
    payload = read(train['reductions'])
    if payload['manifest_sha256'] != receipt['manifest_sha256'] or payload['split'] != 'train' or len(payload['rows']) != 800:
        raise ValueError('SOURCE TRAIN800 manifest mismatch')
    names = ('global_distribution', 'multiscale_spectrum', 'topology')
    measured = {name: sum(row['losses'][name + '.total'] for row in payload['rows']) / 800 for name in names}
    decision_scale_path = contract.get('decision_scales', str(CASE / 'runs/coherence_compute_round6/artifacts/decision_scales.json'))
    frozen = read(decision_scale_path)
    legacy = {name: float(frozen[name + '.total']) for name in names}
    for name in names:
        if not math.isclose(measured[name], legacy[name], rel_tol=1e-5, abs_tol=1e-8):
            raise ValueError(f'SOURCE TRAIN {name} differs from inherited frozen observer scale')
    branch = read(ART / 'observer_branch.json') if (ART / 'observer_branch.json').exists() else {'observer': 'fair'}
    fair = float(train['fair_shape_attraction_mean']) if branch['observer'] == 'fair' else None
    if any(not math.isfinite(value) or value <= 0 for value in [*legacy.values(), *([fair] if fair is not None else [])]):
        raise ValueError('SOURCE telemetry scales must be positive and finite')
    with ledger_transaction() as ledger:
        if ledger['formal_unique'] or ledger['formal_replays']:
            raise ValueError('projection scales must be bound before any learning update')
    configs = {}
    for arm, path in contract['configs'].items():
        if branch['observer'] == 'legacy' and arm == 'T':
            continue
        value = yaml.safe_load(Path(path).read_text())
        scales = copy.deepcopy(legacy)
        if arm == 'T':
            scales['topology'] = fair
        value['optimization']['multitask']['external_policy']['round7_projection_scales'] = scales
        target = CAMPAIGN / 'configs' / f'{arm}.projection_bound.yaml'
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError('projection-bound config already exists')
        target.write_text(yaml.safe_dump(value, sort_keys=False))
        configs[arm] = str(target)
    files = {**contract['files'], decision_scale_path: digest(decision_scale_path), str(receipt_path): digest(receipt_path),
             train['reductions']: train['sha256'], **{p: digest(p) for p in configs.values()}}
    # Bind runtime science/training/inference code, excluding CPU-only reporting.
    files.update({str(path): digest(path) for path in runtime_source_files()})
    bound = {**contract, 'files': files, 'configs': configs, 'arms': list(configs),
             'observer_branch': branch['observer'],
             'projection_scales': {arm: ({**legacy, 'topology': fair} if arm == 'T' else legacy) for arm in configs},
             'projection_scale_provenance': 'frozen SOURCE TRAIN scales validated on TRAIN800; fair Shape uses new attraction; telemetry only'}
    target = ART / 'qualification_contract.json'
    if target.exists():
        raise FileExistsError('qualification contract already frozen')
    write(target, bound)
    return bound['projection_scales']


def freeze_legacy_branch(diagnosis_path):
    """Freeze legacy scoring only after a documented observer-specific failure."""
    diagnosis_path = Path(diagnosis_path).resolve()
    if not diagnosis_path.is_relative_to(ART.resolve()):
        raise ValueError('observer diagnosis must be a case-owned artifact')
    diagnosis = read(diagnosis_path)
    if (diagnosis.get('status') != 'observer_unqualified'
            or not diagnosis.get('reason') or not diagnosis.get('evidence')):
        raise ValueError('an evidenced observer-specific diagnosis is required')
    with ledger_transaction() as ledger:
        if ledger['formal_unique'] or ledger['formal_replays']:
            raise ValueError('observer branch cannot change after learning begins')
    target = ART / 'observer_branch.json'
    if target.exists():
        raise FileExistsError('observer branch already frozen')
    if not (ART / 'formal_contract.json').exists():
        freeze_qualification_inputs(preparation_contract(), legacy=True)
    for name in ('reference_SOURCE.json', 'reference_S100.json', 'qualification_contract.json'):
        path = ART / name
        if path.exists():
            archived = path.with_name(path.stem + '.fair_rejected.' + digest(path)[:16] + path.suffix)
            if archived.exists():
                raise FileExistsError('observer receipt archive already exists')
            path.rename(archived)
    for arm in ('B', 'P', 'T'):
        path = CAMPAIGN / 'configs' / f'{arm}.projection_bound.yaml'
        if path.exists():
            path.rename(path.with_name(path.stem + '.fair_rejected.' + digest(path)[:16] + path.suffix))
    branch = {'observer': 'legacy', 'status': 'frozen_before_learning',
              'diagnosis_path': str(diagnosis_path), 'diagnosis_sha256': digest(diagnosis_path),
              'conditional_target_alignment': 'unresolved', 'omitted_arm': 'T'}
    write(target, branch)
    return branch


def stage_decide(path):
    """Decide from supplied, previously saved reductions; never perform inference."""
    verify_contract('qualification_contract.json')
    from phycoflow_reconstruction.evaluation.round7_decision import (
        Arm, Observation, decide_500, decide_1000, validate_field_coverage)
    data = read(path)
    def hash_value(value):
        return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
    if not all(hash_value(data.get(k)) for k in ('manifest_sha256', 'observer_sha256')):
        raise ValueError('frozen cohort and common observer hashes required')
    if any(item.get('manifest_sha256') != data['manifest_sha256']
           or item.get('observer_sha256') != data['observer_sha256']
           or not hash_value(item.get('checkpoint_sha256')) for item in data['arms']):
        raise ValueError('arm cohort, observer or endpoint identity mismatch')
    arms = [Arm(item['name'], {int(epoch): Observation(**value) for epoch, value in item['observations'].items()},
                item['update_seconds'], item.get('unchanged_terms', False)) for item in data['arms']]
    validate_field_coverage(arms, ['CH4', 'CO', 'T', 'U_1', 'p'])
    epoch = data['epoch']
    approval_data = None
    if epoch == 500:
        context = data.get('proposal_context', {})
        required = {'actual_step_findings', 'shape_attraction_separation', 'native_field_checks',
                    'measured_cost', 'remaining_budget', 'continued_run_identities'}
        if set(context) != required:
            raise ValueError('500-epoch proposal requires actual step, A/R, native, cost, budget and run identities')
        result = decide_500(arms, data['source_validation_scale'])
        if result['continue']:
            resource = read(ART / 'resource_forecast.json')
            measurements_path = resource['measurements_path']
            if digest(measurements_path) != resource['measurement_sha256']:
                raise ValueError('frozen resource measurements changed')
            measured = read(measurements_path)
            rates = [measured['update_seconds'][name] for name in result['continue']]
            n = len(rates)
            projected = (6500 * sum(rates) + (6 * n + 3) * measured['validation_seconds']
                         + (n + 1) * measured['training_panel_seconds']
                         + (n + 3) * measured['native_panel_seconds'] + 650 * max(rates))
            with ledger_transaction() as ledger:
                remaining = 36 * 3600 - total_seconds(ledger['intervals'])
            if projected > remaining:
                result = {**result, 'continue': [], 'mode': 'budget_conflict',
                          'projected_seconds': projected, 'remaining_seconds': remaining}
            else:
                approval = ART / 'continuation_1000.json'
                if approval.exists():
                    raise FileExistsError('continuation decision already frozen')
                approval_data = {'status': 'approved_by_frozen_rules', 'arms': result['continue'],
                                 'decision_input_sha256': digest(path), 'projected_seconds': projected,
                                 'remaining_seconds': remaining}
    elif epoch == 1000:
        result = decide_1000(arms, data['source_validation_scale'])
        from phycoflow_reconstruction.evaluation.round7_decision import extension_allowed
        candidates = [arm for arm in arms if extension_allowed(arm, data['source_validation_scale'], fits_budget=True)]
        selected = next((arm for arm in candidates if arm.name == result['selected']), None)
        if result['selected'] == 'SOURCE' and candidates:
            selected = min(candidates, key=lambda arm: arm.observations[1000].scale)
        if selected is not None:
            resource = read(ART / 'resource_forecast.json')
            if digest(resource['measurements_path']) != resource['measurement_sha256']:
                raise ValueError('frozen resource measurements changed')
            measured = read(resource['measurements_path'])
            with ledger_transaction() as ledger:
                remaining = 36 * 3600 - total_seconds(ledger['intervals'])
            projected = (3250 * measured['update_seconds'][selected.name]
                         + 6 * measured['validation_seconds'] + 2 * measured['training_panel_seconds']
                         + 4 * measured['native_panel_seconds'] + 650 * measured['update_seconds'][selected.name])
            if projected <= remaining:
                approval = ART / 'continuation_1250.json'
                if approval.exists():
                    raise FileExistsError('extension decision already frozen')
                approval_data = {'status': 'approved_by_frozen_rules', 'arms': [selected.name],
                    'decision_input_sha256': digest(path), 'projected_seconds': projected,
                    'remaining_seconds': remaining, 'pinned_1000_fallback': result['selected']}
                result['extension'] = selected.name
        result.setdefault('extension', None)
    elif epoch in (1250, 1500, 1750, 2000):
        from phycoflow_reconstruction.evaluation.round7_decision import eligible, extension_stop
        if len(arms) != 1:
            raise ValueError('extension must contain exactly one continued arm')
        arm = arms[0]
        previous_data = read(CAMPAIGN / 'comparison' / f'decision_input_{epoch - 250}.json')
        previous_arm = next(item for item in previous_data['arms'] if item['name'] == arm.name)
        previous = Observation(**previous_arm['observations'][str(epoch - 250)])
        current = arm.observations[epoch]
        latest_regular = (epoch // 100) * 100 if epoch % 100 else epoch - 100
        prior = arm.observations[latest_regular]
        material_gain = (previous.state_delta - current.state_delta >= .02 or
                         previous.shape_delta - current.shape_delta >= .02)
        stopping = extension_stop(previous, current, preceding_check=prior,
                                  other_views_material_gain=material_gain, next_block_fits=True)
        endpoint_qualified = eligible(arm, epoch, data['source_validation_scale'],
                                      (latest_regular - 100, latest_regular, epoch))
        initial = read(ART / 'continuation_1250.json')
        fallback = initial['pinned_1000_fallback']
        result = {'selected': arm.name if endpoint_qualified else fallback,
                  'qualified': [arm.name] if endpoint_qualified else [], 'extension': None,
                  'block_stop': stopping, 'reached_epoch': epoch,
                  'selected_epoch': epoch if endpoint_qualified else 1000}
        if not stopping and epoch < 2000:
            resource = read(ART / 'resource_forecast.json')
            if digest(resource['measurements_path']) != resource['measurement_sha256']:
                raise ValueError('frozen resource measurements changed')
            measured = read(resource['measurements_path'])
            rate = measured['update_seconds'][arm.name]
            with ledger_transaction() as ledger:
                remaining = 36 * 3600 - total_seconds(ledger['intervals'])
            projected = (3250 * rate + 6 * measured['validation_seconds'] + 2 * measured['training_panel_seconds']
                         + 4 * measured['native_panel_seconds'] + 650 * rate)
            if projected <= remaining:
                approval_data = {'status': 'approved_by_frozen_rules', 'arms': [arm.name],
                    'decision_input_sha256': digest(path), 'projected_seconds': projected,
                    'remaining_seconds': remaining, 'pinned_1000_fallback': fallback}
                result['extension'] = arm.name
    else:
        raise ValueError('fixed 500 or 1000 epoch decisions only')
    output = CAMPAIGN / 'comparison' / f'decision_{epoch}.json'
    if epoch == 500:
        proposal = CAMPAIGN / 'FORMAL_PROPOSAL.md'
        if proposal.exists() or output.exists():
            raise FileExistsError('500-epoch proposal already frozen')
        table = ['| Arm | Validation Scale | State change (SOURCE units) | Shape change (SOURCE units) | Max Q4096 MSE ratio |',
                 '| --- | ---: | ---: | ---: | ---: |']
        for arm in arms:
            o = arm.observations[500]
            table.append(f'| {arm.name} | {o.scale:.8g} | {o.state_delta:.8g} | {o.shape_delta:.8g} | {max(o.q_mse.values()):.8g} |')
        text = ['# Round 7 formal continuation proposal', '',
                f"Frozen validation SOURCE Scale: {data['source_validation_scale']:.8g}.", '', *table, '',
                f"Decision: {result['mode']}; selected {result['selected']}; continue {', '.join(result['continue']) or 'none'}.", '']
        for key in sorted(required):
            text.append(f"- {key.replace('_', ' ').capitalize()}: {json.dumps(context[key], sort_keys=True, allow_nan=False)}")
        text.extend(['', f'Input SHA-256: {digest(path)}.', ''])
        proposal.write_text('\n'.join(text))
    elif output.exists():
        raise FileExistsError('1000-epoch decision already frozen')
    write(output, {'input_sha256': digest(path), 'decision': result})
    if approval_data is not None:
        next_epoch = 1000 if epoch == 500 else epoch + 250
        write(ART / f'continuation_{next_epoch}.json', approval_data)
    if ((epoch == 500 and not result['continue'] and result['selected'] == 'SOURCE')
            or (epoch >= 1000 and result.get('extension') is None)):
        chosen_epoch = result.get('selected_epoch', 1000)
        freeze_final_selection(result['selected'], chosen_epoch, path)
    return result


def freeze_final_selection(selected, epoch, decision_path):
    target = ART / 'final_selection.json'
    if target.exists():
        raise FileExistsError('final selection already frozen before TEST')
    def endpoint(name, boundary):
        run = Path(read(ART / f'owned_run_{name}.json')['run'])
        binding_path = run / 'evaluation' / f'checkpoint_binding_{boundary}.json'
        binding = read(binding_path)
        if digest(binding['checkpoint']) != binding['checkpoint_sha256']:
            raise ValueError('selected endpoint checkpoint changed')
        return {'arm': name, 'epoch': boundary, 'checkpoint': binding['checkpoint'],
                'checkpoint_sha256': binding['checkpoint_sha256'], 'binding': str(binding_path),
                'binding_sha256': digest(binding_path)}
    selected_endpoint = None if selected == 'SOURCE' else endpoint(selected, epoch)
    baseline = endpoint('B', 1000) if (ART / 'B_1000_complete.json').exists() else None
    write(target, {'status': 'frozen_before_test', 'selected': selected, 'endpoint': selected_endpoint,
                   'B1000': baseline, 'source_checkpoint_sha256': '2d1da713538f6e9613560f3a7eb4a9df2a95d1ad1f56dd11a1a8a4f9ef65050c',
                   'decision_input': str(Path(decision_path).resolve()), 'decision_input_sha256': digest(decision_path)})


def close_incomplete():
    """Disposition an unsafe stopped baseline without inventing a 500 endpoint."""
    verify_contract('qualification_contract.json')
    stopped = read(ART / 'B_protective_stop.json')
    if stopped.get('arm') != 'B' or stopped.get('epoch', 500) >= 500:
        raise ValueError('an early baseline protective stop is required')
    if digest(stopped['checkpoint']) != stopped['checkpoint_sha256']:
        raise ValueError('baseline protective checkpoint changed')
    with ledger_transaction() as ledger:
        if any('end' not in interval for interval in ledger['intervals']):
            raise ValueError('wait for all owned GPU stages to terminate before final disposition')
    path = CAMPAIGN / 'comparison' / 'incomplete_baseline_decision.json'
    if path.exists():
        raise FileExistsError('incomplete comparison already closed')
    result = {'status': 'closed_incomplete', 'selected': 'SOURCE',
              'baseline_stopped_epoch': stopped['epoch'], 'protective_stop_sha256': digest(ART / 'B_protective_stop.json'),
              'reason': 'B stopped by its prespecified field-fidelity rule before the matched 500 endpoint'}
    write(path, result)
    freeze_final_selection('SOURCE', 0, path)
    return result


def build_decision(epoch):
    """Create decision inputs solely from frozen, matched saved reductions."""
    from phycoflow_reconstruction.evaluation.round7_summary import summarize_reductions, compare_summaries
    from phycoflow_reconstruction.evaluation.round7_native import compare_round7_native
    fields = ['CH4', 'CO', 'T', 'U_1', 'p']
    source = read(ART / 'reference_SOURCE.json')
    resource = read(ART / 'resource_forecast.json')
    measurements = read(resource['measurements_path'])
    if digest(resource['measurements_path']) != resource['measurement_sha256']:
        raise ValueError('resource measurements changed')
    source_val = summarize_reductions(source['validation']['reductions'], fields)
    source_train = summarize_reductions(source['train']['reductions'], fields)
    observer_sha = hashlib.sha256(json.dumps(source_val['observer_identities'], sort_keys=True,
                                              separators=(',', ':')).encode()).hexdigest()
    shape_key = 'shape.fair' if 'fair' in source_val['observer_identities'] else 'shape.legacy'
    fixed_scales = read(ART / 'qualification_contract.json')['projection_scales']['B']
    shape_scale = (source['train']['fair_shape_attraction_mean'] if shape_key == 'shape.fair'
                   else fixed_scales['topology'])
    scales = {'state': fixed_scales['global_distribution'], 'scale': fixed_scales['multiscale_spectrum'],
              shape_key: shape_scale}
    if epoch == 500:
        names, windows = resource['selected_arms'], (300, 400, 500)
        stopped = [name for name in names if (ART / f'{name}_protective_stop.json').exists()
                   and not (ART / f'{name}_500_complete.json').exists()]
        if 'B' in stopped:
            raise ValueError('baseline stopped before 500; use close-incomplete after all jobs end')
        names = [name for name in names if name not in stopped]
    elif epoch == 1000:
        names, windows = read(ART / 'continuation_1000.json')['arms'], (800, 900, 1000)
    elif epoch in (1250, 1500, 1750, 2000):
        latest_regular = (epoch // 100) * 100 if epoch % 100 else epoch - 100
        names, windows = read(ART / f'continuation_{epoch}.json')['arms'], (latest_regular - 100, latest_regular, epoch)
    else:
        raise ValueError('decision endpoint must be 500 or 1000')
    arms, run_dirs, native, shape = [], {}, {}, {}
    for name in names:
        completion = read(ART / f'{name}_{epoch}_complete.json')
        if completion.get('status') != 'completed':
            raise ValueError('fixed boundary not completed')
        run = Path(completion['run'])
        run_dirs[name] = run
        binding_path = run / 'evaluation' / f'checkpoint_binding_{epoch}.json'
        raw_binding = read(binding_path)
        binding = verify_endpoint_binding(binding_path, model_state_sha256=raw_binding['model_state_sha256'],
                                          manifest_sha256=source_val['manifest_sha256'],
                                          observer_identities=source_val['observer_identities'])
        native_checked = compare_round7_native(Path(source['native_validation']['receipt']).parent,
                                              Path(binding['native_receipt']).parent)
        if native_checked != binding['native_comparison']:
            raise ValueError('fixed native endpoint comparison changed')
        observations = {}
        for window in windows:
            summary = summarize_reductions(run / 'evaluation' / f'validation_epoch_{window}' / 'reductions.json', fields)
            changes = compare_summaries(source_val, summary, positive_scales=scales)['changes']
            q = {field: changes['field.' + field + '.mse']['candidate'] /
                 changes['field.' + field + '.mse']['reference'] for field in fields}
            observation = {'epoch': window, 'scale': summary['scores']['scale'],
                           'state_delta': changes['state']['source_units_change'],
                           'shape_delta': changes[shape_key]['source_units_change'], 'q_mse': q}
            if window == epoch:
                train = summarize_reductions(binding['train_reductions'], fields)
                train_change = compare_summaries(source_train, train)['changes']
                observation['train_mse'] = {field: train_change['field.' + field + '.mse']['candidate'] /
                    train_change['field.' + field + '.mse']['reference'] for field in fields}
                observation['native_mse'] = dict(zip(binding['native_comparison']['field_names'],
                                                     binding['native_comparison']['primary_mse_ratio']))
                native[name] = observation['native_mse']
                shape[name] = {observer: {'losses': values['losses'],
                    'attraction_separation': values.get('diagnostics', {}).get('component_attraction_separation')}
                    for observer, values in summary['shape_observers'].items()}
            observations[str(window)] = observation
        arms.append({'name': name, 'observations': observations,
                     'update_seconds': measurements['update_seconds'][name], 'unchanged_terms': name in ('B', 'P'),
                     'manifest_sha256': source_val['manifest_sha256'],
                     'observer_sha256': observer_sha,
                     'checkpoint_sha256': binding['checkpoint_sha256']})
    data = {'epoch': epoch, 'source_validation_scale': source_val['scores']['scale'],
            'manifest_sha256': source_val['manifest_sha256'],
            'observer_sha256': observer_sha, 'arms': arms}
    if epoch == 500:
        data['protectively_stopped_arms'] = stopped
    if epoch == 500:
        from phycoflow_reconstruction.evaluation.round7_analysis import generate_analysis
        analysis_dir = CAMPAIGN / 'comparison' / 'decision_500_analysis'
        analysis = generate_analysis(run_dirs, analysis_dir)
        data['proposal_context'] = {'actual_step_findings': {'analysis': str(analysis_dir / 'analysis.json'),
            'flags': {arm: {'negative_projection': details['negative_at_least_ten_percent'],
                            'scale_only_two_windows': details['scale_only_at_most_half_in_two_windows']}
                      for arm, details in analysis['arms'].items()}},
            'shape_attraction_separation': shape, 'native_field_checks': native,
            'measured_cost': {'forecast_measurements_sha256': resource['measurement_sha256']},
            'remaining_budget': {'gpu_seconds_used': total_seconds(read(ART / 'budget_ledger.json')['intervals']),
                                 'gpu_seconds_cap': 36 * 3600},
            'continued_run_identities': {name: str(path) for name, path in run_dirs.items()}}
    path = CAMPAIGN / 'comparison' / f'decision_input_{epoch}.json'
    if path.exists():
        raise FileExistsError('fixed decision input already exists')
    write(path, data)
    return stage_decide(path)


def forecast(path):
    verify_contract('qualification_contract.json')
    from phycoflow_reconstruction.evaluation.round7_decision import forecast_matched, choose_resource_design
    data = read(path)
    if not data.get('observer_qualification_receipt') or not data.get('whole_update_timing_receipt'):
        raise ValueError('observer qualification and whole-update timing receipts required')
    for key in ('observer_qualification_receipt', 'whole_update_timing_receipt', 'panel_timing_receipt'):
        if not Path(data[key]).is_file():
            raise ValueError(f'missing measured receipt: {key}')
    qualification = read(data['observer_qualification_receipt'])
    if (qualification.get('status') != 'completed'
            or bool(qualification.get('observer_qualified')) != data['observer_qualified']
            or ('P' in qualification.get('qualified_arms', [])) != data['p_qualified']
            or ('T' in qualification.get('qualified_arms', [])) != data['t_qualified']
            or digest(data['whole_update_timing_receipt']) != qualification.get('timing_sha256')):
        raise ValueError('forecast qualification status or measured timing does not match frozen receipt')
    with ledger_transaction() as ledger:
        if ledger['formal_unique'] or ledger['formal_replays']:
            raise ValueError('resource design must be frozen before learning')
        used = total_seconds(ledger['intervals'])
    times = data['update_seconds']
    kwargs = {key: data[key] for key in ('validation_seconds', 'training_panel_seconds', 'native_panel_seconds')}
    costs = {'BP': forecast_matched(('B', 'P'), times, **kwargs) if data['p_qualified'] else 1e300}
    if data['observer_qualified'] and data['t_qualified']:
        costs['BPT'] = forecast_matched(('B', 'P', 'T'), times, **kwargs) if data['p_qualified'] else 1e300
        costs['BT'] = forecast_matched(('B', 'T'), times, **kwargs)
    else:
        costs.update(BPT=1e300, BT=1e300)
    design = choose_resource_design(three_arm_seconds=costs['BPT'], bt_seconds=costs['BT'],
        bp_seconds=costs['BP'], available_seconds=36 * 3600 - used,
        observer_qualified=data['observer_qualified'], t_qualified=data['t_qualified'])
    target = ART / 'resource_forecast.json'
    if target.exists():
        raise FileExistsError('resource design already frozen')
    result = {'schema': 'round7_resource_forecast_v1', 'measurement_sha256': digest(path),
              'measurements_path': str(path),
              'input_receipts': {key: {'path': data[key], 'sha256': digest(data[key])} for key in
                  ('observer_qualification_receipt', 'whole_update_timing_receipt', 'panel_timing_receipt')},
              'used_gpu_seconds': used, 'projected_additional_seconds': costs,
              'selected_arms': design, 'recovery_attempts_reserved': 650,
              'basis': 'measured full-cohort callbacks and whole updates; protected B paired 1000 and final panels'}
    write(target, result)
    return result


def build_measurements():
    """Extract conservative forecast inputs from completed full-panel receipts."""
    verify_contract('qualification_contract.json')
    qualification_path = ART / 'qualified.json'
    timing_path = ART / 'qualification_timing.json'
    source_path = ART / 'reference_SOURCE.json'
    s100_path = ART / 'reference_S100.json'
    qualification, timing, source, s100 = (read(path) for path in (qualification_path, timing_path, source_path, s100_path))
    if (qualification.get('status') != 'completed' or digest(timing_path) != qualification.get('timing_sha256')
            or digest(source_path) != qualification['evaluation_reference_receipts']['SOURCE']
            or digest(s100_path) != qualification['evaluation_reference_receipts']['S100']):
        raise ValueError('qualification or full-panel evidence changed')
    rates = {}
    for segment in timing['segments']:
        arm = segment['arm']
        if arm not in qualification['qualified_arms']:
            continue
        rows = [json.loads(line) for line in (Path(segment['run']) / 'logs/full_updates.jsonl').read_text().splitlines()]
        samples = [float(row['update_seconds']) for row in rows if 'update_seconds' in row]
        if len(samples) != 12 or any(value <= 0 for value in samples):
            raise ValueError('whole-update qualification timing history is incomplete')
        # Segment wall includes first-update probes and setup; never understate ordinary mean.
        rates[arm] = max(sum(samples) / len(samples), float(segment['seconds']) / 12)
    if set(rates) != set(qualification['qualified_arms']):
        raise ValueError('missing admitted-arm update timing')
    def panel(receipt, split):
        item = receipt[split]
        if digest(item['reductions']) != item['sha256']:
            raise ValueError('full-panel reduction changed')
        timing_data = read(item['reductions'])['timing']
        return max(float(timing_data['condition_seconds']), float(timing_data['invocation_seconds']))
    native_path = source['native_validation']['receipt']
    if digest(native_path) != source['native_validation']['sha256']:
        raise ValueError('SOURCE native timing receipt changed')
    native = read(native_path)
    data = {'observer_qualification_receipt': str(qualification_path),
            'whole_update_timing_receipt': str(timing_path), 'panel_timing_receipt': str(source_path),
            's100_timing_receipt': str(s100_path), 's100_timing_sha256': digest(s100_path),
            'observer_qualified': qualification['observer_qualified'],
            'p_qualified': 'P' in rates, 't_qualified': 'T' in rates,
            'update_seconds': rates,
            'validation_seconds': max(panel(source, 'validation'), panel(s100, 'validation')),
            'training_panel_seconds': panel(source, 'train'),
            'native_panel_seconds': float(native['wrapper_wall_seconds']),
            'basis': 'full TRAIN800 and VALIDATION100 panels, full native VALIDATION100, timed real-model qualification'}
    path = ART / 'resource_measurements.json'
    if path.exists():
        raise FileExistsError('resource measurements already frozen')
    write(path, data)
    return data


def train_arm(arm, target_epoch, resume, *, recovery_receipt=None):
    """Continue a frozen arm, evaluating only its predeclared fixed panels."""
    import copy
    import torch
    import yaml
    from types import SimpleNamespace
    from phycoflow_reconstruction.coherence.compose import build_enabled_families
    from phycoflow_reconstruction.coherence.relation_queries import QueryBank
    from phycoflow_reconstruction.data.factory import open_field_dataset
    from phycoflow_reconstruction.data.manifest import SensorManifest
    from phycoflow_reconstruction.evaluation.loss_aligned import evaluate_split, legacy_metrics, make_batches, model_state_digest
    from phycoflow_reconstruction.evaluation.round7_native import evaluate_round7_native, compare_round7_native
    from phycoflow_reconstruction.training.coherence_timing import HostTiming, instrument_training_timing
    from phycoflow_reconstruction.training.post_training import run_post_training
    from phycoflow_reconstruction.training.run_store import load_project_checkpoint

    contract = verify_contract('qualification_contract.json')
    qualified = read(ART / 'qualified.json')
    resource = read(ART / 'resource_forecast.json')
    if qualified.get('status') != 'completed' or arm not in qualified.get('qualified_arms', []):
        raise ValueError('arm has not completed frozen qualification')
    if arm not in resource['selected_arms']:
        raise ValueError('arm is not in the pre-learning resource design')
    if target_epoch not in (500, 1000, 1250, 1500, 1750, 2000):
        raise ValueError('target must be a planned stage boundary')
    recovering = recovery_receipt is not None
    if target_epoch == 500 and resume is not None and not recovering:
        raise ValueError('initial matched learning starts at SOURCE')
    if target_epoch > 500 and resume is None and not recovering:
        raise ValueError('continuation requires the existing run and state')
    if target_epoch > 500:
        approval = read(ART / f'continuation_{target_epoch}.json')
        if arm not in approval['arms'] or approval.get('status') != 'approved_by_frozen_rules':
            raise ValueError('continuation is not authorized by the saved decision and budget')
    config = yaml.safe_load(Path(contract['configs'][arm]).read_text())
    owned_path = ART / f'owned_run_{arm}.json'
    if recovering:
        receipt_path = Path(recovery_receipt).resolve()
        if not receipt_path.is_relative_to(ART.resolve()) or not receipt_path.name.startswith(f'{arm}_{target_epoch}_failure_'):
            raise ValueError('recovery requires this stage\'s case-owned failure receipt')
        failure = read(recovery_receipt)
        owned = read(owned_path)
        if (failure.get('status') != 'failed' or failure.get('arm') != arm
                or failure.get('target_epoch') != target_epoch
                or owned.get('config_sha256') != digest(contract['configs'][arm])):
            raise ValueError('recovery receipt, arm or frozen config mismatch')
        owned_run = Path(owned['run']).resolve()
        if not owned_run.is_relative_to((CAMPAIGN / arm).resolve()):
            raise ValueError('owned run is outside this arm')
        if resume is not None and Path(resume).resolve() != owned_run:
            raise ValueError('recovery cannot substitute a different run')
        resume = str(owned_run)
    elif target_epoch == 500:
        with ledger_transaction() as ledger:
            if owned_path.exists() or arm in ledger['arm_last_step']:
                raise ValueError('formal arm already attempted; use exact recovery if available')
    else:
        owned = read(owned_path)
        if (Path(resume).resolve() != Path(owned['run']).resolve()
                or owned['config_sha256'] != digest(contract['configs'][arm])):
            raise ValueError('continuation must use the owned run and frozen config')
    manifest = read(contract['manifest'])
    positions = read(contract['positions'])
    source = read(ART / 'reference_SOURCE.json')
    source_metrics = legacy_metrics(source['validation']['reductions'], config['dataset']['field_names'])
    native_source = source['native_validation']
    sensor_manifest = SensorManifest.load(native_source['sensor_manifest'])
    train = open_field_dataset(config['dataset'], split='train')
    validation = open_field_dataset(config['dataset'], split='validation', normalizer=train.normalizer)
    bank = QueryBank(train.field_names, seed=int(config['coherence']['query_coverage']['relation_seed']))
    identities = source['observer_identities']
    start_step = 0
    if resume is not None:
        checkpoint = load_project_checkpoint(Path(resume) / 'checkpoints/last.pt')
        start_step = int(checkpoint['global_step'])
        previous = 0 if target_epoch == 500 else target_epoch - (500 if target_epoch == 1000 else 250)
        if recovering:
            if not previous * 13 <= start_step < target_epoch * 13:
                raise ValueError('recovery checkpoint is outside the failed stage')
        elif start_step != previous * 13:
            raise ValueError('resume checkpoint does not match prior fixed boundary')
        if not recovering:
            prior_binding = read(Path(resume) / 'evaluation' / f'checkpoint_binding_{previous}.json')
            if digest(prior_binding['checkpoint']) != prior_binding['checkpoint_sha256']:
                raise ValueError('prior fixed checkpoint bytes changed')
            pinned = load_project_checkpoint(prior_binding['checkpoint'])
            for key in ('model', 'optimizer', 'external_policy', 'query_coverage', 'rng_state', 'training_aux_state'):
                if key not in checkpoint or key not in pinned or not same_checkpoint_state(checkpoint[key], pinned[key]):
                    raise ValueError(f'continuation state differs from the pinned boundary: {key}')
        if recovering:
            import shutil
            archive = Path(resume) / 'logs' / f'recovery_archive_{time.time_ns()}'
            archive.mkdir(parents=True)
            for filename in ('full_updates.jsonl', 'family_updates.jsonl', 'round7_updates.jsonl', 'history.jsonl'):
                path = Path(resume) / 'logs' / filename
                if not path.exists():
                    continue
                shutil.copy2(path, archive / filename)
                if filename in {'full_updates.jsonl', 'family_updates.jsonl'}:
                    keep = [line for line in path.read_text().splitlines()
                            if int(json.loads(line)['step']) <= start_step]
                    temp = path.with_name('.' + path.name + '.recovery.tmp')
                    temp.write_text(''.join(line + '\n' for line in keep))
                    os.replace(temp, path)
    timer = HostTiming()
    measured = {}
    path_state = {}

    def observers(families, device):
        active = families['topology']
        if contract.get('observer_branch', 'fair') == 'legacy':
            if arm == 'T':
                raise ValueError('T cannot run on the legacy observer branch')
            return {'legacy': active}
        alternative = copy.deepcopy(config['coherence'])
        use_fair = arm != 'T'
        alternative['families']['topology']['persistence']['definition'] = (
            'conditional_fixed_transform_fair_v1' if use_fair else 'finite_primary_v1')
        other = build_enabled_families(alternative, train.data_spec, train.normalizer)['topology'].to(device)
        old = yaml.safe_load(Path(contract['configs']['B']).read_text())
        artifact_path = contract['fair_artifact'] if use_fair else old['coherence']['frozen_family_artifacts']['topology']['path']
        other.load_state_artifact(torch.load(artifact_path, map_location=device, weights_only=True))
        return {'fair': other, 'legacy': active} if use_fair else {'fair': active, 'legacy': other}

    def evaluate(*, model, families, config, run_dir, global_step, source):
        if source:
            return source_metrics
        if global_step % 13:
            raise ValueError('fixed evaluation requires an epoch boundary')
        epoch = global_step // 13
        current_digest = model_state_digest(model)
        if epoch in measured:
            if measured[epoch]['model_state_sha256'] != current_digest:
                raise ValueError('repeated validation endpoint changed weights')
            return measured[epoch]['metrics']
        path_state.update(model=model, families=families, run=Path(run_dir))
        pair = observers(families, next(model.parameters()).device)
        out = Path(run_dir) / 'evaluation' / f'validation_epoch_{epoch}'
        path = evaluate_split(model=model, batches=make_batches(validation, positions['validation'], config, 812711, next(model.parameters()).device),
            families=families, config=config, manifest=manifest, split='validation', checkpoint_sha256=model_state_digest(model),
            weight_identity_kind='model_state_sha256', query_bank=bank, output_dir=out,
            round7_observers=pair, round7_observer_identities=identities)
        metrics = legacy_metrics(path, train.field_names)
        ratios = {field: metrics['endpoint_mse_model_units'][field] / source_metrics['endpoint_mse_model_units'][field]
                  for field in train.field_names}
        measured[epoch] = {'path': str(path), 'model_state_sha256': current_digest, 'ratios': ratios,
                           'metrics': metrics}
        log = Path(run_dir) / 'logs/loss_aligned_validation.jsonl'
        old_rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        if not any(row['step'] == global_step for row in old_rows):
            with log.open('a') as stream:
                stream.write(json.dumps({'step': global_step, 'epoch': epoch, 'metrics': metrics,
                                         'source_ratios': ratios}, allow_nan=False) + '\n')
        return metrics

    def checkpoint(run_dir, checkpoint_path, step):
        owned = {'arm': arm, 'run': str(Path(run_dir).resolve()), 'config_sha256': digest(contract['configs'][arm])}
        if owned_path.exists() and read(owned_path) != owned:
            raise ValueError('a different run is already owned for this arm')
        if not owned_path.exists():
            write(owned_path, owned)
        epoch = step // 13
        if step % 13 or epoch not in measured:
            return
        run = Path(run_dir)
        payload = load_project_checkpoint(checkpoint_path)
        class SavedState:
            def state_dict(self):
                return payload['model']
        model = path_state['model']
        if payload['global_step'] != step or model_state_digest(SavedState()) != measured[epoch]['model_state_sha256']:
            raise ValueError('evaluation and saved checkpoint weights differ')
        log = run / 'logs/loss_aligned_validation.jsonl'
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        if (target_epoch > 1000 and len(rows) >= 2
                and any(all(row['source_ratios'][field] > 1.10 for row in rows[-2:]) for field in train.field_names)):
            interruption = {'arm': arm, 'observed_epoch': epoch, 'checkpoint': str(checkpoint_path),
                            'checkpoint_sha256': digest(checkpoint_path),
                            'reason': 'same-field validation MSE ratio above 1.10 on two consecutive scheduled checks'}
            write(ART / f'{arm}_extension_interrupted_{epoch}.json', interruption)
            fallback = read(ART / 'continuation_1250.json')['pinned_1000_fallback']
            freeze_final_selection(fallback, 1000, CAMPAIGN / 'comparison/decision_input_1000.json')
            raise RuntimeError('extension stopped by consecutive validation fidelity guard; pinned 1000 fallback retained')
        if len(rows) >= 2 and any(all(row['source_ratios'][field] > 1.25 for row in rows[-2:]) for field in train.field_names):
            write(ART / f'{arm}_protective_stop.json', {'arm': arm, 'epoch': epoch,
                  'checkpoint': str(checkpoint_path), 'checkpoint_sha256': digest(checkpoint_path),
                  'reason': 'same-field Q4096 validation MSE ratio above 1.25 at two consecutive scheduled checks',
                  'matched_comparison': 'incomplete' if epoch < 500 else 'see completed fixed endpoints'})
            raise RuntimeError('protective stop: same-field Q4096 ratio exceeded 1.25 twice')
        if epoch not in (500, 1000, 1250, 1500, 1750, 2000):
            return
        binding_path = run / 'evaluation' / f'checkpoint_binding_{epoch}.json'
        if binding_path.exists():
            verify_endpoint_binding(binding_path, model_state_sha256=measured[epoch]['model_state_sha256'],
                                    manifest_sha256=manifest['sha256'], observer_identities=identities)
            return
        import shutil
        pinned = run / 'checkpoints' / f'fixed_endpoint_{epoch}.pt'
        if pinned.exists():
            if digest(pinned) != digest(checkpoint_path):
                raise ValueError('fixed endpoint already exists with different bytes')
        else:
            temp = pinned.with_name('.' + pinned.name + '.tmp')
            shutil.copyfile(checkpoint_path, temp)
            os.replace(temp, pinned)
        pair = observers(path_state['families'], next(model.parameters()).device)
        train_path = evaluate_split(model=model, batches=make_batches(train, positions['train'], config, 812711, next(model.parameters()).device),
            families=path_state['families'], config=config, manifest=manifest, split='train',
            checkpoint_sha256=measured[epoch]['model_state_sha256'], weight_identity_kind='model_state_sha256',
            query_bank=bank, output_dir=run / 'evaluation' / f'train_epoch_{epoch}',
            round7_observers=pair, round7_observer_identities=identities)
        runtime = SimpleNamespace(model=model, device=next(model.parameters()).device, config=config, checkpoint_path=pinned)
        endpoint = {'steps': config['rollout']['steps'], 'solver': config['rollout']['solver'],
                    'observation_config': config['observation_consistency'], 'weight_selection': 'live',
                    'sensor_manifest': sensor_manifest}
        native = evaluate_round7_native(runtime, validation, positions['validation'],
            run / 'evaluation' / f'native_validation_epoch_{epoch}', endpoint, panel_seed_base=812711)
        train_metrics = legacy_metrics(train_path, train.field_names)
        source_train = legacy_metrics(source['train']['reductions'], train.field_names)
        train_ratios = {field: train_metrics['endpoint_mse_model_units'][field] / source_train['endpoint_mse_model_units'][field]
                        for field in train.field_names}
        native_comparison = compare_round7_native(Path(native_source['receipt']).parent, Path(native).parent)
        write(binding_path,
              {'checkpoint': str(pinned), 'checkpoint_sha256': digest(pinned),
               'model_state_sha256': measured[epoch]['model_state_sha256'], 'train_reductions': str(train_path),
               'train_reductions_sha256': digest(train_path),
               'validation_reductions': measured[epoch]['path'], 'native_receipt': str(native),
               'validation_reductions_sha256': digest(measured[epoch]['path']), 'native_receipt_sha256': digest(native),
               'train_primary_mse_ratios': train_ratios, 'native_comparison': native_comparison})

    started = time.time()
    try:
        with full_update_recording(), instrument_training_timing(timer):
            run = run_post_training(config, case_dir=CASE, max_steps=target_epoch * 13 - start_step,
                resume=resume, evaluation_callback=timer.wrap_callback(evaluate),
                checkpoint_callback=timer.wrap_callback(checkpoint),
                update_attempt_callback=lambda step: charge_attempt('formal', arm, step))
        payload = load_project_checkpoint(run / 'checkpoints/last.pt')
        if payload['global_step'] != target_epoch * 13:
            raise ValueError('completed run does not reach fixed stage boundary')
        write(ART / f'{arm}_{target_epoch}_complete.json', {'status': 'completed', 'run': str(run),
            'epoch': target_epoch, 'last_checkpoint_sha256': digest(run / 'checkpoints/last.pt'),
            'elapsed_seconds': time.time() - started, 'timing': timer.report()})
    finally:
        train.close()
        validation.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'qualify', 'qualify-repair', 'reference', 'freeze-legacy', 'bind-scales', 'measurements', 'forecast', 'train', 'finish', 'report', 'stage-decide', 'close-incomplete'])
    parser.add_argument('arm', nargs='?', choices=['B', 'P', 'T'])
    parser.add_argument('--target-epoch', type=int)
    parser.add_argument('--resume')
    parser.add_argument('--recover', help='specific prior failure receipt for same-run recovery')
    parser.add_argument('--floor', type=float)
    parser.add_argument('--floor-units')
    parser.add_argument('--reductions', help='saved decision input JSON')
    parser.add_argument('--reference', choices=['SOURCE', 'S100'])
    parser.add_argument('--measurements', help='measured full-panel and update timings JSON')
    parser.add_argument('--diagnosis', help='case-owned observer failure diagnosis')
    args = parser.parse_args()
    if args.stage == 'prepare':
        if args.floor is None or not args.floor_units:
            parser.error('prepare requires --floor and --floor-units')
        started = time.time()
        try:
            contract = preparation_contract()
            from phycoflow_reconstruction.evaluation.round7_calibrate import main as calibrate
            with occupied('prepare'):
                calibrate(['--config', contract['calibration_config'], '--manifest', contract['manifest'],
                           '--positions', contract['positions'], '--output', contract['fair_artifact'],
                           '--floor', str(args.floor), '--floor-units', args.floor_units])
                frozen = freeze_qualification_inputs(contract)
            write(ART / 'prepare_complete.json', {'status': 'completed', 'elapsed_seconds': time.time() - started,
                  'fair_artifact': contract['fair_artifact'], 'fair_sha256': digest(contract['fair_artifact']),
                  'formal_contract_sha256': digest(ART / 'formal_contract.json'),
                  'arms': frozen['arms'], 'optimizer_attempts': 0, 'observer_qualified': False})
        except BaseException as exc:
            failure = ART / 'prepare_failure.json'
            if failure.exists():
                failure = ART / f'prepare_failure_{time.time_ns()}.json'
            write(failure, {'status': 'failed', 'elapsed_seconds': time.time() - started,
                  'error_type': type(exc).__name__, 'message': str(exc), 'optimizer_attempts': 0})
            raise
    elif args.stage in ('qualify', 'qualify-repair'):
        try:
            with occupied(args.stage):
                (qualify if args.stage == 'qualify' else repair_qualification)()
        except BaseException as exc:
            write(ART / f'qualification_failure_{time.time_ns()}.json', {
                'status': 'failed', 'stage': args.stage,
                'error_type': type(exc).__name__, 'message': str(exc)})
            raise
    elif args.stage == 'reference':
        if not args.reference:
            parser.error('reference requires --reference SOURCE or S100')
        import yaml
        from phycoflow_reconstruction.evaluation.round7_source import evaluate_source
        contract = verify_contract()
        name = args.reference
        branch = read(ART / 'observer_branch.json') if (ART / 'observer_branch.json').exists() else {'observer': 'fair'}
        receipt = ART / f'reference_{name}.json'
        if receipt.exists():
            raise FileExistsError('reference evaluation already completed')
        try:
            with occupied('reference_' + name):
                result = evaluate_source(yaml.safe_load(Path(contract['configs']['B']).read_text()),
                    manifest_path=contract['manifest'], positions_path=contract['positions'],
                    fair_artifact=contract['fair_artifact'] if branch['observer'] == 'fair' else None,
                    output_dir=(CAMPAIGN / 'comparison' / name if branch['observer'] == 'fair'
                                else CAMPAIGN / 'comparison' / 'legacy_observer' / name),
                    native_source=(name == 'SOURCE'),
                    s100_receipt=(CASE / 'runs/coherence_compute_round6/supplement_scale_only/artifacts/train_result.json')
                        if name == 'S100' else None)
            write(receipt, result)
        except BaseException as exc:
            write(ART / f'reference_{name}_failure.json', {'status': 'failed',
                  'error_type': type(exc).__name__, 'message': str(exc)})
            raise
    elif args.stage == 'stage-decide':
        if args.reductions:
            result = stage_decide(args.reductions)
        elif args.target_epoch in (500, 1000, 1250, 1500, 1750, 2000):
            result = build_decision(args.target_epoch)
        else:
            parser.error('stage-decide requires saved reductions or a planned target epoch')
        print(json.dumps(result, sort_keys=True))
    elif args.stage == 'bind-scales':
        print(json.dumps(bind_projection_scales(), sort_keys=True))
    elif args.stage == 'freeze-legacy':
        if not args.diagnosis:
            parser.error('freeze-legacy requires --diagnosis')
        print(json.dumps(freeze_legacy_branch(args.diagnosis), sort_keys=True))
    elif args.stage == 'close-incomplete':
        print(json.dumps(close_incomplete(), sort_keys=True))
    elif args.stage == 'forecast':
        if not args.measurements:
            parser.error('forecast requires --measurements')
        print(json.dumps(forecast(args.measurements), sort_keys=True))
    elif args.stage == 'measurements':
        print(json.dumps(build_measurements(), sort_keys=True))
    elif args.stage == 'train':
        if not args.arm or args.target_epoch is None:
            parser.error('train requires an arm and --target-epoch')
        started = time.time()
        try:
            with occupied(f'{args.arm}_{args.target_epoch}'):
                train_arm(args.arm, args.target_epoch, args.resume, recovery_receipt=args.recover)
        except BaseException as exc:
            write(ART / f'{args.arm}_{args.target_epoch}_failure_{time.time_ns()}.json',
                  {'status': 'failed', 'arm': args.arm, 'target_epoch': args.target_epoch,
                   'elapsed_seconds': time.time() - started,
                   'error_type': type(exc).__name__, 'message': str(exc)})
            raise
    elif args.stage == 'finish':
        import yaml
        from phycoflow_reconstruction.evaluation.round7_final_test import evaluate_frozen_test, validate_completed_test
        contract = verify_contract('qualification_contract.json')
        selection_path = ART / 'final_selection.json'
        if not selection_path.is_file():
            raise ValueError('final validation selection must be frozen before TEST')
        result_path = ART / 'final_test.json'
        try:
            if result_path.exists():
                validate_completed_test(result_path, selection_path)
            else:
                with occupied('finish_test'):
                    result = evaluate_frozen_test(yaml.safe_load(Path(contract['configs']['B']).read_text()),
                        selection_path=selection_path, manifest_path=contract['manifest'], positions_path=contract['positions'],
                        fair_artifact=contract['fair_artifact'] if contract.get('observer_branch', 'fair') == 'fair' else None,
                        output_root=CAMPAIGN / 'comparison' / 'TEST')
                    write(result_path, result)
            write(ART / 'finish_test_complete.json', {'status': 'completed', 'test_receipt': str(result_path),
                                                      'test_receipt_sha256': digest(result_path)})
        except BaseException as exc:
            write(ART / f'finish_failure_{time.time_ns()}.json', {'status': 'failed',
                  'error_type': type(exc).__name__, 'message': str(exc)})
            raise
    elif args.stage == 'report':
        from phycoflow_reconstruction.evaluation.round7_report import generate_report, validate_report_evidence
        result_path, selection_path = ART / 'final_test.json', ART / 'final_selection.json'
        complete = read(ART / 'finish_test_complete.json')
        if complete.get('status') != 'completed' or digest(result_path) != complete.get('test_receipt_sha256'):
            raise ValueError('final TEST completion receipt differs')
        validate_report_evidence(result_path, selection_path)
        with ledger_transaction() as ledger:
            if any('end' not in interval for interval in ledger['intervals']):
                raise ValueError('all GPU stages must be closed before final CPU reporting')
            gpu_time, attempts = total_seconds(ledger['intervals']), ledger['attempt_count']
        report_started = time.time()
        report = generate_report(CAMPAIGN, result_path, selection_path, gpu_seconds=gpu_time, attempt_count=attempts)
        write(ART / 'finish_complete.json', {'status': 'completed', 'cpu_report_seconds': time.time() - report_started, **report})
    else:
        raise RuntimeError('stage unavailable: periodic and native evaluation integration is not qualified')


if __name__ == '__main__':
    main()
