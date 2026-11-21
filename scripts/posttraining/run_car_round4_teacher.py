#!/usr/bin/env python3
"""Bounded numerical qualification for the Round 4 endpoint teacher.

Qualification is execution/timing evidence only, never a learning result. It
consumes existing qualification SOURCE rows and does not create a teacher bank.
"""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
import signal
import sys
import time
import uuid

PROCESS_STARTED = time.time()
PROCESS_CPU_STARTED = time.process_time()
CODE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('CAR_REPO_ROOT', CODE)).resolve()
sys.path[:0] = [str(CODE / 'src'), str(CODE), str(CODE / 'scripts/posttraining')]
from prepare_car_round4 import device_identity, resolve_config, sha

# Mean-risk allowances in model-field squared units from the frozen reference.
EPSILON_MODEL_SQUARED = (0.0010150245189142878, 0.006753054986940697,
                         0.001075031697342638, 0.004402862859424204,
                         0.0014518191469833257)



def frozen_allowance(cfg, manifest):
    budget = cfg['optimization']['multitask']['fidelity_budget']
    declaration = budget['reference_artifact']
    path = Path(declaration['path'])
    if sha(path) != declaration['sha256']:
        raise ValueError('Frozen fidelity reference hash changed')
    document = json.loads(path.read_text())
    ref, settings = document['reference'], document['settings']
    values = ref['absolute_allowance']
    if (ref['field_names'] != manifest['field_order'] or ref['kind'] != 'mean_risk_difference'
            or settings['metric'] != 'physical_mean_risk_model_units'
            or values != list(EPSILON_MODEL_SQUARED)
            or settings['mean_risk_scales'] != budget['mean_risk_scales']
            or settings['member_allowance_fraction'] != budget['member_allowance_fraction']):
        raise ValueError('Fidelity reference or allowance definition changed')
    return values, declaration['sha256']

def frozen_runtime(cfg, dataset, round3, device, manifest, *, seed=None):
    import torch
    from phycoflow_reconstruction.coherence import build_enabled_families
    from phycoflow_reconstruction.coherence.adaptive_fitting import validate_bank_target_binding
    from phycoflow_reconstruction.training.scalable_runtime import ScalableRuntime
    runs = sorted((round3 / 'runs/formal/FFM_SourceScaledRisk_Uniform_Repair').glob('*/artifacts/scalable_coherence.pt'))
    if len(runs) != 1:
        raise ValueError('Expected exactly one closed Round 3 SOURCE-scaled formal artifact')
    artifacts_path = runs[0]
    run = artifacts_path.parent.parent
    run_manifest = json.loads((run / 'run_manifest.json').read_text())
    artifacts = torch.load(artifacts_path, map_location='cpu', weights_only=True)
    provenance = artifacts['provenance']
    for key in ('source_checkpoint_sha256', 'normalizer_sha256', 'field_order'):
        if provenance[key] != manifest[key]:
            raise ValueError(f'Frozen observer {key} differs from TRAIN manifest')
    families = build_enabled_families(cfg['coherence'], dataset.data_spec, dataset.normalizer)
    if set(families) != {'global_distribution', 'multiscale_spectrum', 'topology'}:
        raise ValueError('Expected frozen State, Scale and Shape families')
    hashes = run_manifest.get('coherence_family_state_sha256s', {})
    family_hashes = {}
    for name, family in families.items():
        path = artifacts_path.parent / f'{name}_family.pt'
        family_hashes[name] = sha(path)
        if hashes.get(name) != family_hashes[name]:
            raise ValueError(f'Frozen {name} artifact hash differs from run manifest')
        family.load_state_artifact(torch.load(path, map_location='cpu', weights_only=True))
        family.to(device)
    runtime = ScalableRuntime(artifacts, families, steps_per_epoch=12,
                              expected_provenance=provenance, seed=(int(cfg['runtime']['seed']) + 800048 if seed is None else seed),
                              scoring_mode='conditional', allocation_mode='uniform')
    prefit_decl = cfg['coherence']['scalable_anchored_coherence']['common_prefit']
    prefit_path = Path(prefit_decl['path'])
    if sha(prefit_path) != prefit_decl['sha256']:
        raise ValueError('Frozen common prefit hash changed')
    prefit = json.loads(prefit_path.read_text())
    for key in ('source_checkpoint_sha256', 'normalizer_sha256', 'support_sha256', 'field_order', 'sample_ids'):
        if prefit['binding'].get(key) != provenance.get(key):
            raise ValueError(f'Frozen prefit binding differs for {key}')
    validate_bank_target_binding(prefit['binding'], runtime.bank)
    runtime.calibration.load_prefit(prefit)
    return runtime, {'scalable_sha256': sha(artifacts_path), 'family_sha256s': family_hashes,
                     'prefit_sha256': prefit_decl['sha256'], 'support_sha256': provenance['support_sha256']}


def load_rows(rowdir, count, manifest, manifest_hash, device, normalizer, *, indices=None):
    import torch
    rows = []
    selected = manifest['items'][:count] if indices is None else [manifest['items'][i] for i in indices]
    if len(selected) != count or len({item['sample_index'] for item in selected}) != count:
        raise ValueError('Teacher batch must contain unique manifest conditions')
    for item in selected:
        path = rowdir / f"{item['sample_index']:04d}.pt"
        sidecar = path.with_suffix('.sha256')
        if not sidecar.is_file() or sidecar.read_text().strip() != sha(path):
            raise ValueError(f'SOURCE row checksum missing or changed: {path}')
        row = torch.load(path, map_location='cpu', weights_only=True)
        if (row['sample_id'] != item['sample_id'] or row['sample_index'] != item['sample_index']
                or row['manifest_sha256'] != manifest_hash or list(row['member_seeds']) != item['member_seeds']
                or not torch.equal(row['normalizer_offset'], normalizer.offset.cpu())
                or not torch.equal(row['normalizer_scale'], normalizer.scale.cpu())):
            raise ValueError(f'SOURCE row identity mismatch: {path}')
        if rows and row['source_parameter_buffer_sha256'] != rows[0]['source_parameter_buffer_sha256']:
            raise ValueError('SOURCE rows have different parameter/buffer identities')
        rows.append(row)
    query = torch.tensor(manifest['query_indices'], dtype=torch.long)
    source = torch.cat([r['members_model'] for r in rows], dim=1).to(device=device, dtype=torch.float64)
    reference = torch.cat([r['truth_model'] for r in rows], dim=0).to(device=device, dtype=torch.float64)
    coords = torch.cat([r['coordinates_model'][:, query] for r in rows], dim=0).to(device)
    masses = torch.stack([r['native_masses'].reshape(-1) for r in rows]).to(device=device, dtype=torch.float64)
    if source.shape != (4, count, manifest['native_points'], len(manifest['field_order'])) or reference.shape != source.shape[1:]:
        raise ValueError('SOURCE row field dimensions changed')
    obs = torch.zeros(reference.shape, dtype=torch.bool, device=device)
    for i, r in enumerate(rows):
        idx, field = r['obs_indices'].reshape(-1).long(), r['obs_fields'].reshape(-1).long()
        if idx.tolist() != selected[i]['observation_indices'] or not (field == manifest['field_order'].index('T')).all():
            raise ValueError('SOURCE sensor entries differ from fixed manifest')
        if idx.shape != field.shape or (idx < 0).any() or (field < 0).any():
            raise ValueError('Invalid observation indices')
        observed_values = r['obs_values'].reshape(-1)
        truth_values = r['truth_model'][0, idx, field]
        if not torch.equal(observed_values, truth_values):
            raise ValueError('SOURCE row observation values differ from immutable reference')
        obs[i, idx.to(device), field.to(device)] = True
    mask = torch.ones((count, len(query)), dtype=torch.bool, device=device)
    support_mass = masses[:, query.to(device)]
    context = {'generated_weights': support_mass, 'reference_weights': support_mass,
               'generated_mask': mask, 'reference_mask': mask}
    return source, reference, coords, masses, obs, query.to(device), context, rows[0]['source_parameter_buffer_sha256']



def cached_scale_reference(cache, selected, reference, family):
    """Cache immutable reference transforms by condition in host memory."""
    import torch
    from phycoflow_reconstruction.coherence.query_runtime import _scale_statistics
    missing = [(position, index) for position, index in enumerate(selected) if index not in cache]
    if missing:
        positions = torch.tensor([position for position, _ in missing], device=reference.device)
        with torch.no_grad():
            values = _scale_statistics(family, reference.index_select(0, positions))
        for position, (_, index) in enumerate(missing):
            cache[index] = tuple(value[position].detach().cpu().clone() for value in values)
    return tuple(torch.stack([cache[index][part] for index in selected]).to(reference.device)
                 for part in range(3))

def qualify(args):
    device = device_identity()
    import torch
    from phycoflow_reconstruction.data.factory import open_field_dataset
    from phycoflow_reconstruction.data.training_batches import fixed_query_indices
    from phycoflow_reconstruction.evaluation.adaptive_campaign import atomic_json, locked_ledger, spent
    from phycoflow_reconstruction.coherence.empirical_teacher import empirical_family_scores, normalized_teacher_loss, teacher_step, FAMILIES
    from phycoflow_reconstruction.training.source import source_checkpoint_path
    campaign, rowdir = args.campaign.resolve(), args.rowdir.resolve()
    manifest_path = campaign / 'configs/train_manifest.json'
    manifest_hash = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    cfg, binding = resolve_config(args.round3)
    expected_query = fixed_query_indices(manifest['native_points'], 4096, seed=int(cfg['coherence']['compute_budget']['query_seed']))
    if expected_query.tolist() != manifest['query_indices']:
        raise ValueError('TRAIN manifest support differs from frozen query operator')
    if binding != manifest['config'] or sha(source_checkpoint_path(cfg)) != manifest['source_checkpoint_sha256']:
        raise ValueError('Inherited SOURCE or configuration changed')
    allowance, allowance_sha = frozen_allowance(cfg, manifest)
    token, started = uuid.uuid4().hex, time.time()
    ledger = campaign / 'operations/gpu_ledger.json'
    with locked_ledger(ledger) as state:
        if any('max_seconds' not in entry for entry in state['active'].values()):
            raise ValueError('Cannot budget an existing unbounded GPU lease')
        reserved = sum(max(0., float(x['max_seconds']) - (started - x['start_unix'])) for x in state['active'].values())
        if spent(state) + reserved + args.max_seconds > 86400 or any(x['gpu_uuid'] == device['uuid'] for x in state['active'].values()):
            raise ValueError('GPU ownership or campaign budget unavailable')
        state['active'][token] = {'gpu_uuid':device['uuid'], 'phase':'teacher_numerical_qualification',
                                  'start_unix':started, 'pid':os.getpid(), 'max_seconds':args.max_seconds}
    receipt_path = campaign / 'operations/teacher_jobs' / f'{token}.json'
    receipt = {'schema':'car_round4_teacher_numerical_qualification_v1', 'status':'running',
               'device':device, 'token':token, 'manifest_sha256':manifest_hash, 'rowdir':str(rowdir),
               'start_unix':started, 'max_seconds':args.max_seconds, 'script_sha256':sha(__file__),
               'purpose':'execution and cost qualification only; not learning efficacy', 'checks':[]}
    atomic_json(receipt_path, receipt)
    old = signal.getsignal(signal.SIGALRM)
    def alarm(_signum, _frame): raise TimeoutError('Teacher qualification lease elapsed')
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(args.max_seconds - 5)
    dataset = None
    try:
        torch.set_num_threads(4)
        dataset = open_field_dataset(cfg['dataset'], split='train')
        if len(dataset) != manifest['population'] or dataset.normalizer.digest() != manifest['normalizer_sha256']:
            raise ValueError('TRAIN dataset identity or normalization changed')
        runtime, identities = frozen_runtime(cfg, dataset, args.round3, 'cuda:0', manifest)
        receipt['observer'] = identities
        for count in args.batch_sizes:
            wall = time.perf_counter(); cpu = time.process_time()
            source, reference, coords, masses, obs, query, context, source_identity = load_rows(rowdir, count, manifest, manifest_hash, 'cuda:0', dataset.normalizer)
            if 'source_parameter_buffer_sha256' in receipt and receipt['source_parameter_buffer_sha256'] != source_identity:
                raise ValueError('SOURCE parameter/buffer identity changed between checks')
            receipt['source_parameter_buffer_sha256'] = source_identity
            torch.cuda.synchronize(); io_seconds = time.perf_counter() - wall
            runtime.begin_step(runtime.step)
            def loss_fn(native):
                scores = empirical_family_scores(runtime, native, reference, lambda x: x.index_select(-2, query),
                                                 coordinates=coords, context=context)
                return normalized_teacher_loss(scores, dict.fromkeys(FAMILIES, 1.0))
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            updated, info = teacher_step(source, source, reference, masses,
                                         torch.tensor(allowance, device=source.device, dtype=source.dtype),
                                         obs, loss_fn)
            torch.cuda.synchronize(); update_seconds = time.perf_counter() - start
            runtime.finish_step(True)
            start = time.perf_counter()
            correction = (updated - source).index_select(2, query).cpu()
            if not torch.isfinite(correction).all(): raise ValueError('Nonfinite correction')
            torch.cuda.synchronize(); output_seconds = time.perf_counter() - start
            receipt['checks'].append({'batch_size':count, 'input_wall_seconds':io_seconds,
                 'update_wall_seconds':update_seconds, 'output_wall_seconds':output_seconds,
                 'total_wall_seconds':time.perf_counter()-wall, 'cpu_process_seconds':time.process_time()-cpu,
                 'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
                 'max_gradient_norm':float(info['gradient_norm'].max().cpu()),
                 'min_accepted_alpha':float(info['alpha'].min().cpu()),
                 'all_finite':True, 'all_native_constraints_checked_fp64':True,
                 'field_units':'model field units', 'epsilon_units':'squared model field units'})
            atomic_json(receipt_path, receipt)
            del source, reference, coords, masses, obs, context, updated, info, correction
        receipt['status'] = 'completed'
    except BaseException as exc:
        receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)
        if torch.cuda.is_initialized():
            try: torch.cuda.synchronize()
            except Exception as exc: receipt.update(status='failed', teardown_error=str(exc))
        if dataset is not None: dataset.close()
        receipt['end_unix'] = time.time(); receipt['wall_seconds'] = receipt['end_unix'] - started
        with locked_ledger(ledger) as state:
            entry = state['active'].pop(token)
            state['closed'].append({**entry, 'token':token, 'end_unix':receipt['end_unix'], 'seconds':receipt['wall_seconds']})
        atomic_json(receipt_path, receipt)
        print(receipt_path, flush=True)



def atomic_checkpoint(path, state):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    os.replace(temp, path)



def _semantic_equal(left, right):
    """Compare JSON content while ignoring mapping-key order and number formatting."""
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_semantic_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_semantic_equal(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def _changed_paths(saved, incoming, prefix=''):
    if isinstance(saved, dict) and isinstance(incoming, dict):
        paths = []
        for key in sorted(set(saved) | set(incoming)):
            path = f'{prefix}.{key}' if prefix else key
            if key not in saved or key not in incoming:
                paths.append(path)
            else:
                paths.extend(_changed_paths(saved[key], incoming[key], path))
        return paths
    return [] if _semantic_equal(saved, incoming) else [prefix]


def _ordered_names(values, label, *, columns=False):
    if not isinstance(values, list):
        raise ValueError(f'{label} must be an ordered list')
    names = [item.get('name') if isinstance(item, dict) else None for item in values] if columns else values
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f'{label} must contain named definitions')
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f'{label} contains duplicate definitions: {duplicates}')
    return names


def validate_resume_identity(saved, incoming, *, origin='saved configuration'):
    """Explain scientific and ordered-recording incompatibility without mutation."""
    if not isinstance(saved, dict) or not isinstance(incoming, dict):
        raise ValueError(f'{origin} identity must be a mapping')
    saved_columns = saved.get('recording_columns', [])
    incoming_columns = incoming.get('recording_columns', [])
    old = _ordered_names(saved_columns, f'{origin} recording_columns', columns=True)
    new = _ordered_names(incoming_columns, 'incoming recording_columns', columns=True)
    old_set, new_set = set(old), set(new)
    common = old_set & new_set
    old_common = [name for name in old if name in common]
    new_common = [name for name in new if name in common]
    saved_by_name = {item['name']:item for item in saved_columns}
    incoming_by_name = {item['name']:item for item in incoming_columns}
    def lookup(item, path):
        for part in path.split('.'):
            if not isinstance(item, dict) or part not in item:
                return None
            item = item[part]
        return item
    semantic_changes = {}
    for name in old_common:
        paths = _changed_paths(saved_by_name[name], incoming_by_name[name])
        if paths:
            semantic_changes[name] = {key:{'saved':lookup(saved_by_name[name], key),
                                           'incoming':lookup(incoming_by_name[name], key)} for key in paths}
    added = [name for name in new if name not in old_set]
    missing = [name for name in old if name not in new_set]
    reordered = {'saved':old_common, 'incoming':new_common} if old_common != new_common else None
    old_order = _ordered_names(saved.get('accumulation_order', []), f'{origin} accumulation_order')
    new_order = _ordered_names(incoming.get('accumulation_order', []), 'incoming accumulation_order')
    old_order_set, new_order_set = set(old_order), set(new_order)
    shared_order = old_order_set & new_order_set
    old_shared = [name for name in old_order if name in shared_order]
    new_shared = [name for name in new_order if name in shared_order]
    order_changes = {'added':[name for name in new_order if name not in old_order_set],
                     'missing':[name for name in old_order if name not in new_order_set],
                     'reordered':{'saved':old_shared, 'incoming':new_shared} if old_shared != new_shared else None}
    excluded = {'recording_columns','accumulation_order'}
    scientific = _changed_paths({k:v for k,v in saved.items() if k not in excluded},
                                {k:v for k,v in incoming.items() if k not in excluded})
    if added or missing or reordered or semantic_changes or any(order_changes.values()) or scientific:
        detail = {'columns':{'added':added, 'missing':missing, 'reordered':reordered,
                             'semantically_changed':semantic_changes},
                  'accumulation_order':order_changes, 'scientific_identity_changed':scientific}
        raise ValueError(f'{origin} is incompatible: {json.dumps(detail, sort_keys=True, ensure_ascii=False)}')


def load_resume_checkpoint(scientific_path, checkpoint_path, incoming_identity):
    """Validate saved definitions and checkpoint before history append or writes."""
    import torch
    recorded = json.loads(scientific_path.read_text())
    validate_resume_identity(recorded.get('identity'), incoming_identity, origin='saved configuration')
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    checkpoint_identity = {key:saved[key] for key in recorded['identity'] if key in saved}
    validate_resume_identity(checkpoint_identity, recorded['identity'], origin='saved checkpoint')
    if not _semantic_equal(recorded.get('normalizers'), saved.get('normalizers')):
        raise ValueError('Saved teacher V normalizers differ from checkpoint')
    return recorded, saved

def append_history(path, rows):
    """Append only missing committed rows after checkpoint recovery."""
    existing = [] if not path.exists() else [json.loads(line) for line in path.read_text().splitlines()]
    if existing != rows[:len(existing)]:
        raise ValueError('Teacher history differs from checkpoint definitions or data')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        for row in rows[len(existing):]:
            f.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
        f.flush()
        os.fsync(f.fileno())


def plot_history(path, history, field_names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    epochs = [r['epoch'] for r in history]
    marker = {'marker':'o'} if len(history) == 1 else {}
    for key in ('global_distribution', 'multiscale_spectrum', 'topology'):
        axes[0,0].plot(epochs, [r['family_v'][key] for r in history], label=key, **marker)
    axes[0,0].set_ylabel('Sampled normalized V score'); axes[0,0].legend()
    axes[0,1].plot(epochs, [r['mean_alpha'] for r in history], **marker)
    axes[0,1].set_ylabel('Mean accepted ray factor')
    for ax, key, label in ((axes[1,0], 'field_displacement', 'Displacement'),
                           (axes[1,1], 'field_member_spread', 'Member spread'),
                           (axes[2,0], 'min_field_slack', 'Minimum feasibility slack')):
        for index, name in enumerate(field_names):
            ax.plot(epochs, [r[key][index] for r in history], label=name, **marker)
        ax.set_ylabel(label + ' (squared model units)'); ax.legend()
    axes[2,1].plot(epochs, [r['elapsed_seconds'] for r in history], label='epoch update wall', **marker)
    axes[2,1].plot(epochs, [r['cpu_seconds'] for r in history], label='process CPU', **marker)
    axes[2,1].set_ylabel('Seconds'); axes[2,1].legend()
    for ax in axes.flat: ax.set_xlabel('Reporting epoch')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    for output in (path, path.with_suffix('.png')):
        temp = output.with_name(output.stem + '.tmp' + output.suffix)
        fig.savefig(temp)
        os.replace(temp, output)
    plt.close(fig)



def endpoint_field_sums(source, candidate, reference, masses, observed):
    """Per-field sums from one existing TRAIN replay batch in model units squared.

    The finite-law mean is the actual mean of four jointly optimized atoms; no
    independent-draw variance correction or signed-risk estimator is applied.
    """
    import torch
    if source.ndim != 4 or source.shape[0] != 4 or candidate.shape != source.shape or reference.shape != source.shape[1:]:
        raise ValueError('Endpoint diagnostic requires aligned K4 native fields')
    if observed.shape != reference.shape or masses.shape != reference.shape[:2]:
        raise ValueError('Endpoint diagnostic native mass or sensor shape changed')
    if not all(torch.isfinite(x).all() for x in (source,candidate,reference,masses)) or (masses < 0).any() or (masses.sum(-1) <= 0).any():
        raise ValueError('Endpoint diagnostic requires finite fields and positive physical mass')
    weight=masses/masses.sum(-1,keepdim=True)
    error=candidate-reference.unsqueeze(0)
    mean=candidate.mean(0)
    spread=candidate-mean.unsqueeze(0)
    displacement=candidate-source
    weighted=weight[None,:,:,None]
    member=(error.square()*weighted).sum(2).mean(0).sum(0)
    mean_mse=((mean-reference).square()*weight[:,:,None]).sum(1).sum(0)
    member_spread=(spread.square()*weighted).sum(2).mean(0).sum(0)
    source_displacement=(displacement.square()*weighted).sum(2).mean(0).sum(0)
    observed_sum=(error.square()*observed.unsqueeze(0)).sum((0,1,2))
    observed_count=observed.sum((0,1)).to(torch.int64)*source.shape[0]
    return {'finite_law_mean_mse':mean_mse, 'member_mse':member, 'member_spread':member_spread,
            'source_displacement':source_displacement, 'observed_entry_mse':observed_sum,
            'observation_entry_counts':observed_count}


def finish_endpoint_field_diagnostics(sums, sample_count, field_order, scale, field_units):
    import torch
    if sample_count <= 0 or len(field_order) != len(scale) or len(field_units) != len(field_order):
        raise ValueError('Endpoint diagnostic requires matching field definitions and sample count')
    counts=sums['observation_entry_counts'].cpu().tolist()
    model={}
    for name in ('finite_law_mean_mse','member_mse','member_spread','source_displacement'):
        model[name]=(sums[name].double().cpu()/sample_count).tolist()
    obs=sums['observed_entry_mse'].double().cpu().tolist()
    model['observed_entry_mse']=[value/count if count else None for value,count in zip(obs,counts,strict=True)]
    factors=torch.as_tensor(scale,dtype=torch.float64).cpu().square().tolist()
    physical={name:[None if value is None else value*factor for value,factor in zip(values,factors,strict=True)]
              for name,values in model.items()}
    return {'schema':'car_round4_teacher_train800_fields_v1', 'split':'TRAIN',
            'sample_count':sample_count, 'members':4, 'field_order':list(field_order),
            'physical_field_units':list(field_units),
            'definition':'exact finite-law mean MSE and mean member MSE/spread on normalized native physical mass; observed-entry MSE averages the actual member-sensor entries',
            'model_field_units_squared':model, 'decoded_field_units_squared':physical,
            'observation_entry_counts':counts}

def configure_numerical_execution(cfg):
    """Apply inherited numerical settings before CUDA and checkpoint restore."""
    import torch
    from phycoflow_reconstruction.utils.reproducibility import seed_everything
    deterministic = bool(cfg['runtime'].get('deterministic', True))
    strict = bool(cfg.get('coherence', {}).get('scalable_anchored_coherence', {}).get('common_prefit'))
    seed_everything(int(cfg['runtime'].get('seed', 42)), deterministic,
                    strict_algorithms=strict)
    return {'deterministic': deterministic,
            'strict_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark}


def validate_driver_amendment(origin, execution):
    """Allow only an operational teacher-driver amendment to an intact capsule."""
    import ast
    from phycoflow_reconstruction.evaluation.adaptive_campaign import _capsule_inventory
    origin, execution = Path(origin).resolve(strict=True), Path(execution).resolve(strict=True)
    old, new = _capsule_inventory(origin), _capsule_inventory(execution)
    driver='scripts/posttraining/run_car_round4_teacher.py'
    if set(old)!=set(new) or {name for name in old if old[name]!=new[name]}!={driver}:
        raise ValueError('Teacher amendment must change only the teacher driver')
    def body(root):
        tree=ast.parse((root/driver).read_text())
        defs={node.name:node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
        formal=defs['formal']
        def dump(node): return ast.dump(node, include_attributes=False)
        module_setup=[dump(node) for node in tree.body
                      if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.If))]
        immutable={name:dump(node) for name,node in defs.items()
                   if name not in ('formal','main','append_history','plot_history',
                                   'validate_driver_amendment','validate_resource_admission')}
        nested={node.name:dump(node) for node in ast.walk(formal) if isinstance(node,ast.FunctionDef)
                and node.name in ('rows','score_batch','panel_scores')}
        epoch_loop=next(node for node in ast.walk(formal) if isinstance(node,ast.For)
                        and isinstance(node.target,ast.Name) and node.target.id=='epoch')
        start=next(i for i,node in enumerate(epoch_loop.body) if isinstance(node,ast.Assign)
                   and any(isinstance(target,ast.Name) and target.id=='epoch_start' for target in node.targets))
        end=next(i for i,node in enumerate(epoch_loop.body) if isinstance(node,ast.Expr)
                 and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)
                 and node.value.func.id=='persist')
        return module_setup,immutable,nested,[dump(node) for node in epoch_loop.body[start:end]],[dump(node) for node in epoch_loop.orelse]
    try: old_science=body(origin); new_science=body(execution)
    except (KeyError,StopIteration,SyntaxError) as exc:
        raise ValueError('Teacher amendment scientific bodies are incomplete') from exc
    if old_science != new_science:
        raise ValueError('Teacher amendment changes frozen scoring, update, normalizer or replay bodies')


def validate_resource_admission(path, campaign, manifest_hash, attempt, origin, execution, checkpoint_path, config_path, history_path):
    """Validate a bound, main-authored allocation of the declared GPU reserve."""
    import math
    path=Path(path).resolve(strict=True); campaign=Path(campaign).resolve(strict=True)
    if not path.is_relative_to(campaign): raise ValueError('Resource admission must be campaign-owned')
    receipt=json.loads(path.read_text())
    if (receipt.get('schema')!='car_round4_teacher_resource_admission_v1' or receipt.get('admitted') is not True
            or receipt.get('attempt')!=attempt or receipt.get('train_manifest_sha256')!=manifest_hash
            or receipt.get('origin_code_root')!=str(origin) or receipt.get('execution_code_root')!=str(execution)
            or receipt.get('max_epochs')!=100 or receipt.get('teacher_cpu_cap_seconds')!=21600
            or receipt.get('whole_gpu_cap_seconds')!=86400 or receipt.get('candidate_gpu_cap_seconds')!=35317):
        raise ValueError('Resource admission identity or fixed limits differ')
    def bound(name, expected=None):
        ref=receipt.get(name)
        if not isinstance(ref,dict) or set(ref)!={'path','sha256'} or not Path(ref['path']).is_absolute():
            raise ValueError(f'Invalid resource admission {name} binding')
        target=Path(ref['path']).resolve(strict=True)
        if not target.is_relative_to(campaign) or sha(target)!=ref['sha256'] or (expected is not None and sha(expected)!=ref['sha256']):
            raise ValueError(f'Resource admission {name} binding differs')
        return target
    bound('checkpoint',checkpoint_path); bound('scientific_config',config_path); bound('history',history_path)
    forecast=json.loads(bound('forecast').read_text())
    cap=receipt.get('teacher_gpu_cap_seconds'); allocated=receipt.get('reserve_allocated_seconds'); available=receipt.get('reserve_available_seconds')
    def nonnegative(x): return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>=0
    if (not all(nonnegative(x) for x in (cap,allocated,available)) or not 14400<cap<=32400
            or allocated!=cap-14400 or allocated>available or available>18000):
        raise ValueError('Teacher GPU reserve allocation exceeds the remaining declared reserve')
    envelope=forecast
    if (envelope.get('base_teacher_gpu_allowance_seconds')!=14400 or envelope.get('variation_reserve_gpu_seconds')!=18000
            or envelope.get('candidate_gpu_cap_seconds')!=35317 or envelope.get('whole_gpu_cap_seconds')!=86400
            or any(not nonnegative(envelope.get(k)) for k in ('reserve_committed_elsewhere_gpu_seconds','candidate_other_gpu_seconds','whole_other_gpu_seconds'))
            or available!=18000-envelope['reserve_committed_elsewhere_gpu_seconds']):
        raise ValueError('Warm-cache forecast reserve envelope differs from authorized limits')
    effective=min(14400+available,35317-envelope['candidate_other_gpu_seconds'],86400-envelope['whole_other_gpu_seconds'])
    if forecast.get('teacher_gpu_cap_seconds')!=effective or cap>effective:
        raise ValueError('Teacher allocation exceeds effective reserve, candidate or campaign limit')
    limits={'projected_teacher_gpu_seconds':cap,'projected_teacher_cpu_worker_seconds':21600,
            'projected_whole_gpu_seconds':86400,'projected_candidate_gpu_seconds':35317}
    if any(not nonnegative(receipt.get(k)) or not 0<receipt[k]<=limit for k,limit in limits.items()):
        raise ValueError('Resource admission forecast exceeds an authorized limit')
    if type(receipt.get('completed_epochs')) is not int or not 1<=receipt['completed_epochs']<100:
        raise ValueError('Resource admission requires a partial teacher age')
    if (forecast.get('schema')!='car_round4_teacher_warm_cache_forecast_v1'
            or forecast.get('train_manifest_sha256')!=manifest_hash
            or forecast.get('completed_epochs')!=receipt.get('completed_epochs')
            or forecast.get('teacher_checkpoint_sha256')!=receipt['checkpoint']['sha256']
            or forecast.get('teacher_history_sha256')!=receipt['history']['sha256']
            or forecast.get('teacher_cpu_cap_seconds')!=21600 or forecast.get('safety_factor')!=1.2
            or forecast.get('remaining_epochs')!=100-receipt['completed_epochs']
            or any(forecast.get(k)!=receipt[k] for k in limits)):
        raise ValueError('Resource admission differs from its measured warm-cache forecast')
    bound('ledger',campaign/'operations/gpu_ledger.json')
    validate_driver_amendment(origin,execution)
    return receipt


def formal(args):
    """One bounded teacher attempt. Never called by the qualification command."""
    device = device_identity()
    import torch
    from phycoflow_reconstruction.data.factory import open_field_dataset
    from phycoflow_reconstruction.data.training_batches import fixed_query_indices
    from phycoflow_reconstruction.evaluation.adaptive_campaign import atomic_json, locked_ledger, spent, _capsule_inventory
    from phycoflow_reconstruction.coherence.empirical_teacher import empirical_family_scores, normalized_teacher_loss, teacher_step, FAMILIES
    from phycoflow_reconstruction.coherence.teacher_stream import TeacherStream
    from phycoflow_reconstruction.training.source import source_checkpoint_path
    campaign = args.campaign.resolve()
    manifest_path = campaign / 'configs/train_manifest.json'
    manifest_hash = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    cfg, binding = resolve_config(args.round3)
    if binding != manifest['config'] or sha(source_checkpoint_path(cfg)) != manifest['source_checkpoint_sha256']:
        raise ValueError('Inherited SOURCE or configuration changed')
    query = fixed_query_indices(manifest['native_points'], 4096, seed=int(cfg['coherence']['compute_budget']['query_seed']))
    if query.tolist() != manifest['query_indices']:
        raise ValueError('Frozen support changed')
    gate = json.loads((campaign / 'operations/feasibility_gate.json').read_text())
    if (gate.get('admitted') is not True or gate.get('manifest_sha256') != manifest_hash
            or not 0 < gate.get('candidate_inclusive_gpu_seconds', 0) <= 35317
            or not 0 < gate.get('whole_round_gpu_seconds', 0) <= 86400):
        raise ValueError('Formal teacher needs manifest-bound inclusive-cost admission')
    _capsule_inventory(CODE)
    bank = campaign / 'artifacts/source_bank'
    bank_receipt = json.loads((bank / 'receipt.json').read_text())
    if (bank_receipt.get('schema') != 'car_round4_source_bank_v1' or bank_receipt.get('status') != 'completed'
            or bank_receipt.get('manifest_sha256') != manifest_hash or bank_receipt.get('population') != manifest['population']
            or bank_receipt.get('source_checkpoint_sha256') != manifest['source_checkpoint_sha256']
            or len(bank_receipt.get('rows_sha256', [])) != manifest['population']):
        raise ValueError('Complete matching full-membership SOURCE bank required')
    allowance, allowance_sha = frozen_allowance(cfg, manifest)
    run = campaign / 'runs/formal/SSS_Empirical_Endpoint_Teacher' / args.attempt
    scientific_path = run / 'configs/resolved_scientific_config.json'
    checkpoint_path = run / 'checkpoints/teacher_latest.pt'
    history_path = run / 'logs/teacher_history.jsonl'
    admission=None; origin=CODE; teacher_gpu_cap=14400
    if args.resource_admission:
        if not args.resume: raise ValueError('Reserve admission is only valid for the existing teacher resume')
        saved_definition=json.loads(scientific_path.read_text())
        origin=Path(saved_definition['identity']['code_root']).resolve(strict=True)
        admission=validate_resource_admission(args.resource_admission,campaign,manifest_hash,args.attempt,
                                              origin,CODE,checkpoint_path,scientific_path,history_path)
        teacher_gpu_cap=admission['teacher_gpu_cap_seconds']
    token, started = uuid.uuid4().hex, PROCESS_STARTED
    ledger = campaign / 'operations/gpu_ledger.json'
    with locked_ledger(ledger) as state:
        if admission and sha(ledger)!=admission['ledger']['sha256']:
            raise ValueError('GPU ledger changed since resource admission')
        if any('max_seconds' not in entry for entry in state['active'].values()):
            raise ValueError('Cannot budget an existing unbounded GPU lease')
        reserved = sum(max(0., float(x['max_seconds']) - (started - x['start_unix'])) for x in state['active'].values())
        prior_teacher = [x for x in state['closed'] if x.get('phase') == 'endpoint_teacher']
        if any('cpu_seconds' not in x for x in prior_teacher):
            raise ValueError('Prior teacher CPU cost is unavailable')
        teacher_spent = sum(float(x.get('seconds', 0)) for x in prior_teacher)
        teacher_cpu_spent = sum(float(x['cpu_seconds']) for x in prior_teacher)
        if (spent(state) + reserved + args.max_seconds > 86400 or teacher_spent + args.max_seconds > teacher_gpu_cap
                or any(x['gpu_uuid'] == device['uuid'] for x in state['active'].values())):
            raise ValueError('GPU ownership or campaign/teacher budget unavailable')
        state['active'][token] = {'gpu_uuid':device['uuid'], 'phase':'endpoint_teacher', 'start_unix':started,
                                  'pid':os.getpid(), 'max_seconds':args.max_seconds}
    receipt_path = campaign / 'operations/teacher_jobs' / f'{token}.json'
    receipt = {'schema':'car_round4_teacher_job_v1', 'status':'running', 'device':device, 'token':token,
               'manifest_sha256':manifest_hash, 'start_unix':started, 'max_seconds':args.max_seconds,
               'script_sha256':sha(__file__), 'code_root':str(CODE),
               'reference_cache':'Scale immutable transforms cached on CPU by condition; Shape reference diagrams recomputed'}
    if admission:
        receipt.update(resource_admission={'path':str(Path(args.resource_admission).resolve()),'sha256':sha(args.resource_admission)},
                       origin_code_root=str(origin), teacher_gpu_cap_seconds=teacher_gpu_cap)
    atomic_json(receipt_path, receipt)
    old = signal.getsignal(signal.SIGALRM)
    def alarm(_signum, _frame): raise TimeoutError('Teacher job reached its bounded lease')
    signal.signal(signal.SIGALRM, alarm)
    remaining = args.max_seconds - int(time.time() - started) - 5
    signal.alarm(max(1, remaining))
    dataset = None
    cpu_started = PROCESS_CPU_STARTED; cpu_carried = teacher_cpu_spent
    try:
        torch.set_num_threads(4)
        numerical_execution = configure_numerical_execution(cfg)
        dataset = open_field_dataset(cfg['dataset'], split='train')
        if len(dataset) != manifest['population'] or dataset.normalizer.digest() != manifest['normalizer_sha256']:
            raise ValueError('TRAIN membership or normalization changed')
        runtime, identities = frozen_runtime(cfg, dataset, args.round3, 'cuda:0', manifest)
        ids = [item['sample_id'] for item in manifest['items']]
        id_map = {sid:i for i,sid in enumerate(ids)}
        panel_ids = runtime.artifacts['provenance']['sample_ids']
        if len(panel_ids) != 800 or len(set(panel_ids)) != 800 or not set(panel_ids) <= set(ids):
            raise ValueError('Frozen approximately 10% TRAIN panel identity changed')
        panel = [id_map[sid] for sid in panel_ids]
        expected_source = bank_receipt['source_parameter_buffer_sha256']
        epsilon = torch.tensor(allowance, device='cuda:0', dtype=torch.float64)
        stream = TeacherStream(len(ids), seed=int(cfg['runtime']['seed']) + 221007)
        run.mkdir(parents=True, exist_ok=args.resume)
        columns = [
            {'name':'epoch','type':'int','units':'reporting epochs','meaning':'completed teacher epoch'},
            {'name':'visits','type':'int','units':'conditions','meaning':'objective-bearing condition visits'},
            {'name':'cumulative_visits','type':'int','units':'conditions','meaning':'objective-bearing visits across completed epochs'},
            {'name':'dataset_pass_equivalents','type':'float','units':'dataset passes','meaning':'cumulative visits divided by TRAIN population'},
            {'name':'batch_stream','type':'list[object]','units':'indices and dimensionless factors','meaning':'ordered manifest indices, actual batch sizes and accepted per-condition ray factors'},
            {'name':'family_v','type':'mapping[float]','units':'frozen SOURCE V units','meaning':'sampled family empirical scores'},
            {'name':'mean_alpha','type':'float','units':'dimensionless','meaning':'mean accepted SOURCE-ray factor'},
            {'name':'field_displacement','type':'list[float]','units':'squared model field units','meaning':'mean native physical-mass displacement per field'},
            {'name':'field_member_spread','type':'list[float]','units':'squared model field units','meaning':'mean native physical-mass member variance per field'},
            {'name':'min_field_slack','type':'list[float]','units':'squared model field units','meaning':'minimum of three native constraint slacks across visits'},
            {'name':'cpu_seconds','type':'float','units':'seconds','meaning':'process CPU time including input and descriptor work'},
            {'name':'elapsed_seconds','type':'float','units':'seconds','meaning':'epoch wall duration including data and updates'},
        ]
        identity = {'train_manifest_sha256':manifest_hash, 'observer':identities,
                    'numerical_execution':numerical_execution,
                    'source_parameter_buffer_sha256':expected_source, 'source_bank_receipt_sha256':sha(bank/'receipt.json'),
                    'query_indices':manifest['query_indices'], 'panel_ids':panel_ids,
                    'epsilon_model_squared':allowance, 'epsilon_reference_sha256':allowance_sha,
                    'epsilon_physical_squared':(torch.tensor(allowance, dtype=torch.float64) * dataset.normalizer.scale.cpu().double().square()).tolist(), 'code_root':str(origin),
                    'recording_columns':columns, 'accumulation_order':['global_distribution','multiscale_spectrum','topology']}
        def rows(indices):
            result = load_rows(bank, len(indices), manifest, manifest_hash, 'cuda:0', dataset.normalizer, indices=indices)
            if result[-1] != expected_source: raise ValueError('SOURCE bank parameter identity mismatch')
            for i in indices:
                if (bank/f'{i:04d}.sha256').read_text().strip() != bank_receipt['rows_sha256'][i]:
                    raise ValueError('SOURCE row differs from completed bank receipt')
            return result
        def apply_corrections(source, indices, support, corrections):
            result = source.clone()
            result[:, :, support, :] += corrections[indices].to(device=source.device).permute(1,0,2,3)
            return result
        scale_reference_cache = {}
        def score_batch(observer, source, reference, coords, support, context, selected):
            fixed_reference = cached_scale_reference(scale_reference_cache, selected,
                                reference.index_select(1, support), observer.families['multiscale_spectrum'])
            return empirical_family_scores(observer, source, reference, lambda value:value.index_select(-2, support),
                                           coordinates=coords, context={**context, 'compiled_scale_reference':fixed_reference})
        def panel_scores(seed, corrections, *, field_diagnostics=False):
            observer, _ = frozen_runtime(cfg, dataset, args.round3, 'cuda:0', manifest, seed=seed)
            totals = dict.fromkeys(FAMILIES, 0.0); number = 0; field_sums = None
            for start in range(0, len(panel), 64):
                if cpu_carried + time.process_time() - cpu_started >= 21600:
                    raise TimeoutError('Teacher CPU worker budget exhausted during panel scoring')
                selected = panel[start:start+64]
                source, ref, coords, masses, obs, support, context, _ = rows(selected)
                candidate = source if corrections is None else apply_corrections(source, selected, support, corrections)
                observer.begin_step(observer.step)
                with torch.no_grad():
                    scores = score_batch(observer, candidate, ref, coords, support, context, selected)
                    if field_diagnostics:
                        batch_sums = endpoint_field_sums(source, candidate, ref, masses, obs)
                        if field_sums is None: field_sums = {name:value.detach().cpu() for name,value in batch_sums.items()}
                        else:
                            for name,value in batch_sums.items(): field_sums[name] += value.detach().cpu()
                for name in FAMILIES: totals[name] += float(scores[name].sum().cpu())
                number += len(selected); observer.finish_step(True)
            means = {name: totals[name] / number for name in FAMILIES}
            if not field_diagnostics: return means
            diagnostics = finish_endpoint_field_diagnostics(field_sums, number, manifest['field_order'],
                             dataset.normalizer.scale, dataset.data_spec.field_units)
            return means, diagnostics
        if args.resume:
            recorded, saved = load_resume_checkpoint(scientific_path, checkpoint_path, identity)
            corrections = saved['corrections']; history = saved['history']; normalizers = saved['normalizers']
            if float(saved['cpu_worker_seconds']) > cpu_carried + 1.0:
                raise ValueError('Teacher checkpoint CPU cost exceeds completed job ledger')
            if corrections.dtype != torch.float64 or corrections.shape != (len(ids),4,len(query),len(manifest['field_order'])):
                raise ValueError('Teacher checkpoint correction bank shape changed')
            stream.load_state_dict(saved['stream']); runtime.load_state_dict(saved['runtime'])
            start_epoch = saved['completed_epochs']
            if admission and start_epoch!=admission['completed_epochs']:
                raise ValueError('Resource admission checkpoint age differs')
            if len(history) != start_epoch or runtime.step != 12 * start_epoch:
                raise ValueError('Teacher checkpoint epoch, history, and sampler cursor disagree')
            append_history(history_path, history)
        else:
            if checkpoint_path.exists() or history_path.exists(): raise FileExistsError('Teacher attempt already exists; use --resume')
            baseline = panel_scores(730901, None)
            normalizers = {name:max(float(baseline[name]), 1e-8) for name in FAMILIES}
            scientific_path.parent.mkdir(parents=True, exist_ok=True)
            scientific = {'schema':'car_round4_teacher_scientific_config_v1', 'identity':identity,
                          'normalizers':normalizers, 'normalizer_source_scores':baseline,
                          'normalizer_seed':730901, 'independent_replay_seed':819103,
                          'maximum_epochs':100, 'batches_per_epoch':12, 'maximum_batch_conditions':64,
                          'native_source_units':'model field units', 'correction_dtype':'float64',
                          'teacher_gradient_metric':'normalized native physical mass, mean over atoms and fields',
                          'gradient_clip_norm':1., 'dimensionless_step':1., 'observation_correction':'zero'}
            with scientific_path.open('x') as f:
                json.dump(scientific, f, indent=2, allow_nan=False); f.write('\n')
            corrections = torch.zeros((len(ids),4,len(query),len(manifest['field_order'])), dtype=torch.float64)
            history = []; start_epoch = 0
        def persist(epoch):
            state = {'schema':'car_round4_teacher_checkpoint_v1', **identity, 'corrections':corrections,
                     'completed_epochs':epoch, 'normalizers':normalizers, 'runtime':runtime.state_dict(),
                     'stream':stream.state_dict(), 'visits':stream.visits, 'history':history,
                     'cpu_worker_seconds':cpu_carried + time.process_time() - cpu_started}
            persist_start = time.perf_counter()
            atomic_checkpoint(checkpoint_path, state); append_history(history_path, history)
            if history and (epoch == 1 or epoch % 10 == 0):
                plot_history(run/'logs/teacher_v_scores.pdf', history, manifest['field_order'])
            receipt.setdefault('checkpoint_write_wall_seconds', []).append(time.perf_counter() - persist_start)
            atomic_json(receipt_path, receipt)
        for epoch in range(start_epoch + 1, 101):
            if (time.time() - started + args.epoch_reserve_seconds + args.closing_reserve_seconds >= args.max_seconds
                    or cpu_carried + time.process_time() - cpu_started + args.epoch_reserve_cpu_seconds + args.closing_reserve_cpu_seconds >= 21600):
                if not checkpoint_path.exists(): persist(len(history))
                if history: plot_history(run/'logs/teacher_v_scores.pdf', history, manifest['field_order'])
                receipt.update(status='paused', completed_epochs=len(history), checkpoint=str(checkpoint_path),
                               cpu_worker_seconds=cpu_carried + time.process_time() - cpu_started)
                break
            epoch_start = time.perf_counter(); epoch_cpu = time.process_time(); n = 0; totals = dict.fromkeys(FAMILIES, 0.); alpha_sum = 0.
            batch_stream = []
            displacement_sum = torch.zeros(len(epsilon), dtype=torch.float64, device=epsilon.device)
            spread_sum = torch.zeros_like(displacement_sum); min_slack = torch.full_like(displacement_sum, float('inf'))
            for _ in range(12):
                selected = stream.next_batch(64).tolist()
                source, ref, coords, masses, obs, support, context, _ = rows(selected)
                current = apply_corrections(source, selected, support, corrections)
                runtime.begin_step(runtime.step); captured = {}
                def loss_fn(value):
                    scores = score_batch(runtime, value, ref, coords, support, context, selected)
                    captured.update({name:scores[name].detach() for name in FAMILIES})
                    return normalized_teacher_loss(scores, normalizers)
                updated, info = teacher_step(source, current, ref, masses, epsilon, obs, loss_fn)
                runtime.finish_step(True)
                corrections[selected] = (updated - source).index_select(2, support).permute(1,0,2,3).cpu()
                for name in FAMILIES: totals[name] += float(captured[name].sum().cpu()) / normalizers[name]
                weight = masses / masses.sum(-1, keepdim=True)
                source_error = source - ref.unsqueeze(0)
                source_mean = (source_error.mean(0).square()*weight[:,:,None]).sum(1)
                source_member = (source_error.square()*weight[None,:,:,None]).sum(2).mean(0)
                spread = ((updated-updated.mean(0, keepdim=True)).square()*weight[None,:,:,None]).sum(2).mean(0)
                displacement_sum += info['displacement'].sum(0); spread_sum += spread.sum(0)
                min_slack = torch.minimum(min_slack, torch.stack((source_mean+epsilon-info['mean_error'], source_member+epsilon-info['member_error'], epsilon-info['displacement'])).amin((0,1)))
                batch_stream.append({'condition_indices':selected, 'batch_size':len(selected),
                                     'accepted_alpha':info['alpha'].cpu().tolist()})
                n += len(selected); alpha_sum += float(info['alpha'].sum().cpu())
            torch.cuda.synchronize()
            cumulative_visits = int(stream.visits.sum())
            history.append({'epoch':epoch, 'visits':n, 'cumulative_visits':cumulative_visits,
                            'dataset_pass_equivalents':cumulative_visits / len(ids), 'batch_stream':batch_stream,
                            'family_v':{name:totals[name]/n for name in FAMILIES},
                            'mean_alpha':alpha_sum/n, 'field_displacement':(displacement_sum/n).cpu().tolist(),
                            'field_member_spread':(spread_sum/n).cpu().tolist(), 'min_field_slack':min_slack.cpu().tolist(),
                            'cpu_seconds':time.process_time()-epoch_cpu, 'elapsed_seconds':time.perf_counter()-epoch_start})
            persist(epoch)
        else:
            source_scores, source_fields = panel_scores(819103, None, field_diagnostics=True)
            teacher_scores, teacher_fields = panel_scores(819103, corrections, field_diagnostics=True)
            changes = {name:(source_scores[name]-teacher_scores[name])/normalizers[name] for name in FAMILIES}
            # Every update was checked against all native constraints; replay the final
            # bank separately before issuing any admission receipt.
            violations = 0; zero = 0; total = 0
            for start in range(0, len(ids), 64):
                if cpu_carried + time.process_time() - cpu_started >= 21600:
                    raise TimeoutError('Teacher CPU worker budget exhausted during feasibility check')
                selected = list(range(start, min(start+64, len(ids))))
                source, ref, coords, masses, obs, support, context, _ = rows(selected)
                candidate = apply_corrections(source, selected, support, corrections)
                weight = masses / masses.sum(-1, keepdim=True)
                def measures(value):
                    err=value-ref.unsqueeze(0)
                    return ((err.mean(0).square()*weight[:,:,None]).sum(1),
                            (err.square()*weight[None,:,:,None]).sum(2).mean(0),
                            ((value-source).square()*weight[None,:,:,None]).sum(2).mean(0))
                base = measures(source); cur = measures(candidate)
                valid = ((cur[0] <= base[0]+epsilon).all(-1) & (cur[1] <= base[1]+epsilon).all(-1)
                         & (cur[2] <= epsilon).all(-1) & ((candidate-source).masked_fill(~obs.unsqueeze(0),0)==0).all((0,2,3)))
                violations += int((~valid).sum().cpu()); zero += int(((candidate-source)==0).all((0,2,3)).sum().cpu()); total += len(selected)
            admitted = violations == 0 and all(value >= -.02 for value in changes.values()) and any(value >= .05 for value in changes.values())
            # changes are improvement (SOURCE minus teacher), so regressions are negative.
            gate_doc = {'schema':'car_round4_teacher_qualification_v1', 'admitted':admitted,
                        'train_manifest_sha256':manifest_hash, 'teacher_checkpoint_sha256':sha(checkpoint_path),
                        'checkpoint':str(checkpoint_path.relative_to(campaign)), 'completed_epochs':100,
                        'panel_ids':panel_ids, 'source_scores':source_scores, 'teacher_scores':teacher_scores,
                        'train800_field_diagnostics':{'source':source_fields, 'teacher':teacher_fields},
                        'normalized_improvement':changes, 'violations':violations,
                        'zero_correction_fraction':zero/total, 'replay_seed':819103, 'feasibility_population':total, 'target_precision':'FP64 corrections; downstream student model targets use the native adapter dtype'}
            atomic_json(campaign/'operations/teacher_qualification.json', gate_doc)
            receipt.update(status='completed', completed_epochs=100, teacher_admitted=admitted, checkpoint=str(checkpoint_path))
    except BaseException as exc:
        receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc)); raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)
        if torch.cuda.is_initialized():
            try: torch.cuda.synchronize()
            except Exception as exc: receipt.update(status='failed', teardown_error=str(exc))
        if dataset is not None: dataset.close()
        receipt['end_unix'] = time.time(); receipt['wall_seconds'] = receipt['end_unix'] - started
        receipt['cpu_worker_seconds'] = cpu_carried + time.process_time() - cpu_started
        with locked_ledger(ledger) as state:
            entry = state['active'].pop(token)
            state['closed'].append({**entry, 'token':token, 'end_unix':receipt['end_unix'], 'seconds':receipt['wall_seconds'],
                                    'cpu_seconds':time.process_time() - cpu_started})
        atomic_json(receipt_path, receipt)
        print(receipt_path, flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--round3', type=Path, default=ROOT/'cases/turbulent_combustion/runs/coherence_adaptive_repair/round_3')
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('qualify')
    q.add_argument('--rowdir', type=Path, required=True)
    q.add_argument('--max-seconds', type=int, required=True)
    q.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 8, 64])
    t = sub.add_parser('train')
    t.add_argument('--max-seconds', type=int, required=True)
    t.add_argument('--epoch-reserve-seconds', type=int, required=True)
    t.add_argument('--attempt', required=True, help='Descriptive immutable attempt name')
    t.add_argument('--epoch-reserve-cpu-seconds', type=int, required=True)
    t.add_argument('--closing-reserve-seconds', type=int, required=True)
    t.add_argument('--closing-reserve-cpu-seconds', type=int, required=True)
    t.add_argument('--resume', action='store_true')
    t.add_argument('--resource-admission', type=Path, help='Bound reserve allocation for this same-lineage resume')
    args = p.parse_args()
    if args.command == 'qualify':
        if args.max_seconds <= 10 or args.max_seconds > 7200 or not 1 <= len(args.batch_sizes) <= 3 or any(not 1 <= n <= 64 for n in args.batch_sizes):
            p.error('Qualification permits at most three batch sizes in 1..64 and a lease of 11..7200 seconds')
        qualify(args)
    else:
        if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{7,127}', args.attempt)
                or args.epoch_reserve_seconds < 30 or args.epoch_reserve_cpu_seconds < 1
                or args.closing_reserve_seconds < 1 or args.closing_reserve_cpu_seconds < 1
                or args.max_seconds <= args.epoch_reserve_seconds + args.closing_reserve_seconds
                or args.max_seconds > (32400 if args.resource_admission else 14400)):
            p.error('Teacher training needs a descriptive attempt and measured positive epoch and final-scoring reserves')
        formal(args)

if __name__ == '__main__': main()
