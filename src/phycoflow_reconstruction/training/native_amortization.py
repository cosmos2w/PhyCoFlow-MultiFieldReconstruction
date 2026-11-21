"""Round 4 native RF continuation with fixed conditioning and finite-law targets.

The caller loads the identical SOURCE weight view and supplies its verified full
TRAIN dataset, sensor manifest and (for the repaired path) physical target bank.
This module never changes the reconstruction/inference interface.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch

from ..data.manifest import SensorManifest, dataset_fingerprint
from ..data.sensor_protocols import SensorProtocol, build_observation_batch
from ..utils.reproducibility import seed_everything
from .model_lifecycle import (add_training_aux_state, after_optimizer_step,
                              backward_and_clip_model_loss, load_training_aux_state)
from .run_store import checkpoint_model_state, config_digest, load_model_state_strict

# This ordered definition is owned by each run and is compared on resume.
COLUMNS = [
    {"name": "epoch", "type": "int", "unit": "reporting_epoch", "meaning": "completed reporting epoch"},
    {"name": "native_rf_loss", "type": "float", "unit": "native model loss", "meaning": "visit-weighted mean of adapter native losses"},
    {"name": "learning_rate", "type": "float", "unit": "dimensionless", "meaning": "AdamW learning rate at the last update of this epoch"},
    {"name": "visits", "type": "int", "unit": "conditions", "meaning": "objective-bearing condition visits this epoch"},
    {"name": "cumulative_visits", "type": "int", "unit": "conditions", "meaning": "objective-bearing condition visits since start"},
    {"name": "dataset_pass_equivalents", "type": "float", "unit": "passes", "meaning": "cumulative visits divided by TRAIN membership"},
    {"name": "epoch_seconds", "type": "float", "unit": "s", "meaning": "epoch data and updates excluding history, checkpoint and plot persistence"},
    {"name": "data_seconds", "type": "float", "unit": "s", "meaning": "batch and target preparation elapsed time"},
    {"name": "update_seconds", "type": "float", "unit": "s", "meaning": "forward, backward and optimizer elapsed time"},
    {"name": "batch_stream", "type": "list", "unit": "ordered batches", "meaning": "per batch ordered TRAIN indices, query seed, and sampled teacher atom indices or null for truth"},
]
ACCUMULATION = "sum(adapter native loss * batch condition count) in batch order / total condition visits"


def _plot(rows: list[dict], logs: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r['epoch'] for r in rows], [r['native_rf_loss'] for r in rows],
            marker='o' if len(rows) == 1 else None)
    ax.set(xlabel="Reporting epoch", ylabel="Native RF loss", title="Native training loss")
    fig.tight_layout()
    fig.savefig(logs / "native_loss.pdf")
    fig.savefig(logs / "native_loss.png", dpi=160)
    plt.close(fig)


def _atomic_save(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def run_native_amortization(*, model, dataset, manifest: SensorManifest,
                            config: Mapping, run_dir: str | Path,
                            target_reader: Callable | None = None,
                            resume: bool = False,
                            stop_after_epoch: Callable[[dict], bool] | None = None,
                            inference_config: Mapping | None = None) -> Path:
    """Train through ``config['epochs']`` with checkpointing at epoch boundaries.

    ``target_reader(sample_ids, atom_indices, query_indices)`` returns physical
    targets [B,Q,C], aligned with query indices and original dataset channel
    order. Encoding happens here exactly once. It must bind its bank identity,
    source, manifest, and support before being passed in. ``None`` means truth.
    Config contains seed, epochs, query_points, device, and immutable scientific
    identity (including source/bank/manifest/split hashes) under ``identity``.
    Epoch horizon and runtime device are permitted to change on resume.
    """
    cfg = dict(config)
    seed, epochs = int(cfg['seed']), int(cfg['epochs'])
    deterministic = bool(cfg.get('deterministic', True))
    strict_algorithms = bool(cfg.get('strict_algorithms', False))
    seed_everything(seed, deterministic, strict_algorithms=strict_algorithms)
    numerical_settings = {'deterministic': deterministic, 'strict_algorithms': strict_algorithms,
        'deterministic_algorithms_enabled': torch.are_deterministic_algorithms_enabled(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32}
    if epochs < 1 or len(dataset) < 1:
        raise ValueError('epochs and full TRAIN membership must be positive')
    device = torch.device(cfg.get('device', 'cpu'))
    if manifest.dataset_fingerprint != dataset_fingerprint(dataset.path):
        raise ValueError('sensor manifest dataset fingerprint mismatch')
    if manifest.split.lower() != 'train':
        raise ValueError('sensor manifest must belong to TRAIN')
    if manifest.query_indices is not None:
        raise ValueError('training queries must be sampled, not frozen in sensor manifest')
    expected_kind = 'teacher' if target_reader is not None else 'truth'
    identity = dict(cfg['identity'])
    if identity.get('target_kind') != expected_kind or identity.get('manifest_sha256') != manifest.digest():
        raise ValueError('target kind or manifest identity mismatch')
    if not all(identity.get(k) for k in ('source_sha256', 'membership_sha256', 'manifest_sha256')):
        raise ValueError('source, membership and manifest identities are required')
    if target_reader is not None and not identity.get('bank_sha256'):
        raise ValueError('teacher bank identity is required')
    # Manifest membership must equal the entire dataset, and each fixed sensor
    # set is exactly 256 unique T observations (T field index is explicit).
    t_index = int(cfg['temperature_field_index'])
    if hasattr(dataset, '_items') and hasattr(dataset, 'trajectory_ids'):
        ids = [f"{dataset.trajectory_ids[t]}:{frame if frame is not None else 'all'}"
               for t, frame in dataset._items]
    else:
        ids = [f"{s.trajectory_id}:{s.time_index if s.time_index is not None else 'all'}" for s in dataset]
    membership_hash = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
    if identity['membership_sha256'] != membership_hash:
        raise ValueError('ordered TRAIN membership hash does not match actual membership')
    if len(ids) != len(set(ids)) or set(ids) != set(manifest.indices):
        raise ValueError('manifest must cover exactly the full unique TRAIN membership')
    for sid in ids:
        pairs = manifest.indices[sid]
        if len(pairs) != 256 or len({tuple(p) for p in pairs}) != 256 or any(int(p[1]) != t_index for p in pairs):
            raise ValueError(f'{sid}: fixed manifest must have 256 unique T observations')
    root = Path(run_dir)
    logs = root / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    definition = {'columns': COLUMNS, 'accumulation_order': ACCUMULATION}
    scientific = {'seed': seed, 'query_points': int(cfg['query_points']),
                  'numerical_settings': numerical_settings,
                  'native_query_microbatch_size': getattr(model, 'train_query_microbatch_size', None),
                  'temperature_field_index': t_index, 'identity': identity,
                  'batches_per_epoch': 12, 'batch_size': 64,
                  'teacher_target_cast': 'physical float64 normalized once then cast to native target dtype',
                  'warmup_epochs': 50, 'initial_lr': 5e-6, 'final_lr': 5e-5,
                  'weight_decay': 1e-6, 'grad_clip': 1.0}
    state_path, history_path = root / 'checkpoint.pt', logs / 'history.jsonl'
    generator = torch.Generator().manual_seed(seed + 17)
    query_generator = torch.Generator().manual_seed(seed + 37)
    atom_generator = torch.Generator().manual_seed(seed + 59)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-6)
    if resume:
        checkpoint = torch.load(state_path, map_location='cpu', weights_only=False)
        if checkpoint['scientific'] != scientific:
            old = checkpoint['scientific']
            changed = {k: {'saved': old.get(k), 'incoming': scientific.get(k)} for k in old.keys() | scientific.keys() if old.get(k) != scientific.get(k)}
            raise ValueError(f'incompatible scientific definitions: {changed}')
        saved_definition = json.loads((logs / 'recording_definitions.json').read_text())
        if saved_definition != checkpoint['recording']:
            raise ValueError('saved recording file disagrees with checkpoint recording definitions')
        if saved_definition != definition:
            old_columns = saved_definition['columns']
            new_columns = definition['columns']
            old_names = [column['name'] for column in old_columns]
            new_names = [column['name'] for column in new_columns]
            changes = {'added': [n for n in new_names if n not in old_names],
                       'missing': [n for n in old_names if n not in new_names],
                       'saved_order': old_names, 'incoming_order': new_names,
                       'semantically_changed': [n for n in new_names if n in old_names and
                           new_columns[new_names.index(n)] != old_columns[old_names.index(n)]],
                       'accumulation_changed': saved_definition['accumulation_order'] != definition['accumulation_order']}
            raise ValueError(f'incompatible recording definitions: {changes}')
        load_model_state_strict(model, checkpoint['model'])
        load_training_aux_state(model, checkpoint)
        optimizer.load_state_dict(checkpoint['optimizer'])
        generator.set_state(checkpoint['condition_rng'])
        query_generator.set_state(checkpoint['query_rng'])
        atom_generator.set_state(checkpoint['atom_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        random.setstate(checkpoint['python_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        rows = checkpoint['history']
        # Recovery from an interrupted history/checkpoint write uses the checkpoint
        # as authoritative, with no migration on an ordinary resume.
        on_disk = [json.loads(line) for line in history_path.read_text().splitlines() if line]
        if on_disk != rows:
            raise ValueError('history and checkpoint disagree; preserve both for diagnosis')
        permutation, position = checkpoint['permutation'], checkpoint['position']
        counts = checkpoint['visit_counts']
        start = checkpoint['epoch']
    else:
        if state_path.exists() or history_path.exists():
            raise FileExistsError('run already contains native history or checkpoint')
        torch.manual_seed(seed)
        if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
        random.seed(seed); np.random.seed(seed)
        permutation, position, counts, rows, start = [], 0, [0] * len(dataset), [], 0
        (root / 'resolved_config.json').write_text(json.dumps(cfg, indent=2, sort_keys=True) + '\n')
        (logs / 'recording_definitions.json').write_text(json.dumps(definition, indent=2) + '\n')
    if epochs <= start:
        raise ValueError('requested horizon must exceed completed epochs')
    for epoch in range(start + 1, epochs + 1):
        begin = time.perf_counter()
        total, visits, data_seconds, update_seconds = 0., 0, 0., 0.
        batch_stream = []
        model.train()
        for batch_number in range(12):
            data_start = time.perf_counter()
            if position == len(permutation):
                permutation = torch.randperm(len(dataset), generator=generator).tolist()
                position = 0
            selected = permutation[position:position + 64]
            position += len(selected)
            for index in selected: counts[index] += 1
            query_seed = int(torch.randint(0, 2**31 - 1, (), generator=query_generator))
            protocol = replace(SensorProtocol(**manifest.protocol), seed=query_seed)
            batch = build_observation_batch([dataset[i] for i in selected], protocol,
                                            query_points=scientific['query_points'], manifest_indices=manifest.indices)
            if tuple(ids[i] for i in selected) != batch.sample_ids:
                raise ValueError('dataset membership order changed')
            atom = None
            if target_reader is not None:
                atom = torch.randint(0, 4, (len(selected),), generator=atom_generator)
                query = batch.metadata['query_indices']
                target = torch.as_tensor(target_reader(batch.sample_ids, atom, query))
                if target.shape != batch.target_fields.shape or not torch.isfinite(target).all():
                    raise ValueError('bank targets must be finite and aligned [B,Q,C]')
                batch.target_fields = dataset.normalizer.encode(target).to(dtype=batch.target_fields.dtype)
            batch_stream.append({'condition_indices': selected, 'query_seed': query_seed,
                                 'teacher_atom_indices': None if atom is None else atom.tolist()})
            batch = batch.to(device)
            data_seconds += time.perf_counter() - data_start
            update_start = time.perf_counter()
            completed_updates = (epoch - 1) * 12 + batch_number
            lr = 5e-6 + (5e-5 - 5e-6) * min(completed_updates / (50 * 12), 1.)
            for group in optimizer.param_groups: group['lr'] = lr
            losses, _, _, _ = backward_and_clip_model_loss(model, batch, model.parameters(), 1.0, device=device)
            optimizer.step(); after_optimizer_step(model)
            total += float(losses.total.detach().cpu()) * len(selected)
            update_seconds += time.perf_counter() - update_start
            visits += len(selected)
        if device.type == 'cuda': torch.cuda.synchronize(device)
        cumulative = sum(counts)
        row = {'epoch': epoch, 'native_rf_loss': total / visits, 'learning_rate': lr,
                     'visits': visits, 'cumulative_visits': cumulative,
                     'dataset_pass_equivalents': cumulative / len(dataset),
                     'epoch_seconds': time.perf_counter() - begin,
                     'data_seconds': data_seconds, 'update_seconds': update_seconds,
                     'batch_stream': batch_stream}
        rows.append({column['name']: row[column['name']] for column in definition['columns']})
        with history_path.open('a') as stream: stream.write(json.dumps(rows[-1]) + '\n')
        payload = {'scientific': scientific, 'recording': definition, 'epoch': epoch,
                   'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'condition_rng': generator.get_state(), 'query_rng': query_generator.get_state(),
                   'atom_rng': atom_generator.get_state(), 'torch_rng': torch.get_rng_state(),
                   'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                   'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                   'permutation': permutation, 'position': position, 'visit_counts': counts,
                   'history': rows}
        add_training_aux_state(payload, model)
        _atomic_save(payload, state_path)
        if epoch in (100, 500):
            _atomic_save(payload, root / f'checkpoint_epoch_{epoch}.pt')
        stop = bool(stop_after_epoch(rows[-1])) if stop_after_epoch else False
        if inference_config is not None and (epoch in (100, 500) or epoch == epochs or stop):
            export = {'model': checkpoint_model_state(model), 'global_step': epoch * 12,
                      'model_name': inference_config['model']['name'],
                      'model_config': dict(inference_config['model']),
                      'data_spec': asdict(dataset.data_spec),
                      'normalization': dataset.normalizer.state_dict(),
                      'source_hashes': {'checkpoint': identity['source_sha256']},
                      'config_sha256': config_digest(inference_config),
                      'inference_config_sha256': config_digest(inference_config),
                      'config_sha256_role': 'inherited inference configuration',
                      'native_scientific_definitions': scientific,
                      'reporting_epoch': epoch}
            add_training_aux_state(export, model)
            directory = root / 'checkpoints'
            directory.mkdir(exist_ok=True)
            _atomic_save(export, directory / f'epoch_{epoch}.pt')
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs or stop: _plot(rows, logs)
        if stop:
            break
    return state_path
