#!/usr/bin/env python3
"""Literal FFM SOURCE rollout-checkpointing numerical and B64 execution qualification."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / 'src'))
if os.environ.get('KEOPS_CACHE_FOLDER') and os.environ.get('CUDA_VISIBLE_DEVICES'):
    build = '_'.join(platform.uname()[:3]) + f'_p{sys.version.split()[0]}'
    (Path(os.environ['KEOPS_CACHE_FOLDER']) / f"{build}_CUDA_VISIBLE_DEVICES_{os.environ['CUDA_VISIBLE_DEVICES']}").mkdir(parents=True, exist_ok=True)


def main():
    import torch
    import yaml

    from phycoflow_reconstruction.data.physical_measure import attach_batch_measure
    from phycoflow_reconstruction.data.sensor_protocols import build_observation_batch
    from phycoflow_reconstruction.training.common import sensor_protocol_from_config
    from phycoflow_reconstruction.training.post_training import _scheduled_model_lr
    from phycoflow_reconstruction.training.rollout import differentiable_reconstruction
    from phycoflow_reconstruction.training.run_store import file_sha256
    from phycoflow_reconstruction.training.scalable_fitting import _digest
    from phycoflow_reconstruction.training.source import load_source_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    start = time.time()
    run = args.prepared_run.resolve(strict=True)
    config = yaml.safe_load((run / 'resolved_config.yaml').read_text())
    if config['model']['name'] != 'gl_rbf_cq' or config['rollout'] != {'steps': 4, 'solver': 'euler'}:
        raise ValueError('native four-step FFM prepared run required')
    device = torch.device(args.device)
    model, dataset, identity = load_source_model(config, device)
    try:
        panel_path = run / 'artifacts/source_panel.pt'
        panel = torch.load(panel_path, map_location='cpu', weights_only=True)
        if panel['parameter_buffer_sha256'] != _digest(model.state_dict()) or len(panel['sample_ids']) < 64:
            raise ValueError('literal source checkpoint and B64 TRAIN panel identity required')
        query = panel['query_indices'][:64]
        if query.shape != (64, 4096) or not torch.equal(query, query[0].expand_as(query)):
            raise ValueError('shared Q4096 TRAIN support required')
        samples = [dataset[i] for i in panel['sample_indices'][:64]]
        protocol = replace(sensor_protocol_from_config(config), seed=1129001)
        def make_batch(n):
            batch = build_observation_batch(samples[:n], protocol, query_indices=query[0],
                                            manifest_indices=panel['observation_manifest'])
            batch = attach_batch_measure(dataset, batch)
            if list(batch.sample_ids) != panel['sample_ids'][:n] or not torch.equal(batch.target_fields, panel['reference'][:n]):
                raise ValueError('saved TRAIN sample, observation and target binding differs')
            return replace(batch.to(device), target_fields=None)
        small, large = make_batch(4), make_batch(64)
        model.eval()
        original = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        params = tuple(model.parameters())
        settings = config['optimization']
        lr = _scheduled_model_lr(settings, 0)
        lr = float(settings['lr']) if lr is None else float(lr)
        def members(batch):
            return torch.stack([differentiable_reconstruction(model, batch, steps=4, solver='euler',
                    generator=torch.Generator(device=device).manual_seed(seed),
                    observation_config=dict(config['observation_consistency'])) for seed in (1129001, 1129002)])
        def one(checkpointed):
            model.load_state_dict(original)
            model.rollout_checkpointing = checkpointed
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=float(settings.get('weight_decay', 0.0)))
            before = [p.detach().cpu().clone() for p in params]
            values = members(small)
            grads = torch.autograd.grad(values.square().mean() + (values[0] * values[1]).mean(), params, allow_unused=True)
            for p, grad in zip(params, grads, strict=True):
                p.grad = grad
            optimizer.step()
            return (values.detach().cpu(), [None if g is None else g.detach().cpu() for g in grads],
                    [p.detach().cpu() - old for p, old in zip(params, before, strict=True)])
        direct = one(False)
        checked = one(True)
        def compare(a, b):
            maxima = []
            for left, right in zip(a, b, strict=True):
                if left is None or right is None:
                    if left is not right:
                        raise ValueError('gradient usage differs')
                    continue
                if not torch.isfinite(left).all() or not torch.isfinite(right).all():
                    raise ValueError('nonfinite value or gradient')
                torch.testing.assert_close(left, right, rtol=3e-5, atol=3e-6)
                maxima.append(float((left - right).abs().max()))
            return max(maxima, default=0.)
        small_result = {'output_max_abs': compare(direct[0], checked[0]),
                        'gradient_max_abs': compare(direct[1], checked[1]),
                        'adamw_displacement_max_abs': compare(direct[2], checked[2])}
        del direct, checked
        model.load_state_dict(original)
        model.zero_grad(set_to_none=True)
        model.rollout_checkpointing = True
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        with torch.no_grad():
            reference = members(large).cpu()
        actual = members(large)
        large_error = compare(reference, actual.detach().cpu())
        gradients = torch.autograd.grad(actual.square().mean() + (actual[0] * actual[1]).mean(), params, allow_unused=True)
        count = sum(g is not None for g in gradients)
        if count == 0 or any(not torch.isfinite(g).all() for g in gradients if g is not None):
            raise ValueError('B64 checkpointed gradients must be finite and nonempty')
        result = {'schema': 'car_round2_ffm_checkpointing_qualification_v1', 'passed': True,
                  'scope': 'numerical execution only; no efficacy or reserved-task scores', 'identity': identity,
                  'source_panel_sha256': file_sha256(panel_path), 'sample_ids': panel['sample_ids'][:64],
                  'support': {'small_batch': 4, 'large_batch': 64, 'queries': 4096, 'members': 2, 'steps': 4},
                  'small_direct_checkpoint': small_result, 'large_forward_max_abs': large_error,
                  'large_finite_gradient_tensors': count, 'lr': lr, 'device': str(device),
                  'start_unix': start, 'end_unix': time.time()}
        result['wall_seconds'] = result['end_unix'] - start
        if not math.isfinite(result['wall_seconds']):
            raise ValueError('invalid timer')
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    finally:
        dataset.close()


if __name__ == '__main__':
    main()
