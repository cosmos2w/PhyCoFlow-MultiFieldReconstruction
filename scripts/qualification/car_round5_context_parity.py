#!/usr/bin/env python3
"""Synthetic ordinary CQ context algebra check; this is not model evaluation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from phycoflow_reconstruction.contracts import DataSpec, FieldSample
from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol, build_observation_batch
from phycoflow_reconstruction.models import build_model
from phycoflow_reconstruction.training.rollout import differentiable_rf_rollout


def fixture(device):
    y, x = torch.meshgrid(torch.linspace(0, 1, 8), torch.linspace(0, 1, 8), indexing='ij')
    coords = torch.stack((x, y, x * 0), -1).reshape(-1, 3)
    fields = torch.stack((x + y, (x + 2*y).square(), torch.sin(9*x)*y), -1).reshape(-1, 3)
    field = FieldSample(fields, coords, coords, torch.tensor(0.), 'synthetic', 0,
                        torch.empty(0), ('phi', 'vx', 'vy'), (8, 8))
    batch = build_observation_batch([field], SensorProtocol(name='structured_block_mean',
                                    field_counts={'phi':16, 'vx':4, 'vy':4}))
    batch.obs_coords[:, 1] = batch.obs_coords[:, 0]
    config = {
        'name':'gl_rbf_cq', 'backbone':'GL_rbf_ENH_CQ', 'prior':'iid',
        'hidden_dim':16, 'cond_dim':8, 'field_embed_dim':4, 'latent_dim':16,
        'num_latents':8, 'num_heads':4, 'num_latent_blocks':4, 'ff_mult':2,
        'attn_dropout':0., 'mlp_dropout':0., 'summary_type':'mean',
        'gather_mode':'topk_rbf_glres', 'gather_topk':3, 'gather_query_chunk_size':5,
        'learnable_rbf_sigma':True, 'neighbor_backend':'keops' if device.type == 'cuda' else 'torch',
        'USE_FOURIER_PE':True, 'fourier_pe_num_bands':2, 'fourier_pe_max_freq':4.,
        'sensor_coord_encoding':'fourier', 'latent_sensor_reinject':True,
        'latent_reinject_every':1, 'condition_attention_execution':'cached_kv',
        'sensor_attention_padding_mode':'full', 'glres_scale_init':1e-2,
        'cq_query_dim':8, 'cq_readout_mode':'lowrank', 'cq_readout_rank':4,
        'cq_readout_heads':2, 'cq_fusion_mode':'additive',
        'cq_time_conditioning':'sinusoidal_film', 'cq_time_embed_dim':8,
        'cq_time_max_period':10000., 'cq_time_film_zero_init':True,
        'cq_measurement_support_mode':'rbf_value_support', 'cq_measurement_support_normalize':True,
        'model_ema_enabled':False, 'obs_consistency_mode':'none',
        'obs_consistency_final_clamp':False, 'ode_solver':'euler',
    }
    spec = DataSpec(field.field_names, ('dimensionless',)*3, 3, (8, 8))
    return batch.to(device), config, spec


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_arm(base, batch, device, cache):
    model = deepcopy(base).to(device).eval()
    model.rollout_checkpointing = True
    model.rollout_context_cache = cache
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-6)
    sync(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        baseline_bytes = torch.cuda.memory_allocated(device)
    else:
        baseline_bytes = None
    start = time.perf_counter()
    generator = torch.Generator(device=device).manual_seed(234)
    prediction = differentiable_rf_rollout(model, batch, steps=4, solver='euler',
        generator=generator, observation_config={'mode':'endpoint_smooth', 'final_clamp':True})
    (prediction - batch.target_fields).square().mean().backward()
    gradients = {name: None if p.grad is None else p.grad.detach().cpu().clone()
                 for name, p in model.named_parameters()}
    optimizer.step()
    sync(device)
    seconds = time.perf_counter() - start
    peak_bytes = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
    params = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    return prediction.detach().cpu(), gradients, params, {
        'seconds':seconds, 'baseline_allocated_bytes':baseline_bytes, 'peak_allocated_bytes':peak_bytes}


def compare(left, right, *, rtol, atol):
    if set(left) != set(right):
        raise ValueError('parameter names differ')
    failures, largest, compared = [], 0., 0
    for name in left:
        a, b = left[name], right[name]
        if (a is None) != (b is None):
            failures.append(name + ': gradient presence differs')
            continue
        if a is None:
            continue
        compared += 1
        if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
            failures.append(name + ': nonfinite')
            continue
        largest = max(largest, float((a-b).abs().max()))
        if not torch.allclose(a, b, rtol=rtol, atol=atol):
            failures.append(name)
    return {'passed':not failures and compared > 0, 'tensors_compared':compared,
            'maximum_absolute_difference':largest, 'mismatches':failures, 'rtol':rtol, 'atol':atol}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', required=True, type=torch.device)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get('CAR_ROUND5_LEASE_TOKEN'):
        parser.error('run under the measured Round 5 preparation lease')
    if args.device.type not in ('cpu', 'cuda') or (args.device.type == 'cuda' and not torch.cuda.is_available()):
        parser.error('the requested CPU or CUDA device must be available')
    if args.output.exists():
        parser.error('output receipt already exists')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(216)
    batch, config, spec = fixture(args.device)
    base = build_model(config, spec).eval()
    record = {'schema':'car_round5_context_parity_v1', 'status':'running',
        'scope':'synthetic algebra only; no model accuracy, panel coverage or production speed claim',
        'device':str(args.device), 'device_name':torch.cuda.get_device_name(args.device) if args.device.type == 'cuda' else 'CPU',
        'torch_version':str(torch.__version__), 'neighbor_backend':config['neighbor_backend'],
        'steps':4, 'solver':'euler', 'checkpointing':True,
        'observation_mode':'endpoint_smooth', 'final_clamp':True, 'duplicated_sensor_coordinates':True,
        'optimizer':'AdamW', 'lr':1e-5, 'weight_decay':1e-6}
    try:
        a, ga, pa, a_cost = run_arm(base, batch, args.device, 'none')
        b, gb, pb, b_cost = run_arm(base, batch, args.device, 'static_features')
        endpoint = compare({'endpoint':a}, {'endpoint':b}, rtol=2e-5, atol=3e-6)
        gradients = compare(ga, gb, rtol=2e-4, atol=3e-6)
        updates = compare(pa, pb, rtol=2e-5, atol=2e-7)
        record.update(status='passed' if all(x['passed'] for x in (endpoint, gradients, updates)) else 'failed',
                      endpoint=endpoint, gradients=gradients, parameters_after_action=updates,
                      none=a_cost, static_features=b_cost)
    except Exception as exc:
        record.update(status='error', error_type=type(exc).__name__, error=str(exc))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    if record['status'] != 'passed':
        raise SystemExit(1)
    print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
