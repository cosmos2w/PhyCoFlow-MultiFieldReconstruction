#!/usr/bin/env python3
"""Audit the live outer-pair sampler using actual SOURCE-plus-head gradients.

TEST is never opened. No short panel, endpoint-gradient proxy or fabricated
fixed cost can qualify an allocation. Each outer ID's duplicate multiplicity
shares one inner draw, matching SharedQuerySampler. Expanded leaves and all
complete anchors are enumerated and charged as fixed work.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('CAR_REPO_ROOT', CODE_ROOT)).resolve()
sys.path.insert(0, str(CODE_ROOT / 'src'))
if os.environ.get('KEOPS_CACHE_FOLDER') and os.environ.get('CUDA_VISIBLE_DEVICES'):
    build = '_'.join(platform.uname()[:3]) + f'_p{sys.version.split()[0]}'
    (Path(os.environ['KEOPS_CACHE_FOLDER']) / f"{build}_CUDA_VISIBLE_DEVICES_{os.environ['CUDA_VISIBLE_DEVICES']}").mkdir(parents=True, exist_ok=True)


def shared_inner_mse(gram, outer_count, inner_weights, probabilities, draws):
    """Exact MSE for multinomial outer counts and one shared inner draw per ID.

    Gram entries are full parameter gradients concatenated across Scale and
    Shape. Each gradient is the conditional mean objective for one outer/inner
    cell, before the uniform outer and declared inner target weights.
    """
    import torch
    gram = torch.as_tensor(gram, dtype=torch.float64)
    p = torch.as_tensor(probabilities, dtype=torch.float64)
    inner = torch.as_tensor(inner_weights, dtype=torch.float64)
    k, n = int(outer_count), len(inner)
    if (k < 1 or draws < 1 or gram.shape != (k*n, k*n) or p.shape != (k,)
        or not torch.isfinite(gram).all() or not torch.isfinite(p).all()
        or not torch.isfinite(inner).all() or (p <= 0).any() or (inner <= 0).any()
        or not torch.isclose(p.sum(), p.new_tensor(1.))
        or not torch.isclose(inner.sum(), inner.new_tensor(1.))
        or not torch.allclose(gram, gram.T, atol=1e-10, rtol=1e-10)):
        raise ValueError('invalid Gram or positive normalized sampling probabilities')
    view = gram.reshape(k, n, k, n)
    mean_gram = torch.einsum('a,iajb,b->ij', inner, view, inner)
    second = (gram.diag().reshape(k, n) * inner).sum(1)
    result = (((1 + (1-p)/(draws*p)) * second).sum()
              - (draws-1)/draws * mean_gram.diag().sum()
              - mean_gram.sum()/draws) / (k*k)
    tol = 1e-10 * max(1., float(second.abs().max()))
    if float(result) < -tol:
        raise ValueError('invalid gradient Gram yields negative MSE')
    return max(0., float(result))


def expected_unique_cost(probabilities, costs, draws):
    import torch
    p = torch.as_tensor(probabilities, dtype=torch.float64)
    c = torch.as_tensor(costs, dtype=torch.float64)
    if (draws < 1 or p.ndim != 1 or c.shape != p.shape or (p <= 0).any()
        or (c <= 0).any() or not torch.isfinite(c).all()
        or not torch.isclose(p.sum(), p.new_tensor(1.))):
        raise ValueError('positive costs and normalized proposal required')
    return float(((1 - (1-p).pow(draws)) * c).sum())


def family_gradient_gram(losses, parameters):
    """Concatenate family-specific gradients to prevent cross-family cancellation."""
    import torch
    params = tuple(p for p in parameters if p.requires_grad)
    if not params:
        raise ValueError('no actual trainable residual parameters')
    vectors = []
    for family_losses in losses:
        if len(family_losses) != 2:
            raise ValueError('every cell requires separate Scale and Shape graphs')
        parts = []
        for loss in family_losses:
            if loss.ndim or not torch.isfinite(loss):
                raise ValueError('finite scalar family cell required')
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            parts.extend((torch.zeros_like(p) if g is None else g).detach().flatten().double().cpu()
                         for p, g in zip(params, grads))
        vectors.append(torch.cat(parts))
    matrix = torch.stack(vectors)
    return matrix @ matrix.T, sum(p.numel() for p in params)


def c8_frozen_probabilities(scale_leaves, shape_leaves, inner_weights, batch_weights, *, epsilon=.5):
    """C8's detached nonnegative family mean, conditioned on each outer ID.

    Inputs are [batch, outer, inner] conditional leaf means, never importance
    corrected. Inner and batch weights average actual observations once.
    """
    import torch
    scale = torch.as_tensor(scale_leaves, dtype=torch.float64)
    shape = torch.as_tensor(shape_leaves, dtype=torch.float64)
    inner = torch.as_tensor(inner_weights, dtype=torch.float64)
    batches = torch.as_tensor(batch_weights, dtype=torch.float64)
    if (scale.ndim != 3 or shape.shape != scale.shape or inner.shape != (scale.shape[2],)
            or batches.shape != (scale.shape[0],) or not 0 < epsilon <= 1
            or any(not torch.isfinite(v).all() for v in (scale, shape, inner, batches))
            or (inner <= 0).any() or (batches <= 0).any()
            or not torch.isclose(inner.sum(), inner.new_tensor(1.))
            or not torch.isclose(batches.sum(), batches.new_tensor(1.))):
        raise ValueError('finite conditional leaf values and normalized inner/batch weights required')
    difficulty = ((scale.clamp_min(0) + shape.clamp_min(0)) / 2 * inner[None, None, :]
                  * batches[:, None, None]).sum((0, 2))
    count = scale.shape[1]
    p = torch.full((count,), 1/count, dtype=torch.float64)
    if difficulty.sum() > 0:
        p = epsilon/count + (1-epsilon)*difficulty/difficulty.sum()
    return p, difficulty


def measure_sampler_overhead(fields, probabilities, *, iterations=2000, repeats=5):
    """Warm paired sampling measurements, retaining negative deltas and spread."""
    import statistics

    from phycoflow_reconstruction.coherence.query_sampling import HierarchicalQuerySampler
    if iterations < 100 or repeats < 3:
        raise ValueError('enough measured sampling repetitions required')
    def measure(fixed):
        sampler = HierarchicalQuerySampler(fields, seed=997, hierarchy_weight=0.,
            fixed_probabilities=probabilities if fixed else None,
            profile_sha256='0'*64 if fixed else None)
        start = time.perf_counter_ns()
        for _ in range(iterations):
            sampler.draw(); sampler.finish()
        return (time.perf_counter_ns()-start) / (iterations*1e9)
    measure(False); measure(True)
    rows = []
    for iteration in range(repeats):
        order = (False,True) if iteration % 2 == 0 else (True,False)
        result = {kind:measure(kind) for kind in order}
        rows.append({'uniform_seconds':result[False], 'fixed_seconds':result[True],
                     'difference_seconds':result[True]-result[False]})
    delta = statistics.mean(row['difference_seconds'] for row in rows)
    standard_error = statistics.stdev(row['difference_seconds'] for row in rows)/math.sqrt(repeats)
    return {'iterations_per_measurement':iterations, 'warmup_measurements':2, 'rows':rows,
            'mean_difference_seconds':delta, 'standard_error_seconds':standard_error,
            'charged_overhead_seconds':max(0.,delta+2*standard_error),
            'scope':'paired CPU draw and finish for actual fixed versus uniform hierarchical samplers'}


def measured_prior_research_seconds(research_costs, campaign):
    from phycoflow_reconstruction.evaluation.adaptive_campaign import digest
    if research_costs.get('schema') != 'car_round2_allocator_research_costs_v1' or not isinstance(research_costs.get('jobs'),list):
        raise ValueError('declared allocator research cost allocations required')
    if not research_costs['jobs'] and not research_costs.get('no_prior_allocator_work_reason'):
        raise ValueError('empty research cost allocation requires an explicit no-prior-work reason')
    preparation_seconds = 0.
    for item in research_costs['jobs']:
        path = Path(item['path']).resolve()
        if Path(campaign).resolve() not in path.parents or digest(path) != item['sha256']:
            raise ValueError('allocator research receipt changed or is outside campaign')
        job = json.loads(path.read_text())
        measured = job.get('wall_seconds',job.get('seconds'))
        allocated = item['allocated_seconds']
        if (not isinstance(measured,(int,float)) or not math.isfinite(measured) or measured < 0
                or not isinstance(allocated,(int,float)) or not math.isfinite(allocated) or not 0 <= allocated <= measured):
            raise ValueError('research allocation must be bounded by its measured job wall')
        preparation_seconds += allocated
    return preparation_seconds


def provisional_overhead_fraction(research_seconds, policy_seconds_per_update, minimum_observed_update_seconds, *, updates=12000):
    values = (research_seconds, policy_seconds_per_update, minimum_observed_update_seconds)
    if (any(not math.isfinite(float(v)) for v in values) or research_seconds < 0
            or policy_seconds_per_update < 0 or minimum_observed_update_seconds <= 0 or updates < 1):
        raise ValueError('finite measured research, policy and lower-observed update times required')
    return (research_seconds + updates*policy_seconds_per_update)/(updates*minimum_observed_update_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--prepared-run', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True, help='frozen residual arm config')
    parser.add_argument('--c8-output', type=Path, required=True, help='new TRAIN-panel frozen C8 priority artifact')
    parser.add_argument('--complete-update-receipt', type=Path, required=True, help='measured B64 residual update timing receipt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    parser.add_argument('--save-margin-seconds', type=float, required=True)
    parser.add_argument('--research-costs', type=Path, required=True, help='measured allocator-specific research cost allocations')
    parser.add_argument('--replicates', type=int, default=128)
    args = parser.parse_args()
    import yaml

    from phycoflow_reconstruction.evaluation.adaptive_campaign import (
        atomic_json,
        locked_ledger,
        spent,
    )
    # CUDA_VISIBLE_DEVICES must be set before this process starts, before torch imports.
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_uuid or not os.environ.get('KEOPS_CACHE_FOLDER'):
        raise ValueError('bind the physical GPU and its KeOps cache before invocation')
    if any(not math.isfinite(v) or v <= 0 for v in (args.max_seconds, args.save_margin_seconds)) or args.max_seconds <= args.save_margin_seconds or args.replicates < 2:
        raise ValueError('invalid time bounds or replicate count')
    campaign, run = args.campaign.resolve(strict=True), args.prepared_run.resolve(strict=True)
    if campaign not in run.parents or campaign not in args.output.resolve().parents or '/data2/wanglz/PhyCoFlow-MultiFieldReconstruction/' not in str(campaign) + '/':
        raise ValueError('run and output must belong to DATA2 campaign')
    if args.output.exists():
        raise FileExistsError(args.output)
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location('car_round2_driver', CODE_ROOT / 'scripts/posttraining/run_coherence_adaptive_repair_round2.py')
    driver = module_from_spec(spec); spec.loader.exec_module(driver)
    capsule = driver.check_code_capsule(campaign)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_uuid:
        raise ValueError('invalid visible device')
    config = yaml.safe_load(args.config.read_text())
    source = yaml.safe_load((run / 'resolved_config.yaml').read_text())
    if not config.get('residual', {}).get('enabled') or Path(config['source_run']).resolve() != Path(source['source_run']).resolve():
        raise ValueError('residual config must bind the prepared SOURCE')
    complete_update = json.loads(args.complete_update_receipt.read_text())
    research_costs = json.loads(args.research_costs.read_text())
    preparation_seconds = measured_prior_research_seconds(research_costs, campaign)
    if campaign not in args.c8_output.resolve().parents or args.c8_output.exists():
        raise ValueError('new C8 priority artifact must reside within campaign')
    listed = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
    devices = [tuple(part.strip() for part in row.split(',')) for row in listed.splitlines()]
    selected = [int(index) for index, uuid_value in devices if uuid_value == args.gpu_uuid]
    runtime_settings = json.loads((campaign / 'operations/runtime.json').read_text())
    if len(selected) != 1 or (runtime_settings.get('user_authorized_gpu_indices') is not None
                              and selected[0] not in runtime_settings['user_authorized_gpu_indices']):
        raise ValueError('GPU missing or excluded by runtime configuration')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    if any(row.split(',')[0].strip() == args.gpu_uuid for row in apps.splitlines() if ',' in row):
        raise ValueError('GPU has an existing compute process')
    ledger = campaign / 'operations/gpu_ledger.json'
    token, start = str(uuid.uuid4()), time.time()
    with locked_ledger(ledger) as state:
        if any(x['gpu_uuid'] == args.gpu_uuid for x in state['active'].values()):
            raise ValueError('device already leased')
        if spent(state) + args.max_seconds >= (130-6)*3600:
            raise ValueError('audit exceeds the remaining GPU budget and reserve')
        state['active'][token] = {'gpu_uuid': args.gpu_uuid, 'phase': 'residual_allocator_audit', 'start_unix': start}
    receipt = {'schema': 'car_round2_allocator_job_v1', 'status': 'running', 'token': token,
               'start_unix': start, 'pid': os.getpid(), 'gpu_uuid': args.gpu_uuid,
               'code_capsule_sha256': capsule}
    previous_alarm = signal.getsignal(signal.SIGALRM)
    def timeout(_signal, _frame):
        raise TimeoutError('allocator audit reached its save margin')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(max(1, math.ceil(args.max_seconds - args.save_margin_seconds)))
    dataset = None
    try:
        import torch

        from phycoflow_reconstruction.coherence import build_enabled_families
        from phycoflow_reconstruction.coherence.adaptive_profile import isolate_query_stratum
        from phycoflow_reconstruction.coherence.query_sampling import (
            LINE_WEIGHTS,
            PAIR_SIGNS,
            Draw,
            QueryBatch,
            full_target_query_batch,
            pair_count,
        )
        from phycoflow_reconstruction.coherence.round2_qualification import second_moment_proposal
        from phycoflow_reconstruction.data.physical_measure import (
            attach_batch_measure,
            batch_measure_context,
        )
        from phycoflow_reconstruction.data.sensor_protocols import build_observation_batch
        from phycoflow_reconstruction.models.source_residual import FrozenSourceResidual
        from phycoflow_reconstruction.training.common import sensor_protocol_from_config
        from phycoflow_reconstruction.training.post_training import _physical_endpoint_risk
        from phycoflow_reconstruction.training.rollout import differentiable_reconstruction
        from phycoflow_reconstruction.training.run_store import file_sha256
        from phycoflow_reconstruction.training.scalable_fitting import _digest, validate_scalable_artifacts
        from phycoflow_reconstruction.training.scalable_runtime import ScalableRuntime
        from phycoflow_reconstruction.training.source import load_source_model
        from phycoflow_reconstruction.utils.reproducibility import seed_everything
        seed_everything(int(config['runtime'].get('seed', 42)), bool(config['runtime'].get('deterministic', True)))
        model, dataset, source_identity = load_source_model(source, torch.device('cuda:0'))
        execution = config['optimization'].get('rollout_execution', {})
        if 'checkpointing' in execution:
            model.rollout_checkpointing = execution['checkpointing']
        if dataset.split_name != 'train':
            raise ValueError('allocator audit must use TRAIN only')
        family_name = 'ffm' if model.capabilities.generative else 'senseiver'
        if config['model']['name'] != ('gl_rbf_cq' if family_name == 'ffm' else 'senseiver'):
            raise ValueError('source family and arm config differ')
        panel_path, artifacts_path = run / 'artifacts/source_panel.pt', run / 'artifacts/scalable_coherence.pt'
        panel = torch.load(panel_path, map_location='cpu', weights_only=True)
        artifacts = torch.load(artifacts_path, map_location='cpu', weights_only=True)
        if panel.get('schema') != 'car_source_panel_v1' or panel['provenance'] != artifacts['provenance']:
            raise ValueError('SOURCE panel and common observer provenance differ')
        validate_scalable_artifacts(artifacts, panel['provenance'])
        if panel['parameter_buffer_sha256'] != _digest(model.state_dict()) or panel['source_weight_view'] != source_identity['weight_view']:
            raise ValueError('loaded SOURCE weights/buffers differ from saved panel')
        ids = panel['sample_ids']
        if len(ids) != max(1, math.ceil(.1*len(dataset))) or len(set(ids)) != len(ids) or ids != artifacts['provenance']['sample_ids']:
            raise ValueError('complete matched 10% TRAIN panel required')
        source_sha = artifacts['provenance']['source_checkpoint_sha256']
        update_seconds = complete_update.get('complete_update_seconds')
        lower_update_seconds = complete_update.get('minimum_observed_steady_update_seconds')
        if (complete_update.get('schema') != 'car_round2_complete_update_timing_v1'
                or complete_update.get('source_checkpoint_sha256') != source_sha
                or complete_update.get('config_sha256') != file_sha256(args.config)
                or complete_update.get('gpu_uuid') != args.gpu_uuid
                or complete_update.get('parameterization') != 'residual_head'
                or complete_update.get('batch_size') != 64
                or complete_update.get('measured_updates', 0) < 1
                or not isinstance(update_seconds, (float, int)) or not math.isfinite(update_seconds) or update_seconds <= 0
                or not isinstance(lower_update_seconds, (float, int)) or not math.isfinite(lower_update_seconds)
                or not 0 < lower_update_seconds <= update_seconds):
            raise ValueError('actual measured B64 residual complete-update receipt required')
        for name in ('qualification_receipt', 'pause_launch', 'resume_launch', 'epoch_timing'):
            evidence = Path(complete_update[f'{name}_path']).resolve()
            if campaign not in evidence.parents or file_sha256(evidence) != complete_update[f'{name}_sha256']:
                raise ValueError(f'complete-update timing evidence changed or left campaign: {name}')
        qualification = json.loads(Path(complete_update['qualification_receipt_path']).read_text())
        if (qualification.get('passed') is not True or qualification.get('formal_config_sha256') != file_sha256(args.config)
                or qualification.get('pause_step') != 108 or qualification.get('final_step') != 144):
            raise ValueError('update timing must bind the same qualified final target')
        # Match trainer RNG order: source construction is followed immediately
        # by the randomly initialized, zero-output residual wrapper.
        res = config['residual']
        model = FrozenSourceResidual(model, coordinate_dim=res['coordinate_dim'], field_scales=res['field_scales'], width=res['width'], max_fraction=res['max_fraction']).to('cuda:0').eval()
        parameters = tuple(p for p in model.parameters() if p.requires_grad)
        if not parameters or any(p.requires_grad for p in model.source.parameters()):
            raise ValueError('audit must differentiate exactly the trainable residual head')
        manifest = json.loads((run / 'run_manifest.json').read_text())
        families = build_enabled_families(source['coherence'], dataset.data_spec, dataset.normalizer)
        for name, family in families.items():
            path = run / 'artifacts' / f'{name}_family.pt'
            if file_sha256(path) != manifest['coherence_family_state_sha256s'][name]:
                raise ValueError('prepared family hash changed')
            family.to('cuda:0').load_state_artifact(torch.load(path, map_location='cuda:0', weights_only=True))
        scalable = config['coherence']['scalable_anchored_coherence']
        prefit_path = Path(scalable['common_prefit']['path'])
        if file_sha256(prefit_path) != scalable['common_prefit']['sha256']:
            raise ValueError('frozen common coefficient hash differs')
        prefit = json.loads(prefit_path.read_text())
        for key in ('source_checkpoint_sha256', 'normalizer_sha256', 'support_sha256', 'field_order', 'sample_ids'):
            if prefit.get('binding', {}).get(key) != artifacts['provenance'][key]:
                raise ValueError(f'common prefit binding differs: {key}')
        # Bind exactly the selected expanded leaves to the target and fixed cost.
        bank = scalable['adaptive']['bank_artifact']
        if file_sha256(bank['path']) != bank['sha256']:
            raise ValueError('selected bank hash changed')
        bank_data = json.loads(Path(bank['path']).read_text())
        if bank_data.get('policy_qualified'):
            raise ValueError('audit target must precede the policy qualification')
        if bank_data.get('scoring_mode', 'conditional') != ('conditional' if family_name == 'ffm' else 'deterministic'):
            raise ValueError('selected bank scoring mode differs from SOURCE model')
        from phycoflow_reconstruction.coherence.adaptive_fitting import (
            bank_target_digest,
            validate_bank,
        )
        from phycoflow_reconstruction.training.scalable_fitting import _adaptive_input_binding
        validate_bank(bank_data, field_order=artifacts['provenance']['field_order'],
                      support_identity=artifacts['provenance']['support_sha256'],
                      source_identity=artifacts['provenance']['source_checkpoint_sha256'])
        if prefit.get('binding', {}).get('bank_target_sha256') != bank_target_digest(bank_data):
            raise ValueError('common prefit expanded bank target differs')
        fit_masses = panel['masses'].to(panel['reference'].dtype)
        binding = _adaptive_input_binding(source=panel['members'], reference=panel['reference'],
                                         coordinates=panel['coordinates'], source_weights=fit_masses,
                                         reference_weights=fit_masses, identity=artifacts['provenance'], core=artifacts)
        if bank_data.get('source_fit_binding_sha256') != binding or bank_data.get('population_ids') != ids:
            raise ValueError('selected bank differs from SOURCE panel/core')
        artifacts = copy.deepcopy(artifacts)
        artifacts['adaptive_bank'] = bank_data
        artifacts['sha256'] = _digest({key: value for key, value in artifacts.items() if key != 'sha256'})
        runtime = ScalableRuntime(artifacts, families, steps_per_epoch=12, expected_provenance=artifacts['provenance'], scoring_mode='conditional' if family_name == 'ffm' else 'deterministic')
        runtime.calibration.load_prefit(prefit)
        fields, k = tuple(dataset.field_names), pair_count(len(dataset.field_names))
        inner = [(s, line, w/4) for s in PAIR_SIGNS for line, w in enumerate(LINE_WEIGHTS)]
        weights = torch.tensor([x[2] for x in inner], dtype=torch.float64)
        draws = runtime.sampler.mutual_draws
        if draws < 1:
            raise ValueError('positive live pair draw count required')
        # Group the saved eight-condition rollout chunks into the actual B64
        # update while preserving each chunk's original member streams.
        protocol = sensor_protocol_from_config(source)
        chunks, cursor = [], 0
        for chunk in panel['coupling']:
            size = len(chunk['sample_ids']); end = cursor + size
            samples = [dataset[i] for i in panel['sample_indices'][cursor:end]]
            qids = panel['query_indices'][cursor:end]
            if not torch.equal(qids, qids[0].expand_as(qids)):
                raise ValueError('saved query support differs within a chunk')
            batch = build_observation_batch(samples, protocol, query_indices=qids[0], manifest_indices=panel['observation_manifest'])
            if list(batch.sample_ids) != chunk['sample_ids'] or _digest([batch.obs_indices, batch.obs_values, batch.obs_field_ids, batch.obs_valid_mask]) != chunk['observation_sha256']:
                raise ValueError('saved observation coupling differs')
            batch = attach_batch_measure(dataset, batch)
            for key, actual in (('reference', batch.target_fields), ('coordinates', batch.query_coords), ('masses', batch_measure_context(batch)['generated_weights'])):
                if not torch.equal(panel[key][cursor:end], actual):
                    raise ValueError(f'saved panel {key} differs')
            chunks.append((cursor, end, batch.to('cuda:0'), chunk['member_seeds']))
            cursor = end
        if cursor != len(ids):
            raise ValueError('incomplete source coupling')
        if draws != runtime.sampler.mutual_draws:
            raise ValueError('proposal draw count differs from the actual runtime')
        family_scales = json.loads((run / 'artifacts/coherence_calibration.json').read_text())['resolved_scales']
        family_weights = {name: float(family_scales[name]) * float(getattr(family, 'family_weight', 1.0)) for name, family in families.items()}

        def query(index, inner_index=0, extra=None, unary_field=0, unary_sign=1):
            sign, line, probability = inner[inner_index]
            pair_ids = (index,) if extra is None else (index, extra)
            pair = tuple(Draw(i, 1, 1/k, k, len(pair_ids), sign, line, probability) for i in pair_ids)
            unary = (Draw(unary_field, 1, 1/len(fields), len(fields), 1, (unary_sign,), None, .5),)
            return QueryBatch(unary, pair, 1)

        def evaluate(group, *, verify=False):
            # Preserve SOURCE chunk seeds and observations while forming one B64 loss.
            source_outputs = [[] for _ in range(2 if family_name == 'ffm' else 1)]
            for begin, end, batch, seeds in group:
                input_batch = replace(batch, target_fields=None)
                for member in range(len(source_outputs)):
                    prediction = differentiable_reconstruction(model, input_batch,
                        steps=int(source['rollout']['steps']), solver=source['rollout']['solver'],
                        generator=torch.Generator(device='cuda:0').manual_seed(seeds[member]),
                        observation_config=source['observation_consistency'])
                    if verify:
                        torch.testing.assert_close(prediction.detach().cpu(), panel['members'][begin:end, member], rtol=3e-4, atol=3e-4)
                    source_outputs[member].append(prediction)
            members = [torch.cat(x) for x in source_outputs]
            target = torch.cat([item[2].target_fields for item in group])
            coord = torch.cat([item[2].query_coords for item in group])
            masks = torch.cat([item[2].query_valid_mask for item in group])
            mass = torch.cat([batch_measure_context(item[2])['generated_weights'] for item in group])
            ctx = {'conditional_draws': tuple(members[1:]), 'generated_weights': mass, 'reference_weights': mass}
            return members, target, coord, masks, mass, ctx

        def losses(values, q):
            members, target, coord, _mask, _mass, context = values
            return [runtime._score(name, families[name], members[0], target, coord, context, q)[0].scalar_loss * family_weights[name]
                    for name in ('multiscale_spectrum', 'topology')]

        groups = []
        for pos in range(0, len(chunks), 8):
            group = chunks[pos:pos+8]
            count = sum(end-begin for begin, end, _, _ in group)
            if count != 64 and pos + 8 < len(chunks):
                raise ValueError('SOURCE chunks do not form B64 training batches')
            groups.append(group)
        if not groups or sum(sum(end-begin for begin,end,_,_ in g) for g in groups) != len(ids):
            raise ValueError('streamed B64 groups do not cover the panel')
        for group in groups:
            with torch.no_grad():
                evaluate(group, verify=True)

        # Balanced paired timings include SOURCE, all families, complete anchors,
        # fidelity, extra leaves and backward work. Duplicate pairs cost once.
        from importlib.util import module_from_spec, spec_from_file_location
        timing_spec = spec_from_file_location('old_car_timing', CODE_ROOT / 'scripts/benchmarks/profile_car_allocator.py')
        timing_module = module_from_spec(timing_spec); timing_spec.loader.exec_module(timing_module)
        timing_inner = 0
        def timed_bundle(index, extra):
            torch.cuda.synchronize(); began = time.perf_counter()
            values = evaluate(groups[0])
            members, target, coord, mask, mass, context = values
            q = query(index, timing_inner, extra=extra)
            objectives = [runtime._score(name, family, members[0], target, coord, context, q)[0].scalar_loss * family_weights[name]
                          for name, family in families.items()]
            endpoint = {'prediction': members[0], 'members': tuple(members), 'target': target,
                        'mask': mask, 'measure': {'generated_weights': mass}}
            objectives.append(_physical_endpoint_risk(endpoint, dataset, mean_risk=family_name == 'ffm').mean())
            for objective in objectives:
                torch.autograd.grad(objective, parameters, retain_graph=True, allow_unused=True)
            torch.cuda.synchronize()
            return time.perf_counter() - began
        timing_evidence = []
        receipt['timing'] = timing_evidence
        timings = []
        for timing_inner in range(len(inner)):
            evidence = {'inner_index': timing_inner}
            timing_evidence.append(evidence)
            timings.append(timing_module.measure_bundle_costs(timed_bundle, k, 3, evidence=evidence))
        costs = sum(weights[j] * torch.tensor([row['incremental_seconds'] for row in timing], dtype=torch.float64)
                    for j, timing in enumerate(timings))
        fixed = sum(float(weights[j]) * float(timing[0]['fixed_seconds']) for j, timing in enumerate(timings))
        uniform_p = torch.full((k,), 1/k, dtype=torch.float64)
        # Include ConFIG, clipping, AdamW, sampler and input preparation as
        # measured in a complete actual update, without double-counting pairs.
        measured_residual = float(update_seconds) - expected_unique_cost(uniform_p, costs, draws)
        fixed = max(fixed, measured_residual)
        if fixed <= 0:
            raise ValueError('complete fixed cost is unresolved')

        def concatenate_grads(two_losses):
            pieces = []
            for loss in two_losses:
                gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
                pieces.extend((torch.zeros_like(p) if g is None else g).detach().flatten().double().cpu()
                              for p, g in zip(parameters, gradients))
            return torch.cat(pieces)

        rows, moments, difficulty_rows = [], torch.zeros(k, dtype=torch.float64), []
        for group in groups:
            values = evaluate(group)
            pair_vectors, unary_vectors = [], []
            scale_difficulties = torch.zeros(k, len(inner), dtype=torch.float64)
            shape_difficulties = torch.zeros_like(scale_difficulties)
            for pair in range(k):
                for index in range(len(inner)):
                    q = query(pair, index)
                    family_cells = [isolate_query_stratum(lambda draw, j=j, values=values: losses(values, draw)[j], q, 'pair') for j in range(2)]
                    pair_vectors.append(concatenate_grads(family_cells))
                    members, target, coord, _mask, _mass, context = values
                    for name, destination in (('multiscale_spectrum',scale_difficulties), ('topology',shape_difficulties)):
                        _, _, leaves = runtime._score(name, families[name], members[0], target, coord, context, q)
                        # Returned leaf is already conditional on this outer/inner
                        # draw; never divide it by its selection probability.
                        destination[pair,index] = float(leaves['pair'][pair].detach().mean())
            for field in range(len(fields)):
                for sign in (1, -1):
                    q = query(0, unary_field=field, unary_sign=sign)
                    family_cells = [isolate_query_stratum(lambda draw, j=j, values=values: losses(values, draw)[j], q, 'self') for j in range(2)]
                    unary_vectors.append(concatenate_grads(family_cells))
            pair_matrix, unary_matrix = torch.stack(pair_vectors), torch.stack(unary_vectors)
            exhaustive = full_target_query_batch(len(fields))
            exhaustive_values = (*values[:5], {**values[5], 'exhaustive_shape_queries': True})
            for stratum, matrix, target_weights in (
                ('pair', pair_matrix, weights.repeat(k)/k),
                ('self', unary_matrix, torch.full((2*len(fields),), 1/(2*len(fields)), dtype=torch.float64)),
            ):
                exact = concatenate_grads([isolate_query_stratum(lambda draw, j=j, values=exhaustive_values: losses(values, draw)[j], exhaustive, stratum) for j in range(2)])
                torch.testing.assert_close(target_weights @ matrix, exact, rtol=1e-5, atol=1e-7,
                                           msg=f'{stratum} estimator leaves differ from the full target')
            gram = pair_matrix @ pair_matrix.T
            unary_gram = unary_matrix @ unary_matrix.T
            count = sum(end-begin for begin,end,_,_ in group)
            fraction = count / len(ids)
            difficulty_rows.append((scale_difficulties, shape_difficulties, fraction))
            moments += fraction * (gram.diag().reshape(k, len(inner)) * weights).sum(1)
            rows.append({'ids': [sample for begin,end,_,_ in group for sample in ids[begin:end]], 'fraction': fraction,
                         'pair_gram': gram, 'unary_gram': unary_gram})
            del values, pair_vectors, unary_vectors, pair_matrix, unary_matrix
        if [x for row in rows for x in row['ids']] != ids:
            raise ValueError('parameter audit did not cover the exact TRAIN manifest')
        c8, c8_difficulty = c8_frozen_probabilities(
            torch.stack([row[0] for row in difficulty_rows]), torch.stack([row[1] for row in difficulty_rows]),
            weights, torch.tensor([row[2] for row in difficulty_rows],dtype=torch.float64))
        from phycoflow_reconstruction.evaluation.adaptive_campaign import immutable_json
        c8_artifact = {'schema':'car_round2_c8_priority_v1', 'source_checkpoint_sha256':source_sha,
                       'field_order':list(fields), 'population_ids':ids,
                       'method':'lagged_c8_scale_shape_difficulty', 'pair_probabilities':c8.tolist(),
                       'pair_draws':draws, 'epsilon':.5, 'conditional_difficulty':c8_difficulty.tolist(),
                       'scope':'frozen detached TRAIN-panel conditional leaves; inner and batch weights applied once',
                       'source_panel_sha256':file_sha256(panel_path), 'selected_bank_sha256':bank['sha256'],
                       'frozen_unix':time.time(), 'frozen_before_audit_draws':True,
                       'qualified':False, 'code_capsule_sha256':capsule}
        immutable_json(args.c8_output,c8_artifact)
        proposal = second_moment_proposal(torch.full((k,), 1/k), moments, costs)
        sampler_timing = measure_sampler_overhead(fields, proposal.tolist())
        c8_sampler_timing = measure_sampler_overhead(fields, c8.tolist())
        policy_overhead = sampler_timing['charged_overhead_seconds']
        policies = {'uniform': torch.full((k,), 1/k, dtype=torch.float64), 'c8_priorities': c8,
                    'lagged_gradient_cost': proposal}
        unary_probs = torch.full((len(fields),), 1/len(fields), dtype=torch.float64)
        unary_inner = torch.tensor([.5, .5], dtype=torch.float64)
        results = {}
        elapsed = time.time() - start
        projected_training_seconds = 12000 * float(update_seconds)
        lower_observed_training_seconds = 12000 * float(lower_update_seconds)
        overhead = provisional_overhead_fraction(preparation_seconds + elapsed + args.save_margin_seconds,
                                                 policy_overhead, lower_update_seconds)
        amortized = (preparation_seconds + elapsed + args.save_margin_seconds) / 12000
        generator = torch.Generator().manual_seed(2047)
        def sample_error(gram, outer, inner_probs, p, draw_count):
            counts = torch.bincount(torch.multinomial(p, draw_count, replacement=True, generator=generator), minlength=outer)
            selected = torch.multinomial(inner_probs, outer, replacement=True, generator=generator)
            coefficients = -(inner_probs / outer).repeat(outer)
            for idx in (counts > 0).nonzero().flatten().tolist():
                coefficients[idx*len(inner_probs) + int(selected[idx])] += counts[idx] / (draw_count*outer*p[idx])
            return max(0., float(coefficients @ gram @ coefficients))
        for label, p in policies.items():
            mse = sum(row['fraction'] * (shared_inner_mse(row['pair_gram'], k, weights, p, draws)
                      + shared_inner_mse(row['unary_gram'], len(fields), unary_inner, unary_probs, 1)) for row in rows)
            extra = (amortized + policy_overhead if label == 'lagged_gradient_cost' else
                     amortized + c8_sampler_timing['charged_overhead_seconds'] if label == 'c8_priorities' else 0.)
            seconds = fixed + expected_unique_cost(p, costs, draws) + extra
            replicates = []
            for _ in range(args.replicates):
                draw_error = sum(row['fraction'] * (sample_error(row['pair_gram'], k, weights, p, draws)
                                 + sample_error(row['unary_gram'], len(fields), unary_inner, unary_probs, 1)) for row in rows)
                replicates.append(draw_error * seconds)
            interval = torch.quantile(torch.tensor(replicates, dtype=torch.float64), torch.tensor([.05,.95], dtype=torch.float64)).tolist()
            results[label] = {'gradient_mse': mse, 'full_estimator_seconds': seconds, 'mse_cost': mse * seconds,
                              'descriptive_mse_cost_p05_p95': interval,
                              'pair_probabilities': p.tolist(), 'maximum_importance_weight': float((1/(k*p)).max())}
        baseline = results['uniform']['mse_cost']
        ratio = results['lagged_gradient_cost']['mse_cost'] / baseline if baseline > 0 else None
        receipt.update(status='completed', source_family=family_name, source_identity=source_identity,
                       source_panel_sha256=file_sha256(panel_path), core_sha256=file_sha256(artifacts_path),
                       config_sha256=file_sha256(args.config), selected_bank_sha256=bank['sha256'],
                       prefit_sha256=file_sha256(prefit_path), priorities_sha256=file_sha256(args.c8_output),
                       complete_update_receipt_sha256=file_sha256(args.complete_update_receipt),
                       observed_complete_update_seconds=float(update_seconds),
                       sample_ids=ids, train_population=len(dataset), actual_head_parameters=sum(p.numel() for p in parameters),
                       parameterization='residual_head', pair_draws=draws, policies=results,
                       gradient_cost_uniform_ratio=ratio, overhead_fraction=overhead,
                       projected_training_seconds=projected_training_seconds,
                       lower_observed_training_seconds=lower_observed_training_seconds,
                       overhead_denominator_scope='provisional 12000 times minimum observed steady B64 update; excludes callbacks and requires final actual-cost recheck',
                       qualified_one_path=bool(ratio is not None and ratio <= .8 and overhead <= .05),
                       qualification_status='provisional until final measured candidate training cost and overhead reconciliation',
                       timing=timing_evidence, prior_research_seconds=preparation_seconds,
                       research_costs_sha256=file_sha256(args.research_costs),
                       sampler_timing=sampler_timing, c8_sampler_timing=c8_sampler_timing,
                       audit_through_measurement_seconds=elapsed, save_margin_seconds=args.save_margin_seconds,
                       amortization_updates=12000, expanded_leaves='enumerated and charged in fixed cost',
                       estimator='outer multinomial with one shared inner draw per unique outer ID',
                       interpretation='single-source head-gradient audit; both SOURCE families must qualify before policy adoption')
    except BaseException as exc:
        receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        torch_module = sys.modules.get('torch')
        if torch_module is not None and torch_module.cuda.is_initialized():
            try:
                torch_module.cuda.synchronize()
            except (RuntimeError, OSError) as exc:
                receipt.update(status='failed', teardown_error=f'{type(exc).__name__}: {exc}')
        if dataset is not None:
            try:
                dataset.close()
            except (RuntimeError, OSError) as exc:
                receipt.update(status='failed', teardown_error=f'{type(exc).__name__}: {exc}')
        receipt['end_unix'] = time.time()
        receipt['wall_seconds'] = receipt['end_unix'] - start
        with locked_ledger(ledger) as state:
            entry = state['active'].pop(token)
            state['closed'].append({**entry, 'token': token, 'end_unix': receipt['end_unix'], 'seconds': receipt['wall_seconds']})
        atomic_json(args.output, receipt)

if __name__ == '__main__':
    main()
