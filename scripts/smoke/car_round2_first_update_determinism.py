"""Replay a saved Senseiver full-model first update without optimizer updates."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(args):
    import torch
    import yaml

    from phycoflow_reconstruction.coherence import build_enabled_families
    from phycoflow_reconstruction.coherence.compose import three_family_order
    from phycoflow_reconstruction.data.topology_subset import iter_full_pass_indices
    from phycoflow_reconstruction.data.training_batches import build_training_batch_source
    from phycoflow_reconstruction.training.fidelity_budget import _flat_gradient
    from phycoflow_reconstruction.training.multitask import FamilyScalarizer, OuterAggregator
    from phycoflow_reconstruction.training.post_training import _coherence_objective
    from phycoflow_reconstruction.training.retention import post_training_mode
    from phycoflow_reconstruction.training.run_store import (
        load_model_state_strict,
        load_project_checkpoint,
    )
    from phycoflow_reconstruction.training.scalable_fitting import _digest
    from phycoflow_reconstruction.training.scalable_runtime import ScalableRuntime
    from phycoflow_reconstruction.training.source import load_source_model, set_trainable_scope

    if args.mode == 'deterministic':
        torch.use_deterministic_algorithms(True)
    run = args.run
    cfg = yaml.safe_load((run / 'resolved_config.yaml').read_text())
    if cfg['model']['name'] != 'senseiver' or cfg.get('residual', {}).get('enabled', False):
        raise ValueError('this J1 replay requires the Senseiver full-model qualification')
    checkpoint = load_project_checkpoint(run / 'checkpoints/epoch_000.pt')
    device = torch.device('cuda:0')
    model, dataset, _ = load_source_model(cfg, device)
    set_trainable_scope(model, cfg['trainable'])
    load_model_state_strict(model, checkpoint['model'])
    actual = model.state_dict()
    if any(not torch.equal(actual[k].detach().cpu(), v.cpu()) for k, v in checkpoint['model'].items()):
        raise ValueError('loaded epoch-0 model is not bitwise equal to saved state')
    families = build_enabled_families(cfg['coherence'], dataset.data_spec, dataset.normalizer)
    for name, family in families.items():
        family.to(device).load_state_artifact(torch.load(run/'artifacts'/f'{name}_family.pt', map_location=device, weights_only=True))
        family.load_state_dict(checkpoint['family_states'][name])
    artifact = torch.load(run/'artifacts/scalable_coherence.pt', map_location='cpu', weights_only=True)
    saved_runtime = checkpoint['scalable_runtime']
    runtime = ScalableRuntime(artifact, families, steps_per_epoch=12, expected_provenance=artifact['provenance'],
                              seed=int(cfg['runtime']['seed'])+800048, scoring_mode='deterministic',
                              allocation_mode=saved_runtime['allocation_mode'], uniform_epochs=saved_runtime['uniform_epochs'])
    runtime.calibration.load_prefit(saved_runtime['calibration']['artifact'])
    runtime.load_state_dict(saved_runtime)
    seed = int(cfg['runtime']['seed'])
    indices = iter_full_pass_indices(len(dataset), 1, 64, seed=seed+17, start_step=0)
    source = build_training_batch_source(dataset, indices, cfg, query_points=4096, device=device, start_step=0)
    try:
        batch = next(iter(source))
        post_training_mode(model, cfg)
        torch.set_rng_state(checkpoint['rng_state']['torch_cpu'])
        torch.cuda.set_rng_state_all(checkpoint['rng_state']['torch_cuda'])
        runtime.begin_step(0)
        losses, endpoint = {}, {}
        _result, _ = _coherence_objective(model, batch, families, {k:None for k in families}, cfg, step=0,
            generator=torch.Generator(device=device).manual_seed(seed+1_000_003), family_scales=checkpoint['family_scales'],
            family_losses=losses, endpoint_context=endpoint, scalable_runtime=runtime)
        selected = endpoint['coupling_selected_batch']
        coupling = json.loads((run/'metrics/training_coupling.jsonl').read_text().splitlines()[0])
        obs = _digest([selected.obs_indices, selected.obs_field_ids, selected.obs_values, selected.obs_valid_mask])
        query = _digest([selected.metadata.get('query_indices'), selected.query_coords, selected.query_valid_mask])
        if list(selected.sample_ids) != coupling['sample_ids'] or obs != coupling['observation_sha256'] or query != coupling['query_sha256']:
            raise ValueError('replayed B64/Q4096/observations differ from the saved first update')
        scalarizer = FamilyScalarizer(cfg['optimization']['multitask']['family_scalarization'])
        objectives, _ = scalarizer(losses, scales=checkpoint['family_scales'],
            weights={name:float(getattr(family,'family_weight',1.0)) for name,family in families.items()})
        params = tuple(p for p in model.parameters() if p.requires_grad)
        names = list(three_family_order(families))
        gradients = torch.stack([_flat_gradient(objectives[name], params) for name in names])
        fixed = gradients.detach().to(torch.float64)
        aggregator = OuterAggregator(cfg['optimization']['multitask']['coherence_aggregation'])
        first, _ = aggregator._aggregate(fixed)
        second, _ = aggregator._aggregate(fixed)
        torch.cuda.synchronize()
        torch.save({'mode':args.mode,'names':names,'gradients':gradients.cpu(), 'config_first':first.cpu(),
                    'config_second':second.cpu(), 'family_losses':{k:float(v.detach()) for k,v in objectives.items()},
                    'observation_sha256':obs,'query_sha256':query,'sample_ids':list(selected.sample_ids),
                    'model_bitwise_equal':True,'checkpoint_sha256':file_hash(run/'checkpoints/epoch_000.pt'),
                    'saved_coupling_sha256':file_hash(run/'metrics/training_coupling.jsonl')}, args.output)
    finally:
        source.close()
        dataset.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--worker', action='store_true')
    p.add_argument('--mode', choices=('default','deterministic'))
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--reference-run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--code-root', type=Path, required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--ledger', type=Path, required=True)
    args=p.parse_args()
    sys.path.insert(0, str(args.code_root/'src'))
    sys.path.insert(0, str(args.code_root))
    if args.worker:
        worker(args)
        return
    from phycoflow_reconstruction.evaluation.adaptive_campaign import locked_ledger, spent
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_uuid:
        raise ValueError('visible GPU must match physical UUID')
    start, token = time.time(), str(uuid.uuid4())
    with locked_ledger(args.ledger) as state:
        if any(v['gpu_uuid'] == args.gpu_uuid for v in state['active'].values()) or spent(state)+900 >= 124*3600:
            raise ValueError('GPU leased or diagnostic exceeds protected budget')
        args.output.mkdir(parents=True, exist_ok=False)
        state['active'][token] = {'gpu_uuid':args.gpu_uuid, 'phase':'first_update_determinism', 'start_unix':start}
    previous=signal.getsignal(signal.SIGALRM)
    def timeout(_signal, _frame):
        raise TimeoutError('900-second diagnostic limit reached')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(840)
    receipt={'schema':'car_round2_first_update_determinism_v1','status':'running','token':token,
             'start_unix':start,'gpu_uuid':args.gpu_uuid,'max_wall_seconds':900,'save_margin_seconds':60,
             'helper_sha256':file_hash(__file__), 'code_root':str(args.code_root), 'run':str(args.run), 'workers':[]}
    try:
        if args.reference_run is None:
            raise ValueError('both independent epoch-0 checkpoints are required')
        import torch

        from phycoflow_reconstruction.training.run_store import load_project_checkpoint
        first = load_project_checkpoint(args.run/'checkpoints/epoch_000.pt')['model']
        second = load_project_checkpoint(args.reference_run/'checkpoints/epoch_000.pt')['model']
        equal = first.keys() == second.keys() and all(torch.equal(first[k], second[k]) for k in first)
        receipt['epoch_zero_models_bitwise_equal'] = equal
        receipt['epoch_zero_checkpoint_sha256'] = [file_hash(r/'checkpoints/epoch_000.pt') for r in (args.run,args.reference_run)]
        if not equal:
            raise ValueError('independent epoch-0 model states differ bitwise')
        del first, second
        for mode in ('default','deterministic'):
            for index in range(2):
                output=args.output/f'{mode}_{index}.pt'
                command=[sys.executable, __file__, '--worker', '--mode', mode, '--run',str(args.run), '--output',str(output),
                         '--code-root',str(args.code_root), '--gpu-uuid',args.gpu_uuid, '--ledger',str(args.ledger)]
                env=dict(os.environ)
                if mode == 'deterministic': env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
                else: env.pop('CUBLAS_WORKSPACE_CONFIG',None)
                began=time.time()
                with (args.output/f'{mode}_{index}.stdout').open('x') as out, (args.output/f'{mode}_{index}.stderr').open('x') as err:
                    result=subprocess.run(command,stdout=out,stderr=err,env=env,check=False, timeout=max(1, 840-(time.time()-start)))
                receipt['workers'].append({'mode':mode,'index':index,'returncode':result.returncode,'start_unix':began,'end_unix':time.time(),
                                           'output':str(output) if result.returncode==0 else None,
                                           'stderr':str(args.output/f'{mode}_{index}.stderr'),
                                           'error_tail':(args.output/f'{mode}_{index}.stderr').read_text()[-4000:] if result.returncode else None})
                if result.returncode != 0:
                    break
        comparisons={}
        for mode in ('default','deterministic'):
            successful=[w for w in receipt['workers'] if w['mode']==mode and w['returncode']==0]
            if len(successful)!=2:
                failed=[w for w in receipt['workers'] if w['mode']==mode and w['returncode']!=0]
                comparisons[mode]={'status':'unsupported_or_failed','stderr':failed[-1]['stderr'] if failed else None}
                continue
            a,b=[torch.load(w['output'],map_location='cpu',weights_only=True) for w in successful]
            per_family={}
            for idx,name in enumerate(a['names']):
                x,y=a['gradients'][idx],b['gradients'][idx]
                per_family[name]={'norm_first':float(x.double().norm()),'norm_second':float(y.double().norm()),
                    'max_abs_difference':float((x-y).abs().max()),'bitwise_equal':torch.equal(x,y),
                    'sha256_first':hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest(),
                    'sha256_second':hashlib.sha256(y.contiguous().numpy().tobytes()).hexdigest()}
            comparisons[mode]={'status':'completed','per_family':per_family,
                'config_same_input_bitwise_equal':torch.equal(a['config_first'],a['config_second']),
                'config_same_input_max_abs':float((a['config_first']-a['config_second']).abs().max()),
                'config_cross_replay_max_abs':float((a['config_first']-b['config_first']).abs().max()),
                'model_bitwise_equal':a['model_bitwise_equal'] and b['model_bitwise_equal'],
                'observation_sha256':a['observation_sha256'],'query_sha256':a['query_sha256'],
                'checkpoint_sha256':a['checkpoint_sha256'],'saved_coupling_sha256':a['saved_coupling_sha256']}
        receipt.update(status='completed', comparisons=comparisons)
    except BaseException as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
        end=time.time(); receipt.update(end_unix=end, wall_seconds=end-start)
        with locked_ledger(args.ledger) as state:
            entry=state['active'].pop(token); state['closed'].append({**entry,'token':token,'end_unix':end,'seconds':end-start})
        with (args.output/'job_receipt.json').open('x') as f:
            json.dump(receipt,f,indent=2,allow_nan=False); f.write('\n')


if __name__ == '__main__':
    main()
