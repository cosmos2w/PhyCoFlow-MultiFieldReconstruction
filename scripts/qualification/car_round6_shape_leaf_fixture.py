#!/usr/bin/env python3
"""One mature real-B64 Shape semantic-component gradient fixture."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch
from phycoflow_reconstruction.coherence.query_runtime import _shape_grid
from phycoflow_reconstruction.training.gradient_balance import _flat_gradient
from phycoflow_reconstruction.training.post_training import _coherence_objective
from phycoflow_reconstruction.training.retention import post_training_mode
from phycoflow_reconstruction.training.source import set_trainable_scope
from scripts.qualification.car_round5_panel_replay import load_context,batches
from scripts.qualification.car_round6_joint_audit import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=torch.device,required=True)
    args=p.parse_args()
    if not os.environ.get('CAR_ROUND6_LEASE_TOKEN') or args.device.type!='cuda':p.error('GPU preparation lease required')
    started=time.monotonic();torch.set_num_threads(4)
    cfg,model,dataset,panel,runtime,scales=load_context(args.run/'resolved_config.yaml',args.device)
    ckpt_path=args.run/'checkpoints/epoch_100.pt';ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False)
    if ckpt['global_step']!=1200:raise ValueError('mature epoch-100 checkpoint required')
    set_trainable_scope(model,cfg['trainable']);model.load_state_dict(ckpt['model'],strict=True);post_training_mode(model,cfg)
    params=[p for p in model.parameters() if p.requires_grad]
    source=batches(dataset,panel,cfg,args.device)
    try:
        batch=next(iter(source));runtime.begin_step(0)
        losses,endpoint={},{}
        _coherence_objective(model,batch,runtime.families,{n:None for n in runtime.families},cfg,step=0,
            generator=torch.Generator(device=args.device).manual_seed(47+1_000_003),family_scales=scales,
            family_losses=losses,endpoint_context=endpoint,scalable_runtime=runtime,scalable_probe=False)
        family=runtime.families['topology'];shape=runtime.adapter.shape
        ref=endpoint['target'];first=_shape_grid(family,endpoint['members'][0],ref);second=_shape_grid(family,endpoint['members'][1],ref)
        with torch.no_grad():truth=_shape_grid(family,ref,ref)
        mask=family.domain_mask if family.native_masked else None
        norm=shape._domain(first,mask)
        queries=tuple(shape._sampled_queries(runtime.batch))
        banks=[shape._diagram_bank(grid,queries,mask) for grid in (first,second)]
        with torch.no_grad():target=shape._diagram_bank(truth,queries,mask)
        count=len(batch.sample_ids); dims=len(shape.dimensions)
        attraction=(shape._distance(banks[0],target,norm)+shape._distance(banks[1],target,norm))/2
        separation=-shape._distance(banks[0],banks[1],norm)/2
        pieces={'attraction':attraction.reshape(len(queries),count,dims,2),
                'separation':separation.reshape(len(queries),count,dims,2)}
        floors=torch.stack([shape.scales[q[0]] for q in queries]).to(attraction)
        qweights=attraction.new_tensor([q[4] for q in queries])
        semantic={}
        for kind in ('self','mutual'):
            indices=torch.tensor([i for i,q in enumerate(queries) if q[0][0]==kind],device=args.device,dtype=torch.long)
            coefficient=float(runtime.coefficients[f'shape.{kind}'])*float(family.spatial_objective.weights[f'{kind}.persistence'])
            for sign,values in pieces.items():
                normalized=values/floors[:,None]
                for category,pos,weight in (('finite',0,0.9),('essential',1,0.1)):
                    per_query=normalized[...,pos].mean(-1)*qweights[:,None]
                    semantic[f'{kind}.{category}.{sign}']=coefficient*weight*per_query.index_select(0,indices).sum(0).mean()
        reconstructed=sum(semantic.values());production=losses['topology']
        difference=abs(float((reconstructed-production).detach().cpu()))
        if not torch.allclose(reconstructed,production,rtol=2e-5,atol=2e-6):
            raise ValueError(f'Shape semantic reconstruction mismatch {difference}')
        full_grad=_flat_gradient(production,params).double()
        out={}
        for name,value in semantic.items():
            grad=_flat_gradient(value,params).double()
            out[name]={'value':float(value.detach()),'parameter_gradient_norm':float(torch.linalg.vector_norm(grad)),
                'parameter_gradient_rms':float(grad.square().mean().sqrt()),'dot_full_shape':float(grad @ full_grad)}
        result={'status':'completed','checkpoint_sha256':sha256(ckpt_path),'sample_ids':list(batch.sample_ids),
            'family':'topology','scope':'mature model, first real TRAIN batch, one shared query; same J2 endpoint graph',
            'family_scalar':float(production.detach()),'reconstructed_scalar':float(reconstructed.detach()),
            'absolute_reconstruction_difference':difference,'full_shape_parameter_gradient_norm':float(torch.linalg.vector_norm(full_grad)),
            'components':out,'coefficient_definition':'physical component and frozen prefit weight; finite 0.9 and essential 0.1, normalized per category, sampled query weighting',
            'wall_seconds':time.monotonic()-started}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as out_file:json.dump(result,out_file,indent=2,allow_nan=False);out_file.write('\n')
    finally:source.close();dataset.close()
if __name__=='__main__':main()
