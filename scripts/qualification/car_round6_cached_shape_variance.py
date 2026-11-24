#!/usr/bin/env python3
"""Paired SOURCE TRAIN800 J4 Shape variability: condition, member pair, query."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
import yaml
from phycoflow_reconstruction.coherence import build_enabled_families
from phycoflow_reconstruction.coherence.query_runtime import _shape_grid
from phycoflow_reconstruction.coherence.query_sampling import SharedQuerySampler
from phycoflow_reconstruction.coherence.shape_compute import LazyPooledShape
from phycoflow_reconstruction.data.factory import open_field_dataset
from phycoflow_reconstruction.training.internal_calibration import InternalCalibration
from scripts.qualification.car_round6_joint_audit import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds-limit',type=float,default=900)
    args=p.parse_args()
    started=time.monotonic();torch.set_num_threads(1)
    cfg=yaml.safe_load((args.run/'resolved_config.yaml').read_text())
    panel_path=args.run/'artifacts/source_panel.pt'; core_path=args.run/'artifacts/scalable_coherence.pt'
    panel=torch.load(panel_path,map_location='cpu',weights_only=True)
    if len(panel['sample_ids'])!=800 or panel['members'].shape[1]!=4:
        raise ValueError('matched TRAIN800 SOURCE J4 panel required')
    core=torch.load(core_path,map_location='cpu',weights_only=True)
    ds=open_field_dataset(cfg['dataset'],split='train')
    try:
        family=build_enabled_families(cfg['coherence'],ds.data_spec,ds.normalizer)['topology']
        family_path=args.run/'artifacts/topology_family.pt'
        family.load_state_artifact(torch.load(family_path,map_location='cpu',weights_only=True))
        family._raster_map(panel['coordinates'])
        mask=family.domain_mask if family.native_masked else None
        shape=LazyPooledShape.from_state_dict(core['shape_artifact'])
        pref=cfg['coherence']['scalable_anchored_coherence']['common_prefit']
        if sha256(pref['path'])!=pref['sha256']:
            raise ValueError('prefit hash mismatch')
        prefit=json.loads(Path(pref['path']).read_text())
        cal=InternalCalibration(prefit['priorities'],prefit['units'],prefit['active']);cal.load_prefit(prefit)
        weights={k:float(family.spatial_objective.weights[f'{k}.persistence'])*float(cal.coefficients[f'shape.{k}']) for k in ('self','mutual')}
        values=[]; components=[]
        with torch.no_grad():
            for start in range(0,800,64):
                if time.monotonic()-started>args.seconds_limit:
                    raise TimeoutError('bounded CPU analysis expired')
                end=min(start+64,800); reference=panel['reference'][start:end]
                grids=[_shape_grid(family,panel['members'][start:end,j],reference) for j in range(4)]
                truth=_shape_grid(family,reference,reference)
                q=[SharedQuerySampler(tuple(family.field_names),seed=seed*1000003+start,epsilon=1).draw() for seed in (1,2)]
                rows=[]; blocks=[]
                for pair in ((0,1),(2,3)):
                    row=[]; block=[]
                    for query in q:
                        score=shape.score(grids[pair[0]],grids[pair[1]],truth,query,valid_mask=mask,mode='conditional')
                        row.append((weights['self']*score['self']+weights['mutual']*score['mutual']).double().numpy())
                        block.append({k:float(score[k].double().mean()) for k in ('self','mutual')})
                    rows.append(row);blocks.append(block)
                values.extend(np.asarray(rows).transpose(2,0,1).tolist());components.append(blocks)
        v=np.asarray(values,dtype=np.float64)
        if v.shape!=(800,2,2) or not np.isfinite(v).all():
            raise ValueError('incomplete or nonfinite shape panel')
        condition_means=v.mean(axis=(1,2));member_means=v.mean(axis=2);query_means=v.mean(axis=1)
        output={'schema':'car_round6_cached_shape_variance_v1','status':'completed','sample_count':800,
            'sample_ids':panel['sample_ids'],'source_panel_sha256':sha256(panel_path),'core_sha256':sha256(core_path),
            'family_sha256':sha256(family_path),'prefit_sha256':pref['sha256'],'weights':weights,
            'members':[[0,1],[2,3]],'query_seeds':[1,2], 'batch_size':64,
            'condition_variance_of_four_draw_mean':float(np.var(condition_means,ddof=1)),
            'member_pair_paired_difference_half_mean_square':float(np.mean((member_means[:,0]-member_means[:,1])**2)/2),
            'query_paired_difference_half_mean_square':float(np.mean((query_means[:,0]-query_means[:,1])**2)/2),
            'per_condition_scores':values,'batch_component_means':components,
            'interpretation':'paired finite draws, not independent datasets or an additive variance decomposition; common queries are shared within each batch',
            'limitation':'fixed SOURCE cached J4 selected support and 256 observations; complements, does not replace, mature fresh-J2 action reassessment',
            'wall_seconds':time.monotonic()-started}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as out:json.dump(output,out,indent=2,allow_nan=False);out.write('\n')
    finally:ds.close()
if __name__=='__main__':main()
