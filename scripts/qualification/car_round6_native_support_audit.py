#!/usr/bin/env python3
"""Compare native and selected support on the same cached SOURCE TRAIN800 J4 endpoints."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
import numpy as np
import torch
from phycoflow_reconstruction.coherence.physical_risk import physical_ensemble_risks
from scripts.qualification.car_round6_joint_audit import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    started=time.monotonic()
    torch.set_num_threads(1)
    receipt=json.loads((args.bank/'receipt.json').read_text())
    panel=torch.load(args.panel,map_location='cpu',weights_only=True)
    if receipt['status']!='completed' or receipt['split']!='train' or receipt['members']!=4 or receipt['native_points']!=40300:
        raise ValueError('completed native TRAIN800 J4 bank required')
    if receipt['checkpoint_sha256']!=panel['provenance']['source_checkpoint_sha256']:
        raise ValueError('SOURCE checkpoint mismatch')
    ids=panel['sample_ids']
    if len(ids)!=800 or len(set(ids))!=800 or set(ids)!=set(receipt['sample_ids']):
        raise ValueError('unmatched TRAIN800 sample IDs')
    positions={x:i for i,x in enumerate(ids)}
    result=[]
    for index, expected_id in enumerate(receipt['sample_ids']):
        path=args.bank/f'{index:04d}.pt'
        if sha256(path)!=receipt['files_sha256'][index]:
            raise ValueError(f'native bank hash mismatch: {index}')
        bank=torch.load(path,map_location='cpu',weights_only=True)
        if bank['sample_id']!=expected_id:
            raise ValueError('native bank order mismatch')
        selected=panel['query_indices'][positions[expected_id]].long()
        if selected.numel()!=4096 or selected.unique().numel()!=4096:
            raise ValueError('expected 4096 unique selected native sites')
        members=bank['members_model'][:,0].double()
        truth=bank['truth_model'][0].double()
        masses=bank['native_masses'].double()
        full_member,full_risk=physical_ensemble_risks(members,truth,masses)
        sub_member,sub_risk=physical_ensemble_risks(members[:,selected],truth[selected],masses[selected],
            total_mass=masses.sum(),sampling='without_replacement',probabilities=selected.numel()/masses.numel())
        result.append({'sample_id':expected_id,'native_member_mse':full_member.tolist(), 'selected_member_mse':sub_member.tolist(),
                       'native_mean_risk':full_risk.tolist(), 'selected_mean_risk':sub_risk.tolist()})
    def mean(key): return np.mean([r[key] for r in result],axis=0).tolist()
    output={'status':'completed','schema':'car_round6_source_native_selected_support_v1',
        'sample_count':800,'members':4,'native_points':40300,'selected_points':4096,
        'native_bank_receipt_sha256':sha256(args.bank/'receipt.json'),'selected_panel_sha256':sha256(args.panel),
        'field_order':receipt['field_order'], 'source_checkpoint_sha256':receipt['checkpoint_sha256'],
        'definition':'same cached native J4 predictions and truth; evaluate at full physical measure and the frozen panel sites with inclusion correction',
        'observation_regime':'native fixed 256 observations, not fresh training observations',
        'limitation':'SOURCE support sampling comparison only; says nothing about mature-child support overfitting',
        'mean':{k:mean(k) for k in ('native_member_mse','selected_member_mse','native_mean_risk','selected_mean_risk')},
        'per_condition':result,'wall_seconds':time.monotonic()-started}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as out:
        json.dump(output,out,indent=2,allow_nan=False);out.write('\n')
if __name__=='__main__':main()
