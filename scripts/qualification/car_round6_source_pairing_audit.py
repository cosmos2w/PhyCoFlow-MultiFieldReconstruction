#!/usr/bin/env python3
"""Compare sensor and seed manifests of two cached SOURCE TRAIN800 artifacts."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bank',type=Path,required=True);p.add_argument('--panel',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();start=time.monotonic();torch.set_num_threads(1)
    panel=torch.load(a.panel,map_location='cpu',weights_only=True)
    receipt=json.loads((a.bank/'receipt.json').read_text())
    seeds={sample:row['member_seeds'] for row in panel['coupling'] for sample in row['sample_ids']}
    rows=[]
    for index,sample in enumerate(receipt['sample_ids']):
        data=torch.load(a.bank/f'{index:04d}.pt',map_location='cpu',weights_only=True)
        if data['sample_id']!=sample or sample not in panel['observation_manifest'] or sample not in seeds:raise ValueError('sample mismatch')
        pairs=list(zip(data['obs_indices'].reshape(-1).tolist(),data['obs_fields'].reshape(-1).tolist()))
        rows.append({'sample_id':sample,'same_ordered_sensor_index_field_pairs':pairs==panel['observation_manifest'][sample],
            'native_member_seeds':data['member_seeds'],'panel_member_seeds':seeds[sample],
            'same_member_seeds':data['member_seeds']==seeds[sample]})
    if len(rows)!=800:raise ValueError('incomplete matched panel')
    output={'status':'completed','sample_count':len(rows),'ordered_sensor_match_count':sum(r['same_ordered_sensor_index_field_pairs'] for r in rows),
        'member_seed_match_count':sum(r['same_member_seeds'] for r in rows),'samples':rows,
        'interpretation':'Seed lists differ in representation and may generate different priors; do not pair predictions by sample alone.',
        'wall_seconds':time.monotonic()-start}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:json.dump(output,stream,indent=2);stream.write('\n')
if __name__=='__main__':main()
