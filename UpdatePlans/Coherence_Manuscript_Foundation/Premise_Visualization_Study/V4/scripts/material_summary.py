#!/usr/bin/env python3
"""Summarize supported material contrasts in the complete candidate catalog."""
import argparse
from pathlib import Path
import pandas as pd

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path)
a=p.parse_args()
c=pd.read_csv(a.directory/'candidates.csv')
out=[]
for pair,q0 in c.groupby('pair',sort=True):
 for gate in ['all','matched_10','matched_20','matched_40']:
  q=q0 if gate=='all' else q0[q0[gate]]
  for metric,thresholds in [('state',(0.03,0.05)),('copula',(0.05,0.1))]:
   v=q[q.state_support] if metric=='state' else q
   effect=v.state_gap_gt_y_sd if metric=='state' else v.copula_tv_a-v.copula_tv_b
   for t in thresholds:
    hit=effect.abs()>=t
    decisive=v.l2_dominance!=0
    reversal=decisive & ((effect*v.l2_dominance)<0)
    out.append(dict(pair=pair,stratum=gate,metric=metric,threshold=t,supported_n=len(v),material_n=int(hit.sum()),material_fraction=hit.mean() if len(v) else None,dominant_n=int(decisive.sum()),material_reversal_n=int((hit&reversal).sum()),material_reversal_fraction_of_dominant=(hit&reversal).sum()/decisive.sum() if decisive.sum() else None))
pd.DataFrame(out).to_csv(a.directory/'material_prevalence.csv',index=False)
