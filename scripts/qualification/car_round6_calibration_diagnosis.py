#!/usr/bin/env python3
"""Summarize terminal TRAIN800 calibration without additional model evaluation."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np


def mean(rows,counts):return np.average(np.asarray(rows,dtype=float),axis=0,weights=np.asarray(counts,dtype=float))

def signs(rows,counts):
    values=np.asarray(rows,dtype=float);w=np.asarray(counts,dtype=float)
    return {'mean':mean(values,w).tolist(),'positive_batches':(values>0).sum(0).tolist(),
        'negative_batches':(values<0).sum(0).tolist(),'positive_condition_weight':mean(values>0,w).tolist()}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--history',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();d=json.loads(a.audit.read_text());h=json.loads(a.history.read_text())
    if d['status']!='completed' or d['panel_size']!=800 or sum(d['batch_counts'])!=800:raise ValueError('complete TRAIN800 calibration required')
    fields=d['fields'];counts=d['batch_counts'];cal=d['calibration'];ref=h['reference']
    if fields!=ref['field_names'] or len(d['paired_pre'])!=13:raise ValueError('field or batch identity mismatch')
    pre=np.asarray(cal['pre']);zero=np.asarray(cal['zero']);probe=np.asarray(cal['probe']);gp=float(cal['gamma_probe'])
    b=12*(zero-pre);r=12*(probe-zero)/gp;t=-np.maximum(pre,0)/10
    w=np.asarray(ref['risk_scales'])/np.asarray(ref['absolute_allowance'])
    denominator=float(np.dot(w*r,w*r));numerator=float(np.dot(w*r,w*(t-b)))
    fitted=float(np.clip(numerator/denominator,0,gp)) if denominator>0 else None
    if fitted is None or not math.isclose(fitted,float(cal['gamma_star']),rel_tol=1e-9,abs_tol=1e-12):raise ValueError('gain recomputation mismatch')
    arms={}
    for label,key in [('zero','zero_actions'),('probe','probe_actions'),('star','star_actions')]:
        rows=d[key]
        if len(rows)!=13 or [len(x['sample_ids']) for x in rows]!=counts:raise ValueError('missing matched action batch')
        out={}
        for role in ('family','field'):
            for stage in ('config','composed','clipped','actual_displacement'):
                k=f'{role}_products_{stage}'
                out[k]=signs([row[k] for row in rows],counts)
        post=np.asarray([row['post'] for row in rows]);rawpre=np.asarray(d['paired_pre'])
        out['finite_field_change']=signs(post-rawpre,counts)
        out['clip_pre_norm']=signs([[row['clip_pre_norm']] for row in rows],counts)
        for query in ('fixed','independent'):
            differences=[];batch_mean=[]
            for row,count in zip(rows,counts):
                before=np.asarray(row[f'shape_pre_{query}']['per_condition']);after=np.asarray(row[f'shape_post_{query}']['per_condition'])
                if len(before)!=count or len(after)!=count:raise ValueError('Shape condition count mismatch')
                differences.extend((after-before).tolist());batch_mean.append(float(np.mean(after-before)))
            vals=np.asarray(differences)
            out[f'shape_{query}_change']={'mean':float(vals.mean()),'condition_decreases':int((vals<0).sum()),
                'condition_increases':int((vals>0).sum()),'batch_mean_decreases':int((np.asarray(batch_mean)<0).sum()),
                'batch_mean_increases':int((np.asarray(batch_mean)>0).sum())}
        arms[label]=out
    member=np.asarray(ref['risk_scales']);allow=np.asarray(ref['absolute_allowance']);binding=h['reference']
    output={'status':'completed','sample_count':800,'field_order':fields,'calibration_recomputed':{'b':b.tolist(),
        'r':r.tolist(),'target':t.tolist(),'weights':w.tolist(),'unclipped_gain':numerator/denominator,
        'clipped_gamma_star':fitted,'numerator':numerator,'denominator':denominator,
        'baseline_drift_over_target_magnitude':(np.abs(b)/np.abs(t)).tolist(),
        'baseline_weighted_residual_norm':float(np.linalg.norm(w*(b-t))),
        'probe_weighted_residual_norm':float(np.linalg.norm(w*(b+gp*r-t)))},
        'constraint_reference':binding,'allowance_over_scale':(allow/member).tolist(),
        'coefficients_constant_all_recorded_epochs':h['coefficients_constant'],
        'warmup_action_records':h['lr_update_count'],'first_epoch_mean_lr':h['epoch_lr_first'],'epoch100_mean_lr':h['epoch_lr_100'],
        'arms':arms,'interpretation':'predeclared target-fitting picks zero because measured baseline drift already exceeds target and probe response moves further negative; this is a one-step TRAIN diagnostic, not a learned trajectory or held-out efficacy result',
        'limitations':'separate field VJPs were used for positive probe; zero is exactly the no-feedback action. No moment-reset or preconditioner-isolation experiment has been performed.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:json.dump(output,stream,indent=2,allow_nan=False);stream.write('\n')
if __name__=='__main__':main()
