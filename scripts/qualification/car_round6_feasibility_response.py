#!/usr/bin/env python3
"""Reinterpret recorded paired TRAIN800 actions using one-sided feasibility.

This is a new resumption analysis. It neither alters the frozen target-fit
receipt nor evaluates an additional model, gain, split, or optimizer action.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def read(path):
    data=path.read_bytes();return json.loads(data),hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('audit','parity','diagnosis','decision','checkpoint','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    audit,audit_hash=read(a.audit);parity,parity_hash=read(a.parity)
    diagnosis,diagnosis_hash=read(a.diagnosis);decision,decision_hash=read(a.decision)
    if any(d.get('status')!='completed' for d in (audit,parity,diagnosis,decision)):
        raise ValueError('completed recorded audits required')
    if audit['panel_size']!=800 or sum(audit['batch_counts'])!=800 or parity['sample_count']!=800:
        raise ValueError('matched approximately 10 percent TRAIN800 panel required')
    if parity['audit_sha256']!=audit_hash or parity['amplitude']!='probe':
        raise ValueError('production probe must bind the original audit')
    if decision['evidence']['audit']['sha256']!=audit_hash or decision['evidence']['parity']['sha256']!=parity_hash:
        raise ValueError('original frozen decision does not bind those audits')
    cp_bytes=a.checkpoint.read_bytes();cp_hash=hashlib.sha256(cp_bytes).hexdigest()
    if any(d['checkpoint_sha256']!=cp_hash for d in (audit,parity)):
        raise ValueError('mature checkpoint mismatch')
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if ckpt['global_step']!=1200:raise ValueError('actual epoch 100 checkpoint required')
    ref=ckpt['external_policy']['reference'];fields=ref['field_names']
    if fields!=audit['fields'] or fields!=parity['field_order']:raise ValueError('field order mismatch')
    if [r['sample_ids'] for r in parity['per_batch']]!=[r['sample_ids'] for r in audit['probe_actions']]:
        raise ValueError('production probe is not matched to original batches')
    pre=np.asarray(audit['calibration']['pre'],dtype=float)
    zero=np.asarray(audit['calibration']['zero'],dtype=float)
    probe=np.asarray(parity['combined_post_mean'],dtype=float)
    allowance=np.asarray(ref['absolute_allowance'],dtype=float)
    scale=np.asarray(ref['risk_scales'],dtype=float)
    if np.any(allowance<=0) or np.any(scale<=0):raise ValueError('positive field reference units required')
    w=scale/allowance
    def positive_norm(x):return float(np.linalg.norm(w*np.maximum(x,0)))
    def positive_drift_shortfall(post):
        b=12*(post-pre);target=-np.maximum(pre,0)/10
        return float(np.linalg.norm(w*np.maximum(b-target,0)))
    npre,nzero,nprobe=map(positive_norm,(pre,zero,probe))
    if not (nzero>0 and np.isfinite([npre,nzero,nprobe]).all()):raise ValueError('nonfinite or degenerate reference')
    zero_shape=diagnosis['arms']['zero'];probe_shape=diagnosis['arms']['probe']
    geometry={}
    for query in ('fixed','independent'):
        z=zero_shape[f'shape_{query}_change'];q=probe_shape[f'shape_{query}_change']
        if z['mean']>=0:raise ValueError('baseline Shape decrease is not positive')
        geometry[query]={'zero_mean_change':z['mean'],'probe_mean_change':q['mean'],
            'probe_fraction_of_zero_decrease':float(q['mean']/z['mean']),
            'zero_condition_decreases':z['condition_decreases'],
            'probe_condition_decreases':q['condition_decreases'],
            'zero_batch_mean_decreases':z['batch_mean_decreases'],
            'probe_batch_mean_decreases':q['batch_mean_decreases']}
    states=ckpt['optimizer']['state'];steps=sorted({float(v['step']) for v in states.values()})
    moments={'optimizer_state_tensors':len(states),'unique_steps':steps,
        'nonzero_first_moment_tensors':sum(bool(torch.count_nonzero(v['exp_avg'])) for v in states.values()),
        'nonzero_second_moment_tensors':sum(bool(torch.count_nonzero(v['exp_avg_sq'])) for v in states.values()),
        'saved_actual_multipliers':ckpt['external_policy']['multipliers']}
    out={'schema':'car_round6_feasibility_response_v1','status':'completed',
        'purpose':'resumption analysis under user-authorized one-sided feasibility interpretation',
        'supersedes_or_modifies_prior_receipts':False,'fields':fields,'sample_count':800,'split':'train',
        'mature_checkpoint_sha256':cp_hash,'gamma_zero':0.0,'measured_gamma_probe':parity['gamma'],
        'constraints':'positive part after sample-weighted panel pooling, W=scale/allowance; violations are signed e',
        'panel_mean_e_pre':pre.tolist(),'panel_mean_e_zero':zero.tolist(),'panel_mean_e_production_probe':probe.tolist(),
        'field_weight':w.tolist(),'positive_violation_norm_pre':npre,'positive_violation_norm_zero':nzero,
        'positive_violation_norm_probe':nprobe,'positive_violation_norm_relative_reduction_vs_zero':1-nprobe/nzero,
        'positive_violation_squared_reduction_vs_zero':1-(nprobe/nzero)**2,
        'field_post_excess_change_probe_minus_zero':(probe-zero).tolist(),
        'one_sided_drift_target_shortfall_zero':positive_drift_shortfall(zero),
        'one_sided_drift_target_shortfall_probe':positive_drift_shortfall(probe),
        'shape':geometry,'shape_scope':'fixed and independent query finite scores from original separately accumulated field-VJP replay; production combined-VJP parity verified separately on fields',
        'production_probe_response_difference_norm':parity['combined_vs_separate_weighted_response_norm'],
        'original_symmetric_target_fit_gamma':audit['calibration']['gamma_star'],
        'original_symmetric_target_fit_interpretation':'the clipped zero is correct for its squared drift-matching criterion; the criterion penalizes faster risk correction',
        'recommendation':'Use the already measured probe as a candidate for the authorized fresh-SOURCE 20-epoch pilot; record field/Shape tradeoffs and make no efficacy or optimal-gain claim.',
        'one_sided_selection_limitation':'A one-sided drift shortfall is zero for both measured amplitudes; preferring the probe uses the measured remaining-violation endpoint, not a unique fit or a gain grid.',
        'optimizer_moment_evidence':moments,
        'moment_interpretation':'zero current feedback acts through epoch-100 Adam moments trained with previous nonzero feedback; it is not a fresh-state integral or PI baseline',
        'focused_optional_shadow_question':'At the same mature parameters and inputs compare zero-current-feedback Adam actions with actual versus reset first moment while preserving second moments and counters; this isolates first-moment carryover, not the full fresh-SOURCE trajectory. The user-authorized fresh-state pilot need not wait for it.',
        'limits':['one-step matched TRAIN panel; no VALIDATION task or TEST is used',
                  'positive-part on pooled mean does not assert per-condition or per-batch feasibility',
                  'State and Scale are assessed by first-order products, while finite query-score changes shown here are Shape only',
                  'the two amplitudes do not establish monotonicity between them or beyond the probe'],
        'evidence':{'original_audit':{'path':str(a.audit),'sha256':audit_hash},
          'production_probe_parity':{'path':str(a.parity),'sha256':parity_hash},
          'original_diagnosis':{'path':str(a.diagnosis),'sha256':diagnosis_hash},
          'original_decision':{'path':str(a.decision),'sha256':decision_hash},
          'mature_checkpoint':{'path':str(a.checkpoint),'sha256':cp_hash}}}
    a.output.parent.mkdir(parents=True,exist_ok=False)
    with a.output.open('x') as stream:json.dump(out,stream,indent=2,allow_nan=False);stream.write('\n')
if __name__=='__main__':main()
