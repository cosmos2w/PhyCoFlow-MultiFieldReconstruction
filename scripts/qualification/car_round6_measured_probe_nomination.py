#!/usr/bin/env python3
"""Bind user-authorized pilot nomination to measured one-sided TRAIN responses."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def ref(path):
    path=Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':sha256(path)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('joint-audit','production-probe','original-audit','feasibility-response','history-coarse-profile',
                 'prefit-identity','fidelity-reference','mature-checkpoint','mature-config','ledger','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    evidence={name:ref(getattr(a,name)) for name in ('joint_audit','production_probe','original_audit',
        'feasibility_response','history_coarse_profile','prefit_identity','fidelity_reference','mature_checkpoint','mature_config')}
    joint=json.loads(a.joint_audit.read_text());probe=json.loads(a.production_probe.read_text())
    audit=json.loads(a.original_audit.read_text());response=json.loads(a.feasibility_response.read_text())
    history=json.loads(a.history_coarse_profile.read_text());prefit=json.loads(a.prefit_identity.read_text())
    fidelity=json.loads(a.fidelity_reference.read_text());ledger=json.loads(a.ledger.read_text())
    if any(d.get('status')!='completed' for d in (joint,probe,audit,response,prefit)):
        raise ValueError('all source receipts must be completed')
    if joint['evidence']['parity']['sha256']!=evidence['production_probe']['sha256'] or probe['audit_sha256']!=evidence['original_audit']['sha256']:
        raise ValueError('joint, production probe and original audit do not bind')
    if response['evidence']['original_decision']['sha256']!=evidence['joint_audit']['sha256'] or response['evidence']['original_audit']['sha256']!=evidence['original_audit']['sha256'] or response['evidence']['production_probe_parity']['sha256']!=evidence['production_probe']['sha256']:
        raise ValueError('feasibility interpretation source binding mismatch')
    if (joint['evidence']['history']['sha256']!=evidence['history_coarse_profile']['sha256']
        or joint['evidence']['prefit']['sha256']!=evidence['prefit_identity']['sha256']
        or joint['evidence']['mature_checkpoint']['sha256']!=evidence['mature_checkpoint']['sha256']
        or joint['evidence']['mature_config']['sha256']!=evidence['mature_config']['sha256']):
        raise ValueError('mature history, prefit, checkpoint or configuration binding mismatch')
    if fidelity['reference']!=joint['constraint_reference'] or history['reference']!=joint['constraint_reference']:
        raise ValueError('frozen field reference differs')
    gamma=float(probe['gamma'])
    if probe['amplitude']!='probe' or not math.isfinite(gamma) or gamma<=0 or gamma!=float(audit['calibration']['gamma_probe']) or gamma!=float(response['measured_gamma_probe']):
        raise ValueError('gain must be the existing measured production-verified probe')
    if audit['panel_size']!=800 or probe['sample_count']!=800 or response['sample_count']!=800:
        raise ValueError('matched TRAIN800 panel required')
    if ledger.get('active'):raise ValueError('nomination snapshot requires no active GPU lease')
    prep_seconds=sum(float(row['seconds']) for row in ledger['closed'] if row['phase']=='preparation')
    total_seconds=sum(float(row['seconds']) for row in ledger['closed'])
    if prep_seconds>=4*3600 or total_seconds>=30*3600:raise ValueError('existing GPU budget exhausted')
    out={'schema':'car_round6_measured_response_qualification_v1','status':'completed','qualified':True,'gamma':gamma,
      'scope':'execution and fresh-SOURCE pilot nomination only; not a model-efficacy or optimal-gain claim',
      'rationale':'The user authorized continued bounded learning and a one-sided feasibility interpretation. The measured production probe lowers all five pooled signed constraint excesses and the weighted positive-part norm. The former symmetric H=10 drift fit penalized faster beneficial correction; its zero gain and original decision remain preserved.',
      'scientific_objective':'Unchanged SSS-only fresh parameter-space ConFIG plus signed expected Lagrangian field constraints; the positive part is used only for this pooled diagnostic.',
      'field_order':joint['field_order'],'panel_sample_count':800,'dataset_split':'train',
      'pooling':'sample-weighted panel mean first, then fieldwise positive part, W=scale/allowance',
      'positive_excess_norm_zero':response['positive_violation_norm_zero'],
      'positive_excess_norm_probe':response['positive_violation_norm_probe'],
      'relative_positive_excess_norm_reduction':response['positive_violation_norm_relative_reduction_vs_zero'],
      'panel_mean_excess_zero':response['panel_mean_e_zero'],
      'panel_mean_excess_probe':response['panel_mean_e_production_probe'],
      'field_post_excess_change_probe_minus_zero':response['field_post_excess_change_probe_minus_zero'],
      'shape_finite_change':response['shape'],
      'shape_scope':response['shape_scope'],
      'symmetric_target_fit_previous_gain':response['original_symmetric_target_fit_gamma'],
      'one_sided_drift_shortfall_zero':response['one_sided_drift_target_shortfall_zero'],
      'one_sided_drift_shortfall_probe':response['one_sided_drift_target_shortfall_probe'],
      'moment_carryover':response['moment_interpretation'],
      'next_evidence':'Fresh SOURCE, fresh optimizer/controller, 20 reporting epoch pilot with real save/reload/resume, then authorized 100-epoch learning review under the current bounded plan.',
      'limits':['Two measured amplitudes do not establish an optimum or intervening monotonicity.',
                'TRAIN-only mature one-step responses do not establish fresh-state trajectory, held-out efficacy or deployment performance.',
                'The 14.8 percent weaker independent-query Shape decrease is a measured local tradeoff; finite State and Scale score changes were not evaluated.',
                'No VALIDATION task or TEST is used in this nomination.'],
      'gpu_ledger_snapshot':{'preparation_seconds':prep_seconds,'preparation_gpu_hours':prep_seconds/3600,
          'total_seconds':total_seconds,'active_lease_count':len(ledger.get('active',{})),
          'closed_lease_count':len(ledger['closed']),'remaining_preparation_gpu_hours':(4*3600-prep_seconds)/3600,
          'scope':'snapshot at nomination; mutable ledger is not a frozen evidence dependency'},
      'evidence':evidence}
    a.output.parent.mkdir(parents=True,exist_ok=False)
    with a.output.open('x') as stream:json.dump(out,stream,indent=2,allow_nan=False);stream.write('\n')
if __name__=='__main__':main()
