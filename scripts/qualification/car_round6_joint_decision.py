#!/usr/bin/env python3
"""Assemble a compact decision from completed, separately bound Round 6 audits."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import torch,yaml

def read(path):
 raw=path.read_bytes();data=json.loads(raw)
 if data.get('status')!='completed':raise ValueError(f'incomplete evidence: {path}')
 return data,{'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('run','audit','history','diagnosis','parity','shape','support','pairing','variance','prefit','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();sources={};records={}
 for name in ('audit','diagnosis','parity','shape','support','pairing','variance','prefit'):
  records[name],sources[name]=read(getattr(a,name))
 audit,diag,parity,shape,support,pairing,var,prefit=(records[k] for k in ('audit','diagnosis','parity','shape','support','pairing','variance','prefit'))
 if audit.get('panel_size')!=800 or any(records[k].get('sample_count')!=800 for k in ('diagnosis','parity','support','pairing','variance')):raise ValueError('complete TRAIN800 evidence required')
 if parity['amplitude']!='probe' or len(parity['per_batch'])!=13:raise ValueError('completed production-probe verification required')
 checkpoint=a.run/'checkpoints/epoch_100.pt'; config=a.run/'resolved_config.yaml'
 config_sha=hashlib.sha256(config.read_bytes()).hexdigest()
 cfg=yaml.safe_load(config.read_text()); checkpoint_data=torch.load(checkpoint,map_location='cpu',weights_only=False)
 ckpt_sha=prefit['checkpoint_sha256']
 if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=ckpt_sha:raise ValueError('mature checkpoint hash mismatch')
 if (parity['audit_sha256']!=sources['audit']['sha256'] or audit['checkpoint_sha256']!=ckpt_sha
     or parity['checkpoint_sha256']!=ckpt_sha or shape['checkpoint_sha256']!=ckpt_sha
     or audit['config_sha256']!=config_sha):raise ValueError('audit, checkpoint or config binding mismatch')
 if checkpoint_data['source_hashes']['checkpoint']!=support['source_checkpoint_sha256']:
  raise ValueError('cached support and mature model SOURCE mismatch')
 if (support['selected_panel_sha256']!=var['source_panel_sha256']
     or cfg['coherence']['scalable_anchored_coherence']['common_prefit']['sha256']!=prefit['prefit_sha256']):
  raise ValueError('panel or prefit identity mismatch')
 if diag['field_order']!=audit['fields'] or parity['field_order']!=audit['fields'] or support['field_order']!=audit['fields']:
  raise ValueError('field order mismatch')
 if diag['constraint_reference']!=checkpoint_data['external_policy']['reference']:
  raise ValueError('diagnosis constraint reference differs from checkpoint')
 if (diag['calibration_recomputed']['clipped_gamma_star']!=audit['calibration']['gamma_star']
     or any(not math.isclose(x,y,rel_tol=1e-14,abs_tol=1e-15) for x,y in zip(diag['calibration_recomputed']['b'],audit['calibration']['baseline_drift']))
     or any(not math.isclose(x,y,rel_tol=1e-14,abs_tol=1e-15) for x,y in zip(diag['calibration_recomputed']['r'],audit['calibration']['response_per_gain']))):
  raise ValueError('diagnosis differs from original measured calibration')
 history_bytes=a.history.read_bytes();history=json.loads(history_bytes)
 if history['coefficients_constant']!=diag['coefficients_constant_all_recorded_epochs'] or history['reference']!=diag['constraint_reference']:
  raise ValueError('history and diagnosis differ')
 sources['history']={'path':str(a.history),'sha256':hashlib.sha256(history_bytes).hexdigest()}
 if shape['sample_ids']!=parity['per_batch'][0]['sample_ids']:
  raise ValueError('semantic shape fixture and first mature batch mismatch')
 if [r['sample_id'] for r in support['per_condition']]!=[r['sample_id'] for r in pairing['samples']]:
  raise ValueError('SOURCE support and observation pairing sample mismatch')
 if [len(r['sample_ids']) for r in parity['per_batch']]!=audit['batch_counts']:
  raise ValueError('mature action sample-count mismatch')
 for index,row in enumerate(parity['per_batch']):
  if row['sample_ids']!=audit['probe_actions'][index]['sample_ids']:raise ValueError('mature action sample IDs differ')
 sources['mature_checkpoint']={'path':str(checkpoint),'sha256':ckpt_sha}
 sources['mature_config']={'path':str(config),'sha256':config_sha}
 original=diag['calibration_recomputed'];star=original['clipped_gamma_star'];robust=parity['production_recalculated_gain']
 if not all(math.isfinite(float(x)) for x in (star,robust)):raise ValueError('nonfinite gain')
 status='unqualified_pi_retain_source' if star==0 and robust==0 else 'requires_review_before_freeze'
 zero=diag['arms']['zero'];probe=diag['arms']['probe']
 out={'schema':'car_round6_joint_audit_decision_v1','status':'completed','decision':status,
  'controller_qualification':{'qualified':False,'gamma_star':star,'reason':'predeclared target-fit gain is zero in both diagnostic-row and verified production-probe response' if status=='unqualified_pi_retain_source' else 'production response differs; investigation required',
   'unclipped_gain':original['unclipped_gain'],'production_unclipped_gain':parity['production_unclipped_gain'],
   'production_recalculated_gain':robust,'gamma_probe':parity['gamma'],
   'zero_weighted_residual_norm':original['baseline_weighted_residual_norm'],
   'production_probe_weighted_residual_norm':parity['combined_residual_norm'],
   'production_vs_separate_response_norm':parity['combined_vs_separate_weighted_response_norm'],
   'zero_replay_discrepancy_norm':parity['repeat_replay_noise_norm'],
   'baseline_drift_over_target':original['baseline_drift_over_target_magnitude'],
   'baseline_drift':original['b'],'target_drift':original['target'],
   'production_probe_response_per_gain':parity['production_response_per_gain']},
  'source_checkpoint_sha256':checkpoint_data['source_hashes']['checkpoint'],
  'mature_checkpoint_sha256':ckpt_sha,'mature_config_sha256':config_sha,
  'field_order':diag['field_order'],'family_order':['global_distribution','multiscale_spectrum','topology'],
  'constraint_reference':diag['constraint_reference'],'source_allowance_over_scale':diag['allowance_over_scale'],
  'source_prefit_identity':{'checkpoint_prefit_equal':prefit['prefit_matches_saved_checkpoint_artifact'],
   'prefit_sha256':prefit['prefit_sha256'],'family_scales':prefit['family_scales'],
   'configured_lr':prefit['configured_lr'],'checkpoint_lr':prefit['saved_lr'],'clip_norm':prefit['configured_grad_clip'],
   'logged_coefficients_constant':diag['coefficients_constant_all_recorded_epochs']},
  'action_evidence':{'zero_family_actual_products':zero['family_products_actual_displacement'],
   'zero_field_actual_products':zero['field_products_actual_displacement'],
   'zero_finite_field_excess_changes':zero['finite_field_change'],
   'zero_fixed_query_shape_changes':zero['shape_fixed_change'],'zero_independent_query_shape_changes':zero['shape_independent_change'],
   'probe_separate_field_vjp_fixed_shape_changes':probe['shape_fixed_change'],
   'probe_separate_field_vjp_independent_shape_changes':probe['shape_independent_change']},
  'shape_components':{'scalar':shape['family_scalar'],'reconstructed_scalar':shape['reconstructed_scalar'],
    'absolute_difference':shape['absolute_reconstruction_difference'],'gradient_norm':shape['full_shape_parameter_gradient_norm'],
    'components':shape['components'],'scope':shape['scope']},
  'source_support':{'member_mse_native':support['mean']['native_member_mse'],
   'member_mse_selected':support['mean']['selected_member_mse'],
   'risk_native':support['mean']['native_mean_risk'],'risk_selected':support['mean']['selected_mean_risk'],
   'sensor_pairs_match_count':pairing['ordered_sensor_match_count'],'member_seed_list_match_count':pairing['member_seed_match_count'],
   'scope':support['limitation']},
  'source_shape_variation':{'condition_variance':var['condition_variance_of_four_draw_mean'],
   'paired_member_half_mean_square':var['member_pair_paired_difference_half_mean_square'],
   'paired_query_half_mean_square':var['query_paired_difference_half_mean_square'],'scope':var['limitation']},
  'disposition':'Retain SOURCE/C8. Do not freeze or launch the proposed PI candidate without a qualified gain.' if status=='unqualified_pi_retain_source' else 'Do not freeze; review production response discrepancy.',
  'limits':['TRAIN-only one-step mature diagnostics and fixed-SOURCE cache; no new held-out learning evaluation or TEST access.',
   'No evidence separates Adam moment carryover from preconditioning; the saved local action does not show a dominant rotation failure.',
   'State and Scale have first-order action products only; finite same- and independent-query score changes were measured for Shape.',
   'A zero target-fit gain does not establish failure of Lagrangian methods or efficacy of any alternative controller.'],
  'evidence':sources}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as stream:json.dump(out,stream,indent=2,allow_nan=False);stream.write('\n')
if __name__=='__main__':main()
