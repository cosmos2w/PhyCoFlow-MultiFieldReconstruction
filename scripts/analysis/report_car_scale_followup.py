#!/usr/bin/env python3
"""Render a completed same-parent Scale followup from immutable evidence only."""
from __future__ import annotations
import argparse,json,math,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.analysis.report_car_round5 import bound,doc,sha,save_figure,render
from scripts.analysis.report_car_round6 import repair_round6_pdf,learning,outcomes,regional,case_figures
from scripts.posttraining.assess_car_round6 import intervals,check_terminal

ARM='FFM_SSS_EpochCalibrated_PI_Repair'
FAMILIES=('state','scale','shape')
DEFINITIONS=('Published_Core_Default','Complete_Core_Prefit','Complete_Expanded_Prefit')
FIGURES={'matched_train_validation_scale':'Same complete Scale definitions on cached TRAIN800 and development VALIDATION100 endpoints; paired intervals resample conditions within each split.',
 'native_field_support_gap':'Cached native J4 signed field-risk changes on selected and complement regions, with their own physical-mass normalization; these are not selected-support model rollouts.',
 'actual_adam_complete_scale_changes':'Thirteen independently reset TRAIN J2 shadows of the actual parent AdamW action; negative means an immediate decrease and the batches are not independent trajectories.',
 'scale_components_and_definitions':'VALIDATION100 attraction and separation changes and the three complete-Q4096 Scale definitions in frozen TRAIN units; negative score changes favor repair.',
 'scale_field_pair_band_localization':'Prefit-expanded field, pair and spectral-band changes; joint and log-power norm components are additive attributions, not independently proper field scores.'}

def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def require(condition,message):
 if not condition:raise ValueError(message)
def check_adoption(role,views,endpoint):
 if role=='control':
  raise ValueError('control adoption requires a completed control endpoint task assessment and eligibility evidence')
 if role!='source':
  chosen=views[role]['candidates'][views[role]['arm']]['fidelity']
  require(chosen.get('primary_pass') is True and chosen.get('legacy_member_pass') is True,'replacement requires native field, sensor and member safeguards')
 if role=='candidate':
  require(endpoint['candidates'][endpoint['arm']].get('replacement_eligible') is True,'candidate adoption requires endpoint replacement eligibility')
def endpoint_view(review, review_hash, endpoint, endpoint_hash, decision, decision_hash):
 check_terminal(decision,dict(review,_sha256=review_hash),review['bank_receipt_sha256'])
 require(endpoint.get('review_sha256')==review_hash and endpoint.get('task_decision_sha256')==decision_hash,'endpoint review or decision differs')
 for key in ('schema','epoch','bank_receipt_sha256','checkpoint_sha256','sample_ids','field_order','source_bank_sha256','panel_sha256','contract_sha256','core_sha256','normalizers','arm'):
  require(endpoint.get(key)==review.get(key),f'endpoint differs from task-free review in {key}')
 arm=review['arm']
 for key in ('risk','fidelity','normalized_core_differences','per_condition_scores'):
  require(endpoint['candidates'][arm].get(key)==review['candidates'][arm].get(key),f'endpoint changed task-free {key}')
 require(endpoint.get('task_scope')=='reused development tasks at selected endpoint' and 'reused_tasks' in endpoint['candidates'][arm],'completed endpoint tasks required')
 return endpoint['candidates'][arm]['reused_tasks']
def series(ref,age,fields):
 history=lines(bound(ref['history']));journal=lines(bound(ref['pi_journal']))
 require([r.get('epoch') for r in history]==list(range(1,age+1)) and all(r.get('epoch_complete') is True and r.get('step')==6000+12*r['epoch'] for r in history),'complete phase history required')
 family_keys=('coherence_family/global_distribution/raw','coherence_family/multiscale_spectrum/raw','coherence_family/topology/raw')
 require(all(all(isinstance(row.get(k),(int,float)) and not isinstance(row[k],bool) and math.isfinite(row[k]) for k in family_keys) for row in history),'complete finite three-family histories required')
 require([r.get('step') for r in journal]==list(range(6001,6001+12*age)),'complete phase PI journal required')
 for row in journal:
  require(isinstance(row.get('field_violation'),dict) and set(row['field_violation'])==set(fields) and len(row.get('multipliers_before',[]))==len(fields) and len(row.get('multipliers_after',[]))==len(fields),'PI journal field mapping differs')
  require(all(math.isfinite(float(x)) for x in list(row['field_violation'].values())+row['multipliers_before']+row['multipliers_after']),'nonfinite PI journal')
 return history,journal

def phase100_review_data(reference,start,control,candidate_spec):
 review=doc(reference)
 require(review.get('schema')=='car_scale_phase100_joint_review_v1' and review.get('status')=='completed' and review.get('phase_epoch')==100 and review.get('global_step')==7200 and review.get('cumulative_reporting_epoch')==600 and review.get('test_locked') is True and review.get('decision')=='continue_to_phase500','completed matched phase100 continuation review required')
 refs=review['inputs'];require(refs['common_start_assessment']['sha256']==sha(bound(refs['common_start_assessment'])) and doc(refs['common_start_assessment'])==start,'phase100 common start differs')
 require(doc(refs['control_assessment'])==control,'phase100 control differs')
 assessments={};identities={};specs={}
 for role,variant in (('control','FFM_SSS_Unchanged_Scale_Continuation'),('candidate','FFM_SSS_RotatingComplement_FieldFeedback')):
  a=doc(refs[role+'_assessment']);i=doc(refs[role+'_fork_identity']);s=doc(i['spec']);b=doc(i['bank_receipt']);parent=doc(s['parent_snapshot'])
  for key in ('checkpoint','run_config','source_bank','panel'):bound(s[key])
  bound(parent['checkpoint'])
  require(i.get('schema')=='car_scale_fork_assessment_identity_v1' and i.get('variant')==variant and i.get('phase_epoch')==100 and i.get('phase_origin_step')==6000 and i.get('global_step')==7200 and i.get('test_locked') is True,'phase100 fork identity differs')
  require(i['assessment']['sha256']==refs[role+'_assessment']['sha256'] and a.get('epoch')==100 and a.get('bank_receipt_sha256')==i['bank_receipt']['sha256'] and a.get('checkpoint_sha256')==s['checkpoint']['sha256'] and parent['checkpoint']['sha256']==start['checkpoint_sha256'],'phase100 checkpoint or parent differs')
  require(s.get('variant')==variant and s.get('phase_epoch')==100 and s.get('global_step')==7200 and s['parent_snapshot']['sha256']==candidate_spec['parent_snapshot']['sha256'],'phase100 fork or common parent differs')
  require(b.get('status')=='completed' and b.get('epoch')==100 and b.get('global_step')==7200 and b.get('phase_origin_step')==6000 and b.get('checkpoint_sha256')==s['checkpoint']['sha256'] and b.get('members')==4 and b.get('sample_ids')==start['sample_ids'] and b.get('source_bank_sha256')==start['source_bank_sha256'] and b.get('panel_sha256')==start['panel_sha256'],'phase100 cached bank or coupling differs')
  for key in ('sample_ids','field_order','source_bank_sha256','source_checkpoint_sha256','panel_sha256','contract_sha256','core_sha256','source_baseline_sha256','normalizers'):
   require(a[key]==start[key],f'phase100 {role} differs in {key}')
  fidelity=a['candidates'][a['arm']]['fidelity']
  require(review[role]['fidelity']==fidelity and all(math.isclose(review[role]['field_change_over_allowance'][j],fidelity['field_difference'][j]/fidelity['field_allowance'][j],rel_tol=1e-12,abs_tol=1e-12) for j in range(len(start['field_order']))) and all(review[role]['source_relative_core'][f][k]==a['paired_core'][f][k] for f in FAMILIES for k in ('mean','interval_95')),'phase100 SOURCE-relative results differ')
  assessments[role]=a;identities[role]=i;specs[role]=s
 require(specs['candidate']['run']==candidate_spec['run'],'phase100 and final candidate must be the same fork')
 training=doc(refs['training_result']);require(training.get('status')=='completed_phase' and training.get('phase_target_epoch')==100 and training.get('phase_origin_step')==6000 and training.get('global_step')==7200 and training.get('run')==specs['candidate']['run'] and training['complete_epoch_evidence']['checkpoint']['sha256']==specs['candidate']['checkpoint']['sha256'],'phase100 completed training identity differs')
 for key,left in (('compare_common_start',refs['common_start_assessment']),('compare_control',refs['control_assessment'])):
  comparison=doc(refs[key]);embedded=review['candidate_minus_common_start' if key=='compare_common_start' else 'candidate_minus_control']
  require(comparison['left']['sha256']==left['sha256'] and comparison['right']['sha256']==refs['candidate_assessment']['sha256'] and all(comparison['core_in_frozen_train_units'][f][k]==embedded['core_in_frozen_train_units'][f][k] for f in FAMILIES for k in ('mean','interval_95')),'phase100 paired comparison differs')
 aligned=doc(refs['aligned_scale']);spec=doc(aligned['spec']);comparisons=doc(refs['aligned_comparisons'])
 require(aligned.get('status')=='completed' and aligned.get('split')=='validation' and aligned.get('members')==4 and aligned.get('sample_ids')==start['sample_ids'] and aligned.get('field_order')==start['field_order'] and spec['source_receipt']['sha256']==start['source_bank_sha256'] and spec['panel']['sha256']==start['panel_sha256'] and comparisons['decomposition']['sha256']==refs['aligned_scale']['sha256'] and comparisons.get('sample_count')==100,'phase100 aligned Scale coupling differs')
 for role,variant in (('control','FFM_SSS_Unchanged_Scale_Continuation'),('candidate','FFM_SSS_RotatingComplement_FieldFeedback')):
  require(spec['children'][variant]['receipt']['sha256']==identities[role]['bank_receipt']['sha256'] and spec['children'][variant]['assessment']['sha256']==refs[role+'_assessment']['sha256'],'phase100 aligned Scale child differs')
 require(spec['children']['original_epoch500']['receipt']['sha256']==start['bank_receipt_sha256'],'phase100 aligned common start differs')
 for key,values in review['complete_scale_comparisons']['comparisons'].items():
  for definition in DEFINITIONS:
   require(values[definition]['mean']==comparisons['comparisons'][key][definition]['mean'] and values[definition]['interval_95']==comparisons['comparisons'][key][definition]['interval_95'],'phase100 aligned Scale comparison differs')
 return review

def phase100_review_text(review,fields):
 def interval(x):return f"{x['mean']:+.4f} [{x['interval_95'][0]:+.4f}, {x['interval_95'][1]:+.4f}]"
 out=['## Matched phase-100 continuation review','', 'Both forks had 100 additional epochs and the same original epoch-500 parent. Changes are candidate minus the stated baseline in frozen TRAIN units; intervals are paired condition bootstraps on the reused development panel, with negative favoring the candidate.','', '| Baseline | State | Scale | Shape |','| --- | ---: | ---: | ---: |']
 for label,values in (('SOURCE',review['candidate']['source_relative_core']),('Common start',review['candidate_minus_common_start']['core_in_frozen_train_units']),('Same-age control',review['candidate_minus_control']['core_in_frozen_train_units'])):
  out.append('| '+label+' | '+' | '.join(interval(values[k]) for k in FAMILIES)+' |')
 out+=['','| Candidate contrast | Published core | Complete prefit core | Complete prefit expanded |','| --- | ---: | ---: | ---: |']
 variants=('SOURCE','original_epoch500','FFM_SSS_Unchanged_Scale_Continuation');comparisons=review['complete_scale_comparisons']['comparisons']
 for label,base in zip(('SOURCE','Common start','Same-age control'),variants):
  values=comparisons['FFM_SSS_RotatingComplement_FieldFeedback_minus_'+base];out.append('| '+label+' | '+' | '.join(interval(values[k]) for k in DEFINITIONS)+' |')
 out+=['','| Field | Candidate signed-risk change / allowance | Same-age control signed-risk change / allowance |','| --- | ---: | ---: |']
 for i,field in enumerate(fields):out.append(f"| {field} | {review['candidate']['field_change_over_allowance'][i]:+.3f} | {review['control']['field_change_over_allowance'][i]:+.3f} |")
 out+=['',review['scientific_progress'],'',review['coherence_tradeoff'],'',review['continuation_question'],'',review['limitations']]
 return out

def place_phase100_section(body,review,fields,phase):
 heading='## Matched learning outcome'
 position=body.index(heading)
 if review:body[position:position]=phase100_review_text(review,fields)+['']
 if phase==500:body[body.index(heading)]='## Selected phase-500 outcome and unequal-exposure control'

def validate(inputs):
 require(inputs.get('schema')=='car_scale_followup_report_inputs_v1' and inputs.get('phase') in (100,500),'completed selected phase manifest required')
 phase=inputs['phase'];diagnostic=doc(inputs['diagnostic_readout']);decision=doc(inputs['decision']);cost=doc(inputs['costs']);start=doc(inputs['common_start_assessment']);aligned=doc(inputs['aligned_scale'])
 require(phase!=500 or inputs.get('phase100_review'),'phase500 report requires the matched phase100 review')
 require(diagnostic.get('schema')=='car_scale_followup_diagnostic_readout_v1' and diagnostic.get('status')=='completed' and diagnostic.get('test_locked') is True,'completed diagnostic readout required')
 for r in diagnostic['inputs'].values():bound(r)
 require(decision.get('schema')=='car_scale_followup_disposition_v1' and decision.get('status')=='completed' and decision.get('phase')==phase and decision.get('test_locked') is True and all(isinstance(decision.get(k),str) and decision[k].strip() for k in ('recommendation','reason','scientific_progress','engineering_decision','material_limitations')),'explicit completed disposition required')
 localization=doc(inputs['diagnostic_localization'])
 require(inputs['diagnostic_localization']['sha256']=='96b5b21e3211b72c6570c629a7643adcd67c34e236334c4fb9a88f24952f694d' and localization['status']=='completed' and localization['normalizer']==diagnostic['frozen_train_scale_unit'] and localization['epoch500_vs_source']==diagnostic['scale']['validation']['epoch500'],'completed matched localization evidence required')
 for split in ('train','validation'):
  query=doc(inputs['diagnostic_query_'+split]);decomposition=doc(diagnostic['inputs'][split])
  require(query.get('schema')=='car_scale_cached_grouped_query_estimator_v1' and query.get('status')=='completed' and query.get('split')==split and query.get('members')==4 and query.get('groups')==8 and len(query.get('query_seeds',[]))==4 and query['input']['sha256']==diagnostic['inputs'][split]['sha256'] and query['sample_ids']==decomposition['sample_ids'] and query['original_frozen_train_scale_normalizer']==diagnostic['frozen_train_scale_unit'],'grouped query evidence or panel differs')
 require(cost.get('status')=='completed' and isinstance(cost.get('summary'),str) and cost['summary'].strip() and isinstance(cost.get('measurements'),dict) and cost['measurements'],'completed actual cost evidence required')
 require(all(v is None or (isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)) for v in cost['measurements'].values()),'invalid measured cost')
 if inputs.get('reference_scope'):
  scope=doc(inputs['reference_scope']);require(scope.get('schema')=='car_scale_source_quadrature_reference_scope_v1' and scope.get('status')=='completed' and scope.get('test_locked') is True,'completed SOURCE-reference scope evidence required')
 require(aligned.get('schema')=='car_scale_cached_decomposition_v1' and aligned.get('status')=='completed' and aligned.get('split')=='validation' and aligned.get('members')==4,'completed aligned VALIDATION Scale required')
 spec=doc(aligned['spec']);require(spec.get('split')=='validation','Scale input split differs')
 all_views={'start':start};states={};parents=[]
 for role in ('control','candidate'):
  arm=inputs[role];assessment=doc(arm['assessment']);identity=doc(arm['fork_identity']);fork_spec=doc(identity['spec']);parent=doc(fork_spec['parent_snapshot']);age=100 if role=='control' else phase
  require(identity.get('schema')=='car_scale_fork_assessment_identity_v1' and identity['assessment']['sha256']==arm['assessment']['sha256'] and identity.get('phase_epoch')==age and identity.get('phase_origin_step')==6000 and identity.get('global_step')==6000+12*age and identity.get('test_locked') is True,'fork assessment identity differs')
  expected_variant={'control':'FFM_SSS_Unchanged_Scale_Continuation','candidate':'FFM_SSS_RotatingComplement_FieldFeedback'}[role]
  require(identity.get('variant')==expected_variant and fork_spec.get('variant')==expected_variant and fork_spec.get('phase_epoch')==age and fork_spec.get('global_step')==6000+12*age,'fork variant or phase differs')
  bound(identity['assessment']);bound(identity['bank_receipt']);bound(fork_spec['checkpoint']);bound(fork_spec['run_config']);bound(parent['checkpoint'])
  require(assessment.get('epoch')==age and assessment.get('checkpoint_sha256')==fork_spec['checkpoint']['sha256'] and assessment.get('bank_receipt_sha256')==identity['bank_receipt']['sha256'] and parent['checkpoint']['sha256']==start['checkpoint_sha256'],'fork checkpoint or common parent differs')
  checked_parent=bound(fork_spec['parent_snapshot']);del checked_parent
  require(decision.get(role+'_assessment_sha256')==arm['assessment']['sha256'],'disposition does not bind both assessments')
  all_views[role]=assessment;parents.append(fork_spec['parent_snapshot']['sha256']);states[role]=series(arm,age,start['field_order'])
 require(parents[0]==parents[1],'forks must share the identical original parent snapshot')
 if inputs.get('phase100_review'):phase100_review_data(inputs['phase100_review'],start,all_views['control'],fork_spec)
 require(decision.get('aligned_scale_sha256')==inputs['aligned_scale']['sha256'] and decision.get('common_start_assessment_sha256')==inputs['common_start_assessment']['sha256'],'disposition evidence differs')
 for role,a in all_views.items():
  require(a.get('schema')=='car_round6_matched_validation_v1' and a.get('split')=='validation' and a.get('population')==1000 and a.get('members')==4 and len(a.get('sample_ids',[]))==100,'matched J4 VALIDATION100 required')
  for k in ('sample_ids','field_order','source_bank_sha256','source_checkpoint_sha256','panel_sha256','contract_sha256','core_sha256','source_baseline_sha256','normalizers'):
   require(a[k]==start[k],f'{role} differs in {k}')
  require(a.get('arm') in a.get('candidates',{}),'raw candidate missing')
 require(start['epoch']==500,'epoch 500 common start required')
 endpoint=doc(inputs['candidate_endpoint_assessment']);terminal=doc(inputs['endpoint_decision'])
 endpoint_view(all_views['candidate'],inputs['candidate']['assessment']['sha256'],endpoint,inputs['candidate_endpoint_assessment']['sha256'],terminal,inputs['endpoint_decision']['sha256'])
 require(decision.get('candidate_endpoint_assessment_sha256')==inputs['candidate_endpoint_assessment']['sha256'] and decision.get('endpoint_decision_sha256')==inputs['endpoint_decision']['sha256'],'disposition endpoint identity differs')
 source=doc(inputs['source_bank']);bank=doc(inputs['candidate_bank']);selection=doc(inputs['visual_selection']);event=doc(inputs['event']);coefficients=doc(inputs['coefficients'])['coefficients']
 original=doc(inputs['original_report_inputs']);freeze=doc(spec['formal_freeze'])
 require(inputs['original_report_inputs']['sha256']=='f20a7266f11c6a8ee3ab3bb82904ea05572ce69c3ce1c2ba7529960a1c78512b' and Path(inputs['original_report_inputs']['path']).resolve().parent==Path(spec['formal_freeze']['path']).resolve().parent and original['visual_selection']['sha256']==inputs['visual_selection']['sha256'] and original['coefficients']['sha256']==inputs['coefficients']['sha256'],'original frozen visual selection or coefficients differ')
 require(inputs['source_bank']['sha256']==start['source_bank_sha256'] and inputs['candidate_bank']['sha256']==all_views['candidate']['bank_receipt_sha256'] and source['sample_ids']==start['sample_ids'] and bank['sample_ids']==start['sample_ids'] and bank['source_bank_sha256']==inputs['source_bank']['sha256'] and bank['checkpoint_sha256']==all_views['candidate']['checkpoint_sha256'] and bank['epoch']==phase and bank['arm']==all_views['candidate']['arm'],'cached example bank identity differs')
 require(selection['source_bank_sha256']==inputs['source_bank']['sha256'] and selection.get('candidates_used') is False and selection['event']['sha256']==inputs['event']['sha256'] and selection['panel']['sha256']==start['panel_sha256'] and selection['created_unix']<freeze['created_unix'] and len(selection['cases'])==3 and all(isinstance(c['ordinal'],int) and 0<=c['ordinal']<len(start['sample_ids']) and start['sample_ids'][c['ordinal']]==c['sample_id'] for c in selection['cases']),'original SOURCE-only example selection differs')
 require(isinstance(bank.get('start_unix'),(int,float)) and selection['created_unix']<bank['start_unix'],'candidate generation must follow frozen visual selection')
 require(bool(event) and bool(coefficients) and all(isinstance(v,(int,float)) and math.isfinite(v) for v in coefficients.values()),'event and finite fixed coefficients required')
 require(decision.get('adopted_role') in ('source','control','candidate'),'explicit adopted role required')
 expected_recommendation={'source':'retain SOURCE/C8','control':'FFM_SSS_Unchanged_Scale_Continuation','candidate':'FFM_SSS_RotatingComplement_FieldFeedback'}
 require(decision['recommendation']==expected_recommendation[decision['adopted_role']],'recommendation and adopted role differ')
 check_adoption(decision['adopted_role'],all_views,endpoint)
 require(aligned['sample_ids']==start['sample_ids'] and aligned['field_order']==start['field_order'] and [r['sample_id'] for r in aligned['records']]==start['sample_ids'],'aligned panel differs')
 require(spec['source_receipt']['sha256']==start['source_bank_sha256'] and spec['panel']['sha256']==start['panel_sha256'],'aligned SOURCE or panel differs')
 for role in ('start','control','candidate'):
  key=inputs['aligned_model_keys'][role]; require(all(key in r['models'] and 'SOURCE' in r['models'] for r in aligned['records']) and key in spec['children'],f'aligned {role} missing')
  require(spec['children'][key]['receipt']['sha256']==all_views[role]['bank_receipt_sha256'],f'aligned {role} bank differs')
 require(set(inputs['diagnostic_figures'])==set(FIGURES),'five declared diagnostic figure pairs required')
 for name,refs in inputs['diagnostic_figures'].items():
  require(bound(refs['png']).suffix=='.png' and bound(refs['pdf']).suffix=='.pdf',f'invalid {name} figure')
 return diagnostic,decision,cost,all_views,states,aligned

def relative_score(a,b):
 norm=a['normalizers'];left=a['candidates'][a['arm']]['per_condition_scores'];right=b['candidates'][b['arm']]['per_condition_scores']
 return {k:intervals([r[k] for r in left],[r[k] for r in right],scale=norm[k]) for k in FAMILIES}

def comparison_figures(views,states,aligned,keys,directory):
 import numpy as np
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
 names=('start','control','candidate');labels=('Original epoch 500','Control +100',f"Candidate +{views['candidate']['epoch']}");x=np.arange(3);width=.24
 for i,(name,label) in enumerate(zip(names,labels)):
  a=views[name];candidate=a['candidates'][a['arm']];y=[candidate['normalized_core_differences'][k] for k in FAMILIES];axes[0].bar(x+(i-1)*width,y,width,label=label)
  dif=np.array(candidate['fidelity']['field_difference'])/np.array(candidate['fidelity']['field_allowance']);axes[1].plot(np.arange(len(dif)),dif,marker='o',label=label)
  key=keys[name];diff=np.array([[row['models'][key]['definitions'][k]-row['models']['SOURCE']['definitions'][k] for k in DEFINITIONS] for row in aligned['records']]).mean(0)/aligned_normalizer(aligned)
  axes[2].bar(x+(i-1)*width,diff,width,label=label)
 axes[0].set_xticks(x,['State','Scale','Shape']);axes[0].set_ylabel('Change / frozen TRAIN unit');axes[0].set_title('Matched core vs SOURCE')
 axes[1].set_xticks(range(len(views['start']['field_order'])),views['start']['field_order']);axes[1].axhline(1,color='black',ls='--',lw=.7);axes[1].set_ylabel('Signed-risk change / allowance');axes[1].set_title('Native field changes vs SOURCE')
 axes[2].set_xticks(x,['Published','Prefit core','Prefit expanded'],rotation=15);axes[2].set_ylabel('Change / frozen TRAIN Scale unit');axes[2].set_title('Three aligned Scale definitions')
 for ax in axes:ax.axhline(0,color='black',lw=.6);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=7)
 save_figure(fig,directory,'matched_phase_comparison')
 fig,axes=plt.subplots(5,1,figsize=(11,12),sharex=True,constrained_layout=True)
 family_keys=('coherence_family/global_distribution/raw','coherence_family/multiscale_spectrum/raw','coherence_family/topology/raw')
 colors=plt.get_cmap('tab10').colors
 for role,style in (('control','--'),('candidate','-')):
  history,journal=states[role];epoch=np.asarray([r['epoch'] for r in history])
  for i,key in enumerate(family_keys):axes[i].plot(epoch,[r[key] for r in history],linestyle=style,label=role)
  multipliers=np.array([r['multipliers_after'] for r in journal]).reshape(-1,12,len(views['start']['field_order'])).mean(1)
  for i,field in enumerate(views['start']['field_order']):
   vals=np.array([r['field_violation'][field] for r in journal]).reshape(-1,12).mean(1)
   axes[3].plot(np.arange(1,len(vals)+1),vals,color=colors[i],linestyle=style,label=role+' '+field)
   axes[4].plot(np.arange(1,len(multipliers)+1),multipliers[:,i],color=colors[i],linestyle=style,label=role+' '+field)
 for ax,title in zip(axes,('State sampled raw family','Scale sampled raw family','Shape sampled raw family','Scaled field violations, epoch mean','PI multipliers, epoch mean')):ax.set_ylabel(title);ax.grid(alpha=.2);ax.legend(fontsize=6,ncol=5)
 axes[-1].set_xlabel('Phase reporting epoch')
 save_figure(fig,directory,'phase_learning_and_pi')

def aligned_normalizer(aligned):
 spec=doc(aligned['spec']);value=float(spec['frozen_train_scale_normalizer']);require(math.isfinite(value) and value>0,'frozen Scale unit required');return value

def diagnostic_text(d):
 unit=float(d['frozen_train_scale_unit']);a=d['scale']['train']['epoch500']['Complete_Expanded_Prefit']['mean_change_over_original_frozen_train_scale'];b=d['scale']['validation']['epoch500']['Complete_Expanded_Prefit']['mean_change_over_original_frozen_train_scale']
 return f'The diagnostic comparison used the same complete expanded prefit definition and frozen TRAIN unit {unit:.8g}: epoch-500 change versus SOURCE was {a:+.5f} on TRAIN800 and {b:+.5f} on reused VALIDATION100. The fixed-parent J2 Adam shadow lowered both sampled and complete Scale in all 13 paired batches; these are local actions, not independent learning trajectories.'

def grouped_query_text(train,validation,unit):
 parts=[]
 for label,query in (('TRAIN800',train),('VALIDATION100',validation)):
  summaries=query['summary']['epoch500_minus_SOURCE']; signs=[]
  for definition in DEFINITIONS:
   values=summaries[definition]['grouped_paired_changes_by_seed']
   require(len(values)==4 and all(math.isfinite(x) for x in values),'four finite grouped query changes required')
   signs.append(definition+': '+('all negative' if all(x<0 for x in values) else 'all positive' if all(x>0 for x in values) else 'mixed or zero'))
  values=summaries['Complete_Expanded_Prefit']['grouped_paired_changes_by_seed']
  parts.append(f"{label}: "+', '.join(signs)+f"; expanded range [{min(values)/unit:+.4f}, {max(values)/unit:+.4f}]")
 return 'Four paired grouped-query seed replays of epoch500 minus SOURCE, in frozen TRAIN Scale units: '+ '; '.join(parts)+'. These use fixed cached J4/T256 endpoints; the actual-Adam shadow uses J2 and the variable-observation TRAIN law. Query seeds measure estimator variation on the same conditions, not independent datasets or a causal generalization test.'

def detailed_diagnostic(d,localization,adam):
 unit=d['frozen_train_scale_unit'];rows=[]
 for split,label in (('train','TRAIN800'),('validation','VALIDATION100')):
  for epoch in ('epoch100','epoch500'):
   for definition in DEFINITIONS:
    item=d['scale'][split][epoch][definition];lo,hi=(v/unit for v in item['interval_95_raw'])
    rows.append(f"| {label} | {epoch[5:]} | {definition} | {item['mean_change_over_original_frozen_train_scale']:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
 block=localization['prefit_expanded_block_weighted_changes'];att=localization['weighted_total_attraction_change'];sep=localization['weighted_total_separation_change']
 def formatted(v):return f"{v['mean']:+.4f} [{v['interval_95'][0]:+.4f}, {v['interval_95'][1]:+.4f}]"
 changes=d['actual_adam_finite_changes'];labels=('same_query_train_scale','independent_query_train_scale',*DEFINITIONS)
 weights=[row['sample_count'] for row in adam['results']];total=sum(weights)
 field_delta=[sum(w*(r['field_risk_post'][i]-r['field_risk_pre'][i]) for w,r in zip(weights,adam['results']))/total for i in range(5)]
 field_decreases=[sum(r['field_risk_post'][i]<r['field_risk_pre'][i] for r in adam['results']) for i in range(5)]
 gap=d['spatial_feedback_gap']; fields=gap['train']['field_order']
 gaptext=lambda split:', '.join(f"{f} {v:+.3f} [{lo:+.3f}, {hi:+.3f}]" for f,v,lo,hi in zip(fields,gap[split]['paired_complement_minus_selected_change_over_allowance'],*gap[split]['paired_condition_bootstrap_interval_95']))
 return [
 'The fair score is A − R, where A is the mean declared distance from each of J members to the reference and R is the sum of unordered member-pair distances divided by J(J−1). R is positive separation and is subtracted. Scale definitions on the same frozen Q4096 support are: published core = signed + 0.25 joint + log-power + correlation; complete prefit core = 2 signed + 0.3208755755 joint + 0.5 log-power + 0.8190801090 correlation; complete expanded prefit adds 0.5 times the mean of eight frozen triple queries. Prefit coefficients are shown rounded. Complete means the declared query population on Q4096, not all 40,300 native points.',
 '',f'Paired changes versus SOURCE and 95% condition-bootstrap intervals are in the original frozen TRAIN Scale unit {unit:.8g}; negative is better.', '', '| Split | Epoch | Scale definition | Change [95% interval] |','| --- | ---: | --- | ---: |',*rows,'',
 'VALIDATION500 expanded prefit block changes in frozen units: '+', '.join(f"{name} {formatted(block[key]['score'])}" for name,key in (('joint','joint'),('signed','signed'),('log-power','log_power_self'),('correlation','rho'),('triples','third_order')))+'. Total attraction changed '+formatted(att)+' and separation changed '+formatted(sep)+'; the score subtracts separation. Log-power attraction and separation changed '+formatted(block['log_power_self']['attraction'])+' and '+formatted(block['log_power_self']['separation'])+', respectively.',
 '',f"The weighted CO field attribution is {formatted(localization['weighted_field_attributions']['CO'])}, including band 3 {localization['field_band_attribution'][1][2]:+.4f} and piece means {localization['field_band_attribution'][1][3]:+.4f}; CO–U1 pair attribution is {formatted(localization['weighted_pair_attributions']['CO–U_1'])}. Norm-based field and band allocations are additive attributions, not independent proper scores or causal effects.",
 '', 'Actual parent AdamW finite Scale changes (sample-weighted raw units, 13 independently reset TRAIN J2 batches): '+', '.join(f"{key} {changes[key]['sample_weighted_raw_change']:+.5g} ({changes[key]['negative_batches']}/13 negative)" for key in labels)+'. Simultaneous signed field-risk changes in model units squared: '+', '.join(f'{f} {v:+.5g} ({n}/13 decreases)' for f,v,n in zip(fields,field_delta,field_decreases))+'. These are local shadow updates, not independent learning trajectories.',
 '', 'The epoch500-minus-SOURCE complement-minus-selected native cached field-risk change, divided by the original field allowances, is '+gaptext('train')+' on TRAIN800 and '+gaptext('validation')+' on VALIDATION100 (paired 95% condition-bootstrap intervals). The held-out gaps are uncertain and do not establish a spatial generalization mechanism; the measured TRAIN gap motivated testing complement feedback. These cached native-region comparisons are not identical to the selected-support training rollout.'
 ]

def produce(inputs_path,campaign,output):
 inputs=json.loads(inputs_path.read_text());diagnostic,decision,cost,views,states,aligned=validate(inputs)
 output=output.resolve();campaign=campaign.resolve(strict=True)
 require(campaign/'reports' in output.parents and not output.exists(),'new campaign-owned report directory required')
 started=time.time();output.mkdir(parents=True);figures=output/'figures';figures.mkdir();captions=dict(FIGURES)
 for name,refs in inputs['diagnostic_figures'].items():
  for ext in ('png','pdf'):shutil.copyfile(bound(refs[ext]),figures/f'{name}.{ext}')
 comparison_figures(views,states,aligned,inputs['aligned_model_keys'],figures)
 # Reuse the original scientific renderers with phase-relative journal steps.
 coefficients=doc(inputs['coefficients'])['coefficients']
 for role in ('control','candidate'):
  temporary=figures/('_'+role);temporary.mkdir()
  history,journal=states[role]
  adjusted=[dict(row,step=row['step']-6000) for row in journal]
  leaves=learning(history,adjusted,views['start']['field_order'],coefficients,temporary)
  for name in ['learning_and_action','family_history','fixed_coefficients',*leaves]:
   renamed=role+'_'+name
   for ext in ('png','pdf'):shutil.move(temporary/f'{name}.{ext}',figures/f'{renamed}.{ext}')
   captions[renamed]=f'{role.capitalize()} full phase {name.replace("_"," ")}; sampled losses and actual-displacement diagnostics retain their own units and do not define a scalar ConFIG loss.'
  temporary.rmdir()
 endpoint=doc(inputs['candidate_endpoint_assessment']);outcomes(endpoint,figures);regional(endpoint,figures)
 captions['field_core_task']='Native field accuracy and spread, and paired State, Scale, Shape and task uncertainty on reused VALIDATION100; negative score changes favor the candidate.'
 captions['regional_and_event']='Matched regional covariance and event fair-CRPS scatter on reused development VALIDATION100.'
 from scripts.posttraining.freeze_car_event import FrozenRedesignedEvent
 captions.update(case_figures(endpoint,doc(inputs['visual_selection']),bound(inputs['source_bank']),doc(inputs['source_bank']),bound(inputs['candidate_bank']),doc(inputs['candidate_bank']),FrozenRedesignedEvent(doc(inputs['event'])),figures))
 captions.update(matched_phase_comparison='Matched development VALIDATION100 comparisons against SOURCE: original core changes, native signed field changes relative to fixed allowances and three identical Scale definitions. Dashed line denotes the field allowance.',phase_learning_and_pi='Phase-indexed sampled family histories and per-action PI journal values averaged within each 12-action epoch; raw family units differ and no scalar objective is implied.')
 phase=inputs['phase'];candidate=views['candidate']['candidates'][views['candidate']['arm']];task=doc(inputs['candidate_endpoint_assessment'])['candidates'][views['candidate']['arm']]['reused_tasks'];pair=relative_score(views['control'],views['candidate']);start_pair=relative_score(views['start'],views['candidate'])
 body=['# Scale alignment and rotating field feedback','',f"**Decision:** {decision['recommendation']}. {decision['reason']}",'',f"**Scientific progress:** {decision['scientific_progress']}",'',f"Both arms resume the identical original epoch-500 checkpoint with its actual AdamW, PI and random-stream state. The selected candidate is assessed after {phase} additional phase epochs, at cumulative reporting age {500+phase}; the unchanged control receives 100 additional epochs, at cumulative age 600. "+('Both arms have the same learning age.' if phase==100 else 'The control is younger; this is an unequal-exposure comparison.'),'', 'These are 100 reused development VALIDATION conditions with four paired members. VALIDATION informed selection and is not independent confirmation. TEST remains locked.','', '## Diagnostic evidence','',diagnostic_text(diagnostic),'', '## Matched learning outcome','', 'Candidate minus control core changes in frozen TRAIN units: '+', '.join(f"{k} {pair[k]['mean']:+.5g} (paired 95% interval [{pair[k]['interval_95'][0]:+.5g}, {pair[k]['interval_95'][1]:+.5g}])" for k in FAMILIES)+'.','',f"Candidate regional normalized improvement {task['regional_normalized_improvement']:+.5g}; event normalized score change {task['event_normalized_difference']:+.5g}. Positive regional improvement and negative event change favor the candidate.",'', '| Field | Candidate signed-risk change / allowance | Control signed-risk change / allowance |','| --- | ---: | ---: |']
 position=body.index('## Matched learning outcome');body[position:position]=detailed_diagnostic(diagnostic,doc(inputs['diagnostic_localization']),doc(diagnostic['inputs']['adam']))+['']
 position=body.index('## Matched learning outcome');body[position:position]=[grouped_query_text(doc(inputs['diagnostic_query_train']),doc(inputs['diagnostic_query_validation']),diagnostic['frozen_train_scale_unit']),'']
 position=body.index('## Matched learning outcome');body[position:position]=['The learning experiment retains the original internal Scale coefficients and tests field-feedback coverage. A uniform positive Scale-family multiplier leaves its normalized ConFIG gradient row unchanged, while the assigned direction length and its ratio to field feedback can change; an outer weight increase alone does not isolate the observed transfer problem.','']
 position=body.index('## Diagnostic evidence')
 body[position:position]=['The intervention retains fixed Q4096 SSS and adds a separate paired-RFF 512-point complement only for Lagrangian field risk, with stratified inclusion weights against native physical mass. Original R0, allowances, scales, C0, Adam and PI are retained. Each update uses fresh three-family ConFIG and Lagrangian feedback; no native RF training objective is introduced.','', 'For each condition and action, the additional 512 points are sampled uniformly without replacement from the 36,204 native points outside the fixed support S. With C denoting that complement, C_t its sampled subset and M the complete native physical mass, the estimator is `Rhat_f = [sum_S w_i l_if + (|C|/512) sum_Ct w_i l_if]/M`. For independent J2 members, `l_if = (Y1_if - y_if)(Y2_if - y_if)`. The original observation tensors and per-member RFF coefficients are shared by the two separate rollouts. This inclusion-weighted integral is sampling-unbiased for frozen pointwise endpoints; it does not imply an unbiased AdamW action or an equivalent learning trajectory.','']
 position=body.index('## Matched learning outcome')
 for baseline in ('control','start'):
  comparisons=[]
  for definition in DEFINITIONS:
   left=[r['models'][inputs['aligned_model_keys'][baseline]]['definitions'][definition] for r in aligned['records']]
   right=[r['models'][inputs['aligned_model_keys']['candidate']]['definitions'][definition] for r in aligned['records']]
   q=intervals(left,right,scale=aligned_normalizer(aligned));comparisons.append(f"{definition}: {q['mean']:+.5g} [{q['interval_95'][0]:+.5g}, {q['interval_95'][1]:+.5g}]")
  body[position:position]=[f"Aligned Scale candidate minus {baseline}, frozen TRAIN units, paired 95% intervals: "+'; '.join(comparisons)+'.',''];position+=2
 if inputs.get('reference_scope'):
  position=body.index('## Matched learning outcome')
  body[position:position]=['The unchanged R0 was fitted using selected-support TRAIN800 J2 with training-observation draws; it is not an exact native-support SOURCE risk. Cached native J4/T256 SOURCE and the original R0 use different draw and observation laws, so their difference is descriptive and unpaired. Matched native field assessment uses its own paired J4 SOURCE bank.','']
 place_phase100_section(body,doc(inputs['phase100_review']) if inputs.get('phase100_review') else None,views['start']['field_order'],phase)
 insertion=body.index('| Field | Candidate signed-risk change / allowance | Control signed-risk change / allowance |')
 body[insertion:insertion]=['Candidate minus common-start core changes in frozen TRAIN units: '+', '.join(f"{k} {start_pair[k]['mean']:+.5g} (paired 95% interval [{start_pair[k]['interval_95'][0]:+.5g}, {start_pair[k]['interval_95'][1]:+.5g}])" for k in FAMILIES)+'.','']
 for i,field in enumerate(views['start']['field_order']):
  c=candidate['fidelity'];u=views['control']['candidates'][views['control']['arm']]['fidelity'];body.append(f"| {field} | {c['field_difference'][i]/c['field_allowance'][i]:+.5g} | {u['field_difference'][i]/u['field_allowance'][i]:+.5g} |")
 body+=['','Field metrics use the native physical mask, and MSE is in each field\'s squared physical units. Signed-risk intervals are paired candidate minus SOURCE and divided by the fixed field allowance.','', '| Field | SOURCE member MSE | Candidate member MSE | SOURCE ensemble-mean MSE | Candidate ensemble-mean MSE | Signed-risk delta / allowance (95% interval) |','| --- | ---: | ---: | ---: | ---: | ---: |']
 for i,field in enumerate(views['start']['field_order']):
  source=views['candidate']['source']['risk'];risk=candidate['risk'];paired=views['candidate']['paired_fields']['signed_mean_risk'];allow=candidate['fidelity']['field_allowance'][i]
  body.append(f"| {field} | {source['member_mse'][i]:.5g} | {risk['member_mse'][i]:.5g} | {source['ensemble_mean_mse'][i]:.5g} | {risk['ensemble_mean_mse'][i]:.5g} | {paired['mean'][i]/allow:+.5g} [{paired['interval_95'][0][i]/allow:+.5g}, {paired['interval_95'][1][i]/allow:+.5g}] |")
 body+=['','## Engineering and cost','',decision['engineering_decision'],'',cost['summary'],'',f"**Material limitations:** {decision['material_limitations']}",'', '| Recorded measurement | Value |','| --- | ---: |']
 for key,value in cost['measurements'].items():
  require(value is None or (isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)),'invalid measured cost')
  body.append(f"| {key} | {'unavailable' if value is None else f'{value:.6g}'} |")
 body+=['','The new follow-up has its own 30 occupied GPU-hour ceiling; the original Round 6 cost remains separately recorded. The experiment changes only the candidate field-feedback quadrature, so the same-parent comparison evaluates this package under the declared compute and sampling conditions.','', '## Figure index','']+[f'- [Figure {name.replace("_"," ")}](#figure-{name.replace("_","-")})' for name in captions]
 for name,caption in captions.items():body += ['',f'## Figure {name.replace("_"," ")}','',f'![{caption}](figures/{name}.png)','',f'{caption} [PDF master](figures/{name}.pdf)']
 (output/'report.md').write_text('\n'.join(body)+'\n');pages=render(output,captions);repair_round6_pdf(output,captions)
 # Recreate preview pages after the relative-link-only PDF repair.
 import subprocess
 subprocess.run(['pdftoppm','-png','-r','60',str(output/'report.pdf'),str(output/'page_previews'/'page')],check=True,capture_output=True)
 bindings={k:v for k,v in inputs.items() if isinstance(v,dict) and 'path' in v and 'sha256' in v}
 for role in ('control','candidate'):
  for key,ref in inputs[role].items():
   if isinstance(ref,dict) and 'path' in ref and 'sha256' in ref:bindings[f'{role}_{key}']=ref
 for name,refs in inputs['diagnostic_figures'].items():
  for ext,ref in refs.items():bindings[f'diagnostic_{name}_{ext}']=ref
 evidence={'schema':'car_scale_followup_report_evidence_v1','status':'rendered_pending_visual_inspection','input_manifest':{'path':str(inputs_path.resolve()),'sha256':sha(inputs_path)},'bindings':bindings,'report_md_sha256':sha(output/'report.md'),'report_pdf_sha256':sha(output/'report.pdf'),'figures':{k:{e:sha(figures/f'{k}.{e}') for e in ('png','pdf')} for k in captions},'pages':pages,'elapsed_seconds':time.time()-started,'test_locked':True}
 (output/'evidence_index.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n');return evidence

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(produce(a.inputs.resolve(strict=True),a.campaign,a.output),indent=2))
if __name__=='__main__':main()
