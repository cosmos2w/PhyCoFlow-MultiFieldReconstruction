import copy
import json
import math

import numpy as np
import pytest
import torch

from scripts.posttraining.assess_car_round6 import check_terminal, intervals, summarize_fields
from phycoflow_reconstruction.evaluation.round2_assessment import native_risks, field_sensor_eligibility


def test_signed_risk_allowance_and_matched_intervals():
    # One field, matched J4. Negative unbiased mean risk is a valid observable.
    truth=torch.zeros(1,2,1,dtype=torch.float64)
    source=torch.tensor([[[[1.],[1.]]],[[[-1.],[-1.]]],[[[1.],[1.]]],[[[-1.],[-1.]]]],dtype=torch.float64)
    child=source*.5
    mass=torch.tensor([2.,1.],dtype=torch.float64)
    old=native_risks(source,truth,mass,model_family='ffm')
    new=native_risks(child,truth,mass,model_family='ffm')
    assert old['signed_mean_risk'].item()==pytest.approx(-1/3)
    assert new['signed_mean_risk'].item()==pytest.approx(-1/12)
    # TRAIN model-unit allowance .1 converts by the squared normalization 2**2.
    result=field_sensor_eligibility(old,new,model_family='ffm',absolute_mean_allowance=[.1*2**2],source_sensor_mse=[0.],child_sensor_mse=[0.],sensor_equality_tolerance=[0.])
    assert result['primary_pass'] and result['field_difference'][0]==pytest.approx(.25)
    diff=intervals([[1.,2.],[2.,4.]],[[2.,3.],[3.,5.]],scale=[2.,4.])
    assert diff['mean']==pytest.approx([.5,.25])
    assert np.asarray(diff['interval_95'])==pytest.approx(np.asarray([[.5,.25],[.5,.25]]))
    with pytest.raises(ValueError): intervals([1.,float('nan')],[2.,3.])


def test_endpoint_requires_explicit_decision_matching_review_and_bank():
    review={'epoch':100,'_sha256':'review','task_scope':'both tasks excluded','bank_receipt_sha256':'bank'}
    decision={'epoch':100,'decision':'stop_at_epoch_100','authorize_endpoint_tasks':True,'review_sha256':'review','bank_receipt_sha256':'bank'}
    check_terminal(decision,review,'bank')
    for key,value in [('authorize_endpoint_tasks',False),('review_sha256','other'),('bank_receipt_sha256','other'),('decision','endpoint_at_epoch_500')]:
        altered={**decision,key:value}
        with pytest.raises(ValueError):check_terminal(altered,review,'bank')
    terminal={**decision,'epoch':500,'decision':'endpoint_at_epoch_500'}
    check_terminal(terminal,{**review,'epoch':500},'bank')


def test_report_figure_case_uses_physical_mask_and_predeclared_cases(tmp_path):
    import matplotlib;matplotlib.use('Agg')
    from scripts.analysis.report_car_round6 import case_figures
    from scripts.analysis.report_car_round5 import sha
    from phycoflow_reconstruction.evaluation.adaptive_assessment import _identity
    from scripts.posttraining.freeze_car_event import FrozenRedesignedEvent
    names=['T','CO','CH4']; masses=torch.tensor([1.,2.,0.]); coords=torch.tensor([[0.,0.],[1.,0.],[2.,0.]])
    truth=torch.tensor([[[1.,1.,0.],[0.,1.,0.],[0.,0.,0.]]]);members=truth.repeat(4,1,1,1)
    base={'sample_id':'condition','sample_index':0,'truth_model':truth,'members_model':members,'coordinates_model':coords[None], 'coordinates_raw':coords,'native_masses':masses, 'normalizer_scale':torch.ones(3),'normalizer_offset':torch.zeros(3),'obs_indices':torch.tensor([[0]]),'obs_values':torch.tensor([[[1.]]]),'obs_fields':torch.tensor([[0]]),'member_seeds':[1,2,3,4]}
    source_dir=tmp_path/'source';child_dir=tmp_path/'child';source_dir.mkdir();child_dir.mkdir()
    for i in range(3):
        row={**base,'sample_id':f'condition{i}'}
        torch.save(row,source_dir/f'{i:04d}.pt');torch.save(row,child_dir/f'{i:04d}.pt')
    source={'files_sha256':[sha(source_dir/f'{i:04d}.pt') for i in range(3)],'sample_ids':[f'condition{i}' for i in range(3)]}
    candidate={'files_sha256':[sha(child_dir/f'{i:04d}.pt') for i in range(3)],'start_unix':2.}
    srcpath=source_dir/'receipt.json';srcpath.write_text(json.dumps(source));candidatepath=child_dir/'receipt.json';candidatepath.write_text(json.dumps(candidate))
    selection={'source_bank_sha256':sha(srcpath),'candidates_used':False,'created_unix':1.,'cases':[{'ordinal':i,'sample_id':f'condition{i}'} for i in range(3)]}
    payload={'schema':'car_event_task_redesign_v1','split':'train','field_names':names,'definition':{'T':{'operator':'>','threshold':.5},'CO':{'operator':'>','threshold':.5},'CH4':{'operator':'<','threshold':.5}}}
    event=FrozenRedesignedEvent({**payload,'sha256':_identity(payload)})
    figures=tmp_path/'figures';figures.mkdir()
    captions=case_figures({'arm':'repair','field_order':names},selection,srcpath,source,candidatepath,candidate,event,figures)
    assert len(captions)==9
    assert all((figures/f'{name}.{suffix}').is_file() for name in captions for suffix in ('png','pdf'))
    selection['created_unix']=3.
    with pytest.raises(ValueError,match='before child'):case_figures({'arm':'repair','field_order':names},selection,srcpath,source,candidatepath,candidate,event,figures)


def test_learning_separates_raw_families_leaves_and_coefficients(tmp_path):
    from scripts.analysis.report_car_round6 import learning
    history=[{'epoch':i,'optimizer_lr':1e-4,'coherence_direction_norm':2.,'constraint_gradient_norm':1.,
              'coherence_family/global_distribution/raw':.2,'coherence_family/topology/raw':.4,
              'coherence_component/topology/self/raw':.3,'coherence_component/topology/mutual/raw':.5} for i in (1,2)]
    journal=[{'step':i,'field_violation':{'T':.1},'multipliers_after':[.2],
              'action_family_products':{'actual_displacement':{'state':-.1}},'action_constraint_product':-.2} for i in range(1,25)]
    assert learning(history,journal,['T'],{'state.self':1.},tmp_path)==['leaf_history_1']
    for name in ('learning_and_action','family_history','leaf_history_1','fixed_coefficients'):
        assert (tmp_path/f'{name}.png').is_file() and (tmp_path/f'{name}.pdf').is_file()


def test_preparation_figures_keep_action_products_distinct_from_finite_shape(tmp_path):
    from scripts.analysis.report_car_round6 import preparation_figures
    fields=['one','two']; family=['global_distribution','multiscale_spectrum','topology']
    audit={'field_order':fields,'family_order':family,'controller_qualification':{'baseline_drift':[-.1,-.2], 'target_drift':[-.01,-.02], 'baseline_drift_over_target':[10.,10.]},
      'action_evidence':{'zero_family_actual_products':{'mean':[-.01,-.02,-.03]},'zero_fixed_query_shape_changes':{'mean':-.04},'zero_independent_query_shape_changes':{'mean':-.02}},
      'source_shape_variation':{'condition_variance':.1,'paired_member_half_mean_square':.02,'paired_query_half_mean_square':.3},
      'shape_components':{'components':{'self.finite.attraction':{'parameter_gradient_rms':.01}}},
      'source_support':{'member_mse_native':[1.,2.],'member_mse_selected':[1.1,2.1],'risk_native':[-.1,.3],'risk_selected':[-.05,.4]}}
    profile={'epoch_timing':{'train_seconds':100.,'wall_seconds':110.},'scopes':{k:{'seconds':1.} for k in ('wide_pseudoinverse','shape_pairing_and_transfers','family_and_constraint_backward')}}
    history=[{'epoch':i,'optimizer_lr':.001,'field_violation':{'one':.1,'two':.2},'policy_state':{'multipliers':[.1,.2]},'coherence_family/global_distribution/raw':.1,'coherence_component/topology/self/raw':.2} for i in range(1,4)]
    captions=preparation_figures(audit,{},profile,history,tmp_path)
    assert 'action_and_shape' in captions and 'first-order' in captions['action_and_shape']
    assert all((tmp_path/f'{name}.{suffix}').is_file() for name in captions for suffix in ('png','pdf'))
