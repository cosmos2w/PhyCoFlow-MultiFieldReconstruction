import copy
import pytest
from scripts.analysis.car_round7_preparation_report import preparation_lines, validate_preparation, FAMILIES, OBJECTIVES


def fixture():
    diff={'max_abs':1e-7,'relative_l2':2e-7}
    screen=lambda state: {'state':state,'reference_sha256':'ref','field_row_active':state=='mature_checkpoint',
        'attempts':[{'objective':o,'setting':s,'parameter_row_dtype':'float32',
          'per_family_row_comparison_to_first_separate':{name:diff for name in FAMILIES},
          'comparison_to_first_separate':{name:diff for name in ('field','direction','displacement')}}
          for o in OBJECTIVES for s in ('separate_repeat_1','separate_repeat_2','factored_static_features')]}
    obs={'epoch':1,'optimizer_lr':.1,'round7_scale_kappa':2,'topology.self':1,'topology.mutual':1,
         'coherence_component/multiscale_spectrum/joint/raw':1,'coherence_component/multiscale_spectrum/joint/weighted_contribution':2,'active_field_fraction':1,
         **{f'coherence_family/{name}/{part}':1 for name in FAMILIES for part in ('raw','weighted_contribution')}}
    keys=('strict_scientific_checkpoint_differences','strict_recording_definition_differences','strict_scientific_history_differences','logs/epoch_calibrated_pi_updates.jsonl','metrics/training_coupling.jsonl','logs/scalable_updates.jsonl')
    strict=[dict(comparison_atol=0,comparison_rtol=0,**{k:[] for k in keys}) for _ in range(2)]
    return {'schema':'car_round7_terminal_preparation_readout_v1','status':'completed','fresh_source':screen('fresh_source'),'mature_checkpoint':screen('mature_checkpoint'),
        'production_resume':[{'objective':o,'qualified':True} for o in OBJECTIVES], 'strict_resume_comparison':{'source':{'path':'strict','sha256':'x'},'results':strict},
        'joint_pilot':{'reporting_epochs':10,'contiguous_actions':120,'reference_sha256':'ref','kappa':2,'epoch_observations':[dict(obs,epoch=i) for i in range(1,11)],
           'epoch_timing':[{'wall_seconds':2,'train_seconds':1,'callback_and_reporting_seconds':1} for _ in range(10)], 'curves_decoded':[1]*4,
           'actual_displacement_products':{'products':{'weighted_field':{'negative':1,'zero':0,'positive':0,'minimum':-1,'maximum':-1,'mean':-1}}},
           'scalable_calibration':{'coefficients_constant':True,'calibration_frozen_constant':True,'calibration_frozen_first':True,'calibration_frozen_last':True}}}


def test_preparation_table_and_bound_strict_evidence():
    data=fixture()
    validate_preparation(data,'ref',2,lambda d: iter(()),lambda ref:data['strict_resume_comparison']['results'])
    text='\n'.join(preparation_lines(data))
    assert text.count('factored_static_features')==4 and text.count('separate_repeat_2')==4
    assert '120 contiguous accepted actions' in text and 'zero-tolerance' in text
    broken=copy.deepcopy(data);broken['fresh_source']['attempts'].pop()
    with pytest.raises(ValueError,match='both objective'):validate_preparation(broken,'ref',2,lambda d:iter(()),lambda ref:[])
