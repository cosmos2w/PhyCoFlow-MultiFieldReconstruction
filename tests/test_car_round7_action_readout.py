"""Check paired condition summaries and unequal-batch action aggregation."""
import pytest
from scripts.analysis.car_round7_action_readout import (
    BLOCKS, DEFINITIONS, summarize_conditions, summarize_actions, validate_decomposition,
    query_from_manifest, shape_components,
)
import torch


def model(score):
    category={'attraction':score+1,'separation':1.,'score':score}
    shape={query:{kind:{'finite':category,'essential':category} for kind in ('self','mutual')}
           for query in ('fixed','independent')}
    return {'blocks':{key:{'attraction':score+1,'separation':1.,'score':score} for key in BLOCKS},
            'definitions':{key:score for key in DEFINITIONS},'shape':shape}


def test_paired_condition_bootstrap_keeps_members_and_states_aligned():
    records=[{'sample_id':f'id{index}','models':{'pre':model(index),'composite':model(index+2),
        'joint_anchor':model(index+3)}} for index in range(5)]
    result=summarize_conditions(records,bootstrap=40,seed=7)
    comparisons=result['comparisons']
    delta=comparisons['joint_anchor_minus_composite']['raw_blocks']['joint']['score']
    assert delta == {'mean':1.,'interval_95':[1.,1.]}
    assert result['bootstrap']['unit'].startswith('paired condition')
    records[-1]['sample_id']=records[0]['sample_id']
    with pytest.raises(ValueError,match='unique'):summarize_conditions(records,bootstrap=4,seed=7)


def test_batch_diagnostics_weight_by_actual_condition_counts():
    def row(count,value):
        shape=lambda x:{'scalar':x,'components':{'topology.self':{'raw':x,'weighted':2*x}}}
        action={'direction_norm':value,'displacement_norm':value,'clip_input_norm':value,'learning_rate':value,
            'joint_row_dot_displacement':value,'weighted_field_row_dot_displacement':value,
            'row_dot_displacement':{'topology':value},'field_risk_post':[value+1],
            'shape_fixed_post':shape(value+2),'shape_independent_post':shape(value+3)}
        return {'sample_count':count,'scale_row_cosine':value,
            'scale_row_norms':{'composite':value,'joint_anchor':value},'weighted_field_row_norm':value,
            'family_row_norms':{'topology':value},'field_risk_pre':[value],
            'shape_fixed_pre':shape(value),'shape_independent_pre':shape(value+1),
            'arms':{'composite':action,'joint_anchor':action}}
    result=summarize_actions([row(3,1.),row(1,5.)])
    assert result['sample_count']==4 and result['scale_row_cosine']==2.
    assert result['arms']['composite']['field_risk_change']==[1.]
    assert result['arms']['joint_anchor']['shape']['independent']['change']==2.


def test_decomposition_requires_exact_score_and_expanded_reconstruction():
    prefit={'scale.'+key:1. for key in BLOCKS}
    result=model(1.)
    result['definitions']['Complete_Expanded_Prefit']=4.25
    validate_decomposition(result,prefit)
    result['blocks']['joint']['score']=1.1
    with pytest.raises(ValueError,match='A-R'):validate_decomposition(result,prefit)


def test_shape_group_mapping_and_weighted_component_reconstruction():
    def query(index):
        return {'number':index,'self_queries':({'index':index,'count':1,'probability':1.,'population':1,'draws':1},),
                'pair_queries':()}
    manifest={'batches':(query(0),query(1),query(2))}
    assert [query_from_manifest(manifest,index,8).number for index in range(8)] == [0,0,0,1,1,1,2,2]
    class Shape:
        def score_compiled(self,grids,queries,reference,**kwargs):
            assert kwargs['include_components']
            output={}
            for kind in ('self','mutual'):
                output[kind]=torch.tensor([1.5])
                for category in ('finite','essential'):
                    output[f'{kind}.{category}.attraction']=torch.tensor([1.])
                    output[f'{kind}.{category}.separation']=torch.tensor([.25])
            return output
    result=shape_components(Shape(),((),(),None),torch.zeros(2,1,1,1,1),{'self':2.,'mutual':3.})
    assert result['total']==7.5
    assert result['self']['finite']=={'attraction':2.,'separation':.5,'score':1.5}
    assert result['mutual']['sum_check_abs_error']==0.
