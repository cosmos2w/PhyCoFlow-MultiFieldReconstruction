"""The diagnostic reuses the supplied Shape query and exposes real scorer components."""
from types import SimpleNamespace
import torch
from scripts.qualification.car_round7_action_audit import score_result, shape_summary, detach_tree


def test_shape_query_passes_through_and_summarizes_available_components():
    fixed, independent = object(), object()
    selected = SimpleNamespace(sample_ids=('example',), query_coords=torch.zeros(1,2,1),
        metadata={}, query_valid_mask=torch.ones(1,2,dtype=torch.bool))
    endpoint = {'coupling_selected_batch':selected, 'target':torch.zeros(1,2,1)}
    class Runtime:
        batch = fixed
        scale_objective = 'joint_anchor'
        scale_joint_kappa = 3.0
        families = {'topology': object()}
        def _score(self,name,family,prediction,target,coordinates,context,batch):
            assert batch is independent
            assert self.scale_objective == 'composite'
            assert len(context['conditional_draws']) == 1
            result = SimpleNamespace(scalar_loss=torch.tensor(2.),diagnostics={
                'score_definition':'shape_test', 'shape_sampling':'query_test',
                'components': {'topology.self': {'raw_scalar_loss':1.25,'weight':2.0,
                                                'weighted_scalar_contribution':2.5,'executed':True}}})
            return result, {}, {}
    runtime=Runtime()
    members=(torch.zeros(1,2,1),torch.ones(1,2,1))
    result=score_result(runtime,members,endpoint,'topology','composite',1.0,query=independent)
    observed=shape_summary(result)
    assert observed['components']['topology.self'] == {'raw':1.25,'weight':2.0,'weighted':2.5}
    assert observed['scalar'] == 2.0 and observed['score_definition'] == 'shape_test'
    assert runtime.scale_objective == 'joint_anchor' and runtime.scale_joint_kappa == 3.0
    assert runtime.batch is fixed


def test_detach_tree_frees_nested_graphs_without_changing_values():
    value=torch.tensor([2.],requires_grad=True).square()
    detached=detach_tree({'blocks':[value], 'value':(value,)})
    assert not detached['blocks'][0].requires_grad
    torch.testing.assert_close(detached['value'][0],value)
