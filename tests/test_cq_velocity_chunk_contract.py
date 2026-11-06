"""Autograd rollouts retain production query boundaries and live conditioning."""
from copy import deepcopy

import torch
from helpers.pointcloud import _small_core_config

from phycoflow_reconstruction.contracts import DataSpec, ObservationBatch
from phycoflow_reconstruction.models import build_model


def test_cq_velocity_matches_cached_streamed_boundaries_and_strict_state(monkeypatch):
    config = _small_core_config(2)
    config.pop('model_name'); config.pop('coord_dim')
    config.update(name='gl_rbf_cq', reconstruction_execution_mode='cached_streamed',
                  reconstruction_query_chunk_size=3)
    spec = DataSpec(('u','v'),('1','1'),2,(7,))
    model = build_model(config,spec).eval()
    state_dict = deepcopy(model.state_dict())
    model.load_state_dict(state_dict,strict=True)
    coords = torch.rand(1,7,2)
    values = torch.randn(1,3,1,requires_grad=True)
    batch = ObservationBatch(coords[:,:3],values,torch.tensor([[0,1,0]]),
        torch.ones(1,3,dtype=torch.bool),coords,torch.ones(1,7,dtype=torch.bool),None,('fixture:0',))
    state = torch.randn(1,7,2,requires_grad=True);time=torch.tensor([.25])
    prepare, query = model.model.prepare_condition_context,model.model.forward_query_chunk
    calls=[]; contexts=[]
    def tracked_prepare(*args):
        context=prepare(*args);contexts.append(context);return context
    def tracked_query(t,x,q,context):
        calls.append((q.shape[1],id(context)));return query(t,x,q,context)
    monkeypatch.setattr(model.model,'prepare_condition_context',tracked_prepare)
    monkeypatch.setattr(model.model,'forward_query_chunk',tracked_query)
    actual=model.velocity(batch,state,time)
    assert len(contexts)==1
    assert [n for n,_ in calls]==[3,3,1]
    assert len({context for _,context in calls})==1
    context=prepare(batch.obs_coords,batch.obs_values,batch.obs_valid_mask,batch.obs_field_ids)
    expected=torch.cat([query(time,state[:,i:i+3],coords[:,i:i+3],context) for i in range(0,7,3)],1)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    gx,gv=torch.autograd.grad(actual.square().sum(),(state,values))
    assert torch.isfinite(gx).all() and torch.isfinite(gv).all() and gv.abs().sum()>0
    assert state_dict.keys()==model.state_dict().keys()
