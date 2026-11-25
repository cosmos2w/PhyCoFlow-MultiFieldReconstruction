"""Finite-population estimator and stream properties for rotating PI feedback."""
from itertools import combinations
import pytest
import torch
from phycoflow_reconstruction.coherence.physical_risk import physical_ensemble_risks
from phycoflow_reconstruction.training.fidelity_quadrature import complement_indices, stratified_mean_risks


def test_exact_complement_enumeration_matches_full_signed_risk_and_gradient():
    torch.manual_seed(123)
    x = torch.randn(2, 6, 2, dtype=torch.float64)
    truth = torch.randn(1, 6, 2, dtype=torch.float64)
    masses = torch.tensor([1., 3., 0., 2., 4., 6.], dtype=torch.float64)
    parameter = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    members = (x * parameter).unsqueeze(0)
    full = physical_ensemble_risks(members, truth, masses)[1]
    selected = torch.tensor([[0, 3]])
    estimates=[]
    for indices in combinations([1,2,4,5],2):
        c = torch.tensor([indices])
        estimates.append(stratified_mean_risks(members[:,:,selected[0]], truth[:,selected[0]],
            members[:,:,c[0]], truth[:,c[0]], selected, c, masses))
    expectation = torch.stack(estimates).mean(0)
    assert torch.allclose(expectation, full, atol=1e-12)
    full_grad = torch.autograd.grad(full.sum(), parameter, retain_graph=True)[0]
    estimated_grad = torch.autograd.grad(expectation.sum(), parameter)[0]
    assert torch.allclose(estimated_grad, full_grad, atol=1e-12)


def test_complement_is_valid_deterministic_at_absolute_resume_cursor_without_rng_consumption():
    selected=torch.tensor([[0,2,4],[1,3,5]])
    torch.manual_seed(19);before=torch.get_rng_state().clone()
    first=complement_indices(selected,9,2,seed=20261125,absolute_step=6017)
    assert torch.equal(first,complement_indices(selected,9,2,seed=20261125,absolute_step=6017))
    assert torch.equal(before,torch.get_rng_state())
    for old,new in zip(selected,first,strict=True):
        assert len(set(old.tolist()) & set(new.tolist()))==0 and len(set(new.tolist()))==2
    with pytest.raises(ValueError,match='unique'):
        complement_indices(torch.tensor([[0,0]]),9,2,seed=1,absolute_step=2)


def test_separate_complement_reuses_observations_and_member_rng(monkeypatch):
    from phycoflow_reconstruction.contracts import FieldSample
    from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol, build_observation_batch
    from phycoflow_reconstruction.training.fidelity_quadrature import rotating_field_risks
    from phycoflow_reconstruction.training.post_training import _conditional_draw_seed
    import phycoflow_reconstruction.training.rollout as rollout
    n=4610
    coords=torch.arange(n,dtype=torch.float32).reshape(-1,1)/n
    sample=FieldSample(values=coords.clone(),coordinates=coords,coordinates_raw=coords,time=torch.tensor(0.),trajectory_id='case',time_index=0,conditions=torch.empty(0),field_names=('T',),logical_shape=(n,))
    protocol=SensorProtocol(field_counts={'T':2},seed=2)
    selected=build_observation_batch([sample],protocol,query_indices=list(range(4096)),manifest_indices={'case:0':[(0,0),(4300,0)]})
    class Dataset:
        physical_weights=torch.ones(n)
        trajectory_ids=['case']
        _items=[(0,0)]
        def __getitem__(self,index):assert index==0;return sample
    param=torch.tensor(0.8,requires_grad=True); seed=111
    def fake_rollout(model,batch,*,generator,**kwargs):
        assert batch.target_fields is None
        return param*batch.query_coords + torch.rand((),generator=generator)
    monkeypatch.setattr(rollout,'differentiable_reconstruction',fake_rollout)
    g=torch.Generator().manual_seed(seed);initial=g.get_state().clone()
    members=[fake_rollout(None,selected.__class__(**{**selected.__dict__,'target_fields':None}),generator=g)]
    members.append(fake_rollout(None,selected.__class__(**{**selected.__dict__,'target_fields':None}),generator=torch.Generator().manual_seed(_conditional_draw_seed(seed,1))))
    endpoint={'coupling_selected_batch':selected,'members':members,'target':selected.target_fields,'rollout_generator_state':initial}
    config={'observations':{'fields':{'T':{'count':2}},'seed':2},'rollout':{'steps':4,'solver':'euler'},'observation_consistency':{},'optimization':{'multitask':{'fidelity_budget':{'rotating_complement':{'points':512,'seed':20261125}}}}}
    state=torch.get_rng_state().clone()
    risk,indices=rotating_field_risks(None,endpoint,Dataset(),config,absolute_step=6017,rollout_seed=seed,device=torch.device('cpu'))
    assert risk.shape==(1,1) and indices.shape==(1,512) and torch.equal(state,torch.get_rng_state())
    assert torch.isfinite(torch.autograd.grad(risk.sum(),param)[0])
    assert not torch.isin(indices,torch.arange(4096)).any()


def test_only_declared_complement_recipe_and_metric_are_admitted():
    from phycoflow_reconstruction.config.validate import _validate_fidelity_budget
    budget={'metric':'physical_mean_risk_model_units','mean_risk_scales':[1.0],
            'member_allowance_fraction':0.1,'rotating_complement':{'points':512,'seed':20261125}}
    _validate_fidelity_budget({'fidelity_budget':budget})
    with pytest.raises(ValueError,match='512-point'):
        _validate_fidelity_budget({'fidelity_budget':{**budget,'rotating_complement':{'points':1024,'seed':20261125}}})
    with pytest.raises(ValueError,match='physical mean risk'):
        _validate_fidelity_budget({'fidelity_budget':{**budget,'metric':'physical_member_mse_model_units'}})


def test_real_compact_b64_variable_observation_padding(monkeypatch):
    """Exercise the producer's actual masked slots and copy them byte-for-byte."""
    import os
    from pathlib import Path
    config_path=os.environ.get('CAR_SCALE_REAL_COMPACT_CONFIG')
    if not config_path or not Path(config_path).is_file():
        pytest.skip('set CAR_SCALE_REAL_COMPACT_CONFIG to the admitted candidate for real-HDF5 fixture')
    import yaml
    from phycoflow_reconstruction.data.factory import open_field_dataset
    from phycoflow_reconstruction.data.training_batches import _CompactH5Dataset, fixed_query_indices
    from phycoflow_reconstruction.training.common import sensor_protocol_from_config
    from phycoflow_reconstruction.training.fidelity_quadrature import rotating_field_risks
    import phycoflow_reconstruction.training.rollout as rollout
    cfg=yaml.safe_load(Path(config_path).read_text())
    dataset=open_field_dataset(cfg['dataset'],split='train')
    try:
        fixed=fixed_query_indices(40300,4096,seed=cfg['coherence']['compute_budget']['query_seed'])
        selected=_CompactH5Dataset(dataset,sensor_protocol_from_config(cfg),4096,fixed).__getitems__([(i,6000) for i in range(64)])[0]
        counts=selected.obs_valid_mask.sum(1)
        assert selected.obs_indices.shape==(64,384) and int(counts.min())>=192 and int(counts.max())<=384
        assert int(counts.min()) < int(counts.max()) and bool((~selected.obs_valid_mask).any())
        parameter=torch.tensor(0.8,requires_grad=True);calls=[]
        def fake(model,batch,*,generator,**kwargs):
            for key in ('obs_indices','obs_field_ids','obs_coords','obs_values','obs_valid_mask'):
                assert getattr(batch,key) is getattr(selected,key)
            assert batch.target_fields is None and batch.query_coords.shape[1]==512
            calls.append(generator.get_state().clone())
            return parameter*batch.query_coords[:,:,:1].expand(-1,-1,5)+torch.rand((),generator=generator)
        monkeypatch.setattr(rollout,'differentiable_reconstruction',fake)
        seed=1000093;gen=torch.Generator().manual_seed(seed);initial=gen.get_state().clone()
        # Fake selected endpoint retains a differentiable graph; no model inference.
        base=parameter*selected.query_coords[:,:,:1].expand(-1,-1,5)
        endpoint={'coupling_selected_batch':selected,'members':[base,base+0.1],
                  'target':selected.target_fields,'rollout_generator_state':initial}
        rng=torch.get_rng_state().clone()
        risk, complement=rotating_field_risks(None,endpoint,dataset,cfg,absolute_step=6000,rollout_seed=seed,device=torch.device('cpu'))
        assert risk.shape==(64,5) and complement.shape==(64,512) and len(calls)==2
        assert torch.equal(calls[0],initial) and torch.equal(rng,torch.get_rng_state())
        assert torch.isfinite(torch.autograd.grad(risk.sum(),parameter)[0])
    finally: dataset.close()
