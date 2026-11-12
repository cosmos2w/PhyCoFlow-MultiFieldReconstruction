import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from phycoflow_reconstruction.training.round8_artifacts import fit_artifacts, source_panel_cache_factory
from phycoflow_reconstruction.training.round8_calibration import BLOCKS
from phycoflow_reconstruction.evaluation.round8_campaign import file_hash


def test_train_artifact_fit_risks_and_identity(monkeypatch):
    import phycoflow_reconstruction.training.round8_artifacts as a
    directions = torch.tensor([[1.,0.],[0.,1.],[.70710678,.70710678]],dtype=torch.float64)
    state = SimpleNamespace(field_names=('a','b'),field_ids=(0,1),components_by_key={'cross_joint_topk_swd':SimpleNamespace(directions=directions)},component_weights={'self_marginal_w2':2.,'cross_joint_topk_swd':3.},_in_declared_units=lambda x,y:(x,y))
    shape = SimpleNamespace(field_names=('a','b'),spatial_objective=SimpleNamespace(weights={'self.persistence':.7,'mutual.persistence':.8}),_units_and_fields=lambda x,y:(x,y))
    monkeypatch.setattr(a,'fit_pooled_shape',lambda *args,**kw:{'frozen':True})
    records=[]
    for i in range(3):
        y=torch.arange(8,dtype=torch.float64).reshape(4,2)
        records.append({'id':str(i),'split':'train','predictions_model':torch.stack((y+1,y+3)),'target_model':y,'physical_weights':torch.tensor([.1,.1,.1,.1]),'inclusion_probability':.5,'coordinates':torch.zeros(4,2)})
    keys = [f'{f}.{b}' for f,bs in BLOCKS.items() for b in bs]
    scores = {k:[-1.,2.,-3.] for k in keys}
    provenance = {'split':'train','manifest_sha256':'a'*64,'source_checkpoint_sha256':'b'*64,'observer_identity':'old-common-v1','retained_priorities':{'scale.joint':1.,'scale.log_power_self':1.,'scale.rho':1.}}
    def fit(**changes):
        args=dict(cache_factory=lambda:iter(records),state_family=state,shape_family=shape,candidates=directions,field_order=('a','b'),manifest_sha256='a'*64,source_checkpoint_sha256='b'*64,normalizer_sha256='c'*64,score_provenance=provenance,block_observations=scores,common_scores={'scale':[1.,2.,3.],'shape':[-2.,-3.,-4.],'state':[0.,0.,0.]},score_floors={'scale':.01,'shape':.01,'state':.5},field_scale=[2.,3.],shape_sample_ids=['0','1'],query_batches=[object()],c0=1.)
        args.update(changes)
        return fit_artifacts(**args)
    result=fit()
    assert torch.allclose(result['source_risk_model'],torch.tensor([4.,4.],dtype=torch.float64),atol=1e-7)
    assert torch.allclose(result['source_risk_physical'],torch.tensor([16.,36.],dtype=torch.float64),atol=1e-6)
    assert result['fidelity_reference']['relative_increase'] == .1
    assert result['units']['state.marginal'] == 2
    assert result['priorities']['state.marginal'] == 2
    assert result['common_score_denominators'] == {'scale':2.,'shape':3.,'state':.5}
    with pytest.raises(ValueError,match='provenance'):
        fit(score_provenance={**provenance,'split':'assess'})


def test_panel_loader_validates_shard_and_identity(tmp_path):
    manifest={'TRAIN':[{'id':'a'}]}
    mp=tmp_path/'manifest.json'; mp.write_text(json.dumps(manifest))
    shard=tmp_path/'item.npz'
    np.savez(shard,sample_id='a',field_names=np.array(['u']),inclusion_probability=.5,physical_mask=np.ones(2,dtype=bool),native_ids=np.arange(2),draws_model=np.zeros((2,2,1)),target_model=np.zeros((2,1)),physical_weights=np.array([.2,.3]),model_coordinates=np.zeros((2,2)))
    panel={'schema':'round8_endpoint_panel_v1','status':'completed','population':'TRAIN','endpoint':'SOURCE','manifest_sha256':file_hash(mp),'checkpoint_sha256':'b'*64,'source_checkpoint_sha256':'b'*64,'normalizer_sha256':'c'*64,'field_names':['u'],'inclusion_probability':.5,'conditions':[{'sample_id':'a','dataset_index':0,'path':'item.npz','sha256':file_hash(shard)}]}
    pp=tmp_path/'panel.json'; pp.write_text(json.dumps(panel))
    factory, meta=source_panel_cache_factory(pp,mp)
    assert list(factory())[0]['id']=='a' and meta['normalizer_sha256']=='c'*64
    shard.write_bytes(b'changed')
    with pytest.raises(ValueError,match='shard changed'): list(factory())
