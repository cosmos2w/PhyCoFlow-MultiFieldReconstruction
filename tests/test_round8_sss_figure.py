import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from phycoflow_reconstruction.evaluation import round8_sss_figure as sss
from phycoflow_reconstruction.evaluation.round8_campaign import canonical_hash, file_hash


def test_sss_uses_frozen_observer_descriptor_interfaces(tmp_path,monkeypatch):
    from phycoflow_reconstruction import coherence
    from phycoflow_reconstruction.data import factory
    from phycoflow_reconstruction.training import coherence_sampling, post_training, round8_query_runtime
    dirs=[];panels=[]
    target=np.array([[1.,2.],[3.,4.]],dtype=np.float32)
    for name in ('SOURCE','C8'):
        d=tmp_path/name;d.mkdir();dirs.append(d)
        pred=np.stack([target,target+1]) + (1 if name=='C8' else 0)
        np.savez(d/'part.npz',draws_model=pred,target_model=target,model_coordinates=np.array([[0.,0.],[1.,1.]],dtype=np.float32),physical_mask=np.ones(2,dtype=bool),physical_weights=np.array([.25,.75]),native_ids=np.arange(2))
        panel={'population':'TRAIN','field_names':['a','b'],'conditions':[{'sample_id':'x','path':'part.npz'}]};panels.append(panel)
        (d/'panel.json').write_text(json.dumps(panel))
    monkeypatch.setattr(sss,'validate_matched_panels',lambda *_:tuple(panels))
    state=SimpleNamespace(units='model_units',field_ids=(0,1),components_by_key={'cross_joint_topk_swd':SimpleNamespace(directions=torch.tensor([[.6,.8]]))},_in_declared_units=lambda x,y:(x,y))
    scale=SimpleNamespace(settings=SimpleNamespace(bands=SimpleNamespace(names=('low','high'))),field_ids=(0,1),field_names=('a','b'))
    class Shape:
        spatial_objective=SimpleNamespace(transform_frozen=torch.tensor(True),calibrated=torch.tensor(True),fields=('a','b'),units=(('self.persistence','self.a.positive',(0,),(1,),None),),dimensions=(0,),_diagrams=lambda grid,indices,mask:([SimpleNamespace(finite=torch.tensor([[float(grid.mean()),float(grid.mean())+1]]),essential=torch.tensor([0.]))],None))
        native_masked=False
        def load_state_artifact(self,artifact): self.loaded=artifact
        def fair_prepare_grid(self,member,coordinates): return member
    shape=Shape();families={'global_distribution':state,'multiscale_spectrum':scale,'topology':shape}
    monkeypatch.setattr(coherence,'build_enabled_families',lambda *args:families.copy())
    monkeypatch.setattr(factory,'open_field_dataset',lambda *a,**k:SimpleNamespace(data_spec=None,normalizer=None,close=lambda:None))
    monkeypatch.setattr(coherence_sampling,'_fit_training_references',lambda *a,**k:None)
    monkeypatch.setattr(post_training,'_configure_persistence_workers',lambda *a,**k:None)
    monkeypatch.setattr(round8_query_runtime,'_scale_statistics',lambda family,member:(None,None,torch.tensor([[1.,2.,3.,4.]])+member.mean()))
    fair=tmp_path/'fair.pt';torch.save({'frozen':True},fair)
    config={'dataset':{},'coherence':{'families':{'topology':{'persistence':{}}}},'rollout':{},'observation_consistency':{}}
    saved={'schema':'round8_common_observers_v1','population':'TRAIN','source_panel_sha256':file_hash(dirs[0]/'panel.json'),'endpoint_panel_sha256':file_hash(dirs[1]/'panel.json'),'alpha':.5,'config_sha256':canonical_hash(config),'frozen_family_artifacts_sha256':{},'fair_shape_artifact_sha256':file_hash(fair),'native_observer_identity':None,'rows':[{'sample_id':'x'}]}
    score=tmp_path/'score.json';score.write_text(json.dumps(saved))
    spec={'config':config,'source_panel':dirs[0]/'panel.json','endpoint_panel':dirs[1]/'panel.json','score_report':score,'fair_shape_artifact':fair,'sample_id':'x','field':'a','alpha':.5,'label':'C8 reference-only'}
    meta=sss.build_sss_figure(spec,png_path=tmp_path/'sss.png',pdf_path=tmp_path/'sss.pdf')
    assert meta['direction']==pytest.approx([.6,.8]) and meta['shape_unit']=='self.a.positive'
    assert meta['band_names']==['low','high'] and shape.loaded=={'frozen':True}
    assert meta['label']=='C8 reference-only'
    assert (tmp_path/'sss.png').is_file() and (tmp_path/'sss.pdf').read_bytes().startswith(b'%PDF')
    with pytest.raises(ValueError,match='score report differ'):
        sss.build_sss_figure({**spec,'alpha':.25},png_path=tmp_path/'bad.png',pdf_path=tmp_path/'bad.pdf')
