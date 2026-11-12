import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from phycoflow_reconstruction.evaluation import round8_observer_profile as mod
from phycoflow_reconstruction.evaluation.round8_campaign import file_hash


def test_source_profile_uses_six_existing_ids_and_saves_only_timings(tmp_path,monkeypatch):
    from phycoflow_reconstruction.training import post_training
    from phycoflow_reconstruction.evaluation import loss_aligned
    manifest={'schema':'round8_manifest_v1','TRAIN':[{'id':f't{i}'} for i in range(7)],'ASSESS':[{'id':f'a{i}'} for i in range(7)],'inputs':{'source_checkpoint':{'sha256':'a'}}}
    mp=tmp_path/'manifest.json';mp.write_text(json.dumps(manifest))
    panels={}
    for population,prefix in (('TRAIN','t'),('ASSESS','a')):
        d=tmp_path/population;d.mkdir();rows=[]
        for i in range(7):
            p=d/f'{i}.npz'
            np.savez(p,sample_id=f'{prefix}{i}',field_names=np.array(['u']),draws_model=np.zeros((2,3,1),dtype=np.float32),target_model=np.zeros((3,1),dtype=np.float32),draws_physical=np.zeros((2,3,1),dtype=np.float32),target_physical=np.zeros((3,1),dtype=np.float32),native_ids=np.arange(3),raw_coordinates=np.zeros((3,2)),inclusion_probability=np.asarray(1.),model_coordinates=np.zeros((3,2),dtype=np.float32),physical_weights=np.array([.2,.3,.5]),physical_mask=np.ones(3,dtype=bool))
            rows.append({'sample_id':f'{prefix}{i}','path':p.name,'sha256':file_hash(p)})
        panel={'population':population,'endpoint':'SOURCE','manifest_sha256':file_hash(mp),'checkpoint_sha256':'a','source_checkpoint_sha256':'a','normalizer_sha256':'n','field_names':['u'],'conditions':rows}
        (d/'panel.json').write_text(json.dumps(panel));panels[population]=d/'panel.json'
    monkeypatch.setattr(mod,'validate_panel',lambda path:json.loads(Path(path).read_text()))
    family_calls=[]
    class Family:
        conditional_draws=2
        def __call__(self,pred,target,*,coordinates,context):
            family_calls.append((context['sample_ids'][0],len(context['conditional_draws'])))
            return SimpleNamespace(diagnostics={})
    def setup(config,pop,fair,native,device):
        assert __import__('os').environ['PHYCOFLOW_TOPOLOGY_WORKERS']=='4'
        train=SimpleNamespace(physical_measure_receipt={'sha256':'measure'},normalizer=SimpleNamespace(digest=lambda:'n'),close=lambda:None)
        return train,{name:Family() for name in ('global_distribution','multiscale_spectrum','topology')}, {'native':'id'} if pop=='ASSESS' else None
    monkeypatch.setattr(mod,'_setup',setup)
    monkeypatch.setattr(post_training,'_configure_persistence_workers',lambda _:None)
    monkeypatch.setattr(loss_aligned,'_collect_family_result',lambda *a:None)
    fair=tmp_path/'fair.pt';fair.write_bytes(b'fair');native=tmp_path/'native.pt';native.write_bytes(b'native')
    config={'dataset':{},'coherence':{},'rollout':{},'observation_consistency':{}}
    output=tmp_path/'profile.json'
    mod.profile_observers(config,train_panel=panels['TRAIN'],assess_panel=panels['ASSESS'],manifest_path=mp,fair_shape_artifact=fair,native_bundle=native,output_path=output,device='cpu')
    profile=json.loads(output.read_text())
    assert profile['schema']=='round8_common_observer_resource_profile_v1' and profile['persistence_workers']==4
    assert profile['populations']['TRAIN']['sample_ids']==[f't{i}' for i in range(6)]
    assert profile['populations']['ASSESS']['sample_ids']==[f'a{i}' for i in range(6)]
    assert len(family_calls)==36 and all(n==1 for _,n in family_calls)
    assert 'losses' not in output.read_text() and 'scores' not in profile['populations']['TRAIN']
    with pytest.raises(FileExistsError):mod.profile_observers(config,train_panel=panels['TRAIN'],assess_panel=panels['ASSESS'],manifest_path=mp,fair_shape_artifact=fair,native_bundle=native,output_path=output,device='cpu')
