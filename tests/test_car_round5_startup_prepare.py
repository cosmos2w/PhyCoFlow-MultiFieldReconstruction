import json
from pathlib import Path
import pytest
import yaml
from scripts.qualification import prepare_car_round5_startup as startup


def test_startup_configs_and_bounded_correction(tmp_path, monkeypatch):
    monkeypatch.setattr(startup, 'validate_config', lambda config: None)
    monkeypatch.setattr(startup, 'source_hashes', lambda config: {'checkpoint': 'sourcehash'})
    ref = tmp_path/'reference.json'
    ref.write_text(json.dumps({'risk_reference_panel':{'source_member_mse':[1.]*5}}))
    def checked(declaration):
        if not declaration: return ref
        path = Path(declaration['path'])
        if startup.digest(path) != declaration['sha256']: raise ValueError('frozen artifact changed')
        return path
    monkeypatch.setattr(startup, 'checked', checked)
    cfg = {'source':{'weight_view':'ema_materialized'},
           'optimization':{'batch_size':64,'steps_per_epoch':12,'sampling':'full_pass',
                           'multitask':{'fidelity_budget':{'reference_artifact':{}}}},
           'coherence':{'compute_budget':{'batch_size':64,'point_count':4096},
                        'scalable_anchored_coherence':{'common_core':{'source_panel_sha256':'panelhash'}}},
           'rollout':{'steps':4,'solver':'euler'}, 'runtime':{},'evaluation':{},'output':{},'checkpointing':{}}
    config = tmp_path/'historical.yaml'; config.write_text(yaml.safe_dump(cfg))
    units = tmp_path/'units.json'; units.write_text(json.dumps({'schema':'car_round5_source_units_v1','status':'completed',
        'sample_count':800,'split':'train','members':2,'floor_active':{},'source_checkpoint_sha256':'sourcehash',
        'source_panel_sha256':'panelhash','units':dict.fromkeys(('native','state','scale','shape'), 1.)}))
    action = tmp_path/'action.json'; action.write_text(json.dumps({'schema':'car_round5_action_replay_v1',
        'status':'completed','sample_count':800,'units_sha256':startup.digest(units),'field_tolerance_max':[1e-7]*5}))
    args=dict(campaign=tmp_path,config_path=config,units_path=units,action_path=action)
    receipt=startup.prepare(**args,attempt_id='Initial_Scalar')
    candidate=yaml.safe_load(Path(receipt['configs'][startup.ARMS[1]]['path']).read_text())
    native=yaml.safe_load(Path(receipt['configs'][startup.ARMS[0]]['path']).read_text())
    historical=yaml.safe_load(Path(receipt['configs']['Historical_SourceScaledRisk_Timing']['path']).read_text())
    assert candidate['optimization']['direct_supervision']['mode']=='safeguarded'
    assert candidate['optimization']['epochs']==500 and candidate['optimization']['lr_warmup']['epochs']==50
    assert native['coherence']['scalable_anchored_coherence']=={'enabled':False}
    assert 'multitask' in historical['optimization'] and 'direct_supervision' not in historical['optimization']
    with pytest.raises(ValueError,match='already prepared'):
        startup.prepare(**args,attempt_id='Duplicate_Initial')
    evidence=tmp_path/'correction.json'; evidence.write_text(json.dumps({'demonstrated_failure':True,'correction':'lower LR and beta1 zero',
        'finite_step_overshoot_demonstrated':True,'ascending_proposal_fraction':.2}))
    corrected=startup.prepare(**args,attempt_id='Combined_Correction',kind='correction',correction_path=evidence)
    assert corrected['lr_target']==2.5e-5 and corrected['beta1']==0.
    assert corrected['configs']['Historical_SourceScaledRisk_Timing'] == receipt['configs']['Historical_SourceScaledRisk_Timing']
    with pytest.raises(ValueError,match='already prepared'):
        startup.prepare(**args,attempt_id='Duplicate_Correction',kind='correction',correction_path=evidence)
