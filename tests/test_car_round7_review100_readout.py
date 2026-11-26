import copy
import json
from pathlib import Path
import pytest
import torch
import yaml
from scripts.analysis.car_round7_review100_readout import capture_training_evidence, matched_configs, ref, streams, window


def test_terminal_training_snapshot_survives_continuation_and_status_change(tmp_path):
    names={'history':'history.jsonl','journal':'epoch_calibrated_pi_updates.jsonl',
           'coupling':'training_coupling.jsonl','query':'scalable_updates.jsonl','status':'status.json'}
    originals={}
    for key,name in names.items():
        path=tmp_path/'run'/name;path.parent.mkdir(exist_ok=True);path.write_text('{"step":1200}\n')
        originals[key]=ref(path)
    saved=capture_training_evidence(originals,tmp_path/'review'/'training_evidence'/'Composite_Scale_Control')
    for key in names:
        path=tmp_path/'run'/names[key]
        path.write_text('{"step":6000}\n' if key=='status' else path.read_text()+'{"step":1201}\n')
        assert ref(saved[key]['path'])['sha256']==originals[key]['sha256']
        assert saved[key]['original']==originals[key]
        assert ref(path)['sha256']!=saved[key]['sha256']
    assert ref(saved['snapshot_receipt']['path'])==saved['snapshot_receipt']
from scripts.analysis.car_round7_paired_summary import summarize_scale


def test_streams_detect_changed_member_or_group_query():
    exposure=[{'condition_group':i,'index':i} for i in range(8)]
    coupled={'step':1,'sample_ids':['train:1'],'observation_sha256':'obs','query_sha256':'query',
        'member_seeds':[1,2],'first_member_generator_state_sha256':'state',
        'field_complement_sha256':'comp','field_complement_seed':42}
    left=(None,None,None,[copy.deepcopy(coupled) for _ in range(1200)],
          [{'step':i+1,'exposure':copy.deepcopy(exposure)} for i in range(1200)])
    right=copy.deepcopy(left)
    assert streams(left,right)['matched_actions']==1200
    right[3][0]['member_seeds'][1]=99
    with pytest.raises(ValueError,match='stream mismatch'): streams(left,right)
    right=copy.deepcopy(left)
    right[4][0]['exposure'].pop()
    with pytest.raises(ValueError,match='exposure mismatch'): streams(left,right)


def test_late_windows_use_action_level_controller_and_epoch_means():
    hist=[{'coherence_component/multiscale_spectrum/joint/raw':float(i)} for i in range(1,101)]
    campaign=Path('cases/turbulent_combustion/runs/coherence_adaptive_repair/round_7')
    path=next((campaign/'runs/qualification/Balanced_Mature_Executor_Epochs_v1/Epoch_02_factored').glob('*/logs/epoch_calibrated_pi_updates.jsonl'))
    example=json.loads(path.open().readline())
    fields=json.loads((campaign/'artifacts/TrainingLaw_SOURCE_Reference_v2/reference.json').read_text())['reference']['field_names']
    journal=[]
    for i in range(1200):
        row=copy.deepcopy(example)
        row['field_violation']={field:float(i) for field in reversed(fields)}
        row['multipliers_before']=[1.0 if i%2 else 0.0]*len(fields)
        journal.append(row)
    result=window(hist,journal,51,75,fields)
    assert result['history_means']['coherence_component/multiscale_spectrum/joint/raw']==63
    assert result['active_field_row_fraction']==.5
    assert list(result['field_violation']['last'])==fields
    assert result['field_violation']['last'][fields[0]]==899.0
    assert list(result['multipliers_after']['mean'])==fields


def test_scientific_config_mismatch_rejected(tmp_path):
    root=Path('cases/turbulent_combustion/runs/coherence_adaptive_repair/round_6/configs')
    cfg=yaml.safe_load((root/'FFM_SSS_EpochCalibrated_PI_Repair.yaml').read_text())
    cfg['coherence']['scalable_anchored_coherence'].update(scale_objective='composite',scale_joint_kappa=1.)
    candidate=copy.deepcopy(cfg)
    candidate['coherence']['scalable_anchored_coherence'].update(scale_objective='joint_anchor',scale_joint_kappa=2.)
    p,q=tmp_path/'control.yaml',tmp_path/'joint.yaml'
    p.write_text(yaml.safe_dump(cfg)); q.write_text(yaml.safe_dump(candidate))
    assert matched_configs(p,q)['candidate_kappa']==2.
    candidate['optimization']['lr_warmup']['epochs']=10
    q.write_text(yaml.safe_dump(candidate))
    with pytest.raises(ValueError,match='scientific configurations differ'): matched_configs(p,q)


def test_terminal_source_baseline_and_bank_match_readout_contract():
    campaign=Path('cases/turbulent_combustion/runs/coherence_adaptive_repair/round_7')
    baseline=campaign/'assessment/SOURCE_Baseline_VALIDATION100_v1'
    source=campaign/'assessment/SOURCE_Selected_VALIDATION100_v1'
    cached=torch.load(baseline/'0000.pt',map_location='cpu',weights_only=True)
    bank=torch.load(source/'0000.pt',map_location='cpu',weights_only=True)
    assert cached['sample_id']==bank['sample_id'] and len(bank['member_seeds'])==4
    assert {'member_mse','ensemble_mean_mse','signed_mean_risk','unbiased_member_variance','sensor_mse'} <= cached['native_fields'].keys()
    assert {'state','scale','shape','shape.self','shape.mutual'} <= cached['scores'].keys()
    config=yaml.safe_load((campaign/'configs/training_law_base.yaml').read_text())
    prefit=json.loads(Path(config['coherence']['scalable_anchored_coherence']['common_prefit']['path']).read_text())['coefficients']
    model=cached['decomposition']
    result=summarize_scale({'records':[{'sample_id':cached['sample_id'],
        'models':{'SOURCE':model,'candidate':model}}], 'sample_ids':[cached['sample_id']],
        'members':4,'split':'validation','prefit_coefficients':prefit},'candidate',1.,bootstrap=5)
    assert result['definitions']['Complete_Expanded_Prefit']['mean']==pytest.approx(0)
    assert set(result['weighted_blocks'])=={'signed','joint','log_power_self','rho','third_order'}
