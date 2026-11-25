"""Phase clocks, immutable lineage, and matched cached validation comparison."""
import json
from copy import deepcopy
from pathlib import Path
import pytest
import yaml
from phycoflow_reconstruction.training.run_store import config_digest
from scripts.qualification.car_scale_fork_validation import validate_clock, reference, compare


def test_fork_clock_binds_parent_config_checkpoint_and_phase(tmp_path):
    run=tmp_path/'descriptive_fork';(run/'checkpoints').mkdir(parents=True)
    cfg={'stage':'post_training','optimization':{'epochs':500},'output':{'experiment_name':'DescriptiveFork'}}
    config=run/'resolved_config.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    parent=tmp_path/'parent500.pt';parent.write_bytes(b'immutable parent')
    snapshot={'run':str(tmp_path/'parent_run'),'checkpoint':reference(parent)}
    origin={'parent_run':snapshot['run'],'parent_checkpoint_sha256':snapshot['checkpoint']['sha256'],'absolute_step':6000}
    for phase in (100,225,500):
        checkpoint=run/'checkpoints'/f'epoch_{phase:03d}.pt';checkpoint.write_bytes(f'phase {phase}'.encode())
        spec={'variant':'ScaleRepairedContinuation','phase_epoch':phase,'phase_origin_step':6000,'global_step':6000+12*phase,
              'run_config':reference(config),'checkpoint':reference(checkpoint)}
        manifest={'run_id':run.name,'fork_origin':origin,'config_sha256':config_digest(cfg)}
        state={'fork_origin':origin,'global_step':spec['global_step'],'config_sha256':config_digest(cfg)}
        assert validate_clock(spec,manifest,state,run,config,checkpoint,snapshot)==spec['global_step']
        wrong=deepcopy(state);wrong['global_step']-=12
        with pytest.raises(ValueError,match='lineage'):validate_clock(spec,manifest,wrong,run,config,checkpoint,snapshot)
        wrong=deepcopy(manifest);wrong['fork_origin']['absolute_step']=0
        with pytest.raises(ValueError,match='lineage'):validate_clock(spec,wrong,state,run,config,checkpoint,snapshot)
        wrong=deepcopy(spec);wrong['phase_origin_step']=0
        with pytest.raises(ValueError,match='clocks'):validate_clock(wrong,manifest,state,run,config,checkpoint,snapshot)


def test_paired_cache_compare_preserves_units_and_rejects_panel_mismatch(tmp_path):
    campaign=tmp_path/'campaign';(campaign/'assessment').mkdir(parents=True)
    record={'scores':{'state':1.,'scale':2.,'shape':3.},'signed_mean_risk':[1.,2.],
            'member_mse':[2.,4.],'ensemble_mean_mse':[1.,2.],'unbiased_member_variance':[1.,2.]}
    a={'split':'validation','population':1000,'members':4,'field_order':['T','p'],'sample_ids':[str(i) for i in range(100)],
       'source_checkpoint_sha256':'s','source_bank_sha256':'b','panel_sha256':'p','contract_sha256':'c','core_sha256':'x',
       'source_baseline_sha256':'z','normalizers':{'state':1.,'scale':2.,'shape':1.},'native_allowance':[1.,1.],
       'task_scope':'both tasks excluded','per_condition_fields':{'child':[deepcopy(record) for _ in range(100)]}}
    b=deepcopy(a)
    for row in b['per_condition_fields']['child']:row['scores']['scale']+=1.;row['signed_mean_risk'][0]+=.5
    left=tmp_path/'left.json';right=tmp_path/'right.json';left.write_text(json.dumps(a));right.write_text(json.dumps(b))
    dest=campaign/'assessment'/'paired.json';compare(campaign,left,right,dest)
    out=json.loads(dest.read_text());assert out['core_in_frozen_train_units']['scale']['mean']==pytest.approx(.5)
    assert out['physical_fields']['signed_mean_risk']['mean'][0]==pytest.approx(.5)
    b['panel_sha256']='other';right.write_text(json.dumps(b))
    with pytest.raises(ValueError,match='identities'):compare(campaign,left,right,campaign/'assessment'/'bad.json')


@pytest.mark.parametrize(('phase','decision'),[(100,'stop_at_epoch_100'),(225,'endpoint_at_epoch_225'),(500,'endpoint_at_epoch_500')])
def test_explicit_phase_terminal_task_decision_binds_same_review_and_bank(tmp_path,monkeypatch,phase,decision):
    import scripts.qualification.car_scale_fork_validation as driver
    from scripts.posttraining.assess_car_round6 import check_terminal
    campaign=tmp_path/'campaign';bank=campaign/'assessment'/'bank';bank.mkdir(parents=True)
    receipt=bank/'receipt.json';receipt.write_text('{"status":"completed"}')
    spec_path=campaign/'spec.json';spec_path.write_text('{}')
    spec={'phase_epoch':phase,'global_step':6000+12*phase,'variant':'ScaleRepairedContinuation',
          'bank_output':str(bank),'checkpoint':{'sha256':'checkpoint'}}
    review={'epoch':phase,'task_scope':'both tasks excluded','bank_receipt_sha256':reference(receipt)['sha256'],
            'prepared_sha256':reference(spec_path)['sha256'],'checkpoint_sha256':'checkpoint'}
    review_path=campaign/'assessment'/'review.json';review_path.write_text(json.dumps(review))
    monkeypatch.setattr(driver,'validate_fork',lambda *_:None)
    output=campaign/'assessment'/'task_decision.json'
    driver.authorize_endpoint(campaign,spec_path,spec,review_path,bank,decision,'Selected terminal assessment.',output)
    review['_sha256']=reference(review_path)['sha256'];check_terminal(json.loads(output.read_text()),review,reference(receipt)['sha256'])
    with pytest.raises(ValueError,match='terminal choice'):
        driver.authorize_endpoint(campaign,spec_path,spec,review_path,bank,'endpoint_at_epoch_225' if phase!=225 else 'endpoint_at_epoch_500','wrong phase',campaign/'assessment'/'bad.json')
