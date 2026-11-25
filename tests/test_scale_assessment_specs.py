"""Bounded assessment-spec checks without inference or training."""
import json
from pathlib import Path
import pytest
import torch
import yaml
from phycoflow_reconstruction.training.run_store import config_digest
from scripts.qualification import car_scale_assessment_specs as helper


def test_fork_spec_rejects_wrong_clock_and_snapshots_manifest(tmp_path,monkeypatch):
    campaign=tmp_path/'scale_alignment_followup';(campaign/'configs').mkdir(parents=True);(campaign/'assessment').mkdir();run=campaign/'runs/formal/FFM_SSS_Unchanged_Scale_Continuation/run001';(run/'checkpoints').mkdir(parents=True)
    parent=tmp_path/'parent';(parent/'checkpoints').mkdir(parents=True);parent_ckpt=parent/'checkpoints/epoch_500.pt';parent_ckpt.write_bytes(b'parent')
    snap=tmp_path/'round_6/assessment/Epoch500_Validation100_J4_20261125T042223Z/snapshot/snapshot.json';snap.parent.mkdir(parents=True);snap.write_text(json.dumps({'run':str(parent),'checkpoint':helper.reference(parent_ckpt)}))
    cfg={'optimization':{'epochs':500}};(run/'resolved_config.yaml').write_text(yaml.safe_dump(cfg))
    origin={'parent_run':str(parent),'parent_checkpoint_sha256':helper.hash_file(parent_ckpt),'absolute_step':6000}
    manifest={'run_id':run.name,'fork_origin':origin,'config_sha256':config_digest(cfg)};(run/'run_manifest.json').write_text(json.dumps(manifest))
    checkpoint=run/'checkpoints/epoch_100.pt';state={'fork_origin':origin,'global_step':7199,'config_sha256':config_digest(cfg),'source_hashes':{'checkpoint':'source'}};torch.save(state,checkpoint)
    source=tmp_path/'source.json';source.write_text('{}');panel=tmp_path/'panel.json';panel.write_text('{}');start=tmp_path/'start.json';start.write_text('{}');code=tmp_path/'code';code.mkdir();(code/'execution_receipt.json').write_text('{}')
    monkeypatch.setattr(helper,'ROOT',code);monkeypatch.setattr(helper,'verify_execution',lambda _:None);monkeypatch.setattr(helper,'check_cached_rows',lambda *_:None)
    monkeypatch.setattr(helper,'base_context',lambda _:({'source_receipt':source,'panel_path':panel,'source':{'source_sha256':'source'}},start,{}))
    output=campaign/'configs/fork_spec.json';bank=campaign/'assessment/bank'
    with pytest.raises(ValueError,match='lineage'):helper.fork_spec(campaign,run,100,run.parent.name,bank,output)
    assert not output.exists()
    state['global_step']=7200;torch.save(state,checkpoint)
    result=helper.fork_spec(campaign,run,100,run.parent.name,bank,output)
    assert result['global_step']==7200 and result['run_manifest']['path']!=str((run/'run_manifest.json').resolve())
    (run/'run_manifest.json').write_text('{"changed":true}')
    assert helper.checked(result['run_manifest']).is_file()


def test_three_aligned_scale_paired_values_and_order_guard(tmp_path):
    campaign=tmp_path/'campaign';(campaign/'assessment').mkdir(parents=True)
    definitions=helper.DEFINITIONS
    records=[]
    for i in range(100):
        records.append({'sample_id':str(i),'models':{'SOURCE':{'definitions':{k:1. for k in definitions}},
                    'FFM_SSS_Unchanged_Scale_Continuation':{'definitions':{k:1.2 for k in definitions}}}})
    units=tmp_path/'units.json';units.write_text(json.dumps({'frozen_train_scale_normalizer':0.5}))
    doc={'status':'completed','split':'validation','members':4,'sample_ids':[str(i) for i in range(100)],'records':records,'spec':helper.reference(units)};path=tmp_path/'decomposition.json';path.write_text(json.dumps(doc));out=campaign/'assessment/comparison.json'
    result=helper.scale_compare(campaign,path,0.5,[('SOURCE','FFM_SSS_Unchanged_Scale_Continuation')],out)
    values=result['comparisons']['FFM_SSS_Unchanged_Scale_Continuation_minus_SOURCE']
    assert set(values)==set(definitions) and all(v['mean']==pytest.approx(.4) for v in values.values())
    doc['records'][0]['sample_id']='other';path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='matched VALIDATION100'):helper.scale_compare(campaign,path,.5,[('SOURCE','FFM_SSS_Unchanged_Scale_Continuation')],campaign/'assessment/bad.json')
