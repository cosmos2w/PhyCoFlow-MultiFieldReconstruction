import json

import pytest

from scripts.posttraining import score_car_round7_review as lane


def test_commands_are_two_sequential_cached_cpu_scores(tmp_path):
    context={split:{'spec_path':tmp_path/f'{split}_spec.json','baseline':tmp_path/f'{split}_baseline.json',
                    'source_selected':tmp_path/f'{split}_source.json',
                    'banks':{'native':tmp_path/f'{split}_native.json','selected':tmp_path/f'{split}_selected.json'}}
             for split in lane.SPLITS}
    plan=lane.commands(context,tmp_path/'output',python='/env/python',code='/frozen/code')
    assert [item['split'] for item in plan]==['train','validation']
    for item in plan:
        cmd=item['command']
        assert cmd[:3]==['/env/python','/frozen/code/scripts/posttraining/assess_car_round7.py','score']
        assert cmd[cmd.index('--threads')+1]=='2'
        assert '--device' not in cmd and 'generate' not in cmd and 'tasks' not in cmd
        assert cmd[cmd.index('--baseline')+1]==str(context[item['split']]['baseline'])


@pytest.mark.parametrize('epoch',[100,500])
def test_ready_binds_each_bank_to_exact_split_spec_and_checkpoint(tmp_path,monkeypatch,epoch):
    def save(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return lane.ref(path)
    campaign=tmp_path/'campaign';assessment=campaign/'assessment';assessment.mkdir(parents=True)
    run=tmp_path/'run';(run/'checkpoints').mkdir(parents=True)
    checkpoint=run/f'checkpoints/epoch_{epoch:03d}.pt';checkpoint.write_bytes(b'checkpoint');checkpoint_ref=lane.ref(checkpoint)
    saved_config=save(run/'resolved_config.json',{'config':'frozen'})
    specs={};banks=[]
    for split,panel_name in (('train','TRAIN800'),('validation','VALIDATION100')):
        source_ref={'path':'native','sha256':f'{split}_native_source'}
        spec={'split':split,'campaign_root':str(campaign),'reporting_epoch':epoch,'checkpoint':checkpoint_ref,
              'source_native_receipt':source_ref,'run_config':saved_config}
        spec_ref=save(assessment/f'{split}_spec.json',spec);specs[split]=spec_ref
        selected_ref=save(assessment/f'{split}_source_selected.json',{'split':split,'support':'selected'})
        save(assessment/f'SOURCE_Baseline_{panel_name}_v1'/'baseline.json',{'schema':'car_round7_source_baseline_v1',
            'status':'completed','binding':{'split':split},'sample_ids':[f'{split}_id'],
            'source_selected':selected_ref,'source_native':source_ref,'files_sha256':['hash']})
        for support in lane.SUPPORTS:
            banks.append(save(assessment/f'{split}_{support}.json',{'status':'completed','split':split,'support':support,
                'checkpoint_sha256':checkpoint_ref['sha256'],'run_config_sha256':saved_config['sha256'],'spec_sha256':spec_ref['sha256']}))
    ready={'schema':'car_round7_review_arm_ready_v1','arm':lane.ARMS[0],'checkpoint':checkpoint_ref,'specs':specs,'banks':banks}
    attempt=assessment/lane.ARMS[0]
    training={'schema':'car_round7_training_result_v1','arm':lane.ARMS[0],
        'target_epoch':epoch,'target_reached':True,'global_step':12*epoch,'run_dir':str(run),'saved_config':saved_config}
    save(attempt/'training_result.json',training)
    ready_ref=save(attempt/'ready.json',ready)
    monkeypatch.setattr(lane,'validate_spec',lambda spec:({'sample_ids':[spec['split']+'_id']},{}))
    monkeypatch.setattr(lane,'baseline_binding',lambda spec:{'split':spec['split']})
    monkeypatch.setattr(lane,'verify_bank',lambda path,spec,panel,source,support:(path,{}))
    prepared,_,contexts=lane.prepare(ready_ref['path'])
    assert prepared['arm']==lane.ARMS[0] and set(contexts['train']['banks'])=={'native','selected'}
    assert contexts['train']['epoch']==epoch
    if epoch==500:
        training['global_step']=1200;save(attempt/'training_result.json',training)
        with pytest.raises(ValueError,match='canonical endpoint'):
            lane.prepare(ready_ref['path'])
        training['global_step']=6000;save(attempt/'training_result.json',training)
        last=run/'checkpoints/last.pt';last.symlink_to(checkpoint)
        ready['checkpoint']={'path':str(last),'sha256':checkpoint_ref['sha256']};save(attempt/'ready.json',ready)
        with pytest.raises(ValueError,match='canonical endpoint'):
            lane.prepare(ready_ref['path'])
        ready['checkpoint']=checkpoint_ref;save(attempt/'ready.json',ready)
    changed=json.loads((assessment/'train_selected.json').read_text());changed['spec_sha256']='other'
    banks[1]=save(assessment/'train_selected.json',changed);ready['banks']=banks
    save(attempt/'ready.json',ready)
    with pytest.raises(ValueError,match='exact split spec'):
        lane.prepare(ready_ref['path'])
