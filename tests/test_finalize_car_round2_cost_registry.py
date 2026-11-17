import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


finalizer = module('car_finalize', ROOT/'scripts/analysis/finalize_car_round2_cost_registry.py')
assembler = module('car_assemble', ROOT/'scripts/analysis/assemble_car_round2_cost_manifest.py')


def test_terminal_graph_round_trip_and_guards(tmp_path):
    from phycoflow_reconstruction.evaluation.round2_costs import reconcile

    counter = 0
    ledger = {'schema':'car_gpu_ledger_v1', 'active':{}, 'closed':[]}
    outputs = []
    def write(name, data):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return finalizer.binding(path)
    def gpu(arm, epoch, device, kind):
        nonlocal counter
        counter += 1
        start = float(counter*10)
        token = f'token-{counter}'
        ledger['closed'].append({'token':token, 'gpu_uuid':device, 'start_unix':start, 'end_unix':start+2, 'seconds':2.})
        doc = {'schema':'car_round2_native_bank_v1' if kind == 'generate' else 'car_round2_deployment_timing_v1', 'status':'completed', 'token':token, 'start_unix':start, 'end_unix':start+2,
               'wall_seconds':2., 'arm':arm, 'epoch':epoch}
        doc['gpu_uuid']=device
        name = f'out/{token}/receipt.json' if kind == 'generate' else f'out/{token}.json'
        ref = write(name, doc); outputs.append(ref)
        outarg = str(Path(ref['path']).parent) if kind == 'generate' else ref['path']
        return outarg
    def child(name, kind, cmd):
        return write(f'children/{name}.json', {'schema':'car_round2_assessment_sequence_child_v1',
            'name':name, 'kind':kind, 'returncode':0, 'independent_cpu':kind=='cpu',
            'start_unix':1., 'end_unix':2., 'wall_seconds':1., 'command':cmd,
            'wall_accounting':'encloses the GPU lease; do not add both' if kind=='gpu' else 'CPU child wall includes its scoring receipt; do not add both'})
    initial = []
    for scope, owner in [('source_preparation','all_shared'),('bank_selection','all_shared'),
                         ('qualification','all_shared'),('allocation','residual_shared')]:
        ref=write(f'initial/{scope}.json',{'status':'completed','independent_cpu':True,'start_unix':1.,'end_unix':2.,'wall_seconds':1.})
        initial.append({'kind':'cpu','scope':scope,'owner':owner,'receipt':ref['path'],'independent':True})
    base = write('base.json', {'schema':'car_round2_cost_registry_v1','registrations':initial, 'open_items':['historical gap'],
                                'receipt_inventory_globs':[str(tmp_path/'empty_inventory'/'*.json')],
                                'allocator_disposition':{'ffm':'uniform','senseiver':'uniform'}})
    formal = []
    formal_starts = {}
    for family, arms in [('ffm',finalizer.ARMS[:1]), ('ffm',finalizer.ARMS[1:2]), ('senseiver',finalizer.ARMS[2:])]:
        records = []
        for arm in arms:
            nonlocal_dummy = gpu(arm, 1000, 'formal-device', 'generate')
            del nonlocal_dummy
            original = outputs.pop(); old = json.loads(Path(original['path']).read_text())
            formal_starts[arm] = old['start_unix']
            launch = write(f'formal/{arm}_launch.json', {'token':old['token'], 'started_unix':old['start_unix'], 'gpu_uuid':'formal-device'})
            run=tmp_path/'runs'/'formal'/arm/'completed';run.mkdir(parents=True)
            completion = write(f'formal/{arm}_completion.json', {'run_dir':str(run),'token':old['token'], 'finished_unix':old['end_unix'], 'status':'completed','global_step':12000})
            records.append(write(f'formal/{arm}_child.json', {'schema':'car_round2_formal_child_v1','arm':arm,'returncode':0,
                'launch_receipt':launch['path'],'launch_sha256':launch['sha256'],'completion_receipt':completion['path'],
                'completion_sha256':completion['sha256'],'wall_accounting':'encloses the child GPU lease; do not add both'}))
        launch = write(f'formal/group{len(formal)}_launch.json', {'schema':'launch'})
        formal.append(write(f'formal/group{len(formal)}.json', {'schema':'car_round2_formal_group_v1','status':'completed',
            'failure':None,'family':family,'launch_receipt':launch['path'],'launch_sha256':launch['sha256'], 'children':records,
            'wall_accounting':'encloses child leases; do not add as independent GPU cost'}))
    freeze=write('freeze.json',{'schema':'car_round2_freeze_receipt_v1','manifest':{'status':'frozen'}})
    groups=[]
    devices={'ffm':'profile-ffm','senseiver':'profile-senseiver'}
    for arm in finalizer.ARMS:
        family='ffm' if arm.startswith('FFM_') else 'senseiver'
        for epoch in finalizer.EPOCHS:
            stem=f'train_{arm}_{epoch:03d}'
            launch=write(f'train/{stem}_launch.json',{'schema':'car_round2_assessment_sequence_launch_v1','train_batch_binding':{'arm':arm,'checkpoints':{str(epoch):{}}}})
            # Generation can intentionally use a different device from profiling.
            gen=gpu(arm,epoch,'generation-device','generate'); prof=gpu(arm,epoch,devices[family],'profile')
            children=[child(stem+'_generate','gpu',['generate','--gpu-uuid','generation-device','--output',gen]),
                      child(stem+'_deployment','gpu',['profile-deployment','--gpu-uuid',devices[family],'--output',prof]),
                      child(stem+'_score','cpu',['score'])]
            g=write(f'train/{stem}.json',{'schema':'car_round2_assessment_sequence_v1','status':'completed','phase':'train-batch',
                'launch_receipt':launch['path'],'launch_sha256':launch['sha256'],'children':children,
                'wall_accounting':'group encloses GPU leases and independent CPU score; nonadditive'})
            groups.append({'arm':arm,'epoch':epoch,'group':g['path'],'group_sha256':g['sha256']})
    train=write('train/result.json',{'schema':'car_round2_train_checkpoint_successor_result_v1','status':'completed','groups':groups,'gpu_devices':devices,'freeze_sha256':freeze['sha256']})
    nominations=[write(f'nominations/{family}.json',{'status':'completed','returncode':0,'start_unix':1.,'end_unix':2.,'command':['--family',family]}) for family in devices]
    barrier=write('validation/barrier.json',{'status':'passed','train_result':train['path'],'train_result_sha256':train['sha256'],'nominations':nominations})
    families=[]
    for family, device in devices.items():
        launch=write(f'validation/{family}_launch.json',{'schema':'car_round2_validation_successor_launch_v1','family':family,'gpu_uuid':device})
        children=[child(f'{family}_normalizer','cpu',['normalizer'])]
        # Names are preserved exactly as in the sequencer.
        doc=json.loads(Path(children[0]['path']).read_text());doc['name']='freeze_source_reserved_normalizer';children[0]=write(f'validation/{family}_normalizer.json',doc)
        out=gpu(None,None,device,'generate');children.append(child('validation_source_generate_'+family,'gpu',['generate','--gpu-uuid',device,'--output',out]))
        doc=json.loads(Path(children[-1]['path']).read_text());doc['name']='validation_source_generate';children[-1]=write(f'validation/{family}_source.json',doc)
        for arm in finalizer.ARMS:
            if not arm.lower().startswith(family):continue
            for epoch in finalizer.EPOCHS:
                name=f'validation_{arm}_{epoch:03d}'
                out=gpu(arm,epoch,device,'generate')
                children.extend([child(name+'_generate','gpu',['generate','--gpu-uuid',device,'--output',out]),child(name+'_score','cpu',['score'])])
        g=write(f'validation/{family}.json',{'schema':'car_round2_validation_successor_v11','status':'completed','failure':None,
            'launch_receipt':launch['path'],'launch_sha256':launch['sha256'],'children':children,
            'wall_accounting':'Encloses GPU lease and independent CPU scores; group wall is nonadditive.'})
        families.append({'family':family,**g})
    validation=write('validation/result.json',{'schema':'car_round2_validation_successor_result_v1','status':'completed',
                         'barrier':barrier['path'],'barrier_sha256':barrier['sha256'],'families':families,'formal_freeze_sha256':freeze['sha256']})
    names=['train_checkpoint_successor_v13_job.json','train_checkpoint_successor_v13_ffm_cpu_job.json','train_checkpoint_successor_v13_senseiver_cpu_job.json','train_checkpoint_successor_v13_ffm_prefetch_cpu_job.json','validation_successor_v11_job.json','validation_ffm_successor_v11_coordinator.json','validation_senseiver_successor_v11_coordinator.json']
    controllers=[]
    for name in names:
        doc={'schema':'car_round2_independent_cpu_job_v1','status':'completed','independent_cpu':True,'start_unix':1.,'end_unix':2.,'wall_seconds':1.,'blocked_wait_seconds':0.,'nested_child_wall_seconds':0.,'active_wall_seconds':1.}
        root = name in ('train_checkpoint_successor_v13_job.json','validation_successor_v11_job.json')
        if root:
            doc={'schema':'car_round2_cpu_coordination_v1','status':'completed','controller_active_wall_independent_cpu':True,
                 'start_unix':1.,'end_unix':2.,'wall_seconds':1.,'blocked_wait_seconds':0.,
                 'nomination_child_wall_seconds':0.,'parallel_family_enclosure_seconds':0.,'controller_active_wall_seconds':1.}
        elif name.startswith('validation_'):
            doc={'schema':'car_round2_cpu_coordination_v1','status':'completed','independent_cpu':True,
                 'start_unix':1.,'end_unix':2.,'wall_seconds':1.,'normalizer_child_wait_seconds':0.,
                 'gpu_child_wait_seconds':0.,'score_parent_wait_seconds':0.,'active_wall_seconds':1.}
        owner='ffm_shared' if '_ffm_' in name else 'senseiver_shared' if '_senseiver_' in name else 'all_shared'
        ref=write('controllers/'+name,doc);controllers.append({'receipt':ref,'owner':owner,'scope':'campaign_setup','independent':True,'active_wall_field':'controller_active_wall_seconds' if root else 'active_wall_seconds'})
    evidence=write('evidence.json',{'formal_freeze':freeze,'output_receipts':outputs,'independent_cpu_jobs':controllers})
    clocks={}
    for arm in finalizer.ARMS:
        epochs = {}
        for e in finalizer.EPOCHS:
            ck=write(f'checkpoints/{arm}_{e}.json',{'value':arm,'epoch':e})
            epochs[str(e)]={'schema':'checkpoint_clock_v1','epoch':e,'global_step':12*e,
                            'checkpoint_path':ck['path'],'checkpoint_sha256':ck['sha256'],
                            'recorded_unix':formal_starts[arm]+e/1000*1.8}
        status=write(f'clocks/{arm}_status.json',{'status':'completed','global_step':12000})
        journal=tmp_path/f'clocks/{arm}_journal.jsonl'
        journal.write_text(''.join(json.dumps(v)+'\n' for v in epochs.values()))
        clocks[arm]=write(f'clocks/{arm}.json',{'schema':'car_round2_checkpoint_clock_index_v1','arm':arm,'run':str(tmp_path/'runs'/'formal'/arm/'completed'),'epochs':epochs,
            'journal_path':str(journal),'journal_sha256':finalizer.digest(journal),
            'status_path':status['path'],'status_sha256':status['sha256']})
    result=finalizer.finalize(tmp_path,base,formal,train,validation,clocks,evidence)
    assert 'historical gap' in result['open_items']
    manifest=assembler.assemble(tmp_path,result,ledger)
    assert len(manifest['gpu_jobs']) == 4+48+26
    assert len(manifest['cpu_jobs']) == 24+26+2+7+4
    # Simulate later explicit resolution of documented inventory/gap items; the finalizer never clears them.
    assert manifest['assembly_missing']
    manifest['assembly_missing'] = []
    report=reconcile(campaign=tmp_path,ledger=ledger,manifest=manifest,family='ffm')
    assert len(report['allocated_jobs']) == len(manifest['gpu_jobs'])+len(manifest['cpu_jobs'])
    full = finalizer.ARMS[0]
    extras = manifest['checkpoint_extras'][full]['100']
    profile = [j for j in manifest['gpu_jobs'] if j['scope']=='deployment' and j['allocations'].get(full)==1. and 100 in j.get('checkpoint_epochs', [])]
    assert len(profile) == 1 and profile[0]['id'] in extras
    from phycoflow_reconstruction.evaluation.round2_costs import _checkpoint_cost_bounds
    own = [j for j in report['allocated_jobs'] if j['weights'].get(full,0.) > 0]
    without = [j for j in own if j['id'] != profile[0]['id']]
    cutoff = formal_starts[full]+0.18
    assert len(report['arms'][full]['checkpoint_cost_bounds_seconds']) == 6
    assert '100' in report['arms'][full]['checkpoint_cost_bounds_seconds'], [x for x in report['missing'] if 'checkpoint evidence' in x]
    included = report['arms'][full]['checkpoint_cost_bounds_seconds']['100']
    reduced = _checkpoint_cost_bounds(without, [x for x in extras if x != profile[0]['id']], full, cutoff)
    assert included == [reduced[0]+2., reduced[1]+2.]
    with pytest.raises(ValueError,match='duplicate'):
        finalizer.finalize(tmp_path,base,[formal[0],formal[0],formal[2]],train,validation,clocks,evidence)
    train_doc=json.loads(Path(train['path']).read_text());train_doc['status']='failed';bad=write('train/bad.json',train_doc)
    with pytest.raises(ValueError,match='TRAIN root'):
        finalizer.finalize(tmp_path,base,formal,bad,validation,clocks,evidence)
    missing = write('missing-evidence.json',{'formal_freeze':freeze,'output_receipts':outputs[1:],'independent_cpu_jobs':controllers})
    with pytest.raises(ValueError,match='unbound'):
        finalizer.finalize(tmp_path,base,formal,train,validation,clocks,missing)
    first = Path(groups[0]['group']); original = json.loads(first.read_text())
    original['children'][0]['sha256'] = '0'*64
    broken = write('train/broken.json',original)
    from phycoflow_reconstruction.evaluation.round2_costs import validate_excluded_enclosure
    with pytest.raises(ValueError,match='binding changed'):
        validate_excluded_enclosure(json.loads(Path(broken['path']).read_text()))
