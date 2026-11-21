import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/posttraining/review_car_round4_teacher_cost.py'
spec=importlib.util.spec_from_file_location('teacher_cost_review',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_warm_projection_uses_matched_five_and_all_lease_cpu_overhead():
    history=[{'epoch':i,'elapsed_seconds':100+i,'cpu_seconds':90+i} for i in range(1,9)]
    writes=[2,3,4,5,6,7]
    result=module.calculate(history,writes,8,650,1000,800,11058,15061,0)
    overhead=(650-sum(x['cpu_seconds'] for x in history[2:]))/6
    assert result['window']['epochs']==[4,5,6,7,8]
    assert result['training_window_epochs']==[4,5,6,7,8]
    assert result['window']['checkpoint_write_wall_seconds']==[3,4,5,6,7]
    assert result['window']['cpu_persistence_overhead_per_epoch_seconds']==pytest.approx(overhead)
    assert result['projected_teacher_gpu_seconds']==pytest.approx(1000+60+1.2*92*(106+5)+1599)
    assert result['projected_teacher_cpu_worker_seconds']==pytest.approx(800+60+1.2*92*(96+overhead)+1698)
    assert result['teacher_gpu_cap_seconds']==min(32400,35317-11058,86400-15061)
    assert result['projected_candidate_gpu_seconds']==pytest.approx(11058+result['projected_teacher_gpu_seconds'])
    assert 'admitted' not in result


def test_partial_warm_window_and_nonfinite_costs_rejected():
    rows=[{'epoch':i,'elapsed_seconds':1,'cpu_seconds':1} for i in range(1,6)]
    with pytest.raises(ValueError): module.calculate(rows,[1]*4,5,10,10,10,1,2,0)
    with pytest.raises(ValueError): module.calculate(rows,[1]*5,5,10,float('nan'),10,1,2,0)
    with pytest.raises(ValueError): module.calculate(rows,[1]*5,5,10,10,10,1,2,18001)
    rows[2]['epoch']=7
    with pytest.raises(ValueError): module.calculate(rows,[1]*5,5,10,10,10,1,2,0)


def test_active_lease_rejected_before_any_teacher_artifact_read(tmp_path):
    ops=tmp_path/'operations'; ops.mkdir()
    (ops/'gpu_ledger.json').write_text(json.dumps({'schema':'car_gpu_ledger_v1','active':{'current':{}},'closed':[]}))
    with pytest.raises(ValueError,match='every GPU lease to be closed'):
        module.review(tmp_path,SimpleNamespace())
    assert not (tmp_path/'assessment').exists()


def test_closed_fixture_produces_immutable_review(tmp_path, monkeypatch):
    import hashlib
    import torch
    c=tmp_path
    def put(path, value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
        return path
    manifest=put(c/'configs/train_manifest.json',{'split':'train'})
    mh=module.sha(manifest)
    initial=put(c/'operations/feasibility_gate.json',{'schema':'car_round4_feasibility_gate_v1','admitted':True,'manifest_sha256':mh})
    put(c/'operations/teacher_launch_forecast.json',{'schema':'car_round4_teacher_launch_forecast_v1','admitted':True,
        'manifest_sha256':mh,'bindings':{'initial_gate':{'sha256':module.sha(initial)},'failed_startups':[]},
        'measured_qualification_gpu_seconds':10,'measured_generation_gpu_seconds':20,
        'remaining_gpu_forecast_seconds':{'teacher':100},'candidate_inclusive_gpu_seconds':500,
        'whole_round_gpu_seconds':700})
    codebase=c/'operations/code'
    content=b'print("teacher")\n'
    files={'scripts/posttraining/run_car_round4_teacher.py':hashlib.sha256(content).hexdigest()}
    capsule_hash=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    capsule=codebase/capsule_hash
    script=capsule/'scripts/posttraining/run_car_round4_teacher.py'
    script.parent.mkdir(parents=True); script.write_bytes(content)
    put(capsule/'execution_receipt.json',{'schema':'car_execution_code_v1','sha256':capsule_hash,'files':files})
    code_receipt=put(c/'configs/execution_code_teacher_resource.json',{'execution_code_root':str(capsule)})
    run=c/'runs/formal/SSS_Empirical_Endpoint_Teacher/Teacher_Initial'
    (run/'checkpoints').mkdir(parents=True)
    (run/'logs').mkdir()
    ident={'train_manifest_sha256':mh,'recording_columns':[{'name':'epoch'}],'accumulation_order':['shape']}
    rows=[{'epoch':i,'elapsed_seconds':10,'cpu_seconds':9} for i in range(1,6)]
    cp=run/'checkpoints/teacher_latest.pt'
    torch.save({'schema':'car_round4_teacher_checkpoint_v1',**ident,'completed_epochs':5,
                'normalizers':{'shape':1},'history':rows},cp)
    (run/'logs/teacher_history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    put(run/'configs/resolved_scientific_config.json',{'schema':'car_round4_teacher_scientific_config_v1',
                                                       'identity':ident,'normalizers':{'shape':1}})
    entries=[{'token':'prep','phase':'generate','seconds':30},
             {'token':'fail','phase':'endpoint_teacher','seconds':12,'cpu_seconds':4},
             {'token':'resume','phase':'endpoint_teacher','seconds':100,'cpu_seconds':60}]
    for e in entries[1:]:
        receipt={'schema':'car_round4_teacher_job_v1','token':e['token'],'manifest_sha256':mh,
                 'status':'paused' if e['token']=='resume' else 'failed', 'wall_seconds':e['seconds'],
                 'code_root':str(capsule),'script_sha256':files['scripts/posttraining/run_car_round4_teacher.py']}
        if e['token']=='resume': receipt.update(checkpoint=str(cp),completed_epochs=5,checkpoint_write_wall_seconds=[1]*5)
        put(c/'operations/teacher_jobs'/f'{e["token"]}.json',receipt)
    put(c/'operations/gpu_ledger.json',{'schema':'car_gpu_ledger_v1','active':{},'closed':entries})
    args=SimpleNamespace(attempt='Teacher_Initial',resumed_token='resume',review_name='Teacher_Warm_Review',
                         reserve_committed_elsewhere=0,execution_code_receipt=code_receipt)
    module.review(c,args)
    review=c/'assessment/teacher_cost_reviews/Teacher_Warm_Review'
    result=json.loads((review/'warm_forecast.json').read_text())
    assert result['closed_teacher_gpu_seconds']==112 and result['closed_teacher_cpu_worker_seconds']==64
    assert result['teacher_checkpoint_sha256']==module.sha(cp)
    assert result['training_window_epochs']==[1,2,3,4,5]
    assert result['window']['checkpoint_write_wall_seconds']==[1]*5
    assert result['bindings']['continuation_code_receipt']['sha256']==module.sha(code_receipt)
    assert result['review_timer']['process_cpu_seconds'] >= 0
    assert json.loads(initial.read_text())['admitted']
    # Exercise the actual shared admission and stop validators against this
    # forecast's field contract, without authoring a campaign decision.
    close_spec=importlib.util.spec_from_file_location('closeout_test',path.parent/'closeout_car_round4_cost.py')
    close=importlib.util.module_from_spec(close_spec); close_spec.loader.exec_module(close)
    stop_forecast={**result,'projected_teacher_gpu_seconds':result['teacher_gpu_cap_seconds']+1}
    stop={'schema':'car_round4_teacher_budget_stop_v1','decision':'stop_cost_infeasible','train_manifest_sha256':mh,
          'completed_epochs':5,'teacher_gpu_cap_seconds':result['teacher_gpu_cap_seconds'],'teacher_cpu_cap_seconds':21600,
          'checkpoint':result['bindings']['checkpoint'],'history':result['bindings']['history'],
          'projected_teacher_gpu_seconds':stop_forecast['projected_teacher_gpu_seconds'],
          'projected_teacher_cpu_worker_seconds':result['projected_teacher_cpu_worker_seconds']}
    assert close.validate_budget_stop(stop,stop_forecast,mh)==5
    teacher_spec=importlib.util.spec_from_file_location('teacher_validator_test',path.parent/'run_car_round4_teacher.py')
    teacher=importlib.util.module_from_spec(teacher_spec); teacher_spec.loader.exec_module(teacher)
    monkeypatch.setattr(teacher,'validate_driver_amendment',lambda origin,execution: None)
    forecast_path=review/'warm_forecast.json'
    fields=('projected_teacher_gpu_seconds','projected_teacher_cpu_worker_seconds','projected_candidate_gpu_seconds','projected_whole_gpu_seconds')
    # A synthetic larger but valid allocation exercises the resource validator.
    allocation=15000
    admission={'schema':'car_round4_teacher_resource_admission_v1','admitted':True,'attempt':args.attempt,
               'train_manifest_sha256':mh,'origin_code_root':str(capsule),'execution_code_root':str(capsule),
               'max_epochs':100,'teacher_cpu_cap_seconds':21600,'whole_gpu_cap_seconds':86400,'candidate_gpu_cap_seconds':35317,
               'teacher_gpu_cap_seconds':allocation,'reserve_allocated_seconds':allocation-14400,'reserve_available_seconds':18000,
               'completed_epochs':5,**{k:result[k] for k in fields},'checkpoint':result['bindings']['checkpoint'],
               'history':result['bindings']['history'],'scientific_config':result['bindings']['scientific_config'],
               'forecast':module.bind(forecast_path),'ledger':module.bind(c/'operations/gpu_ledger.json')}
    admission_path=put(c/'operations/synthetic_admission.json',admission)
    assert teacher.validate_resource_admission(admission_path,c,mh,args.attempt,capsule,capsule,cp,
           run/'configs/resolved_scientific_config.json',run/'logs/teacher_history.jsonl')['admitted']
    old=module.sha(review/'teacher_history.jsonl')
    with (run/'logs/teacher_history.jsonl').open('a') as f: f.write('later append\n')
    assert module.sha(review/'teacher_history.jsonl')==old
    with pytest.raises(FileExistsError): module.review(c,args)
