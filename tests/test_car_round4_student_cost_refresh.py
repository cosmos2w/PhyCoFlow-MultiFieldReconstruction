import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

path=Path(__file__).resolve().parents[1]/'scripts/posttraining/refresh_car_round4_student_cost.py'
spec=importlib.util.spec_from_file_location('student_cost_refresh',path)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_native_first_epoch_stream_uses_independent_generators():
    seed=47
    rows=list(m.first_epoch_stream(seed,8000,torch))
    assert len(rows)==12 and all(len(x[0])==len(x[2])==64 for x in rows)
    expected=torch.randperm(8000,generator=torch.Generator().manual_seed(seed+17)).tolist()
    assert sum((x[0] for x in rows),[])==expected[:768]
    queries=torch.Generator().manual_seed(seed+37)
    atoms=torch.Generator().manual_seed(seed+59)
    for _,q,a in rows:
        assert q==int(torch.randint(0,2**31-1,(),generator=queries))
        assert torch.equal(a,torch.randint(0,4,(64,),generator=atoms))


def test_sparse_correction_is_decoded_once():
    values=torch.arange(24,dtype=torch.float32).reshape(4,1,3,2)
    row={'members_model':values,'normalizer_scale':torch.tensor([2.,4.]),'normalizer_offset':torch.tensor([3.,5.])}
    corrections=torch.zeros((1,4,2,2),dtype=torch.float64)
    corrections[0,2,0]=torch.tensor([1.,2.]); corrections[0,2,1]=torch.tensor([3.,4.])
    support_lookup=torch.tensor([-1,0,1])
    actual=m.decoded_target(row,0,2,torch.tensor([2,0,1]),support_lookup,corrections,torch)
    expected=(values[2,0,[2,0,1]].double()+torch.tensor([[3.,4.],[0.,0.],[1.,2.]],dtype=torch.float64))*row['normalizer_scale']+row['normalizer_offset']
    assert torch.equal(actual,expected)


def test_inclusive_forecast_counts_io_once_and_checks_both_horizons():
    out=m.forecast(prior_gpu=1000,teacher_gpu=4000,update_per_batch=2,qualified_batch_input=1,
                   measured_batch_input=10,measured_target_io=30,assessment_gpu=900,
                   assessment_cpu=500,assessment_wall=2000)
    truth=1.2*(24+12+2); teacher=1.2*(24+12+30+2)
    assert out['student_epoch_forecast_seconds']['dense_truth']==pytest.approx(truth)
    assert out['student_epoch_forecast_seconds']['teacher_targets']==pytest.approx(teacher)
    assert out['review_100']['candidate_inclusive_gpu_seconds']==pytest.approx(1000+4000+60+100*teacher+900)
    assert out['conditional_endpoint_500']['whole_round_gpu_seconds']==pytest.approx(1000+4000+120+500*teacher+900+120+500*truth)
    assert out['review_100']['startup_allowance_per_student_seconds']==60
    assert out['conditional_endpoint_500']['startup_allowance_per_student_seconds']==120
    assert out['review_100']['candidate_twofold_budget_fits'] and not out['conditional_endpoint_500']['candidate_twofold_budget_fits']


def test_active_guard_reads_no_campaign_artifacts(tmp_path):
    ops=tmp_path/'operations'; ops.mkdir()
    (ops/'gpu_ledger.json').write_text(json.dumps({'schema':'car_gpu_ledger_v1','active':{'teacher':{}},'closed':[]}))
    with pytest.raises(ValueError,match='all GPU leases must be closed'):
        m.execute(tmp_path,SimpleNamespace())
    assert sorted(p.name for p in ops.iterdir())==['gpu_ledger.json']


def test_closed_campaign_missing_teacher_checkpoint_does_not_create_receipts(tmp_path):
    def put(path, doc):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(doc))
    ops=tmp_path/'operations'
    put(ops/'gpu_ledger.json',{'schema':'car_gpu_ledger_v1','active':{},'closed':[]})
    manifest=tmp_path/'configs/train_manifest.json'; put(manifest,{'population':8000})
    mh=m.sha(manifest)
    gate=ops/'feasibility_gate.json'; put(gate,{'admitted':True,'manifest_sha256':mh})
    put(ops/'teacher_launch_forecast.json',{'admitted':True,'manifest_sha256':mh,
        'bindings':{'initial_gate':{'sha256':m.sha(gate)}}})
    put(ops/'teacher_qualification.json',{'schema':'car_round4_teacher_qualification_v1','admitted':True,
        'completed_epochs':100,'train_manifest_sha256':mh,'checkpoint':'missing/teacher.pt','teacher_checkpoint_sha256':'x'})
    put(tmp_path/'artifacts/source_bank/receipt.json',{})
    with pytest.raises(FileNotFoundError):
        m.execute(tmp_path,SimpleNamespace())
    assert not (ops/'student_target_io_timing.json').exists()
    assert not (ops/'student_cost_forecast.json').exists()


def test_standalone_timing_fields_match_auxiliary_cost_accounting():
    close_path=path.parent/'closeout_car_round4_cost.py'
    close_spec=importlib.util.spec_from_file_location('closeout_cost_fixture',close_path)
    close=importlib.util.module_from_spec(close_spec); close_spec.loader.exec_module(close)
    auxiliary,clocks,unknown=[],[],[]
    doc={'status':'completed','gpu_used':False,'start_unix':100.,'end_unix':104.,
         'wall_seconds':4.,'process_cpu_seconds':2.}
    close.add_cpu_only_auxiliary(auxiliary,clocks,unknown,scope='student target I/O',ref={},doc=doc,cpu_key='process_cpu_seconds')
    assert auxiliary[0]['process_cpu_seconds']==2 and auxiliary[0]['wall_seconds']==4
    assert not unknown


def test_closed_costs_charge_failed_and_completed_teacher_once(tmp_path):
    ops=tmp_path/'operations'; (ops/'teacher_jobs').mkdir(parents=True); (ops/'failed_launches').mkdir(); (ops/'preparation_jobs').mkdir()
    (ops/'preparation_jobs/prep.json').write_text(json.dumps({'token':'prep','wall_seconds':100}))
    teacher=tmp_path/'teacher.pt'; teacher.write_bytes(b'checkpoint')
    for token,status,seconds in [('failure','failed',12),('success','completed',500)]:
        receipt={'token':token,'manifest_sha256':'manifest','status':status,'wall_seconds':seconds}
        if status=='completed': receipt.update(completed_epochs=100,checkpoint=str(teacher))
        (ops/'teacher_jobs'/f'{token}.json').write_text(json.dumps(receipt))
    failure=ops/'failed_launches/start.json'; failure.write_text(json.dumps({'conservative_gpu_seconds_charge':20}))
    ledger={'closed':[{'token':'prep','phase':'generate','seconds':100},
                      {'token':'failure','phase':'endpoint_teacher','seconds':12,'cpu_seconds':5},
                      {'token':'success','phase':'endpoint_teacher','seconds':500,'cpu_seconds':450}]}
    launch={'measured_qualification_gpu_seconds':0,'measured_generation_gpu_seconds':100,
            'unmeasured_startup_allowance_gpu_seconds':60,
            'bindings':{'failed_startups':[{'path':str(failure),'sha256':m.sha(failure)}]}}
    prior,teacher_cost,clocks,refs,_=m.closed_costs(tmp_path,ledger,launch,'manifest',teacher)
    assert prior==180 and teacher_cost==512 and len(refs)==3
    assert clocks[0]['cpu_seconds'] is None
    (ops/'teacher_jobs/success.json').write_text(json.dumps({'token':'success','manifest_sha256':'manifest','status':'paused','wall_seconds':500}))
    with pytest.raises(ValueError,match='completed terminal teacher lease missing'):
        m.closed_costs(tmp_path,ledger,launch,'manifest',teacher)
