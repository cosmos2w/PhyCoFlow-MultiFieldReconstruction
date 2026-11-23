import json
from scripts.analysis.collect_car_round5_costs import collect


def test_costs_include_failed_and_resumed_leases_without_double_counting(tmp_path):
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path
    def jsonlines(path,rows):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    campaign=tmp_path/'campaign';arm='Native_GT_Continuation'
    write(campaign/'configs/formal_freeze.json',{'schema':'car_round5_freeze_v1','active_arms':[arm],'forecast':{'planned_seconds':999}})
    leases=[('prep','startup_initial',10,0,10,'failed',[]),
            ('first','formal_training',20,10,30,'completed',['driver','--arm',arm]),
            ('resumed','formal_training',30,30,60,'completed',['outer',['driver','--arm',arm]]),
            ('assessment','assessment',5,60,65,'completed',['assessment'])]
    rows=[]
    for token,phase,seconds,start,end,status,command in leases:
        rows.append({'token':token,'phase':phase,'seconds':seconds,'start_unix':start,'end_unix':end})
        write(campaign/'operations/completions'/f'{token}.json',{'token':token,'phase':phase,'seconds':seconds,
             'start_unix':start,'end_unix':end,'status':status,'command':command,
             'child_user_cpu_seconds':2 if token=='first' else 3,
             'child_system_cpu_seconds':.5 if token=='first' else 1})
    write(campaign/'operations/gpu_ledger.json',{'active':{},'closed':rows})
    run=tmp_path/'run'
    jsonlines(run/'logs/history.jsonl',[{'epoch':1,'epoch_complete':True},{'epoch':2,'epoch_complete':True}])
    jsonlines(run/'logs/epoch_timing.jsonl',[{'epoch':1,'step':2,'wall_seconds':10,'train_seconds':8,'callback_and_reporting_seconds':2},
        {'epoch':2,'step':4,'wall_seconds':10,'train_seconds':8,'callback_and_reporting_seconds':2}])
    jsonlines(run/'logs/direct_updates.jsonl',[{'step':i,'update_seconds':3,'data_wait_seconds':.5,'update_accepted':i!=2} for i in range(1,5)])
    cpu=write(tmp_path/'cpu.json',{'start_unix':64,'end_unix':75,'cpu_seconds':9,'status':'completed'})
    inputs=write(tmp_path/'inputs.json',{'schema':'car_round5_cost_inputs_v1','arms':{arm:{'run_dir':str(run)}},
        'cpu_jobs':[{'path':str(cpu),'kind':'assessment','inside_gpu_lease':False}],
        'assessment_windows':[{'gpu_tokens':['assessment'],'cpu_paths':[str(cpu)]}]})
    result=collect(campaign,inputs);training=result['training'][arm]
    assert result['campaign_gpu_seconds']==65 and result['failed_attempt_gpu_seconds']==10
    assert training['occupied_gpu_seconds']==50 and training['update_work_seconds']==12
    assert training['inside_lease_child_user_cpu_seconds']==5
    assert training['inside_lease_child_system_cpu_seconds']==1.5
    assert training['inside_lease_child_cpu_seconds']==6.5
    assert training['data_wait_seconds']==2 and training['data_wait_and_loop_overhead_seconds']==4
    assert training['callback_and_reporting_seconds']==4 and training['seconds_per_epoch']==10
    assert result['assessment']=={'gpu_seconds':5,'cpu_seconds':9,'wall_seconds':15,'queue_tail_seconds':10}
    assert result['independent_cpu_seconds'] is None and result['forecast_is_measured'] is False


def test_cpu_endpoint_wrapper_rusage_without_absolute_clocks(tmp_path):
    from scripts.analysis.collect_car_round5_costs import cpu_jobs
    path=tmp_path/'cpu_endpoint_query_completion.json'
    path.write_text(json.dumps({'status':'completed','user_cpu_seconds':3.5,'system_cpu_seconds':1.25,'wall_seconds':9}))
    result=cpu_jobs([{'path':str(path),'kind':'independent','inside_gpu_lease':False}])
    assert result[0]['cpu_seconds']==4.75
    assert result[0]['start_unix'] is None and result[0]['end_unix'] is None
