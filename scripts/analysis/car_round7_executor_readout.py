#!/usr/bin/env python3
"""Bind completed Round 7 mature executor epochs into a provisional report adapter."""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]


def ref(path):
    path=Path(path).resolve(strict=True); digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return {'path':str(path),'sha256':digest.hexdigest()}


def jsonlines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def run(campaign, output):
    started=time.time();cpu_started=time.process_time()
    campaign=campaign.resolve(strict=True);output=output.resolve()
    screen=campaign/'runs/qualification/Targeted_Executor_Static_Cache_Screen_v1/receipt.json'
    complete=campaign/'runs/qualification/Balanced_Mature_Executor_Epochs_v1/receipt.json'
    launch=campaign/'operations/Balanced_Mature_Executor_Epochs_v1_launch.json'
    profile_readout=campaign/'assessment/Executor_Static_Cache_Engineering_Readout_v1/engineering_readout.json'
    epoch=json.loads(complete.read_text()); launch_data=json.loads(launch.read_text())
    screen_data=json.loads(screen.read_text()); profile=json.loads(profile_readout.read_text())
    if epoch.get('status')!='completed' or not epoch.get('matched_coupling') or not epoch.get('parent_checkpoint_unchanged'):
        raise ValueError('completed matched qualification with unchanged parent required')
    if len(epoch['attempts'])!=4 or [x['method'] for x in epoch['attempts']]!=['separate','factored','factored','separate']:
        raise ValueError('balanced four-attempt order required')
    capsule=Path(launch_data['code_root'])/'execution_receipt.json'
    capsule_data=json.loads(capsule.read_text())
    if capsule_data['sha256']!=capsule.parent.name:raise ValueError('execution capsule identity differs')
    trace=Path(screen_data['trace']['path']);trace_ref=ref(trace)
    if trace_ref['sha256']!=screen_data['trace']['sha256'] or trace_ref['sha256']!=profile['source_trace']['sha256']:
        raise ValueError('screen and trace evidence differ')
    parent=Path(launch_data['command'][launch_data['command'].index('--parent')+1])/'checkpoints/epoch_500.pt'
    parent_ref=ref(parent)
    if parent_ref['sha256']!=epoch['parent_sha256']:raise ValueError('parent checkpoint changed')
    measurements=[]; evidence=[ref(complete),ref(launch),ref(capsule),parent_ref];couplings=[];journals=[];source_hash=None
    keys=('step','sample_ids','observation_sha256','query_sha256','member_seeds','first_member_generator_state_sha256','field_complement_sha256')
    for attempt in epoch['attempts']:
        run=Path(attempt['run']);paths={
            'history':run/'logs/history.jsonl','timing':run/'logs/epoch_timing.jsonl',
            'coupling':run/'metrics/training_coupling.jsonl','journal':run/'logs/epoch_calibrated_pi_updates.jsonl',
            'definitions':run/'logs/recording_definitions.json','manifest':run/'run_manifest.json',
            'status':run/'status.json','config':run/'resolved_config.yaml'}
        evidence.extend(ref(path) for path in paths.values())
        history=jsonlines(paths['history']);timing=jsonlines(paths['timing']);coupling=jsonlines(paths['coupling']);journal=jsonlines(paths['journal'])
        manifest=json.loads(paths['manifest'].read_text());status=json.loads(paths['status'].read_text())
        if len(history)!=1 or len(timing)!=1 or len(coupling)!=12 or len(journal)!=12: raise ValueError('each run requires one complete epoch and twelve coupled actions')
        h=history[0];t=timing[0]
        if t!=attempt['epoch_timing'] or not t['complete_epoch_duration'] or t['accepted_updates']!=12 or t['timed_updates']!=12:
            raise ValueError('epoch timing and receipt do not agree')
        if status['global_step']!=6012 or status['status']!='budget_paused' or not status['source_immutable_verified']:
            raise ValueError('terminal diagnostic checkpoint/source status differs')
        if h.get('round7_active_field_row_fraction')!=1.0 or h['finite_counts'].get('data_wait_seconds')!=12:
            raise ValueError('active field row and twelve recorded waits required')
        if any(not row.get('accepted',False) for row in coupling):raise ValueError('all coupling actions must be accepted')
        coupling_rows=[{key:row[key] for key in keys} for row in coupling]
        if any(row['step']!=6001+i for i,row in enumerate(coupling_rows)):raise ValueError('unexpected coupling clock')
        couplings.append(coupling_rows);journals.append(journal)
        executor=h['round7_executor'];expected_chunk=1 if attempt['method']=='separate' else 2
        if executor['family_vjp']!=attempt['method'] or executor['vjp_chunk_size']!=expected_chunk or not executor['rollout_checkpointing']:
            raise ValueError('history executor disagrees with effective configuration')
        if attempt['context_cache'] != ('none' if attempt['method']=='separate' else 'static_features'):
            raise ValueError('unexpected cache policy')
        meta=(manifest['source_metadata']['materialized_view_sha256'],manifest['source_hashes']['checkpoint'])
        if source_hash is None:source_hash=meta
        if source_hash!=meta:raise ValueError('SOURCE identities differ across epochs')
        if manifest['fork_origin']['parent_checkpoint_sha256']!=epoch['parent_sha256']:
            raise ValueError('fork parent differs')
        wait_sum=float(h['data_wait_seconds'])*12
        measurements.append({'method':('Separate, repeat 1' if attempt['label']=='Epoch_01_separate' else
            'Factored cached, repeat 1' if attempt['label']=='Epoch_02_factored' else
            'Factored cached, repeat 2' if attempt['label']=='Epoch_03_factored' else 'Separate, repeat 2'),
            'label':attempt['label'],'seconds':t['wall_seconds'],'run_return_seconds':attempt['wall_seconds'],
            'train_seconds':t['train_seconds'],'callback_seconds':t['callback_and_reporting_seconds'],
            'host_data_wait_seconds':wait_sum,'active_field_fraction':h['round7_active_field_row_fraction'],
            'effective_row_chunk_size':expected_chunk,'receipt_chunk_value':attempt['chunk'],
            'context_cache':attempt['context_cache'],'checkpointing':True,
            'peak_allocated_gib':attempt['peak_allocated_bytes']/2**30,
            'peak_reserved_gib':attempt['peak_reserved_bytes']/2**30,'device_name':screen_data['device_name'],
            'references':{name:ref(path) for name,path in paths.items()}})
    if any(rows!=couplings[0] for rows in couplings[1:]):raise ValueError('condition/query/member/complement stream mismatch')
    traces=[]
    for index in range(12):
        pair_differences=[]
        for baseline in (0,3):
            for candidate in (1,2):
                left=journals[baseline][index]['field_violation'];right=journals[candidate][index]['field_violation']
                if set(left)!=set(right):raise ValueError('field names differ')
                pair_differences.append(max(abs(left[name]-right[name]) for name in left))
        traces.append({'action':index+1,'global_step':6001+index,'max_abs_normalized_field_violation_difference':max(pair_differences)})
    baseline=[x for x in measurements if x['label'].endswith('separate')]
    candidate=[x for x in measurements if x['label'].endswith('factored')]
    saving=lambda key:1-statistics.mean(x[key] for x in candidate)/statistics.mean(x[key] for x in baseline)
    measured_breakdown=[{'scope':'Profiled: '+name,'seconds':data['cpu_inclusive_s'],
        'kind':'host inclusive elapsed; instrumented one action; nested, not additive',
        'gpu_kernel_window_intersection_seconds':data['gpu_kernel_intersection_s']}
        for name,data in profile['trace_ranges'].items()]
    adapter={'schema':'car_round7_executor_provisional_adapter_v1','status':'provisional_waiting_for_terminal_ledger',
        'screen_evidence':[ref(screen)],'complete_epoch_evidence':evidence,'trace_evidence':[trace_ref,ref(profile_readout)],
        'complete_epoch_measurements':measurements,'measured_breakdown':measured_breakdown,
        'costs':[{'measure':'Occupied GPU time including failures','value':None,'units':'hours'},
                 {'measure':'CPU scoring and delivery tail','value':None,'units':'seconds'}],
        'statistics':{'mean_run_return_saving_fraction':saving('run_return_seconds'),
            'mean_epoch_span_saving_fraction':saving('seconds'),'matched_coupling_steps':12,
            'normalized_signed_field_violation_max_abs_by_action':traces},
        'source_materialized_view_sha256':source_hash[0],'source_checkpoint_sha256':source_hash[1],
        'capsule_sha256':capsule_data['sha256'],'parent_checkpoint_sha256':epoch['parent_sha256'],
        'interpretation':'Composite-objective mature-state execution comparison only; no learning/held-out claim. Batch waits and callbacks overlap enclosing durations. Float32 nonsmooth sensitivity is a possible, unproven explanation of trajectory divergence; do not claim trajectory equivalence.',
        'cost_limit':'Individual epoch timings only; do not treat as whole-package cost. Terminal lease ledger and failed attempts must be reconciled before final report.'}
    output.mkdir(parents=True,exist_ok=False)
    (output/'engineering_adapter.json').write_text(json.dumps(adapter,indent=2,allow_nan=False)+'\n')
    # Reuse the same figure adapter that will be used by the final report.
    from scripts.analysis.report_car_round7 import engineering_figure
    figures=output/'figures';figures.mkdir();caption=engineering_figure(adapter,figures)
    text=['# Mature executor epoch comparison','',
        'Four separately restored mature-state, composite-objective epochs completed in balanced order with all 12 actions and identical condition, observation, query, member and complement identities. The weighted field row was active for every action. These are execution measurements, not learning or held-out results.','',
        '| Attempt | Run-return wall (s) | Epoch span (s) | Update span (s) | Callback/reporting (s) | Host batch wait sum (s) | Allocated / reserved (GiB) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for item in measurements:
        text.append(f"| {item['method']} | {item['run_return_seconds']:.3f} | {item['seconds']:.3f} | {item['train_seconds']:.3f} | {item['callback_seconds']:.3f} | {item['host_data_wait_seconds']:.3f} | {item['peak_allocated_gib']:.2f} / {item['peak_reserved_gib']:.2f} |")
    text.extend(['',f"Mean run-return time fell by {100*saving('run_return_seconds'):.3f}%; mean recorded epoch-span time fell by {100*saving('seconds'):.3f}%. Run-return time is the enclosing trainer call; epoch span includes normal callbacks and excludes offline pauses. The callback and host-wait values are components of enclosing clocks and must not be added to them.",'',
        'The separately computed 12-action trajectories diverged despite close same-state initial rows and products: the largest normalized signed-field violation difference was 0 at action 1, 2.60e-7 at action 2, 7.41e-6 at action 3, 7.65e-4 at action 6, 7.47e-3 at action 10, and 5.45e-3 at action 12. Differences were not monotone. Nonsmooth float32 sensitivity is a possible explanation, not an established cause; these observations do not establish trajectory equivalence.','',
        'The separate baseline uses the effective chunk-one path; its worker receipt retains an unused chunk-two argument. The candidate uses chunk two, static-feature caching and checkpoints. The side-by-side profiler scope bars are an instrumented single composite action: host scopes are inclusive and CUDA intersections are asynchronous, nonexclusive and not causal attributions.','',
        '![Epoch cost, peak memory and inclusive traced scopes](figures/executor_time_memory_breakdown.png)',
        f'*{caption}* [PDF master](figures/executor_time_memory_breakdown.pdf)','','## Figure index','','- Executor time, memory and trace scopes: `figures/executor_time_memory_breakdown.png` and `figures/executor_time_memory_breakdown.pdf`.','',
        'Whole-package costs, failures and CPU scoring/delivery are pending the terminal ledger. The adapter deliberately marks those values unavailable and must be refreshed before final publication.'])
    (output/'ENGINEERING_READOUT.md').write_text('\n'.join(text)+'\n')
    manifest={'status':'completed_provisional','adapter':ref(output/'engineering_adapter.json'),
        'markdown':ref(output/'ENGINEERING_READOUT.md'),'figure_png':ref(figures/'executor_time_memory_breakdown.png'),
        'figure_pdf':ref(figures/'executor_time_memory_breakdown.pdf'),'wall_seconds':time.time()-started,'cpu_seconds':time.process_time()-cpu_started}
    (output/'receipt.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();run(a.campaign,a.output)


if __name__=='__main__':main()
