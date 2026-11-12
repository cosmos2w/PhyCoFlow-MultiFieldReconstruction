"""TRAIN-frozen Round 8 endpoint decisions from matched cached panels.

Score reports are produced by round8_inference.score_cached_panel; coherence is
always evaluated on actual blended fields. No policy is fitted from ASSESS.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import torch
from .round8_campaign import file_hash
from .round8_inference import validate_matched_panels, score_cached_panel
from ..training.round8_policy import alpha_cap, select_train_policy, MATERIALITY, PRESERVATION

VERSION='round8_frozen_policy_v1'
KEYS={'state':'global_distribution.total','scale':'multiscale_spectrum.total','shape':'topology.total'}
SIGNED='multiscale_spectrum.signed_band_energy_score.conditional'
JOINT='multiscale_spectrum.joint_spectral_mean_energy_score.conditional'
FRONTIER=(0.,.25,.5,.75,1.)


def _save(path, payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n'
    try:
        with path.open('x') as stream: stream.write(encoded)
    except FileExistsError:
        if json.loads(path.read_text()) != payload:
            raise ValueError(f'existing frozen artifact differs: {path}')


def _load(path):
    with Path(path).open() as stream: return json.load(stream)


def panel_quadratic(source_path, endpoint_path):
    """Stream matched paired physical fields and calculate per-condition R0,B,A."""
    source, endpoint = validate_matched_panels(source_path,endpoint_path)
    rows=[]
    for s,e in zip(source['conditions'],endpoint['conditions']):
        with np.load(Path(source_path).parent/s['path'],allow_pickle=False) as x, np.load(Path(endpoint_path).parent/e['path'],allow_pickle=False) as y:
            for key in ('target_physical','physical_weights','native_ids','physical_mask','inclusion_probability'):
                if not np.array_equal(x[key],y[key]): raise ValueError('matched support, target or weights differ')
            p=float(x['inclusion_probability']); w=x['physical_weights'].astype(np.float64)/p
            if not 0<p<=1 or (w<0).any() or not x['physical_mask'].all(): raise ValueError('invalid sampled physical measure')
            if source['population']=='ASSESS' and (p != 1 or not np.isclose(w.sum(),1)):
                raise ValueError('ASSESS requires normalized native physical weights')
            target=x['target_physical'].astype(np.float64)[None]
            a=x['draws_physical'].astype(np.float64); d=y['draws_physical'].astype(np.float64)-a
            err=a-target
            def integrate(v): return (v*w[None,:,None]).sum(axis=1).mean(axis=0)
            r,b,q=integrate(err*err),integrate(err*d),integrate(d*d)
            direct=integrate((err+.5*d)**2)
            if not all(np.isfinite(v).all() for v in (r,b,q)) or not np.allclose(direct,r+b+.25*q,rtol=1e-11,atol=1e-12): raise ValueError('risk quadratic mismatch')
            rows.append({'sample_id':s['sample_id'],'legacy_source':s['source'],'r0':r.tolist(),'b':b.tolist(),'a':q.tolist()})
    def mean(k): return np.mean([r[k] for r in rows],axis=0)
    return {'source_panel':source,'endpoint_panel':endpoint,'r0':mean('r0'),'b':mean('b'),'a':mean('a'),'rows':rows}


def _mean_scores(report):
    rows=report['rows']
    if not rows: raise ValueError('common score report is empty')
    out={k:float(np.mean([r['losses'][v] for r in rows])) for k,v in KEYS.items()}
    if not all(math.isfinite(v) for v in out.values()): raise ValueError('nonfinite common scores')
    return out


def _check_report(report, source_sha, endpoint_sha, population, alpha, *, observer=None):
    if report.get('schema')!='round8_common_observers_v1' or report.get('population')!=population or report.get('source_panel_sha256')!=source_sha or report.get('endpoint_panel_sha256')!=endpoint_sha or not math.isclose(float(report['alpha']),float(alpha),rel_tol=0,abs_tol=1e-12):
        raise ValueError('score report identity or alpha mismatch')
    identity={k:report.get(k) for k in ('config_sha256','frozen_family_artifacts_sha256','fair_shape_artifact_sha256','native_observer_identity')}
    if observer is not None and identity != observer: raise ValueError('common observer identity changed')
    _mean_scores(report)
    return identity


def _compute_report(config, source, endpoint, fair, directory, alpha, *, assessment_policy=None, native_bundle=None, device='cpu'):
    path=Path(directory)/('alpha_'+format(alpha,'.17g').replace('.','p')+'.json')
    if not path.exists():
        score_cached_panel(config,source,endpoint,fair_shape_artifact=fair,output_path=path,alpha=float(alpha),assessment_policy=assessment_policy,native_bundle=native_bundle,device=device)
    return path


def _reuse_source_report(path, target_path, *, source_sha, endpoint_sha, population, expected_ids, policy_path=None):
    """Rebind a validated alpha-zero baseline after matched-panel validation."""
    old=_load(path)
    if old.get('schema')!='round8_common_observers_v1' or old.get('population')!=population or old.get('source_panel_sha256')!=source_sha or old.get('alpha')!=0. or [row['sample_id'] for row in old.get('rows',())]!=list(expected_ids):
        raise ValueError('shared SOURCE baseline identity differs')
    _mean_scores(old)
    new=dict(old)
    new.update({'endpoint_panel_sha256':endpoint_sha,'assessment_policy_sha256':file_hash(policy_path) if policy_path else None,'shared_source_baseline_sha256':file_hash(path),'shared_source_baseline_path':str(path)})
    if Path(target_path).exists():
        if _load(target_path)!=new: raise ValueError('existing rebound SOURCE baseline differs')
    else: _save(target_path,new)
    return Path(target_path)


def freeze_train_policy(config, source_panel, endpoint_panel, *, fair_shape_artifact, score_floors, output_dir, device='cpu', source_report_path=None):
    """Score clipped candidate blends on TRAIN and atomically freeze a policy."""
    q=panel_quadratic(source_panel,endpoint_panel)
    if q['source_panel']['population']!='TRAIN' or q['endpoint_panel']['endpoint']=='SOURCE': raise ValueError('TRAIN SOURCE and repair required')
    source_sha, endpoint_sha=file_hash(source_panel),file_hash(endpoint_panel)
    floors={k:float(score_floors[k]) for k in KEYS}
    if any(not math.isfinite(v) or v<=0 for v in floors.values()): raise ValueError('positive frozen TRAIN score floors required')
    cap=alpha_cap(q['r0'],q['b'],q['a'])
    candidates=sorted({min(.25,cap),min(.5,cap)}-{0.})
    score_dir=Path(output_dir)/'train_scores'
    paths={v:_compute_report(config,source_panel,endpoint_panel,fair_shape_artifact,score_dir,v,device=device) for v in [1.,*candidates]}
    paths[0.]=(_compute_report(config,source_panel,endpoint_panel,fair_shape_artifact,score_dir,0.,device=device) if source_report_path is None else _reuse_source_report(source_report_path,score_dir/'alpha_0.json',source_sha=source_sha,endpoint_sha=endpoint_sha,population='TRAIN',expected_ids=[x['sample_id'] for x in q['source_panel']['conditions']]))
    reports={k:_load(v) for k,v in paths.items()}
    observer=_check_report(reports[0.],source_sha,endpoint_sha,'TRAIN',0.)
    for alpha,report in reports.items(): _check_report(report,source_sha,endpoint_sha,'TRAIN',alpha,observer=observer)
    source_scores,raw_scores=_mean_scores(reports[0.]),_mean_scores(reports[1.])
    units={k:max(abs(source_scores[k]),floors[k]) for k in KEYS}
    risks={a:torch.as_tensor(q['r0']+2*a*q['b']+a*a*q['a']) for a in candidates}
    for a in (0.,1.,*candidates):
        expected=q['r0']+2*a*q['b']+a*a*q['a']
        if not np.allclose(expected,reports[a]['member_risk_physical'],rtol=1e-5,atol=1e-10): raise ValueError('scored risk differs from quadratic')
    policy=select_train_policy(cap,torch.as_tensor(q['r0']),risks,source_scores,raw_scores,{a:_mean_scores(reports[a]) for a in candidates},units)
    # This reference is predeclared even when the endpoint has no deployable policy.
    ref=policy.alpha if policy.alpha is not None else min(.5,cap)
    payload={'schema':VERSION,'population':'TRAIN','endpoint':q['endpoint_panel']['endpoint'],'alpha':policy.alpha,'families':list(policy.families),'reference_alpha':ref,'reference_only':policy.alpha is None,
             'alpha_cap':cap,'candidates':candidates,'source_panel_sha256':source_sha,'endpoint_panel_sha256':endpoint_sha,'manifest_sha256':q['source_panel']['manifest_sha256'],
             'source_checkpoint_sha256':q['source_panel']['source_checkpoint_sha256'],'endpoint_checkpoint_sha256':q['endpoint_panel']['checkpoint_sha256'],'observer_identity':observer,
             'train_source_scores':source_scores,'train_raw_scores':raw_scores,'train_score_units':units,'score_floors':floors,'denominator_provenance':'max(abs(TRAIN SOURCE score), TRAIN-frozen positive floor)',
             'train_source_risk':q['r0'].tolist(),'risk_quadratic':{'b':q['b'].tolist(),'a':q['a'].tolist()},'score_reports':{str(k):{'path':str(v),'sha256':file_hash(v)} for k,v in paths.items()}}
    path=Path(output_dir)/'frozen_policy.json';_save(path,payload)
    return path


def _summary(values):
    a=np.asarray(values,dtype=np.float64)
    if len(a)==0 or not np.isfinite(a).all(): raise ValueError('finite nonempty observations required')
    return {'mean':float(a.mean()),'median':float(np.median(a)),'q05':float(np.quantile(a,.05)),'q95':float(np.quantile(a,.95)),'condition_standard_error':float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else None,'n':len(a)}


def _gain(source,candidate,units): return {k:(source[k]-candidate[k])/units[k] for k in KEYS}


def assess_fixed_policy(policy, source_report, raw_report, blend_report, *, field_names, regime_by_id=None):
    """Evaluate the frozen policy; subgroup/uncertainty summaries add no gates."""
    if policy['alpha'] is None or not policy['families']:
        return {'eligible':False,'reason':'no TRAIN-nominated blend'}
    pop=source_report['population']; source_sha=source_report['source_panel_sha256']; endpoint_sha=source_report['endpoint_panel_sha256']
    if pop!='ASSESS': raise ValueError('ASSESS reports required')
    identity=_check_report(source_report,source_sha,endpoint_sha,pop,0.)
    _check_report(raw_report,source_sha,endpoint_sha,pop,1.,observer=identity)
    _check_report(blend_report,source_sha,endpoint_sha,pop,policy['alpha'],observer=identity)
    ids=[r['sample_id'] for r in source_report['rows']]
    if any([r['sample_id'] for r in report['rows']]!=ids for report in (raw_report,blend_report)): raise ValueError('score condition IDs differ')
    src,raw,blend=map(_mean_scores,(source_report,raw_report,blend_report))
    units={k:max(abs(src[k]),policy['score_floors'][k]) for k in KEYS}
    gain,rawgain=_gain(src,blend,units),_gain(src,raw,units)
    baseline=np.asarray(source_report['member_risk_physical'],dtype=float); risk=np.asarray(blend_report['member_risk_physical'],dtype=float)
    if baseline.shape!=risk.shape or len(field_names)!=len(risk) or not np.isfinite(baseline).all() or not np.isfinite(risk).all(): raise ValueError('invalid per-field risks')
    risk_ok=bool(np.all(risk<=1.1*baseline)); preserve=all(gain[k]>=-PRESERVATION[k] for k in KEYS)
    qualified=[k for k in policy['families'] if rawgain[k]>0 and gain[k]>=MATERIALITY[k] and gain[k]>=.3*rawgain[k]]
    rows=[]
    for a,b in zip(source_report['rows'],blend_report['rows']):
        sr=np.asarray(a['member_mse_physical'],dtype=float).mean(0); br=np.asarray(b['member_mse_physical'],dtype=float).mean(0)
        rows.append({'sample_id':a['sample_id'],'legacy_source':a['legacy_source'],'regime':(regime_by_id or {}).get(a['sample_id'],'unknown'),'source_risk':sr.tolist(),'blend_risk':br.tolist(),'source_scores':{k:float(a['losses'][v]) for k,v in KEYS.items()},'blend_scores':{k:float(b['losses'][v]) for k,v in KEYS.items()}})
    grouped={}
    for category in ('legacy_source','regime'):
        grouped[category]={}
        for value in sorted({r[category] for r in rows}):
            group=[r for r in rows if r[category]==value]
            grouped[category][value]={'n':len(group),'field_risk_delta':{name:_summary([r['blend_risk'][i]-r['source_risk'][i] for r in group]) for i,name in enumerate(field_names)},'score_gain':{k:_summary([(r['source_scores'][k]-r['blend_scores'][k])/units[k] for r in group]) for k in KEYS}}
    pooled={'field_risk_delta':{name:_summary([r['blend_risk'][i]-r['source_risk'][i] for r in rows]) for i,name in enumerate(field_names)},'score_gain':{k:_summary([(r['source_scores'][k]-r['blend_scores'][k])/units[k] for r in rows]) for k in KEYS}}
    return {'eligible':risk_ok and preserve and bool(qualified),'pooled':pooled,'risk_pass':risk_ok,'preservation_pass':preserve,'qualifying_families':qualified,'gain':gain,'raw_gain':rawgain,'source_scores':src,'blend_scores':blend,'score_units':units,'denominator_provenance':'ASSESS SOURCE score with frozen TRAIN floor','field_names':list(field_names),'source_risk':baseline.tolist(),'blend_risk':risk.tolist(),'risk_ratio':[None if x==0 else float(y/x) for x,y in zip(baseline,risk)],'rmse_change':[None if x==0 else float(np.sqrt(y/x)-1) for x,y in zip(baseline,risk)],'groups':grouped,'per_condition':rows,'uncertainty_note':'Condition standard errors are descriptive; correlated frames need not be independent.'}


def assess_endpoint(config, source_panel, endpoint_panel, policy_path, *, fair_shape_artifact, native_bundle, output_dir, regime_by_id=None, device='cpu', source_report_path=None):
    """Score the frozen ASSESS endpoint, then the descriptive fixed frontier."""
    policy=_load(policy_path); source,endpoint=validate_matched_panels(source_panel,endpoint_panel)
    if (policy.get('schema')!=VERSION or source['population']!='ASSESS' or policy['manifest_sha256']!=source['manifest_sha256'] or policy['source_checkpoint_sha256']!=source['source_checkpoint_sha256'] or policy['endpoint_checkpoint_sha256']!=endpoint['checkpoint_sha256'] or policy['endpoint']!=endpoint['endpoint']):
        raise ValueError('ASSESS endpoint is not bound to the TRAIN-frozen policy')
    source_sha, endpoint_sha=file_hash(source_panel),file_hash(endpoint_panel)
    directory=Path(output_dir)/'assess_scores'
    def run(alpha,reference_only=False):
        path=directory/('alpha_'+format(alpha,'.17g').replace('.','p')+('_reference' if reference_only else '')+'.json')
        if not path.exists():
            score_cached_panel(config,source_panel,endpoint_panel,fair_shape_artifact=fair_shape_artifact,output_path=path,alpha=float(alpha),assessment_policy=policy_path,native_bundle=native_bundle,device=device,reference_only=reference_only)
        report=_load(path);_check_report(report,source_sha,endpoint_sha,'ASSESS',alpha)
        if report.get('assessment_policy_sha256')!=file_hash(policy_path): raise ValueError('ASSESS report policy identity differs')
        return path,report
    if source_report_path is None:
        source_path,source=run(0.)
    else:
        source_path=_reuse_source_report(source_report_path,directory/'alpha_0.json',source_sha=source_sha,endpoint_sha=endpoint_sha,population='ASSESS',expected_ids=[x['sample_id'] for x in source['conditions']],policy_path=policy_path)
        source=_load(source_path)
    raw_path,raw=run(1.)
    train_identity=policy['observer_identity']
    if any(source.get(k)!=train_identity.get(k) for k in ('config_sha256','frozen_family_artifacts_sha256','fair_shape_artifact_sha256')):
        raise ValueError('ASSESS observer is not the TRAIN-frozen common observer')
    _check_report(raw,source_sha,endpoint_sha,'ASSESS',1.,observer=_check_report(source,source_sha,endpoint_sha,'ASSESS',0.))
    if policy['alpha'] is None:
        result={'eligible':False,'reason':'no TRAIN-nominated blend'}
        selected_path=None
        if policy['endpoint']=='B8':
            reference_path,reference=run(policy['reference_alpha'],reference_only=True)
        else: reference_path=reference=None
    else:
        selected_path,selected=run(policy['alpha'])
        result=assess_fixed_policy(policy,source,raw,selected,field_names=source['field_names'] if 'field_names' in source else endpoint['field_names'],regime_by_id=regime_by_id)
        reference_path,reference=selected_path,selected
    decision_path=Path(output_dir)/'assessment_decision.json'
    _save(decision_path,{'schema':'round8_assessment_decision_v1','endpoint':policy['endpoint'],'policy_sha256':file_hash(policy_path),'result':result})
    # The frozen decision is saved before any descriptive frontier score.
    frontier={}
    for alpha in FRONTIER:
        path,report=run(alpha)
        _check_report(report,source_sha,endpoint_sha,'ASSESS',alpha,observer=_check_report(source,source_sha,endpoint_sha,'ASSESS',0.))
        frontier[str(alpha)]={'path':str(path),'sha256':file_hash(path),'scores':_mean_scores(report),'member_risk_physical':report['member_risk_physical']}
    result.update({'schema':'round8_assessment_v1','endpoint':policy['endpoint'],'alpha':policy['alpha'],'policy_sha256':file_hash(policy_path),'source_report':str(source_path),'raw_report':str(raw_path),'selected_report':None if selected_path is None else str(selected_path),'b8_reference_report':None if reference_path is None else str(reference_path),'b8_reference_only':policy['reference_only'] if policy['endpoint']=='B8' else None,'decision_sha256':file_hash(decision_path),'frontier':frontier,'observer_identity':_check_report(source,source_sha,endpoint_sha,'ASSESS',0.)})
    path=Path(output_dir)/'assessment.json';_save(path,result)
    return path


def strong_scale(source, candidate, b8_reference, *, score_floor, assessment):
    """Additional claim gate; all inputs are matched ASSESS common reports."""
    reports=(source,candidate,b8_reference)
    if any(x['population']!='ASSESS' for x in reports): raise ValueError('ASSESS reports required')
    if len({x['source_panel_sha256'] for x in reports})!=1: raise ValueError('SOURCE panel differs')
    identity={k:source.get(k) for k in ('config_sha256','frozen_family_artifacts_sha256','fair_shape_artifact_sha256','native_observer_identity')}
    if any({k:x.get(k) for k in identity}!=identity for x in reports): raise ValueError('observer identity differs')
    ids=[r['sample_id'] for r in source['rows']]
    if any([r['sample_id'] for r in x['rows']]!=ids for x in reports): raise ValueError('condition identity differs')
    src,cur,base=map(_mean_scores,reports);unit=max(abs(src['scale']),float(score_floor))
    components={}
    for name,key in (('signed',SIGNED),('joint',JOINT)):
        s=float(np.mean([r['losses'][key] for r in source['rows']])); c=float(np.mean([r['losses'][key] for r in candidate['rows']]))
        components[name]={'source':s,'candidate':c,'pass':s>0 and c<=1.005*s}
    source_gain=(src['scale']-cur['scale'])/unit
    b8_gain=(base['scale']-cur['scale'])/unit
    return {'pass':bool(assessment.get('eligible')) and source_gain>=.01 and b8_gain>=.005 and all(v['pass'] for v in components.values()),'source_gain':source_gain,'b8_gain':b8_gain,'unit':unit,'components':components}


def computation_decision(source_report, b8_raw, c8_raw, *, floors, b8_cost, c8_cost):
    """Assess raw endpoints and amortized measured cost under the fixed rule.

    Each cost has per-update GPU seconds including callbacks, calibration and
    half shared warmup, actual update count, measurement device identity and
    campaign GPU seconds. Same-class devices use their actual occupied times;
    different device types require caller-supplied same-device amortized
    measurements with a shared comparison_device.
    """
    if b8_raw['alpha']!=1 or c8_raw['alpha']!=1: raise ValueError('raw endpoint scores required')
    reports=(source_report,b8_raw,c8_raw)
    if len({r['population'] for r in reports})!=1 or len({r['source_panel_sha256'] for r in reports})!=1: raise ValueError('matched population and SOURCE panel required')
    ids=[r['sample_id'] for r in source_report['rows']]
    if any([r['sample_id'] for r in report['rows']]!=ids for report in reports): raise ValueError('matched condition IDs required')
    identity={k:source_report.get(k) for k in ('config_sha256','frozen_family_artifacts_sha256','fair_shape_artifact_sha256','native_observer_identity')}
    if any({k:r.get(k) for k in identity}!=identity for r in reports): raise ValueError('observer identity differs')
    source,b,c=map(_mean_scores,reports); units={k:max(abs(source[k]),float(floors[k])) for k in KEYS}
    if any(not math.isfinite(v) or v<=0 for v in units.values()): raise ValueError('positive score units required')
    change={k:(c[k]-b[k])/units[k] for k in KEYS}; limits={'scale':.005,'shape':.02,'state':.02}
    r0,rb,rc=(np.asarray(x['member_risk_physical'],dtype=float) for x in reports)
    quality=all(change[k]<=limits[k] for k in KEYS) and bool(np.all(rc-rb<=.02*r0))
    gain=any(-change[k]>=MATERIALITY[k] for k in KEYS)
    for cost in (b8_cost,c8_cost):
        if cost['actual_updates']<=0 or cost['amortized_seconds_per_update']<=0 or cost['campaign_gpu_seconds']<0 or not cost.get('comparison_device'): raise ValueError('complete positive cost measurements required')
    if b8_cost['comparison_device']!=c8_cost['comparison_device'] or b8_cost['actual_updates']!=c8_cost['actual_updates']: raise ValueError('same-class or normalized same-device comparison and matched updates required')
    ratio=c8_cost['amortized_seconds_per_update']/b8_cost['amortized_seconds_per_update']
    adopted=quality and (ratio<=.9 or (gain and ratio<=1.1))
    return {'retain_c8_default':adopted,'quality_preserved':quality,'qualifying_quality_gain':gain,'score_change':change,'field_risk_increase':(rc-rb).tolist(),'cost_ratio':ratio,'b8_cost':b8_cost,'c8_cost':c8_cost}


def operational_choice(assessments, benchmarks, *, source_benchmark, device_vram_bytes):
    """Choose lowest latency under matched J2 batch-one device benchmarks.

    Every benchmark declares device_uuid, batch_size, member_count,
    warmup_policy, request_ids_sha256, synchronized and both_models_resident.
    """
    source_latency=float(source_benchmark['latency_seconds'])
    if not math.isfinite(source_latency) or source_latency<=0 or device_vram_bytes<=0: raise ValueError('valid SOURCE benchmark required')
    keys=('device_uuid','batch_size','member_count','warmup_policy','request_ids_sha256','synchronized')
    if any(k not in source_benchmark for k in keys) or source_benchmark['batch_size']!=1 or source_benchmark['member_count']!=2 or source_benchmark['synchronized'] is not True:
        raise ValueError('SOURCE benchmark must specify synchronized batch-one J2 conditions')
    eligible=[]; exclusions={}
    for name in ('C8','B8','T2000'):
        if name not in assessments: continue
        outcome=assessments[name]
        if not outcome.get('eligible'): exclusions[name]='ASSESS rule failed or no nomination'; continue
        if name not in benchmarks: exclusions[name]='benchmark missing'; continue
        v=benchmarks[name]
        if any(v.get(k)!=source_benchmark[k] for k in keys): raise ValueError('endpoint benchmark conditions differ from SOURCE')
        if 0 < float(outcome['alpha']) < 1 and v.get('both_models_resident') is not True:
            raise ValueError('blend benchmark must include both resident models')
        latency=float(v['latency_seconds']); memory=int(v['peak_resident_bytes'])
        if not math.isfinite(latency) or latency<=0 or memory<=0: raise ValueError('invalid endpoint benchmark')
        if latency>2.2*source_latency or memory>.9*device_vram_bytes: exclusions[name]='latency or memory guard'; continue
        eligible.append((latency,('C8','B8','T2000').index(name),name))
    winner=min(eligible)[2] if eligible else 'SOURCE'
    return {'choice':winner,'eligible':[x[2] for x in sorted(eligible)],'exclusions':exclusions,'source_benchmark':source_benchmark,'benchmarks':benchmarks,'device_vram_bytes':device_vram_bytes}


def decide_campaign(assessment_paths, *, source_benchmark, benchmarks, device_vram_bytes,
                    computation_reports, b8_cost, c8_cost, score_floors, output_path):
    """Produce a reviewable final decision from frozen reports and measurements.

    ``computation_reports`` maps population to source, b8 and c8 common-report
    file paths; each population is evaluated independently. Assessment files are
    keyed by T2000/B8/C8. Missing endpoints remain absent, never imputed.
    """
    assessments={name:_load(path) for name,path in assessment_paths.items()}
    if any(name not in ('T2000','B8','C8') or value.get('endpoint')!=name for name,value in assessments.items()):
        raise ValueError('invalid endpoint assessment mapping')
    operational=operational_choice(assessments,benchmarks,source_benchmark=source_benchmark,device_vram_bytes=device_vram_bytes)
    computation={}
    for population, paths in computation_reports.items():
        if population not in ('TRAIN','ASSESS') or set(paths)!={'source','b8','c8'}: raise ValueError('invalid computation report mapping')
        reports={name:_load(path) for name,path in paths.items()}
        if any(r['population']!=population for r in reports.values()): raise ValueError('computation population mismatch')
        computation[population]=computation_decision(reports['source'],reports['b8'],reports['c8'],floors=score_floors,b8_cost=b8_cost,c8_cost=c8_cost)
    claims={}
    b8=assessments.get('B8')
    if b8 and b8.get('b8_reference_report'):
        ref=_load(b8['b8_reference_report'])
        for name, result in assessments.items():
            if name=='B8' or not result.get('selected_report'): continue
            claims[name]=strong_scale(_load(result['source_report']),_load(result['selected_report']),ref,score_floor=score_floors['scale'],assessment=result)
    payload={'schema':'round8_campaign_decision_v1','operational':operational,'computation_by_population':computation,'strong_scale_claims':claims,'assessment_sha256':{name:file_hash(path) for name,path in assessment_paths.items()},'selection_note':'Operational choice uses the fixed ASSESS panel; it is not an independent post-selection generalization estimate.'}
    _save(output_path,payload)
    return Path(output_path)
