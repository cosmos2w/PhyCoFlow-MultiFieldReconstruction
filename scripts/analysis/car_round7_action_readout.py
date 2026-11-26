#!/usr/bin/env python3
"""CPU-only complete Scale decomposition of paired Round 7 shadow-action banks.

This reads already generated J2 endpoints, performs no inference, and reports
one-action diagnostics separately for SOURCE and mature optimizer states.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('CAR_REPO_ROOT', CODE)).resolve()
sys.path[:0] = [str(CODE/'src'), str(CODE)]
import numpy as np
import torch
import yaml
from phycoflow_reconstruction.coherence import build_enabled_families
from phycoflow_reconstruction.coherence.adaptive_fitting import spectral_filter, validate_bank_target_binding
from phycoflow_reconstruction.coherence.query_runtime import _scale_statistics, _shape_grid
from phycoflow_reconstruction.coherence.query_sampling import Draw, QueryBatch, ConditionQueryGroups
from phycoflow_reconstruction.coherence.shape_compute import LazyPooledShape
from phycoflow_reconstruction.data.factory import open_field_dataset
from phycoflow_reconstruction.training.scalable_fitting import _digest, validate_scalable_artifacts
from scripts.analysis.car_round7_paired_summary import paired_summary
from scripts.analysis.car_round7_scale_decompose import decompose

BLOCKS = ('signed','joint','log_power_self','rho','third_order')
TERMS = ('attraction','separation','score')
DEFINITIONS = ('Published_Core_Default','Complete_Core_Prefit','Complete_Expanded_Prefit')
STATES = ('source','mature')
ARMS = ('composite','joint_anchor')


def query_from_manifest(manifest, offset, count):
    def make_batch(value):
        def draw(item):
            item=dict(item)
            if item.get('inner_sign') is not None: item['inner_sign']=tuple(item['inner_sign'])
            return Draw(**item)
        return QueryBatch(tuple(draw(item) for item in value['self_queries']),
                          tuple(draw(item) for item in value['pair_queries']),value['number'])
    if 'batches' not in manifest: return make_batch(manifest)
    groups=ConditionQueryGroups(tuple(make_batch(batch) for batch in manifest['batches']))
    for selected,query in groups.slices(count):
        if selected.start <= offset < selected.stop: return query
    raise ValueError('condition offset is outside the saved Shape query groups')


def shape_components(shape, compiled, grids, physical_weights):
    scores=shape.score_compiled(grids,compiled[0],compiled[1],valid_mask=compiled[2],include_components=True)
    output={}
    for kind in ('self','mutual'):
        weight=physical_weights[kind]
        categories={}
        for category in ('finite','essential'):
            attraction=float(scores[f'{kind}.{category}.attraction'][0])*weight
            separation=float(scores[f'{kind}.{category}.separation'][0])*weight
            categories[category]={'attraction':attraction,'separation':separation,'score':attraction-separation}
        total=float(scores[kind][0])*weight
        reconstructed=sum(item['score'] for item in categories.values())
        tolerance=64*torch.finfo(torch.float32).eps*max(1.0,abs(total),abs(reconstructed))
        if abs(total-reconstructed)>tolerance: raise ValueError('Shape finite/essential A-R sum mismatch')
        output[kind]={'score':total,'finite':categories['finite'],'essential':categories['essential'],
                      'sum_check_abs_error':abs(total-reconstructed),'sum_check_tolerance':tolerance}
    output['total']=sum(output[kind]['score'] for kind in ('self','mutual'))
    return output


def digest(path):
    hasher=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''): hasher.update(chunk)
    return hasher.hexdigest()


def check_declaration(declaration):
    path=Path(declaration['path']).resolve(strict=True)
    if digest(path) != declaration['sha256']: raise ValueError(f'artifact hash mismatch: {path}')
    return path


def validate_decomposition(result,prefit):
    for block in BLOCKS:
        item=result['blocks'][block]
        if not all(math.isfinite(float(item[k])) for k in TERMS): raise ValueError('non-finite Scale term')
        if not math.isclose(item['attraction']-item['separation'],item['score'],rel_tol=1e-8,abs_tol=1e-10):
            raise ValueError('Scale A-R does not reconstruct '+block)
    expanded=sum(float(prefit['scale.'+key])*(0.25 if key=='joint' else 1.0)*result['blocks'][key]['score'] for key in BLOCKS)
    if not math.isclose(expanded,result['definitions']['Complete_Expanded_Prefit'],rel_tol=1e-8,abs_tol=1e-10):
        raise ValueError('expanded prefit Scale does not reconstruct')


def summarize_conditions(records, *, bootstrap, seed):
    if not records or bootstrap < 1: raise ValueError('nonempty records and positive bootstrap count required')
    ids=[row['sample_id'] for row in records]
    if len(set(ids)) != len(ids): raise ValueError('condition IDs must be unique within state')
    indices=np.random.default_rng(seed).integers(0,len(records),size=(bootstrap,len(records)))
    def compare(left,right):
        return paired_summary(np.asarray(left),np.asarray(right),indices)
    def extract(name,section,key,term=None):
        values=[row['models'][name][section][key] for row in records]
        return [value[term] for value in values] if term else values
    mean=lambda values:float(np.mean(np.asarray(values,dtype=float)))
    levels={name:{
        'definitions':{key:mean(extract(name,'definitions',key)) for key in DEFINITIONS},
        'raw_blocks':{key:{term:mean(extract(name,'blocks',key,term)) for term in TERMS} for key in BLOCKS}}
        for name in ('pre','composite','joint_anchor')}
    comparisons={}
    for label,left,right in (('composite_minus_pre','pre','composite'),
                             ('joint_anchor_minus_pre','pre','joint_anchor'),
                             ('joint_anchor_minus_composite','composite','joint_anchor')):
        comparisons[label]={
            'definitions':{key:compare(extract(left,'definitions',key),extract(right,'definitions',key)) for key in DEFINITIONS},
            'raw_blocks':{key:{term:compare(extract(left,'blocks',key,term),extract(right,'blocks',key,term))
                               for term in TERMS} for key in BLOCKS}}
    shape={}
    for query in ('fixed','independent'):
        shape[query]={}
        for kind in ('self','mutual'):
            shape[query][kind]={}
            for category in ('finite','essential'):
                shape[query][kind][category]={}
                for term in TERMS:
                    def vals(model): return [r['models'][model]['shape'][query][kind][category][term] for r in records]
                    shape[query][kind][category][term]={
                        'levels':{model:mean(vals(model)) for model in ('pre','composite','joint_anchor')},
                        'composite_minus_pre':compare(vals('pre'),vals('composite')),
                        'joint_anchor_minus_pre':compare(vals('pre'),vals('joint_anchor')),
                        'joint_anchor_minus_composite':compare(vals('composite'),vals('joint_anchor'))}
    return {'sample_count':len(ids),'sample_ids':ids,
        'bootstrap':{'replicates':bootstrap,'seed':seed,'unit':'paired condition; J2 members retained together'},
        'levels':levels, 'comparisons':comparisons, 'shape_decomposition':shape}


def weighted_batch_mean(rows, extract):
    values=[np.asarray(extract(row),dtype=float) for row in rows]
    weights=np.asarray([row['sample_count'] for row in rows],dtype=float)
    if not values or any(not np.isfinite(value).all() for value in values): raise ValueError('non-finite or empty batch diagnostics')
    return np.average(np.stack(values),axis=0,weights=weights).tolist()


def summarize_actions(rows):
    """Batch-level update diagnostics, weighted by represented condition count."""
    mean=lambda getter:weighted_batch_mean(rows,getter)
    result={'sample_count':sum(row['sample_count'] for row in rows), 'batch_count':len(rows),
        'weighting':'sample-count weighted batch diagnostics; no condition bootstrap because each action is shared by the batch',
        'scale_row_cosine':mean(lambda r:r['scale_row_cosine']),
        'scale_row_norms':{arm:mean(lambda r:r['scale_row_norms'][arm]) for arm in ARMS},
        'field_row_norm':mean(lambda r:r['weighted_field_row_norm']),'arms':{}}
    family_order=tuple(rows[0]['family_row_norms'])
    for arm in ARMS:
        result['arms'][arm]={key:mean(lambda r,key=key:r['arms'][arm][key]) for key in
            ('direction_norm','displacement_norm','clip_input_norm','learning_rate','joint_row_dot_displacement','weighted_field_row_dot_displacement')}
        result['arms'][arm]['family_row_dot_displacement']={name:mean(lambda r,name=name:r['arms'][arm]['row_dot_displacement'][name]) for name in family_order}
        result['arms'][arm]['field_risk_change']=mean(lambda r:np.asarray(r['arms'][arm]['field_risk_post'])-np.asarray(r['field_risk_pre']))
        result['arms'][arm]['shape']={}
        for query in ('fixed','independent'):
            pre='shape_'+query+'_pre';post='shape_'+query+'_post'
            info={'pre_scalar':mean(lambda r:r[pre]['scalar']),
                  'post_scalar':mean(lambda r:r['arms'][arm][post]['scalar']),
                  'change':mean(lambda r:r['arms'][arm][post]['scalar']-r[pre]['scalar'])}
            component_names=tuple(rows[0][pre]['components'])
            info['component_change']={name:{term:mean(lambda r:r['arms'][arm][post]['components'][name][term]-r[pre]['components'][name][term])
                                             for term in ('raw','weighted')} for name in component_names}
            result['arms'][arm]['shape'][query]=info
        result['arms'][arm]['shape']['independent_minus_fixed_pre']=mean(lambda r:r['shape_independent_pre']['scalar']-r['shape_fixed_pre']['scalar'])
        result['arms'][arm]['shape']['independent_minus_fixed_post']=mean(lambda r:r['arms'][arm]['shape_independent_post']['scalar']-r['arms'][arm]['shape_fixed_post']['scalar'])
    return result


def run(args):
    started,cpu_started=time.time(),time.process_time()
    args.output.mkdir(parents=True,exist_ok=False)
    receipt={'schema':'car_round7_action_readout_v1','status':'failed','audit':str(args.audit.resolve()),
        'config_sha256':digest(args.config),'bootstrap_seed':args.seed,'bootstrap_replicates':args.bootstrap,
        'scope':'single independently reset TRAIN J2 action; complete selected-support Scale and sampled action/Shape diagnostics; no inference, training trajectory or held-out efficacy',
        'test_locked':True}
    dataset=None
    try:
        torch.set_num_threads(args.threads)
        audit_receipt=json.loads((args.audit/'receipt.json').read_text())
        if audit_receipt.get('status') not in ('completed','completed_fixture_only'):
            raise ValueError('a completed audit or completed fixture is required')
        if audit_receipt['config_sha256'] != digest(args.config): raise ValueError('audit config identity mismatch')
        rows_path=args.audit/'batches.jsonl'
        if audit_receipt.get('batches_sha256') != digest(rows_path): raise ValueError('audit records hash mismatch')
        rows=[json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
        by_state={state:[row for row in rows if row['state']==state] for state in STATES}
        if any(not value for value in by_state.values()): raise ValueError('SOURCE and mature records required')
        if len(by_state['source']) != len(by_state['mature']): raise ValueError('paired state batch counts differ')
        for left,right in zip(by_state['source'],by_state['mature'],strict=True):
            for key in ('batch','sample_ids','sample_count','selected_sha256','complement_sha256',
                        'observation_sha256','member_seeds','first_member_generator_state_sha256',
                        'shape_fixed_query_sha256','shape_independent_query_sha256'):
                if left[key]!=right[key]: raise ValueError('paired states differ: '+key)
        cfg=yaml.safe_load(args.config.read_text())
        dataset=open_field_dataset(cfg['dataset'],split='train')
        family=build_enabled_families(cfg['coherence'],dataset.data_spec,dataset.normalizer)['multiscale_spectrum']
        declaration=cfg['coherence']['frozen_family_artifacts']['multiscale_spectrum']
        family_path=check_declaration(declaration)
        family.load_state_artifact(torch.load(family_path,map_location='cpu',weights_only=True))
        shape_family=build_enabled_families(cfg['coherence'],dataset.data_spec,dataset.normalizer)['topology']
        shape_family_path=check_declaration(cfg['coherence']['frozen_family_artifacts']['topology'])
        shape_family.load_state_artifact(torch.load(shape_family_path,map_location='cpu',weights_only=True))
        core_path=args.audit/'scalable_coherence.pt'
        core=torch.load(core_path,map_location='cpu',weights_only=True)
        validate_scalable_artifacts(core,core['provenance'])
        if (core['provenance']['source_checkpoint_sha256'] != audit_receipt['source_identity_before']['checkpoint']
            or core['provenance']['normalizer_sha256'] != dataset.normalizer.digest()):
            raise ValueError('audit Shape core SOURCE or normalizer identity differs')
        shape=LazyPooledShape.from_state_dict(core['shape_artifact'])
        settings=cfg['coherence']['scalable_anchored_coherence']
        bank_path=check_declaration(settings['adaptive']['bank_artifact'])
        bank_doc=json.loads(bank_path.read_text()); bank=bank_doc.get('bank',bank_doc)
        prefit_path=check_declaration(settings['common_prefit']); prefit_doc=json.loads(prefit_path.read_text())
        prefit=prefit_doc['coefficients']; validate_bank_target_binding(prefit_doc['binding'],bank)
        shape_weights={kind:float(shape_family.spatial_objective.weights[kind+'.persistence'])*float(prefit['shape.'+kind])
                       for kind in ('self','mutual')}
        if any(not math.isfinite(value) or value <= 0 for value in shape_weights.values()):
            raise ValueError('Shape physical weights and prefit coefficients must be positive finite')
        if dataset.normalizer.digest() != prefit_doc['binding']['normalizer_sha256']:
            raise ValueError('normalizer differs from frozen prefit')
        receipt.update(audit_receipt_sha256=digest(args.audit/'receipt.json'),audit_records_sha256=digest(rows_path),
            frozen_family_sha256=digest(family_path), frozen_shape_family_sha256=digest(shape_family_path),
            scalable_core_sha256=digest(core_path), shape_artifact_content_sha256=_digest(core['shape_artifact']),
            shape_physical_prefit_weights=shape_weights, bank_sha256=digest(bank_path), prefit_sha256=digest(prefit_path),
            source_identity=audit_receipt['source_identity_before'],parent_checkpoint_sha256=audit_receipt['parent_checkpoint_sha256'],
            field_reference_sha256=audit_receipt['reference_sha256'],kappa=audit_receipt['kappa'],
            expected_samples=audit_receipt['sample_count'],cpu_threads=args.threads)
        records={state:[] for state in STATES}
        shape_comparisons=[]
        path=args.output/'conditions.jsonl'
        with path.open('x') as stream, torch.no_grad():
            for state in STATES:
                for row in by_state[state]:
                    bank_path=check_declaration(row['endpoint_bank'])
                    payload=torch.load(bank_path,map_location='cpu',weights_only=True)
                    if payload['schema']!='car_round7_selected_action_bank_v1' or payload['member_layout']!='J,B,Q,C':
                        raise ValueError('unsupported endpoint bank layout')
                    if payload['sample_ids']!=row['sample_ids'] or payload['target'].shape[0]!=row['sample_count']:
                        raise ValueError('endpoint bank condition IDs or counts differ')
                    for query_name in ('fixed','independent'):
                        if _digest(payload['shape_'+query_name+'_query']) != row['shape_'+query_name+'_query_sha256']:
                            raise ValueError('saved Shape query manifest identity differs')
                    batch_shape_totals={query:{label:[] for label in ('pre','composite','joint_anchor')}
                                        for query in ('fixed','independent')}
                    for offset,sample_id in enumerate(payload['sample_ids']):
                        target=payload['target'][offset:offset+1]
                        coords=payload['query_coords'][offset:offset+1]
                        if not bool(payload['query_valid_mask'][offset].all()): raise ValueError('masked Scale query support unsupported')
                        family._require_ready(target,coords)
                        target_stats=_scale_statistics(family,target)
                        target_filtered=spectral_filter(family,target)
                        shape_family._raster_map(coords)
                        target_grid=_shape_grid(shape_family,target,target)
                        models={}
                        for label,key in (('pre','pre_members'),('composite','composite_post_members'),('joint_anchor','joint_anchor_post_members')):
                            members=payload[key]
                            if members.ndim!=4 or members.shape[0]!=2 or members.shape[1]!=row['sample_count']:
                                raise ValueError('exactly two matched members per condition required')
                            result=decompose(family,members[:,offset:offset+1],target,bank,prefit,
                                target_statistics=target_stats,target_filtered=target_filtered)
                            validate_decomposition(result,prefit)
                            models[label]={'blocks':result['blocks'],'definitions':result['definitions']}
                        grids={label:torch.stack([_shape_grid(shape_family,value,target) for value in payload[key][:,offset:offset+1]])
                               for label,key in (('pre','pre_members'),('composite','composite_post_members'),('joint_anchor','joint_anchor_post_members'))}
                        for query_name in ('fixed','independent'):
                            query=query_from_manifest(payload['shape_'+query_name+'_query'],offset,row['sample_count'])
                            queries=shape.compile_queries((query,))
                            mask=shape_family.domain_mask if shape_family.native_masked else None
                            compiled=(queries,shape.compile_reference(target_grid,queries,valid_mask=mask),mask)
                            for label in ('pre','composite','joint_anchor'):
                                part=shape_components(shape,compiled,grids[label],shape_weights)
                                models[label].setdefault('shape',{})[query_name]=part
                                batch_shape_totals[query_name][label].append(part['total'])
                        item={'state':state,'sample_id':sample_id,'batch':row['batch'],'models':models}
                        records[state].append(item)
                        stream.write(json.dumps(item,allow_nan=False)+'\n')
                    for query_name in ('fixed','independent'):
                        for label in ('pre','composite','joint_anchor'):
                            computed=float(np.mean(batch_shape_totals[query_name][label]))
                            recorded=(row['shape_'+query_name+'_pre']['scalar'] if label=='pre'
                                      else row['arms'][label]['shape_'+query_name+'_post']['scalar'])
                            tolerance=64*torch.finfo(torch.float32).eps*max(1.0,abs(computed),abs(recorded))
                            discrepancy=abs(computed-recorded)
                            shape_comparisons.append({'state':state,'batch':row['batch'],'query':query_name,'model':label,
                                'sampled_scalar':recorded,'compiled_scalar':computed,'absolute_difference':discrepancy,
                                'float32_reduction_tolerance':tolerance,'within_tolerance':discrepancy<=tolerance})
                    stream.flush()
                    receipt['completed_condition_counts']={key:len(value) for key,value in records.items()}
                    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
        if [r['sample_id'] for r in records['source']] != [r['sample_id'] for r in records['mature']]:
            raise ValueError('SOURCE and mature condition ordering differs')
        if any(len(records[state]) != audit_receipt['sample_count'] for state in STATES):
            raise ValueError('record count differs from declared panel')
        summary={'schema':'car_round7_action_summary_v1','interpretation':'Paired same-state one-action diagnostics on TRAIN-selected support; negative score change is improvement; A-R terms are in unchanged representation units. No learning or held-out claim.',
            'shape_reconstruction':{'interpretation':'Compiled A/R uses the exact saved sampled query and endpoint banks; original sampled scalars are retained alongside arithmetic differences.',
                                    'batch_comparisons':shape_comparisons,
                                    'all_within_float32_reduction_tolerance':all(x['within_tolerance'] for x in shape_comparisons)},
            'states':{state:{'complete_scale':summarize_conditions(records[state],bootstrap=args.bootstrap,seed=args.seed),
                             'action_and_shape':summarize_actions(by_state[state])} for state in STATES}}
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        receipt.update(status='completed' if audit_receipt['status']=='completed' else 'completed_fixture_only',
            conditions_sha256=digest(path),summary_sha256=digest(args.output/'summary.json'))
    except BaseException as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error)); raise
    finally:
        if dataset is not None: dataset.close()
        receipt.update(wall_seconds=time.time()-started,cpu_seconds=time.process_time()-cpu_started)
        (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',required=True,type=Path);p.add_argument('--config',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--seed',type=int,default=6042026)
    a=p.parse_args()
    if a.threads<1 or a.bootstrap<1:p.error('threads and bootstrap must be positive')
    run(a)


if __name__=='__main__':main()
