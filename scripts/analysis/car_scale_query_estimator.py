#!/usr/bin/env python3
"""Compare fixed-endpoint grouped Scale query estimates with exact enumeration."""
from __future__ import annotations
import argparse,hashlib,itertools,json,math,os,sys,time
from pathlib import Path
ROOT=Path(os.environ.get('CAR_REPO_ROOT',Path(__file__).resolve().parents[2])).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
from phycoflow_reconstruction.coherence.query_sampling import ConditionQueryGroups,HierarchicalQuerySampler

SEEDS=(1,2,3,4)
DEFINITIONS=('Published_Core_Default','Complete_Core_Prefit','Complete_Expanded_Prefit')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def query_score(model,batch,band_weights,coefficients):
 """Reweight exact field/pair leaves with one actual sampler draw."""
 c=len(model['localization']['signed_self_band_field']['score'][0]);loc=model['localization'];bw=np.asarray(band_weights)
 a=np.asarray(loc['signed_self_band_field']['score']);p=np.asarray(loc['signed_pair_band']['score']);r=np.asarray(loc['rho_pair_band']['score'])
 unary=sum(d.weight*float(np.dot(bw,a[:,d.index])) for d in batch.self_queries)
 pair=sum(d.weight*float(np.dot(bw,p[:,d.index])) for d in batch.pair_queries)
 signed=unary/c+(1-1/c)*pair
 rho=sum(d.weight*float(r[:,d.index].mean()) for d in batch.pair_queries)
 blocks=model['blocks'];joint=blocks['joint']['score'];logp=blocks['log_power_self']['score'];triple=blocks['third_order']['score']
 base=signed+.25*joint+logp+rho
 weighted=coefficients['scale.signed']*signed+.25*coefficients['scale.joint']*joint+coefficients['scale.log_power_self']*logp+coefficients['scale.rho']*rho
 return {'Published_Core_Default':base,'Complete_Core_Prefit':weighted,'Complete_Expanded_Prefit':weighted+coefficients['scale.third_order']*triple}

def compare(doc,normalizer):
 if doc.get('schema')!='car_scale_cached_decomposition_v1' or doc.get('status')!='completed' or doc['members']!=4:raise ValueError('completed J4 exhaustive decomposition required')
 rows=doc['records'];n=len(rows);fields=doc['field_order'];models=list(rows[0]['models']);weights=doc['band_weights'];co=doc['prefit_coefficients']
 if n!=doc['sample_count'] or [r['sample_id'] for r in rows]!=doc['sample_ids'] or not math.isfinite(normalizer) or normalizer<=0:raise ValueError('sample or frozen scale unit mismatch')
 # Within a seed, each real B64 uses eight independent samplers on contiguous groups;
 # across SOURCE/children the exact same query draw is reused (paired comparison).
 records=[{'sample_id':r['sample_id'],'models':{name:{'exhaustive':r['models'][name]['definitions'],'grouped_by_seed':{}} for name in models}} for r in rows]
 draws=[]
 for seed in SEEDS:
  samplers=[HierarchicalQuerySampler(fields,seed=seed+7919*i,hierarchy_weight=0.0) for i in range(8)]
  for start in range(0,n,64):
   count=min(64,n-start);queries=tuple(s.draw() for s in samplers);groups=ConditionQueryGroups(queries);batch_entry={'seed':seed,'start':start,'count':count,'groups':[]}
   for sl,q in groups.slices(count):
    batch_entry['groups'].append({'begin':start+sl.start,'end':start+sl.stop,'self_queries':[vars(d) for d in q.self_queries],'pair_queries':[vars(d) for d in q.pair_queries]})
    for index in range(start+sl.start,start+sl.stop):
     for name in models:records[index]['models'][name]['grouped_by_seed'][str(seed)]=query_score(rows[index]['models'][name],q,weights,co)
   draws.append(batch_entry)
   for sampler in samplers:sampler.finish()
 summary={}
 for name in models:
  summary[name]={}
  for definition in DEFINITIONS:
   exact=np.asarray([r['models'][name]['exhaustive'][definition] for r in records]);values=np.asarray([[r['models'][name]['grouped_by_seed'][str(seed)][definition] for r in records] for seed in SEEDS]);panel=values.mean(axis=1);err=panel-exact.mean()
   summary[name][definition]={'exhaustive_panel_mean':float(exact.mean()),'grouped_panel_means_by_seed':panel.tolist(),'grouped_minus_exhaustive_by_seed':err.tolist(),'grouped_minus_exhaustive_mean':float(err.mean()),'grouped_minus_exhaustive_in_frozen_train_scale':float(err.mean()/normalizer),'query_seed_standard_deviation_panel_means':float(panel.std(ddof=1)),'condition_mean_query_variance':float(values.var(axis=0,ddof=1).mean())}
 for left,right in itertools.combinations(models,2):
  key=f'{right}_minus_{left}';summary[key]={}
  for definition in DEFINITIONS:
   xa=np.asarray([r['models'][left]['exhaustive'][definition] for r in records]);xb=np.asarray([r['models'][right]['exhaustive'][definition] for r in records]);pairs=np.asarray([[r['models'][right]['grouped_by_seed'][str(seed)][definition]-r['models'][left]['grouped_by_seed'][str(seed)][definition] for r in records] for seed in SEEDS]);pooled=pairs.mean(axis=1);truth=float((xb-xa).mean())
   summary[key][definition]={'exhaustive_paired_change':truth,'exhaustive_paired_change_in_frozen_train_scale':truth/normalizer,'grouped_paired_changes_by_seed':pooled.tolist(),'query_seed_standard_deviation_paired_change':float(pooled.std(ddof=1)),'mean_grouped_minus_exhaustive':float(pooled.mean()-truth)}
 return {'schema':'car_scale_cached_grouped_query_estimator_v1','status':'completed','split':doc['split'],'sample_count':n,'sample_ids':doc['sample_ids'],'fields':fields,'members':4,'member_policy':'all four cached J4 members; query law varies while endpoints are fixed','query_seeds':list(SEEDS),'batch_size':64,'groups':8,'query_law':'uniform HierarchicalQuerySampler, hierarchy_weight=0; one unary and three pair draws, group streams offset by 7919; same sampled queries across compared models','scope':'four query seeds on one matched panel; estimator variation only, not independent datasets, model efficacy, trained trajectory or native-40300 support','original_frozen_train_scale_normalizer':normalizer,'summary':summary,'query_draws':draws,'records':records}

def main():
 p=argparse.ArgumentParser();p.add_argument('--decomposition',type=Path,required=True);p.add_argument('--normalizer',type=float,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.time();cpu=time.process_time();d=json.loads(a.decomposition.read_text());result=compare(d,a.normalizer);out=a.output.resolve()
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);result.update(input={'path':str(a.decomposition.resolve(strict=True)),'sha256':sha(a.decomposition)},script_sha256=sha(__file__),elapsed_seconds=time.time()-start,process_cpu_seconds=time.process_time()-cpu)
 (out/'query_estimator.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
