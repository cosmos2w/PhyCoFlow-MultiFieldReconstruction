#!/usr/bin/env python3
"""Matched cached J4 physical field risk inside and outside frozen Q4096."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(os.environ.get('CAR_REPO_ROOT',Path(__file__).resolve().parents[2])).resolve();sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
from scripts.posttraining.assess_car_round2_checkpoints import matched_rows

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def checked(r):
 p=Path(r['path']).resolve(strict=True)
 if sha(p)!=r['sha256']:raise ValueError('hash mismatch: '+str(p))
 return p

def field_risks(members,truth,masses,selected,scale):
 """Returns physical-unit field risks for one matched condition."""
 x=members.to(torch.float64);y=truth.to(torch.float64);m=masses.to(torch.float64).reshape(-1);scale=scale.to(torch.float64).reshape(-1)
 if x.ndim!=4 or x.shape[0]!=4 or x.shape[1]!=1 or x.shape[1:]!=y.shape or x.shape[2]!=len(m) or x.shape[-1]!=len(scale) or not torch.isfinite(x).all() or not torch.isfinite(y).all() or not torch.isfinite(m).all() or not (m>=0).all():raise ValueError('invalid J4 fields, truth, scales or masses')
 count=len(m)
 if selected.numel()!=4096 or bool((selected<0).any()) or bool((selected>=count).any()):raise ValueError('invalid selected indices')
 mask=torch.zeros(count,dtype=torch.bool);mask[selected]=True
 if selected.numel()!=4096 or mask.sum().item()!=4096 or count!=40300:raise ValueError('invalid fixed/native support')
 total=m.sum();ms=m[mask].sum();mc=m[~mask].sum()
 if total<=0 or ms<=0 or mc<=0:raise ValueError('empty physical mass')
 err=(x-y.unsqueeze(0)).squeeze(1)*scale[None,None,:]
 member=err.square().mean(0);mean=err.mean(0);signed=mean.square()-err.var(0,unbiased=True)/4
 def sums(v):
  ins=(v[mask]*m[mask,None]).sum(0);outs=(v[~mask]*m[~mask,None]).sum(0)
  return {'native':((ins+outs)/total).tolist(),'selected_region':(ins/ms).tolist(),'complement_region':(outs/mc).tolist(),'selected_ht':(ins/total*(count/selected.numel())).tolist()}
 return {'selected_mass_fraction':float(ms/total),'complement_mass_fraction':float(mc/total),'member_mse':sums(member),'signed_mean_risk':sums(signed)}

def analyze(spec,limit):
 src=Path(spec['source_receipt']['path']);source=json.loads(checked(spec['source_receipt']).read_text());panel=json.loads(checked(spec['panel']).read_text());query=torch.tensor(panel['query_indices'],dtype=torch.long);ids=source['sample_ids'];children={name:json.loads(checked(item['receipt']).read_text()) for name,item in spec['children'].items()}
 if source['status']!='completed' or source['members']!=4 or source['panel_sha256']!=spec['panel']['sha256'] or panel['sample_ids']!=ids or len(ids)!=len(set(ids)):raise ValueError('source and panel identity mismatch')
 for name,child in children.items():
  if child['status']!='completed' or child['members']!=4 or child['sample_ids']!=ids or child['panel_sha256']!=source['panel_sha256'] or child['source_bank_sha256']!=spec['source_receipt']['sha256'] or child.get('field_order',source['field_order'])!=source['field_order'] or child.get('normalizer_sha256',source['normalizer_sha256'])!=source['normalizer_sha256']:raise ValueError('child receipt mismatch '+name)
 assessments={name:json.loads(checked(item['assessment']).read_text()) for name,item in spec['children'].items() if item.get('assessment')}
 for name,assessment in assessments.items():
  if assessment['sample_ids']!=ids or assessment['bank_receipt_sha256']!=spec['children'][name]['receipt']['sha256'] or assessment['source_bank_sha256']!=spec['source_receipt']['sha256'] or assessment['checkpoint_sha256']!=children[name]['checkpoint_sha256']:raise ValueError('saved assessment binding differs')
 rows=[]
 for i,sid in enumerate(ids[:limit]):
  src_path=src.parent/f'{i:04d}.pt'
  if sha(src_path)!=source['files_sha256'][i]:raise ValueError('SOURCE row hash mismatch')
  srow=torch.load(src_path,map_location='cpu',weights_only=True)
  if sid!=srow['sample_id']:raise ValueError('source ID mismatch')
  models={}
  for name,child in [('SOURCE',source),*children.items()]:
   if name=='SOURCE':r=srow
   else:
    path=Path(spec['children'][name]['receipt']['path']).parent/f'{i:04d}.pt'
    if sha(path)!=child['files_sha256'][i]:raise ValueError('child row hash mismatch')
    r=torch.load(path,map_location='cpu',weights_only=True);matched_rows(srow,r)
   models[name]=field_risks(r['members_model'],r['truth_model'],r['native_masses'],query,r['normalizer_scale'])
  for name,assessment in assessments.items():
   for model,label in (('SOURCE','source'),(name,'child')):
    for kind in ('signed_mean_risk','member_mse'):
     expected=assessment['per_condition_fields'][label][i][kind]
     error=np.abs(np.asarray(models[model][kind]['native'])-np.asarray(expected));member=np.asarray(assessment['per_condition_fields'][label][i]['member_mse'])
     if np.any(error>1e-7+1e-5*np.maximum(np.abs(expected),member)):raise ValueError(f'native risk reconstruction differs: {sid} {model} {kind}')
  rows.append({'sample_id':sid,'member_seeds':srow['member_seeds'],'models':models})
 return rows,source,children

def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,action='append',required=True);p.add_argument('--fidelity-reference',type=Path,required=True);p.add_argument('--reference-sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--fixture',action='store_true');a=p.parse_args();start=time.time();cpu=time.process_time();torch.set_num_threads(2)
 if sha(a.fidelity_reference)!=a.reference_sha256:raise ValueError('fidelity reference changed')
 ref=json.loads(a.fidelity_reference.read_text())['reference'];output=a.output.resolve();specs=[json.loads(p.read_text()) for p in a.spec];campaign=Path(specs[0]['campaign']).resolve(strict=True)
 if campaign/'assessment' not in output.parents or output.exists() or any(Path(s['campaign']).resolve()!=campaign for s in specs):raise ValueError('new shared campaign assessment required')
 output.mkdir(parents=True);results={}
 for spath,spec in zip(a.spec,specs):
  rows,source,children=analyze(spec,1 if a.fixture else None);split=spec['split'];scale=torch.as_tensor(torch.load(Path(spec['source_receipt']['path']).parent/'0000.pt',map_location='cpu',weights_only=True)['normalizer_scale'],dtype=torch.float64);allow=np.asarray(ref['absolute_allowance'])*scale.square().numpy()
  if list(ref['field_names'])!=source['field_order']:raise ValueError('field allowance order differs')
  n=len(rows);rng=np.random.default_rng(6042026);idx=rng.integers(0,n,size=(2000,n));summary={}
  for left,right in [('SOURCE',name) for name in children]+list(__import__('itertools').combinations(children,2)):
   contrasts={}
   for kind in ('signed_mean_risk','member_mse'):
    contrasts[kind]={}
    for region in ('native','selected_region','complement_region','selected_ht'):
     delta=np.asarray([np.asarray(row['models'][right][kind][region])-np.asarray(row['models'][left][kind][region]) for row in rows]);mean=delta.mean(0);interval=np.quantile(delta[idx].mean(1),[.025,.975],axis=0)
     contrasts[kind][region]={'mean_physical_units_squared':mean.tolist(),'interval_95_physical_units_squared':interval.tolist(),'mean_over_original_frozen_allowance':(mean/allow).tolist(),'interval_95_over_original_frozen_allowance':(interval/allow).tolist()}
   summary[right+'_minus_'+left]=contrasts
  # The same region masses reproduce native risk for every condition, model and field.
  for row in rows:
   for model in row['models'].values():
    for kind in ('signed_mean_risk','member_mse'):
     v=model[kind];reconstructed=model['selected_mass_fraction']*np.asarray(v['selected_region'])+model['complement_mass_fraction']*np.asarray(v['complement_region'])
     if not np.allclose(reconstructed,v['native'],rtol=1e-11,atol=1e-12):raise ValueError('native mass partition does not reconstruct')
  results[split]={'spec':{'path':str(spath.resolve()),'sha256':sha(spath)},'field_order':source['field_order'],'sample_count':n,'physical_allowance':allow.tolist(),'selected_mass_fraction_range':[min(r['models']['SOURCE']['selected_mass_fraction'] for r in rows),max(r['models']['SOURCE']['selected_mass_fraction'] for r in rows)],'summary':summary,'records':rows}
 result={'schema':'car_cached_field_support_v1','status':'completed_fixture_only' if a.fixture else 'completed','script':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},'definitions':'J4 signed mean risk = squared ensemble-mean error minus unbiased member variance/4; member MSE = mean per-member squared error; model residuals are converted to physical units using saved field scales and accumulated with native physical masses in float64. Selected HT = selected mass-weighted sum / total native mass * 40300/4096. Region means use region mass only.','interpretation':'same cached native endpoint values sliced by frozen Q4096 and complement; not selected-support model rollout, fresh-observation training law, or training-trajectory counterfactual. Float64 accumulation can differ slightly from historical float32 physical-risk arithmetic.','fidelity_reference':{'path':str(a.fidelity_reference.resolve()),'sha256':a.reference_sha256},'splits':results,'elapsed_seconds':time.time()-start,'process_cpu_seconds':time.process_time()-cpu,'test_locked':True}
 (output/'field_support.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
