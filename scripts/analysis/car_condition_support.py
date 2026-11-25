#!/usr/bin/env python3
"""Fixed TRAIN/VALID source-cache condition support summaries, no model evaluation."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ref(p):
 p=Path(p).resolve(strict=True);return {'path':str(p),'sha256':sha(p)}
def load_rows(receipt_ref,panel_ref,*,fixture):
 receipt_path=Path(receipt_ref['path']);panel_path=Path(panel_ref['path'])
 if sha(receipt_path)!=receipt_ref['sha256'] or sha(panel_path)!=panel_ref['sha256']:raise ValueError('SOURCE receipt or panel hash changed')
 receipt=json.loads(receipt_path.read_text());panel=json.loads(panel_path.read_text());ids=receipt['sample_ids'];split=panel['split']
 if (receipt['status']!='completed' or receipt['split']!=split or receipt['population']!=panel['population'] or receipt['panel_sha256']!=panel_ref['sha256'] or panel['sample_ids']!=ids or receipt['members']!=4 or receipt['native_points']!=40300 or len(set(ids))!=len(ids) or receipt['normalizer_sha256']!=panel['normalizer_sha256'] or receipt['field_order']!=panel['field_order']):raise ValueError('matched SOURCE panel identities required')
 fields=receipt['field_order'];t_index=fields.index('T');rows=[]
 for i,sid in enumerate(ids[:1] if fixture else ids):
  path=receipt_path.parent/f'{i:04d}.pt'
  if sha(path)!=receipt['files_sha256'][i]:raise ValueError('cached SOURCE row changed')
  row=torch.load(path,map_location='cpu',weights_only=True);x=row['truth_model'].double().squeeze(0);m=row['native_masses'].double().reshape(-1);obs=row['obs_values'].double().reshape(-1);obs_fields=row['obs_fields'].reshape(-1)
  if (row['sample_id']!=sid or row['sample_index']!=panel['sample_indices'][i] or x.shape!=(40300,len(fields)) or m.shape!=(40300,) or obs.numel()!=256 or obs_fields.numel()!=256 or not bool((obs_fields==t_index).all()) or not torch.isfinite(x).all() or not torch.isfinite(obs).all() or not torch.isfinite(m).all() or not (m>=0).all() or not (m.sum()>0)):raise ValueError('cached fields, observation or native physical mask mismatch')
  m=m/m.sum();mean=(x*m[:,None]).sum(0);rms=(x.square()*m[:,None]).sum(0).sqrt()
  rows.append({'sample_id':sid,'sample_index':row['sample_index'],'features':[float(obs.mean()),float(obs.std(unbiased=False)),*[float(y) for y in mean],*[float(y) for y in rms]]})
 return rows,receipt,panel

def subset_summary(v,z,train,tail,fields):
 means=v.mean(0);sd=v.std(0,ddof=0);rng=np.quantile(train,[.025,.975],axis=0);rng_tail=np.quantile(tail,[.025,.975],axis=0)
 out={}
 for i,name in enumerate(fields):
  out[name]={'mean':float(means[i]),'std_population':float(sd[i]),'mean_z_in_train_units':float(z[:,i].mean()),'outside_train_minmax_count':int(((v[:,i]<train[:,i].min())|(v[:,i]>train[:,i].max())).sum()),'outside_train_central95_count':int(((v[:,i]<rng[0,i])|(v[:,i]>rng[1,i])).sum()),'outside_tail_minmax_count':int(((v[:,i]<tail[:,i].min())|(v[:,i]>tail[:,i].max())).sum()),'outside_tail_central95_count':int(((v[:,i]<rng_tail[0,i])|(v[:,i]>rng_tail[1,i])).sum())}
 return out

def dist_summary(v,train,tail,mask):
 x=v[:,mask];a=train[:,mask];b=tail[:,mask];d=((x[:,None,:]-a[None,:,:])**2).sum(-1)**.5;dt=((x[:,None,:]-b[None,:,:])**2).sum(-1)**.5
 def stats(v):return {'mean':float(v.mean()),'median':float(np.median(v)),'min':float(v.min()),'max':float(v.max()),'p90':float(np.quantile(v,.9))}
 return {'feature_count':int(sum(mask)),'nearest_full_train':stats(d.min(1)),'nearest_last20pct_train':stats(dt.min(1)),'per_validation_condition_nearest_full':d.min(1).tolist(),'per_validation_condition_nearest_tail':dt.min(1).tolist()}

def main():
 p=argparse.ArgumentParser();p.add_argument('--train-spec',type=Path,required=True);p.add_argument('--validation-spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--fixture',action='store_true');a=p.parse_args();start=time.time();cpu=time.process_time();torch.set_num_threads(2)
 ts=json.loads(a.train_spec.read_text());vs=json.loads(a.validation_spec.read_text());campaign=Path(ts['campaign']).resolve(strict=True);out=a.output.resolve()
 if campaign/'assessment' not in out.parents or out.exists() or Path(vs['campaign']).resolve()!=campaign:raise ValueError('new same-campaign assessment attempt required')
 tr,trr,trp=load_rows(ts['source_receipt'],ts['panel'],fixture=a.fixture);va,var,vap=load_rows(vs['source_receipt'],vs['panel'],fixture=a.fixture)
 if (trr['source_sha256']!=var['source_sha256'] or trr['normalizer_sha256']!=var['normalizer_sha256'] or trr['field_order']!=var['field_order'] or trr['population']!=8000 or var['population']!=1000):raise ValueError('source, units, fields or population differ')
 fields=trr['field_order'];names=['observation_T_mean','observation_T_std']+['target_'+n+'_mass_mean' for n in fields]+['target_'+n+'_mass_rms' for n in fields]
 result={'schema':'car_condition_support_v1','status':'completed_fixture_only' if a.fixture else 'completed','script':ref(__file__),'inputs':{'train_spec':ref(a.train_spec),'validation_spec':ref(a.validation_spec),'train_source_receipt':ts['source_receipt'],'validation_source_receipt':vs['source_receipt'],'train_panel':ts['panel'],'validation_panel':vs['panel']},'feature_order':names,'feature_definition':'256 observed T values mean and population std; target native mass-normalized field mean and root-mean-square, all in frozen model units; physical mask represented by nonnegative native physical masses','standardization':'each feature uses TRAIN800 population mean and population standard deviation; no VALIDATION fitting','interpretation':'target-derived features are retrospective and unavailable for input-only deployment; panel and nearest-neighbor descriptions on one chronological trajectory, not causal or independent generalization evidence','train':tr,'validation':va,'test_locked':True}
 if not a.fixture:
  if len(tr)!=800 or len(va)!=100:raise ValueError('TRAIN800 and VALIDATION100 required')
  train=np.array([x['features'] for x in tr]);valid=np.array([x['features'] for x in va]);tail_idx=np.array([6400<=x['sample_index']<8000 for x in tr]);tail=train[tail_idx]
  if not len(tail):raise ValueError('TRAIN last20% panel subset empty')
  mean=train.mean(0);std=train.std(0,ddof=0)
  if not np.isfinite(train).all() or not np.isfinite(valid).all() or np.any(std<=1e-12):raise ValueError('finite nondegenerate TRAIN feature scales required')
  zt=(train-mean)/std;zv=(valid-mean)/std;z_tail=zt[tail_idx];mask_obs=[True]*2+[False]*10;mask_target=[False]*2+[True]*10
  result.update(train_count=len(tr),validation_count=len(va),tail_count=len(tail),tail_definition='training sample indices 6400–7999; subset of fixed TRAIN800',train_feature_mean=mean.tolist(),train_feature_std=std.tolist(),train_summary=subset_summary(train,zt,train,tail,names),tail_summary=subset_summary(tail,z_tail,train,tail,names),validation_summary=subset_summary(valid,zv,train,tail,names),nearest_distances={'all_12_features':dist_summary(zv,zt,z_tail,[True]*12),'observation_only':dist_summary(zv,zt,z_tail,mask_obs),'target_only':dist_summary(zv,zt,z_tail,mask_target)})
 out.mkdir(parents=True);result.update(elapsed_seconds=time.time()-start,process_cpu_seconds=time.process_time()-cpu);(out/'condition_support.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
