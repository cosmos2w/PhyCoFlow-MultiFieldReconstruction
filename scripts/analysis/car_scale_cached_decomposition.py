#!/usr/bin/env python3
"""Decompose cached matched Scale scores without model inference or Shape scoring."""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, os, time
from pathlib import Path
import sys
ROOT=Path(os.environ.get('CAR_REPO_ROOT',Path(__file__).resolve().parents[2])).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
import yaml
from phycoflow_reconstruction.coherence import build_enabled_families
from phycoflow_reconstruction.coherence.adaptive_fitting import spectral_filter, validate_bank, validate_bank_target_binding
from phycoflow_reconstruction.coherence.query_runtime import _scale_statistics
from phycoflow_reconstruction.coherence.query_sampling import full_target_query_batch, index_pair
from phycoflow_reconstruction.coherence.scale_compute import sampled_scale, third_order_scores
from phycoflow_reconstruction.data.factory import open_field_dataset
from scripts.posttraining.assess_car_round2_checkpoints import matched_rows

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def ref(path):return {'path':str(Path(path).resolve(strict=True)),'sha256':sha(path)}
def check_ref(item):
 p=Path(item['path']).resolve(strict=True)
 if sha(p)!=item['sha256']:raise ValueError(f'hash mismatch: {p}')
 return p

def fair_terms(draw,target):
 """Vector energy along last axis, keeping all non-feature dimensions."""
 j=len(draw)
 if j<2:raise ValueError('at least two independent draws required')
 attraction=torch.linalg.vector_norm(draw-target,dim=-1).mean(0)
 separation=sum((torch.linalg.vector_norm(draw[a]-draw[b],dim=-1) for a,b in itertools.combinations(range(j),2)),torch.zeros_like(attraction))/(j*(j-1))
 return attraction,separation

def allocate(parts):
 """Each part is a per-draw squared distance group; sum parts / total norm."""
 total=sum((x.sum(dim=tuple(range(1,x.ndim))) for x in parts),torch.zeros(parts[0].shape[0],dtype=parts[0].dtype))
 root=total.sqrt();out=[]
 for x in parts:
  denom=root.reshape((-1,)+(1,)*(x.ndim-1));out.append(torch.where(denom>0,x/denom.clamp_min(torch.finfo(x.dtype).tiny),torch.zeros_like(x)))
 return out

def attributed_fair(parts_target,parts_pairs,j):
 a=allocate(parts_target);r=[torch.zeros_like(x[0]) for x in a]
 for parts in parts_pairs:
  for index,value in enumerate(allocate(parts)):r[index]+=value[0]/(j*(j-1))
 return [(x.mean(0),y) for x,y in zip(a,r)]

def triple_parts(family,draws,target,bank):
 fm=spectral_filter(family,draws);ft=spectral_filter(family,target)
 mass=torch.as_tensor(bank['region_masses'],dtype=fm.dtype);mass=mass/mass.sum(-1,keepdim=True)
 scales=torch.as_tensor(bank['triple_scales'],dtype=fm.dtype)
 xs=[];ys=[]
 for region,terms in bank['triples']:
  w=mass[region]
  def desc(x):
   factors=[]
   for band,field in terms:
    value=x[...,band,:,field];center=(value*w).sum(-1,keepdim=True)
    factors.append((value-center)/scales[region,band,field])
   return torch.asinh((factors[0]*factors[1]*factors[2]*w).sum(-1))
  xs.append(desc(fm));ys.append(desc(ft))
 x=torch.stack(xs,-1);y=torch.stack(ys,-1)
 a,r=fair_terms(x.unsqueeze(-1),y.unsqueeze(-1))
 expected=third_order_scores(fm,ft,bank['triples'],bank['region_masses'],bank['triple_scales'])
 torch.testing.assert_close(a-r,expected,rtol=2e-9,atol=2e-10)
 return a,r

def decompose(family,draws,target,bank,prefit):
 """Inputs J,B,N,C; all transformed inputs are model-unit matched endpoints."""
 ts=_scale_statistics(family,target);ds=[_scale_statistics(family,x) for x in draws]
 coef,means,desc=(torch.stack([x[i] for x in ds]) for i in range(3));tc,tm,td=ts
 j,b,k,c=coef.shape;f=family.filters.to(coef);s=family.band_scale.to(coef);m=family.mass.to(coef);bw=family._band_weights(family.compute_dtype,coef.device);floor=family.power_floor.to(coef)
 z=coef.unsqueeze(-3)*f[None,None,:,:,None]/s[None,None,:,None,:]/m.sum().sqrt()
 y=tc.unsqueeze(-3)*f[None,:,:,None]/s[None,:,None,:]/m.sum().sqrt()
 # Convert [J,B,S,K,C] to [J,B,S,C,K].
 z=z.permute(0,1,2,4,3);y=y.permute(0,1,3,2)
 sa,sr=fair_terms(z,y)
 signed_a=(sa*bw[None,:,None]).sum((-1,-2))/(c*c);signed_r=(sr*bw[None,:,None]).sum((-1,-2))/(c*c)
 pa=[];pr=[];ra=[];rr=[]
 def rho(x,i,l):
  xi=x[...,i];xl=x[...,l];ff=f.square();pi=torch.einsum('...k,sk->...s',xi.square(),ff);pl=torch.einsum('...k,sk->...s',xl.square(),ff);cross=torch.einsum('...k,sk->...s',xi*xl,ff)
  return cross/((pi+floor[:,i])*(pl+floor[:,l])).sqrt()
 for i,l in itertools.combinations(range(c),2):
  plus=fair_terms((z[:,:,:,i]+z[:,:,:,l])/math.sqrt(2),(y[:,:,i]+y[:,:,l])/math.sqrt(2));minus=fair_terms((z[:,:,:,i]-z[:,:,:,l])/math.sqrt(2),(y[:,:,i]-y[:,:,l])/math.sqrt(2))
  pa.append((plus[0]+minus[0])*.5);pr.append((plus[1]+minus[1])*.5)
  a,r=fair_terms(rho(coef,i,l).unsqueeze(-1),rho(tc,i,l).unsqueeze(-1));ra.append(a);rr.append(r)
 pa=torch.stack(pa,-1);pr=torch.stack(pr,-1);ra=torch.stack(ra,-1);rr=torch.stack(rr,-1);npair=pa.shape[-1]
 signed_a+=(pa*bw[None,:,None]).sum((-1,-2))*(1-1/c)/npair;signed_r+=(pr*bw[None,:,None]).sum((-1,-2))*(1-1/c)/npair
 rho_a=ra.mean((-1,-2));rho_r=rr.mean((-1,-2))
 # Joint group squares [J,S,C] plus piece means [J,P,C], for B=1.
 if b!=1:raise ValueError('one condition per cached row required')
 pw=family.piece_measures.sum(dim=1).to(coef);rootmean=(family.settings.mean_weight*pw/pw.sum()).sqrt()[:,None]/family.piece_mean_scale.to(coef)
 diffz=z[:,0]-y[0];diffm=(means[:,0]-tm[0])*rootmean
 p0=[diffz.square().sum(-1),diffm.square()]
 pairs=[]
 for u,v in itertools.combinations(range(j),2):pairs.append([(z[u:u+1,0]-z[v:v+1,0]).square().sum(-1),((means[u:u+1,0]-means[v:v+1,0])*rootmean).square()])
 joint=attributed_fair(p0,pairs,j)
 dz=(desc[:,0]-td[0]).reshape(j,f.shape[0],c);lpairs=[[(desc[u:u+1,0]-desc[v:v+1,0]).reshape(1,f.shape[0],c).square()] for u,v in itertools.combinations(range(j),2)]
 logp=attributed_fair([dz.square()],lpairs,j)[0]
 ta,tr=triple_parts(family,draws,target,bank);tw=torch.as_tensor(bank['target_weights']['scale.third_order'],dtype=ta.dtype)
 blocks={'signed':(signed_a[0],signed_r[0]),'joint':(sum(x[0].sum() for x in joint),sum(x[1].sum() for x in joint)),'log_power_self':(logp[0].sum(),logp[1].sum()),'rho':(rho_a[0],rho_r[0]),'third_order':((ta*tw).sum(),(tr*tw).sum())}
 expected=sampled_scale(coef,tc,full_target_query_batch(c),filters=f,scales=s,mass=m,power_floor=floor,band_weights=bw,self_descriptors=desc,target_self_descriptor=td,mean_kwargs={'draw_piece_means':means,'target_piece_means':tm,'piece_weights':pw,'mean_scales':family.piece_mean_scale,'mean_weight':family.settings.mean_weight})
 for key,(a,r) in blocks.items():
  if key!='third_order':torch.testing.assert_close(a-r,expected[key][0],rtol=2e-9,atol=2e-10)
 def vals(pair):
  a,r=pair;return {'attraction':a.detach().cpu().tolist(),'separation':r.detach().cpu().tolist(),'score':(a-r).detach().cpu().tolist()}
 block={k:vals(v) for k,v in blocks.items()};scores={k:float(v[0]-v[1]) for k,v in blocks.items()}
 base=scores['signed']+.25*scores['joint']+scores['log_power_self']+scores['rho']
 weighted=sum((.25 if k=='joint' else 1)*prefit['scale.'+k]*scores[k] for k in ('signed','joint','log_power_self','rho'))
 expanded=weighted+prefit['scale.third_order']*scores['third_order']
 return {'blocks':block,'definitions':{'Published_Core_Default':base,'Complete_Core_Prefit':weighted,'Complete_Expanded_Prefit':expanded},
  'localization':{'signed_self_band_field':vals((sa[0],sr[0])),'signed_pair_band':vals((pa[0],pr[0])),'rho_pair_band':vals((ra[0],rr[0])),'joint_spectral_band_field_attribution':vals(joint[0]),'joint_piece_field_attribution':vals(joint[1]),'log_power_band_field_attribution':vals(logp),'third_order_leaf':vals((ta[0],tr[0]))}}

def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();torch.set_num_threads(2);started=time.time();cpu_started=time.process_time()
 spec=read(args.spec);out=args.output.resolve();campaign=Path(spec['campaign']).resolve(strict=True)
 if campaign/'assessment' not in out.parents or out.exists():raise ValueError('new campaign assessment attempt required')
 freeze=read(check_ref(spec['formal_freeze']));formal_cfg=yaml.safe_load(check_ref(freeze['arms']['FFM_SSS_EpochCalibrated_PI_Repair']).read_text());declared=formal_cfg['coherence']['scalable_anchored_coherence']
 if (check_ref(declared['adaptive']['bank_artifact'])!=check_ref(spec['expanded_bank']) or check_ref(declared['common_prefit'])!=check_ref(spec['prefit'])):raise ValueError('bank or prefit differs from frozen formal recipe')
 source_ref=spec['source_receipt'];source=read(check_ref(source_ref));panel=read(check_ref(spec['panel']));core_run=Path(spec['core_run']).resolve(strict=True);cfg=yaml.safe_load((core_run/'resolved_config.yaml').read_text());ds=open_field_dataset(cfg['dataset'],split='train')
 try:
  family=build_enabled_families(cfg['coherence'],ds.data_spec,ds.normalizer)['multiscale_spectrum'];state=core_run/'artifacts/multiscale_spectrum_family.pt';manifest=read(core_run/'run_manifest.json')
  if sha(state)!=manifest['coherence_family_state_sha256s']['multiscale_spectrum']:raise ValueError('frozen family mismatch')
  family.load_state_artifact(torch.load(state,map_location='cpu',weights_only=True));bank=read(check_ref(spec['expanded_bank']));prefit_doc=read(check_ref(spec['prefit']));prefit=prefit_doc['coefficients'];binding=prefit_doc['binding']
  validate_bank(bank,field_order=source['field_order'],support_identity=binding['support_sha256'],source_identity=source['source_sha256']);validate_bank_target_binding(binding,bank)
  if binding['source_checkpoint_sha256']!=source['source_sha256'] or binding['normalizer_sha256']!=source['normalizer_sha256'] or ds.normalizer.digest()!=source['normalizer_sha256'] or binding['field_order']!=source['field_order']:raise ValueError('SOURCE/prefit scientific identities differ')
  expected={'scale.signed':2.0,'scale.joint':1.2835023021293974,'scale.log_power_self':.5,'scale.rho':.8190801090444172,'scale.third_order':.5}
  if any(not math.isclose(prefit.get(k,float('nan')),v,rel_tol=1e-12) for k,v in expected.items()) or family.settings.joint_energy_score_weight!=.25 or family.relation_extension['self_weight']!=1 or family.relation_extension['rho_weight']!=1:raise ValueError('Scale recipe does not match the declared definitions')
  ids=source['sample_ids'];query=torch.tensor(panel['query_indices'],dtype=torch.long)
  if source['status']!='completed' or source['members']!=4 or len(ids)!=len(set(ids)) or source['panel_sha256']!=spec['panel']['sha256'] or query.numel()!=4096 or query.unique().numel()!=4096:raise ValueError('invalid source panel')
  children={name:read(check_ref(item['receipt'])) for name,item in spec['children'].items()}; assessments={name:read(check_ref(item['assessment'])) for name,item in spec['children'].items() if item.get('assessment')}
  for name,rec in children.items():
   if rec['status']!='completed' or rec['members']!=4 or rec['sample_ids']!=ids or rec['panel_sha256']!=source['panel_sha256'] or rec['source_bank_sha256']!=source_ref['sha256']:raise ValueError('unmatched candidate receipt: '+name)
   if name in assessments:
    a=assessments[name]
    if (a['sample_ids']!=ids or a['bank_receipt_sha256']!=spec['children'][name]['receipt']['sha256'] or a['source_bank_sha256']!=source_ref['sha256'] or a['panel_sha256']!=source['panel_sha256'] or a['checkpoint_sha256']!=rec['checkpoint_sha256'] or a['normalizers']['scale']!=spec['frozen_train_scale_normalizer']):raise ValueError('saved assessment identity differs: '+name)
  records=[];out.mkdir(parents=True)
  for i,sid in enumerate(ids):
   path=Path(source_ref['path']).parent/f'{i:04d}.pt'
   if sha(path)!=source['files_sha256'][i]:raise ValueError('source row mismatch')
   srow=torch.load(path,map_location='cpu',weights_only=True);target=srow['truth_model'][:,query];family._require_ready(target,srow['coordinates_model'][:,query]);row={'sample_id':sid,'member_seeds':srow['member_seeds'],'models':{}}
   for name,rec in [('SOURCE',source),*children.items()]:
    member=srow if name=='SOURCE' else torch.load(Path(spec['children'][name]['receipt']['path']).parent/f'{i:04d}.pt',map_location='cpu',weights_only=True)
    if name!='SOURCE':
     if sha(Path(spec['children'][name]['receipt']['path']).parent/f'{i:04d}.pt')!=rec['files_sha256'][i]:raise ValueError('child row mismatch')
     matched_rows(srow,member)
    if srow['sample_id']!=sid:raise ValueError('source sample order mismatch')
    with torch.no_grad():row['models'][name]=decompose(family,member['members_model'][:,:,query],target,bank,prefit)
   for name,assessment in assessments.items():
    child=assessment['candidates'][assessment['arm']]['per_condition_scores'][i]['scale'];source_saved=assessment['source']['per_condition_scores'][i]['scale']
    for key,value in ((name,child),('SOURCE',source_saved)):
     got=row['models'][key]['definitions']['Published_Core_Default'];delta=abs(got-value)
     if delta>1e-7+1e-5*abs(value):raise ValueError(f'published Scale reconstruction mismatch {sid} {key}: {got} vs {value}')
   records.append(row)
  definitions=('Published_Core_Default','Complete_Core_Prefit','Complete_Expanded_Prefit');summary={};rng=np.random.default_rng(6042026);indices=rng.integers(0,len(ids),size=(2000,len(ids)))
  for name in children:
   summary[name]={}
   for key in definitions:
    delta=np.asarray([row['models'][name]['definitions'][key]-row['models']['SOURCE']['definitions'][key] for row in records]);boots=delta[indices].mean(axis=1)
    summary[name][key]={'mean_raw_change':float(delta.mean()),'interval_95_raw':np.quantile(boots,[.025,.975]).tolist(),'mean_change_over_original_frozen_train_scale':float(delta.mean()/spec['frozen_train_scale_normalizer'])}
  for left,right in itertools.combinations(children,2):
   summary[f'{right}_minus_{left}']={}
   for key in definitions:
    delta=np.asarray([row['models'][right]['definitions'][key]-row['models'][left]['definitions'][key] for row in records]);summary[f'{right}_minus_{left}'][key]={'mean_raw_change':float(delta.mean()),'interval_95_raw':np.quantile(delta[indices].mean(axis=1),[.025,.975]).tolist(),'mean_change_over_original_frozen_train_scale':float(delta.mean()/spec['frozen_train_scale_normalizer'])}
  component_pairs=[('SOURCE',name) for name in children]+list(itertools.combinations(children,2));component_summary={}
  for left,right in component_pairs:
   views={}
   for section in ('blocks','localization'):
    views[section]={}
    for key in records[0]['models'][left][section]:
     views[section][key]={}
     for term in ('attraction','separation','score'):
      delta=np.asarray([np.asarray(row['models'][right][section][key][term])-np.asarray(row['models'][left][section][key][term]) for row in records])
      views[section][key][term]={'mean_raw_change':np.mean(delta,axis=0).tolist(),'interval_95_raw':np.quantile(np.mean(delta[indices],axis=1),[.025,.975],axis=0).tolist()}
   component_summary[f'{right}_minus_{left}']=views
  output={'schema':'car_scale_cached_decomposition_v1','status':'completed','spec':ref(args.spec),'script':ref(__file__),'sample_count':len(ids),'sample_ids':ids,'field_order':source['field_order'],'split':spec['split'],'members':4,'support':'Q4096 frozen support; complete field/pair queries, not native 40300 support','prefit_coefficients':{k:v for k,v in prefit.items() if k.startswith('scale.')},'band_weights':family._band_weights(family.compute_dtype,torch.device('cpu')).tolist(),'pair_order':[list(x) for x in itertools.combinations(source['field_order'],2)],'third_order_queries':bank['triples'],'third_order_weights':bank['target_weights']['scale.third_order'],'attribution':'joint and log-power allocations are squared group distance divided by the total norm, zero at zero distance; additive attribution, not independent proper scores','fair_score':'attraction minus positive separation; all six member pairs, denominator J*(J-1)','summary':summary,'component_paired_summary':component_summary,'records':records,'elapsed_seconds':time.time()-started,'process_cpu_seconds':time.process_time()-cpu_started}
  (out/'decomposition.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
 finally:ds.close()
if __name__=='__main__':main()
