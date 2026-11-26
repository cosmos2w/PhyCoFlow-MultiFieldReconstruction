"""Round 7 Scale decomposition with reusable frozen target statistics."""
import itertools
import math
import torch
from phycoflow_reconstruction.coherence.query_runtime import _scale_statistics
from phycoflow_reconstruction.coherence.query_sampling import full_target_query_batch
from phycoflow_reconstruction.coherence.scale_compute import sampled_scale
from phycoflow_reconstruction.coherence.adaptive_fitting import spectral_filter
from phycoflow_reconstruction.coherence.scale_compute import third_order_scores
from scripts.analysis.car_scale_cached_decomposition import fair_terms, attributed_fair

def triple_parts(family,draws,target,bank, *, target_filtered=None):
 fm=spectral_filter(family,draws);ft=spectral_filter(family,target) if target_filtered is None else target_filtered
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

def decompose(family,draws,target,bank,prefit, *, target_statistics=None, target_filtered=None):
 """Inputs J,B,N,C; all transformed inputs are model-unit matched endpoints."""
 ts=_scale_statistics(family,target) if target_statistics is None else target_statistics;ds=[_scale_statistics(family,x) for x in draws]
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
 ta,tr=triple_parts(family,draws,target,bank,target_filtered=target_filtered);tw=torch.as_tensor(bank['target_weights']['scale.third_order'],dtype=ta.dtype)
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
