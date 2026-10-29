#!/usr/bin/env python3
"""Build V4 cohort and paired-endpoint figures from saved arrays and metric CSVs."""
from __future__ import annotations
import argparse, json, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import metrics as met

BASE=Path(__file__).resolve().parents[2]
DISPLAY={'gl_rbf_cq':'FFM','deeponet':'DON','diffusion_pde':'DiffusionPDE','geofno':'GeoFNO','mimonet':'MIMONet','senseiver':'Senseiver'}
PALETTE={'gl_rbf_cq':'#007f82','deeponet':'#89802a','diffusion_pde':'#bd5a4c','geofno':'#426dab','mimonet':'#93688d','senseiver':'#d88724'}
CMAPS=('cividis','magma','inferno','RdBu_r','RdBu_r')
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'pdf.fonttype':42,'figure.dpi':130})

def save(fig, path):
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'),dpi=170,bbox_inches='tight')
    plt.close(fig)

def run():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics',type=Path,default=BASE/'V4/data/metrics')
    p.add_argument('--out',type=Path,default=BASE/'V4/figures')
    p.add_argument('--gt',type=Path,default=BASE/'data/gt_test100.npz')
    p.add_argument('--model',action='append',required=True,help='NAME=NPZ path, matching metrics inputs')
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    paths={x.split('=',1)[0]:Path(x.split('=',1)[1]) for x in a.model}
    definitions=json.loads((a.metrics/'definitions.json').read_text())
    names=definitions['models']
    if set(paths)!=set(names): raise ValueError('model registry does not match metrics')
    for name in names:
        if met.digest(paths[name])!=definitions['inputs'][name]['sha256']: raise ValueError('model archive hash differs from metrics: '+name)
    if met.digest(a.gt)!=definitions['inputs']['gt']['sha256']: raise ValueError('GT archive differs from metrics')
    frame=pd.read_csv(a.metrics/'frame_metrics.csv'); cand=pd.read_csv(a.metrics/'candidates.csv'); cohort=pd.read_csv(a.metrics/'cohort.csv')
    f=frame[frame.bins==12].copy(); f['mean_l2']=(f.l2_x+f.l2_y)/2
    support=cand.groupby(['global_index','pair'],as_index=False).shape_support.agg(['min','max']).reset_index()
    if (support['min'] != support['max']).any(): raise ValueError('GT shape support disagrees across model comparisons')
    f=f.merge(support[['global_index','pair','min']].rename(columns={'min':'shape_support'}),on=['global_index','pair'],validate='many_to_one')
    color={n:PALETTE.get(n,'#444444') for n in names}
    inventory=[]
    for metric,label in [('joint_tv','Joint occupation TV'),('copula_tv','Rank-copula TV'),('shape_geometry_mean','Joint mask Jaccard distance'),('shape_component_abs_mean','Joint component count discrepancy')]:
        fig,axs=plt.subplots(2,5,figsize=(16,6),sharex=True,sharey=True,constrained_layout=True)
        for ax,pair in zip(axs.flat,definitions['pairs']):
            for n in names:
                q=f[(f.pair==pair)&(f.model==n)]
                if metric.startswith('shape_'): q=q[q.shape_support]
                ax.scatter(q.mean_l2,q[metric],s=9,alpha=.45,color=color[n],label=DISPLAY.get(n,n))
            ax.set_title(pair); ax.grid(alpha=.15)
        fig.supxlabel('Mean relative physical L2 of the two fields');fig.supylabel(label)
        handles,labels=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.04),ncol=len(names),frameon=False)
        stem='cohort_'+metric;save(fig,a.out/stem);inventory.append(dict(stem=stem,caption=f"{label} versus mean paired-field relative L2 for ten pairs and models. " + ("Shape points require GT joint area at least 0.001 of domain at every threshold. " if metric.startswith('shape_') else "All frames are included. ") + "Each point is a paired snapshot; comparisons share frames."))
    # Pairwise descriptors need common support, so retain one point per model comparison.
    for metric,label in [('state','Conditional mean discrepancy'),('scale','Signed complex cross-band discrepancy')]:
        fig,axs=plt.subplots(2,5,figsize=(16,6),sharex=True,sharey=True,constrained_layout=True)
        for ax,pair in zip(axs.flat,definitions['pairs']):
            q=cand[cand.pair==pair]; q=q[q.state_support] if metric=='state' else q[q.scale_bands>0]
            for n in names:
                aa=q[q.model_a==n]; bb=q[q.model_b==n]
                ax.scatter((aa.l2_a_x+aa.l2_a_y)/2,aa[metric+'_a'],s=9,alpha=.35,color=color[n],label=DISPLAY.get(n,n))
                ax.scatter((bb.l2_b_x+bb.l2_b_y)/2,bb[metric+'_b'],s=9,alpha=.35,color=color[n])
            ax.set_title(pair);ax.grid(alpha=.15)
        fig.supxlabel('Mean relative physical L2 of the two fields');fig.supylabel(label+' on comparison-specific common support')
        handles,labels=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.04),ncol=len(names),frameon=False)
        stem='cohort_'+metric;save(fig,a.out/stem);inventory.append(dict(stem=stem,caption=f'{label} versus paired-field relative L2, on support shared by GT and the two compared models. Points may repeat as comparison support changes.'))
    # Explicit model-pair heatmaps with counts; reversals denominator is dominance only.
    fig,axs=plt.subplots(2,2,figsize=(11,9),constrained_layout=True)
    for ax,metric in zip(axs.flat,['state','scale','shape_geometry','shape_topology']):
        mat=np.full((len(names),len(names)),np.nan);den=np.zeros_like(mat,int)
        for i,j in itertools.combinations(range(len(names)),2):
            q=cand[(cand.model_a==names[i])&(cand.model_b==names[j])]
            if metric=='state':q=q[q.state_support]
            if metric=='scale':q=q[q.scale_bands>0]
            if metric.startswith('shape_'):q=q[q.shape_support]
            q=q[q.l2_dominance!=0]; d=q[metric+'_a']-q[metric+'_b']; rev=(d*q.l2_dominance)<0
            den[i,j]=len(q);mat[i,j]=rev.mean() if len(q) else np.nan
            ax.text(j,i,f'{mat[i,j]:.2f}\nn={den[i,j]}' if len(q) else 'n=0',ha='center',va='center',fontsize=7)
        im=ax.imshow(mat,vmin=0,vmax=1,cmap='viridis');ax.set_title(metric);ax.set_xticks(range(len(names)),[DISPLAY.get(n,n) for n in names],rotation=55,ha='right');ax.set_yticks(range(len(names)),[DISPLAY.get(n,n) for n in names])
    fig.colorbar(im,ax=axs,shrink=.7,label='Reversal fraction among L2-dominant comparisons');save(fig,a.out/'inversion_heatmaps');inventory.append(dict(stem='inversion_heatmaps',caption='Fraction of comparisons where the model with worse L2 in both fields has lower descriptor error. Labels show eligible comparison counts across frames and field pairs; they are not independent samples.'))
    fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,metric in zip(axs.flat,['state','scale','shape_geometry','shape_topology']):
        subsets=[];labels=[]
        for gate in ['all','matched_40','matched_20','matched_10']:
            q=cand if gate=='all' else cand[cand[gate]]
            if metric=='state':q=q[q.state_support]
            if metric=='scale':q=q[q.scale_bands>0]
            if metric.startswith('shape_'):q=q[q.shape_support]
            d=np.abs(q[metric+'_a']-q[metric+'_b']).dropna();subsets.append(d);labels.append(f'{gate}\nn={len(d)}')
        ax.boxplot(subsets,showfliers=False);ax.set_xticks(range(1,5),labels);ax.set_title(metric);ax.set_ylabel('Absolute discrepancy contrast');ax.grid(axis='y',alpha=.2)
    save(fig,a.out/'fidelity_strata');inventory.append(dict(stem='fidelity_strata',caption='Absolute model-pair discrepancy differences across nested fieldwise L2 match gates. Boxes summarize all supported comparisons; n counts are descriptive and share frames.'))
    with np.load(a.gt) as z: gt=z['fields'];ids=z['global_indices'];coords=z['coordinates'];scale=z['scale'];offset=z['offset']
    predictions={n:np.load(paths[n])['fields'] for n in names}
    # Favor complete support and stable State ordering; spread model combinations.
    chosen=[]; used_frames=set()
    combinations=list(itertools.combinations(names,2))
    for ma,mb in combinations:
        q=cand[(cand.model_a==ma)&(cand.model_b==mb)&cand.state_support&cand.state_stable].copy()
        if q.empty:continue
        q=q[~q.global_index.isin(used_frames)]
        if q.empty:continue
        # Favor chemically interpretable field pairs, with an explicit fallback.
        focus=q[q.pair.isin(['CH4-CO','CO-T'])]
        if not focus.empty: q=focus
        # Quality first among moderate-effect, >=20% L2-match comparisons, with fallback explicitly recorded.
        match=q[q.matched_20];pool=match if len(match) else q
        pool=pool.copy();pool['effect']=(pool.state_a-pool.state_b).abs();pool=pool[pool.effect>=pool.effect.quantile(.65)]
        pool['quality']=pool[['l2_a_x','l2_a_y','l2_b_x','l2_b_y']].max(axis=1)
        r=pool.sort_values(['quality','effect'],ascending=[True,False]).iloc[0];chosen.append(r);used_frames.add(r.global_index)
        if len(chosen)>=8:break
    payload={};selected=[]
    for z,r in enumerate(chosen):
        k=int(np.where(ids==r.global_index)[0][0]);i,j=[met.FIELDS.index(x) for x in r.pair.split('-')];ma,mb=r.model_a,r.model_b
        arrays=[gt[k],predictions[ma][k],predictions[mb][k]]; labels=['GT',ma,mb]
        fig=plt.figure(figsize=(15,11),constrained_layout=True);gs=fig.add_gridspec(4,3,height_ratios=[1,1,1,1.2]);
        for row,fi in enumerate([i,j]):
            lo=min(v[...,fi].min() for v in arrays);hi=max(v[...,fi].max() for v in arrays)
            for col,(arr,label) in enumerate(zip(arrays,labels)):
                ax=fig.add_subplot(gs[row,col]);im=ax.pcolormesh(coords[:,:,0],coords[:,:,1],arr[...,fi]*scale[fi]+offset[fi],shading='nearest',cmap=CMAPS[fi],vmin=lo*scale[fi]+offset[fi],vmax=hi*scale[fi]+offset[fi],rasterized=True)
                l2=np.linalg.norm((arr[...,fi]-gt[k,...,fi])*scale[fi])/max(np.linalg.norm(gt[k,...,fi]*scale[fi]+offset[fi]),1e-30)
                partner=j if fi==i else i; level=(gt[k,...,partner].min()+.5*np.ptp(gt[k,...,partner])) if partner==0 and j in (1,2) else np.quantile(gt[k,...,partner],.7); values=arr[...,partner]
                if values.min()<level<values.max(): ax.contour(coords[:,:,0],coords[:,:,1],values,levels=[level],colors='white',linewidths=.8)
                ax.set_title(f'{DISPLAY.get(label,label)} | {met.FIELDS[fi]}');ax.text(.98,.04,f'L2 {100*l2:.1f}%',transform=ax.transAxes,ha='right',va='bottom',color='white',fontsize=8,bbox=dict(facecolor='black',alpha=.6,edgecolor='none',pad=2));ax.set_box_aspect(.52);fig.colorbar(im,ax=ax,shrink=.7)
        low_x=i==0 and j in (1,2)
        thresholds=[(gt[k,...,i].min()+.5*np.ptp(gt[k,...,i])) if low_x else np.quantile(gt[k,...,i],.7),np.quantile(gt[k,...,j],.7)]
        gm=((gt[k,...,i]<=thresholds[0]) if low_x else (gt[k,...,i]>=thresholds[0]))&(gt[k,...,j]>=thresholds[1])
        for col,(arr,label) in enumerate(zip(arrays[1:],labels[1:])):
            ax=fig.add_subplot(gs[2,col]); pm=((arr[...,i]<=thresholds[0]) if low_x else (arr[...,i]>=thresholds[0]))&(arr[...,j]>=thresholds[1]); diff=pm.astype(int)-gm.astype(int)
            im=ax.pcolormesh(coords[:,:,0],coords[:,:,1],diff,shading='nearest',cmap='RdBu_r',vmin=-1,vmax=1,rasterized=True);ax.set_box_aspect(.52);ax.set_title(f'{DISPLAY.get(label,label)}: missing / excess joint region');fig.colorbar(im,ax=ax,ticks=[-1,0,1],shrink=.7)
        ax=fig.add_subplot(gs[2,2]);ax.pcolormesh(coords[:,:,0],coords[:,:,1],gm,shading='nearest',cmap='Greys',vmin=0,vmax=1,rasterized=True);ax.set_box_aspect(.52);ax.set_title('GT joint region, fixed GT levels')
        ax=fig.add_subplot(gs[3,0]); details=[]
        for arr,label in zip(arrays[1:],labels[1:]):details.append(met.example_payload(arr,gt[k],i,j))
        mask=(details[0]['gt_counts']>=64)&(details[0]['model_counts']>=64)&(details[1]['model_counts']>=64)
        centers=(details[0]['x_edges'][:-1]+details[0]['x_edges'][1:])/2
        for d,key,label in [(details[0],'gt','GT'),(details[0],'model',ma),(details[1],'model',mb)]:ax.plot(centers[mask],d[key+'_conditional_mean'][mask],marker='o',color=PALETTE.get(label,'#28282b'),label=DISPLAY.get(label,label))
        ax.set_title(f'Own-X conditional mean; GT coverage {r.state_coverage:.2f}');ax.set_xlabel(met.FIELDS[i]+' (normalized)');ax.set_ylabel(met.FIELDS[j]+' (normalized)');ax.legend(fontsize=8)
        for col,(d,label) in enumerate(zip(details,[ma,mb]),start=1):
            ax=fig.add_subplot(gs[3,col]); delta=d['model_joint']-d['gt_joint']; lim=max(np.abs(details[0]['model_joint']-details[0]['gt_joint']).max(),np.abs(details[1]['model_joint']-details[1]['gt_joint']).max(),1e-12)
            im=ax.imshow(delta.T,origin='lower',extent=(d['x_edges'][0],d['x_edges'][-1],d['y_edges'][0],d['y_edges'][-1]),aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim);fig.colorbar(im,ax=ax,label='Model minus GT probability');ax.set_title(f'{DISPLAY.get(label,label)}: joint occupation difference');ax.set_xlabel(met.FIELDS[i]);ax.set_ylabel(met.FIELDS[j])
        fig.suptitle(f'Frame {int(r.global_index)} | {r.pair}',fontsize=12)
        stem=f'example_{z+1:02d}';save(fig,a.out/stem)
        rec={k:(int(v) if k=='global_index' else str(v)) for k,v in {'global_index':r.global_index,'pair':r.pair,'model_a':ma,'model_b':mb}.items()};rec.update(stem=stem,matched_20=bool(r.matched_20),state_coverage=float(r.state_coverage),state_gap=float(r.state_gap));selected.append(rec)
        for field,arr in zip(labels,arrays):payload[f'{stem}_{field}_maps']=arr[...,[i,j]]
        for label,d in zip([ma,mb],details):
            for key,val in d.items():payload[f'{stem}_{label}_{key}']=val
        payload[f'{stem}_common_mask']=mask
        inventory.append(dict(stem=stem,caption=f"Frame {int(r.global_index)}, {r.pair}, {ma} versus {mb}: native-coordinate fields with white partner contours at fixed GT tail levels (CH4 midrange for CH4-CO/CH4-T; otherwise GT 70th percentile), missing/excess joint excursion region, common-support conditional means, and model-minus-GT joint occupation; L2 match within 20%: {bool(r.matched_20)}. Selection is exploratory."))
    # Companion diagnostics use the same selected maps, with unary controls.
    if chosen:
        r=next((r for r in chosen if r.scale_bands>0),chosen[0]); k=int(np.where(ids==r.global_index)[0][0]);i,j=[met.FIELDS.index(x) for x in r.pair.split('-')]; pi=met.PAIRS.index((i,j))
        with np.load(a.metrics/'spectrum.npz') as archive:
            au=archive['auto'];cross=archive['cross']; snames=list(archive['method_names']);bands=archive['band_names']; fractions=au/np.maximum(au.sum(axis=-1,keepdims=True),1e-30)
            fig,axs=plt.subplots(1,4,figsize=(17,4),constrained_layout=True)
            mids=[snames.index(x) for x in ('gt',r.model_a,r.model_b)]
            mask=np.logical_and.reduce([fractions[m,k,i]>=.01 for m in mids]+[fractions[m,k,j]>=.01 for m in mids])
            for m,raw in zip(mids,['gt',r.model_a,r.model_b]):
                label=DISPLAY.get(raw,raw); co=PALETTE.get(raw,'#28282b')
                for ax,fi in zip(axs[:2],[i,j]):ax.plot(range(len(bands)),fractions[m,k,fi],marker='o',color=co,label=label)
                ratio=cross[m,k,pi]/np.maximum(np.sqrt(au[m,k,i]*au[m,k,j]),1e-30)
                axs[2].plot(np.flatnonzero(mask),ratio.real[mask],marker='o',color=co,label=label+' Re');axs[2].plot(np.flatnonzero(mask),ratio.imag[mask],marker='x',color=co,linestyle='--',label=label+' Im')
            # Rectangular-window sensitivity on the exact same selected fields and native modes.
            tr=met.spec._physical_band_transform(coords); rect=met.spec._window_transform(tr,'rectangular')
            ra=[];rc=[]
            for arr in [gt[k],predictions[r.model_a][k],predictions[r.model_b][k]]:
                aa,cc=met.spec._reduce(met.spec._transform_block(arr[None,...],rect),rect);ra.append(aa[0]);rc.append(cc[0])
            ra=np.asarray(ra);rc=np.asarray(rc);rf=ra/np.maximum(ra.sum(axis=-1,keepdims=True),1e-30)
            both=mask & np.logical_and.reduce([rf[m,i]>=.01 for m in range(3)]+[rf[m,j]>=.01 for m in range(3)])
            rr=rc[:,pi]/np.maximum(np.sqrt(ra[:,i]*ra[:,j]),1e-30); hr=cross[mids,k,pi]/np.maximum(np.sqrt(au[mids,k,i]*au[mids,k,j]),1e-30)
            for mi,raw in [(1,r.model_a),(2,r.model_b)]:
                label=DISPLAY.get(raw,raw); co=PALETTE.get(raw,'#28282b')
                axs[3].plot(np.flatnonzero(both),np.abs(hr[mi,both]-hr[0,both]),marker='o',color=co,label=label+' Hann')
                axs[3].plot(np.flatnonzero(both),np.abs(rr[mi,both]-rr[0,both]),marker='x',color=co,linestyle='--',label=label+' rectangular')
            axs[3].set_title('Window sensitivity, shared support');axs[3].set_ylabel('absolute complex difference to GT')
            payload['scale_rect_auto']=ra;payload['scale_rect_cross']=rc;payload['scale_window_common']=both
            for ax,fi in zip(axs[:2],[i,j]):ax.set_title(met.FIELDS[fi]+' unary power fractions');ax.set_ylabel('fraction of supported power')
            axs[2].set_title('Signed complex cross ratio');axs[2].set_ylabel('normalized cross component')
            for ax in axs:ax.set_xticks(range(len(bands)),bands,rotation=40,ha='right');ax.legend(fontsize=7);ax.grid(alpha=.2)
            save(fig,a.out/'example_scale'); inventory.append(dict(stem='example_scale',caption=f"Frame {int(r.global_index)}, {r.pair}, {r.model_a} versus {r.model_b}: unary spectral fractions and signed complex cross ratios over bands with common 1% support. Hann-window native-coordinate quadrature; rectangular-window sensitivity uses bands supported under both windows."))
            payload['scale_auto']=au[mids,k];payload['scale_cross']=cross[mids,k];payload['scale_common']=mask;payload['scale_bands']=bands
        shape_case=next((r for r in chosen if r.shape_support),None)
        if shape_case is None: raise ValueError('no selected case satisfies Shape GT area support')
        r=shape_case; da=np.array(json.loads(r.shape_details_a));db=np.array(json.loads(r.shape_details_b)); x=np.arange(9)
        fig,axs=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
        for arr,raw in [(da,r.model_a),(db,r.model_b)]:
            label=DISPLAY.get(raw,raw); co=PALETTE.get(raw,'#28282b')
            axs[0].plot(x,arr[:,0],marker='o',color=co,label=label);axs[1].plot(x,arr[:,1],marker='o',color=co,label=label+' unary X');axs[1].plot(x,arr[:,2],marker='x',color=co,linestyle='--',label=label+' unary Y');axs[2].plot(x,arr[:,3],marker='o',color=co,label=label)
        for ax,title in zip(axs,['Joint region Jaccard distance','Unary region controls','Joint component count discrepancy']):
            ax.set_title(title);ax.set_xticks(x,[f'{q:.1f}/{fl:.3f}' for q in met.THRESHOLDS for fl in met.FLOORS],rotation=50,ha='right');ax.set_xlabel('GT tail level / native area floor');ax.legend(fontsize=7);ax.grid(alpha=.2)
        save(fig,a.out/'example_shape'); inventory.append(dict(stem='example_shape',caption=f"Frame {int(r.global_index)}, {r.pair}, {r.model_a} versus {r.model_b}: joint-region geometry, unary controls, and joint component discrepancies across GT thresholds and native-area floors; GT joint area is at least 0.001 of domain at every level. Component counts are not persistence."))
        payload['shape_a']=da;payload['shape_b']=db
        payload['shape_global_index']=np.asarray(r.global_index);payload['shape_pair']=np.asarray(r.pair)
        payload['shape_model_a']=np.asarray(r.model_a);payload['shape_model_b']=np.asarray(r.model_b)
    np.savez_compressed(a.metrics/'figure_examples.npz',coordinates=coords,offset=offset,scale=scale,**payload)
    (a.metrics/'selected_examples.json').write_text(json.dumps(selected,indent=2)+'\n')
    (a.out/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(f'wrote {len(inventory)} figures and {len(selected)} selected examples')
if __name__=='__main__':run()
