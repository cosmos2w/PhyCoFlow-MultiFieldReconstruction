#!/usr/bin/env python3
"""GT-defined coupled-region diagnostics. No training, checkpoints or GPU.

Default requires frozen endpoint archives for every model in scripts/registry.json.
After inference extends that registry and writes its endpoint, rerun this script;
no event-code edit is needed. --allow-incomplete enables explicit subset development.
--redraw renders selected cases entirely from the retained event case archive.
"""
from pathlib import Path
import argparse,csv,json,itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

V4=Path(__file__).resolve().parents[1]; STUDY=V4.parent
PREFERRED_MODELS=['gl_rbf_cq','senseiver','mimonet','geofno','deeponet','diffusion_pde']
REGISTRY=V4/'scripts/registry.json'

def registry_models():
    registry=json.loads(REGISTRY.read_text())
    return [m for m in PREFERRED_MODELS if m in registry]+sorted(m for m in registry if m not in PREFERRED_MODELS)

def model_label(name):
    return LABELS.get(name,name.replace('_',' ').title())
LABELS={'gl_rbf_cq':'FFM','senseiver':'Senseiver','mimonet':'MIMONet','geofno':'GeoFNO','deeponet':'DeepONet','diffusion_pde':'DiffusionPDE'}
EVENTS=['lowCH4_highCO','transitionCH4_highCO','highCO_hotT','lowCH4_hotT']
EVENT_LABELS=['Low CH₄ / high CO','CH₄ transition / high CO','High CO / hot T','Low CH₄ / hot T']
FIELDS=['CH4','CO','T','U1','p']; PAIRS=[(0,1),(0,1),(1,2),(0,2)]
CMAPS=['cividis','magma','inferno','RdBu_r','RdBu_r']
DATA=V4/'data/events_cases.npz'; META=V4/'metadata/events_summary.json'; FIG=V4/'figures'

def weights(axis):
    a=np.asarray(axis,dtype=float); d=np.diff(a)
    if not np.all(d>0): raise ValueError('Expected ascending native coordinate axes')
    w=np.empty_like(a);w[0]=d[0]/2;w[-1]=d[-1]/2;w[1:-1]=(d[:-1]+d[1:])/2
    return w

def wquantile(a,w,q):
    ix=np.argsort(a.ravel()); vals=a.ravel()[ix]; ws=w.ravel()[ix]
    return float(vals[np.searchsorted(np.cumsum(ws),q*ws.sum(),side='left')])

def specs(gt,w):
    low=float(gt[:,:,0].min()+.25*np.ptp(gt[:,:,0])); hi=float(gt[:,:,0].min()+.75*np.ptp(gt[:,:,0]))
    co=wquantile(gt[:,:,1],w,.90); hot=wquantile(gt[:,:,2],w,.75)
    return np.array([[low,co],[low,hi],[co,hot],[low,hot]],float),co

def masks(z,event,thresholds,co):
    low=thresholds[0,0];hi=thresholds[1,1];hot=thresholds[2,1]
    if event==0:return z[:,:,0] < low,z[:,:,1] > co
    if event==1:return (z[:,:,0]>=low)&(z[:,:,0]<=hi),z[:,:,1]>co
    if event==2:return z[:,:,1]>co,z[:,:,2]>hot
    return z[:,:,0]<low,z[:,:,2]>hot

def area(mask,w):return float(np.sum(w*mask))

def region(ref,pred,w):
    tp=area(ref&pred,w); fp=area(~ref&pred,w); fn=area(ref&~pred,w)
    return {'gt_area':area(ref,w),'pred_area':area(pred,w),'missing':fn,'excess':fp,
            'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
            'iou':tp/(tp+fp+fn) if tp+fp+fn else None}

def metrics(gt,z,event,thresholds,co,w):
    r1,r2=masks(gt,event,thresholds,co); p1,p2=masks(z,event,thresholds,co)
    a,b,j=region(r1,p1,w),region(r2,p2,w),region(r1&r2,p1&p2,w)
    pair=PAIRS[event]; l2=[]
    for f in pair:
        den=float(np.sum(w*gt[:,:,f]**2)); l2.append(float(np.sqrt(np.sum(w*(z[:,:,f]-gt[:,:,f])**2)/den)) if den>0 else float('inf'))
    return {'l2_a':l2[0],'l2_b':l2[1],
       **{f'unary_a_{k}':v for k,v in a.items()}, **{f'unary_b_{k}':v for k,v in b.items()},
       **{f'joint_{k}':v for k,v in j.items()},
       'gt_partner_given_a':area(r1&r2,w)/area(r1,w) if area(r1,w) else None,
       'pred_partner_given_a':area(p1&p2,w)/area(p1,w) if area(p1,w) else None,
       'partner_inside_gt_a':area(r1&p2,w)/area(r1,w) if area(r1,w) else None}

def dumpcsv(path,rows):
    with path.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

def prepare(allow_incomplete):
    models=registry_models()
    files={m:V4/'data'/f'{m}_test100.npz' for m in models}
    missing=[m for m,p in files.items() if not p.exists()]
    if missing and not allow_incomplete:raise RuntimeError('Missing endpoint archives: '+', '.join(missing)+'; rerun after all registered endpoints are available')
    available=[m for m in models if m not in missing]
    if len(available)<2:raise RuntimeError('At least two models are needed')
    with np.load(STUDY/'data/gt_test100.npz') as f:
        gt=f['fields'].astype(float); ids=f['global_indices'].copy(); coord=f['coordinates'].copy(); scale=f['scale'].copy();offset=f['offset'].copy()
    x=coord[0,:,0].astype(float);y=coord[:,0,1].astype(float)
    w=weights(y)[:,None]*weights(x)[None,:];w/=w.sum()
    gt=gt*scale+offset
    predictions={}
    for model in available:
        with np.load(files[model]) as f:
            z=f['fields']
            if z.shape!=gt.shape:raise ValueError(f'Unexpected shape for {model}: {z.shape}')
            predictions[model]=z
    records=[]; thresholds_by_frame=[]; lookup={}
    for fi,g in enumerate(gt):
        t,co=specs(g,w); thresholds_by_frame.append((t,co))
        for ei,event in enumerate(EVENTS):
            for model in available:
                pred=predictions[model][fi].astype(float)*scale+offset
                rec={'frame':int(ids[fi]),'event':event,'model':model,**metrics(g,pred,ei,t,co,w)}
                records.append(rec);lookup[(fi,ei,model)]=rec
    rows=[]
    for fi in range(len(ids)):
        for ei,event in enumerate(EVENTS):
            for a,b in itertools.combinations(available,2):
                ra,rb=lookup[(fi,ei,a)],lookup[(fi,ei,b)]
                def ratio(k):
                    x,y=ra[k],rb[k];return abs(x-y)/max(x,y,1e-15)
                d1,d2=ratio('l2_a'),ratio('l2_b');gt_area=ra['joint_gt_area']
                gap=abs(ra['joint_iou']-rb['joint_iou']) if ra['joint_iou'] is not None and rb['joint_iou'] is not None else 0
                quality=max(ra['l2_a'],ra['l2_b'],rb['l2_a'],rb['l2_b'])
                rows.append({'frame':int(ids[fi]),'event':event,'model_a':a,'model_b':b,'match_a':d1,'match_b':d2,
                    'max_field_l2':quality,'quality_stratum':'<0.1' if quality<.1 else '<0.25' if quality<.25 else '<0.5' if quality<.5 else '>=0.5',
                    'joint_gt_area':gt_area,'joint_iou_gap':gap,'valid_support':gt_area>=.005,
                    'match_20':d1<=.2 and d2<=.2,'match_40':d1<=.4 and d2<=.4,
                    'joint_iou_a':ra['joint_iou'],'joint_iou_b':rb['joint_iou']})
    # Prioritize distinct event families and representation while ranking by IoU contrast.
    eligible=[r for r in rows if r['valid_support'] and r['match_40']]
    selected=[]; usage={m:0 for m in available}
    for event in EVENTS:
        candidates=[r for r in eligible if r['event']==event]
        if not candidates:continue
        # Hard preference 20%-matched; otherwise disclose fallback 40%.
        tight=[r for r in candidates if r['match_20']];pool=tight or candidates
        pool.sort(key=lambda r:(-(r['joint_iou_gap']-.015*(usage[r['model_a']]+usage[r['model_b']])),r['frame'],r['model_a'],r['model_b']))
        chosen=pool[0];selected.append(chosen)
        for model in (chosen['model_a'],chosen['model_b']):usage[model]+=1
    # A fifth case can expose a model omitted by the four event-first choices.
    uncovered={m for m in available if not usage[m]}
    if uncovered and len(selected)<5:
        pool=[r for r in eligible if (r['model_a'] in uncovered or r['model_b'] in uncovered) and r not in selected]
        tight=[r for r in pool if r['match_20']]
        pool=tight or pool
        if pool:
            pool.sort(key=lambda r:(-r['joint_iou_gap'],r['frame'],r['model_a'],r['model_b']))
            selected.append(pool[0])
    # Search matched-error inversions: worse fieldwise L2 in both fields but
    # higher joint IoU. Prioritize close unary IoUs and distinct event families.
    inversions=[]
    for r in eligible:
        if r['joint_iou_gap'] < .1: continue
        fi=int(np.flatnonzero(ids==r['frame']).item());ei=EVENTS.index(r['event'])
        xr=lookup[(fi,ei,r['model_a'])];yr=lookup[(fi,ei,r['model_b'])]
        da=xr['l2_a']-yr['l2_a'];db=xr['l2_b']-yr['l2_b'];di=xr['joint_iou']-yr['joint_iou']
        if da*db<=0 or da*di<=0: continue
        r=dict(r)
        r['selection_type']='field_error_joint_iou_inversion'
        r['worse_fields_better_joint']=r['model_a'] if da>0 else r['model_b']
        r['max_unary_iou_gap']=max(abs(xr['unary_a_iou']-yr['unary_a_iou']),abs(xr['unary_b_iou']-yr['unary_b_iou']))
        inversions.append(r)
    inversions.sort(key=lambda r:(not r['match_20'],r['max_unary_iou_gap'],r['max_field_l2'],r['frame']))
    inversion_selected=[]
    for r in inversions:
        if not inversion_selected or r['event']!=inversion_selected[0]['event']:
            inversion_selected.append(r)
        if len(inversion_selected)==2:break
    selected.extend(inversion_selected)
    if not selected: raise RuntimeError('No eligible comparisons: check support and matching counts')
    cases=[]; meta_cases=[]
    for row in selected:
        fi=int(np.flatnonzero(ids==row['frame']).item());ei=EVENTS.index(row['event']);t,co=thresholds_by_frame[fi]
        cases.append(np.stack([gt[fi],predictions[row['model_a']][fi].astype(float)*scale+offset,predictions[row['model_b']][fi].astype(float)*scale+offset]))
        meta_cases.append({**row,'thresholds':t.tolist(),'co_q90':co,'records':[lookup[(fi,ei,m)] for m in (row['model_a'],row['model_b'])]})
    DATA.parent.mkdir(parents=True,exist_ok=True); META.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(DATA,cases=np.stack(cases),x=x,y=y,weights=w)
    dumpcsv(V4/'data/events_metrics.csv',records);dumpcsv(V4/'data/events_pairs.csv',rows)
    count={}
    for event in EVENTS:
        rr=[r for r in rows if r['event']==event];count[event]={'pairs_all':len(rr),'support_ge_0.005':sum(r['valid_support'] for r in rr),
          'support_and_match20':sum(r['valid_support'] and r['match_20'] for r in rr), 'support_and_match40':sum(r['valid_support'] and r['match_40'] for r in rr),
          'by_quality_and_match40':{q:sum(r['valid_support'] and r['match_40'] and r['quality_stratum']==q for r in rr) for q in ['<0.1','<0.25','<0.5','>=0.5']}}
    doc={'models_requested':models,'models_available':available,'missing':missing,'complete_six':all(m in available for m in PREFERRED_MODELS),'complete_registry':not missing,'frame_count':len(ids),'selection':meta_cases,'counts':count,'inversion_candidates_match40':len(inversions),'inversion_candidates_match20':sum(r['match_20'] for r in inversions),
       'thresholds':'Per-frame native-area weighted CO q90 and T q75; CH4 below 25% of GT per-frame range, or transition 25–75% inclusive. Thresholds are frozen from GT and applied to all predictions. Strict > for high thresholds; actual event areas are reported.',
       'area':'Tensor-product trapezoidal native-coordinate weights normalized to domain area. Unary and joint missing/excess, precision/recall/IoU use these weights. partner_inside_gt_a evaluates predicted second event inside the GT first region.',
       'matching':'Each relevant field relative L2 uses native-area weighting and decoded values; absolute difference divided by larger model error <=20% or <=40% for each field. Joint GT support >=0.5% of domain. Greedy maximum IoU gap after preferring 20% with light model-coverage penalty; one case per event plus at most one case for uncovered model coverage. Selection is targeted, not a population estimate.',
       'limitations':'Events are empirical thresholded co-occurrences, not chemical laws or causal mechanisms. Raster topology and small regions depend on grid and threshold. Screened on the same 100 frames shown; no inferential generalization. Model T is evaluated as output and is not assumed observation-consistent.'}
    META.write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n');return doc

def render(doc):
    with np.load(DATA) as f:
        cases=f['cases'];x=f['x'];y=f['y']
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.titlesize':13,'figure.titlesize':15,'pdf.fonttype':42})
    for case_index,(z,info) in enumerate(zip(cases,doc['selection'])):
        ei=EVENTS.index(info['event']); pair=PAIRS[ei]; t=np.asarray(info['thresholds']);co=info['co_q90']
        ref=masks(z[0],ei,t,co); ref_joint=ref[0]&ref[1]
        fig,ax=plt.subplots(3,3,figsize=(13.5,9),constrained_layout=True)
        for row,f in enumerate(pair):
            lo=float(z[:,:,:,f].min());hi=float(z[:,:,:,f].max())
            for col in range(3):
                im=ax[row,col].pcolormesh(x,y,z[col,:,:,f],cmap=CMAPS[f],vmin=lo,vmax=hi,shading='nearest',rasterized=True)
                # Actual partner region and fixed GT reference use the same
                # GT-defined thresholds; transition masks show both edges.
                fld=z[col,:,:,f]; levels=np.linspace(lo,hi,14)[1:-1]
                levels=levels[(levels>fld.min()) & (levels<fld.max())]
                if len(levels):
                    ax[row,col].contour(x,y,fld,levels=levels,colors='#c8c8c8',linewidths=.24,alpha=.65)
                partner=masks(z[col],ei,t,co)[1-row].astype(float)
                fixed=ref[1-row].astype(float)
                if col and fixed.min()<.5<fixed.max():
                    ax[row,col].contour(x,y,fixed,levels=[.5],colors='#bdbdbd',linewidths=.8,linestyles='dashed')
                if partner.min()<.5<partner.max():
                    ax[row,col].contour(x,y,partner,levels=[.5],colors='white',linewidths=1.0)
                if col:
                    error=info['records'][col-1][['l2_a','l2_b'][row]]
                    ax[row,col].text(.985,.04,f'L₂ {100*error:.1f}%',transform=ax[row,col].transAxes,
                        ha='right',va='bottom',fontsize=10,color='white',
                        bbox={'facecolor':'#111111','edgecolor':'none','alpha':.75,'pad':1.5})
                if row==0:ax[row,col].set_title(['Ground truth',model_label(info['model_a']),model_label(info['model_b'])][col])
                ax[row,col].set_box_aspect(.54);ax[row,col].set_xticks([]);ax[row,col].set_yticks([])
                if col==0:ax[row,col].set_ylabel(FIELDS[f],fontsize=12)
            fig.colorbar(im,ax=list(ax[row,:]),shrink=.78,label='Decoded '+FIELDS[f],pad=.012)
        cm=ListedColormap(['#f1f1f1','#4b9e83','#df7439','#6d59a0'])
        for col in range(3):
            m=masks(z[col],ei,t,co);j=m[0]&m[1]
            # 0 neither, 1 true joint, 2 missing, 3 excess
            category=np.where(ref_joint&j,1,np.where(ref_joint&~j,2,np.where(~ref_joint&j,3,0)))
            ax[2,col].pcolormesh(x,y,category,cmap=cm,vmin=-.5,vmax=3.5,shading='nearest',rasterized=True)
            ax[2,col].set_box_aspect(.54);ax[2,col].set_xticks([]);ax[2,col].set_yticks([])
            if col==0:ax[2,col].set_ylabel('Joint region')
        fig.legend(handles=[Patch(color='#4b9e83',label='GT joint retained'),Patch(color='#df7439',label='GT joint missing'),Patch(color='#6d59a0',label='Predicted joint excess'),Line2D([0],[0],color='#666666',lw=1.5,label='Solid white: own partner'),Line2D([0],[0],color='gray',lw=1,ls='--',label='Dashed: GT partner')],
                   loc='lower center',bbox_to_anchor=(.5,-.015),ncol=5,frameon=False,fontsize=10)
        fig.suptitle(EVENT_LABELS[ei]+f' · frame {info["frame"]}',fontsize=16)
        name='events_'+str(case_index+1).zfill(2)+'_'+info['event']
        fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--allow-incomplete',action='store_true');p.add_argument('--redraw',action='store_true');args=p.parse_args()
    doc=json.loads(META.read_text()) if args.redraw else prepare(args.allow_incomplete)
    render(doc)
    print(json.dumps({'complete_six':doc['complete_six'],'complete_registry':doc.get('complete_registry',doc['complete_six']),'available':doc['models_available'],'counts':doc['counts'],'selected':[{k:v for k,v in r.items() if k not in ('records','thresholds')} for r in doc['selection']]},indent=2))
if __name__=='__main__':main()
