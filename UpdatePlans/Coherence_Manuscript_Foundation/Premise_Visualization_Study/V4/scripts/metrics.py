#!/usr/bin/env python3
"""Cohort descriptors for fixed paired TEST100 endpoint arrays; no model inference."""
from __future__ import annotations
import argparse, csv, hashlib, itertools, json, sys
from pathlib import Path
import numpy as np
from scipy import ndimage, stats

BASE = Path(__file__).resolve().parents[2]
import physical_spectrum as spec

FIELDS = ('CH4', 'CO', 'T', 'U_1', 'p')
PAIRS = tuple(itertools.combinations(range(5), 2))
BINS = (8, 10, 12, 16, 20)
THRESHOLDS = (.5, .7, .9)
FLOORS = (.003, .006, .01)

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        w.writeheader(); w.writerows(rows)

def state_stats(z, g, i, j, bins, zr=None, gr=None):
    """Equal-node occupation; own-X membership, GT range, tails clamped."""
    x, y = z[..., i].ravel(), z[..., j].ravel()
    gx, gy = g[..., i].ravel(), g[..., j].ravel()
    ex = np.linspace(gx.min(), gx.max(), bins+1); ey = np.linspace(gy.min(), gy.max(), bins+1)
    def ids(v, edges): return np.searchsorted(edges[1:-1], v, side='right')
    ix, iy, igx, igy = ids(x, ex), ids(y, ey), ids(gx, ex), ids(gy, ey)
    cnt = np.bincount(ix, minlength=bins); gcnt = np.bincount(igx, minlength=bins)
    means = np.bincount(ix, weights=y, minlength=bins) / np.maximum(cnt, 1)
    gmeans = np.bincount(igx, weights=gy, minlength=bins) / np.maximum(gcnt, 1)
    h = np.bincount(ix*bins+iy, minlength=bins*bins).reshape(bins,bins) / x.size
    gh = np.bincount(igx*bins+igy, minlength=bins*bins).reshape(bins,bins) / x.size
    # Rank-based copula occupies a common uniform rank grid, removing marginal transforms.
    def cop(a,b):
        ra = np.minimum((a*bins).astype(int), bins-1)
        rb = np.minimum((b*bins).astype(int), bins-1)
        return np.bincount(ra*bins+rb,minlength=bins*bins).reshape(bins,bins)/len(a)
    if zr is None: zr = [stats.rankdata(v, method='average')/(len(v)+1) for v in (x,y)]
    if gr is None: gr = [stats.rankdata(v, method='average')/(len(v)+1) for v in (gx,gy)]
    return cnt, gcnt, means, gmeans, .5*np.abs(h-gh).sum(), .5*np.abs(h.sum(1)-gh.sum(1)).sum(), .5*np.abs(h.sum(0)-gh.sum(0)).sum(), .5*np.abs(cop(*zr)-cop(*gr)).sum()

def example_payload(z, g, i, j, bins=12):
    """Plot-ready conditional and joint occupation arrays for one endpoint and GT.

    z and g are [y,x,5] normalized fields; plot maps using original arrays and
    native coordinates in the GT archive. Joint histograms include clamped tails.
    """
    gx, gy = g[..., i].ravel(), g[..., j].ravel()
    ex = np.linspace(gx.min(), gx.max(), bins+1)
    ey = np.linspace(gy.min(), gy.max(), bins+1)
    output = {'x_edges': ex, 'y_edges': ey}
    for label, a in [('gt', g), ('model', z)]:
        x, y = a[..., i].ravel(), a[..., j].ravel()
        ix = np.searchsorted(ex[1:-1], x, side='right')
        iy = np.searchsorted(ey[1:-1], y, side='right')
        count = np.bincount(ix, minlength=bins)
        output[label+'_counts'] = count
        output[label+'_conditional_mean'] = np.bincount(ix, weights=y, minlength=bins)/np.maximum(count,1)
        output[label+'_joint'] = np.bincount(ix*bins+iy, minlength=bins*bins).reshape(bins,bins)/len(x)
    return output

def conditional(a, b):
    cnt, gcnt, means, gmeans = a[:4]
    mask = (gcnt >= 64) & (cnt >= 64) & (b[0] >= 64)
    coverage = gcnt[mask].sum() / gcnt.sum()
    if not mask.any(): return np.nan, np.nan, coverage, 0
    return (np.average(np.abs(means[mask]-gmeans[mask]), weights=gcnt[mask]),
            np.average(np.abs(b[2][mask]-gmeans[mask]), weights=gcnt[mask]), coverage, int(mask.sum()))

def area_weights(coords):
    def trap(x):
        d = np.diff(x)
        if np.any(d <= 0): raise ValueError('coordinates must be strictly increasing')
        return np.r_[d[0]/2, (d[:-1]+d[1:])/2, d[-1]/2]
    x, y = coords[0,:,0], coords[:,0,1]
    if not (np.allclose(coords[:,:,0], x[None,:]) and np.allclose(coords[:,:,1], y[:,None])): raise ValueError('not rectilinear')
    a = np.outer(trap(y), trap(x)); return a/a.sum()

def jaccard(a,b,w):
    union = np.sum(w*(a|b))
    return 0. if union == 0 else float(np.sum(w*(a^b))/union)

def components(mask,w,floor):
    labels, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(2,1))
    sizes = np.bincount(labels.ravel(), weights=w.ravel(), minlength=n+1)[1:]
    return int(np.count_nonzero(sizes >= floor))

def shape_stats(z,g,i,j,w):
    # Each threshold uses GT field quantiles, same physical level for all models.
    out=[]
    for q in THRESHOLDS:
        low_x = i == 0 and j in (1, 2)
        ti=(g[...,i].min() + {0.5:.75,0.7:.5,0.9:.25}[q]*np.ptp(g[...,i])) if low_x else np.quantile(g[...,i],q)
        tj=np.quantile(g[...,j],q)
        za,ga=(z[...,i]<=ti,g[...,i]<=ti) if low_x else (z[...,i]>=ti,g[...,i]>=ti)
        zb,gb=z[...,j]>=tj,g[...,j]>=tj
        zm,gm=za&zb,ga&gb
        for floor in FLOORS:
            gz,pz=components(gm,w,floor),components(zm,w,floor)
            out.append((jaccard(zm,gm,w), jaccard(za,ga,w),jaccard(zb,gb,w),abs(pz-gz),gz,pz,float(np.sum(w*gm)),float(np.sum(w*zm)),float(ti),float(tj)))
    return np.asarray(out)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gt',type=Path,default=BASE/'data/gt_test100.npz')
    p.add_argument('--model',action='append',required=True,help='NAME=NPZ path, fields [frame,y,x,5]')
    p.add_argument('--out',type=Path,default=BASE/'V4/data/metrics')
    p.add_argument('--chunk',type=int,default=4)
    a=p.parse_args(); paths=dict(s.split('=',1) for s in a.model)
    if len(paths)!=len(a.model) or 'gt' in paths or len(paths)<2: p.error('supply at least two distinct model names, excluding gt')
    paths={k:Path(v) for k,v in paths.items()}; a.out.mkdir(parents=True,exist_ok=True)
    with np.load(a.gt,allow_pickle=False) as q:
        gt=q['fields']; coords=q['coordinates']; scale=q['scale']; offset=q['offset']; ids=q['global_indices']
    pred={}
    for name,path in paths.items():
        with np.load(path,allow_pickle=False) as q:
            pred[name]=q['fields']
            if 'global_indices' in q and not np.array_equal(q['global_indices'],ids): raise ValueError(f'{name} frame identity mismatch')
        if pred[name].shape!=gt.shape: raise ValueError(f'{name} shape mismatch')
        if not np.isfinite(pred[name]).all(): raise ValueError(f'{name} contains nonfinite values')
    if gt.shape[0]!=len(ids) or gt.shape[-1]!=5 or not np.isfinite(gt).all(): raise ValueError('invalid GT')
    names=list(pred); n=len(ids); w=area_weights(coords)
    # l2 convention matches the previous frozen study: unweighted Euclidean physical values.
    l2=np.empty((len(names),n,5)); mse=np.empty_like(l2)
    for m,name in enumerate(names):
        for k in range(n):
            err=(pred[name][k].astype(float)-gt[k])*scale
            l2[m,k]=np.linalg.norm(err,axis=(0,1))/np.maximum(np.linalg.norm(gt[k]*scale+offset,axis=(0,1)),1e-30)
            mse[m,k]=np.mean((pred[name][k]-gt[k])**2,axis=(0,1))
    # Reuse V2 spectral summaries only for byte-identical archives and identical GT.
    cache=BASE/'data/v2/spectrum_octave_payloads.npz'; reuse={}
    if cache.exists() and digest(a.gt)==digest(BASE/'data/gt_test100.npz'):
        with np.load(cache) as c:
            for name,path in paths.items():
                old=BASE/f'data/{name}_test100.npz'
                if old.exists() and digest(path)==digest(old) and name in c['method_names']:
                    idx=list(c['method_names']).index(name); reuse[name]=(c['per_frame_auto_power_by_band'][idx],c['per_frame_cross_spectrum_by_band'][idx])
            reuse['gt']=(c['per_frame_auto_power_by_band'][0],c['per_frame_cross_spectrum_by_band'][0])
    tr=spec._physical_band_transform(coords)
    auto=[]; cross=[]
    for name,path in [('gt',a.gt),*paths.items()]:
        if name in reuse: au,cr=reuse[name]
        else: au,cr,_=spec._measure(path,tr,a.chunk)
        auto.append(au); cross.append(cr)
    auto=np.asarray(auto); cross=np.asarray(cross); frac=spec._fractions(auto)
    np.savez_compressed(a.out/'spectrum.npz',method_names=np.array(['gt',*names]),global_indices=ids,auto=auto,cross=cross,band_names=np.array(spec.BAND_NAMES),kappa_edges=spec.KAPPA_EDGES)
    frame=[]; candidates=[]
    for k,gid in enumerate(ids):
        g=gt[k]; zs=[pred[name][k] for name in names]
        ranks=[[stats.rankdata(z[...,f].ravel(),method='average')/(g.shape[0]*g.shape[1]+1) for f in range(5)] for z in [g,*zs]]
        for pi,(i,j) in enumerate(PAIRS):
            pair=f'{FIELDS[i]}-{FIELDS[j]}'
            ss=[[state_stats(z,g,i,j,b,(ranks[m+1][i],ranks[m+1][j]),(ranks[0][i],ranks[0][j])) for b in BINS] for m,z in enumerate(zs)]
            sh=[shape_stats(z,g,i,j,w) for z in zs]
            for m,name in enumerate(names):
                for bi,b in enumerate(BINS):
                    s=ss[m][bi]
                    frame.append(dict(global_index=int(gid),pair=pair,model=name,bins=b,l2_x=l2[m,k,i],l2_y=l2[m,k,j],mse_x=mse[m,k,i],mse_y=mse[m,k,j],joint_tv=s[4],marginal_x_tv=s[5],marginal_y_tv=s[6],copula_tv=s[7],shape_geometry_mean=sh[m][:,0].mean(),shape_unary_x=sh[m][:,1].mean(),shape_unary_y=sh[m][:,2].mean(),shape_component_abs_mean=sh[m][:,3].mean()))
            for ma,mb in itertools.combinations(range(len(names)),2):
                gap=np.abs(l2[ma,k,[i,j]]-l2[mb,k,[i,j]])/np.maximum(np.maximum(l2[ma,k,[i,j]],l2[mb,k,[i,j]]),1e-30)
                state=[conditional(ss[ma][bi],ss[mb][bi]) for bi in range(len(BINS))]
                primary=state[2]; support=primary[2]>=.8
                common=(frac[[0,ma+1,mb+1],k,i,:]>=.01).all(axis=0)&(frac[[0,ma+1,mb+1],k,j,:]>=.01).all(axis=0)
                rat=cross[:,k,pi,:]/np.maximum(np.sqrt(auto[:,k,i,:]*auto[:,k,j,:]),1e-30)
                errs=[float(np.sqrt(np.mean(np.abs(rat[m+1,common]-rat[0,common])**2)/2)) if common.any() else np.nan for m in (ma,mb)]
                topology=sh[ma][:,3]-sh[mb][:,3]; geometry=sh[ma][:,0]-sh[mb][:,0]
                candidates.append(dict(global_index=int(gid),pair=pair,model_a=names[ma],model_b=names[mb],l2_a_x=l2[ma,k,i],l2_a_y=l2[ma,k,j],l2_b_x=l2[mb,k,i],l2_b_y=l2[mb,k,j],l2_gap_x=gap[0],l2_gap_y=gap[1],matched_10=bool(max(gap)<=.1),matched_20=bool(max(gap)<=.2),matched_40=bool(max(gap)<=.4),l2_dominance=int(np.sign(l2[ma,k,i]-l2[mb,k,i])) if (l2[ma,k,i]-l2[mb,k,i])*(l2[ma,k,j]-l2[mb,k,j])>0 else 0,state_a=primary[0],state_b=primary[1],state_gap=primary[0]-primary[1],state_gap_gt_y_sd=(primary[0]-primary[1])/max(float(np.std(g[...,j])),1e-30),state_coverage=primary[2],state_support=bool(support),state_stable=bool(all(s[2]>=.8 for s in state) and (all(s[0]<s[1] for s in state) or all(s[0]>s[1] for s in state))),joint_tv_a=ss[ma][2][4],joint_tv_b=ss[mb][2][4],copula_tv_a=ss[ma][2][7],copula_tv_b=ss[mb][2][7],scale_a=errs[0],scale_b=errs[1],scale_gap=errs[0]-errs[1],scale_bands=int(common.sum()),shape_gt_area_min=float(sh[ma][:,6].min()),shape_support=bool(np.all(sh[ma][:,6]>=.001)),shape_geometry_a=sh[ma][:,0].mean(),shape_geometry_b=sh[mb][:,0].mean(),shape_geometry_stable=bool(np.all(geometry>0) or np.all(geometry<0)),shape_topology_a=sh[ma][:,3].mean(),shape_topology_b=sh[mb][:,3].mean(),shape_topology_stable=bool(np.all(topology>0) or np.all(topology<0)),shape_details_a=json.dumps(sh[ma].tolist(),separators=(',',':')),shape_details_b=json.dumps(sh[mb].tolist(),separators=(',',':')),state_bin_details=json.dumps(state,separators=(',',':'))))
        if (k+1)%10==0: print(f'frames {k+1}/{n}',flush=True)
    expected_frame=n*len(PAIRS)*len(names)*len(BINS)
    expected_candidates=n*len(PAIRS)*len(names)*(len(names)-1)//2
    if len(frame)!=expected_frame or len(candidates)!=expected_candidates: raise RuntimeError('incomplete cohort catalog')
    write_csv(a.out/'frame_metrics.csv',frame); write_csv(a.out/'candidates.csv',candidates)
    cohort=[]
    for pair in [f'{FIELDS[i]}-{FIELDS[j]}' for i,j in PAIRS]:
      for metric in ('state','scale','shape_geometry','shape_topology'):
       for gate in ('all','matched_10','matched_20','matched_40'):
        rows=[r for r in candidates if r['pair']==pair and (gate=='all' or r[gate]) and (r['state_support'] if metric=='state' else r['scale_bands']>0 if metric=='scale' else r['shape_support'])]
        diff=np.array([r[metric+'_a']-r[metric+'_b'] for r in rows]); dom=np.array([r['l2_dominance'] for r in rows]); valid=np.isfinite(diff)
        d=diff[valid]; dd=dom[valid]; decisive=dd!=0; rev=decisive & (np.sign(d)*dd<0)
        cohort.append(dict(pair=pair,metric=metric,stratum=gate,n=len(d),median_absolute_gap=float(np.median(np.abs(d))) if len(d) else np.nan,dominant_l2_n=int(decisive.sum()),reversals=int(rev.sum()),reversal_fraction=float(rev.sum()/decisive.sum()) if decisive.sum() else np.nan))
    write_csv(a.out/'cohort.csv',cohort)
    # Compact navigational index; the complete, unfiltered catalog remains CSV.
    index = {}
    for metric in ('state','scale','shape_geometry','shape_topology'):
        index[metric] = {}
        for gate in ('all','matched_10','matched_20','matched_40','inversion'):
            rows = [r for r in candidates if (gate in ('all','inversion') or r[gate])
                    and (r['state_support'] if metric=='state' else r['scale_bands']>0 if metric=='scale' else r['shape_support'])]
            if gate == 'inversion':
                rows = [r for r in rows if r['l2_dominance'] and (r[metric+'_a']-r[metric+'_b'])*r['l2_dominance'] < 0]
            rows.sort(key=lambda r: abs(r[metric+'_a']-r[metric+'_b']), reverse=True)
            index[metric][gate] = [dict(global_index=r['global_index'],pair=r['pair'],model_a=r['model_a'],model_b=r['model_b'],effect=float(r[metric+'_a']-r[metric+'_b'])) for r in rows[:20]]
    (a.out/'candidate_index.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
    definitions={'inputs':{k:{'path':str(v),'sha256':digest(v)} for k,v in [('gt',a.gt),*paths.items()]},'shape':list(gt.shape),'models':names,'pairs':[f'{FIELDS[i]}-{FIELDS[j]}' for i,j in PAIRS],'state':'equal-node empirical occupation. own-X GT-range bins; GT occupancy weighted absolute conditional mean error on common >=64-node bins. GT coverage >=0.8. TV of 2D histogram, separate 1D marginals, rank copula histogram, 12 bins primary; 8,10,12,16,20 stability. Values outside GT range enter end bins.','scale':'V2 native-coordinate quadrature NUDFT; physical Hann taper, signed complex cross ratios, kappa edges 0,4,8,16,32,64,infinity; common pair band >=1% each field GT and both models. RMS complex error / sqrt(2), no Parseval claim. Auto and cross saved for unary attribution.','spectrum_reused':list(reuse),'shape_definition':'GT fieldwise quantiles .5,.7,.9 used as fixed upper-tail thresholds. For CH4-CO and CH4-T only, CH4 uses lower tails at GT minimum plus .75,.5,.25 of GT range, paired with upper q=.5,.7,.9 of the second field; this avoids tied CH4 floor quantiles; all other pairs use upper tails in both fields. Joint intersection geometry uses area-weighted Jaccard distance; unary Jaccard separately. Topology uses absolute count discrepancy of 4-connected joint excursion components, each with native area >= .003,.006,.01 of full domain; descriptor is not persistent homology. Nine threshold/floor values, including native GT and model joint area fractions and the actual normalized threshold values, saved in candidates. shape_support requires GT joint area >=0.001 at every threshold; stable means strict same order across all nine.','l2':'Euclidean relative L2 in physical units; matching gaps relative to larger L2, each field <=10%,20%,40%. Dominance requires one model strictly worse in both fields. Reversal requires dominant model better on descriptor. No causal or universal validity assertion.','expected_frame_rows':expected_frame,'expected_candidate_rows':expected_candidates,'aggregation':'rows are comparisons within 100 paired frames; pairs/model comparisons share frames, not independent statistical samples. Denominators explicit. Counts, median absolute effects and rank reversals are descriptive.'}
    (a.out/'definitions.json').write_text(json.dumps(definitions,indent=2,allow_nan=False)+'\n')
    print(f'wrote {len(frame)} frame rows and {len(candidates)} candidate rows to {a.out}',flush=True)
if __name__=='__main__': main()
