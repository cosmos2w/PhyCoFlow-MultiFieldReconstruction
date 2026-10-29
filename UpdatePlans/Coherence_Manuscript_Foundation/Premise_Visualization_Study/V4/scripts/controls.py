#!/usr/bin/env python3
"""Construct and redraw CPU-only controlled evidence from frozen V3 and GT arrays.

--redraw reads only V4/data/controls.npz, never the original sources.
"""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage

V4 = Path(__file__).resolve().parents[1]
STUDY = V4.parent
DATA = V4 / 'data' / 'controls.npz'
META = V4 / 'metadata' / 'controls.json'
FIG = V4 / 'figures'
NAMES = ['CH4', 'CO', 'T', 'U1', 'p']
FAMILIES = ['noise', 'convolution', 'bandphase', 'warp']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def corr(a,b): return float(np.corrcoef(a.ravel(),b.ravel())[0,1])


def make_sweeps(gt_archive, a):
    from scipy.stats import rankdata
    slots=np.linspace(0,len(gt_archive['global_indices'])-1,10,dtype=int)
    shifts=np.array([1,2,4,8,16]); pairs=np.array([[0,1],[0,3],[1,3]])
    # metrics: field errors 1/2, signed Pearson r, reference-fixed rank-bin TV.
    out=np.empty((len(slots),3,len(shifts),2,4),dtype=float)
    ids=gt_archive['global_indices'][slots]
    for si,slot in enumerate(slots):
        frame=gt_archive['fields'][slot].astype(float)
        for pi,(p,q) in enumerate(pairs):
            x,y=frame[:,:,p],frame[:,:,q]
            bx=np.minimum((rankdata(x.ravel(),method='average').reshape(x.shape)*10/x.size).astype(int),9)
            by=np.minimum((rankdata(y.ravel(),method='average').reshape(y.shape)*10/y.size).astype(int),9)
            refhist=np.bincount((bx*10+by).ravel(),minlength=100)/x.size
            for ki,k in enumerate(shifts):
                xp=np.roll(x,int(k),axis=1); bxp=np.roll(bx,int(k),axis=1)
                for vi,sgn in enumerate((1,-1)):
                    yp=np.roll(y,int(sgn*k),axis=1); byp=np.roll(by,int(sgn*k),axis=1)
                    hist=np.bincount((bxp*10+byp).ravel(),minlength=100)/x.size
                    out[si,pi,ki,vi]=[np.mean((xp-x)**2),np.mean((yp-y)**2),corr(xp,yp),.5*np.abs(hist-refhist).sum()]
                assert np.allclose(out[si,pi,ki,0,:2],out[si,pi,ki,1,:2],rtol=1e-12,atol=1e-12)
                assert out[si,pi,ki,0,3] < 1e-14
    # Analytic real-field Fourier control: for each amplitude, conjugate phases
    # yield identical per-field MSE, exact power and unchanged common cross.
    coeff=a['scale_coeff_reference']; delta=a['scale_phase_delta']; factors=np.linspace(0,4,21)
    scale=np.empty((len(factors),6)); cross0=coeff[0]*coeff[1].conj()
    for i,factor in enumerate(factors):
        phase=np.exp(1j*factor*delta)
        common=coeff*phase; opposed=coeff*np.stack((phase,phase.conj()))
        mse_c=np.mean(abs(common-coeff)**2,axis=(1,2)); mse_o=np.mean(abs(opposed-coeff)**2,axis=(1,2))
        power=max(np.max(abs(abs(common)**2-abs(coeff)**2)),np.max(abs(abs(opposed)**2-abs(coeff)**2)))
        imaginary=max(np.max(abs(np.fft.ifft2(common,axes=(-2,-1),norm='ortho').imag)),np.max(abs(np.fft.ifft2(opposed,axes=(-2,-1),norm='ortho').imag)))
        assert np.allclose(mse_c,mse_o,rtol=1e-12,atol=1e-12) and power<1e-10 and imaginary<1e-12
        cross_c=common[0]*common[1].conj(); cross_o=opposed[0]*opposed[1].conj()
        scale[i]=[mse_c[0],mse_c[1],np.linalg.norm(cross_c-cross0)/np.linalg.norm(cross0),np.linalg.norm(cross_o-cross0)/np.linalg.norm(cross0),power,imaginary]
    return dict(sweep_frame_ids=ids,sweep_shifts=shifts,sweep_pairs=pairs,sweep_metrics=out,
                sweep_scale_factors=factors,sweep_scale_metrics=scale)


def generate():
    toy_path = STUDY/'data/v3/toys.npz'
    gt_path = STUDY/'data/gt_test100.npz'
    candidate_path = STUDY/'data/v3/controlled_candidates.npz'
    with np.load(toy_path) as f: toy = {k: f[k] for k in f.files}
    with np.load(gt_path) as f:
        slot = int(np.flatnonzero(f['global_indices'] == 9575).item())
        gt = f['fields'][slot].astype(np.float64)
        obs = f['obs_indices'][slot].copy()
        sweeps = make_sweeps(f,toy)
    with np.load(candidate_path) as f:
        ix = np.flatnonzero(f['frame_ids'] == 9575)
        candidates = f['fields'][ix].copy()
        ids = f['candidate_ids'][ix].copy()
    # On the equal-weight index grid, a permutation and its inverse have
    # identical squared distance from the original vector. Common permutation
    # preserves CH4-CO joint occupation exactly; opposing it changes pairing.
    common = gt.copy(); opposed = gt.copy(); k = 8
    for j in (0,1): common[:,:,j] = np.roll(gt[:,:,j], k, axis=1)
    opposed[:,:,0] = common[:,:,0]
    opposed[:,:,1] = np.roll(gt[:,:,1], -k, axis=1)
    arrays = dict(toy, real_gt=gt, real_common=common, real_opposed=opposed,
                  real_candidates=candidates, candidate_ids=ids, obs_indices=obs,
                  shift=np.array(k), **sweeps)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA, **arrays)
    return arrays, {str(p.relative_to(STUDY)): sha(p) for p in (toy_path, gt_path, candidate_path)}


def audit(a, sources=None):
    ref=a['real_gt']; c=a['real_common']; d=a['real_opposed']; obs=a['obs_indices']
    mse=lambda x: np.mean((x-ref)**2, axis=(0,1))
    mc,md=mse(c),mse(d)
    assert np.allclose(mc,md,rtol=1e-12,atol=1e-14)
    assert np.array_equal(c.reshape(-1,5)[obs,2],ref.reshape(-1,5)[obs,2])
    assert np.array_equal(d.reshape(-1,5)[obs,2],ref.reshape(-1,5)[obs,2])
    for j in (0,1):
        assert np.array_equal(np.sort(c[:,:,j].ravel()),np.sort(ref[:,:,j].ravel()))
        assert np.array_equal(np.sort(d[:,:,j].ravel()),np.sort(ref[:,:,j].ravel()))
    assert np.array_equal(np.sort(c[:,:,:2].reshape(-1,2),axis=0),np.sort(ref[:,:,:2].reshape(-1,2),axis=0))
    assert np.array_equal(c[:,:,:2], np.roll(ref[:,:,:2],int(a['shift']),axis=1))
    tests=[]
    for name, x in zip(a['candidate_ids'],a['real_candidates']):
        m=mse(x)
        assert np.max(np.abs(m-0.004)) < 1e-10
        assert np.array_equal(x.reshape(-1,5)[obs,2],ref.reshape(-1,5)[obs,2])
        tests.append({'id': str(name), 'mse':dict(zip(NAMES,m.tolist())),
                      'ch4_co_pearson':corr(x[:,:,0],x[:,:,1])})
    toy_metrics={}
    for name, refkeys, opts in [
        ('state', ['state_u_reference','state_v_reference'], [('state_u_hat_preserving','state_v_preserving'),('state_u_hat_broken','state_v_broken')]),
        ('shape', ['shape_A_reference','shape_B_reference'], [('shape_A_estimate','shape_B_preserving'),('shape_A_estimate','shape_B_broken')])]:
        values=[[float(np.mean((a[k]-a[r])**2)) for k,r in zip(option,refkeys)] for option in opts]
        assert np.allclose(values[0],values[1],atol=1e-12,rtol=1e-10), (name,values)
        toy_metrics[name+'_mse']=values
    scale_mse=[[float(np.mean((a['scale_field_'+variant][j]-a['scale_field_reference'][j])**2)) for j in range(2)] for variant in ('shared','opposed')]
    assert np.allclose(scale_mse[0],scale_mse[1],atol=1e-12,rtol=1e-10)
    power_error=max(float(np.max(np.abs(np.abs(a['scale_coeff_'+variant])**2-np.abs(a['scale_coeff_reference'])**2))) for variant in ('shared','opposed'))
    assert power_error < 1e-10
    toy_metrics['scale_mse']=scale_mse
    toy_metrics['scale_auto_power_max_absolute_difference']=power_error
    toy_metrics['shape_joint_beta0_at_zero']=[int(ndimage.label(a['shape_joint_'+v+'_tau0'],structure=np.ones((3,3)))[1]) for v in ('preserving','broken')]
    assert toy_metrics['shape_joint_beta0_at_zero']==[1,2]
    sm=a['sweep_metrics']; ss=a['sweep_scale_metrics']
    assert np.max(np.abs(sm[:,:,:,0,:2]-sm[:,:,:,1,:2])) < 1e-12
    assert np.max(np.abs(sm[:,:,:,0,3])) < 1e-14
    assert np.max(ss[:,4]) < 1e-10 and np.max(ss[:,5]) < 1e-12
    aa=a['shape_A_estimate']; ar=a['shape_A_reference']
    aa=aa / (np.sum(aa*ar)/np.sum(ar**2))
    def betti(mask):
        b0=int(ndimage.label(mask,structure=np.ones((3,3)))[1])
        labels,n=ndimage.label(~mask,structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        boundary=set(np.concatenate((labels[0],labels[-1],labels[:,0],labels[:,-1])))
        return b0,sum(i not in boundary for i in range(1,n+1))
    for i,tau in enumerate(a['shape_thresholds']):
        ma=aa>tau; mp=a['shape_B_preserving']>tau; mb=a['shape_B_broken']>tau
        assert betti(ma)==betti(mp)==betti(mb)
        for name,mask in (('preserving',ma & mp),('broken',ma & mb)):
            assert betti(mask)[0]==a['shape_joint_beta0_'+name][i]
            assert np.isclose(mask.mean(),a['shape_joint_area_fraction_'+name][i],rtol=0,atol=1e-15)
    toy_metrics['shape_unary_betti_equal_at_all_15_thresholds']=True
    result={'reference_frame':9575, 'toy_checks_from_retained_arrays':toy_metrics, 'sources_sha256':sources,
      'permutation': {'shift_columns':int(a['shift']), 'mse_common':dict(zip(NAMES,mc.tolist())),
        'mse_opposed':dict(zip(NAMES,md.tolist())), 'max_mse_difference':float(np.max(abs(mc-md))),
        'ch4_co_pearson':{'reference':corr(ref[:,:,0],ref[:,:,1]),'common':corr(c[:,:,0],c[:,:,1]),'opposed':corr(d[:,:,0],d[:,:,1])},
        'contract':'Exact unweighted sample-site marginal permutations and paired MSE for all five fields; CH4-CO joint occupation preserved by common shift, not necessarily their joint with unchanged fields. T is unchanged including observed sites.',
        'limitations':'Cyclic index-grid wrap introduces a nonphysical seam; index spacing is nonuniform, so this is not a physical translation or area-weighted invariance. This is an intentionally strong constructed perturbation, not a model prediction.'},
      'matched_interventions':tests,
      'matched_intervention_contract':'V3 calibrated residuals: equal 0.004 MSE in training-normalized units per field; observed T unchanged; clipping and individual amplitude calibration mean shared maps need not preserve joint occupancy, topology or exact power. Noise, convolution, bandphase, and warp are synthetic interventions, not learned errors.',
      'toy_contract':'Retained V3 strict constructed toys: State matched per-field MSE; Scale periodic synthetic field with matched unary Fourier power and MSE and nonrigid shared/opposed phases; Shape smooth warps with matched field MSE, matching unary beta0/beta1 over declared thresholds and different joint beta0. These do not establish natural or causal mechanisms.',
      'sweeps': {'frames':a['sweep_frame_ids'].tolist(),'pairs':[[NAMES[i],NAMES[j]] for i,j in a['sweep_pairs']], 'shifts_columns':a['sweep_shifts'].tolist(), 'rank_histogram':'10 by 10 rank bins, average ranks for ties, reference bin assignments permuted with values; unweighted sample sites', 'phase_factors':a['sweep_scale_factors'].tolist(), 'phase_sweep':'Synthetic periodic Hermitian Fourier fields; max power and imaginary residue are in retained sweep metrics. Shape threshold sweep reuses V3 fields and is evaluated at the 15 declared thresholds.', 'limitations':'Cyclic wrapping and nonuniform physical grid apply to the real-data sweep. Same-site correlation and rank-bin TV are selected descriptors, not complete physical coherence; shift and frame comparisons are not independent model evaluations.'},
      'negative_findings':'A common permutation preserves CH4-CO occupation despite spatial relocation; matching unary power cannot detect the Scale phase contrast; matching unary beta0/beta1 cannot detect the Shape joint intersection contrast. Sensitivity depends on the declared observable.'}
    return result


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight')
    plt.close(fig)


def plot(a, audit_data):
    plt.rcParams.update({'font.size':12,'axes.titlesize':13,'axes.labelsize':12,'figure.titlesize':16,'savefig.facecolor':'white'})
    # Exact real-field State pairing; common limits by field across panels.
    fig, ax=plt.subplots(2,4,figsize=(17,7.6),constrained_layout=True)
    real=[a['real_gt'],a['real_common'],a['real_opposed']]
    for row,j in enumerate((0,1)):
        lo=min(np.min(x[:,:,j]) for x in real); hi=max(np.max(x[:,:,j]) for x in real)
        for col,x in enumerate(real):
            im=ax[row,col].imshow(x[:,:,j],origin='lower',aspect='auto',vmin=lo,vmax=hi,cmap='viridis')
            if row==0: ax[row,col].set_title(['Reference','Common shift','Opposed shift'][col])
            ax[row,col].set_ylabel(NAMES[j]+' · grid row'); ax[row,col].set_xlabel('Grid column')
        fig.colorbar(im,ax=list(ax[row,:3]),shrink=.78,label='Normalized value')
    # Fixed subsample for faithful, readable joint plot.
    for x,label,col in zip(real,['Reference','Common','Opposed'],['#222222','#0072B2','#D55E00']):
        ax[0,3].scatter(x[::3,::5,0],x[::3,::5,1],s=2,alpha=.12,label=label,color=col)
    ax[0,3].set(xlabel='CH4 normalized',ylabel='CO normalized',title='Same-site pairs (subsample)'); ax[0,3].legend(markerscale=4)
    v=audit_data['permutation']['ch4_co_pearson']
    ax[1,3].bar(['Reference','Common','Opposed'],list(v.values()),color=['#777777','#0072B2','#D55E00'])
    ax[1,3].set(ylabel='Pearson correlation',title='Full-grid CH4–CO pairing'); ax[1,3].axhline(0,color='black',lw=.7)
    fig.suptitle('Real GT intervention: exact equal field errors, different pairing')
    save(fig,'controls_real_state')
    # Strict examples display actual arrays and diagnostics; each metric is scoped.
    fig,ax=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    for i,(variant,label) in enumerate([('preserving','Shared'),('broken','Differential')]):
        u=a['state_u_hat_'+variant]; v=a['state_v_'+variant]
        ax[0,0].scatter(u[::4,::4],v[::4,::4],s=2,alpha=.18,label=label)
    ax[0,0].set(xlabel='Estimated U',ylabel='Estimated V',title='State: same-site pairs'); ax[0,0].legend()
    ax[1,0].plot(a['state_x'],a['state_v_preserving'][len(a['state_y'])//2],label='Shared')
    ax[1,0].plot(a['state_x'],a['state_v_broken'][len(a['state_y'])//2],label='Differential')
    ax[1,0].set(xlabel='x',ylabel='V',title='State: center-row profile'); ax[1,0].legend()
    k=a['scale_shell_k']
    for variant,label in [('reference','Reference'),('shared','Shared'),('opposed','Opposed')]:
        ax[0,1].plot(k,a['scale_cross_normalized_product_'+variant+'_re_im'][0],label=label)
        ax[1,1].plot(k,a['scale_auto_'+variant][0],label=label)
    ax[0,1].set(xlabel='Shell index',ylabel='Real cross product',title='Scale: signed normalized cross product'); ax[0,1].legend()
    ax[1,1].set(xlabel='Shell index',ylabel='U auto power',title='Scale: unary power'); ax[1,1].set_yscale('symlog',linthresh=1e-8); ax[1,1].legend()
    ax[0,2].imshow(a['shape_joint_preserving_tau0'],origin='lower',cmap='Blues',vmin=0,vmax=1,aspect='auto')
    ax[0,2].set_title('Shape: shared joint excursion, τ = 0')
    ax[1,2].imshow(a['shape_joint_broken_tau0'],origin='lower',cmap='Oranges',vmin=0,vmax=1,aspect='auto')
    ax[1,2].set_title('Shape: differential joint excursion, τ = 0')
    for z in ax[:,2]: z.set(xlabel='Grid column',ylabel='Grid row')
    fig.suptitle('Strict constructed controls: three complementary observables')
    save(fig,'controls_strict_toys')
    # Matched-MSE intervention family response, including negative/small responses.
    fig,ax=plt.subplots(2,4,figsize=(17,8),constrained_layout=True)
    gt=a['real_gt']; xs=a['real_candidates']; base=corr(gt[:,:,0],gt[:,:,1])
    global_lim=float(np.quantile(np.abs(xs[:,:,:,1]-gt[:,:,1]),.99))
    all_changes=[corr(x[:,:,0],x[:,:,1])-base for x in xs]
    bar_lim=max(abs(x) for x in all_changes)*1.15
    for j,fam in enumerate(FAMILIES):
        pair=xs[2*j:2*j+2]
        for i,x in enumerate(pair):
            delta=x[:,:,1]-gt[:,:,1]
            lim=global_lim
            im=ax[0,j].imshow(delta,origin='lower',aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim)
            if i==0: break
        ax[0,j].set(title=fam.capitalize()+': shared CO residual',xlabel='Grid column',ylabel='Grid row')
        fig.colorbar(im,ax=ax[0,j],shrink=.75)
        vals=[corr(x[:,:,0],x[:,:,1])-base for x in pair]
        ax[1,j].bar(['Shared','Mismatched'],vals,color=['#0072B2','#D55E00'])
        ax[1,j].axhline(0,color='black',lw=.8)
        ax[1,j].set_ylim(-bar_lim,bar_lim)
        ax[1,j].set(title='Change in CH4–CO correlation',ylabel='Δ Pearson r')
    fig.suptitle('Four synthetic real-GT intervention families: each field MSE = 0.004')
    save(fig,'controls_real_families')


def plot_sweeps(a):
    m=a['sweep_metrics']; shifts=a['sweep_shifts']; pairs=a['sweep_pairs']
    fig,axes=plt.subplots(3,3,figsize=(15,11),constrained_layout=True)
    for j,(u,v) in enumerate(pairs):
        for h,idx in enumerate((0,1)):
            val=m[:,j,:,0,idx]
            axes[0,j].plot(shifts,np.median(val,axis=0),marker='o',label=NAMES[(u,v)[h]])
            axes[0,j].fill_between(shifts,np.min(val,axis=0),np.max(val,axis=0),alpha=.1)
        axes[0,j].set(title=NAMES[u]+'–'+NAMES[v],ylabel='Per-field MSE',xlabel='Shift in columns')
        axes[0,j].legend()
        vals=m[:,j,:,1,3]
        for row in vals: axes[1,j].plot(shifts,row,color='#D55E00',alpha=.19,lw=1)
        axes[1,j].plot(shifts,np.median(vals,axis=0),color='#D55E00',marker='o',label='Opposed median')
        axes[1,j].plot(shifts,np.median(m[:,j,:,0,3],axis=0),color='#0072B2',marker='s',label='Common median')
        axes[1,j].set(ylabel='Rank-bin joint TV',xlabel='Shift in columns',ylim=(0,1)); axes[1,j].legend(fontsize=9)
        vals=m[:,j,:,1,2]-m[:,j,:,0,2]
        for row in vals: axes[2,j].plot(shifts,row,color='#D55E00',alpha=.19,lw=1)
        axes[2,j].plot(shifts,np.median(vals,axis=0),color='#D55E00',marker='o')
        axes[2,j].axhline(0,color='black',lw=.7)
        axes[2,j].set(ylabel='Opposed − common Pearson r',xlabel='Shift in columns')
        for row in range(3): axes[row,j].set_xticks(shifts)
    fig.suptitle('Real-field permutation sensitivity: 10 frozen frames, 3 pairs')
    save(fig,'controls_permutation_sweep')
    scale=a['sweep_scale_metrics']; factor=a['sweep_scale_factors']; tau=a['shape_thresholds']
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    axes[0,0].plot(factor,scale[:,0],label='Field 1',marker='o',ms=3)
    axes[0,0].plot(factor,scale[:,1],label='Field 2',marker='o',ms=3)
    axes[0,0].set(xlabel='Nonrigid phase multiplier',ylabel='Per-field MSE',title='Identical shared / opposed errors'); axes[0,0].legend()
    axes[0,1].plot(factor,scale[:,2],label='Shared',marker='o',ms=3)
    axes[0,1].plot(factor,scale[:,3],label='Opposed',marker='o',ms=3)
    axes[0,1].set(xlabel='Nonrigid phase multiplier',ylabel='Relative complex cross-product change',title='Cross-spectrum sensitivity');axes[0,1].legend()
    for v,label in [('preserving','Shared'),('broken','Differential')]:
        axes[1,0].step(tau,a['shape_joint_beta0_'+v],where='mid',label=label)
        axes[1,1].plot(tau,a['shape_joint_area_fraction_'+v],marker='o',ms=3,label=label)
    axes[1,0].set(xlabel='Excursion threshold',ylabel='Joint β₀ (8-neighbor)',title='Joint connectivity across thresholds',ylim=(0,3));axes[1,0].legend()
    axes[1,1].set(xlabel='Excursion threshold',ylabel='Joint area fraction',title='Joint overlap across thresholds');axes[1,1].legend()
    fig.suptitle('Strict synthetic sensitivity: fixed unary power; matched unary Betti numbers')
    save(fig,'controls_strict_sweep')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--redraw',action='store_true'); args=p.parse_args()
    if args.redraw:
        with np.load(DATA,allow_pickle=False) as f: arrays={k:f[k] for k in f.files}
        old=json.loads(META.read_text()) if META.exists() else {}; sources=old.get('sources_sha256')
    else: arrays,sources=generate()
    result=audit(arrays,sources)
    META.parent.mkdir(parents=True,exist_ok=True); META.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    plot(arrays,result)
    plot_sweeps(arrays)
    print(json.dumps({'data':str(DATA),'metadata':str(META),'permutation':result['permutation'],'figures':['controls_real_state','controls_strict_toys','controls_real_families','controls_permutation_sweep','controls_strict_sweep']},indent=2))
if __name__=='__main__': main()
