"""Render a source-backed Round 8 Markdown/HTML report and scientific figures.

All inputs are existing run artifacts. No scoring, model inference, or decision
fitting is performed. ``build_report`` never creates a final case report itself;
callers select an output directory after producing the artifacts.
"""
from __future__ import annotations
import html
import json
import math
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path
import numpy as np

from .round8_campaign import file_hash
from .round8_decision import KEYS, _mean_scores
from .round8_inference import validate_matched_panels


def _json(path):
    with Path(path).open() as stream: return json.load(stream)


def _num(value,fmt='.4g'):
    if value is None: return 'unavailable'
    v=float(value)
    return format(v,fmt) if math.isfinite(v) else 'unavailable'


def _escape(value): return str(value).replace('|','\\|').replace('\n',' ')


def _save_figure(fig,directory,stem):
    directory.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','png'): fig.savefig(directory/f'{stem}.{ext}',bbox_inches='tight',dpi=170)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return f'figures/{stem}.png',f'figures/{stem}.pdf'


def _training_figure(histories,figures):
    """Read actual run log round8.scores; never synthesize a ConFIG total."""
    import matplotlib.pyplot as plt
    series=[]
    for arm,path in histories.items():
        rows=[]
        with Path(path).open() as stream:
            for line in stream:
                if line.strip(): rows.append(json.loads(line))
        for family in ('state','scale','shape'):
            for kind in ('raw','calibrated'):
                points=[]
                for row in rows:
                    epoch=row.get('epoch'); value=row.get('round8',{}).get('scores',{}).get(family,{}).get(kind)
                    if isinstance(epoch,(int,float)) and isinstance(value,(int,float)) and math.isfinite(epoch) and math.isfinite(value): points.append((epoch,value))
                if points: series.append((arm,family,kind,points))
    if not series: return None
    fig,axes=plt.subplots(3,1,figsize=(8,8),sharex=True,layout='constrained')
    for ax,family in zip(axes,('state','scale','shape')):
        for arm,f,k,points in series:
            if f==family:
                x,y=zip(*points);ax.plot(x,y,label=f'{arm} {k}',linestyle='-' if k=='raw' else '--',linewidth=1)
        ax.set_ylabel(f'{family} score');ax.grid(alpha=.2);ax.axvline(10,color='0.5',lw=.7,ls=':')
        if ax.get_legend_handles_labels()[1]: ax.legend(fontsize=7,ncol=2)
        else: ax.text(.5,.5,'No recorded scores',transform=ax.transAxes,ha='center',va='center')
    axes[-1].set_xlabel('Epoch');fig.suptitle('Sampled TRAIN family scores; dotted line: epoch 10')
    return _save_figure(fig,figures,'training_scores')


def _diagnostic_values(value, terms):
    """Find explicitly named scalar diagnostics without deriving missing values."""
    result={}
    def walk(node,path):
        if isinstance(node,dict):
            for key,item in node.items(): walk(item,path+(str(key),))
        elif isinstance(node,list) and node and all(isinstance(x,(int,float)) and math.isfinite(x) for x in node):
            key=path[-1].lower() if path else ''
            if key in terms: result['.'.join(path)]=float(np.mean(node))
        elif isinstance(node,(int,float)) and not isinstance(node,bool) and math.isfinite(node):
            key=path[-1].lower() if path else ''
            if key in terms: result['.'.join(path)]=float(node)
    walk(value,())
    return result


def _assessment_figure(series,figures):
    """Rows: {'endpoint','epoch','population','path'} to common score reports."""
    import matplotlib.pyplot as plt
    points=[]; diagnostics=[]
    for item in series:
        report=_json(item['path'])
        if report.get('schema')!='round8_common_observers_v1' or report.get('population')!=item['population']:
            raise ValueError('assessment history report identity mismatch')
        if 'alpha' in item and not math.isclose(float(item['alpha']),float(report['alpha']),abs_tol=1e-12): raise ValueError('assessment history alpha mismatch')
        means=_mean_scores(report)
        for family in KEYS: points.append((item['endpoint'],item['population'],family,float(item['epoch']),means[family],float(report['alpha'])))
        for family in ('global_distribution','multiscale_spectrum','topology'):
            columns={}
            for row in report['rows']:
                for key,value in _diagnostic_values(row.get('diagnostics',{}).get(family,{}),{'attraction','separation','repulsion','attraction_per_sample','separation_per_sample','repulsion_per_sample','weighted_attraction','weighted_separation'}).items(): columns.setdefault(key,[]).append(value)
            for key,values in columns.items():
                if len(values)==len(report['rows']): diagnostics.append((item['endpoint'],item['population'],family,key,float(item['epoch']),float(np.mean(values)),float(report['alpha'])))
    if not points: return []
    fig,axes=plt.subplots(3,1,figsize=(8,8),sharex=True,layout='constrained')
    for ax,family in zip(axes,('state','scale','shape')):
        labels=sorted({(a,p,alpha) for a,p,f,_,_,alpha in points if f==family})
        for arm,pop,alpha in labels:
            xy=sorted((epoch,value) for a,p,f,epoch,value,al in points if (a,p,f,al)==(arm,pop,family,alpha))
            if xy: ax.plot(*zip(*xy),marker='o',label=f'{arm} {pop} α={alpha:g}',lw=1)
        ax.set_ylabel(family);ax.grid(alpha=.2);ax.legend(fontsize=6,ncol=2)
    axes[-1].set_xlabel('Epoch');fig.suptitle('Common observer assessments on declared panel/support')
    output=[('assessments',_save_figure(fig,figures,'assessment_scores'))]
    if diagnostics:
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
        labels=sorted({(a,p,f,k,al) for a,p,f,k,_,_,al in diagnostics})
        for arm,pop,family,key,alpha in labels:
            xy=sorted((epoch,value) for a,p,f,k,epoch,value,al in diagnostics if (a,p,f,k,al)==(arm,pop,family,key,alpha))
            ax.plot(*zip(*xy),marker='o',label=f'{arm} {pop} {family} {key} α={alpha:g}',lw=1)
        ax.set(xlabel='Epoch',ylabel='Recorded diagnostic',title='Recorded attraction and separation diagnostics');ax.legend(fontsize=5,ncol=2);ax.grid(alpha=.2)
        output.append(('attraction_separation',_save_figure(fig,figures,'attraction_separation')))
    return output


def _reconstruction(spec,figures,index):
    """One condition/field/member native scatter from matched cached endpoints."""
    import matplotlib.pyplot as plt
    source, endpoint=validate_matched_panels(spec['source_panel'],spec['endpoint_panel'])
    found=[(a,b) for a,b in zip(source['conditions'],endpoint['conditions']) if a['sample_id']==spec['sample_id']]
    if len(found)!=1: raise ValueError('reconstruction sample absent or ambiguous')
    s,e=found[0];alpha=float(spec['alpha']);member=int(spec.get('member',0));field=str(spec['field'])
    if not 0<=alpha<=1 or member not in (0,1) or field not in source['field_names']: raise ValueError('invalid reconstruction selection')
    fi=source['field_names'].index(field)
    with np.load(Path(spec['source_panel']).parent/s['path'],allow_pickle=False) as a, np.load(Path(spec['endpoint_panel']).parent/e['path'],allow_pickle=False) as b:
        for key in ('target_physical','raw_coordinates','physical_weights','physical_mask','native_ids','field_units'):
            if not np.array_equal(a[key],b[key]): raise ValueError('reconstruction matched support differs')
        mask=a['physical_mask'].astype(bool);coords=a['raw_coordinates'][mask]; weights=a['physical_weights'][mask].astype(float)
        if coords.ndim!=2 or coords.shape[1]<2 or weights.sum()<=0: raise ValueError('physical 2D support required')
        truth=a['target_physical'][mask,fi].astype(float)
        pred=(a['draws_physical'][member,mask,fi]+alpha*(b['draws_physical'][member,mask,fi]-a['draws_physical'][member,mask,fi])).astype(float)
        if not all(np.isfinite(v).all() for v in (coords,weights,truth,pred)) or (weights<0).any(): raise ValueError('invalid reconstruction data')
        rmse=float(np.sqrt(np.sum((weights/weights.sum())*(pred-truth)**2)))
        unit=str(a['field_units'][fi]); source_id=str(s['sample_id'])
        fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
        low,high=float(min(truth.min(),pred.min())),float(max(truth.max(),pred.max()))
        for ax,data,title in zip(axes,(truth,pred,pred-truth),('Reference','Reconstruction',f'Error; weighted RMSE={rmse:.4g} {unit}')):
            kwargs={'vmin':low,'vmax':high} if title!='Error' and not title.startswith('Error;') else {}
            im=ax.scatter(coords[:,0],coords[:,1],c=data,s=3,cmap='viridis' if kwargs else 'coolwarm',**kwargs)
            ax.set_title(title);ax.set_aspect('equal');fig.colorbar(im,ax=ax,shrink=.8)
        fig.suptitle(f"{field}, {source_id}, member {member}, α={alpha:g}; {'native physical mask' if source['population']=='ASSESS' else 'sampled physical mask'}")
    return _save_figure(fig,figures,f'reconstruction_{index:02d}'),{'rmse':rmse,'field':field,'units':unit,'sample_id':source_id,'support':source['population'],'alpha':alpha,'member':member}


def _condition_summary(values, *, bootstrap_samples=1000, seed=47):
    """Descriptive frame-resampling intervals, not an independence guarantee."""
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all(): raise ValueError('invalid condition values')
    interval=None
    if len(x)>1:
        rng=np.random.default_rng(seed)
        draws=np.empty(bootstrap_samples,dtype=np.float64)
        # Bound memory even for large panels.
        for i in range(bootstrap_samples): draws[i]=x[rng.integers(len(x),size=len(x))].mean()
        interval=[float(v) for v in np.quantile(draws,[.025,.975])]
    return {'mean':float(x.mean()),'n':len(x),'frame_bootstrap_95':interval}


def cached_field_diagnostics(spec):
    """Stream physical J2 fields; calculate spread, displacement and moments.

    Spread is unbiased J2 member variance, sum_w (z1-z2)^2/2. Displacement is
    mean_j sum_w (z_alpha,j-z_source,j)^2. Pair relationship is the ensemble
    mean of spatial raw cross-moments sum_w z_a z_b minus sum_w target_a target_b,
    in physical product units. All use the declared HT weights on TRAIN and
    native masses on ASSESS; no random weight renormalization is used.
    """
    source,endpoint=validate_matched_panels(spec['source_panel'],spec['endpoint_panel'])
    alpha=float(spec['alpha'])
    if not 0<=alpha<=1: raise ValueError('invalid fixed coefficient')
    fields=tuple(source['field_names'])
    if spec.get('regime_manifest') and 'regime_by_id' in spec: raise ValueError('provide one regime source')
    if spec.get('regime_manifest'):
        manifest_path=spec['regime_manifest']
        if file_hash(manifest_path)!=source['manifest_sha256']: raise ValueError('regime manifest differs from panel manifest')
        manifest=_json(manifest_path)
        regimes={str(row['id']):str(row.get('regime','unknown')) for row in manifest[source['population']]}
        regime_provenance={'manifest_sha256':file_hash(manifest_path)}
    else:
        regimes=spec.get('regime_by_id',{})
        regime_provenance={'description':'caller-supplied mapping' if regimes else 'regime metadata unavailable'}
    rows=[]; units=None
    for a,b in zip(source['conditions'],endpoint['conditions']):
        with np.load(Path(spec['source_panel']).parent/a['path'],allow_pickle=False) as x, np.load(Path(spec['endpoint_panel']).parent/b['path'],allow_pickle=False) as y:
            for key in ('target_physical','physical_weights','physical_mask','native_ids','inclusion_probability','field_units'):
                if not np.array_equal(x[key],y[key]): raise ValueError('field diagnostic endpoint support differs')
            mask=x['physical_mask'].astype(bool)
            p=float(x['inclusion_probability']); w=x['physical_weights'][mask].astype(np.float64)/p
            z=x['draws_physical'][:,mask,:].astype(np.float64); target=x['target_physical'][mask].astype(np.float64)
            repair=y['draws_physical'][:,mask,:].astype(np.float64)
            if not 0<p<=1 or z.shape!=(2,len(w),len(fields)) or repair.shape!=z.shape or target.shape!=z.shape[1:] or not all(np.isfinite(v).all() for v in (w,z,target,repair)) or (w<0).any(): raise ValueError('invalid cached field diagnostic input')
            if source['population']=='ASSESS' and (p!=1 or not np.isclose(w.sum(),1)): raise ValueError('ASSESS needs native normalized masses')
            actual_units=tuple(str(u) for u in x['field_units'])
            if units is None: units=actual_units
            elif units!=actual_units: raise ValueError('physical field units differ across conditions')
            blend=z+alpha*(repair-z)
            def spread(v): return np.sum(w[:,None]*(v[0]-v[1])**2/2,axis=0)
            displacement=np.sum(w[:,None]*np.mean((blend-z)**2,axis=0),axis=0)
            mean_error=np.sum(w[:,None]*(blend.mean(axis=0)-target),axis=0)
            relations=[]
            for i in range(len(fields)):
                for j in range(i+1,len(fields)):
                    ref=np.sum(w*target[:,i]*target[:,j])
                    src=np.mean([np.sum(w*v[:,i]*v[:,j]) for v in z])-ref
                    candidate=np.mean([np.sum(w*v[:,i]*v[:,j]) for v in blend])-ref
                    relations.append({'fields':[fields[i],fields[j]],'source_error':float(src),'blend_error':float(candidate)})
            rows.append({'sample_id':a['sample_id'],'legacy_source':a['source'],'regime':str(regimes.get(a['sample_id'],'unknown')),'source_spread_variance':spread(z).tolist(),'blend_spread_variance':spread(blend).tolist(),'displacement_mse':displacement.tolist(),'ensemble_mean_error':mean_error.tolist(),'relationships':relations})
    if not rows: raise ValueError('no cached field conditions')
    def summarize(selected):
        out={'n':len(selected),'fields':{},'relationships':{}}
        for i,name in enumerate(fields):
            src=np.array([r['source_spread_variance'][i] for r in selected]); blend=np.array([r['blend_spread_variance'][i] for r in selected]); disp=np.array([r['displacement_mse'][i] for r in selected])
            out['fields'][name]={'unit':units[i],'source_rms_spread':float(np.sqrt(src.mean())),'blend_rms_spread':float(np.sqrt(blend.mean())),'rms_displacement':float(np.sqrt(disp.mean())),'spread_variance_change':_condition_summary(blend-src),'ensemble_mean_error':_condition_summary([r['ensemble_mean_error'][i] for r in selected])}
        for index in range(len(selected[0]['relationships'])):
            relation=selected[0]['relationships'][index]; key=' / '.join(relation['fields'])
            out['relationships'][key]={'fields':relation['fields'],'units':units[fields.index(relation['fields'][0])]+' * '+units[fields.index(relation['fields'][1])], 'source_absolute_error':_condition_summary([abs(r['relationships'][index]['source_error']) for r in selected]),'blend_absolute_error':_condition_summary([abs(r['relationships'][index]['blend_error']) for r in selected]),'paired_absolute_error_change':_condition_summary([abs(r['relationships'][index]['blend_error'])-abs(r['relationships'][index]['source_error']) for r in selected])}
        return out
    groups={category:{value:summarize([r for r in rows if r[category]==value]) for value in sorted({r[category] for r in rows})} for category in ('legacy_source','regime')}
    return {'schema':'round8_cached_field_diagnostics_v1','endpoint':spec['endpoint'],'population':source['population'],'alpha':alpha,'source_panel_sha256':file_hash(spec['source_panel']),'endpoint_panel_sha256':file_hash(spec['endpoint_panel']),'field_names':fields,'regime_provenance':regime_provenance,'pooled':summarize(rows),'groups':groups,'per_condition':rows,'support':'native physical mask' if source['population']=='ASSESS' else 'fixed Q spatial Horvitz-Thompson proxy','uncertainty':'95% frame-bootstrap intervals are descriptive; correlated frames are not independent realizations.','relationship_definition':'physical area mean of member cross-products minus target cross-product, per field pair'}


def exposure_diagnostics(path,field_names):
    """Summarize streamed sparse query IDs and multiplicities without inference."""
    from ..coherence.round8_queries import index_pair, pair_count
    fields=tuple(field_names); counts={'self':{},'pair':{}}; batches=set()
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip(): continue
            row=json.loads(line);kind=row['kind'];index=int(row['index']); count=int(row['count']); probability=float(row['probability'])
            population=len(fields) if kind=='self' else pair_count(len(fields)) if kind=='pair' else 0
            if not 0<=index<population or count<1 or not 0<probability<=1: raise ValueError('invalid query exposure record')
            counts[kind][index]=counts[kind].get(index,0)+count;batches.add(int(row['batch']))
    result={}
    for kind in counts:
        total=len(fields) if kind=='self' else pair_count(len(fields))
        result[kind]={'population':total,'unique':len(counts[kind]),'draws':sum(counts[kind].values()),'per_query':[{'fields':[fields[i]] if kind=='self' else [fields[j] for j in index_pair(len(fields),i)],'draws':n} for i,n in sorted(counts[kind].items())]}
    return {'batches':len(batches),'queries':result,'sha256':file_hash(path)}


def cost_ledger_summary(path):
    """Add completed GPU lease intervals across devices and report open leases."""
    ledger=_json(path); groups={}; open_count=0; by_device={}
    for row in ledger['intervals']:
        if 'end' not in row: open_count+=1;continue
        start,end=float(row['start']),float(row['end'])
        if not math.isfinite(start) or not math.isfinite(end) or end<start: raise ValueError('invalid GPU lease interval')
        if not row.get('device_uuid'): raise ValueError('completed GPU lease needs a device UUID')
        by_device.setdefault(row['device_uuid'],[]).append((start,end))
        key=str(row['stage']); groups[key]=groups.get(key,0.)+end-start
    for spans in by_device.values():
        spans.sort()
        if any(spans[i][0] < spans[i-1][1] for i in range(1,len(spans))):
            raise ValueError('overlapping GPU leases on the same device')
    return {'seconds_by_stage':groups,'total_completed_gpu_seconds':sum(groups.values()),'open_intervals_excluded':open_count,'sha256':file_hash(path)}


class _Links(HTMLParser):
    def __init__(self): super().__init__(); self.links=[]; self.images=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='a' and a.get('href'): self.links.append(a['href'])
        if tag=='img' and a.get('src'): self.images.append(a['src'])


def validate_report(directory):
    """Render HTML and validate relative links, PNGs, and every PDF figure page.

    Verification checks rendering and file decoding; it does not claim a human
    visual review. PDF masters are rasterized page by page for decoding.
    """
    from markdown_it import MarkdownIt
    from PIL import Image
    import tempfile
    root=Path(directory); text=(root/'REPORT.md').read_text()
    rendered=MarkdownIt('commonmark',{'html':False}).enable('table').render(text)
    document='<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Round 8 report</title><style>body{font:16px system-ui;max-width:1050px;margin:auto;padding:2rem;line-height:1.5}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:.35rem}code{background:#eee}</style></head><body>'+rendered+'</body></html>'
    (root/'REPORT.html').write_text(document)
    # Keep the Markdown/HTML Unicode text and use explicit TeX math in the
    # print input so the installed pdfLaTeX engine renders scientific symbols.
    print_text = text
    for symbol, replacement in {'α':r'$\alpha$', 'θ':r'$\theta$', '≤':r'$\le$',
                                '≥':r'$\ge$', '×':r'$\times$', '−':'-'}.items():
        print_text = print_text.replace(symbol, replacement)
    print_input = root / '.report_print.md'
    print_input.write_text(print_text)
    try:
        subprocess.run(['pandoc', str(print_input.resolve()), '-o', str((root/'REPORT.pdf').resolve()),
                        '--pdf-engine=pdflatex', '-V', 'geometry:margin=0.7in',
                        '-V', 'fontsize=10pt'], cwd=root, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finally:
        print_input.unlink(missing_ok=True)
    rendered_pages = root / 'rendered_pages'
    rendered_pages.mkdir(exist_ok=True)
    for old in rendered_pages.glob('page-*.png'): old.unlink()
    subprocess.run(['pdftoppm', '-png', '-r', '90', str(root/'REPORT.pdf'),
                    str(rendered_pages/'page')], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    report_pages = sorted(rendered_pages.glob('page-*.png'))
    if not report_pages: raise ValueError('report PDF did not render')
    for page in report_pages:
        with Image.open(page) as image: image.load()
    parsed=_Links();parsed.feed(document)
    checked=[]
    for target in parsed.links+parsed.images:
        if target.startswith(('http://','https://','#','mailto:')): continue
        path=(root/target).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError(f'unresolved local report link: {target}')
        checked.append(target)
    for target in parsed.images:
        with Image.open(root/target) as image:
            image.verify()
        with Image.open(root/target) as image:
            image.load()
    pdfs=[p for p in parsed.links if p.lower().endswith('.pdf')]
    pages=0
    with tempfile.TemporaryDirectory() as temp:
        for index,target in enumerate(pdfs):
            prefix=str(Path(temp)/f'figure_{index}')
            subprocess.run(['pdftoppm','-png','-r','60',str(root/target),prefix],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            raster=sorted(Path(temp).glob(f'figure_{index}-*.png'))
            if not raster: raise ValueError(f'PDF did not render: {target}')
            for path in raster:
                with Image.open(path) as page: page.load()
                pages+=1
    verification={'html_rendered':True,'local_links_checked':len(checked),'png_images_decoded':len(parsed.images),'pdf_figures_rendered':len(pdfs),'pdf_pages_decoded':pages,'report_pdf_pages':[str(p.relative_to(root)) for p in report_pages],'human_visual_review':'not performed by automated validator'}
    (root/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    return verification


def build_report(output_dir, *, decision_path=None, assessment_paths=None, policy_paths=None,
                 histories=None, assessment_history=None, reconstructions=None, field_panels=None,
                 exposure_logs=None, cost_ledger=None, sss_figures=None, title='Round 8 comparison'):
    """Build report from actual saved artifacts, including optional evidence.

    ``assessment_paths`` and ``policy_paths`` map endpoint names to JSON files;
    ``histories`` maps arm names to training JSONL; ``assessment_history`` is a
    list of {endpoint, epoch, population, path, optional alpha} common score
    report references. ``reconstructions`` lists {source_panel, endpoint_panel,
    sample_id, field, alpha, optional member} against actual paired NPZ panels.
    ``field_panels`` are {endpoint, source_panel, endpoint_panel, alpha,
    optional regime_manifest or regime_by_id} cached-field inputs. ``exposure_logs`` maps arm to
    {path, field_names}; ``cost_ledger`` is a saved campaign ledger JSON.
    ``sss_figures`` supplies build_sss_figure inputs for frozen common observer descriptors. Missing inputs are labeled unavailable. The output directory must be empty.
    """
    output=Path(output_dir)
    if output.exists() and any(output.iterdir()): raise FileExistsError('report output directory must be empty')
    output.mkdir(parents=True,exist_ok=True); figures=output/'figures'
    assessment_paths=assessment_paths or {};policy_paths=policy_paths or {}; histories=histories or {}; assessment_history=assessment_history or []; reconstructions=reconstructions or []; field_panels=field_panels or []; exposure_logs=exposure_logs or {}; sss_figures=sss_figures or []
    decisions=None if decision_path is None else _json(decision_path)
    if decisions is not None and decisions.get('schema')!='round8_campaign_decision_v1': raise ValueError('invalid Round 8 campaign decision')
    assessments={k:_json(v) for k,v in assessment_paths.items()}
    policies={k:_json(v) for k,v in policy_paths.items()}
    for k,v in assessments.items():
        if v.get('schema')!='round8_assessment_v1' or v.get('endpoint')!=k: raise ValueError('assessment endpoint identity mismatch')
        if k in policies and v['policy_sha256']!=file_hash(policy_paths[k]): raise ValueError('assessment policy hash mismatch')
    for k,v in policies.items():
        if v.get('schema')!='round8_frozen_policy_v1' or v.get('endpoint')!=k: raise ValueError('invalid TRAIN policy')
    lines=[f'# {_escape(title)}','', 'The results below use the frozen TRAIN and ASSESS condition panels. TRAIN uses the declared Q4096 physical-weight proxy; ASSESS uses the native physical mask. Values are unavailable where an input artifact is absent.','']
    if decisions is None: lines+=['Operational and computation decisions: unavailable; no campaign decision artifact was supplied.','']
    else:
        operational=decisions.get('operational',{}); choice=operational.get('choice','unavailable')
        lines += [f"Operational choice: **{_escape(choice)}**. {_escape(decisions.get('selection_note',''))}",'']
        chosen=assessments.get(choice)
        if choice=='SOURCE':
            excluded=operational.get('exclusions',{})
            reason='; '.join(f'{k}: {v}' for k,v in excluded.items()) or 'no eligible endpoint recorded'
            lines += [f"The retained SOURCE J2 latency is {_num(operational.get('source_benchmark',{}).get('latency_seconds'))} seconds. Endpoint disposition: {_escape(reason)}.",'']
        if chosen and chosen.get('eligible'):
            gains=chosen.get('gain',{}); ratios=[v for v in chosen.get('risk_ratio',[]) if v is not None]
            benchmark=operational.get('benchmarks',{}).get(choice,{})
            lines += [f"The selected alpha is {_num(chosen.get('alpha'))}; ASSESS SOURCE-normalized gains are State {_num(gains.get('state'))}, Scale {_num(gains.get('scale'))}, and Shape {_num(gains.get('shape'))}. The largest defined pooled member-MSE ratio is {_num(max(ratios) if ratios else None)}; J2 inference latency is {_num(benchmark.get('latency_seconds'))} seconds.",'']
        for population,comp in decisions.get('computation_by_population',{}).items():
            choice='retain C8 as default' if comp['retain_c8_default'] else 'do not retain C8 as default'
            lines += [f"{_escape(population)} computation decision: **{choice}**; quality preserved: {comp['quality_preserved']}; qualifying quality gain: {comp['qualifying_quality_gain']}; C8/B8 amortized GPU time ratio: {_num(comp.get('cost_ratio'))}.",'']
    lines+=['## Fixed endpoint results','','| Endpoint | Representation | Family | TRAIN | ASSESS |','|---|---|---|---:|---:|']
    for name in ('T2000','B8','C8'):
        p=policies.get(name);a=assessments.get(name)
        if p is None and a is None: continue
        selected=None
        if p is not None and p.get('alpha') is not None:
            item=p.get('score_reports',{}).get(str(p['alpha']))
            if item:
                if file_hash(item['path'])!=item['sha256']: raise ValueError('TRAIN selected score report hash differs')
                selected=_mean_scores(_json(item['path']))
        raw_assess=None
        if a is not None and a.get('raw_report'):
            raw_report=_json(a['raw_report'])
            if raw_report.get('schema')!='round8_common_observers_v1' or raw_report.get('population')!='ASSESS' or raw_report.get('alpha')!=1.:
                raise ValueError('invalid ASSESS raw score report')
            raw_assess=_mean_scores(raw_report)
        for representation,train_scores,assess_scores in (
            ('SOURCE',None if p is None else p.get('train_source_scores'),None if a is None else a.get('source_scores')),
            ('Raw endpoint',None if p is None else p.get('train_raw_scores'),raw_assess),
            ('Selected blend',selected,None if a is None else a.get('blend_scores'))):
            for family in ('state','scale','shape'):
                lines.append(f"| {name} | {representation} | {family} | {_num(None if train_scores is None else train_scores.get(family))} | {_num(None if assess_scores is None else assess_scores.get(family))} |")
        lines.append(f"| {name} | fixed alpha / eligibility | - | {_num(None if p is None else p.get('alpha'))} | {'unavailable' if a is None else str(a.get('eligible'))} |")
    lines += ['', 'The family scores are the recorded common observers; lower scores are better under their frozen definitions. The two columns have different condition populations and spatial support.','']
    for name,p in policies.items():
        if p.get('risk_quadratic') and p.get('train_source_risk'):
            r0=np.asarray(p['train_source_risk'],dtype=float); b=np.asarray(p['risk_quadratic']['b'],dtype=float); q=np.asarray(p['risk_quadratic']['a'],dtype=float)
            names=assessments.get(name,{}).get('field_names',[str(i) for i in range(len(r0))])
            lines += [f'### {_escape(name)} TRAIN member risks','', '| Field | SOURCE MSE | Raw endpoint MSE | Selected blend MSE |','|---|---:|---:|---:|']
            for i,field in enumerate(names):
                blend=None if p.get('alpha') is None else r0[i]+2*p['alpha']*b[i]+p['alpha']**2*q[i]
                lines.append(f"| {_escape(field)} | {_num(r0[i])} | {_num(r0[i]+2*b[i]+q[i])} | {_num(blend)} |")
            lines += ['', 'TRAIN field risks are physical member MSE estimated on the fixed Q support; selected values use the saved quadratic.','']
    for name,a in assessments.items():
        lines += [f'### {_escape(name)} assessment','',f"Eligible: **{a.get('eligible',False)}**; qualifying frozen families: {_escape(', '.join(a.get('qualifying_families',[])) or 'none')}. {_escape(a.get('reason',''))}",'']
        if 'risk_ratio' in a:
            names=a.get('field_names',[str(i) for i in range(len(a['risk_ratio']))])
            lines += ['| Field | SOURCE member MSE | Blend member MSE | MSE ratio | RMSE change |','|---|---:|---:|---:|---:|']
            for field,source_risk,blend_risk,ratio,change in zip(names,a['source_risk'],a['blend_risk'],a['risk_ratio'],a['rmse_change']):
                lines.append(f"| {_escape(field)} | {_num(source_risk)} | {_num(blend_risk)} | {_num(ratio)} | {'unavailable' if change is None else format(100*change,'.2f')+'%'} |")
            lines.append('')
        for category,groups in a.get('groups',{}).items():
            lines += [f"{_escape(category)} subgroup diagnostics:",'', '| Group | n | State gain | Scale gain | Shape gain |','|---|---:|---:|---:|---:|']
            for group,values in groups.items():
                gains=values.get('score_gain',{})
                def display(k):
                    item=gains.get(k,{})
                    return _num(item.get('mean'))+' ± '+_num(item.get('condition_standard_error'))
                lines.append(f"| {_escape(group)} | {values['n']} | {display('state')} | {display('scale')} | {display('shape')} |")
            lines += ['', '| Group | Field | Member-risk change mean [frame SE] |','|---|---|---:|']
            for group,values in groups.items():
                for field,summary in values.get('field_risk_delta',{}).items():
                    lines.append(f"| {_escape(group)} | {_escape(field)} | {_num(summary.get('mean'))} [{_num(summary.get('condition_standard_error'))}] |")
            lines.append('')
        if a.get('uncertainty_note'): lines += [_escape(a['uncertainty_note']),'']
    if decisions is not None:
        lines+=['## Cost and additional claims','']
        for population,comp in decisions.get('computation_by_population',{}).items():
            for arm in ('b8','c8'):
                cost=comp.get(f'{arm}_cost',{})
                lines.append(f"{_escape(population)} {arm.upper()}: {_num(cost.get('amortized_seconds_per_update'))} GPU seconds per update; {_num(cost.get('campaign_gpu_seconds'))} campaign GPU seconds; {_num(cost.get('actual_updates'))} updates.")
            lines.append('')
        for endpoint,benchmark in operational.get('benchmarks',{}).items():
            lines += [f"{_escape(endpoint)} inference: {_num(benchmark.get('latency_seconds'))} seconds per J2 request; peak resident memory {_num(benchmark.get('peak_resident_bytes'))} bytes.",'']
        for endpoint,claim in decisions.get('strong_scale_claims',{}).items(): lines += [f"{_escape(endpoint)} strong Scale rule: {'passed' if claim['pass'] else 'not established'}; SOURCE gain {_num(claim.get('source_gain'))}, B8 gain {_num(claim.get('b8_gain'))} in frozen SOURCE units.",'']
    field_details=[cached_field_diagnostics(spec) for spec in field_panels]
    lines += ['## Physical spread, displacement and relationship means','']
    if not field_details:
        lines += ['Cached field diagnostics: unavailable; no matched field panels were supplied.','']
    for detail in field_details:
        lines += [f"### {_escape(detail['endpoint'])} {_escape(detail['population'])}, alpha={detail['alpha']:g}",'',f"Support: {detail['support']}. {detail['uncertainty']}",'', '| Field | Units | SOURCE J2 RMS spread | Blend J2 RMS spread | RMS SOURCE displacement | Ensemble mean error | Spread variance change [95% interval] |','|---|---|---:|---:|---:|---:|---:|']
        for field,row in detail['pooled']['fields'].items():
            spread=row['spread_variance_change']; bounds=spread['frame_bootstrap_95']
            interval='unavailable' if bounds is None else f"[{_num(bounds[0])}, {_num(bounds[1])}]"
            lines.append(f"| {_escape(field)} | {_escape(row['unit'])} | {_num(row['source_rms_spread'])} | {_num(row['blend_rms_spread'])} | {_num(row['rms_displacement'])} | {_num(row['ensemble_mean_error']['mean'])} | {_num(spread['mean'])} {interval} |")
        lines += ['', 'J2 spread is the square root of pooled unbiased two-member variance. SOURCE displacement is the square root of pooled member-average squared output change. The signed ensemble-mean error uses the same physical measure.','', 'Relationship-mean diagnostic: '+detail['relationship_definition']+'. Absolute errors are pooled across conditions; the paired change includes a descriptive frame-bootstrap 95% interval.','','| Field pair | Product units | SOURCE mean absolute error | Blend mean absolute error | Paired change [95% interval] |','|---|---|---:|---:|---:|']
        for pair,row in detail['pooled']['relationships'].items():
            change=row['paired_absolute_error_change']; interval=change['frame_bootstrap_95']
            ci='unavailable' if interval is None else f"[{_num(interval[0])}, {_num(interval[1])}]"
            lines.append(f"| {_escape(pair)} | {_escape(row['units'])} | {_num(row['source_absolute_error']['mean'])} | {_num(row['blend_absolute_error']['mean'])} | {_num(change['mean'])} {ci} |")
        lines.append('')
        for category,groups in detail['groups'].items():
            lines += [f"{_escape(category)} physical subgroup diagnostics:",'','| Group | n | Field | Blend J2 RMS spread | RMS SOURCE displacement |','|---|---:|---|---:|---:|']
            for group,values in groups.items():
                for field,row in values['fields'].items():
                    lines.append(f"| {_escape(group)} | {values['n']} | {_escape(field)} | {_num(row['blend_rms_spread'])} | {_num(row['rms_displacement'])} |")
            lines += ['', '| Group | Field pair | Paired absolute relationship-error change [95% interval] |','|---|---|---:|']
            for group,values in groups.items():
                for pair,row in values['relationships'].items():
                    change=row['paired_absolute_error_change']; interval=change['frame_bootstrap_95']
                    ci='unavailable' if interval is None else f"[{_num(interval[0])}, {_num(interval[1])}]"
                    lines.append(f"| {_escape(group)} | {_escape(pair)} | {_num(change['mean'])} {ci} |")
            lines.append('')
    lines += ['## Query exposure and additive GPU cost','']
    if not exposure_logs: lines += ['Query exposure: unavailable; no exposure logs were supplied.','']
    for arm,spec in exposure_logs.items():
        exposure=exposure_diagnostics(spec['path'],spec['field_names'])
        lines += [f"{_escape(arm)}: {exposure['batches']} sampled batches.",'', '| Query type | Unique queries | Population | Draws including repeats |','|---|---:|---:|---:|']
        for kind,row in exposure['queries'].items(): lines.append(f"| {kind} | {row['unique']} | {row['population']} | {row['draws']} |")
        lines += ['', '| Query | Draws |','|---|---:|']
        for kind,row in exposure['queries'].items():
            for query in row['per_query']: lines.append(f"| {kind}: {_escape(' / '.join(query['fields']))} | {query['draws']} |")
        lines.append('')
    if cost_ledger is None: lines += ['Additive GPU cost ledger: unavailable.','']
    else:
        ledger=cost_ledger_summary(cost_ledger)
        lines += [f"Completed occupied GPU time across devices: {_num(ledger['total_completed_gpu_seconds'])} GPU seconds; open intervals excluded: {ledger['open_intervals_excluded']}.",'', '| Recorded stage | Occupied GPU seconds |','|---|---:|']
        for stage,seconds in sorted(ledger['seconds_by_stage'].items()): lines.append(f"| {_escape(stage)} | {_num(seconds)} |")
        lines.append('')
    (output/'cached_field_diagnostics.json').write_text(json.dumps(field_details,indent=2,allow_nan=False)+'\n')
    entries=[]
    training=_training_figure(histories,figures) if histories else None
    if training: entries.append(('training_scores',training,'Sampled TRAIN family scores are plotted separately as raw and calibrated values. They are not a scalar ConFIG objective.'))
    for kind,paths in _assessment_figure(assessment_history,figures):
        entries.append((kind,paths,'Recorded common-observer scores on the indicated population and repair coefficient.' if kind=='assessments' else 'Recorded attraction, separation, or repulsion diagnostics; only available saved scalar entries are shown.'))
    sss_metadata=[]
    if not sss_figures: lines += ['Common State/Scale/Shape descriptor figure: unavailable; no matched scored condition and frozen observer inputs were supplied.','']
    if sss_figures:
        from .round8_sss_figure import build_sss_figure
        for i,spec in enumerate(sss_figures,1):
            stem=f'sss_{i:02d}'; metadata=build_sss_figure(spec,png_path=figures/(stem+'.png'),pdf_path=figures/(stem+'.pdf')); sss_metadata.append(metadata)
            caption=(f"Common frozen descriptors for {metadata['sample_id']} ({metadata['support']}), {metadata['field']}, alpha={metadata['alpha']:g}: weighted State marginal and fixed-direction quantiles, normalized spectral band-power descriptor, and {metadata['shape_unit']} H{metadata['dimension']} persistence diagrams. SOURCE and blend members are paired; these panels display descriptors, not physical field errors.")
            entries.append((stem,(f'figures/{stem}.png',f'figures/{stem}.pdf'),caption))
    (output/'sss_figure_provenance.json').write_text(json.dumps(sss_metadata,indent=2,allow_nan=False)+'\n')
    reconstruction_metadata=[]
    for i,spec in enumerate(reconstructions,1):
        paths,metadata=_reconstruction(spec,figures,i); reconstruction_metadata.append(metadata)
        support='native' if metadata['support']=='ASSESS' else 'sampled, self-normalized plotted support'
        entries.append((f'reconstruction_{i:02d}',paths,f"{metadata['field']} for {metadata['sample_id']}, member {metadata['member']}, α={metadata['alpha']:g}; weighted RMSE is computed from the displayed physical field on the {support}."))
    lines+=['## Figure index','']
    if not entries: lines+=['No figure input artifacts were supplied.','']
    for number,(name,(png,pdf),caption) in enumerate(entries,1):
        lines += [f'{number}. [{_escape(name)}]({png})']
    lines.append('')
    for name,(png,pdf),caption in entries:
        lines += [f'### {name.replace("_","-")}','',f'![{_escape(caption)}]({png})','',f'{caption} [PDF master]({pdf}).','']
    lines+=['## Evidence and limitations','', 'Fieldwise accuracy and the finite State, Scale and Shape observers are reported separately. The conditional scores do not certify complete physical validity or unseen-condition reliability.','']
    inputs={**{f'assessment:{k}':str(v) for k,v in assessment_paths.items()},**{f'policy:{k}':str(v) for k,v in policy_paths.items()},**{f'history:{k}':str(v) for k,v in histories.items()}}
    if decision_path is not None: inputs['decision']=str(decision_path)
    inputs.update({f"assessment_history:{i}":str(v['path']) for i,v in enumerate(assessment_history)})
    for i,v in enumerate(field_panels):
        inputs[f'field_source_panel:{i}']=str(v['source_panel'])
        inputs[f'field_endpoint_panel:{i}']=str(v['endpoint_panel'])
        if v.get('regime_manifest'): inputs[f'field_regime_manifest:{i}']=str(v['regime_manifest'])
    for arm,spec in exposure_logs.items(): inputs[f'exposure:{arm}']=str(spec['path'])
    if cost_ledger is not None: inputs['cost_ledger']=str(cost_ledger)
    for i,v in enumerate(sss_figures):
        for key in ('source_panel','endpoint_panel','score_report','fair_shape_artifact','native_bundle'):
            if v.get(key): inputs[f'sss_{i}:{key}']=str(v[key])
    for i,v in enumerate(reconstructions):
        inputs[f'reconstruction_source_panel:{i}']=str(v['source_panel'])
        inputs[f'reconstruction_endpoint_panel:{i}']=str(v['endpoint_panel'])
    for key,path in inputs.items(): lines.append(f'- {_escape(key)}: `{_escape(path)}`; SHA-256 `{file_hash(path)}`.')
    lines.append('')
    (output/'REPORT.md').write_text('\n'.join(lines))
    verification=validate_report(output)
    return {'markdown':str(output/'REPORT.md'),'html':str(output/'REPORT.html'),'pdf':str(output/'REPORT.pdf'),'verification':verification,'reconstructions':reconstruction_metadata,'sss_figures':sss_metadata}
