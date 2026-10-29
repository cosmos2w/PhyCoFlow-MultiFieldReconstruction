#!/usr/bin/env python3
"""Verify frozen inputs, cohort artifacts, report links and rendered preview."""
from pathlib import Path
from urllib.parse import unquote, urlsplit
import argparse, csv, hashlib, json, os, subprocess, sys
import numpy as np
from PIL import Image
from markdown_it import MarkdownIt

V4=Path(__file__).resolve().parents[1]; STUDY=V4.parent; ROOT=V4.parents[3]
MODELS=('deeponet','diffusion_pde','geofno','mimonet','senseiver','gl_rbf_cq')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def arrayhash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def count(path):
    with path.open(newline='') as f: return sum(1 for _ in csv.DictReader(f))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,default=V4/'REPORT.md');p.add_argument('--pdf',action='store_true',help='Also render PDF with PyMuPDF in the fig environment');p.add_argument('--inputs-only',action='store_true',help='Check frozen model and cohort identities without requiring report or metrics');args=p.parse_args()
    report=args.report.resolve()
    snapshot=json.loads((V4/'metadata/checkpoints.json').read_text()); old=json.loads((STUDY/'metadata/study_manifest.json').read_text())
    assert set(MODELS)<=set(snapshot['checkpoints'])
    assert digest(V4/'scripts/registry.json')==snapshot['registry_sha256']
    audit=json.loads((V4/'metadata/model_audit.json').read_text())
    assert set(audit['models'])==set(snapshot['checkpoints']), 'model audit does not cover registry'
    for name in snapshot['checkpoints']:
        rec=snapshot['checkpoints'][name]; m=json.loads((V4/f'metadata/{name}.json').read_text())
        assert audit['models'][name]['checkpoint_sha256']==rec['sha256'] and audit['models'][name]['config_sha256']==rec['resolved_config_sha256'],name
        assert m['checkpoint']==rec and digest(rec['checkpoint'])==rec['sha256'],name
        assert digest(Path(rec['run'])/'resolved_config.yaml')==rec['resolved_config_sha256'],name
        f=V4/f'data/{name}_test100.npz'
        if f.is_symlink():
            assert not os.path.isabs(os.readlink(f)) and f.resolve().is_file()
            assert m['source']=='verified previous study' and old['checkpoints'][name]['sha256']==rec['sha256']
        with np.load(f,allow_pickle=False) as d:
            a=d['fields']; assert a.shape==(100,100,403,5) and a.dtype==np.float32 and np.isfinite(a).all()
            assert arrayhash(a)==m['prediction_sha256'],name
    with np.load(STUDY/'data/gt_test100.npz',allow_pickle=False) as d:
        gt=d['fields']; ids=d['local_indices']; glob=d['global_indices']; oi=d['obs_indices']; ov=d['obs_values']; coord=d['coordinates']; offset=d['offset']; scale=d['scale']
    assert gt.shape==(100,100,403,5) and arrayhash(gt)==old['normalized_gt_sha256']
    assert ids.tolist()==old['local_indices'] and glob.tolist()==old['global_indices']
    assert oi.shape==(100,256) and ov.shape==(100,256) and coord.shape==(100,403,2)
    assert np.array_equal(offset,old['offset']) and np.array_equal(scale,old['scale'])
    # Rebuild all observations from the locked cohort and compare the saved inputs.
    os.environ.setdefault('KEOPS_CACHE_FOLDER','/tmp/premise_v4_keops'); os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
    import platform
    cache=Path(os.environ['KEOPS_CACHE_FOLDER'])/(platform.system()+'_'+platform.node()+'_'+platform.release()+'_p'+platform.python_version()+'_CUDA_VISIBLE_DEVICES_'+os.environ['CUDA_VISIBLE_DEVICES']);cache.mkdir(parents=True,exist_ok=True)
    import torch
    sys.path.insert(0,str(ROOT/'src'))
    from phycoflow_reconstruction.config import load_config
    from phycoflow_reconstruction.data.factory import open_field_dataset
    from phycoflow_reconstruction.data.manifest import dataset_fingerprint
    from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol,build_observation_batch
    c=load_config(Path(snapshot['checkpoints']['deeponet']['run'])/'resolved_config.yaml'); ds=open_field_dataset(c['dataset'],split='test')
    assert len(ds)==1000 and tuple(ds.grid_shape)==(100,403) and ds.normalizer.digest()==old['normalization_digest']
    assert dataset_fingerprint(ds.path)==old['dataset_fingerprint']
    for j,i in enumerate(ids):
        s=ds[int(i)]; assert s.time_index==int(glob[j]); assert np.array_equal(s.values.numpy().reshape(100,403,5),gt[j])
        b=build_observation_batch([s],SensorProtocol(name='random_uniform',field_counts={'T':256},seed=42+int(i)),query_points=None)
        assert np.array_equal(b.obs_indices[0].numpy(),oi[j]) and np.array_equal(b.obs_values[0,:,0].numpy(),ov[j])
        assert (b.obs_field_ids==2).all() and torch.equal(b.metadata['query_indices'][0],torch.arange(40300))
    ds.close()
    if args.inputs_only:
        print(f'Inputs verified: {len(snapshot["checkpoints"])} models and {len(ids)} frames')
        return
    assert report.is_file(),f'missing report {report}'
    metrics=V4/'data/metrics'; definitions=json.loads((metrics/'definitions.json').read_text())
    assert set(definitions['models'])==set(snapshot['checkpoints']) and definitions['shape']==[100,100,403,5]
    n=len(definitions['models']); expected_frames=100*10*n*5; expected_pairs=100*10*n*(n-1)//2
    assert definitions['expected_frame_rows']==expected_frames and count(metrics/'frame_metrics.csv')==expected_frames
    assert definitions['expected_candidate_rows']==expected_pairs and count(metrics/'candidates.csv')==expected_pairs
    with (metrics/'frame_metrics.csv').open(newline='') as f:
        assert {r['model'] for r in csv.DictReader(f)}==set(snapshot['checkpoints'])
    assert count(metrics/'cohort.csv')==160
    for name,item in definitions['inputs'].items(): assert digest(item['path'])==item['sha256'],name
    eventdoc=json.loads((V4/'metadata/events_summary.json').read_text())
    assert set(eventdoc['models_available'])==set(snapshot['checkpoints']) and not eventdoc['missing']
    expected_events=100*4*n; expected_event_pairs=100*4*n*(n-1)//2
    assert count(V4/'data/events_metrics.csv')==expected_events and count(V4/'data/events_pairs.csv')==expected_event_pairs
    with (V4/'data/events_metrics.csv').open(newline='') as f:
        assert {r['model'] for r in csv.DictReader(f)}==set(snapshot['checkpoints'])
    md=MarkdownIt('commonmark',{'html':False}).enable('table'); tokens=md.parse(report.read_text()); links=[]; images=[]
    for block in tokens:
        for t in block.children or []:
            if t.type=='image': images.append(t.attrGet('src'))
            elif t.type=='link_open': links.append(t.attrGet('href'))
    for url in images+links:
        if not url or url.startswith('#') or urlsplit(url).scheme or url.startswith('//'): continue
        path=(report.parent/unquote(urlsplit(url).path)).resolve(); assert path.exists() or path==V4/'metadata/verification.json',f'broken link: {url}'
    for url in images:
        path=(report.parent/unquote(urlsplit(url).path)).resolve(); assert path.suffix.lower()=='.png',f'not PNG: {url}'
        with Image.open(path) as im: im.verify()
        assert path.with_suffix('.pdf').is_file(),f'missing PDF master: {url}'
        assert any((report.parent/unquote(urlsplit(link).path)).resolve()==path.with_suffix('.pdf') for link in links if link and not urlsplit(link).scheme and not link.startswith('#')),f'PDF master not linked: {url}'
    html='<!doctype html><html><head><meta charset="utf-8"><style>body{font:16px/1.55 sans-serif;max-width:1100px;margin:2em auto;padding:0 1em}img{max-width:100%;height:auto}table{border-collapse:collapse;max-width:100%;table-layout:fixed}td,th{border:1px solid #bbb;padding:.35em;overflow-wrap:anywhere}pre{overflow:auto} @media print{body{font-size:10pt} img{max-height:85vh;object-fit:contain} h2,h3{break-after:avoid}}</style></head><body>'+md.render(report.read_text())+'</body></html>'
    htmlpath=report.with_suffix('.preview.html'); htmlpath.write_text(html)
    if args.pdf:
        subprocess.run(['/home/wanglz/miniconda3/envs/fig/bin/python',str(V4/'scripts/render_pdf.py'),str(htmlpath),str(report.with_suffix('.pdf'))],check=True)
    result={'models':list(snapshot['checkpoints']),'frames':100,'observation_checks':100,'frame_metric_rows':expected_frames,'candidate_rows':expected_pairs,'event_rows':expected_events,'event_pair_rows':expected_event_pairs,'images':len(images),'local_links_checked':len(links),'html':str(htmlpath),'pdf_rendered':args.pdf}
    (V4/'metadata/verification.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__':main()
