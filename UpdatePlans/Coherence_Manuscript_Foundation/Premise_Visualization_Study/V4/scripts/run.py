#!/usr/bin/env python3
"""Run registry-driven CPU analysis and reporting; inference is a separate step."""
from pathlib import Path
import argparse, json, subprocess, sys

V4=Path(__file__).resolve().parents[1]
SCRIPTS=V4/'scripts'

def call(script,*args):
    command=[sys.executable,str(SCRIPTS/script),*map(str,args)]
    print('+',' '.join(command),flush=True)
    subprocess.run(command,check=True)

def update_audit(names):
    path=V4/'metadata/model_audit.json'
    audit=json.loads(path.read_text())
    for name in names:
        if name in audit['models']: continue
        m=json.loads((V4/f'metadata/{name}.json').read_text()); c=m['checkpoint']
        audit['models'][name]={
            'source':'src/phycoflow_reconstruction/models/__init__.py',
            'architecture':'See native_model_config in model metadata; individual source review is pending.',
            'observation_handling':'Not yet individually audited; inspect the adapter before making claims.',
            'epoch':m['epoch'],'global_step':m['global_step'],'evaluation_weights':m['evaluation_weights'],
            'generation_steps':m['native_evaluation_config']['generation_steps'],
            'checkpoint_sha256':c['sha256'],'config_sha256':c['resolved_config_sha256'],
            'strict_load_evidence':m.get('strict_state_loaded',m.get('checks',{}).get('strict_state_loaded')),
            'review_status':'individual source audit pending'}
    path.write_text(json.dumps(audit,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analyze',action='store_true',help='recompute all metrics and event catalog')
    p.add_argument('--redraw',action='store_true',help='redraw all figures from retained data')
    p.add_argument('--verify',action='store_true',help='verify inputs, report and render HTML/PDF')
    p.add_argument('--all',action='store_true',help='analyze, redraw, rebuild report and verify')
    args=p.parse_args()
    if not any(vars(args).values()): p.error('choose --analyze, --redraw, --verify or --all')
    names=list(json.loads((SCRIPTS/'registry.json').read_text()))
    for name in names:
        if not (V4/f'data/{name}_test100.npz').exists() or not (V4/f'metadata/{name}.json').exists():
            raise SystemExit(f'Missing {name} endpoint or metadata: extract it first with inference.py')
    model_args=[item for name in names for item in ('--model',f'{name}={V4}/data/{name}_test100.npz')]
    if args.analyze or args.all:
        update_audit(names)
        call('metrics.py',*model_args)
        call('coupling_events.py')
    if args.redraw or args.all:
        call('figures.py',*model_args)
        call('coupling_events.py','--redraw')
        call('controls.py','--redraw')
        call('paused_loss.py')
    if args.all:
        call('build_report.py')
        print('Review report claims and any pending individual model source audits before use.',flush=True)
    if args.verify or args.all:
        call('verify.py','--pdf')
if __name__=='__main__':main()
