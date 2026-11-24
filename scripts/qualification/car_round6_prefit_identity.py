#!/usr/bin/env python3
"""Reconcile the mature checkpoint, saved prefit and recorded scientific scales."""
import argparse,hashlib,json
from pathlib import Path
import torch,yaml

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 cp=a.run/'checkpoints/epoch_100.pt';config=a.run/'resolved_config.yaml'
 c=torch.load(cp,map_location='cpu',weights_only=False);cfg=yaml.safe_load(config.read_text())
 pref=cfg['coherence']['scalable_anchored_coherence']['common_prefit'];raw=Path(pref['path']).read_bytes();digest=hashlib.sha256(raw).hexdigest();saved=c['scalable_runtime']['calibration']['artifact'];artifact=json.loads(raw)
 if digest!=pref['sha256'] or saved!=artifact or c['global_step']!=1200:raise ValueError('prefit or mature checkpoint mismatch')
 out={'status':'completed','prefit_sha256':digest,'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),
      'prefit_matches_saved_checkpoint_artifact':True,'global_step':c['global_step'],'runtime_step':c['scalable_runtime']['step'],
      'family_scales':c['family_scales'],'source_objective_scale':c['external_policy']['reference']['objective_scale'],
      'field_order':c['external_policy']['reference']['field_names'],'configured_lr':cfg['optimization']['lr'],
      'saved_lr':c['optimizer']['param_groups'][0]['lr'],'saved_weight_decay':c['optimizer']['param_groups'][0]['weight_decay'],
      'saved_betas':c['optimizer']['param_groups'][0]['betas'],'saved_epsilon':c['optimizer']['param_groups'][0]['eps'],
      'configured_grad_clip':cfg['optimization']['grad_clip'],'saved_multipliers':c['external_policy']['multipliers']}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as stream:json.dump(out,stream,indent=2,allow_nan=False);stream.write('\n')
if __name__=='__main__':main()
