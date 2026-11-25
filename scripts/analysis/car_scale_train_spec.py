#!/usr/bin/env python3
"""Bind completed matched TRAIN800 banks to the existing Scale decomposition spec."""
import argparse,hashlib,json
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ref(path):
 p=Path(path).resolve(strict=True);return {'path':str(p),'sha256':sha(p)}
def read(path):return json.loads(Path(path).read_text())

def main():
 p=argparse.ArgumentParser();p.add_argument('--validation-spec',type=Path,required=True);p.add_argument('--epoch100-receipt',type=Path,required=True);p.add_argument('--epoch500-receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 spec=read(a.validation_spec);campaign=Path(spec['campaign']).resolve(strict=True)
 if campaign/'configs' not in a.output.resolve().parents or a.output.exists():raise ValueError('new campaign config required')
 root=campaign.parent;source_path=root/'round_2/assessment/source_train_ffm_v1/receipt.json';panel_path=root/'round_2/configs/panels/train.json';source=read(source_path);panel=read(panel_path);source_ref=ref(source_path);panel_ref=ref(panel_path)
 if (source.get('schema')!='car_round2_native_bank_v1' or source.get('status')!='completed' or source.get('split')!='train' or source.get('population')!=8000 or source.get('members')!=4 or len(source.get('sample_ids',[]))!=800 or source.get('panel_sha256')!=panel_ref['sha256'] or source['sample_ids']!=panel['sample_ids'] or len(panel['query_indices'])!=4096):raise ValueError('TRAIN SOURCE panel not matched')
 children={}
 for name,path,epoch in (('epoch100',a.epoch100_receipt,100),('epoch500',a.epoch500_receipt,500)):
  receipt=read(path);validation=read(spec['children'][name]['receipt']['path'])
  if (receipt.get('schema')!='car_round6_scale_alignment_train_bank_v1' or receipt.get('status')!='completed' or receipt.get('split')!='train' or receipt.get('epoch')!=epoch or receipt.get('members')!=4 or receipt.get('sample_ids')!=source['sample_ids'] or receipt.get('source_bank_sha256')!=source_ref['sha256'] or receipt.get('panel_sha256')!=panel_ref['sha256'] or receipt.get('source_checkpoint_sha256')!=source['source_sha256'] or receipt.get('checkpoint_sha256')!=validation['checkpoint_sha256'] or receipt.get('field_order')!=source['field_order'] or receipt.get('normalizer_sha256')!=source['normalizer_sha256'] or len(receipt.get('files_sha256',[]))!=800):raise ValueError('incompatible completed TRAIN child '+name)
  children[name]={'receipt':ref(path)}
 spec.update(split='train',source_receipt=source_ref,panel=panel_ref,children=children)
 with a.output.open('x') as f:json.dump(spec,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'status':'completed','spec':ref(a.output),'source':source_ref,'samples':800,'members':4}))
if __name__=='__main__':main()
