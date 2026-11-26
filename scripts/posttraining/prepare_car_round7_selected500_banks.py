#!/usr/bin/env python3
"""Prepare four matched epoch-500 bank commands from a completed selected run.

This CPU-only adapter writes a new assessment attempt and never launches inference.
Run the resulting bank lease only after ownership and budget checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
ARM = 'FFM_Round7_Joint_Anchor_Scale'


def ref(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def canonical_checkpoint(result):
    from phycoflow_reconstruction.training.run_store import load_project_checkpoint
    from scripts.posttraining.qualify_car_round2_resume import differences
    run = Path(result['run_dir']).resolve(strict=True)
    last = run/'checkpoints/last.pt'
    if Path(result['checkpoint']['path']).resolve(strict=True) != last.resolve(strict=True) or ref(last) != result['checkpoint']:
        raise ValueError('terminal last checkpoint differs from result')
    canonical = ref(run/'checkpoints/epoch_500.pt')
    last_payload = load_project_checkpoint(last)
    canonical_payload = load_project_checkpoint(canonical['path'])
    if last_payload.get('global_step') != 6000 or canonical_payload.get('global_step') != 6000:
        raise ValueError('both endpoint checkpoints must have global step 6000')
    delta = differences(last_payload, canonical_payload, atol=0, rtol=0)
    if delta: raise ValueError('canonical epoch-500 checkpoint full payload differs: '+str(delta[:5]))
    parity = {'last':ref(last),'canonical':canonical,'global_step':6000,
              'comparison':'full loaded checkpoint payload; no exclusions',
              'comparison_atol':0,'comparison_rtol':0,'differences':delta}
    return canonical, parity


def prepare(campaign, result_path, attempt, device):
    campaign = campaign.resolve(strict=True)
    result_path = result_path.resolve(strict=True)
    result = json.loads(result_path.read_text())
    if (result.get('schema') != 'car_round7_training_result_v1' or result.get('arm') != ARM or
        result.get('global_step') != 6000 or result.get('target_epoch') != 500 or
        result.get('target_reached') is not True or result.get('disposition') != 'target_reached'):
        raise ValueError('completed selected joint-anchor epoch-500 training receipt required')
    freeze = json.loads((campaign/'configs/formal_freeze.json').read_text())
    prepared = json.loads((campaign/'configs/prepared.json').read_text())
    code = Path(freeze['code_root']).resolve(strict=True)
    if ref(campaign/'configs/prepared.json')['sha256'] != freeze['prepared_sha256']:
        raise ValueError('prepared configuration does not match frozen receipt')
    for key in ('checkpoint', 'saved_config'):
        if ref(result[key]['path']) != result[key]: raise ValueError(key + ' changed after training')
    if ref(prepared['reference']['path']) != prepared['reference']: raise ValueError('training reference changed')
    checkpoint, parity = canonical_checkpoint(result)
    root = attempt.resolve()/ARM
    root.mkdir(parents=True, exist_ok=False)
    with (root/'training_result.json').open('xb') as stream: stream.write(result_path.read_bytes())
    specs = {}
    for split in ('train','validation'):
        output = root/(split+'_spec.json')
        subprocess.run([sys.executable, str(code/'scripts/posttraining/prepare_car_round7_assessment.py'),
            '--campaign',str(campaign),'--config',result['saved_config']['path'],
            '--reference',prepared['reference']['path'],'--split',split,
            '--checkpoint',checkpoint['path'],'--reporting-epoch','500','--output',str(output)],check=True)
        specs[split] = ref(output)
    plan = {'schema':'car_round7_four_banks_v1','arm':ARM,'checkpoint':checkpoint,
            'training_result':ref(result_path),'checkpoint_payload_parity':parity,'specs':specs,'commands':[]}
    for split in ('train','validation'):
        for support in ('native','selected'):
            output = root/(split+'_'+support+'_bank')
            command = [sys.executable,str(code/'scripts/posttraining/assess_car_round7.py'),'generate',
                       '--spec',specs[split]['path'],'--output',str(output),'--support',support,'--device',device]
            plan['commands'].append({'split':split,'support':support,'output':str(output),'command':command})
    path=root/'bank_commands.json'
    with path.open('x') as stream: json.dump(plan,stream,indent=2,allow_nan=False);stream.write('\n')
    print(path)


def finalize(attempt):
    root = attempt.resolve(strict=True)/ARM
    plan = json.loads((root/'bank_commands.json').read_text())
    progress = json.loads((root/'bank_progress.json').read_text())
    local_result = root/'training_result.json'
    result = json.loads(local_result.read_text())
    checkpoint, parity = canonical_checkpoint(result)
    if (plan['checkpoint'] != checkpoint or plan['checkpoint_payload_parity'] != parity or ref(local_result)['sha256'] != plan['training_result']['sha256'] or
        progress.get('status') != 'completed' or progress.get('arm') != ARM or len(progress['steps']) != 4):
        raise ValueError('completed matching four-bank run required')
    banks=[]; identities=set()
    for item in progress['steps']:
        binding=item['bank_receipt']
        if ref(binding['path']) != binding: raise ValueError('bank receipt changed')
        bank=json.loads(Path(binding['path']).read_text())
        split,support=bank['split'],bank['support']
        identities.add((split,support))
        if (bank.get('status')!='completed' or bank.get('checkpoint_sha256')!=checkpoint['sha256'] or
            bank.get('spec_sha256')!=plan['specs'][split]['sha256']):
            raise ValueError('bank does not match canonical checkpoint and split spec')
        banks.append(binding)
    if identities != {(split,support) for split in ('train','validation') for support in ('native','selected')}:
        raise ValueError('four distinct split/support banks required')
    for binding in plan['specs'].values():
        if ref(binding['path']) != binding: raise ValueError('assessment spec changed')
    ready={'schema':'car_round7_review_arm_ready_v1','arm':ARM,'checkpoint':checkpoint,'specs':plan['specs'],'banks':banks}
    with (root/'ready.json').open('x') as stream: json.dump(ready,stream,indent=2,allow_nan=False);stream.write('\n')
    print(root/'ready.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--training-result',type=Path,required=True)
    parser.add_argument('--attempt',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--finalize',action='store_true',help='verify completed bank progress and write ready.json')
    args=parser.parse_args()
    if args.finalize: finalize(args.attempt)
    else: prepare(args.campaign,args.training_result,args.attempt,args.device)


if __name__=='__main__': main()
