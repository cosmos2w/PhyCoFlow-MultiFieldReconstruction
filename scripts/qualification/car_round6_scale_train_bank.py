#!/usr/bin/env python3
"""Replay a bound native TRAIN800 SOURCE bank at a Round 6 checkpoint."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import time

from scripts.posttraining.assess_car_round2_checkpoints import hash_file, matched_rows

ARM = 'FFM_SSS_EpochCalibrated_PI_Repair'
SOURCE_SHA = '2d1da713538f6e9613560f3a7eb4a9df2a95d1ad1f56dd11a1a8a4f9ef65050c'

def validate_source(source):
    if (source.get('schema') != 'car_round2_native_bank_v1' or source.get('status') != 'completed'
        or source.get('split') != 'train' or source.get('population') != 8000
        or source.get('members') != 4 or source.get('native_points') != 40300
        or source.get('source_weight_view') != 'ema_materialized'
        or source.get('checkpoint_sha256') != SOURCE_SHA
        or len(source.get('sample_ids', [])) != 800 or len(set(source['sample_ids'])) != 800
        or len(source.get('files_sha256', [])) != 800):
        raise ValueError('expected completed native SOURCE TRAIN800 J4 bank')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--epoch', type=int, choices=(100, 500), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--preflight', action='store_true')
    args = p.parse_args()
    import torch
    import yaml
    from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol, build_observation_batch
    from phycoflow_reconstruction.training.model_lifecycle import load_training_aux_state
    from phycoflow_reconstruction.training.rollout import differentiable_reconstruction
    from phycoflow_reconstruction.training.run_store import load_model_state_strict, load_project_checkpoint
    from phycoflow_reconstruction.training.source import load_source_model
    from phycoflow_reconstruction.training.scalable_fitting import _digest

    campaign = args.campaign.resolve(strict=True)
    round6 = (campaign.parent/'round_6').resolve(strict=True)
    source_dir = campaign.parent/'round_2/assessment/source_train_ffm_v1'
    source_receipt = source_dir/'receipt.json'
    source = json.loads(source_receipt.read_text()); validate_source(source)
    freeze_path = round6/'configs/formal_freeze.json'; freeze = json.loads(freeze_path.read_text())
    code = Path(freeze['code_root']).resolve(strict=True)
    inventory = json.loads((code/'execution_receipt.json').read_text())
    if code.name != freeze['code_sha256'] or inventory['sha256'] != freeze['code_sha256']:
        raise ValueError('frozen scientific code identity mismatch')
    for name, expected in inventory['files'].items():
        if hash_file(code/name) != expected: raise ValueError('frozen code changed: '+name)
    if Path(__import__('phycoflow_reconstruction').__file__).resolve().is_relative_to(code) is False:
        raise ValueError('scientific imports must use frozen code')
    snapshot = json.loads((round6/'assessment/Epoch500_Validation100_J4_20261125T042223Z/snapshot/snapshot.json').read_text())
    run = Path(snapshot['run']).resolve(strict=True)
    if (round6/'runs/formal'/ARM).resolve() not in run.parents: raise ValueError('wrong formal run')
    config_path = run/'resolved_config.yaml'; cfg = yaml.safe_load(config_path.read_text())
    if hash_file(config_path) != freeze['arms'][ARM]['sha256']: raise ValueError('saved config changed')
    checkpoint = run/'checkpoints'/f'epoch_{args.epoch:03d}.pt'
    validation = round6/('assessment/Epoch100_Validation100_J4_20261124/bank/receipt.json' if args.epoch == 100 else 'assessment/Epoch500_Validation100_J4_20261125T042223Z/bank/receipt.json')
    val = json.loads(validation.read_text()); checkpoint_sha = hash_file(checkpoint)
    if val['checkpoint_sha256'] != checkpoint_sha: raise ValueError('checkpoint differs from completed assessment')
    state = load_project_checkpoint(checkpoint)
    if state.get('global_step') != args.epoch*12 or state.get('source_hashes',{}).get('checkpoint') != SOURCE_SHA:
        raise ValueError('checkpoint step or source differs')
    output = args.output.resolve()
    if (campaign/'assessment').resolve() not in output.parents:
        raise ValueError('output must be in followup assessment')
    if output.exists(): raise FileExistsError(output)
    # Check the first and last source rows without generating any predictions.
    for i in (0, 799):
        path = source_dir/f'{i:04d}.pt'
        if hash_file(path) != source['files_sha256'][i]: raise ValueError('SOURCE row hash mismatch')
        row = torch.load(path, map_location='cpu', weights_only=True)
        if row['sample_id'] != source['sample_ids'][i] or len(row['member_seeds']) != 4: raise ValueError('SOURCE row identity mismatch')
    if args.preflight:
        print(json.dumps({'status':'preflight_passed','epoch':args.epoch,'source_receipt_sha256':hash_file(source_receipt),'checkpoint_sha256':checkpoint_sha,'code_sha256':freeze['code_sha256'],'samples':800,'members':4,'test_locked':True})); return
    token = os.environ.get('CAR_ROUND6_LEASE_TOKEN')
    if not token: raise ValueError('enclosing campaign assessment lease required')
    output.mkdir(parents=True, exist_ok=False)
    start=time.time(); cpu=time.process_time()
    receipt={'schema':'car_round6_scale_alignment_train_bank_v1','status':'failed','split':'train','population':8000,'epoch':args.epoch,'members':4,'sample_ids':source['sample_ids'],'source_bank_sha256':hash_file(source_receipt),'panel_sha256':source['panel_sha256'],'field_order':source['field_order'],'native_points':source['native_points'],'normalizer_sha256':source['normalizer_sha256'],'source_checkpoint_sha256':SOURCE_SHA,'source_view':'ema_materialized','child_view':'raw_checkpoint_model','checkpoint_sha256':checkpoint_sha,'config_sha256':hash_file(config_path),'frozen_code_sha256':freeze['code_sha256'],'driver_sha256':hash_file(Path(__file__)),'test_locked':True,'campaign':str(campaign),'round6_source_campaign':str(round6),'lease_token':token,'start_unix':start,'device':str(torch.cuda.get_device_name(0)),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
    dataset=None
    try:
        device=torch.device('cuda:0'); model,dataset,_=load_source_model(cfg,device)
        if len(dataset)!=8000 or dataset.normalizer.digest()!=source['normalizer_sha256'] or _digest(model.state_dict())!=source['source_parameter_buffer_sha256']:
            raise ValueError('dataset, normalizer or materialized SOURCE differs')
        load_model_state_strict(model,state['model']); load_training_aux_state(model,state); model.eval()
        protocol=SensorProtocol(field_counts={'T':256},seed=2047); files=[]
        with torch.no_grad():
            for ordinal, sample_id in enumerate(source['sample_ids']):
                src=source_dir/f'{ordinal:04d}.pt'
                if hash_file(src)!=source['files_sha256'][ordinal]: raise ValueError('SOURCE row hash mismatch')
                row=torch.load(src,map_location='cpu',weights_only=True); index=int(row['sample_index']); sample=dataset[index]
                actual=f'{dataset.trajectory_ids[dataset._items[index][0]]}:{dataset._items[index][1]}'
                if actual!=sample_id or row['sample_id']!=sample_id or sample.values.shape[0]!=40300: raise ValueError('sample/support identity mismatch')
                obs={sample_id:list(zip(row['obs_indices'][0].tolist(),row['obs_fields'][0].tolist(),strict=True))}
                batch=build_observation_batch([sample],protocol,query_indices=None,manifest_indices=obs).to(device)
                members=[differentiable_reconstruction(model,replace(batch,target_fields=None),steps=int(cfg['rollout']['steps']),solver=cfg['rollout']['solver'],generator=torch.Generator(device=device).manual_seed(seed),observation_config=cfg['observation_consistency']).detach().cpu() for seed in row['member_seeds']]
                child=dict(sample_id=sample_id,sample_index=index,members_model=torch.stack(members),truth_model=batch.target_fields.detach().cpu(),coordinates_model=batch.query_coords.detach().cpu(),coordinates_raw=sample.coordinates_raw.cpu(),native_masses=dataset.physical_weights.cpu(),obs_indices=batch.obs_indices.detach().cpu(),obs_values=batch.obs_values.detach().cpu(),obs_fields=batch.obs_field_ids.detach().cpu(),member_seeds=row['member_seeds'],normalizer_offset=dataset.normalizer.offset.cpu(),normalizer_scale=dataset.normalizer.scale.cpu())
                matched_rows(row,child)
                path=output/f'{ordinal:04d}.pt';torch.save(child,path);files.append(hash_file(path))
        receipt.update(status='completed',files_sha256=files,units='model-field values; normalizer and native physical masses retained')
    except BaseException as exc:
        receipt.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:
        if dataset is not None: dataset.close()
        receipt.update(end_unix=time.time(),wall_seconds=time.time()-start,cpu_seconds=time.process_time()-cpu)
        with (output/'receipt.json').open('x') as f: json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
if __name__=='__main__':main()
