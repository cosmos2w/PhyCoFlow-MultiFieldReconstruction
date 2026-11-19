#!/usr/bin/env python3
"""Fixed raw-output early closeout using the existing native assessment functions.

`freeze` binds the three literal checkpoints and fits SOURCE task units from the
already-saved TRAIN bank. `generate` and `score` require that freeze. No output
amplitude or checkpoint is chosen using VALIDATION results.
"""
import time

START = time.time()

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get('CAR_REPO_ROOT', Path(__file__).resolve().parents[2])).resolve()
CODE = Path(os.environ.get('CAR_CODE_ROOT', ROOT)).resolve()
sys.path.insert(0, str(CODE/'src'))
sys.path.insert(0, str(CODE))
ARMS = {'FFM_Selected_Bank_Uniform_FullModel': 750,
        'Senseiver_Selected_Bank_Uniform_FullModel': 1000,
        'Senseiver_Selected_Bank_Residual_Repair': 1000}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def bind(campaign, path, expected=None):
    path = Path(path).resolve(strict=True)
    if campaign not in path.parents:
        raise ValueError(f'artifact outside campaign: {path}')
    actual = sha(path)
    if expected is not None and actual != expected:
        raise ValueError(f'artifact changed: {path}')
    return {'path':str(path), 'sha256':actual}


def load_assessment():
    path = CODE/'scripts/posttraining/assess_car_round2_checkpoints.py'
    spec = importlib.util.spec_from_file_location('round2_original_assessment', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze(args, campaign, base):
    import torch
    import yaml

    from phycoflow_reconstruction.evaluation.round2_assessment import scalar_task_score
    from phycoflow_reconstruction.evaluation.round2_task import regional_covariance
    from phycoflow_reconstruction.training.run_store import load_project_checkpoint

    plan_path = Path(args.plan).resolve(strict=True)
    if campaign not in plan_path.parents:
        raise ValueError('plan must be campaign-owned')
    plan = json.loads(plan_path.read_text())
    if plan.get('schema') != 'car_round2_raw_closeout_plan_v1' or set(plan.get('arms', {})) != set(ARMS):
        raise ValueError('explicit three-arm raw closeout plan required')
    output = Path(args.output).resolve()
    if campaign/'assessment/early_closeout' not in output.parents:
        raise ValueError('freeze output must be in early-closeout assessment storage')
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    formal_path = campaign/'configs/formal_freeze.json'
    formal = json.loads(formal_path.read_text())
    if formal.get('schema') != 'car_round2_freeze_receipt_v1' or formal['manifest']['status'] != 'frozen':
        raise ValueError('existing formal scientific freeze required')
    manifest = formal['manifest']
    panel_path = campaign/'configs/panels/validation.json'
    panel = json.loads(panel_path.read_text())
    ids = base.panel_contract(panel, 'validation')
    if len(ids) != 100 or panel['members'] != {'ffm':4, 'senseiver':1}:
        raise ValueError('matched VALIDATION100 and genuine J4/J1 required')
    selections = {}
    for arm, epoch in ARMS.items():
        item = plan['arms'][arm]
        if item.get('epoch') != epoch or item.get('alpha') != 1:
            raise ValueError(f'{arm}: fixed raw checkpoint and alpha required')
        run_value = Path(item['run'])
        run = (run_value if run_value.is_absolute() else campaign/run_value).resolve(strict=True)
        if (campaign/'runs/formal'/arm).resolve() not in run.parents:
            raise ValueError('checkpoint run is not the named formal arm')
        checkpoint = run/'checkpoints'/f'epoch_{epoch:03d}.pt'
        checkpoint_ref = bind(campaign, checkpoint, item['checkpoint_sha256'])
        state = load_project_checkpoint(checkpoint)
        if state.get('global_step') != 12*epoch:
            raise ValueError('checkpoint optimizer age differs')
        config = yaml.safe_load((run/'resolved_config.yaml').read_text())
        config_ref = manifest['arms'][arm]['config']
        bind(campaign, config_ref['path'], config_ref['sha256'])
        if not base.same_scientific_config(config, yaml.safe_load(Path(config_ref['path']).read_text())):
            raise ValueError('run configuration differs from frozen arm')
        selections[arm] = {'epoch':epoch, 'alpha':1., 'run':str(run),
                           'checkpoint':checkpoint_ref, 'config':config_ref}
    task_ref = manifest['artifacts']['reserved_task']
    task = json.loads(Path(bind(campaign, task_ref['path'], task_ref['sha256'])['path']).read_text())
    if task.get('schema') != 'car_round2_reserved_task_v1':
        raise ValueError('fixed reserved task required')
    train_panel = campaign/'configs/panels/train.json'
    train_ids = base.panel_contract(json.loads(train_panel.read_text()), 'train')
    if len(train_ids) != 800:
        raise ValueError('TRAIN800 source task normalization required')
    selection = {'schema':'car_round2_raw_closeout_selection_v1', 'plan':bind(campaign, plan_path),
                 'formal_freeze':bind(campaign, formal_path), 'validation_panel':bind(campaign, panel_path),
                 'train_panel':bind(campaign, train_panel), 'reserved_task':task_ref,
                 'event':manifest['artifacts']['development_event'], 'arms':selections,
                 'output_policy':'SOURCE and fixed raw child alpha=1; no VALIDATION search',
                 'rejected_ffm_residual':'TRAIN event continuity failure; no held-out rescue',
                 'created_unix':time.time()}
    selection_path = output/'selection.json'
    write(selection_path, selection)
    units, artifacts = {}, {}
    setup = json.loads((campaign/'configs/topology_stable_setup_v1/setup_references.json').read_text())
    for family in ('ffm','senseiver'):
        ref = manifest['artifacts'][f'assessment_contract_{family}']
        contract = json.loads(Path(bind(campaign, ref['path'], ref['sha256'])['path']).read_text())
        if contract.get('schema') != 'car_round2_assessment_contract_v2' or contract.get('reserved_task_sha256') != task_ref['sha256']:
            raise ValueError('source assessment contract or reserved task differs')
        bank = campaign/f'assessment/source_train_{family}_v1'
        receipt_path = bank/'receipt.json'
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get('status') != 'completed' or receipt.get('kind') != 'source' or receipt.get('split') != 'train'
                or receipt.get('family') != family or receipt.get('sample_ids') != train_ids
                or sha(receipt_path) != contract['source_bank_sha256'] or receipt['panel_sha256'] != sha(train_panel)):
            raise ValueError('fixed TRAIN source bank differs from assessment contract')
        values=[]
        for i,sample_id in enumerate(train_ids):
            path=bank/f'{i:04d}.pt'
            if sha(path)!=receipt['files_sha256'][i]: raise ValueError('TRAIN SOURCE row changed')
            row=torch.load(path,map_location='cpu',weights_only=True)
            if row['sample_id']!=sample_id: raise ValueError('TRAIN SOURCE IDs differ')
            mass=row['native_masses'].double(); scale=row['normalizer_scale']; offset=row['normalizer_offset']
            members=row['members_model']*scale+offset; truth=row['truth_model']*scale+offset
            member_values=torch.stack([regional_covariance(member,mass,task,tuple(receipt['field_order'])) for member in members])
            target=regional_covariance(truth,mass,task,tuple(receipt['field_order']))
            values.append(float(scalar_task_score(member_values,target,model_family=family).mean()))
        source_score=sum(values)/len(values)
        fallback=contract['reserved_task_normalizer']; rule=contract['reserved_task_normalization_rule']
        threshold=rule['near_zero_threshold']
        if (not math.isfinite(source_score) or not math.isfinite(fallback) or fallback <= 0
                or rule['primary'] != 'final_train_source_task_score'
                or rule['fallback'] != 'train_truth_absolute_covariance' or threshold != 1e-12*fallback):
            raise ValueError('invalid predeclared TRAIN task unit rule')
        units[family]={'source_score':source_score,'normalizer':fallback if source_score <= threshold else source_score,
                       'used_truth_fallback':source_score <= threshold,'threshold':threshold,
                       'source_bank':bind(campaign,receipt_path), 'contract':ref}
        source=setup['sources'][family]['source_fit']
        artifacts[family]={'core_run':source['run'],'core':bind(campaign,source['core'],source['core_sha256']),
                           'source_checkpoint':bind(campaign,source['source_checkpoint'],source['source_checkpoint_sha256']),
                           'contract':ref}
    freeze_doc={'schema':'car_round2_raw_closeout_freeze_v1','selection':bind(campaign,selection_path),
                'assessment_script_sha256':sha(CODE/'scripts/posttraining/assess_car_round2_checkpoints.py'),
                'adapter_sha256':sha(Path(__file__)), 'artifacts':artifacts,'task_units':units,
                'validation_panel':selection['validation_panel'],'event':selection['event'],
                'reserved_task':task_ref,'arms':selections,'created_unix':time.time()}
    write(output/'freeze.json',freeze_doc)
    print(output/'freeze.json')


def apply_freeze(base, campaign, freeze_path):
    doc = json.loads(Path(bind(campaign,freeze_path)['path']).read_text())
    if doc.get('schema')!='car_round2_raw_closeout_freeze_v1' or set(doc.get('arms',{}))!=set(ARMS):
        raise ValueError('sealed raw closeout freeze required')
    selection=json.loads(Path(bind(campaign,doc['selection']['path'],doc['selection']['sha256'])['path']).read_text())
    if selection['arms'] != doc['arms'] or selection['validation_panel'] != doc['validation_panel']:
        raise ValueError('sealed checkpoint or validation panel selection changed')
    for values in doc['artifacts'].values():
        for key in ('core', 'source_checkpoint', 'contract'):
            bind(campaign,values[key]['path'],values[key]['sha256'])
    for values in doc['task_units'].values():
        bind(campaign,values['source_bank']['path'],values['source_bank']['sha256'])
    for key in ('validation_panel','event','reserved_task'):
        bind(campaign,doc[key]['path'],doc[key]['sha256'])
    if sha(CODE/'scripts/posttraining/assess_car_round2_checkpoints.py') != doc['assessment_script_sha256'] or sha(Path(__file__)) != doc['adapter_sha256']:
        raise ValueError('frozen assessment or adapter code differs')
    for arm, epoch in ARMS.items():
        item=doc['arms'][arm]
        if item['epoch']!=epoch or item['alpha']!=1:
            raise ValueError('raw checkpoint selection changed')
        bind(campaign,item['checkpoint']['path'],item['checkpoint']['sha256'])
        bind(campaign,item['config']['path'],item['config']['sha256'])
    def barrier(current):
        if current.resolve()!=campaign: raise ValueError('wrong campaign')
    def nomination(current, family, arm=None, epoch=None):
        barrier(current)
        if arm is None: return doc, None
        item=doc['arms'].get(arm)
        if item is None or item['epoch']!=epoch or ('ffm' if arm.startswith('FFM_') else 'senseiver')!=family:
            raise ValueError('arm and epoch are outside frozen raw closeout')
        return doc, {'selected_alpha':1., 'nomination_qualified':True,
                     'nomination_reason':'predeclared raw closeout; no TRAIN amplitude selection',
                     'child_checkpoint_sha256':item['checkpoint']['sha256'],
                     'source_checkpoint_sha256':doc['artifacts'][family]['source_checkpoint']['sha256']}
    base.global_train_choice_barrier=barrier
    base.validation_freeze=nomination
    base.validate_reserved_freeze=lambda current,family: {'normalizer':doc['task_units'][family]['normalizer']}
    base.assessment_budget_limit=lambda current,final: 130*3600 if final else 124*3600
    # Preserve the scoring function verbatim except allow the predeclared FFM 750
    # endpoint to use the same final reserved-task evaluation as epoch 1000.
    source=inspect.getsource(base.score)
    old="split == 'validation' and epoch == 1000"
    new="split == 'validation' and (epoch == 1000 or (family == 'ffm' and epoch == 750))"
    old_receipt="hash_file(campaign / 'assessment/train' / f'reserved_normalizer_{family}.json')"
    if source.count(old)!=1 or source.count(old_receipt)!=1:
        raise ValueError('underlying score source differs from reviewed adaptation')
    source=source.replace(old,new).replace(old_receipt,'hash_file(freeze_path)')
    namespace=base.__dict__
    namespace['freeze_path']=Path(freeze_path)
    exec(compile(source,str(CODE/'scripts/posttraining/assess_car_round2_checkpoints.py'),'exec'),namespace)  # noqa: S102
    return doc


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('freeze'); prep.add_argument('--plan',type=Path,required=True);prep.add_argument('--output',type=Path,required=True)
    gen=sub.add_parser('generate');gen.add_argument('--freeze',type=Path,required=True);gen.add_argument('--kind',choices=('source','child'),required=True)
    gen.add_argument('--arm',choices=tuple(ARMS),required=True);gen.add_argument('--source-bank',type=Path);gen.add_argument('--output',type=Path,required=True)
    gen.add_argument('--gpu-uuid',required=True);gen.add_argument('--max-seconds',type=float,required=True);gen.add_argument('--save-margin-seconds',type=float,required=True)
    score=sub.add_parser('score');score.add_argument('--freeze',type=Path,required=True);score.add_argument('--arm',choices=tuple(ARMS),required=True)
    score.add_argument('--source-bank',type=Path,required=True);score.add_argument('--child-bank',type=Path,required=True);score.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();campaign=args.campaign.resolve(strict=True)
    if '/data2/wanglz/PhyCoFlow-MultiFieldReconstruction/' not in str(campaign)+'/': raise ValueError('DATA2 campaign required')
    base=load_assessment()
    if args.command=='freeze':
        freeze(args,campaign,base);return
    doc=apply_freeze(base,campaign,args.freeze)
    if campaign/'assessment/early_closeout' not in Path(args.output).resolve().parents:
        raise ValueError('early closeout output must remain in its named assessment directory')
    item=doc['arms'][args.arm];family='ffm' if args.arm.startswith('FFM_') else 'senseiver'
    if args.command=='generate':
        args.run=Path(item['run']);args.epoch=item['epoch'] if args.kind=='child' else None
        args.split='validation';args.preformal_source=False;args.final_assessment=True
        base.generate(args,campaign)
    else:
        # The original scorer imports this constant inside score(); isolate the
        # authorized raw comparison to exactly SOURCE and alpha one.
        from phycoflow_reconstruction.evaluation import round2_assessment
        round2_assessment.AMPLITUDES = (0., 1.)
        child_receipt=json.loads((args.child_bank/'receipt.json').read_text())
        if (child_receipt.get('arm') != args.arm or child_receipt.get('epoch') != item['epoch']
                or child_receipt.get('checkpoint_sha256') != item['checkpoint']['sha256']):
            raise ValueError('child bank does not match the selected raw checkpoint')
        args.core_run=Path(doc['artifacts'][family]['core_run']);args.contract=Path(doc['artifacts'][family]['contract']['path'])
        args.event=Path(doc['event']['path']);args.reserved_task=Path(doc['reserved_task']['path'])
        args.deployment_timing=None;args.allow_missing_latency=True
        base.score(args,campaign)
        assessment=json.loads((args.output/'assessment.json').read_text())
        if ([row['alpha'] for row in assessment['rows']] != [0., 1.]
                or assessment.get('selected_alpha') != 1.
                or assessment.get('epoch') != item['epoch']):
            raise ValueError('closeout scoring did not produce exactly SOURCE and raw rows')


if __name__=='__main__':
    main()
