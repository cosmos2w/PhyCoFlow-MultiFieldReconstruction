"""Synchronized timing of existing qualification updates on disposable copies."""
from __future__ import annotations
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
from time import perf_counter

import torch
import yaml

from ..evaluation.round8_campaign import file_hash
from .run_store import config_digest, load_project_checkpoint
from .round8_execution import execute_training
from .round8_runtime import Round8Runtime
from .fidelity_budget import ExternalPolicy
from . import fidelity_budget


@contextmanager
def synchronized_training_timings(*, device, records, synchronize=None):
    """Wrap existing calls only; restore methods even when a call fails.

    Synchronization makes each measurement wall-clock GPU completion time. The
    nested gradient timings are parts of, and must not be added to, policy time.
    """
    sync = synchronize if synchronize is not None else lambda: torch.cuda.synchronize(torch.device(device))
    original_score = Round8Runtime.score_family
    original_policy = ExternalPolicy.update_with_coherence_aggregation
    original_flat = fidelity_budget._flat_gradient
    active = None
    current_step = None

    def timed_score(self, name, family, prediction, reference, *, coordinates, context):
        nonlocal current_step
        sync(); started=perf_counter()
        try:
            return original_score(self,name,family,prediction,reference,coordinates=coordinates,context=context)
        finally:
            sync(); records.append({'kind':'score_family','family':name,'step':int(self.step),
                                    'seconds':perf_counter()-started})
            current_step=int(self.step)

    def timed_flat(loss, parameters):
        if active is None:
            return original_flat(loss, parameters)
        index=len(active['gradient_calls'])
        names=active['family_order']
        label=names[index] if index < len(names) else 'constraint'
        sync(); started=perf_counter()
        try:
            return original_flat(loss,parameters)
        finally:
            sync(); active['gradient_calls'].append({'ordinal':index+1,'label':label,
                                                      'seconds':perf_counter()-started})

    def timed_policy(self, model, optimizer, family_losses, *args, **kwargs):
        nonlocal active
        if active is not None: raise RuntimeError('nested policy profiling is unsupported')
        names=list(family_losses)
        if names != ['global_distribution','multiscale_spectrum','topology']:
            raise ValueError('profile requires the three ordered Round 8 families')
        active={'kind':'policy_update','step':current_step,'family_order':names,'gradient_calls':[]}
        sync(); started=perf_counter()
        try:
            result = original_policy(self,model,optimizer,family_losses,*args,**kwargs)
            if (result.get('family_gradient_evaluations') != 3 or
                    len(active['gradient_calls']) != 3 + result.get('constraint_gradient_evaluations', -1)):
                raise ValueError('profile observed unexpected policy gradient calls')
            return result
        finally:
            sync(); active['seconds']=perf_counter()-started
            records.append(active); active=None

    Round8Runtime.score_family = timed_score
    ExternalPolicy.update_with_coherence_aggregation = timed_policy
    fidelity_budget._flat_gradient = timed_flat
    try:
        yield
    finally:
        Round8Runtime.score_family = original_score
        ExternalPolicy.update_with_coherence_aggregation = original_policy
        fidelity_budget._flat_gradient = original_flat


def _validate_records(records, start_step):
    names=['global_distribution','multiscale_spectrum','topology']
    scores=[r for r in records if r['kind']=='score_family']
    policies=[r for r in records if r['kind']=='policy_update']
    if len(scores)!=6 or len(policies)!=2:
        raise ValueError('profile did not execute exactly two complete three-family updates')
    for offset, policy in enumerate(policies):
        step=start_step+offset
        group=[row for row in scores if row['step']==step]
        if (len(group) != 3 or {row['family'] for row in group} != set(names) or policy['step'] != step
                or [row['label'] for row in policy['gradient_calls'][:3]] != names):
            raise ValueError('profile family or gradient call order differs')
        if len(policy['gradient_calls']) < 3:
            raise ValueError('profile lacks the three family gradients')
    if any(not isinstance(row['seconds'],(int,float)) or row['seconds'] < 0
           for row in scores + policies):
        raise ValueError('invalid profile timing')


def profile_training(*, qualification_receipt, artifacts_path, output_path, campaign,
                     case_dir, charge_attempt, synchronize=None):
    """Two updates per qualification arm from verified independent clones."""
    receipt_path=Path(qualification_receipt).resolve()
    if receipt_path != (Path(campaign).resolve() / 'artifacts' / 'qualified.json'):
        raise ValueError('training profile requires the latest qualification receipt')
    receipt_hash=file_hash(receipt_path)
    receipt=json.loads(receipt_path.read_text())
    if (receipt.get('schema')!='round8_disposable_qualification_v1' or receipt.get('status')!='completed'
            or receipt.get('optimizer_attempts')!=14 or receipt.get('fork',{}).get('qualification_only') is not True):
        raise ValueError('a completed fourteen-attempt qualification receipt is required')
    artifact_path=Path(artifacts_path).resolve()
    artifact_hash=file_hash(artifact_path)
    artifacts=torch.load(artifact_path,map_location='cpu',weights_only=True)
    output=Path(output_path).resolve()
    if output.exists(): raise FileExistsError(f'profile receipt already exists: {output}')
    output.parent.mkdir(parents=True,exist_ok=True)
    clone_root=output.parent/(output.stem+'_clones')
    clone_root.mkdir(exist_ok=False)
    campaign_qualification=(Path(campaign).resolve()/'qualification')
    results={}; total_attempts=0
    for arm, mode in (('B8','control'),('C8','candidate')):
        segment=receipt.get('segments',{}).get(arm,{})
        fork=receipt.get('fork',{}).get('arms',{}).get(arm,{})
        parent=Path(segment.get('run','')).resolve()
        if (not parent.is_relative_to(campaign_qualification) or not parent.is_dir()
                or parent != Path(fork.get('run','')).resolve() or len(segment.get('update_seconds',()))!=2):
            raise ValueError(f'{arm} completed qualification child identity differs')
        checkpoint_path=parent/'checkpoints'/'last.pt'
        parent_hash=file_hash(checkpoint_path)
        checkpoint=load_project_checkpoint(checkpoint_path)
        start=checkpoint.get('global_step')
        runtime=checkpoint.get('round8_runtime',{})
        if (isinstance(start,bool) or not isinstance(start,int) or start!=12
                or checkpoint.get('qualification_only') is not True
                or runtime.get('mode')!=mode or runtime.get('step')!=start):
            raise ValueError(f'{arm} child is not the completed two-update qualification endpoint')
        config=yaml.safe_load((parent/'resolved_config.yaml').read_text())
        manifest=json.loads((parent/'run_manifest.json').read_text())
        if (config.get('optimization',{}).get('steps_per_epoch')!=1
                or config_digest(config)!=manifest.get('config_sha256')
                or checkpoint.get('config_sha256')!=manifest.get('config_sha256')):
            raise ValueError(f'{arm} child config binding differs')
        clone=clone_root/arm
        shutil.copytree(parent,clone,symlinks=False,copy_function=shutil.copy2)
        if file_hash(clone/'checkpoints'/'last.pt') != parent_hash:
            raise ValueError('copied qualification checkpoint differs')
        if any(p.is_symlink() for p in clone.rglob('*')):
            raise ValueError('profile clone must contain regular independent files')
        records=[]; attempt_steps=[]
        def charge(phase, *, arm, step):
            if phase!='qualification' or arm!=mode or step!=start+len(attempt_steps) or len(attempt_steps)>=2:
                raise ValueError('profile optimizer attempts differ from the fixed two-update budget')
            charge_attempt(phase,arm=arm,step=step)
            attempt_steps.append(int(step))
        with synchronized_training_timings(device=config['runtime']['device'],records=records,synchronize=synchronize):
            returned=execute_training(config,case_dir=case_dir,artifacts=artifacts,phase='qualification',
                                      charge_attempt=charge,mode=mode,resume=clone,max_steps=2)
        if Path(returned).resolve()!=clone.resolve() or attempt_steps!=[start,start+1]:
            raise ValueError('profile did not execute exactly the requested cloned updates')
        final_checkpoint=load_project_checkpoint(clone/'checkpoints'/'last.pt')
        if final_checkpoint.get('global_step')!=start+2:
            raise ValueError('profile cloned checkpoint did not advance exactly two updates')
        _validate_records(records,start)
        if file_hash(checkpoint_path)!=parent_hash:
            raise ValueError('original qualification checkpoint changed while profiling')
        results[arm]={'parent_run':str(parent),'parent_checkpoint_sha256':parent_hash,
                      'clone_run':str(clone),'clone_start_checkpoint_sha256':parent_hash,
                      'clone_final_checkpoint_sha256':file_hash(clone/'checkpoints'/'last.pt'),
                      'start_step':start,'end_step':start+2,'attempt_steps':attempt_steps,'timings':records}
        total_attempts+=len(attempt_steps)
    if file_hash(receipt_path) != receipt_hash or file_hash(artifact_path) != artifact_hash:
        raise ValueError('qualification receipt or runtime artifacts changed during profiling')
    report={'schema':'round8_training_compute_profile_v1','status':'completed','qualification_only':True,
            'qualification_receipt':str(receipt_path),'qualification_receipt_sha256':receipt_hash,
            'artifacts':str(artifact_path),'artifacts_sha256':artifact_hash,
            'optimizer_attempts':total_attempts,'arms':results,
            'timing_semantics':'CUDA-synchronized wall times; family calls and policy call are separate; gradient calls are nested within policy time. Synchronization instrumentation perturbs timings.',
            'scope':'Two existing qualification updates per disposable clone; no model ranking or efficacy inference.'}
    with output.open('x') as stream: stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return output
