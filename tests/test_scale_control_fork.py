import json
import os
from pathlib import Path
import pytest
import torch
import yaml
from phycoflow_reconstruction.training.fork_control import create_control_fork
from phycoflow_reconstruction.training.monitoring import TrainingMonitor
from phycoflow_reconstruction.training.checkpointing import PeriodicCheckpointManager
from phycoflow_reconstruction.training.run_store import RunStore, file_sha256

PARENT=Path(os.environ['CAR_FORK_PARENT_RUN']) if os.environ.get('CAR_FORK_PARENT_RUN') else None
SHA='d3d7122ba402467a38ee1061134e6b22840550258b71a0930df99359586f7e62'

@pytest.mark.skipif(PARENT is None or not PARENT.is_dir(), reason='set CAR_FORK_PARENT_RUN to the actual parent for integration test')
def test_actual_parent_fork_is_independent_and_unchanged_science(tmp_path):
 cfg=yaml.safe_load((PARENT/'resolved_config.yaml').read_text());cfg['output']['experiment_name']='Test_Fork_Control';cfg['runtime']['device']='cpu'
 run=create_control_fork(tmp_path,cfg,PARENT,parent_checkpoint_sha256=SHA)
 old=torch.load(PARENT/'checkpoints/epoch_500.pt',map_location='cpu',weights_only=False)
 new=torch.load(run/'checkpoints/last.pt',map_location='cpu',weights_only=False)
 assert new['global_step']==6000 and new['external_policy']==old['external_policy']
 assert all(torch.equal(old['model'][k],new['model'][k]) for k in old['model'])
 assert torch.equal(old['rng_state']['index_generator'],new['rng_state']['index_generator'])
 assert not (run/'logs/history.jsonl').exists()
 assert RunStore.resume(run,cfg).run_dir==run
 bad=yaml.safe_load(yaml.safe_dump(cfg));bad['optimization']['lr']*=2
 with pytest.raises(ValueError,match='identical science'):create_control_fork(tmp_path,bad,PARENT,parent_checkpoint_sha256=SHA)
 with pytest.raises(ValueError,match='scientific configuration'):RunStore.resume(run,bad)
 assert file_sha256(PARENT/'checkpoints/epoch_500.pt')==SHA

def test_origin_monitor_and_checkpoint_mapping(tmp_path):
 run=tmp_path/'run';(run/'logs').mkdir(parents=True)
 monitor=TrainingMonitor(run,start_step=6000,final_step=6024,configured_steps=12000,steps_per_epoch=12,description='fork',enabled=False,origin_step=6000)
 for step in range(6001,6013):monitor.record({'step':step,'epoch':1,'total':1.0},lr=1e-5)
 monitor.finish_step();monitor.close()
 row=json.loads((run/'logs/history.jsonl').read_text().splitlines()[0]); assert (row['step'],row['epoch'],row['absolute_epoch'])==(6012,1,501)
 store=RunStore(run,{'stage':'post_training'});config={'checkpointing':{'enabled':True,'epochs':[100,500]},'evaluation':{'external_assessment':True}}
 c=PeriodicCheckpointManager(config,store=store,steps_per_epoch=12,origin_step=6000)
 assert c.milestone_due(7200) and c.milestone_due(12000) and not c.milestone_due(6000)
 from phycoflow_reconstruction.training.post_training import _phase_milestone_path
 assert _phase_milestone_path(run,7200,12,6000)==run/'checkpoints/epoch_100.pt'
 assert _phase_milestone_path(run,12000,12,6000)==run/'checkpoints/epoch_500.pt'
 assert _phase_milestone_path(run,1200,12)==run/'checkpoints/epoch_100.pt'

def test_fork_fresh_process_partial_recording_matches_full(tmp_path):
 import os, subprocess, sys
 worker=tmp_path/'worker.py'
 worker.write_text('''\nimport json,sys\nfrom pathlib import Path\nfrom phycoflow_reconstruction.training.monitoring import TrainingMonitor\nroot,mode=Path(sys.argv[1]),sys.argv[2]\n(root/'logs').mkdir(parents=True,exist_ok=True)\n(root/'resolved_config.yaml').write_text("coherence:\\n  families:\\n    state: {enabled: true}\\n    scale: {enabled: true}\\n  scalable_anchored_coherence:\\n    common_prefit: {path: unused}\\n")\nTrainingMonitor._plot=lambda self: None\nstate=json.loads((root/'state.json').read_text()) if mode=='resume' else None\nstart=6017 if mode=='resume' else 6000\nm=TrainingMonitor(root,start_step=start,final_step=6024,configured_steps=12000,steps_per_epoch=12,origin_step=6000,description='fork',enabled=False,resume_state=state)\nsteps=range(6001,6025) if mode=='full' else range(6001,6018) if mode=='partial' else range(6018,6025)\nfor step in steps:\n m.record({'step':step,'coherence_loss':float(step),'update_accepted':True});m.finish_step()\nif mode=='partial':(root/'state.json').write_text(json.dumps(m.checkpoint_state()))\nm.close()\n''')
 for name,mode in [('full','full'),('resume','partial'),('resume','resume')]:subprocess.run([sys.executable,str(worker),str(tmp_path/name),mode],env=os.environ,check=True)
 full=tmp_path/'full/logs'; resumed=tmp_path/'resume/logs'
 assert (full/'history.jsonl').read_bytes()==(resumed/'history.jsonl').read_bytes()
 assert (full/'recording_definitions.json').read_bytes()==(resumed/'recording_definitions.json').read_bytes()
 columns={c['name']:c for c in json.loads((full/'recording_definitions.json').read_text())['columns']}
 assert columns['step']['meaning']=='Absolute accepted updates, including the 6000 accepted parent updates before this fork.'
 assert columns['epoch']['meaning']=='Reporting epoch within this new fork phase, starting at 1.'
 assert columns['absolute_epoch']['meaning']=='Absolute reporting epoch including the 500 parent epochs before this fork.'
 assert columns['step']['units']=='updates' and columns['epoch']['units']==columns['absolute_epoch']['units']=='reporting_epochs'
 rows=[json.loads(x) for x in (full/'history.jsonl').read_text().splitlines()]
 assert [(r['step'],r['epoch'],r['absolute_epoch']) for r in rows]==[(6012,1,501),(6024,2,502)]

@pytest.mark.skipif(PARENT is None or not PARENT.is_dir(), reason='set CAR_FORK_PARENT_RUN to the actual parent for integration test')
def test_candidate_fork_admits_only_explicit_quadrature(tmp_path):
 from phycoflow_reconstruction.training.fork_control import create_fidelity_quadrature_fork
 cfg=yaml.safe_load((PARENT/'resolved_config.yaml').read_text())
 cfg['output']['experiment_name']='RotatingComplement_FieldFeedback_Fork_Test'
 cfg['runtime']['device']='cpu'
 cfg['optimization']['multitask']['fidelity_budget']['rotating_complement']={'points':512,'seed':20261125}
 bad=yaml.safe_load(yaml.safe_dump(cfg));bad['optimization']['lr']*=2
 with pytest.raises(ValueError,match='identical science'):
  create_fidelity_quadrature_fork(tmp_path,bad,PARENT,parent_checkpoint_sha256=SHA)
 with pytest.raises(ValueError,match='identical science'):
  create_control_fork(tmp_path,cfg,PARENT,parent_checkpoint_sha256=SHA)
 run=create_fidelity_quadrature_fork(tmp_path,cfg,PARENT,parent_checkpoint_sha256=SHA)
 ckpt=torch.load(run/'checkpoints/last.pt',map_location='cpu',weights_only=False)
 old=torch.load(PARENT/'checkpoints/epoch_500.pt',map_location='cpu',weights_only=False)
 assert ckpt['global_step']==old['global_step']==6000
 assert ckpt['external_policy']==old['external_policy']
 assert ckpt['fork_origin']['scientific_transition']=='rotating_complement_field_quadrature_512_v1'
 assert torch.equal(ckpt['rng_state']['index_generator'],old['rng_state']['index_generator'])
 assert file_sha256(PARENT/'checkpoints/epoch_500.pt')==SHA
