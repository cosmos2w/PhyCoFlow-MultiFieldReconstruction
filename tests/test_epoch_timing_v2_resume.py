"""Real fresh-process partial-epoch recording distinguishes observed timer scope."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.mark.parametrize("enabled", [False, True])
def test_partial_epoch_timing_preserves_full_scientific_history(tmp_path, enabled):
    worker=tmp_path/'worker.py'
    worker.write_text('''
import json,os,sys
from pathlib import Path
from phycoflow_reconstruction.training.monitoring import TrainingMonitor
root,mode=Path(sys.argv[1]),sys.argv[2]
(root/'logs').mkdir(parents=True,exist_ok=True)
config=root/'resolved_config.yaml'
if not config.exists():
 config.write_text("coherence:\\n  families:\\n    state: {enabled: true}\\n    scale: {enabled: true}\\n  scalable_anchored_coherence:\\n    common_prefit: {path: unused}\\n")
TrainingMonitor._plot=lambda self: None
state_path=root/'state.json'
state=json.loads(state_path.read_text()) if mode=='resume' else None
start=2 if mode=='resume' else 0
monitor=TrainingMonitor(root,start_step=start,final_step=8,configured_steps=8,
 steps_per_epoch=4,description='timing fixture',enabled=os.environ['TIMING_TEST_PROGRESS']=='1',resume_state=state)
steps=range(1,9) if mode=='full' else range(1,3) if mode=='partial' else range(3,9)
for step in steps:
 monitor.record({'step':step,'coherence_loss':float(step),'update_accepted':True,
                 'attempted_condition_visits':64,'accepted_objective_condition_visits':64})
 monitor.finish_step()
if mode=='partial':state_path.write_text(json.dumps(monitor.checkpoint_state()))
monitor.close()
''')
    for name,mode in [('full','full'),('resumed','partial'),('resumed','resume')]:
        subprocess.run([sys.executable,str(worker),str(tmp_path/name),mode],env=dict(os.environ, TIMING_TEST_PROGRESS='1' if enabled else '0'),check=True)
    full=tmp_path/'full/logs';resumed=tmp_path/'resumed/logs'
    assert (full/'history.jsonl').read_bytes()==(resumed/'history.jsonl').read_bytes()
    assert (full/'recording_definitions.json').read_bytes()==(resumed/'recording_definitions.json').read_bytes()
    original=[json.loads(x) for x in (full/'epoch_timing.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (resumed/'epoch_timing.jsonl').read_text().splitlines()]
    assert len(original)==len(rows)==2
    assert [x['timed_updates'] for x in original]==[4,4]
    assert [x['complete_epoch_duration'] for x in original]==[True,True]
    assert [x['attempted_updates'] for x in rows]==[4,4]
    assert [x['accepted_updates'] for x in rows]==[4,4]
    assert [x['timed_updates'] for x in rows]==[2,4]
    assert [x['complete_epoch_duration'] for x in rows]==[False,True]
    assert all(x['version']=='complete_epoch_timing_v2' for x in rows)
    assert all(x['duration_scope']=='current-process epoch span including callbacks; excludes prior-process time and offline pauses' for x in rows)
    assert all(x['wall_seconds']>=x['train_seconds']>0 for x in rows)
