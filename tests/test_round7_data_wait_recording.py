"""Round 7 batch wait is an ordered, explicitly defined epoch time."""
import json
import pytest
import yaml
from phycoflow_reconstruction.training.monitoring import TrainingMonitor


def test_round7_data_wait_column_is_seconds_and_epoch_mean(tmp_path):
    (tmp_path/'resolved_config.yaml').write_text(yaml.safe_dump({
        'coherence':{'scalable_anchored_coherence':{'common_prefit':True},
                     'families':{'state':{},'scale':{},'shape':{}}},
        'optimization':{'multitask':{'fidelity_budget':{'reference_law':'training_selected_complement_v1'}}}}))
    monitor=TrainingMonitor(tmp_path,start_step=0,final_step=2,configured_steps=2,
                            steps_per_epoch=2,description='wait_fixture',enabled=False)
    monitor._plot=lambda:None
    executor={'family_vjp':'factored','vjp_chunk_size':2,'rollout_checkpointing':True,
              'rollout_context_cache':'static_features'}
    monitor.record({'step':1,'data_wait_seconds':0.2,'coherence_loss':1.0, 'round7_executor':executor})
    monitor.record({'step':2,'data_wait_seconds':0.4,'coherence_loss':0.5, 'round7_executor':executor})
    monitor.close()
    definitions=json.loads((tmp_path/'logs/recording_definitions.json').read_text())
    column=next(item for item in definitions['columns'] if item['name']=='data_wait_seconds')
    assert column['type']=='number' and column['units']=='seconds'
    assert 'batch-source waiting' in column['meaning'] and 'no GPU synchronization' in column['meaning']
    executor_column=next(item for item in definitions['columns'] if item['name']=='round7_executor')
    assert executor_column['type']=='object' and 'context-cache' in executor_column['meaning']
    history=[json.loads(line) for line in (tmp_path/'logs/history.jsonl').read_text().splitlines()]
    assert history[0]['data_wait_seconds']==pytest.approx(0.3)
    assert history[0]['round7_executor']==executor
