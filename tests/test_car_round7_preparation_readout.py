"""Terminal preparation readout validates complete pilot clocks and curve bytes."""
import json
from pathlib import Path
import pytest
import yaml
from PIL import Image
from scripts.analysis.car_round7_preparation_readout import FAMILIES, ref, summarize_pilot, summarize_screen


def write_lines(path, values):
    path.write_text(''.join(json.dumps(value)+'\n' for value in values))


def pilot_fixture(tmp_path):
    run=tmp_path/'pilot';logs=run/'logs';metrics=run/'metrics';checkpoint=run/'checkpoints'
    for folder in (logs,metrics,checkpoint):folder.mkdir(parents=True)
    saved=run/'resolved_config.yaml'
    saved.write_text(yaml.safe_dump({'coherence':{'scalable_anchored_coherence':{'scale_objective':'joint_anchor','scale_joint_kappa':2.}},
        'optimization':{'multitask':{'fidelity_budget':{'reference_artifact':{'sha256':'sample-reference'}}}}}))
    ck=checkpoint/'last.pt';ck.write_bytes(b'checkpoint fixture')
    write_lines(logs/'history.jsonl',[{'epoch':e,'step':e*12,'epoch_complete':True,'optimizer_lr':0.001,
        'round7_scale_kappa':2.,'topology.self':0.2,'topology.mutual':0.3,
        'coherence_component/multiscale_spectrum/joint/raw':0.4,
        'coherence_component/multiscale_spectrum/joint/weighted_contribution':0.8,
        'round7_active_field_row_fraction':0.5, **{f'coherence_family/{name}/{part}':0.1 for name in ('global_distribution','multiscale_spectrum','topology') for part in ('raw','weighted_contribution')}} for e in range(1,11)])
    write_lines(logs/'epoch_timing.jsonl',[{'epoch':e,'complete_epoch_duration':True,'timed_updates':12,'accepted_updates':12} for e in range(1,11)])
    write_lines(logs/'epoch_calibrated_pi_updates.jsonl',[{'step':i,'field_violation':{'field':0.1},
        'multipliers_before':[0.1],'multipliers_after':[0.2],'z_after':[0.1],
        'action_family_products':{'actual_displacement':{name:-0.1 for name in ('global_distribution','multiscale_spectrum','topology')}},
        'action_constraint_product':0.1,'action_displacement_norm':0.2} for i in range(1,121)])
    write_lines(metrics/'training_coupling.jsonl',[{'step':i,'accepted':True,'sample_ids':['one'], 'observation_sha256':'obs','query_sha256':'query','member_seeds':[1,2], 'first_member_generator_state_sha256':'member','field_complement_sha256':'complement'} for i in range(1,121)])
    write_lines(logs/'scalable_updates.jsonl',[{'step':i,'update_accepted':True,'coefficients':{'one':1}, 'calibration_frozen':True,'exposure':[{'condition_group':g,'kind':kind,'count':1} for g in range(8) for kind in ('self','pair')]} for i in range(1,121)])
    (logs/'recording_definitions.json').write_text('{}')
    image=Image.new('RGB',(8,8),'white')
    for name in ('loss_history','coherence_history'):
        image.save(logs/(name+'.png'));image.save(logs/(name+'.pdf'),'PDF')
    result=tmp_path/'pilot_result.json';result.write_text(json.dumps({'schema':'car_round7_training_result_v1',
        'arm':'FFM_Round7_Joint_Anchor_Scale','target_epoch':10,'global_step':120,'target_reached':True,
        'disposition':'target_reached','run_dir':str(run),'checkpoint':ref(ck),'saved_config':ref(saved)}))
    return result,logs,metrics


def test_complete_pilot_checks_clocks_and_decodes_curves(tmp_path):
    result,_,_=pilot_fixture(tmp_path)
    summary=summarize_pilot(result)
    assert summary['contiguous_actions']==120 and summary['reporting_epochs']==10
    assert summary['kappa']==2. and len(summary['curves_decoded'])==4
    assert summary['scalable_calibration']['coefficients_constant']
    assert summary['query_group_exposure']['7']['pair']==120
    assert summary['actual_displacement_products']['products']['topology']['negative']==120
    assert summary['actual_displacement_products']['source']==ref(result.parent/'pilot/logs/epoch_calibrated_pi_updates.jsonl')
    assert all(min(curve['decoded_dimensions'])>0 for curve in summary['curves_decoded'])


def test_incomplete_action_journal_is_rejected(tmp_path):
    result,logs,_=pilot_fixture(tmp_path)
    journal=logs/'epoch_calibrated_pi_updates.jsonl'
    journal.write_text('\n'.join(journal.read_text().splitlines()[:-1])+'\n')
    with pytest.raises(ValueError,match='120 contiguous actions'):summarize_pilot(result)


def test_screen_family_mapping_order_is_not_scientific(tmp_path):
    difference={'max_abs':0.,'l2_abs':0.,'relative_l2':0.}
    family={**difference,'baseline_norm':1.,'candidate_norm':1.}
    attempts=[]
    for objective in ('composite','joint_anchor'):
        for setting in ('separate_repeat_1','separate_repeat_2','factored_static_features'):
            candidate=setting=='factored_static_features'
            attempts.append({'objective':objective,'setting':setting,'method':'factored' if candidate else 'separate',
                'context_cache':'static_features' if candidate else 'none','checkpointing':True,'chunk':2,
                'selected_query_sha256':'query','complement_sha256':'complement','profiler_enabled':False,
                'parameter_row_dtype':'float32','per_family_row_comparison_to_first_separate':{name:family for name in reversed(FAMILIES)},
                'comparison_to_first_separate':{name:difference for name in ('values','rows','field','direction','displacement','risk')},
                'weighted_field_row_norms':{'baseline':0.},'clip_input_norm':1.,'forward_seconds':1.,'reverse_seconds':1.,
                'solver_optimizer_seconds':1.,'peak_allocated_bytes':1,'peak_reserved_bytes':1})
    receipt={'schema':'car_round7_executor_screen_v1','status':'completed_screen','state':'fresh_source',
        'family_order':list(FAMILIES),'field_row_active':False,'backend_policy':{'deterministic':True,
        'strict_algorithms':True,'deterministic_algorithms_enabled':True,'cudnn_deterministic':True,
        'cudnn_benchmark':False,'cublas_workspace_config':':4096:8'},'attempts':attempts,
        'reference_sha256':'reference','config_sha256':'config','sample_ids':['condition'],'wall_seconds':1.}
    path=tmp_path/'screen.json';path.write_text(json.dumps(receipt))
    summary=summarize_screen(path,'fresh_source')
    assert list(summary['attempts'][0]['per_family_row_comparison_to_first_separate'])==list(FAMILIES)
    receipt['family_order'].reverse();path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='family order'):summarize_screen(path,'fresh_source')
