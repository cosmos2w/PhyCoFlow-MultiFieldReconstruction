import hashlib
import json
from argparse import Namespace
from pathlib import Path
import pytest
from scripts.posttraining.review_car_round6_learning import decide_review, decide_disposition, historical_comparison


def put(path, value):
    path.write_text(json.dumps(value))
    return path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def sample(tmp_path):
    checkpoint=put(tmp_path/'checkpoint.json',{'checkpoint':'fixture'})
    prepared=put(tmp_path/'prepared.json',{'prepared':'fixture'})
    config=put(tmp_path/'config.json',{'config':'fixture'})
    protocol=(tmp_path/'resumption.md');protocol.write_text('explicit resumed decision protocol')
    fidelity={'primary_field_pass':True,'sensor_pass':True,'primary_pass':True,'legacy_member_pass':True,
              'field_difference':[.09,.02],'field_allowance':[.1,.1]}
    arm='FFM_SSS_EpochCalibrated_PI_Repair'
    candidate={'fidelity':fidelity,'normalized_core_differences':{'state':-.049,'scale':.001,'shape':-.01}}
    bank=put(tmp_path/'bank.json',{'status':'completed','epoch':100,'arm':arm,'members':4,
        'sample_ids':[f'sample_{i}' for i in range(100)],'source_bank_sha256':'source-hash','panel_sha256':'panel-hash',
        'checkpoint_sha256':sha(checkpoint),'prepared_sha256':sha(prepared),'config_sha256':sha(config)})
    assessment={'schema':'car_round6_matched_validation_v1','split':'validation','population':1000,
        'members':4,'sample_ids':[f'sample_{i}' for i in range(100)],'epoch':100,
        'task_scope':'both tasks excluded','arm':arm,'candidates':{arm:candidate},'source':{},
        'field_order':['one','two'],'bank_receipt_sha256':sha(bank),'checkpoint_sha256':sha(checkpoint),
        'prepared_sha256':sha(prepared),'source_bank_sha256':'source-hash','panel_sha256':'panel-hash'}
    review=put(tmp_path/'review.json',assessment)
    history=tmp_path/'history.jsonl';history.write_text(''.join(json.dumps({'epoch':i,'epoch_complete':True,'step':12*i})+'\n' for i in range(1,101)))
    journal=tmp_path/'pi_journal.jsonl'
    journal.write_text(''.join(json.dumps({'step':i,'epoch':(i-1)//12+1,
        'field_violation':{'one':2. if i%2 else -2.,'two':1. if i%2 else -3.},
        'multipliers_before':[1. if i%2 else 0.,.25],
        'action_constraint_product':-.2 if i%2 else .1})+'\n' for i in range(1,1201)))
    args=Namespace(assessment=review,history=history,pi_journal=journal,checkpoint=checkpoint,bank_receipt=bank,prepared=prepared,
        resolved_config=config,resumption_protocol=protocol,historical_review=None,decision='continue_to_500',
        remaining_gpu_hours=10.,forecast_remaining_gpu_hours=9.,rationale='Field retention is improving and Scale is stable.',
        scientific_progress='Shape improves while all fields satisfy their guards.')
    return args,assessment,bank,protocol


def test_learning_continuation_is_explicit_and_not_numeric_margin_gate(tmp_path):
    args,assessment,bank,protocol=sample(tmp_path)
    result=decide_review(args)
    assert result['continue_to_500'] is True
    assert result['authorize_endpoint_tasks'] is False
    assert result['metrics']['original_reference']['any_core_improvement_at_least_0_05'] is False
    assert result['assessment']['sha256']==sha(args.assessment)
    args.forecast_remaining_gpu_hours=11.
    with pytest.raises(ValueError,match='remaining GPU budget'):decide_review(args)
    args.forecast_remaining_gpu_hours=9.;assessment['candidates'][assessment['arm']]['reused_tasks']={'not_allowed':True}
    put(args.assessment,assessment)
    with pytest.raises(ValueError,match='task-free'):decide_review(args)


def test_disposition_permits_margin_miss_but_never_silent_physical_failure(tmp_path):
    args,assessment,bank,protocol=sample(tmp_path)
    args.decision='stop_at_epoch_100';review_decision=decide_review(args)
    assert review_decision['authorize_endpoint_tasks'] is True
    prior=put(tmp_path/'decision.json',review_decision)
    assessment['task_scope']='reused development tasks at selected endpoint'
    assessment['review_sha256']=review_decision['review_sha256']
    assessment['task_decision_sha256']=sha(prior)
    assessment['source']['reused_tasks']={'event_score':1.}
    candidate=assessment['candidates'][assessment['arm']]
    candidate['reused_tasks']={'regional_normalized_improvement':.049,'event_normalized_difference':.051}
    candidate['replacement_eligible']=False
    terminal=put(tmp_path/'terminal.json',assessment)
    disposition=Namespace(assessment=terminal,review_decision=prior,bank_receipt=bank,resumption_protocol=protocol,
        recommendation='replace',rationale='Small margin misses are outweighed by field retention and coherent gains.',
        scientific_progress='The shared model shows physical retention and useful Shape improvement.')
    result=decide_disposition(disposition)
    assert result['recommendation']==assessment['arm']
    assert result['metrics']['original_reference']['replacement_eligible'] is False
    assert result['original_task_reference']['regional_improvement_at_least_0_05'] is False
    assessment['task_decision_sha256']='different-decision-hash';put(terminal,assessment)
    with pytest.raises(ValueError,match='terminal assessment must reuse'):decide_disposition(disposition)
    assessment['task_decision_sha256']=sha(prior)
    candidate['fidelity'].update(primary_field_pass=False,primary_pass=False)
    put(terminal,assessment)
    with pytest.raises(ValueError,match='physical safeguard'):decide_disposition(disposition)


def test_resolved_config_and_core_unit_bindings(tmp_path):
    args,assessment,bank,protocol=sample(tmp_path)
    args.resolved_config.write_text('changed after bank generation')
    with pytest.raises(ValueError,match='resolved configuration'):decide_review(args)
    keys=('split','population','members','field_order','sample_ids','source_checkpoint_sha256','source_bank_sha256',
          'panel_sha256','contract_sha256','core_sha256','source_baseline_sha256')
    current={k:assessment.get(k) for k in keys}
    current.update(epoch=100,normalizers={'state':1.,'scale':2.,'shape':3.})
    prior=dict(current,normalizers={**current['normalizers'],'event':4.},candidates={
        'FFM_SourceScaledRisk_Uniform_Repair':{'normalized_core_differences':{'state':0.,'scale':0.,'shape':0.},
        'fidelity':{'field_difference':[.1,.2],'field_allowance':[1.,1.]}}})
    assert historical_comparison(current,prior,{})['sample_ids_match'] is True
    prior['normalizers']['shape']=4.
    with pytest.raises(ValueError,match='core normalizers'):historical_comparison(current,prior,{})


def test_pi_window_pools_signed_actions_and_rejects_missing_or_reordered(tmp_path):
    args,_,_,_=sample(tmp_path)
    result=decide_review(args)
    window=result['pi_action_windows_51_75_76_100']['windows']['epochs_51_75']
    assert window['action_count']==300
    assert window['mean_signed_e']=={'one':0.,'two':-1.}
    assert window['positive_part_of_mean_e']=={'one':0.,'two':0.}
    assert window['mean_multipliers_before']=={'one':.5,'two':.25}
    assert window['fraction_nonzero_multiplier_by_field']=={'one':.5,'two':1.}
    assert window['fraction_any_nonzero_multiplier']==1.
    assert window['action_constraint_product']['negative_count']==150
    assert window['action_constraint_product']['positive_count']==150
    assert window['action_constraint_product']['mean']==pytest.approx(-.05)
    assert result['evidence']['pi_journal']['sha256']==sha(args.pi_journal)
    lines=args.pi_journal.read_text().splitlines()
    args.pi_journal.write_text('\n'.join(lines[:-1])+'\n')
    with pytest.raises(ValueError,match='exactly 12 actions'):decide_review(args)
    lines[601],lines[602]=lines[602],lines[601]
    args.pi_journal.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError,match='contiguous and ordered'):decide_review(args)
    lines[601],lines[602]=lines[602],lines[601]
    reordered=json.loads(lines[600])
    reordered['field_violation']=dict(reversed(list(reordered['field_violation'].items())))
    lines[600]=json.dumps(reordered)
    args.pi_journal.write_text('\n'.join(lines)+'\n')
    assert decide_review(args)['pi_action_windows_51_75_76_100']==result['pi_action_windows_51_75_76_100']
    wrong=json.loads(lines[600]);wrong['field_violation'].pop('two');lines[600]=json.dumps(wrong)
    args.pi_journal.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError,match='field identities'):decide_review(args)
    args.pi_journal=None
    with pytest.raises(ValueError,match='requires its immutable'):decide_review(args)
