import hashlib
import json
from pathlib import Path

import pytest

from scripts.analysis import report_car_scale_followup as report


def binding(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def test_real_completed_diagnostic_text():
    path = Path('cases/turbulent_combustion/runs/coherence_adaptive_repair/scale_alignment_followup_20261125/assessment/ScaleAlignment_CompletedDiagnostics_20261125_v1/readout.json')
    if not path.is_file():
        pytest.skip('campaign-owned completed diagnostic is unavailable')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '950ce3925e4f9bf534f0a2411133388ba5d8173b083038b47a7c639b8b338579'
    text = report.diagnostic_text(json.loads(path.read_text()))
    assert '-0.07198' in text and '+0.27398' in text
    assert 'not independent learning trajectories' in text


def test_missing_or_unfinished_decision_cannot_render(monkeypatch):
    diagnostic = {'schema':'car_scale_followup_diagnostic_readout_v1', 'status':'completed', 'test_locked':True, 'inputs':{}}
    docs = {'diag':diagnostic,'decision':{'status':'running'},'cost':{},'start':{},'aligned':{}}
    monkeypatch.setattr(report, 'doc', lambda ref:docs[ref])
    inputs = {'schema':'car_scale_followup_report_inputs_v1','phase':100,'diagnostic_readout':'diag','decision':'decision','costs':'cost','common_start_assessment':'start','aligned_scale':'aligned'}
    with pytest.raises(ValueError,match='explicit completed disposition'):
        report.validate(inputs)
    with pytest.raises(ValueError,match='selected phase manifest'):
        report.validate({'schema':'car_scale_followup_report_inputs_v1'})


def test_phase_history_and_journal_require_order_and_bindings(tmp_path):
    history=tmp_path/'history.jsonl';journal=tmp_path/'journal.jsonl'
    family={'coherence_family/global_distribution/raw':1.,'coherence_family/multiscale_spectrum/raw':2.,'coherence_family/topology/raw':3.}
    history.write_text(json.dumps({'epoch':1,'epoch_complete':True,'step':6012,**family})+'\n')
    rows=[{'step':step,'field_violation':{'T':-.1},'multipliers_before':[.1],'multipliers_after':[.1]} for step in range(6001,6013)]
    journal.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    refs={'history':binding(history),'pi_journal':binding(journal)}
    assert len(report.series(refs,1,['T'])[1])==12
    rows[5]['step']=6008;journal.write_text(''.join(json.dumps(row)+'\n' for row in rows));refs['pi_journal']=binding(journal)
    with pytest.raises(ValueError,match='complete phase PI journal'):
        report.series(refs,1,['T'])


def test_completed_endpoint_enrichment_binds_task_free_review_and_decision():
    import copy
    candidate = {'risk': {'member_mse':[1.]}, 'fidelity': {'primary_pass':True},
                 'normalized_core_differences': {'scale':.1}, 'per_condition_scores':[{'scale':1.}]}
    review = {'schema':'car_round6_matched_validation_v1','epoch':100,'bank_receipt_sha256':'bank',
              'checkpoint_sha256':'checkpoint','sample_ids':[1],'field_order':['T'],
              'source_bank_sha256':'source','panel_sha256':'panel','contract_sha256':'contract',
              'core_sha256':'core','normalizers':{'scale':1.},'arm':'repair',
              'task_scope':'both tasks excluded','candidates':{'repair':candidate}}
    decision={'epoch':100,'decision':'stop_at_epoch_100','authorize_endpoint_tasks':True,
              'review_sha256':'review','bank_receipt_sha256':'bank'}
    endpoint=copy.deepcopy(review)
    endpoint.update(task_scope='reused development tasks at selected endpoint',
                    review_sha256='review',task_decision_sha256='decision')
    endpoint['candidates']['repair']['reused_tasks']={'event_normalized_difference':.2}
    assert report.endpoint_view(review,'review',endpoint,'endpoint',decision,'decision')['event_normalized_difference']==.2
    with pytest.raises(ValueError, match='review or decision'):
        report.endpoint_view(review,'review',endpoint,'endpoint',decision,'other')
    endpoint['candidates']['repair']['risk']['member_mse']=[2.]
    with pytest.raises(ValueError, match='changed task-free risk'):
        report.endpoint_view(review,'review',endpoint,'endpoint',decision,'decision')


def test_candidate_promotion_requires_terminal_eligibility():
    views={'candidate':{'arm':'repair','candidates':{'repair':{'fidelity':{'primary_pass':True,'legacy_member_pass':True}}}}}
    endpoint={'arm':'repair','candidates':{'repair':{'replacement_eligible':False}}}
    report.check_adoption('source',views,endpoint)
    with pytest.raises(ValueError,match='endpoint replacement eligibility'):
        report.check_adoption('candidate',views,endpoint)
    endpoint['candidates']['repair']['replacement_eligible']=True
    report.check_adoption('candidate',views,endpoint)


def test_control_adoption_requires_own_task_evidence():
    with pytest.raises(ValueError,match='control endpoint task assessment'):
        report.check_adoption('control',{}, {})


def test_completed_diagnostic_detail_uses_paired_evidence():
    base=Path('cases/turbulent_combustion/runs/coherence_adaptive_repair/scale_alignment_followup_20261125/assessment')
    readout=base/'ScaleAlignment_CompletedDiagnostics_20261125_v1/readout.json'
    summary=base/'ScaleDefinition_Diagnosis_20261125_v1/summary.json'
    if not readout.is_file() or not summary.is_file():
        pytest.skip('completed campaign evidence unavailable')
    d=json.loads(readout.read_text());localization=json.loads(summary.read_text())
    adam=json.loads(report.bound(d['inputs']['adam']).read_text())
    text='\n'.join(report.detailed_diagnostic(d,localization,adam))
    assert '+0.2235' in text and '+0.2740' in text and '-0.0720' in text
    assert '+0.3030' in text and '+0.1129' in text and '+0.00012118' in text
    assert '13 independently reset' in text and 'held-out gaps are uncertain' in text


def test_grouped_query_replays_use_matched_complete_evidence():
    base=Path('cases/turbulent_combustion/runs/coherence_adaptive_repair/scale_alignment_followup_20261125/assessment')
    paths=[base/'TRAIN800_GroupedScaleEstimator_20261125_v1/query_estimator.json',base/'Validation100_GroupedScaleEstimator_20261125_v1/query_estimator.json']
    if not all(p.is_file() for p in paths):pytest.skip('completed grouped query evidence unavailable')
    train,validation=(json.loads(p.read_text()) for p in paths)
    text=report.grouped_query_text(train,validation,train['original_frozen_train_scale_normalizer'])
    assert text.count('all negative')==3 and text.count('all positive')==3
    assert '[-0.0733, -0.0712]' in text and '[+0.2234, +0.3051]' in text
    assert 'J4/T256' in text and 'variable-observation TRAIN law' in text
