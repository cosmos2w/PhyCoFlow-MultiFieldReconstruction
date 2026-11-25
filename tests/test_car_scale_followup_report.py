import hashlib
import json
from pathlib import Path

import pytest

from scripts.analysis import report_car_scale_followup as report


@pytest.fixture
def relocated_campaign():
    old=Path('/data2/wanglz/PhyCoFlow-MultiFieldReconstruction/cases/turbulent_combustion/runs/coherence_adaptive_repair/scale_alignment_followup_20261125')
    new=old.parent/'round_6'/old.name
    report.configure_relocation({'from':str(old),'to':str(new)})
    try:
        yield new
    finally:
        report.configure_relocation(None)


def binding(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def test_real_completed_diagnostic_text(relocated_campaign):
    path = relocated_campaign/'assessment/ScaleAlignment_CompletedDiagnostics_20261125_v1/readout.json'
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


def test_completed_diagnostic_detail_uses_paired_evidence(relocated_campaign):
    base=relocated_campaign/'assessment'
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


def test_grouped_query_replays_use_matched_complete_evidence(relocated_campaign):
    base=relocated_campaign/'assessment'
    paths=[base/'TRAIN800_GroupedScaleEstimator_20261125_v1/query_estimator.json',base/'Validation100_GroupedScaleEstimator_20261125_v1/query_estimator.json']
    if not all(p.is_file() for p in paths):pytest.skip('completed grouped query evidence unavailable')
    train,validation=(json.loads(p.read_text()) for p in paths)
    text=report.grouped_query_text(train,validation,train['original_frozen_train_scale_normalizer'])
    assert text.count('all negative')==3 and text.count('all positive')==3
    assert '[-0.0733, -0.0712]' in text and '[+0.2234, +0.3051]' in text
    assert 'J4/T256' in text and 'variable-observation TRAIN law' in text


def test_phase100_review_uses_completed_matched_evidence_and_rejects_mismatch(monkeypatch,relocated_campaign):
    import copy
    base=relocated_campaign/'operations/decisions/CandidatePhase100_JointReview_Continue500_20261125_v1/phase100_joint_review.json'
    if not base.is_file():pytest.skip('completed phase100 review unavailable')
    ref=binding(base)
    assert ref['sha256']=='3d6f23b93eb0d76799ab79161269b7a90431562830275a8a1866fb3b36038be0'
    review=json.loads(base.read_text());start=report.doc(review['inputs']['common_start_assessment']);control=report.doc(review['inputs']['control_assessment'])
    identity=report.doc(review['inputs']['candidate_fork_identity']);spec=report.doc(identity['spec'])
    checked=report.phase100_review_data(ref,start,control,spec)
    text='\n'.join(report.phase100_review_text(checked,start['field_order']))
    assert '+0.3275' in text and '+0.1040' in text and '+7.745' in text
    assert 'Same-age control' in text and 'paired condition bootstraps' in text and 'continue' in text.lower()
    body=['## Diagnostic evidence','Detailed diagnostic','Grouped query','Reference scope','## Matched learning outcome','Final outcome']
    report.place_phase100_section(body,checked,start['field_order'],500)
    assert body.index('Reference scope') < body.index('## Matched phase-100 continuation review') < body.index('## Selected phase-500 outcome and unequal-exposure control') < body.index('Final outcome')
    body225=['## Diagnostic evidence','## Matched learning outcome']
    report.place_phase100_section(body225,checked,start['field_order'],225)
    assert body225.index('## Matched phase-100 continuation review') < body225.index('## Selected phase-225 outcome and unequal-exposure control')
    bad=copy.deepcopy(review);bad['phase_epoch']=500
    old=report.doc
    monkeypatch.setattr(report,'doc',lambda x: bad if x is ref else old(x))
    with pytest.raises(ValueError,match='matched phase100 continuation review'):report.phase100_review_data(ref,start,control,spec)
    monkeypatch.setattr(report,'doc',old)
    altered=copy.deepcopy(spec);altered['run']+='-other'
    with pytest.raises(ValueError,match='same fork'):report.phase100_review_data(ref,start,control,altered)


def test_phase500_requires_phase100_review(monkeypatch):
    docs={'diag':{},'decision':{},'cost':{},'start':{},'aligned':{}}
    monkeypatch.setattr(report,'doc',lambda ref:docs[ref])
    inputs={'schema':'car_scale_followup_report_inputs_v1','phase':500,'diagnostic_readout':'diag','decision':'decision','costs':'cost','common_start_assessment':'start','aligned_scale':'aligned'}
    with pytest.raises(ValueError,match='phase500 report requires'):report.validate(inputs)
    inputs['phase']=225
    with pytest.raises(ValueError,match='phase225 report requires'):report.validate(inputs)


def test_relocated_campaign_evidence_and_bank_rows(tmp_path):
    import shutil
    import torch
    from scripts.analysis.report_car_round5 import rows_for_case
    old=tmp_path/'scale_alignment_followup_20261125'
    new=tmp_path/'round_6'/old.name
    source=tmp_path/'source_bank';source.mkdir()
    bank=old/'bank';bank.mkdir(parents=True)
    row={name:torch.tensor([1.]) for name in ('truth_model','coordinates_model','coordinates_raw','native_masses','normalizer_offset','normalizer_scale','obs_indices','obs_values','obs_fields')}
    row.update(sample_id='condition_1',sample_index=1,member_seeds=[7,8,9,10])
    torch.save(row,source/'0000.pt');torch.save(row,bank/'0000.pt')
    src_receipt=source/'receipt.json';src_receipt.write_text(json.dumps({'files_sha256':[hashlib.sha256((source/'0000.pt').read_bytes()).hexdigest()]}))
    child_receipt=bank/'receipt.json';child_receipt.write_text(json.dumps({'files_sha256':[hashlib.sha256((bank/'0000.pt').read_bytes()).hexdigest()]}))
    src_ref=binding(src_receipt);child_ref=binding(child_receipt)
    new.parent.mkdir();shutil.move(str(old),str(new));assert not old.exists()
    report.configure_relocation({'from':str(old),'to':str(new)})
    try:
        assert report.bound(src_ref)==src_receipt
        mapped=report.bound(child_ref);assert mapped==new/'bank/receipt.json'
        src,children=rows_for_case(report.bound(src_ref),report.doc(src_ref),{'candidate':(mapped,report.doc(child_ref))},0,'condition_1')
        assert src['member_seeds']==children['candidate']['member_seeds']
        mapped.write_text('{}')
        with pytest.raises(ValueError,match='evidence changed'):report.doc(child_ref)
        with pytest.raises(ValueError,match='relocation must map'):report.configure_relocation({'from':str(old),'to':str(tmp_path/'elsewhere')})
    finally:report.configure_relocation(None)


def test_selected_phase225_source_tables_use_saved_intervals(relocated_campaign):
    manifest=relocated_campaign/'configs/report_phase225_final_inputs_20261125_v1.json'
    if not manifest.is_file():pytest.skip('completed phase225 report inputs unavailable')
    inputs=json.loads(manifest.read_text());endpoint=report.doc(inputs['candidate_endpoint_assessment']);aligned=report.doc(inputs['aligned_scale'])
    text='\n'.join(report.selected_source_text(endpoint,aligned,inputs['aligned_model_keys']['candidate']))
    assert '+0.19946673 [+0.07184266, +0.32063100]' in text
    assert '+0.29046679 [-0.04936431, +0.67767068]' in text
    assert 'reused VALIDATION100' in text
