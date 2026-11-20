"""Round 3 reports expose missing evidence and render saved evidence only."""
import json

import pytest

from scripts.analysis import report_car_round3 as report


def write_ref(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) if not isinstance(data, str) else data)
    return {'path':str(path), 'sha256':report.digest(path)}


def assessment(epoch, tasks=False):
    risk = {'member_mse':[1.,2.], 'signed_mean_risk':[.8,1.8]}
    fidelity = {'field_difference':[0.,0.], 'field_allowance':[.1,.2], 'primary_field_pass':True, 'sensor_pass':True, 'legacy_member_pass':True}
    candidates = {}
    for arm in report.ARMS:
        row = {'risk':risk, 'fidelity':fidelity, 'normalized_core_differences':{'state':-.06,'scale':0.,'shape':0.}}
        if tasks:
            row['reused_tasks'] = {'event_normalized_difference':0., 'regional_normalized_improvement':.1}
        candidates[arm] = row
    return {'schema':'car_round3_matched_validation_v1', 'epoch':epoch, 'split':'validation', 'members':4,
        'sample_ids':[f'sample-{i}' for i in range(100)], 'panel_sha256':'panel', 'source_bank_sha256':'source',
        'contract_sha256':'contract', 'core_sha256':'core', 'source_checkpoint_sha256':'source-checkpoint',
        'field_order':['u','v'], 'source':{'risk':risk}, 'candidates':candidates,
        'task_scope':'reused development tasks at endpoint' if tasks else 'both tasks excluded',
        'checkpoint_sha256':dict.fromkeys(report.ARMS,'checkpoint'),
        **({'predeclared_recommendation':report.ARMS[0]} if tasks else {})}


def inputs_for(tmp_path, complete, endpoint_epoch=100):
    campaign = tmp_path/'campaign'; (campaign/'reports').mkdir(parents=True)
    inputs = {'schema':'car_round3_report_inputs_v1', 'review':write_ref(tmp_path/'review.json',assessment(100)), 'histories':{}}
    history = []
    for epoch in range(1,(endpoint_epoch if complete else 100)+1):
        row={'epoch':epoch, 'data_loss':.2}
        for family,_ in report.FAMILIES: row[f'coherence_family/{family}/weighted_contribution'] = .3
        history.append(json.dumps(row))
    for arm in report.ARMS:
        inputs['histories'][arm] = write_ref(tmp_path/f'{arm}.jsonl','\n'.join(history)+'\n')
    if complete:
        endpoint = assessment(endpoint_epoch,True)
        if endpoint_epoch == 100:
            endpoint.update(task_free_review_sha256=inputs['review']['sha256'], terminal_decision_sha256='terminal')
        inputs['endpoint']=write_ref(tmp_path/'endpoint.json',endpoint)
        decision={'schema':'car_round3_final_decision_v1','epoch':endpoint_epoch,'assessment_sha256':inputs['endpoint']['sha256'],
                  'recommendation':report.ARMS[0], 'reason':'The matched endpoint meets the declared criteria.',
                  'arm_dispositions':{report.ARMS[0]:'Adopt.',report.ARMS[1]:'Close.'}}
        inputs['decision']=write_ref(tmp_path/'decision.json',decision)
        inputs['qualification']=write_ref(tmp_path/'qualification.json',{'status':'passed'})
        inputs['exact_compute']=write_ref(tmp_path/'exact.json',{'schema':'car_round3_exact_compute_disposition_v1',
            'disposition':'rejected','reason':'No meaningful timing improvement.', 'baseline_epoch_seconds':115.,'reused_epoch_seconds':114.})
        v={'physical_weighted_total':{'mean':1.,'sample_sd':.1}}
        seeds = [{'seed':i, 'shared':{'scores':{'physical_weighted_total':1.+i*.1}},
                  'condition_specific':{'scores':{'physical_weighted_total':1.+i*.02}}} for i in (1,2)]
        inputs['shape']=write_ref(tmp_path/'shape.json',{'schema':'car_round3_shape_query_diagnostic_v1','seeds':seeds,
            'shared_dispersion':v,'condition_specific_dispersion':v,'scorer_worker_seconds':100.,
            'sample_count':800,'members':[0,1],'elapsed_worker_seconds':110.,'worker_seconds_limit':200.,
            'prior_conservative_worker_seconds':2812.,'accounted_upper_bound_worker_seconds':2922.,
            'total_worker_seconds_ceiling':7200.})
        inputs['costs']=write_ref(tmp_path/'cost.json',{'schema':'car_round3_costs_v1',
            'training':{arm:{'seconds_per_epoch':110.,'occupied_gpu_seconds':11000.} for arm in report.ARMS},
            'assessment':{'gpu_seconds':200.,'cpu_seconds':300.,'wall_seconds':180.},'failed_attempt_gpu_seconds':None})
    input_path=tmp_path/'inputs.json'; input_path.write_text(json.dumps(inputs))
    return campaign,input_path


def test_incomplete_report_does_not_claim_completion(tmp_path):
    campaign,inputs=inputs_for(tmp_path,False)
    result=report.produce(inputs,campaign,campaign/'reports/incomplete')
    assert result['status']=='incomplete'
    assert 'matched endpoint assessment' in result['pending']
    text=(campaign/'reports/incomplete/report.md').read_text()
    assert '**Status:** INCOMPLETE' in text
    assert 'Both event and regional task scores were excluded' in text
    assert 'RawRisk uses unit risk scales and a dual step of 0.01' in text
    assert 'max(TRAIN SOURCE member MSE, 1e-8) risk scales and a dual step of 0.0001' in text
    assert 'identical fixed allowances and the same feasible set' in text
    assert 'does not causally isolate denominator scaling' in text
    assert '![Sampled TRAIN' in text and '[PDF master]' in text
    assert 'Fixed TRAIN allowance' in text
    assert '| u | 1 | 1 | 1 | 0 | 0 | 0.1 |' in text
    assert 'max-width: 100%' in (campaign/'reports/incomplete/report.html').read_text()


@pytest.mark.parametrize("endpoint_epoch", [100, 500])
def test_complete_report_renders_figures_and_rejects_evidence_drift(tmp_path, endpoint_epoch):
    campaign,inputs=inputs_for(tmp_path,True,endpoint_epoch)
    result=report.produce(inputs,campaign,campaign/'reports/complete')
    assert result['status']=='completed' and result['pending']==[]
    output=campaign/'reports/complete'
    assert (output/'report.html').is_file()
    assert len(result['figures'])==3
    text = (output/'report.md').read_text()
    assert 'figures/shape_query_variation.png' in text and 'figures/shape_query_variation.pdf' in text
    assert 'sampled query variation, not learning or gradient variance' in text
    assert '800 cached TRAIN conditions' in text and 'SOURCE member pair [0, 1]' in text
    assert 'recovery-attempt scorer worker time 100 s' in text
    assert 'recovery-attempt elapsed worker time 110 s' in text
    assert 'recovery-attempt worker limit 200 s' in text
    assert 'prior-attempt conservative worker bound 2812 s' in text
    assert 'all-attempt accounted worker upper bound 2922 s' in text
    assert 'total worker ceiling 7200 s' in text
    assert 'Exact prior-attempt CPU costs are unavailable' in text
    for figure in result['figures']:
        assert (output/'figures'/f'{figure}.png').is_file()
        assert (output/'figures'/f'{figure}.pdf').is_file()
    (tmp_path/'endpoint.json').write_text('{}')
    with pytest.raises(ValueError, match='hash mismatch'):
        report.produce(inputs,campaign,campaign/'reports/invalid')


def test_terminal_inconclusive_shape_disposition_closes_diagnostic(tmp_path):
    campaign, inputs_path = inputs_for(tmp_path, True)
    inputs = json.loads(inputs_path.read_text())
    shape = {'schema':'car_round3_shape_query_diagnostic_v1', 'seeds':[],
             'disposition':'budget-exhausted', 'reason':'Only one paired replay fit the budget.',
             'conclusion':'The scalar variance comparison is inconclusive.'}
    inputs['shape'] = write_ref(tmp_path/'shape_terminal.json', shape)
    inputs_path.write_text(json.dumps(inputs))
    result = report.produce(inputs_path, campaign, campaign/'reports/terminal_shape')
    assert result['status'] == 'completed'
    text = (campaign/'reports/terminal_shape/report.md').read_text()
    assert 'closed as budget-exhausted' in text
    assert 'The scalar variance comparison is inconclusive.' in text
    assert 'Only one paired replay fit the budget.' in text
    shape['reason'] = ''
    assert not report.shape_evidence_complete(shape)
    shape['reason'] = None
    assert not report.shape_evidence_complete(shape)


def test_shape_conclusion_is_shown_with_dispersion(tmp_path):
    campaign, inputs_path = inputs_for(tmp_path, True)
    inputs = json.loads(inputs_path.read_text())
    shape = json.loads((tmp_path/'shape.json').read_text())
    shape.update(disposition='inconclusive', reason='The replicates do not resolve the comparison.',
                 conclusion='The result is inconclusive.')
    inputs['shape'] = write_ref(tmp_path/'shape_with_conclusion.json', shape)
    inputs_path.write_text(json.dumps(inputs))
    report.produce(inputs_path, campaign, campaign/'reports/shape_conclusion')
    text = (campaign/'reports/shape_conclusion/report.md').read_text()
    assert 'shared-query mean' in text
    assert 'The result is inconclusive.' in text
    assert 'The replicates do not resolve the comparison.' in text


def test_optional_campaign_gpu_accounting_and_device_labels(tmp_path):
    campaign, inputs_path = inputs_for(tmp_path, True)
    inputs = json.loads(inputs_path.read_text())
    costs = json.loads((tmp_path/'cost.json').read_text())
    costs['training'][report.ARMS[0]]['runtime_device'] = 'RTX device 0'
    costs['training'][report.ARMS[0]].update(update_work_seconds=8000.,
        data_wait_and_loop_overhead_seconds=2000., callback_and_reporting_seconds=1000.)
    costs['training'][report.ARMS[1]]['runtime_device'] = 'RTX device 1'
    costs['campaign_gpu_accounting'] = {
        'setup_upper_bound_gpu_seconds': 500.,
        'measured_training_gpu_seconds': 22000.,
        'measured_assessment_gpu_seconds': 200.,
        'accounted_upper_bound_gpu_seconds': 22700.,
    }
    inputs['costs'] = write_ref(tmp_path/'cost_with_accounting.json', costs)
    inputs_path.write_text(json.dumps(inputs))
    report.produce(inputs_path, campaign, campaign/'reports/accounting')
    text = (campaign/'reports/accounting/report.md').read_text()
    assert 'Raw-risk control on RTX device 0' in text
    assert 'SOURCE-scaled feedback on RTX device 1' in text
    assert 'update work: 8000 s; data waits plus loop overhead: 2000 s; callbacks and reporting: 1000 s' in text
    assert 'Pure data-wait time was not separately measured.' in text
    assert 'SOURCE-scaled feedback recorded timing:' not in text
    assert 'against the 45 GPU-hour ceiling' in text
    assert 'conservative setup and qualification upper bound: 500 GPU s' in text
    assert 'measured training occupancy: 22000 GPU s' in text
    assert 'measured assessment occupancy: 200 GPU s' in text
    assert 'supplied total campaign upper bound: 22700 GPU s' in text
    assert 'other disjoint measured occupancy:' not in text
    assert 'parallel speedup' not in text


def test_paired_shape_figure_uses_recorded_values_and_rejects_duplicate_seeds():
    data = {'seeds': [
        {'seed':7, 'shared':{'scores':{'physical_weighted_total':1.2}},
         'condition_specific':{'scores':{'physical_weighted_total':1.4}}},
        {'seed':8, 'shared':{'scores':{'physical_weighted_total':1.6}},
         'condition_specific':{'scores':{'physical_weighted_total':1.3}}},
    ]}
    assert report.paired_shape_values(data) == [(7,1.2,1.4), (8,1.6,1.3)]
    data['seeds'][1]['seed'] = 7
    with pytest.raises(ValueError, match='unique'):
        report.paired_shape_values(data)


def test_labelled_native_and_sensor_tables_preserve_field_order(tmp_path):
    campaign, inputs_path = inputs_for(tmp_path, False)
    inputs = json.loads(inputs_path.read_text())
    result = json.loads((tmp_path/'review.json').read_text())
    rows = [result['source'], *(result['candidates'][arm] for arm in report.ARMS)]
    for i, row in enumerate(rows):
        row['risk'] = dict(row['risk'])
        row['risk']['ensemble_mean_mse'] = [11.+i, 21.+i]
        row['risk']['unbiased_member_variance'] = [31.+i, 41.+i]
        row['sensor_mse'] = [None, .1+i*.1]
    inputs['review'] = write_ref(tmp_path/'review_tables.json', result)
    inputs_path.write_text(json.dumps(inputs))
    report.produce(inputs_path, campaign, campaign/'reports/labelled')
    text = (campaign/'reports/labelled/report.md').read_text()
    assert '| u | 11 | 12 | 13 | 31 | 32 | 33 |' in text
    assert '| v | 21 | 22 | 23 | 41 | 42 | 43 |' in text
    assert '| v | 0.1 | 0.2 | 0.3 |' in text
    assert '| u | unavailable | unavailable | unavailable |' not in text
    assert "{'state':" not in text


def test_paired_endpoint_effects_are_labelled_and_field_ordered(tmp_path):
    campaign, inputs_path = inputs_for(tmp_path, True)
    inputs = json.loads(inputs_path.read_text())
    endpoint = json.loads((tmp_path/'endpoint.json').read_text())
    endpoint['paired_scaled_minus_raw'] = {'regional_normalized_improvement':.07, 'event_score':-.03,
        'normalized_core_differences':{'state':.01, 'scale':.02, 'shape':-.03},
        'signed_mean_risk':[4.,5.], 'member_mse':[6.,7.]}
    inputs['endpoint'] = write_ref(tmp_path/'endpoint_paired.json', endpoint)
    decision = json.loads((tmp_path/'decision.json').read_text())
    decision['assessment_sha256'] = inputs['endpoint']['sha256']
    inputs['decision'] = write_ref(tmp_path/'decision_paired.json', decision)
    inputs_path.write_text(json.dumps(inputs))
    report.produce(inputs_path, campaign, campaign/'reports/paired')
    text = (campaign/'reports/paired/report.md').read_text()
    assert 'regional normalized improvement 0.07; event score -0.03' in text
    assert '| Scaled minus raw | 0.01 | 0.02 | -0.03 |' in text
    assert '| u | 4 | 6 |' in text and '| v | 5 | 7 |' in text
    assert 'Both event and regional task scores were excluded from the learning review.' not in text
