from copy import deepcopy

from scripts.qualification.assess_car_round5_estimator import assess, load_action_cost, sha


def gram(n):
    values = []
    for i in range(n):
        offset = i - (n - 1) / 2
        values.extend(([10., offset], [10., offset * .1]))
    return [[sum(a*b for a,b in zip(left,right)) for right in values] for left in values]


def receipts():
    endpoint = {'schema':'car_round5_endpoint_query_replay_v1', 'status':'completed',
        'sample_count':800, 'completed_conditions':800, 'seeds':8, 'panel_sha256':'panel',
        'units_sha256':'units', 'config_sha256':'config', 'gradient_gram':gram(8), 'scores':[1.]*16}
    parameter = {'schema':'car_round5_parameter_query_replay_v1', 'status':'completed',
        'sample_count':800, 'completed_conditions':800, 'seeds':4, 'source_panel_sha256':'panel',
        'source_checkpoint_sha256':'checkpoint', 'units_sha256':'units', 'config_sha256':'config',
        'shape':{'gradient_gram':gram(4)}, 'direct_scalar':{'gradient_gram':gram(4)}}
    context = {'schema':'car_round5_context_parity_v1', 'status':'passed', 'device':'cuda:0',
        'neighbor_backend':'keops', 'steps':4, 'checkpointing':True, 'duplicated_sensor_coordinates':True,
        'observation_mode':'endpoint_smooth', 'final_clamp':True,
        'endpoint':{'passed':True}, 'gradients':{'passed':True}, 'parameters_after_action':{'passed':True}}
    arm = {'device':'cuda:0', 'proposals':8, 'accepted':7, 'descending':6,
        'median_same_query_decrease':1., 'median_independent_query_decrease':.5,
        'full_attempted_epoch_seconds':100.}
    action = {'schema':'car_round5_estimator_action_cost_v1', 'status':'completed', 'sample_count':800,
        'matched_inputs':True, 'source_panel_sha256':'panel', 'checkpoint_sha256':'checkpoint',
        'units_sha256':'units', 'device':'cuda:0', '_artifacts_verified':True,
        'action_artifact':{'path':'actions.jsonl','sha256':'hash'},
        'timing_artifact':{'path':'epoch.json','sha256':'hash'}, 'shared':deepcopy(arm), 'grouped':deepcopy(arm)}
    return endpoint, parameter, context, action


def test_group_selection_requires_all_evidence_and_reports_variation():
    endpoint, parameter, context, action = receipts()
    result = assess(endpoint, parameter, context, action)
    assert result['query_law'] == 'condition_groups_8'
    assert abs(result['shape_parameter_gradient']['centered_sd_ratio'] - .1) < 1e-8
    assert abs(result['shape_parameter_gradient']['centered_variance_ratio'] - .01) < 1e-8
    assert result['context'] == 'static_features_synthetic_cuda_parity_passed'
    assert 'not measured' in result['context_speed']
    assert all(result['gates'].values())
    for missing in (None, {**endpoint, 'status':'forecast_exceeds_budget'}):
        assert assess(missing, parameter, context, action)['query_law'] == 'shared'
    assert assess(endpoint, {**parameter, 'completed_conditions':64}, context, action)['query_law'] == 'shared'
    assert assess(endpoint, parameter, context, None)['query_law'] == 'shared'


def test_reject_identity_cost_action_and_direction_failures():
    endpoint, parameter, context, action = receipts()
    assert assess({**endpoint, 'panel_sha256':'other'}, parameter, context, action)['query_law'] == 'shared'
    bad = deepcopy(action)
    bad['grouped']['full_attempted_epoch_seconds'] = 111.
    assert assess(endpoint, parameter, context, bad)['query_law'] == 'shared'
    bad = deepcopy(action)
    bad['grouped']['median_independent_query_decrease'] = -.1
    assert assess(endpoint, parameter, context, bad)['query_law'] == 'shared'
    bad_param = deepcopy(parameter)
    vectors = [[10., -1.], [10., 3.], [10., 1.], [10., 5.], [10., -1.], [10., 3.], [10., 1.], [10., 5.]]
    bad_param['shape']['gradient_gram'] = [[sum(a*b for a,b in zip(x,y)) for y in vectors] for x in vectors]
    assert assess(endpoint, bad_param, context, action)['query_law'] == 'shared'
    assert assess(endpoint, parameter, {**context, 'device':'cpu'}, action)['context'] == 'unqualified'


def test_variance_gate_accepts_reduction_without_twenty_percent_sd_reduction():
    import math
    endpoint, parameter, context, action = receipts()
    vectors = []
    for i in range(4):
        offset = i - 1.5
        vectors.extend(([10., offset], [10., offset * math.sqrt(.75)]))
    parameter['shape']['gradient_gram'] = [[sum(a*b for a,b in zip(x,y)) for y in vectors] for x in vectors]
    result = assess(endpoint, parameter, context, action)
    assert abs(result['shape_parameter_gradient']['centered_variance_ratio'] - .75) < 1e-8
    assert result['shape_parameter_gradient']['centered_sd_ratio'] > .8
    assert result['gates']['shape_parameter_variance_reduced_at_least_20_percent']
    assert result['query_law'] == 'condition_groups_8'


def test_action_artifact_hashes_must_resolve(tmp_path):
    import json
    endpoint, parameter, context, action = receipts()
    for name, key in (('actions.jsonl', 'action_artifact'), ('epoch.json', 'timing_artifact')):
        path = tmp_path / name
        path.write_text('evidence\n')
        action[key] = {'path':name, 'sha256':sha(path)}
    receipt = tmp_path / 'receipt.json'
    receipt.write_text(json.dumps(action))
    assert assess(endpoint, parameter, context, load_action_cost(receipt))['query_law'] == 'condition_groups_8'
    (tmp_path / 'epoch.json').write_text('changed\n')
    assert assess(endpoint, parameter, context, load_action_cost(receipt))['query_law'] == 'shared'
