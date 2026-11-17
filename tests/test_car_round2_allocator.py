"""Exact estimator checks against enumeration of the actual duplicate rule."""
import importlib.util
import itertools
from pathlib import Path

import pytest
import torch

path = Path(__file__).resolve().parents[1] / 'scripts/benchmarks/profile_car_round2_allocator.py'
spec = importlib.util.spec_from_file_location('round2_allocator', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


@pytest.mark.parametrize('stratum', ['pair', 'self'])
@pytest.mark.parametrize('exhaustive', [False, True])
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
def test_joint_family_isolation_matches_real_family_values_and_gradients(stratum, exhaustive, dtype):
    from phycoflow_reconstruction.coherence.adaptive_profile import isolate_query_stratum
    from phycoflow_reconstruction.coherence.query_sampling import (
        LINE_WEIGHTS,
        Draw,
        QueryBatch,
        full_target_query_batch,
    )
    from phycoflow_reconstruction.coherence.scale_compute import sampled_scale
    from phycoflow_reconstruction.coherence.shape_compute import LazyPooledShape

    torch.manual_seed(610)
    c = 3
    head = torch.nn.Parameter(torch.randn(c, 4, 4, dtype=dtype) * .2)
    prediction = torch.randn(1, c, 4, 4, dtype=dtype) + head.unsqueeze(0)
    truth = torch.randn_like(prediction)
    shape = LazyPooledShape([0] * c, [1] * c, dimensions=(0,), projections=3)
    full = full_target_query_batch(c)
    shape.accumulate(prediction.detach(), truth, full)
    shape.freeze(provenance={'split': 'train', 'sample_ids': ['synthetic'], 'source_checkpoint_sha256': 'a' * 64})
    spectral = prediction.flatten(2)[..., :4].transpose(1, 2).unsqueeze(0)
    target = truth.flatten(2)[..., :4].transpose(1, 2)
    desc = prediction.flatten(2)[..., :2].reshape(1, 1, 2*c)
    target_desc = truth.flatten(2)[..., :2].reshape(1, 2*c)
    mean_kw = {'draw_piece_means': prediction.mean((-1, -2)).unsqueeze(0).unsqueeze(2),
               'target_piece_means': truth.mean((-1, -2)).unsqueeze(1),
               'piece_weights': torch.ones(1, dtype=dtype),
               'mean_scales': torch.ones(1, c, dtype=dtype)}
    query = full if exhaustive else QueryBatch(
        (Draw(0, 1, 1/c, c, 1, (1,), None, .5),),
        (Draw(0, 1, 1/3, 3, 1, (1, 1), 0, LINE_WEIGHTS[0]/4),), 1)
    calls = []

    def score_both(batch):
        calls.append(batch)
        scale = sampled_scale(spectral, target, batch, filters=torch.ones(2, 4, dtype=dtype),
            scales=torch.ones(2, c, dtype=dtype), mass=torch.ones(4, dtype=dtype),
            power_floor=torch.ones(2, c, dtype=dtype), band_weights=torch.tensor([.5, .5], dtype=dtype),
            self_descriptors=desc, target_self_descriptor=target_desc, mean_kwargs=mean_kw, mode='paired')
        topology = shape.score(prediction, None, truth, batch, exhaustive=exhaustive, mode='paired')
        anchor = prediction.square().mean() * .31
        return ([1.7 * (scale['signed'] + scale['rho'] + scale['joint'] + scale['log_power_self']).mean() + anchor,
                 .6 * (topology['self'] + topology['mutual']).mean() + anchor], [scale, topology])

    old = [isolate_query_stratum(lambda draw, j=j: score_both(draw)[0][j], query, stratum) for j in range(2)]
    assert len(calls) == 4
    calls.clear()
    new, leaves = mod.isolate_family_query_stratum(score_both, query, stratum)
    assert len(calls) == 2 and calls[1] is query
    assert len(leaves) == 2 and 'signed' in leaves[0] and 'mutual' in leaves[1]
    for original, optimized in zip(old, new):
        torch.testing.assert_close(optimized, original, rtol=0, atol=0)
        old_grad = torch.autograd.grad(original, head, retain_graph=True)[0]
        new_grad = torch.autograd.grad(optimized, head, retain_graph=True)[0]
        torch.testing.assert_close(new_grad, old_grad, rtol=0, atol=0)


def test_shared_inner_mse_matches_exhaustive_correlated_duplicates():
    # Four scalar parameter gradients: two outer IDs, each with two inner outcomes.
    vectors = torch.tensor([[1., -2.], [3., 1.], [-1., 2.], [2., 4.]], dtype=torch.float64)
    gram = vectors @ vectors.T
    proposal = [0.7, 0.3]
    inner = [0.25, 0.75]
    draws = 3
    target = sum(inner[j] * vectors[2*i+j] / 2 for i in range(2) for j in range(2))
    exact = 0.
    for outer_draws in itertools.product(range(2), repeat=draws):
        p_outer = 1.
        for item in outer_draws:
            p_outer *= proposal[item]
        counts = [outer_draws.count(i) for i in range(2)]
        for inner_draws in itertools.product(range(2), repeat=2):
            probability = p_outer * inner[inner_draws[0]] * inner[inner_draws[1]]
            estimate = sum(counts[i] / (draws * 2 * proposal[i]) * vectors[2*i+inner_draws[i]] for i in range(2))
            exact += probability * float((estimate-target).square().sum())
    assert mod.shared_inner_mse(gram, 2, inner, proposal, draws) == pytest.approx(exact, rel=1e-12, abs=1e-12)


def test_source_panel_guard_uses_producer_tensor_digest():
    import ast

    from phycoflow_reconstruction.training.scalable_fitting import _digest as producer_digest

    tree = ast.parse(path.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    digest_imports = [node for node in ast.walk(main) if isinstance(node, ast.ImportFrom)
                      and any(alias.name == '_digest' for alias in node.names)]
    assert len(digest_imports) == 1
    scope = {alias.asname or alias.name: getattr(importlib.import_module(node.module), alias.name)
             for node in digest_imports for alias in node.names}
    assert scope['_digest'] is producer_digest
    guard = next(node.test for node in ast.walk(main) if isinstance(node, ast.If)
                 and "panel['parameter_buffer_sha256']" in ast.unparse(node.test))
    guard_code = compile(ast.Expression(guard), str(path), 'eval')
    model = torch.nn.BatchNorm1d(3)
    panel = {'parameter_buffer_sha256': producer_digest(model.state_dict()), 'source_weight_view': 'source'}
    scope.update(model=model, panel=panel, source_identity={'weight_view': 'source'})
    assert not eval(guard_code, scope)
    with torch.no_grad():
        model.weight[0].add_(1)
    assert eval(guard_code, scope)
    with torch.no_grad():
        model.weight[0].sub_(1)
        model.running_mean[0].add_(1)
    assert eval(guard_code, scope)


def test_profiler_accepts_adapter_default_detached_difficulty(monkeypatch):
    import ast

    from phycoflow_reconstruction.coherence import query_runtime

    leaf = torch.tensor([2., 4.], requires_grad=True)
    monkeypatch.setattr(query_runtime, 'score', lambda *args, **kwargs: (None, {}, {'pair': {0: leaf}, 'self': {}}))
    adapter = object.__new__(query_runtime.ScalableQueryAdapter)
    adapter.families = {'multiscale_spectrum': object()}
    adapter.shape = None
    _, _, leaves = adapter.score('multiscale_spectrum', None, None, coordinates=None,
                                 context={}, batch=None, coefficients={})
    assert leaves['pair'][0] == 3.0 and isinstance(leaves['pair'][0], float)
    tree = ast.parse(path.read_text())
    assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                      and 'destination[pair, index]' in ast.unparse(node))
    destination = torch.zeros((1, 1), dtype=torch.float64)
    destination[0, 0] = eval(compile(ast.Expression(assignment.value), str(path), 'eval'),
                             {'pair': 0, 'index': 0, 'leaves': leaves})
    assert destination[0, 0].item() == 3.0


def test_unique_cost_counts_duplicate_pairs_once():
    # Three draws, the two possible IDs have occurrence probabilities 7/8.
    assert mod.expected_unique_cost([.5, .5], [2., 4.], 3) == pytest.approx(5.25)


def test_single_inner_draw_reduces_to_iid_outer_formula():
    from phycoflow_reconstruction.coherence.round2_qualification import importance_mse
    vectors = torch.tensor([[2., 1.], [-3., 4.], [1., -2.]], dtype=torch.float64)
    gram = vectors @ vectors.T
    proposal = [.2, .3, .5]
    expected = importance_mse(gram, [1/3]*3, proposal, 4)
    assert mod.shared_inner_mse(gram, 3, [1.], proposal, 4) == pytest.approx(expected)


def test_family_gradients_cannot_cancel():
    x = torch.tensor(1., requires_grad=True)
    gram, count = mod.family_gradient_gram([(x, -x), (2*x, -2*x)], [x])
    assert count == 1
    torch.testing.assert_close(gram, torch.tensor([[2., 4.], [4., 8.]], dtype=torch.float64))


def test_oracle_diagnostics_split_actual_parameter_width_and_count_mismatches():
    expected = torch.tensor([1., 0., 2., 0.], dtype=torch.float64)
    actual = torch.tensor([1. + 2e-5, 0., 2., float('nan')], dtype=torch.float64)
    report = mod.oracle_family_diagnostics(actual, expected, 2)
    assert report['scale']['tolerance_mismatch_count'] == 1
    assert report['scale']['finite_pairs'] == 2
    assert report['scale']['max_abs'] == pytest.approx(2e-5)
    assert report['shape']['finite_pairs'] == 1
    assert report['shape']['finite_weighted'] == 1
    assert report['shape']['finite_direct'] == 2
    assert report['shape']['tolerance_mismatch_count'] == 1
    with pytest.raises(ValueError, match='two complete'):
        mod.oracle_family_diagnostics(actual, expected, 1)


def test_c8_conditional_difficulty_uses_each_leaf_once():
    scale = torch.tensor([[[2., 4.], [0., -8.]], [[6., 2.], [4., 0.]]])
    shape = torch.tensor([[[0., 2.], [2., 2.]], [[2., 0.], [0., 2.]]])
    probabilities, difficulty = mod.c8_frozen_probabilities(scale, shape, [.25,.75], [.75,.25])
    # C8 clamps each family before averaging; the values are conditional on
    # outer ID and never divided again by outer or inner proposal probability.
    expected = torch.tensor([2.3125, 1.0625], dtype=torch.float64)
    torch.testing.assert_close(difficulty, expected)
    torch.testing.assert_close(probabilities, .25 + .5*expected/expected.sum())


def test_sampler_overhead_preserves_measured_components():
    report = mod.measure_sampler_overhead(['CH4','CO','T'], [1/3]*3, iterations=100, repeats=3)
    assert len(report['rows']) == 3
    assert all(row['uniform_seconds'] > 0 and row['fixed_seconds'] > 0 for row in report['rows'])
    assert report['charged_overhead_seconds'] == pytest.approx(max(0., report['mean_difference_seconds'] + 2*report['standard_error_seconds']))


def test_overhead_gate_uses_the_lower_observed_update_bound():
    lower = mod.provisional_overhead_fraction(120., .005, .5)
    upper = mod.provisional_overhead_fraction(120., .005, 1.)
    assert lower == pytest.approx(2*upper)


def test_epoch_timing_records_complete_update_and_callbacks(tmp_path, monkeypatch):
    import json

    from phycoflow_reconstruction.training.monitoring import TrainingMonitor

    monitor = TrainingMonitor(tmp_path, start_step=0, final_step=2, configured_steps=2,
                              steps_per_epoch=2, description='qualification', enabled=False)
    monkeypatch.setattr(monitor, '_plot', lambda: None)
    monitor.record({'step':1, 'total':1.})
    monitor.finish_step()
    monitor.record({'step':2, 'total':.5})
    monitor.finish_step()
    monitor.close()
    rows = [json.loads(line) for line in (tmp_path/'logs/epoch_timing.jsonl').read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]['accepted_updates'] == 2 and rows[0]['step'] == 2
    assert rows[0]['wall_seconds'] >= rows[0]['train_seconds'] > 0
    assert rows[0]['callback_and_reporting_seconds'] == pytest.approx(rows[0]['wall_seconds']-rows[0]['train_seconds'])


def test_timing_receipt_uses_qualified_complete_epochs(tmp_path):
    import json

    import yaml

    timing_path = Path(__file__).resolve().parents[1] / 'scripts/analysis/summarize_car_round2_timing.py'
    spec = importlib.util.spec_from_file_location('round2_timing', timing_path)
    timing = importlib.util.module_from_spec(spec); spec.loader.exec_module(timing)
    source = tmp_path/'source'; (source/'checkpoints').mkdir(parents=True)
    (source/'checkpoints/last.pt').write_bytes(b'source')
    run = tmp_path/'run'; (run/'checkpoints').mkdir(parents=True); (run/'logs').mkdir()
    (run/'checkpoints/last.pt').write_bytes(b'checkpoint')
    (run/'status.json').write_text(json.dumps({'status':'completed','global_step':144}))
    epoch_rows = [{'version':'complete_epoch_timing_v1','epoch':i,'step':i*12,'accepted_updates':12,
                   'train_seconds':float(i), 'wall_seconds':float(i)+.5,
                   'callback_and_reporting_seconds':.5} for i in range(1,13)]
    (run/'logs/epoch_timing.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in epoch_rows))
    config = tmp_path/'formal.yaml'
    config.write_text(yaml.safe_dump({'source_run':str(source),'residual':{'enabled':True},
                                      'optimization':{'batch_size':64,'steps_per_epoch':12,'epochs':1000}}))
    pause, resume = tmp_path/'pause.json',tmp_path/'resume.json'
    pause.write_text(json.dumps({'gpu_uuid':'GPU-actual','arm':'FFM_Selected_Bank_Residual_Repair', 'token':'pause', 'started_unix':0.}))
    resume.write_text(json.dumps({'gpu_uuid':'GPU-actual','arm':'FFM_Selected_Bank_Residual_Repair','resume':str(run), 'token':'resume', 'started_unix':100.}))
    pause_end, resume_end = tmp_path/'pause_end.json', tmp_path/'resume_end.json'
    pause_end.write_text(json.dumps({'token':'pause','finished_unix':80.,'status':'budget_paused','global_step':108,'run_dir':str(run)}))
    resume_end.write_text(json.dumps({'token':'resume','finished_unix':140.,'status':'completed','global_step':144,'run_dir':str(run)}))
    qualification = tmp_path/'qualification.json'
    qualification.write_text(json.dumps({'schema':'car_round2_qualification_v1','passed':True,
        'formal_config_sha256':timing.digest(config),'resumed':str(run),'pause_step':108,'final_step':144,
        'pause_launch_sha256':timing.digest(pause),'resume_launch_sha256':timing.digest(resume),
        'pause_completion_sha256':timing.digest(pause_end),
        'checkpoint_sha256':['uninterrupted',timing.digest(run/'checkpoints/last.pt')], 'arm':'FFM_Selected_Bank_Residual_Repair'}))
    result = timing.summarize(formal_config=config,run=run,qualification=qualification,
                              pause_launch=pause,resume_launch=resume,pause_completion=pause_end,resume_completion=resume_end)
    assert result['measured_updates'] == 96
    assert result['complete_update_seconds'] == pytest.approx(11/12)
    assert result['minimum_observed_steady_update_seconds'] == pytest.approx(2/12)
    assert result['callback_seconds_by_epoch'] == [.5]*12
    assert result['outside_epoch_seconds'] == pytest.approx(36.)
    epoch_rows[-1]['accepted_updates'] = 11
    (run/'logs/epoch_timing.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in epoch_rows))
    with pytest.raises(ValueError,match='exactly 12'):
        timing.summarize(formal_config=config,run=run,qualification=qualification,
                         pause_launch=pause,resume_launch=resume,pause_completion=pause_end,resume_completion=resume_end)


def test_research_costs_are_bounded_by_measured_receipts(tmp_path):
    import json

    from phycoflow_reconstruction.evaluation.adaptive_campaign import digest

    job = tmp_path/'job.json'; job.write_text(json.dumps({'wall_seconds':4.}))
    entry = {'path':str(job),'sha256':digest(job),'allocated_seconds':3.}
    costs = {'schema':'car_round2_allocator_research_costs_v1','jobs':[entry]}
    assert mod.measured_prior_research_seconds(costs,tmp_path) == 3.
    entry['allocated_seconds'] = 4.1
    with pytest.raises(ValueError,match='bounded'):
        mod.measured_prior_research_seconds(costs,tmp_path)
    with pytest.raises(ValueError,match='no-prior-work'):
        mod.measured_prior_research_seconds({'schema':costs['schema'],'jobs':[]},tmp_path)
