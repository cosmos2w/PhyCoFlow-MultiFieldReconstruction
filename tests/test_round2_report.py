"""Saved-data report contracts and native reconstruction annotations."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location('round2_report', ROOT / 'scripts/analysis/report_car_round2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deployment_memory_table_uses_selected_and_raw_values():
    report = load()
    selected = {'alpha': .5, 'deployment_memory': {'baseline_allocated_bytes': 2**20,
                                                     'peak_allocated_bytes': 3*2**20}}
    raw = {'alpha': 1., 'deployment_memory': {'baseline_allocated_bytes': 2*2**20,
                                               'peak_allocated_bytes': 5*2**20}}
    doc = {'selected_alpha': .5, 'rows': [selected, raw]}
    docs = {arm: {'validation': {1000: (None, doc)}} for arm in report.ARMS}
    table = '\n'.join(report.deployment_memory_table(docs))
    assert '| Nominated | 0.5 | 1.00 | 3.00 |' in table
    assert '| Raw | 1 | 2.00 | 5.00 |' in table
    assert 'SOURCE and child models resident' in table
    assert 'not isolated standalone' in table
    raw['deployment_memory'] = None
    assert '| Raw | 1 | unavailable | unavailable |' in '\n'.join(report.deployment_memory_table(docs))


def test_training_and_deployment_device_provenance_labels_cross_device_and_unknown():
    report = load()
    docs = {arm: {'validation': {1000: (None, {'gpu_uuid': 'deployment-gpu'})}} for arm in report.ARMS}
    full, residual = report.ARMS[:2]
    costs = {'ffm': {'complete': True, 'allocated_jobs': [
        {'kind': 'gpu', 'scope': 'training', 'weights': {full: 1}, 'gpu_uuid': 'training-gpu-0'},
        {'kind': 'gpu', 'scope': 'training', 'weights': {residual: 1}, 'gpu_uuid': 'training-gpu-1'}]},
        'senseiver': {'complete': False}}
    text = '\n'.join(report.device_provenance_lines(docs, costs))
    assert 'training-gpu-0' in text and 'training-gpu-1' in text
    assert 'FFM observed training clocks and attempts use different devices or multiple devices' in text
    assert 'FFM final deployment measurements use the same device' in text
    assert 'device identity is not evidence of a hardware speedup' in text
    assert 'SENSEIVER training device comparison: unavailable' in text
    costs['ffm']['complete'] = False
    text = '\n'.join(report.device_provenance_lines(docs, costs))
    assert 'FFM observed training clocks and attempts use different devices or multiple devices' in text
    assert 'FFM cost evidence is incomplete, so the observed training device coverage may be incomplete' in text
    docs[full]['validation'][1000][1].pop('gpu_uuid')
    assert 'FFM deployment device comparison: unavailable' in '\n'.join(report.device_provenance_lines(docs, costs))


def test_allocator_table_reports_saved_policy_costs_ranges_and_partial_groups():
    report = load()
    policies = {name: {'gradient_mse': mse, 'full_estimator_seconds': seconds,
                       'mse_cost': mse * seconds, 'descriptive_mse_cost_p05_p95': [1., 3.],
                       'pair_probabilities': [.25, .75], 'maximum_importance_weight': 2.}
                for name, mse, seconds in [('uniform', 1., 2.), ('c8_priorities', 2., 2.),
                                           ('lagged_gradient_cost', 1.5, 2.)]}
    group = {'group_ids': ['first', 'second'], 'families': {'scale': {'max_abs': .01, 'relative_l2': .02,
             'tolerance_mismatch_count': 0, 'elements': 5, 'finite_direct': 5, 'finite_weighted': 5, 'finite_pairs': 5}}}
    audit = {'policies': policies, 'oracle_diagnostics': [group, {**group, 'stratum': 'self'}],
             'actual_head_parameters': 5, 'prior_research_seconds': 100.,
             'audit_through_measurement_seconds': 50., 'save_margin_seconds': 10., 'amortization_updates': 12000}
    text = '\n'.join(report.allocation_audit_table(audit, {'command': ['--replicates', '128']}))
    assert '| lagged_gradient_cost | 1.5 | 2 | 3 | 1.5 | 1–3 | 0.25–0.75 | 2 |' in text
    assert '128 saved-dispatch draws' in text and 'not confidence intervals' in text
    assert 'Oracle group sizes: [2]' in text and 'residual-head parameters: 5' in text
    assert '(prior research 100 s + audit through measurement 50 s + safe-save allowance 10 s) / 12000' in text
    assert 'unavailable saved-dispatch draws' in '\n'.join(report.allocation_audit_table({}))


def test_allocator_table_marks_missing_and_invalid_oracle_metrics_unavailable():
    report = load()
    audit = {'oracle_diagnostics': [{'group_ids': ['sample'], 'families': {'scale': {
        'max_abs': True, 'relative_l2': float('nan'), 'elements': 4, 'finite_direct': 4}}}]}
    text = '\n'.join(report.allocation_audit_table(audit))
    assert 'coverage: unavailable' in text
    assert 'maximum tolerance mismatch count: unavailable' in text
    assert 'worst absolute error: unavailable' in text
    assert 'worst relative L2 error: unavailable' in text


def test_qualification_ledger_binds_sources_and_relative_markdown(tmp_path):
    report = load()
    directory = tmp_path / 'assessment'; directory.mkdir()
    source = directory / 'source.json'; source.write_text('{}')
    markdown = directory / 'ledger.md'; markdown.write_text('# Completed execution qualification\n')
    ledger = directory / 'ledger.json'
    ledger.write_text(json.dumps({'schema': 'car_round2_definition_qualification_ledger_v1',
        'execution_qualification_passed': True, 'markdown_path': 'assessment/ledger.md',
        'markdown_sha256': report.sha(markdown),
        'source_receipts': [{'path': 'assessment/source.json', 'sha256': report.sha(source)}]}))
    inputs = {'definition_qualification_ledger': 'assessment/ledger.json'}
    lines, bound_ledger, bound_markdown = report.qualification_ledger_lines(inputs, tmp_path, tmp_path / 'reports/final')
    assert 'qualification/ledger.md' in '\n'.join(lines) and 'pre-learning qualification' in '\n'.join(lines)
    assert bound_ledger == ledger and bound_markdown == markdown
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='source changed'):
        report.qualification_ledger_lines(inputs, tmp_path, tmp_path / 'reports/final')
    with pytest.raises(ValueError, match='ledger required'):
        report.qualification_ledger_lines({}, tmp_path, tmp_path / 'reports/final')


def test_adjacent_blocks_do_not_bridge_missing_frames_or_trajectories():
    report = load()
    ids = [f'first:{i}' for i in range(12)] + ['first:15', 'second:2']
    blocks = report.adjacent_blocks(ids, range(len(ids)))
    assert [(row['trajectory'], row['first'], row['last']) for row in blocks] == [
        ('first', 0, 9), ('first', 10, 11), ('first', 15, 15), ('second', 2, 2)]
    assert blocks[0]['mean'] == 4.5


def test_report_rejects_missing_four_mature_runs(tmp_path):
    report = load()
    with pytest.raises(ValueError, match='four named mature runs'):
        report.validate_inputs({'schema': 'car_round2_report_inputs_v1', 'runs': {}}, tmp_path)


def test_training_series_requires_actual_three_family_losses():
    report = load()
    history = [{'data_loss': 0., 'coherence_family/global_distribution/raw': 1.25,
                'coherence_family/multiscale_spectrum/raw': 2.5, 'coherence_family/topology/raw': 3.75}]
    assert report.training_series(history)['global_distribution'] == [1.25]
    del history[0]['coherence_family/topology/raw']
    with pytest.raises(ValueError, match='sampled-training family history'):
        report.training_series(history)


def test_missing_costs_do_not_hide_measured_science_or_qualify_unknown_routes(tmp_path):
    report = load()
    arms = list(report.ARMS[:2])
    cost = {'complete': False, 'missing': ['assessment timer unavailable'],
            'arms': {arm: {'serial_cost_seconds': None, 'checkpoint_cost_seconds': {}} for arm in arms}}
    decision = {'costs_complete': False, 'quality_route': None, 'learning_efficiency_route': None,
                'deployment_comparison_route': None,
                'deployment_disposition': {'opt_in_function': None, 'operational_function': None,
                    'decisions': {'Residual': {'task_useful': True, 'source_deployment_route': None,
                                             'operational_eligible': None}}}}
    report.validate_cost_evidence(cost, decision, arms, tmp_path)
    decision['quality_route'] = False
    with pytest.raises(ValueError, match='unavailable cost-dependent routes'):
        report.validate_cost_evidence(cost, decision, arms, tmp_path)
    decision['quality_route'] = None
    decision['deployment_disposition']['opt_in_function'] = 'Residual'
    with pytest.raises(ValueError, match='cannot qualify a deployment adoption'):
        report.validate_cost_evidence(cost, decision, arms, tmp_path)


def test_complete_cost_evidence_still_requires_all_checkpoint_clocks(tmp_path):
    report = load()
    evidence = tmp_path / 'cost_evidence.json'; evidence.write_text('{}')
    arms = list(report.ARMS[:2])
    cost = {'complete': True, 'missing': [],
            'source_receipts': [{'path': str(evidence), 'sha256': report.sha(evidence)}],
            'arms': {arm: {'serial_cost_seconds': 2.,
                           'checkpoint_cost_seconds': {str(epoch): 1. for epoch in report.EPOCHS}}
                     for arm in arms}}
    decision = {'costs_complete': True}
    report.validate_cost_evidence(cost, decision, arms, tmp_path)
    cost['arms'][arms[0]]['checkpoint_cost_seconds'].pop('250')
    with pytest.raises(ValueError, match='six checkpoint costs'):
        report.validate_cost_evidence(cost, decision, arms, tmp_path)


def test_exact_full_costs_accept_honest_checkpoint_bounds_without_fabricated_times(tmp_path):
    report = load()
    evidence = tmp_path / 'cost_evidence.json'; evidence.write_text('{}')
    arms = list(report.ARMS[:2])
    cost = {'complete': True, 'missing': [],
            'source_receipts': [{'path': str(evidence), 'sha256': report.sha(evidence)}],
            'arms': {arm: {'serial_cost_seconds': 2.,
                           'checkpoint_cost_seconds': {str(epoch): 1. for epoch in report.EPOCHS},
                           'checkpoint_cost_bounds_seconds': {str(epoch): [1., 1.] for epoch in report.EPOCHS}}
                     for arm in arms}}
    item = cost['arms'][arms[0]]
    item['checkpoint_cost_seconds']['100'] = None
    item['checkpoint_cost_bounds_seconds']['100'] = [.5, 1.5]
    report.validate_cost_evidence(cost, {'costs_complete': True}, arms, tmp_path)
    assert report.route_label(None, 'checkpoint cost bounds overlap') == 'unavailable (checkpoint cost bounds overlap)'
    item['checkpoint_cost_bounds_seconds']['100'] = [1., 1.]
    with pytest.raises(ValueError, match='nondegenerate measured bound'):
        report.validate_cost_evidence(cost, {'costs_complete': True}, arms, tmp_path)


def test_native_plot_has_per_field_physical_error_annotations(tmp_path):
    import subprocess

    report = load()
    source = tmp_path / 'source'
    source.mkdir()
    selected = tmp_path / 'assessment/selected'
    selected.mkdir(parents=True)
    fields = ['T', 'CO']
    truth = torch.tensor([[[1., 2.], [2., 3.], [3., 5.], [4., 8.]]])
    common = {'sample_id': 'trajectory:0', 'truth_model': truth, 'member_seeds': [42]}
    torch.save({**common, 'coordinates_raw': torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]),
                'native_masses': torch.tensor([1., 1., 1., 0.]), 'normalizer_offset': torch.tensor([0., 0.]),
                'normalizer_scale': torch.tensor([1., 1.])}, source / '0000.pt')
    torch.save({**common, 'members_model': (truth + .1).unsqueeze(0)}, selected / '0000.pt')
    receipt = {'schema': 'car_round2_native_bank_v1', 'status': 'completed', 'split': 'validation',
               'kind': 'source', 'field_order': fields, 'files_sha256': [report.sha(source / '0000.pt')]}
    (source / 'receipt.json').write_text(json.dumps(receipt))
    assessment = {'source_bank_sha256': report.sha(source / 'receipt.json'),
                  'selected_files_sha256': [report.sha(selected / '0000.pt')], 'sample_ids': ['trajectory:0']}
    figures = tmp_path / 'figures'; figures.mkdir()
    report.plot_native(figures, 'senseiver', source, selected.parent / 'assessment.json', assessment)
    text = subprocess.check_output(['pdftotext', str(figures / 'native_senseiver.pdf'), '-'], text=True)
    assert text.count('physical relative L2') == 2
    assert (figures / 'native_senseiver.png').is_file()


@pytest.mark.parametrize('complete_costs', [True, False])
def test_synthetic_report_renders_every_indexed_preview_and_master(tmp_path, monkeypatch, complete_costs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    report = load()
    histories = {arm: [{'step': epoch * 12, 'data_loss': 0.,
                        'coherence_family/global_distribution/raw': float(1 / epoch),
                        'coherence_family/multiscale_spectrum/raw': float(2 / epoch),
                        'coherence_family/topology/raw': float(3 / epoch)} for epoch in range(1, 1001)]
                 for arm in report.ARMS}
    scores = {'state': 1., 'scale': 1., 'shape': 1., 'event': 1.}
    row = {'alpha': 0., 'scores': scores, 'risk': {'member_mse': [1., 2.]},
           'per_condition_scores': [{'state': 1., 'scale': 1., 'shape': 1.}],
           'fidelity': {'primary_pass': True}, 'latency_seconds': 1., 'sensor_mse': [.1, None]}
    raw = {**row, 'alpha': 1.}
    doc = {'selected_alpha': 0., 'rows': [row, raw], 'normalizers': scores,
           'sample_ids': ['trajectory:1'], 'reserved_task': {'normalized_improvement': 0.},
           'source_latency_seconds': 1., 'native_abs_error_tail_quantiles': [.95, .99, 1.],
           'raw_native_abs_error_tails': [[[.1, .2], [.3, .4], [.5, .6]]],
           'selected_native_abs_error_tails': [[[.1, .2], [.3, .4], [.5, .6]]],
           'represented_decomposition': {label: [{'attraction': {'state': 2., 'scale': 3., 'shape': 4.},
               'fair_separation_correction': {'state': 1., 'scale': 2., 'shape': 3.}}]
               for label in ('raw', 'selected')}}
    docs = {arm: {split: {epoch: (tmp_path / 'unused.json', doc) for epoch in report.EPOCHS}
                  for split in ('train', 'validation')} for arm in report.ARMS}
    decisions = {family: {'deployment_disposition': {'operational_recommendation': 'retain SOURCE/C8',
                                                     'opt_in_function': None},
                          'quality_route': False if complete_costs else None,
                          'learning_efficiency_route': False if complete_costs else None,
                          'deployment_comparison_route': False if complete_costs else None,
                          'allocation_decision': {'training_mode': 'uniform', 'status': 'uniform_fallback', 'adopt': False}}
                 for family in ('ffm', 'senseiver')}
    costs = {family: {'complete': complete_costs, 'missing': [] if complete_costs else ['missing assessment timer'],
                      'allocator_final': {'status': 'uniform_fallback', 'qualified': False},
                      'arms': {arm: {'serial_cost_seconds': 1. if complete_costs else None, 'gpu_seconds': .5,
                                     'independent_cpu_seconds': .5, 'failed_attempt_seconds': 0., 'paused_attempt_seconds': 0.}
                               for arm in report.ARMS}}
             for family in ('ffm', 'senseiver')}
    banks = {family: {'selected_count': 1, 'candidate_count': 2} for family in ('ffm', 'senseiver')}
    monkeypatch.setattr(report, 'validate_inputs', lambda *_: (docs, histories, decisions, costs, banks, {'mode': 'uniform'}))
    def native(directory, family, *_):
        figure, axis = plt.subplots()
        axis.text(.1, .5, 'synthetic native physical relative L2 = 0')
        report.save_figure(figure, directory, f'native_{family}')
    monkeypatch.setattr(report, 'plot_native', native)
    sources = {}
    for family in ('ffm', 'senseiver'):
        path = tmp_path / family; path.mkdir()
        (path / 'receipt.json').write_text(json.dumps({'field_order': ['T', 'CO']}))
        sources[family] = str(path)
    audit = tmp_path / 'artifacts/allocation/synthetic_head/audit.json'
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps({'schema': 'car_round2_allocator_job_v1', 'status': 'completed',
        'code_capsule_sha256': 'synthetic-capsule', 'gradient_cost_uniform_ratio': 1.5,
        'policies': {'uniform': {'gradient_mse': 1., 'full_estimator_seconds': 2., 'mse_cost': 2.},
                     'lagged_gradient_cost': {'gradient_mse': 1.5, 'full_estimator_seconds': 2., 'mse_cost': 3.}}}))
    operations = tmp_path / 'operations'; operations.mkdir()
    (operations / 'synthetic_head_launch.json').write_text(json.dumps({
        'output': str(audit.relative_to(tmp_path)), 'code_capsule_sha256': 'synthetic-capsule',
        'command': ['--replicates', '128']}))
    ledger = tmp_path / 'qualification_ledger.json'
    markdown = tmp_path / 'qualification_ledger.md'; markdown.write_text('# Synthetic qualification ledger\n')
    ledger.write_text(json.dumps({'schema': 'car_round2_definition_qualification_ledger_v1',
        'execution_qualification_passed': True, 'markdown_path': str(markdown),
        'markdown_sha256': report.sha(markdown),
        'source_receipts': [{'path': str(audit), 'sha256': report.sha(audit)}]}))
    report_inputs = {'source_banks': sources, 'allocator_audits': {'ffm': [], 'senseiver': [str(audit)]},
                     'definition_qualification_ledger': str(ledger)}
    output = tmp_path / 'report'
    report.produce(report_inputs, tmp_path, output)
    assert (output / 'report.md').is_file()
    assert (output / 'report.html').is_file()
    rendered_html = (output / 'report.html').read_text()
    assert '<table>' in rendered_html and '<th>Field</th>' in rendered_html
    assert 'Analytic gradient MSE</th>' in rendered_html
    assert '128 saved-dispatch draws' in rendered_html and 'lagged_gradient_cost' in rendered_html
    assert 'qualification/ledger.md' in rendered_html and 'pre-learning qualification' in rendered_html
    assert (output / 'qualification/ledger.json').read_bytes() == ledger.read_bytes()
    assert (output / 'qualification/ledger.md').read_bytes() == markdown.read_bytes()
    assert '</head><body><h1' in rendered_html and rendered_html.endswith('</body></html>')
    receipt = json.loads((output / 'report_receipt.json').read_text())
    assert receipt['report_inputs_sha256'] == report.sha(output / 'report_inputs.json')
    assert receipt['definition_qualification_ledger_sha256'] == report.sha(ledger)
    assert receipt['definition_qualification_markdown_sha256'] == report.sha(markdown)
    assert receipt['script_sha256'] == report.sha(ROOT / 'scripts/analysis/report_car_round2.py')
    assert json.loads((output / 'report_inputs.json').read_text()) == report_inputs
    assert len(list((output / 'figures').glob('*.png'))) == 8
    assert len(list((output / 'figures').glob('*.pdf'))) == 8
    assert json.loads((output / 'paired_condition_blocks.json').read_text())['ffm']['blocks'][0]['count'] == 1
    text = (output / 'report.md').read_text()
    assert 'Two-family nonuniform allocation decision: **reject**' in text
    assert 'Raw separation correction' in text
    assert 'Sensor MSE: raw / nominated' in text
    if not complete_costs:
        import subprocess
        assert 'Quality route: unavailable (incomplete costs)' in text
        assert 'complete serial unavailable' in text
        pdf_text = subprocess.check_output(['pdftotext', str(output / 'figures/task_deployment_cost.pdf'), '-'], text=True)
        assert 'Full model: unavailable' in pdf_text and 'Residual: unavailable' in pdf_text


def test_native_tables_distinguish_conditional_quantile_means_from_maximum():
    report = load()
    row = {'alpha': 0., 'risk': {'member_mse': [1., 2.]}, 'sensor_mse': [.2, None]}
    doc = {'selected_alpha': 0., 'rows': [row, {**row, 'alpha': 1.}],
           'sample_ids': ['trajectory:0', 'trajectory:1'],
           'native_abs_error_tail_quantiles': [.95, .99, 1.],
           'raw_native_abs_error_tails': [[[1., 2.], [3., 4.], [5., 6.]],
                                          [[3., 4.], [5., 6.], [9., 10.]]],
           'selected_native_abs_error_tails': [[[1., 2.], [3., 4.], [5., 6.]],
                                               [[3., 4.], [5., 6.], [9., 10.]]],
           'represented_decomposition': {label: [{'attraction': {'state': a, 'scale': a, 'shape': a},
                'fair_separation_correction': {'state': .5, 'scale': .5, 'shape': .5}} for a in (2., 4.)]
                for label in ('raw', 'selected')}}
    docs = {arm: {'validation': {1000: (None, doc)}} for arm in report.ARMS}
    text = '\n'.join(report.native_diagnostic_tables(docs, {'ffm': ['T', 'CO'], 'senseiver': ['T', 'CO']}))
    assert '| T | 1 / 1 | 0.2 / 0.2 | 2, 4, 9 | 2, 4, 9 |' in text
    assert '| State | 3 | 0.5 | 3 | 0.5 |' in text
    assert 'unobserved / unobserved' in text
    doc['raw_native_abs_error_tails'][1][2][0] = float('nan')
    with pytest.raises(ValueError, match='complete finite per-condition native tails'):
        report.native_diagnostic_tables(docs, {'ffm': ['T', 'CO'], 'senseiver': ['T', 'CO']})
