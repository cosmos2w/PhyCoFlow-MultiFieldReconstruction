import json
from pathlib import Path
from phycoflow_reconstruction.evaluation.round7_report import generate_report


def test_report_renders_frozen_source_only(tmp_path, monkeypatch):
    from phycoflow_reconstruction.evaluation import round7_summary
    monkeypatch.setattr(round7_summary, 'summarize_reductions', lambda path, fields: {
        'conditions': 100, 'scores': {'scale': 1}, 'fields': {}})
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps({'status': 'frozen_before_test', 'selected': 'SOURCE', 'endpoint': None}))
    test = tmp_path / 'test.json'
    test.write_text(json.dumps({'endpoints': {'SOURCE': {'reductions': 'source.json', 'checkpoint_sha256': 'source'}}}))
    output = generate_report(tmp_path, test, selection, gpu_seconds=3600, attempt_count=48)
    assert output['disposition'] == 'SOURCE'
    assert Path(output['report_pdf']).is_file()
    assert list((tmp_path / 'comparison/postprocess').glob('rendered_page-*.png'))


def test_report_withholds_on_test_failure_without_rerank(tmp_path, monkeypatch):
    from phycoflow_reconstruction.evaluation import round7_summary
    fields = ['CH4', 'CO', 'T', 'U_1', 'p']
    summary = {'conditions': 100, 'scores': {'scale': .9, 'state': .1, 'shape.fair': 1., 'shape.legacy': 1.},
               'observer_identities': {'fair': 'a', 'legacy': 'b'},
               'fields': {f: {'relative_l2': .1} for f in fields},
               'component_diagnostics': {},
               'loss_means': {k: 1. for k in ('multiscale_spectrum.signed_band_energy_score.conditional',
                   'multiscale_spectrum.joint_spectral_mean_energy_score.conditional',
                   'multiscale_spectrum.relation_self_energy_score.conditional',
                   'multiscale_spectrum.relation_rho_crps.conditional')},
               'shape_observers': {name: {'losses': {'topology.self.persistence': .5,
                   'topology.mutual.persistence': .5}} for name in ('fair', 'legacy')},
               'j2_physical_measure_model_units': {'variance_proxy': [1] * 5,
                                                   'bias_plus_reference_variance_proxy': [2] * 5}}
    monkeypatch.setattr(round7_summary, 'summarize_reductions', lambda *args, **kwargs: summary)
    changes = {key: {'reference': 1., 'candidate': .9, 'raw_change': -.1,
                     'paired_median_change': -.1, 'paired_p05_change': -.2,
                     'paired_p95_change': .1, 'conditions_improved': 70, 'conditions': 100,
                     'source_units_change': -.1, 'fractional_change': -.1}
               for key in ('state', 'scale', 'shape.fair', 'shape.legacy')}
    changes.update({'field.' + f + '.mse': {'reference': 1., 'candidate': 1.2 if f == 'T' else 1.} for f in fields})
    monkeypatch.setattr(round7_summary, 'compare_summaries', lambda *args, **kwargs: {'changes': changes})
    binding = tmp_path / 'binding.json'
    binding.write_text(json.dumps({'train_primary_mse_ratios': {f: 1. for f in fields},
        'native_comparison': {'field_names': fields, 'primary_mse_ratio': [1.] * 5}, 'train_reductions': 'train.json',
        'validation_reductions': 'validation.json'}))
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps({'status': 'frozen_before_test', 'selected': 'P',
                                      'endpoint': {'checkpoint_sha256': 'abc', 'binding': str(binding), 'epoch': 1000}}))
    test = tmp_path / 'test.json'
    test.write_text(json.dumps({'endpoints': {'SOURCE': {'checkpoint_sha256': 'source', 'reductions': 'source.json', 'native_receipt': 'source/native.json'},
        'selected': {'checkpoint_sha256': 'abc', 'reductions': 'selected.json', 'native_receipt': 'selected/native.json'}},
        'native_comparisons': {'selected': {'field_names': fields, 'primary_mse_ratio': [1.] * 5}}}))
    (tmp_path / 'artifacts').mkdir()
    (tmp_path / 'artifacts/reference_SOURCE.json').write_text(json.dumps({
        'train': {'reductions': 'train.json', 'fair_shape_attraction_mean': .7},
        'validation': {'reductions': 'validation.json'}}))
    (tmp_path / 'artifacts/qualification_contract.json').write_text(json.dumps({'projection_scales': {
        'B': {'global_distribution': .1, 'multiscale_spectrum': .9, 'topology': 1.}}}))
    output = generate_report(tmp_path, test, selection, gpu_seconds=3600, attempt_count=100)
    assert output['disposition'] == 'SOURCE'
    assert output['test_q_ratios']['T'] == 1.2
    assert (tmp_path / 'comparison/postprocess/figures/matched_components_fidelity.png').exists()


def test_fixed_evidence_retains_all_boundaries_when_source_selected(tmp_path):
    from phycoflow_reconstruction.evaluation.round7_report import _fixed_panel_evidence, _SCALE_TERMS
    artifacts = tmp_path / 'artifacts'
    artifacts.mkdir()
    (artifacts / 'reference_SOURCE.json').write_text(json.dumps({
        'train': {'reductions': 'source_train'}, 'validation': {'reductions': 'source_val'}}))
    for arm, epochs in [('B', (500, 1000)), ('P', (500,))]:
        run = tmp_path / arm
        (run / 'evaluation').mkdir(parents=True)
        (artifacts / f'owned_run_{arm}.json').write_text(json.dumps({'run': str(run)}))
        for epoch in epochs:
            (artifacts / f'{arm}_{epoch}_complete.json').write_text('{}')
            (run / 'evaluation' / f'checkpoint_binding_{epoch}.json').write_text(json.dumps({
                'checkpoint_sha256': f'{arm}{epoch}', 'train_reductions': 'train', 'validation_reductions': 'val'}))
    signed = _SCALE_TERMS['signed'][0]
    joint = _SCALE_TERMS['joint'][0]
    panel = {'scores': {'state': 2., 'scale': 3., 'shape.fair': -1., 'shape.legacy': 4.},
             'loss_means': {key: 2. for key, _ in _SCALE_TERMS.values()},
             'component_diagnostics': {signed: {'weighted_attraction': 3., 'weighted_separation': 1.},
                                       joint: {'attraction': 8., 'repulsion': 4.}},
             'shape_observers': {'fair': {'losses': {'topology.self.persistence': 1., 'topology.mutual.persistence': -2.},
                                         'diagnostics': {'component_attraction_separation': {'self': [1., 2.]}}}}}
    result = _fixed_panel_evidence(tmp_path, [], lambda path, fields: panel, lambda a, b: {'matched': True})
    assert set(result) == {'SOURCE', 'B', 'P'}
    assert set(result['B']) == {'500', '1000'} and set(result['P']) == {'500'}
    validation = result['P']['500']['splits']['validation']['summary']
    assert validation['weighted_scale_terms']['joint'] == .5
    assert validation['weighted_scale_attraction_separation']['joint'] == {'attraction': 2., 'separation': 1.}
    assert validation['weighted_scale_attraction_separation']['signed']['separation'] == 1.
    assert validation['shape_observers']['fair']['losses']['topology.mutual.persistence'] == -2.


def test_fixed_test_evidence_keeps_baseline_and_selected():
    from phycoflow_reconstruction.evaluation.round7_report import _test_panel_evidence
    entries = {name: {'checkpoint_sha256': name, 'reductions': name} for name in ('SOURCE', 'B1000', 'selected')}
    native = {name: {'field_names': ['f'], 'primary_mse_ratio': [1.05]} for name in ('B1000', 'selected')}
    result = _test_panel_evidence(entries, {'native_comparisons': native}, {'scores': {}}, ['f'],
                                  lambda p, f: {'scores': {'scale': 2}}, lambda a, b: {'changes': {'scale': {}}})
    assert set(result) == {'SOURCE', 'B1000', 'selected'}
    assert result['B1000']['native_comparison']['primary_mse_ratio'] == [1.05]
    assert result['selected']['source_comparison'] == {'changes': {'scale': {}}}


def test_report_timing_does_not_sum_overlapping_gpu_intervals(tmp_path):
    from phycoflow_reconstruction.evaluation.round7_report import _timing_evidence
    artifacts = tmp_path / 'artifacts'
    artifacts.mkdir()
    (artifacts / 'budget_ledger.json').write_text(json.dumps({'intervals': [
        {'stage': 'prepare', 'start': 10, 'end': 20}, {'stage': 'qualify', 'start': 15, 'end': 25}]}))
    (artifacts / 'prepare_complete.json').write_text(json.dumps({'elapsed_seconds': 10}))
    result = _timing_evidence(tmp_path, 15)
    assert result['elapsed_first_to_last_gpu_stage_seconds'] == 15
    assert result['occupied_gpu_union_seconds'] == 15
    assert result['stage_wall_seconds_summed_may_overlap'] == {'prepare': 10, 'qualify': 10}
    assert result['calibration']['elapsed_seconds'] == 10


def test_negative_selection_still_renders_fixed_endpoint_figure(tmp_path):
    from phycoflow_reconstruction.evaluation.round7_report import _fixed_endpoint_figure, _panel_snapshot
    fields = ['CH4', 'CO', 'T', 'U_1', 'p']
    panel = _panel_snapshot({
        'scores': {'scale': 2.},
        'loss_means': {'multiscale_spectrum.joint_spectral_mean_energy_score.conditional': 4.},
        'component_diagnostics': {'multiscale_spectrum.joint_spectral_mean_energy_score.conditional':
                                  {'attraction': 8., 'repulsion': 4.}},
        'shape_observers': {'fair': {'losses': {'topology.self.persistence': 1., 'topology.mutual.persistence': -.2},
                                     'diagnostics': {'component_attraction_separation': {
                                         'self.persistence': {'attraction': [2.], 'separation': [1.]},
                                         'mutual.persistence': {'attraction': [1.], 'separation': [.5]}}}},
                            'legacy': {'losses': {'topology.self.persistence': .7, 'topology.mutual.persistence': .2}}},
        'j2_physical_measure_model_units': {'variance_proxy': [1., 2., 3., 4., 5.]}})
    changes = {'field.' + f + '.mse': {'candidate': 1.05, 'reference': 1.} for f in fields}
    item = {'splits': {'validation': {'summary': panel, 'source_comparison': {'changes': changes}}},
            'native_comparison': {'field_names': fields, 'primary_mse_ratio': [1.02] * 5}}
    evidence = {'SOURCE': {'validation': panel}, 'B': {'500': item}, 'P': {'500': item}}
    _fixed_endpoint_figure(evidence, fields, tmp_path)
    assert panel['j2_physical_measure_model_units']['variance_proxy'][2] == 3.
    assert (tmp_path / 'fixed_endpoint_evidence.png').stat().st_size > 0
    assert (tmp_path / 'fixed_endpoint_evidence.pdf').stat().st_size > 0
    from PIL import Image
    with Image.open(tmp_path / 'fixed_endpoint_evidence.png') as image:
        image.verify()


def test_s100_evidence_uses_matched_saved_validation_only(tmp_path):
    from phycoflow_reconstruction.evaluation.round7_report import _s100_validation_evidence
    artifacts = tmp_path / 'artifacts'
    artifacts.mkdir()
    for name in ('SOURCE', 'S100'):
        (artifacts / f'reference_{name}.json').write_text(json.dumps({
            'validation': {'reductions': name + '_validation.json'}}))
    calls = []
    def summarize(path, fields):
        calls.append((path, fields))
        return {'identity': path, 'scores': {'scale': 1.}}
    def compare(source, candidate):
        assert source['identity'] == 'SOURCE_validation.json'
        assert candidate['identity'] == 'S100_validation.json'
        return {'changes': {'scale': {'reference': 1., 'candidate': .9,
                                      'raw_change': -.1, 'fractional_change': -.1}}}
    result = _s100_validation_evidence(tmp_path, ['T'], summarize, compare)
    assert calls == [('SOURCE_validation.json', ['T']), ('S100_validation.json', ['T'])]
    assert result['support'] == 'VALIDATION100 Q4096'
    assert result['source_comparison']['changes']['scale']['raw_change'] == -.1
    assert result['s100_summary']['identity'] == 'S100_validation.json'


def test_accounting_uses_recorded_counts_and_never_invents_cache_counts(tmp_path):
    import hashlib
    import pickle
    import zipfile
    from phycoflow_reconstruction.evaluation.round7_report import _accounting_evidence
    artifacts = tmp_path / 'artifacts'
    artifacts.mkdir()
    artifact = artifacts / 'calibration.pt'
    with zipfile.ZipFile(artifact, 'w') as archive:
        archive.writestr('fixture/data.pkl', pickle.dumps({'round7_calibration_receipt': {
            'source_forwards': 800, 'reference_visits': 800,
            'calibration_endpoint_filtrations': 100, 'calibration_dimension_diagrams': 200}}, protocol=2))
    (artifacts / 'prepare_complete.json').write_text(json.dumps({'fair_artifact': str(artifact),
        'fair_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest()}))
    (artifacts / 'budget_ledger.json').write_text(json.dumps({'intervals': [{'start': 1, 'end': 2}],
        'attempt_count': 60, 'qualification_attempts': 49, 'formal_unique': 10, 'formal_replays': 1}))
    (artifacts / 'reference_SOURCE.json').write_text(json.dumps({'train': {'conditions': 800},
                                                                  'validation': {'conditions': 100}}))
    (artifacts / 'B_500_complete.json').write_text(json.dumps({'run': 'b', 'epoch': 500,
        'timing': {'calls': {'reconstruction': 155}}}))
    result = _accounting_evidence(tmp_path)
    assert result['optimizer_attempts']['formal_replays'] == 1
    assert result['calibration_recorded_counts']['source_forwards'] == 800
    assert result['reference_panels']['SOURCE']['train']['fresh_rollouts'] is None
    assert result['reference_panels']['SOURCE']['train']['conditional_draws'] is None
    assert result['completed_formal_segments'][0]['reconstruction_calls'] == 155


def test_closed_500_narrative_is_specific_and_uses_saved_scores():
    from phycoflow_reconstruction.evaluation.round7_report import _closed_500_narrative
    selection = {'selected': 'SOURCE', 'endpoint': None, 'B1000': None}
    verdict = {'mode': 'close', 'selected': 'SOURCE', 'continue': []}
    def score(scale, fair=2., legacy=3.):
        return {'scale': scale, 'shape.fair': fair, 'shape.legacy': legacy}
    def arm(values):
        return {'500': {'splits': {'validation': {'summary': {'scores': values}}}}}
    fixed = {'SOURCE': {'validation': {'scores': score(1.)}},
             'B': arm(score(1.016)), 'P': arm(score(1.009)), 'T': arm(score(.995, 1.7, 2.85))}
    observations = [{'name': name, 'observations': {'500': {'q_mse': {'T': 1.11}}}} for name in ('P', 'T')]
    text = ' '.join(_closed_500_narrative(selection, {'SOURCE': {}}, verdict, fixed, observations))
    assert 'No 1000-epoch continuation or extension' in text
    assert 'candidate TEST was not performed' in text
    assert '0.70% of SOURCE Scale' in text and 'field-fidelity endpoint requirements failed' in text
    assert 'fair Shape -15.00%' in text and 'legacy paired Shape -5.00%' in text
    assert _closed_500_narrative(selection, {'SOURCE': {}, 'B1000': {}}, verdict, fixed, observations) == []
    assert _closed_500_narrative(selection, {'SOURCE': {}}, {**verdict, 'mode': 'eligible'}, fixed, observations) == []
