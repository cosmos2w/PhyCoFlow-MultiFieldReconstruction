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


def test_synthetic_report_renders_every_indexed_preview_and_master(tmp_path, monkeypatch):
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
           'fidelity': {'primary_pass': True}, 'latency_seconds': 1.}
    raw = {**row, 'alpha': 1.}
    doc = {'selected_alpha': 0., 'rows': [row, raw], 'normalizers': scores,
           'sample_ids': ['trajectory:1'], 'reserved_task': {'normalized_improvement': 0.},
           'source_latency_seconds': 1.}
    docs = {arm: {split: {epoch: (tmp_path / 'unused.json', doc) for epoch in report.EPOCHS}
                  for split in ('train', 'validation')} for arm in report.ARMS}
    decisions = {family: {'deployment_disposition': {'operational_recommendation': 'retain SOURCE/C8',
                                                     'opt_in_function': None}, 'quality_route': False,
                          'learning_efficiency_route': False, 'deployment_comparison_route': False}
                 for family in ('ffm', 'senseiver')}
    costs = {family: {'allocator_final': {'status': 'uniform_fallback', 'qualified': False},
                      'arms': {arm: {'serial_cost_seconds': 1., 'gpu_seconds': .5,
                                     'independent_cpu_seconds': .5, 'failed_attempt_seconds': 0.}
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
    output = tmp_path / 'report'
    report.produce({'source_banks': sources, 'allocator_audits': {'ffm': [], 'senseiver': []}}, tmp_path, output)
    assert (output / 'report.md').is_file()
    assert (output / 'report.html').is_file()
    receipt = json.loads((output / 'report_receipt.json').read_text())
    assert receipt['report_inputs_sha256'] == report.sha(output / 'report_inputs.json')
    assert receipt['script_sha256'] == report.sha(ROOT / 'scripts/analysis/report_car_round2.py')
    assert json.loads((output / 'report_inputs.json').read_text()) == {
        'source_banks': sources, 'allocator_audits': {'ffm': [], 'senseiver': []}}
    assert len(list((output / 'figures').glob('*.png'))) == 8
    assert len(list((output / 'figures').glob('*.pdf'))) == 8
    assert json.loads((output / 'paired_condition_blocks.json').read_text())['ffm']['blocks'][0]['count'] == 1
