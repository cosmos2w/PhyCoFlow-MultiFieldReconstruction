import json

import pytest
from PIL import Image

from phycoflow_reconstruction.evaluation.round7_analysis import analyze_updates, generate_analysis


def _record(step, epoch, *, live=None, shadow=None):
    measured = {'family_benefits': live or [1., .3, 1.], 'component_products': {}, 'shadows': {},
                'diagnostic_seconds': .2, 'gradient_and_direction_seconds': .03,
                'extra_gradient_evaluations': 0}
    if shadow is not None:
        measured['component_products'] = {'joint_attraction': .4, 'joint_separation': -.1,
                                           'temperature_risk': -.2}
        measured['component_gradient_norms'] = {'joint_attraction': 4., 'joint_separation': 3.}
        measured['extra_gradient_evaluations'] = 8
        measured['shadows']['scale_only'] = {'family_benefits': [0., shadow, 0.],
            'component_products': {'joint_attraction': .1, 'joint_separation': -.2}, 'seconds': .05}
    return {'step': step, 'epoch': epoch, 'gradient': {'round7_actual_step': measured}}


def _window(epoch, live=.3, shadow=1.):
    return [_record(epoch * 13 + i, epoch, live=[-1. if i <= 2 else 1., live, 1.],
                    shadow=shadow if i in (1, 7, 13) else None) for i in range(1, 14)]


def test_flags_use_complete_paired_whole_scale_and_exclude_warmup():
    rows = [_record(1, 1, live=[-9., -9., -9.], shadow=2.), _record(76, 6, live=[-9., -9., -9.])]
    rows += _window(100) + _window(300)
    result = analyze_updates(rows, arm='P')
    assert result['warmup_records_excluded'] == 2
    assert result['post_warmup_families']['global_distribution']['negative_count'] == 4
    assert result['negative_at_least_ten_percent']['global_distribution'] is True
    assert result['negative_at_least_ten_percent']['topology'] is False
    assert result['scale_only_at_most_half_in_two_windows'] is True
    assert result['windows']['100']['scale_only_comparison']['live_mean'] == pytest.approx(.3)
    assert result['windows']['100']['scale_only_comparison']['shadow_mean'] == pytest.approx(1.)
    assert result['post_warmup_components']['joint_separation']['median'] == -.1
    assert result['post_warmup_component_gradient_norms']['joint_separation']['median'] == 3
    assert result['post_warmup_shadow_components']['scale_only']['joint_attraction']['median'] == .1
    assert result['telemetry_cost']['shadow_proposals'] == 7


def test_missing_incomplete_and_nonpositive_shadow_are_unassessed():
    assert analyze_updates([], arm='B')['scale_only_at_most_half_in_two_windows'] is None
    incomplete = _window(100)[:-1] + _window(300, shadow=-1.)
    result = analyze_updates(incomplete, arm='T')
    assert result['scale_only_at_most_half_in_two_windows'] is None
    assert not result['windows']['100']['scale_only_comparison']['assessed']
    assert not result['windows']['300']['scale_only_comparison']['assessed']
    with pytest.raises(ValueError, match='duplicate'):
        analyze_updates([_record(1, 1), _record(1, 1)], arm='B')


def test_generate_offline_png_pdf_and_train_family_series(tmp_path):
    root = tmp_path / 'B'; log = root / 'logs'; log.mkdir(parents=True)
    (log / 'round7_updates.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in _window(100)))
    rows = [{'epoch': 1, 'coherence_family/global_distribution/raw': .4,
             'coherence_family/global_distribution/weighted_contribution': .2,
             'coherence_family/topology/raw': .5, 'coherence_family/topology/weighted_contribution': .7}]
    (log / 'family_updates.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    output = tmp_path / 'out'; report = generate_analysis({'B': root, 'T': tmp_path / 'missing'}, output)
    assert report['sources']['B']['family_record_count'] == 1
    assert report['sources']['T']['actual_updates'] is None
    assert report['arms']['T']['negative_at_least_ten_percent']['topology'] is None
    for name in ('actual_step_projections', 'family_training_losses'):
        with Image.open(output / (name + '.png')) as image:
            image.verify()
        assert (output / (name + '.pdf')).read_bytes().startswith(b'%PDF-')
    assert json.loads((output / 'analysis.json').read_text())['schema'] == 'round7_saved_evidence_analysis_v1'
