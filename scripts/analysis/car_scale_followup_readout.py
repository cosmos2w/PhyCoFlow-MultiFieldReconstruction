#!/usr/bin/env python3
"""Assemble completed Scale diagnostics without inference or optimizer actions."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

INPUTS = {
    'train': ('TRAIN800_ExhaustiveScale_20261125_v1/decomposition.json', 'cc6813d632e1be4ea5818f04f6195d2b87cd32dba62edb648ef3cb51b41ab0a7'),
    'validation': ('Validation100_ExhaustiveScale_20261125_v1/decomposition.json', '5e1d7c408dcc9475b64b70e2c666bc48dc39854e305aa2176e8a010fd4f9a4d1'),
    'adam': ('Epoch500_ActualAdam_TRAIN800_20261125_v3/receipt.json', '4f04daf02907b9705dc5b25ca3262f8696e36a1147056d014db0b5d1679f0c39'),
    'fields': ('CachedFieldSupport_TRAIN800_VALID100_20261125_v1/field_support.json', 'e89d56b1d76f14112f38b1f055fd9759a41ff409641534a5dac73194fb165d82'),
    'conditions': ('ConditionSupport_TRAIN800_VALID100_20261125_v1/condition_support.json', '95ec9f87bc85bf3d7e4553829f225e728876806b2de6ee71c878bbe7d582d6b0'),
}
UNIT = 0.25978530736085925
DEFINITIONS = ('Published_Core_Default', 'Complete_Core_Prefit', 'Complete_Expanded_Prefit')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assemble(campaign):
    data, identities = {}, {}
    for name, (relative, expected) in INPUTS.items():
        path = campaign / 'assessment' / relative
        if digest(path) != expected:
            raise ValueError(f'completed evidence changed: {path}')
        data[name] = json.loads(path.read_text())
        if data[name]['status'] != 'completed':
            raise ValueError(f'incomplete evidence: {path}')
        identities[name] = {'path': str(path), 'sha256': expected}
    rows = data['adam']['results']
    weights = np.array([row['sample_count'] for row in rows])
    if weights.sum() != 800 or len(rows) != 13:
        raise ValueError('actual-Adam complete TRAIN800 shadows required')
    action = {}
    for key in ('same_query_train_scale', 'independent_query_train_scale', *DEFINITIONS):
        delta = np.array([row[key]['post'] - row[key]['pre'] if key in DEFINITIONS
                          else row[key + '_post'] - row[key + '_pre'] for row in rows])
        action[key] = {'sample_weighted_raw_change': float(np.average(delta, weights=weights)),
                       'negative_batches': int((delta < 0).sum()), 'batch_changes': delta.tolist()}
    first_order = {}
    for key in ('full_core_gradient_dot_actual_displacement', 'family_gradient_dot_actual_displacement',
                'weighted_field_gradient_dot_actual_displacement'):
        values = np.array([row[key] for row in rows])
        first_order[key] = {'sample_weighted_mean': np.average(values, weights=weights, axis=0).tolist(),
                            'negative_batches': (values < 0).sum(axis=0).tolist()}
    support = {}
    for split, item in data['fields']['splits'].items():
        records = item['records']
        allowance = np.array(item['physical_allowance'])
        paired = np.array([
            (np.array(row['models']['epoch500']['signed_mean_risk']['complement_region'])
             - np.array(row['models']['SOURCE']['signed_mean_risk']['complement_region'])
             - np.array(row['models']['epoch500']['signed_mean_risk']['selected_region'])
             + np.array(row['models']['SOURCE']['signed_mean_risk']['selected_region'])) / allowance
            for row in records])
        indices = np.random.default_rng(6042026).integers(0, len(records), size=(2000, len(records)))
        support[split] = {'field_order': item['field_order'], 'sample_count': len(records),
                          'paired_complement_minus_selected_change_over_allowance': paired.mean(0).tolist(),
                          'paired_condition_bootstrap_interval_95': np.quantile(paired[indices].mean(1), [.025, .975], axis=0).tolist(),
                          'native_and_regions': item['summary']['epoch500_minus_SOURCE']['signed_mean_risk']}
    conditions = data['conditions']
    train = np.array([r['features'] for r in conditions['train']])
    valid = np.array([r['features'] for r in conditions['validation']])
    tail = train[[6400 <= r['sample_index'] < 8000 for r in conditions['train']]]
    std = np.array(conditions['train_feature_std'])
    centroid = {}
    for label, indices in (('observations', slice(0, 2)), ('targets', slice(2, 12))):
        centroid[label] = {name: float(np.linalg.norm(((valid.mean(0) - pool.mean(0)) / std)[indices]))
                           for name, pool in (('full_train', train), ('last20pct_train', tail))}
    return {'schema': 'car_scale_followup_diagnostic_readout_v1', 'status': 'completed', 'inputs': identities,
            'frozen_train_scale_unit': UNIT, 'scale': {k: data[k]['summary'] for k in ('train', 'validation')},
            'actual_adam_finite_changes': action, 'actual_adam_first_order': first_order,
            'action_interpretation': '13 independently reset J2 TRAIN-law selected-support shadows; sample-weighted descriptive changes, not 13 independent datasets or a continuation trajectory; no shadow-action uncertainty interval inferred.',
            'spatial_feedback_gap': support, 'condition_centroid_distances_in_train_units': centroid,
            'condition_interpretation': 'Fixed retrospective features; no validation fitting, causal claim, or justified recency weighting.',
            'test_locked': True}


def figures(readout, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
    destination = output / 'figures'
    destination.mkdir()
    def save(fig, name):
        fig.savefig(destination / (name + '.pdf'), bbox_inches='tight')
        fig.savefig(destination / (name + '.png'), dpi=160, bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    x = np.arange(3)
    for ax, split in zip(axes, ('train', 'validation')):
        for offset, epoch, color in ((-.18, 'epoch100', '#3267a8'), (.18, 'epoch500', '#d2702e')):
            values = [readout['scale'][split][epoch][key] for key in DEFINITIONS]
            y = np.array([v['mean_raw_change'] / UNIT for v in values])
            ci = np.array([v['interval_95_raw'] for v in values]).T / UNIT
            ax.bar(x + offset, y, .34, color=color, label=epoch.replace('epoch', 'Epoch '))
            ax.errorbar(x + offset, y, yerr=np.vstack((y-ci[0], ci[1]-y)), fmt='none', color='#222', capsize=3)
        ax.axhline(0, color='#555', lw=.8)
        ax.set_xticks(x, ['Published core', 'Prefit core', 'Prefit + triples'])
        ax.set_title(('TRAIN800' if split == 'train' else 'VALIDATION100') + ', native cached J4')
        ax.set_ylabel('Scale change / frozen TRAIN unit (lower is better)')
        ax.legend(frameon=False)
    save(fig, 'matched_train_validation_scale')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for ax, split in zip(axes, ('train', 'validation')):
        item = readout['spatial_feedback_gap'][split]
        for offset, region, label, color in ((-.25, 'selected_region', 'Fixed Q4096 region', '#3267a8'),
                                           (0, 'complement_region', 'Native complement', '#d2702e'),
                                           (.25, 'native', 'Full native mask', '#6c777e')):
            ax.bar(np.arange(5)+offset, item['native_and_regions'][region]['mean_over_original_frozen_allowance'], .24, label=label, color=color)
        ax.axhline(0, color='#555', lw=.8)
        ax.axhline(1, color='#555', lw=.8, ls='--', label='Original allowance')
        ax.set_xticks(np.arange(5), item['field_order'])
        ax.set_title(split.upper() + ': original epoch 500 minus SOURCE')
        ax.set_ylabel('Signed field-risk change / original allowance')
        ax.legend(frameon=False, fontsize=8)
    save(fig, 'native_field_support_gap')
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    labels = ('Same-query training', 'Independent-query training', 'Complete published core', 'Complete prefit core', 'Complete prefit + triples')
    for key, label in zip(readout['actual_adam_finite_changes'], labels):
        ax.plot(np.arange(1, 14), readout['actual_adam_finite_changes'][key]['batch_changes'], marker='o', ms=3, lw=1, label=label)
    ax.axhline(0, color='#555', lw=.8)
    ax.set(xlabel='Independently reset TRAIN shadow batch (13 batches cover 800 samples)', ylabel='Raw Scale finite change after one actual Adam action', title='Epoch-500 model and actual Adam/PI state; J2, Q4096')
    ax.legend(frameon=False, fontsize=8, ncols=2)
    save(fig, 'actual_adam_complete_scale_changes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start, cpu = time.time(), time.process_time()
    campaign, output = args.campaign.resolve(strict=True), args.output.resolve()
    if campaign / 'assessment' not in output.parents or output.exists():
        raise ValueError('a new campaign-owned assessment output is required')
    readout = assemble(campaign)
    output.mkdir(parents=True)
    figures(readout, output)
    readout.update(script_sha256=digest(__file__), elapsed_seconds=time.time()-start, process_cpu_seconds=time.process_time()-cpu)
    (output / 'readout.json').write_text(json.dumps(readout, indent=2, allow_nan=False) + '\n')
    (output / 'figure_index.md').write_text('# Completed Scale diagnostic figures\n\n'
        'These completed diagnostics select a learning experiment; they do not establish its efficacy. Negative Scale changes are better. TEST remains locked.\n\n'
        '![Matched complete Scale changes on TRAIN800 and VALIDATION100](figures/matched_train_validation_scale.png)\n\n'
        'Figure 1. Identical complete definitions improve on TRAIN but deteriorate on the reused validation panel; intervals resample matched conditions within each split, without asserting independent trajectories. [PDF master](figures/matched_train_validation_scale.pdf)\n\n'
        '![Cached native signed field risk partitioned into fixed support and complement](figures/native_field_support_gap.png)\n\n'
        'Figure 2. The fixed-support field-risk gap is clear on TRAIN, while validation deterioration spans both regions; region means use their own physical mass and the native mean uses the complete physical mask. [PDF master](figures/native_field_support_gap.pdf)\n\n'
        '![Finite complete Scale changes for all 13 actual Adam shadow batches](figures/actual_adam_complete_scale_changes.png)\n\n'
        'Figure 3. Every independently reset J2 training-law shadow lowers both queried and complete Scale; these are real Adam actions at fixed parent state, not a sequential continuation or native J4 inference. [PDF master](figures/actual_adam_complete_scale_changes.pdf)\n')
    print(json.dumps({'output': str(output), 'readout_sha256': digest(output/'readout.json'), 'wall_seconds': readout['elapsed_seconds']}))


if __name__ == '__main__':
    main()
