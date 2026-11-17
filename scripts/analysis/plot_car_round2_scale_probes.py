#!/usr/bin/env python3
"""Plot completed, independently reset one-update Scale probes without inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefit', type=Path, required=True)
    parser.add_argument('--family', choices=('ffm', 'senseiver'), required=True)
    parser.add_argument('--target-label', choices=('provisional target', 'final selected target'),
                        default='provisional target')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    started, cpu = time.time(), time.process_time()
    source = args.prefit.resolve(strict=True)
    receipt_path = source.with_suffix(source.suffix + '.receipt.json')
    completed = json.loads(receipt_path.read_text())
    if completed.get('status') != 'completed':
        raise ValueError('completed prefit receipt required')
    doc = json.loads(source.read_text())
    probe = doc['scale_probe']
    rows = probe['rows']
    seeds = sorted({row['seed'] for row in rows})
    if len(seeds) != 3 or probe.get('reserved_task_used') is not False:
        raise ValueError('three matched, reserved-task-free RNG manifests required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter

    values = []
    for kind in ('full_model', 'residual_head'):
        for ratio in (1, 2, 4):
            group = [row for row in rows if row['parameterization'] == kind and row['ratio'] == ratio]
            for seed in seeds:
                selected = [row for row in group if row['seed'] == seed]
                ids = [sample for row in selected for sample in row['sample_ids']]
                if ids != doc['binding']['sample_ids'] or any(row['sample_count'] != len(row['sample_ids']) for row in selected):
                    raise ValueError('every plotted cell must cover the exact matched TRAIN panel')
                count = sum(row['sample_count'] for row in selected)
                before = sum(row['scale_before'] * row['sample_count'] for row in selected) / count
                after = sum(row['scale_after'] * row['sample_count'] for row in selected) / count
                values.append({'parameterization': kind, 'ratio': ratio, 'seed': seed,
                               'sample_count': count, 'before': before, 'after': after,
                               'decrease': before - after,
                               'pooled_field_sensor_feasible': all(row['field_and_sensor_feasible'] for row in selected)})
    if len(rows) != 234 or any(value['sample_count'] != 800 for value in values):
        raise ValueError('this figure requires the actual B64 TRAIN800, 234-trial probe')
    threshold = probe['decision']['minimum_pooled_decrease']
    decreases = [value['decrease'] for value in values]
    if not np.isfinite(decreases).all() or threshold <= 0:
        raise ValueError('finite plotted decreases and positive frozen response threshold required')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    figure, axes = plt.subplots(1, 2, figsize=(10, 5.3), sharey=True)
    floor = min(threshold, max(abs(value) for value in decreases)) * .01
    for axis, kind, title in zip(axes, ('full_model', 'residual_head'),
                                  ('Full model', 'Frozen SOURCE + residual head')):
        axis.axhline(0, color='#8a8a8a', linewidth=.7)
        axis.axhline(threshold, color='#333333', linestyle='--', linewidth=1.1,
                     label='Required decrease')
        for index, ratio in enumerate((1, 2, 4)):
            selected = [value for value in values if value['parameterization'] == kind and value['ratio'] == ratio]
            for offset, value, marker in zip((-.12, 0, .12), selected, ('o', 's', '^')):
                axis.scatter(index + offset, value['decrease'], marker=marker, s=34,
                             facecolor='white', edgecolor='#33658a', linewidth=1, zorder=3)
            pooled = sum(value['decrease'] * value['sample_count'] for value in selected) / sum(value['sample_count'] for value in selected)
            axis.scatter(index, pooled, marker='D', s=29, color='#172f40', zorder=4)
            axis.annotate(f'{pooled:.3g}', (index, pooled), xytext=(0, 12),
                          textcoords='offset points', ha='center', fontsize=9, color='#172f40')
        axis.set(title=title, xlabel='Scale joint coefficient multiplier', xticks=range(3),
                 xticklabels=['1', '2', '4'], xlim=(-.4, 2.4))
        axis.set_yscale('symlog', linthresh=floor)
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f'{value:.0e}' if value else '0'))
        axis.grid(axis='y', color='#e4e4e4', linewidth=.65)
        axis.set_axisbelow(True)
    low, high = min(0., min(decreases)), max(threshold, max(decreases))
    axes[0].set_ylim(low * 2 - floor, high * 3)
    limit = max(abs(low * 2 - floor), high * 3)
    first_exponent = int(np.ceil(np.log10(floor)))
    last_exponent = int(np.floor(np.log10(limit)))
    exponents = range(first_exponent, last_exponent + 1)
    ticks = [0., *[10. ** exponent for exponent in exponents if 10. ** exponent <= high * 3]]
    ticks += [-10. ** exponent for exponent in exponents if -10. ** exponent >= low * 2 - floor]
    axes[0].set_yticks(sorted(ticks))
    axes[0].set_ylabel('Complete Scale decrease (symmetric log scale)')
    axes[0].legend(loc='lower left', frameon=False, fontsize=9)
    label = 'FFM' if args.family == 'ffm' else 'Senseiver'
    figure.suptitle(f'{label}: one accepted-update Scale response', x=.08, ha='left', fontsize=14)
    figure.text(.08, .89, f'TRAIN800 · three matched RNG manifests · threshold {threshold:.5g} · {args.target_label}', fontsize=10)
    figure.text(.08, .075, 'Open marks: RNG manifests; diamonds and labels: sample-weighted pooled decreases.', fontsize=9)
    figure.text(.08, .04, 'Each trial restores initial state. These are local responses, not a learning trajectory or independent trained replicas.', fontsize=9)
    figure.subplots_adjust(left=.1, right=.98, bottom=.2, top=.8, wspace=.12)
    pdf, png = args.output_dir / 'scale_response.pdf', args.output_dir / 'scale_response.png'
    figure.savefig(pdf)
    figure.savefig(png, dpi=180)
    plt.close(figure)
    source_relative = os.path.relpath(source, args.output_dir)
    caption = f'{label} complete Scale decreases for the {args.target_label} after one production optimizer update, with initial state restored for every trial. Every point pools the same 800 TRAIN conditions; RNG-manifest spread is descriptive. The frozen response threshold applies to each parameterization and matched manifest. The saved decision retains ratio {probe["decision"]["chosen_ratio"]}; no mature efficacy conclusion follows.'
    (args.output_dir / 'README.md').write_text(f'# {label} Scale response: {args.target_label}\n\nFigure index: one two-panel figure.\n\n![{label} one-update Scale response for full-model and residual-head paths](scale_response.png)\n\n{caption}\n\n[PDF master](scale_response.pdf) · [Completed probe source]({source_relative})\n')
    (args.output_dir / 'plotted_values.json').write_text(json.dumps({'source_scale': probe['source_scale'],
        'decision': probe['decision'], 'cells': values}, indent=2, allow_nan=False) + '\n')
    shutil.copyfile(Path(__file__), args.output_dir / 'render_scale_probe.py')
    result = {'schema': 'car_round2_scale_probe_figure_v1', 'status': 'completed', 'family': args.family,
              'target_label': args.target_label,
              'prefit': str(source), 'prefit_sha256': sha(source), 'prefit_receipt_sha256': sha(receipt_path),
              'script_sha256': sha(Path(__file__)), 'trial_count': len(rows), 'condition_count': 800,
              'gpu_lease': None, 'start_unix': started, 'end_unix': time.time(),
              'cpu_seconds': time.process_time() - cpu,
              'files_sha256': {path.name: sha(path) for path in sorted(args.output_dir.iterdir())}}
    result['wall_seconds'] = result['end_unix'] - started
    (args.output_dir / 'job_receipt.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(args.output_dir / 'job_receipt.json')


if __name__ == '__main__':
    main()
