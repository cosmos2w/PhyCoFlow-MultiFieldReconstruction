"""Analyze saved Round 7 scalar evidence without importing a model or torch.

A diagnostic window is one reporting epoch. The Scale-only explanation flag
uses the mean live and shadow projections on their *paired* component-probe
updates in each window, and requires positive mean shadow benefit in at least
two windows. These are temporal explanation flags, not efficacy tests.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

FAMILIES = ('global_distribution', 'multiscale_spectrum', 'topology')
COMPONENTS = ('signed', 'joint', 'self', 'rho', 'joint_attraction', 'joint_separation',
              'temperature_risk', 'temperature_violation')


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError('expected a finite numeric diagnostic')
    return float(value)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(_number(v) for v in values)
    if not ordered:
        return {'count': 0, 'mean': None, 'median': None, 'p10': None, 'p90': None,
                'minimum': None, 'maximum': None, 'negative_count': 0, 'negative_fraction': None}
    def quantile(p: float) -> float:
        position = (len(ordered) - 1) * p
        i = int(position)
        return ordered[i] + (ordered[min(i + 1, len(ordered) - 1)] - ordered[i]) * (position - i)
    negative = sum(v < 0 for v in ordered)
    return {'count': len(ordered), 'mean': statistics.fmean(ordered), 'median': statistics.median(ordered),
            'p10': quantile(.1), 'p90': quantile(.9), 'minimum': ordered[0], 'maximum': ordered[-1],
            'negative_count': negative, 'negative_fraction': negative / len(ordered)}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f'{path}:{line_number} is not an object')
                rows.append(value)
    return rows


def analyze_updates(rows: Sequence[Mapping[str, Any]], *, arm: str) -> dict[str, Any]:
    """Summarize one arm's saved updates; missing evidence remains unassessed."""
    windows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    post: dict[str, list[float]] = {k: [] for k in FAMILIES}
    components: dict[str, list[float]] = {k: [] for k in COMPONENTS}
    component_norms: dict[str, list[float]] = {k: [] for k in COMPONENTS}
    shadow_components: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    times = {'diagnostic_seconds': 0.0, 'gradient_and_direction_seconds': 0.0,
             'shadow_seconds': 0.0, 'extra_gradient_evaluations': 0, 'shadow_proposals': 0}
    seen: set[int] = set()
    warmup_records = 0
    for row in rows:
        step, epoch = row.get('step'), row.get('epoch')
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise ValueError('invalid update step')
        if step in seen:
            raise ValueError(f'duplicate update step {step}')
        seen.add(step)
        gradient = row.get('gradient')
        if not isinstance(gradient, Mapping):
            raise ValueError('update record lacks gradient mapping')
        measured = gradient.get('round7_actual_step')
        if measured is None:
            continue
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1 or not isinstance(measured, Mapping):
            raise ValueError('invalid measured epoch or actual-step record')
        values = measured.get('family_benefits')
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError('actual-step benefits must contain the ordered three families')
        values = [_number(v) for v in values]
        warmup = step <= 76
        warmup_records += int(warmup)
        entry = {'step': step, 'live': values, 'shadows': {}, 'components': {},
                 'component_gradient_norms': {}, 'shadow_components': {}}
        if not warmup:
            for name, value in zip(FAMILIES, values):
                post[name].append(value)
        for name, value in measured.get('component_products', {}).items():
            if name not in COMPONENTS:
                raise ValueError(f'unknown component product {name}')
            entry['components'][name] = _number(value)
            if not warmup:
                components[name].append(entry['components'][name])
        for name, value in measured.get('component_gradient_norms', {}).items():
            if name not in COMPONENTS:
                raise ValueError(f'unknown component gradient norm {name}')
            value = _number(value)
            if value < 0:
                raise ValueError('gradient norm must be nonnegative')
            entry['component_gradient_norms'][name] = value
            if not warmup:
                component_norms[name].append(value)
        for name, shadow in measured.get('shadows', {}).items():
            if not isinstance(shadow, Mapping) or not isinstance(shadow.get('family_benefits'), (list, tuple)) or len(shadow['family_benefits']) != 3:
                raise ValueError('invalid shadow benefit record')
            entry['shadows'][name] = [_number(v) for v in shadow['family_benefits']]
            entry['shadow_components'][name] = {}
            for component, product in shadow.get('component_products', {}).items():
                if component not in COMPONENTS:
                    raise ValueError(f'unknown shadow component product {component}')
                value = _number(product)
                entry['shadow_components'][name][component] = value
                if not warmup:
                    shadow_components[name][component].append(value)
            times['shadow_seconds'] += _number(shadow.get('seconds', 0.0))
            times['shadow_proposals'] += 1
        for name in ('diagnostic_seconds', 'gradient_and_direction_seconds'):
            times[name] += _number(measured.get(name, 0.0))
        times['extra_gradient_evaluations'] += int(measured.get('extra_gradient_evaluations', 0))
        if not warmup:
            windows[epoch].append(entry)

    window_reports = {}
    qualified = []
    for epoch, entries in sorted(windows.items()):
        paired = [(x['live'][1], x['shadows']['scale_only'][1]) for x in entries if 'scale_only' in x['shadows']]
        live_mean = statistics.fmean(x for x, _ in paired) if paired else None
        shadow_mean = statistics.fmean(y for _, y in paired) if paired else None
        assessed = len(entries) == 13 and len(paired) == 3 and shadow_mean > 0
        hit = live_mean <= .5 * shadow_mean if assessed else None
        if assessed:
            qualified.append(hit)
        window_reports[str(epoch)] = {
            'step_count': len(entries),
            'families': {name: _distribution([x['live'][i] for x in entries]) for i, name in enumerate(FAMILIES)},
            'scale_only_comparison': {'paired_count': len(paired), 'live_mean': live_mean,
                                      'shadow_mean': shadow_mean, 'assessed': assessed, 'at_most_half': hit},
            'components': {name: _distribution([x['components'][name] for x in entries if name in x['components']])
                           for name in COMPONENTS},
            'component_gradient_norms': {name: _distribution([
                x['component_gradient_norms'][name] for x in entries if name in x['component_gradient_norms']])
                for name in COMPONENTS},
            'scale_only_component_pairs': {
                component: [{'step': x['step'], 'live': x['components'][component],
                             'shadow': x['shadow_components']['scale_only'][component]}
                            for x in entries if component in x['components']
                            and component in x['shadow_components'].get('scale_only', {})]
                for component in COMPONENTS},
        }
    negative_flags = {}
    for name in FAMILIES:
        distribution = _distribution(post[name])
        negative_flags[name] = None if not distribution['count'] else distribution['negative_fraction'] >= .1
    return {'arm': arm, 'family_order': list(FAMILIES), 'warmup_records_excluded': warmup_records,
            'post_warmup_families': {name: _distribution(post[name]) for name in FAMILIES},
            'post_warmup_components': {name: _distribution(components[name]) for name in COMPONENTS},
            'post_warmup_component_gradient_norms': {name: _distribution(component_norms[name]) for name in COMPONENTS},
            'post_warmup_shadow_components': {shadow: {component: _distribution(values)
                                                       for component, values in grouped.items()}
                                               for shadow, grouped in shadow_components.items()},
            'windows': window_reports, 'negative_at_least_ten_percent': negative_flags,
            'scale_only_at_most_half_in_two_windows': None if len(qualified) < 2 else sum(qualified) >= 2,
            'scale_only_assessed_windows': len(qualified), 'telemetry_cost': times,
            'semantics': {'scale_comparison': 'paired live and Scale-only shadow whole-Scale gradient products; means within each complete 13-update epoch with three paired probes',
                          'null_flag': 'unassessed', 'joint_separation': 'signed negative separation contribution',
                          'shape_basis': 'arm-specific objective; Shape products are not a common cross-arm metric',
                          'cost': 'diagnostic_seconds includes live optimizer time and overlaps shadow_seconds; do not sum them',
                          'step_products': 'local predictions from actual parameter displacement, not finite loss changes'}}


def _loss_series(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, list[list[float]]]]:
    series = {name: {'raw': [], 'weighted': []} for name in FAMILIES}
    for row in rows:
        epoch = row.get('epoch')
        if isinstance(epoch, bool) or not isinstance(epoch, (int, float)) or not math.isfinite(epoch):
            continue
        for family in FAMILIES:
            for output, suffix in [('raw', 'raw'), ('weighted', 'weighted_contribution')]:
                value = row.get(f'coherence_family/{family}/{suffix}')
                if value is not None:
                    series[family][output].append([float(epoch), _number(value)])
    return series


def _plot_actual(reports: Mapping[str, dict[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    arms = list(reports)
    fig, axes = plt.subplots(3, len(arms), figsize=(4.1 * len(arms), 7.4), squeeze=False, sharex='col')
    for column, arm in enumerate(arms):
        windows = reports[arm]['windows']
        for row, family in enumerate(FAMILIES):
            ax = axes[row][column]
            if not windows:
                ax.text(.5, .5, 'No measured data', transform=ax.transAxes,
                        ha='center', va='center', color='0.45')
            for epoch, record in windows.items():
                dist = record['families'][family]
                if dist['count']:
                    x = int(epoch)
                    ax.vlines(x, dist['p10'], dist['p90'], color='#405f86', lw=1.5)
                    ax.plot(x, dist['median'], 'o', color='#1d4168', ms=4)
                    if row == 1:
                        comparison = record['scale_only_comparison']
                        if comparison['paired_count']:
                            ax.plot(x, comparison['live_mean'], 'x', color='#ad4f26', ms=6)
                            ax.plot(x, comparison['shadow_mean'], '^', color='#558243', ms=4)
            ax.axhline(0, color='0.55', lw=.7)
            ax.set_ylabel(('State', 'Scale', 'Shape')[row] + '\nprojected benefit')
            ax.grid(alpha=.2)
            if row == 0:
                ax.set_title(f'{arm}: actual step')
            if row == 2:
                ax.set_xlabel('Reporting epoch')
    fig.suptitle('Actual projections: median and 10–90%; Scale paired live mean ×, same-state Scale-only mean △', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .95))
    for suffix in ('pdf', 'png'):
        fig.savefig(output / f'actual_step_projections.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def _plot_losses(series: Mapping[str, dict[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    arms = list(series)
    fig, axes = plt.subplots(3, len(arms), figsize=(4.1 * len(arms), 7.4), squeeze=False, sharex='col')
    for column, arm in enumerate(arms):
        for row, family in enumerate(FAMILIES):
            ax = axes[row][column]
            if not any(series[arm][family].values()):
                ax.text(.5, .5, 'No training data', transform=ax.transAxes,
                        ha='center', va='center', color='0.45')
            for name, style in [('raw', '--'), ('weighted', '-')]:
                points = series[arm][family][name]
                if points:
                    ax.plot([p[0] for p in points], [p[1] for p in points], style, lw=.7, alpha=.8, label=name)
            ax.set_ylabel(('State', 'Scale', 'Shape')[row] + ' training loss')
            ax.grid(alpha=.2)
            if row == 0:
                ax.set_title(arm)
                if ax.lines:
                    ax.legend(fontsize=7)
            if row == 2:
                ax.set_xlabel('Reporting epoch')
    fig.suptitle('Sampled training family losses: raw and weighted; Shape uses each arm’s own definition', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .95))
    for suffix in ('pdf', 'png'):
        fig.savefig(output / f'family_training_losses.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def generate_analysis(arms: Mapping[str, str | Path], output_dir: str | Path) -> dict[str, Any]:
    """Read run-owned logs and write scalar JSON and compact PDF/PNG figures.

    Missing logs are reported as absent evidence. Call after a checkpoint or
    terminal handoff; no live trainer state is queried or changed.
    """
    if not arms:
        raise ValueError('at least one arm path is required')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reports, series, sources = {}, {}, {}
    for arm, root in arms.items():
        root = Path(root)
        update_path = root / 'logs/round7_updates.jsonl'
        full = root / 'logs/full_updates.jsonl'
        family = root / 'logs/family_updates.jsonl'
        training_path = full if full.is_file() else family
        update_rows = read_jsonl(update_path)
        training_rows = read_jsonl(training_path)
        reports[arm] = analyze_updates(update_rows, arm=arm)
        series[arm] = _loss_series(training_rows)
        sources[arm] = {'actual_updates': str(update_path) if update_path.is_file() else None,
                        'family_updates': str(training_path) if training_path.is_file() else None,
                        'actual_record_count': len(update_rows), 'family_record_count': len(training_rows)}
    report = {'schema': 'round7_saved_evidence_analysis_v1', 'sources': sources, 'arms': reports,
              'figures': {'actual_step': 'actual_step_projections.png', 'actual_step_pdf': 'actual_step_projections.pdf',
                          'family_losses': 'family_training_losses.png', 'family_losses_pdf': 'family_training_losses.pdf'}}
    _plot_actual(reports, output)
    _plot_losses(series, output)
    (output / 'analysis.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return report
