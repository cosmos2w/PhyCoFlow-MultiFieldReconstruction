"""Build the final Round 7 report from frozen reductions and receipts only."""
from __future__ import annotations
import json
import hashlib
import re
from pathlib import Path
import subprocess
import time


def _file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def validate_report_evidence(test_receipt, selection_path):
    """Verify frozen receipt and artifact bytes without importing model code."""
    test = json.loads(Path(test_receipt).read_text())
    selection = json.loads(Path(selection_path).read_text())
    if test.get('schema') != 'round7_fixed_test_v1' or test.get('selection_sha256') != _file_sha256(selection_path):
        raise ValueError('TEST receipt does not match frozen selection')
    expected = {selection['source_checkpoint_sha256']}
    expected.update(item['checkpoint_sha256'] for item in (selection.get('endpoint'), selection.get('B1000')) if item)
    if expected != {item['checkpoint_sha256'] for item in test['endpoints'].values()}:
        raise ValueError('TEST endpoint set differs from frozen selection')
    for item in test['endpoints'].values():
        for key in ('reductions', 'native_receipt'):
            if _file_sha256(item[key]) != item[key + '_sha256']:
                raise ValueError('TEST evidence bytes changed')
    return test


def _comparison_figures(campaign, output, selection, test_source, test_selected, q_ratios, native_ratios):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .round7_summary import summarize_reductions, compare_summaries
    fields = ['CH4', 'CO', 'T', 'U_1', 'p']
    source_receipt = json.loads((campaign / 'artifacts/reference_SOURCE.json').read_text())
    source_val = summarize_reductions(source_receipt['validation']['reductions'], fields)
    source_train = summarize_reductions(source_receipt['train']['reductions'], fields)
    shape_key = 'shape.fair' if 'fair' in source_val['observer_identities'] else 'shape.legacy'
    fixed_scales = json.loads((campaign / 'artifacts/qualification_contract.json').read_text())['projection_scales']['B']
    shape_scale = (source_receipt['train']['fair_shape_attraction_mean'] if shape_key == 'shape.fair'
                   else fixed_scales['topology'])
    scales = {'state': fixed_scales['global_distribution'], 'scale': fixed_scales['multiscale_spectrum'], shape_key: shape_scale}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    dual_axis = axes[1, 1].twinx()
    for arm in ('B', 'P', 'T'):
        owned = campaign / 'artifacts' / f'owned_run_{arm}.json'
        if not owned.exists():
            continue
        run = Path(json.loads(owned.read_text())['run'])
        series = []
        for path in (run / 'evaluation').glob('validation_epoch_*/reductions.json'):
            epoch = int(path.parent.name.rsplit('_', 1)[1])
            summary = summarize_reductions(path, fields)
            comparison = compare_summaries(source_val, summary, positive_scales=scales)['changes']
            q = max(comparison['field.' + f + '.mse']['candidate'] /
                    comparison['field.' + f + '.mse']['reference'] for f in fields)
            series.append((epoch, comparison['state']['source_units_change'],
                           comparison['scale']['fractional_change'],
                           comparison[shape_key]['source_units_change'], q))
        if series:
            series.sort()
            for i, ax in enumerate(axes.flat, start=1):
                ax.plot([row[0] for row in series], [row[i] for row in series], marker='.', label=arm)
        updates = run / 'logs/full_updates.jsonl'
        if updates.exists():
            per_epoch = {}
            for line in updates.read_text().splitlines():
                row = json.loads(line)
                dual = row.get('multipliers_after')
                if isinstance(dual, list) and dual:
                    per_epoch.setdefault(int(row['epoch']), []).append(max(dual))
            if per_epoch:
                epochs = sorted(per_epoch)
                dual_axis.plot(epochs, [sum(per_epoch[e]) / len(per_epoch[e]) for e in epochs],
                               linestyle=':', linewidth=.7, alpha=.6)
    labels = ['State change / SOURCE scale', 'Complete Scale relative change',
              f'{shape_key} change / SOURCE scale', 'Maximum field MSE / SOURCE']
    for ax, title, threshold in zip(axes.flat, labels, (.02, -.01, .02, 1.10)):
        ax.axhline(threshold, color='black', linestyle='--', linewidth=.8)
        ax.set_title(title); ax.set_xlabel('Reporting epoch'); ax.grid(alpha=.2)
        if ax.lines and len(ax.lines) > 1:
            ax.legend()
    dual_axis.set_ylabel('Mean maximum dual multiplier (dotted)', fontsize=8)
    fig.savefig(output / 'validation_trajectory.pdf')
    fig.savefig(output / 'validation_trajectory.png', dpi=150)
    plt.close(fig)

    if test_selected is None:
        return
    binding = json.loads(Path(selection['endpoint']['binding']).read_text())
    selected_val = summarize_reductions(binding['validation_reductions'], fields)
    val_change = compare_summaries(source_val, selected_val)['changes']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    keys = [('signed', 'multiscale_spectrum.signed_band_energy_score.conditional', 1.),
            ('joint', 'multiscale_spectrum.joint_spectral_mean_energy_score.conditional', .25),
            ('self', 'multiscale_spectrum.relation_self_energy_score.conditional', .025605941746750686),
            ('rho', 'multiscale_spectrum.relation_rho_crps.conditional', .21887008719850062)]
    x = list(range(len(keys)))
    for offset, label, panel in ((-.18, 'SOURCE', source_val), (.18, 'selected', selected_val)):
        vals = [panel['loss_means'][key] * weight for _, key, weight in keys]
        axes[0, 0].bar([v + offset for v in x], vals, width=.36, label=label)
    axes[0, 0].set_xticks(x, [k[0] for k in keys]); axes[0, 0].set_title('Weighted complete Scale terms'); axes[0, 0].legend()
    shape_names = [name for name in ('fair', 'legacy') if name in source_val['shape_observers']]
    shape_labels = [(name, part) for name in shape_names for part in ('self', 'mutual')]
    x = list(range(len(shape_labels)))
    for offset, label, panel in ((-.18, 'SOURCE', source_val), (.18, 'selected', selected_val)):
        values = [panel['shape_observers'][name]['losses']['topology.' + part + '.persistence'] for name, part in shape_labels]
        axes[0, 1].bar([v + offset for v in x], values, width=.36, label=label)
    axes[0, 1].set_xticks(x, [f'{n}\n{p}' for n, p in shape_labels]); axes[0, 1].set_title('Shape observer terms'); axes[0, 1].legend()
    x = list(range(len(fields)))
    train_ratios = binding['train_primary_mse_ratios']
    val_ratios = {f: val_change['field.' + f + '.mse']['candidate'] / val_change['field.' + f + '.mse']['reference'] for f in fields}
    val_native = dict(zip(binding['native_comparison']['field_names'], binding['native_comparison']['primary_mse_ratio']))
    for offset, label, vals in ((-.32, 'TRAIN Q4096', train_ratios), (-.16, 'VAL Q4096', val_ratios),
                                (0., 'VAL native', val_native), (.16, 'TEST Q4096', q_ratios),
                                (.32, 'TEST native', native_ratios)):
        axes[1, 0].bar([v + offset for v in x], [vals[f] for f in fields], width=.15, label=label)
    axes[1, 0].axhline(1.1, color='black', linestyle='--', linewidth=.8)
    axes[1, 0].set_xticks(x, fields); axes[1, 0].set_title('Primary MSE / SOURCE'); axes[1, 0].legend(fontsize=7)
    for offset, label, panel in ((-.18, 'SOURCE', test_source), (.18, 'selected', test_selected)):
        values = panel['j2_physical_measure_model_units']['variance_proxy']
        axes[1, 1].bar([v + offset for v in x], values, width=.36, label=label)
    axes[1, 1].set_xticks(x, fields); axes[1, 1].set_title('TEST Q4096 weighted variance proxy'); axes[1, 1].legend()
    fig.savefig(output / 'matched_components_fidelity.pdf')
    fig.savefig(output / 'matched_components_fidelity.png', dpi=150)
    plt.close(fig)


def _compute_coverage_figure(campaign, output, gpu_seconds, attempt_count):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    log = campaign / 'artifacts/attempts.jsonl'
    counts = {'qualification': 0, 'B': 0, 'P': 0, 'T': 0}
    if log.exists():
        for line in log.read_text().splitlines():
            row = json.loads(line)
            key = 'qualification' if row['phase'] == 'qualification' else row['arm']
            if key in counts:
                counts[key] += 1
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    axes[0].bar(list(counts), list(counts.values()))
    axes[0].set_title(f'Live optimizer attempts ({attempt_count} charged)')
    axes[0].tick_params(axis='x', rotation=20)
    axes[1].bar(['TRAIN', 'VALIDATION', 'TEST'], [800, 100, 100])
    axes[1].set_title(f'Fixed condition counts; GPU union {gpu_seconds / 3600:.2f} h')
    axes[1].set_ylabel('Conditions; each panel is approximately 10%')
    fig.savefig(output / 'compute_coverage.pdf')
    fig.savefig(output / 'compute_coverage.png', dpi=150)
    plt.close(fig)


_SCALE_TERMS = {
    'signed': ('multiscale_spectrum.signed_band_energy_score.conditional', 1.),
    'joint': ('multiscale_spectrum.joint_spectral_mean_energy_score.conditional', .25),
    'self': ('multiscale_spectrum.relation_self_energy_score.conditional', .025605941746750686),
    'rho': ('multiscale_spectrum.relation_rho_crps.conditional', .21887008719850062),
}


def _panel_snapshot(panel):
    """Keep raw scores and all recorded decompositions, with declared weights."""
    losses = panel.get('loss_means', {})
    diagnostics = panel.get('component_diagnostics', {})
    terms, attraction_separation = {}, {}
    for name, (key, weight) in _SCALE_TERMS.items():
        terms[name] = losses[key] * weight if key in losses else None
        d = diagnostics.get(key, {})
        if name == 'signed':
            attraction_separation[name] = {'attraction': d.get('weighted_attraction'),
                                          'separation': d.get('weighted_separation')}
        else:
            attraction_separation[name] = {
                'attraction': d['attraction'] * weight if d.get('attraction') is not None else None,
                'separation': d['repulsion'] * weight if d.get('repulsion') is not None else None}
    return {'scores': panel.get('scores', {}), 'fields': panel.get('fields', {}),
            'weighted_scale_terms': terms, 'weighted_scale_attraction_separation': attraction_separation,
            'shape_observers': panel.get('shape_observers', {}),
            'loss_means': losses, 'family_diagnostics': panel.get('family_diagnostics', {}),
            'component_diagnostics': diagnostics}


def _fixed_panel_evidence(campaign, fields, summarize, compare):
    artifacts = campaign / 'artifacts'
    reference = artifacts / 'reference_SOURCE.json'
    if not reference.exists():
        return {}
    receipt = json.loads(reference.read_text())
    panels = {split: summarize(receipt[split]['reductions'], fields) for split in ('train', 'validation')}
    result = {'SOURCE': {split: _panel_snapshot(panel) for split, panel in panels.items()}}
    for arm in ('B', 'P', 'T'):
        owned = artifacts / f'owned_run_{arm}.json'
        if not owned.exists():
            continue
        run = Path(json.loads(owned.read_text())['run'])
        for epoch in (500, 1000):
            if not (artifacts / f'{arm}_{epoch}_complete.json').exists():
                continue
            binding = json.loads((run / 'evaluation' / f'checkpoint_binding_{epoch}.json').read_text())
            item = {'checkpoint_sha256': binding.get('checkpoint_sha256'), 'splits': {}}
            for split in ('train', 'validation'):
                panel = summarize(binding[f'{split}_reductions'], fields)
                item['splits'][split] = {'summary': _panel_snapshot(panel),
                    'source_comparison': compare(panels[split], panel)}
            result.setdefault(arm, {})[str(epoch)] = item
    return result


def _test_panel_evidence(entries, test, source, fields, summarize, compare):
    results = {}
    for name, entry in entries.items():
        panel = source if name == 'SOURCE' else summarize(entry['reductions'], fields)
        result = {'checkpoint_sha256': entry['checkpoint_sha256'], 'summary': panel}
        if name != 'SOURCE':
            result['source_comparison'] = compare(source, panel)
            result['native_comparison'] = test['native_comparisons'][name]
        results[name] = result
    return results


def _timing_evidence(campaign, gpu_seconds):
    ledger = campaign / 'artifacts/budget_ledger.json'
    intervals = json.loads(ledger.read_text()).get('intervals', []) if ledger.exists() else []
    closed = [item for item in intervals if item.get('end') is not None]
    stages = {}
    for item in closed:
        stage = item.get('stage', 'unknown')
        stages[stage] = stages.get(stage, 0.) + item['end'] - item['start']
    result = {'occupied_gpu_union_seconds': gpu_seconds,
              'elapsed_first_to_last_gpu_stage_seconds': (max(x['end'] for x in closed) - min(x['start'] for x in closed)) if closed else None,
              'stage_wall_seconds_summed_may_overlap': stages,
              'unclosed_gpu_intervals': len(intervals) - len(closed)}
    for key, filename in (('calibration', 'prepare_complete.json'),
                          ('source_evaluation', 'reference_SOURCE.json'),
                          ('s100_evaluation', 'reference_S100.json'),
                          ('qualification', 'qualification_timing.json'),
                          ('test_evaluation', 'finish_test_complete.json')):
        path = campaign / 'artifacts' / filename
        if path.exists():
            data = json.loads(path.read_text())
            result[key] = {k: data[k] for k in ('elapsed_seconds', 'wall_seconds', 'segments', 'failures') if k in data}
    return result


def _fmt(value):
    return 'unavailable' if value is None else f'{value:.6g}'

def generate_report(campaign, test_receipt, selection_path, *, gpu_seconds, attempt_count):
    from .round7_summary import summarize_reductions, compare_summaries
    from .round7_analysis import generate_analysis
    report_started = time.perf_counter()
    campaign = Path(campaign)
    selection = json.loads(Path(selection_path).read_text())
    test = json.loads(Path(test_receipt).read_text())
    if selection['status'] != 'frozen_before_test':
        raise ValueError('pre-TEST frozen selection required')
    fields = ['CH4', 'CO', 'T', 'U_1', 'p']
    entries = test['endpoints']
    source = summarize_reductions(entries['SOURCE']['reductions'], fields)
    chosen = selection['endpoint']
    label = next((name for name, value in entries.items()
                  if chosen and value['checkpoint_sha256'] == chosen['checkpoint_sha256']), None)
    selected = summarize_reductions(entries[label]['reductions'], fields) if label else None
    native = test['native_comparisons'][label] if label else None
    fixed_evidence = _fixed_panel_evidence(campaign, fields, summarize_reductions, compare_summaries)
    test_evidence = _test_panel_evidence(entries, test, source, fields, summarize_reductions, compare_summaries)
    timing = _timing_evidence(campaign, gpu_seconds)
    q_ratios, native_ratios = {}, {}
    if selected:
        changes = compare_summaries(source, selected)['changes']
        q_ratios = {f: changes['field.' + f + '.mse']['candidate'] / changes['field.' + f + '.mse']['reference'] for f in fields}
        native_ratios = dict(zip(native['field_names'], native['primary_mse_ratio']))
        binding = json.loads(Path(chosen['binding']).read_text())
        train_ok = all(value <= 1.10 for value in binding['train_primary_mse_ratios'].values())
        val_native_ok = all(value <= 1.10 for value in binding['native_comparison']['primary_mse_ratio'])
        disposition = selection['selected'] if (train_ok and val_native_ok and all(x <= 1.10 for x in q_ratios.values())
                                                    and all(x <= 1.10 for x in native_ratios.values())) else 'SOURCE'
    else:
        changes, disposition = {}, 'SOURCE'
    output = campaign / 'comparison/postprocess'
    output.mkdir(parents=True, exist_ok=True)
    arms = {}
    for arm in ('B', 'P', 'T'):
        receipt = campaign / 'artifacts' / f'owned_run_{arm}.json'
        if receipt.exists():
            arms[arm] = json.loads(receipt.read_text())['run']
    figure_dir = output / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)
    _compute_coverage_figure(campaign, figure_dir, gpu_seconds, attempt_count)
    analysis = None
    if arms:
        analysis = generate_analysis(arms, figure_dir)
    has_comparison_figures = (campaign / 'artifacts/reference_SOURCE.json').exists()
    if has_comparison_figures:
        _comparison_figures(campaign, figure_dir, selection, source, selected, q_ratios, native_ratios)
    lines = ['# Round 7: coherent updates and conditional-target alignment', '',
             f'Frozen validation selection: {selection["selected"]}. Operational disposition after the fixed TEST100 guards: {disposition}.', '',
             'SOURCE-start post-training retained complete State and Scale, the original fidelity bounds and C0. P used a fixed norm-matched direction preference; T replaced the Shape target with a fixed-transform conditional fair score. Arm admission and observer choice are recorded in the case artifacts.', '',
             f'The fixed TEST panel contains {source["conditions"]} conditions on Q4096 and 40,300 native spatial sites. TEST was evaluated after endpoint selection and does not rerank endpoints.', '']
    for boundary in (500, 1000):
        decision = campaign / 'comparison' / f'decision_{boundary}.json'
        inputs = campaign / 'comparison' / f'decision_input_{boundary}.json'
        if decision.is_file() and inputs.is_file():
            verdict = json.loads(decision.read_text())['decision']
            observations = json.loads(inputs.read_text())['arms']
            values = ', '.join(f"{item['name']} Scale {item['observations'][str(boundary)]['scale']:.6g}" for item in observations)
            lines.extend([f'Epoch {boundary} fixed validation: {values}. Decision: {json.dumps(verdict, sort_keys=True)}.', ''])
            if boundary == 500:
                lines.extend(['State and Shape changes are in fixed SOURCE units; maxima are fieldwise MSE ratios to SOURCE.', '',
                              '| Arm | State delta | Shape delta | Train max | Val Q max | Val native max |',
                              '| --- | ---: | ---: | ---: | ---: | ---: |'])
                for item in observations:
                    row = item['observations']['500']
                    lines.append(f"| {item['name']} | {row['state_delta']:+.5f} | {row['shape_delta']:+.5f} | {max(row['train_mse'].values()):.5f} | {max(row['q_mse'].values()):.5f} | {max(row['native_mse'].values()):.5f} |")
                lines.extend(['', '[Formal 500-epoch proposal](../../FORMAL_PROPOSAL.md).', ''])
            by_name = {item['name']: item for item in observations}
            if boundary == 1000 and selection['selected'] in ('P', 'T') and 'B' in by_name and selection['selected'] in by_name:
                b = by_name['B']['observations']['1000']['scale']
                candidate = by_name[selection['selected']]['observations']['1000']['scale']
                gain = b - candidate
                required = .005 * json.loads(inputs.read_text())['source_validation_scale']
                classification = 'incremental Scale gain' if gain >= required else 'Scale tied for the incremental claim'
                lines.extend([f'Epoch 1000 candidate versus B complete Scale difference B minus candidate: {gain:+.6g}; required increment: {required:.6g}. Classification: {classification}.', ''])
                if 'B' not in verdict.get('qualified', []):
                    lines.extend(['The candidate met the fixed endpoint requirements while B did not; this is a guard-feasibility advantage.', ''])
    incomplete = campaign / 'comparison/incomplete_baseline_decision.json'
    if incomplete.exists():
        event = json.loads(incomplete.read_text())
        lines.extend([f"The matched comparison is incomplete: B stopped at epoch {event['baseline_stopped_epoch']} under the prespecified fidelity rule, before the fixed 500-epoch endpoint.", ''])
    if selection.get('endpoint') and selection['endpoint']['epoch'] > 1000:
        lines.extend([f"The selected recipe continued alone to reporting epoch {selection['endpoint']['epoch']}; its treatment comparison remains the matched epoch-1000 result.", ''])
    s100_path = campaign / 'artifacts/reference_S100.json'
    if s100_path.exists():
        s100 = json.loads(s100_path.read_text())
        s100_val = summarize_reductions(s100['validation']['reductions'], fields)
        lines.extend([f"S100's fixed-validation complete Scale was {s100_val['scores']['scale']:.6g}; its earlier A200-plus-100 lineage is a secondary benchmark with different optimizer history.", ''])
        if selection.get('endpoint'):
            selected_binding = json.loads(Path(selection['endpoint']['binding']).read_text())
            selected_val = summarize_reductions(selected_binding['validation_reductions'], fields)
            retained = selected_val['scores']['scale'] <= s100_val['scores']['scale'] + .02 * .1585698355665164
            lines.extend([f"Selected validation Scale versus S100 plus 0.02 frozen Scale units: {'retained' if retained else 'not retained'}.", ''])
    if fixed_evidence:
        lines.extend(['Fixed TRAIN and VALIDATION observer scores and weighted Scale components (A/R and full Shape decompositions are recorded in the linked numerical summary).', '',
                      '| Endpoint | Panel | State | Scale | Fair Shape | Legacy Shape | Signed | Joint | Self | Rho |',
                      '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
        for name, data in fixed_evidence.items():
            groups = [('SOURCE', data)] if name == 'SOURCE' else [(f'{name} {epoch}', item['splits']) for epoch, item in data.items()]
            for endpoint, splits in groups:
                for split, item in splits.items():
                    panel = item if name == 'SOURCE' else item['summary']
                    scores, terms = panel['scores'], panel['weighted_scale_terms']
                    vals = [scores.get(k) for k in ('state', 'scale', 'shape.fair', 'shape.legacy')] + [terms.get(k) for k in ('signed', 'joint', 'self', 'rho')]
                    lines.append('| ' + endpoint + ' | ' + split + ' | ' + ' | '.join(_fmt(v) for v in vals) + ' |')
        lines.extend(['', 'Weighted Scale attraction / separation (unavailable where the observer did not record a component).', '',
                      '| Endpoint | Panel | Signed A/R | Joint A/R | Self A/R | Rho A/R |',
                      '| --- | --- | ---: | ---: | ---: | ---: |'])
        for name, data in fixed_evidence.items():
            groups = [('SOURCE', data)] if name == 'SOURCE' else [(f'{name} {epoch}', item['splits']) for epoch, item in data.items()]
            for endpoint, splits in groups:
                for split, item in splits.items():
                    panel = item if name == 'SOURCE' else item['summary']
                    diag = panel['weighted_scale_attraction_separation']
                    vals = [f"{_fmt(diag[k]['attraction'])} / {_fmt(diag[k]['separation'])}" for k in ('signed', 'joint', 'self', 'rho')]
                    lines.append('| ' + endpoint + ' | ' + split + ' | ' + ' | '.join(vals) + ' |')
        lines.extend(['', 'Shape self and mutual components by observer; attraction/separation details are in the numerical summary.', '',
                      '| Endpoint | Panel | Fair self | Fair mutual | Legacy self | Legacy mutual |',
                      '| --- | --- | ---: | ---: | ---: | ---: |'])
        for name, data in fixed_evidence.items():
            groups = [('SOURCE', data)] if name == 'SOURCE' else [(f'{name} {epoch}', item['splits']) for epoch, item in data.items()]
            for endpoint, splits in groups:
                for split, item in splits.items():
                    panel = item if name == 'SOURCE' else item['summary']
                    observers = panel['shape_observers']
                    vals = [observers.get(obs, {}).get('losses', {}).get('topology.' + part + '.persistence')
                            for obs in ('fair', 'legacy') for part in ('self', 'mutual')]
                    lines.append('| ' + endpoint + ' | ' + split + ' | ' + ' | '.join(_fmt(v) for v in vals) + ' |')
        lines.append('')
    lines.extend(['Fixed TEST endpoint scores; all endpoints were frozen before TEST and are reported without reranking.', '',
                  '| Endpoint | State | Scale | Fair Shape | Legacy Shape | Max Q MSE / SOURCE | Max native MSE / SOURCE |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for endpoint, data in test_evidence.items():
        scores = data['summary']['scores']
        if endpoint == 'SOURCE':
            qmax = nmax = 1.
        else:
            c = data['source_comparison']['changes']
            qmax = max(c['field.' + f + '.mse']['candidate'] / c['field.' + f + '.mse']['reference'] for f in fields)
            nmax = max(data['native_comparison']['primary_mse_ratio'])
        lines.append('| ' + endpoint + ' | ' + ' | '.join(_fmt(scores.get(k)) for k in ('state', 'scale', 'shape.fair', 'shape.legacy')) + f' | {qmax:.5f} | {nmax:.5f} |')
    lines.append('')
    if selected:
        lines += ['| TEST measure | SOURCE | Selected | Change |', '| --- | ---: | ---: | ---: |']
        for key in ('state', 'scale', 'shape.fair', 'shape.legacy'):
            if key in changes:
                x = changes[key]
                lines.append(f'| {key} | {x["reference"]:.6g} | {x["candidate"]:.6g} | {x["raw_change"]:+.6g} |')
        lines += ['', '| Field | Q4096 primary MSE / SOURCE | Native primary MSE / SOURCE | Selected Q4096 relative L2 |',
                  '| --- | ---: | ---: | ---: |']
        for field in fields:
            l2 = selected['fields'][field]['relative_l2']
            lines.append(f'| {field} | {q_ratios[field]:.5f} | {native_ratios[field]:.5f} | {l2:.5f} |' if l2 is not None
                         else f'| {field} | {q_ratios[field]:.5f} | {native_ratios[field]:.5f} | unavailable |')
        lines += ['', 'Each fidelity guard uses primary-member MSE divided by SOURCE MSE on the declared support. The native guard uses physical area weights; Q4096 uses unweighted model units. TEST failures withhold deployment and do not trigger another selection.', '']
        delta = changes['scale']
        lines.extend([f"Per-condition TEST Scale changes: median {delta['paired_median_change']:+.6g}, 5th percentile {delta['paired_p05_change']:+.6g}, 95th percentile {delta['paired_p95_change']:+.6g}; {delta['conditions_improved']} of {delta['conditions']} conditions improved.", ''])
        binding = json.loads(Path(chosen['binding']).read_text())
        train = summarize_reductions(binding['train_reductions'], fields)
        validation = summarize_reductions(binding['validation_reductions'], fields)
        lines.extend([f"Selected complete Scale was {train['scores']['scale']:.6g} on TRAIN800 and {selected['scores']['scale']:.6g} on TEST100; these are separate fixed condition panels.", ''])
        source_receipt = json.loads((campaign / 'artifacts/reference_SOURCE.json').read_text()) if (campaign / 'artifacts/reference_SOURCE.json').exists() else None
        if source_receipt:
            source_train = summarize_reductions(source_receipt['train']['reductions'], fields)
            source_val = summarize_reductions(source_receipt['validation']['reductions'], fields)
            lines += ['| Observer | Source train | Chosen train | Source val | Chosen val |',
                      '| --- | ---: | ---: | ---: | ---: |']
            for key in ('state', 'scale', 'shape.fair', 'shape.legacy'):
                if all(key in item['scores'] for item in (source_train, train, source_val, validation)):
                    lines.append(f"| {key} | {source_train['scores'][key]:.6g} | {train['scores'][key]:.6g} | {source_val['scores'][key]:.6g} | {validation['scores'][key]:.6g} |")
            lines.append('')
        diag = validation['component_diagnostics']
        signed = diag.get('multiscale_spectrum.signed_band_energy_score.conditional', {})
        joint = diag.get('multiscale_spectrum.joint_spectral_mean_energy_score.conditional', {})
        lines.extend([f"Selected VALIDATION signed-band weighted attraction and separation: {signed.get('weighted_attraction')} and {signed.get('weighted_separation')}; joint attraction and separation after the 0.25 coefficient: {None if joint.get('attraction') is None else .25 * joint['attraction']} and {None if joint.get('repulsion') is None else .25 * joint['repulsion']}.", ''])
        for name, observer in validation['shape_observers'].items():
            values = observer.get('diagnostics', {}).get('component_attraction_separation', 'unavailable')
            lines.extend([f"Selected VALIDATION {name} Shape self/mutual attraction and separation: {json.dumps(values, sort_keys=True)}.", ''])
        lines += [f'Selected TEST Q4096 J2 physical-measure generated-variance proxy by field: {selected["j2_physical_measure_model_units"]["variance_proxy"]}. The paired bias-plus-reference-variance proxy is {selected["j2_physical_measure_model_units"]["bias_plus_reference_variance_proxy"]}; it is not isolated conditional bias.', '']
    else:
        lines += ['The planned selection did not establish a candidate endpoint, so no candidate TEST ranking was performed.', '']
    if arms:
        for arm, details in analysis['arms'].items():
            lines.extend([f"{arm} actual-step negative-projection flags: {json.dumps(details['negative_at_least_ten_percent'], sort_keys=True)}; Scale-only shadow flag across two assessed windows: {details['scale_only_at_most_half_in_two_windows']}.", ''])
            lines.extend([f"{arm} diagnostic cost and counts: {json.dumps(details['telemetry_cost'], sort_keys=True)}. Diagnostic time overlaps shadow and live-step time and is not added to GPU time.", ''])
        lines += ['![Training family loss histories](figures/family_training_losses.png)',
                  'Sampled training losses are shown separately from fixed-panel scores. [PDF master](figures/family_training_losses.pdf)', '',
                  '![Actual parameter-step projection distributions](figures/actual_step_projections.png)',
                  'Actual parameter-step products are local projected reductions; they do not establish finite or held-out improvement. [PDF master](figures/actual_step_projections.pdf)', '']
    if has_comparison_figures:
        lines += ['![Fixed validation score and fidelity trajectories](figures/validation_trajectory.png)',
                  'Dashed lines show the declared preservation, Scale improvement, and field fidelity thresholds. [PDF master](figures/validation_trajectory.pdf)', '']
        if selected:
            lines += ['![Matched Scale, Shape, fidelity and spread comparison](figures/matched_components_fidelity.png)',
                      'SOURCE and selected fixed-panel scores are shown with weighted Scale terms, fieldwise fidelity and generated spread. [PDF master](figures/matched_components_fidelity.pdf)', '']
    lines += ['![Optimizer attempts and fixed-panel coverage](figures/compute_coverage.png)',
              'Attempt counts include qualification and replay; GPU time is the sum of physical-device interval unions. [PDF master](figures/compute_coverage.pdf)', '']
    lines += [f'Charged optimizer attempts: {attempt_count}; occupied physical GPU time: {gpu_seconds / 3600:.3f} hours. Elapsed time from the first to the last recorded GPU stage: {_fmt(timing["elapsed_first_to_last_gpu_stage_seconds"])} seconds. Stage wall times, calibration, evaluation, qualification, and CPU reporting time are recorded in the numerical summary; stage times may overlap.', '']
    cost_rows = []
    for path in sorted((campaign / 'artifacts').glob('*_complete.json')):
        record = json.loads(path.read_text())
        if record.get('run') and record.get('epoch') and record.get('elapsed_seconds') is not None:
            cost_rows.append((Path(record['run']).parent.name, record['epoch'], record['elapsed_seconds']))
    if cost_rows:
        lines.extend(['| Arm | Segment endpoint epoch | Segment wall seconds |', '| --- | ---: | ---: |'])
        lines.extend(f'| {arm} | {epoch} | {seconds:.1f} |' for arm, epoch, seconds in cost_rows)
        lines.append('')
    lines += [
              'The comparison uses one training seed and a previously exposed TEST panel. State, Scale and Shape describe declared statistical observers; they do not by themselves establish physical validity or conditional-law identification.', '']
    lines.extend(['[Numerical summary and per-condition paired differences](numeric_summary.json).', ''])
    report = output / 'REPORT.md'
    numeric = {'disposition': disposition, 'test_changes': changes,
        'test_q_mse_ratios': q_ratios, 'test_native_mse_ratios': native_ratios,
        'selected_test_summary': selected, 'source_test_summary': source,
        'fixed_train_validation_endpoints': fixed_evidence, 'fixed_test_endpoints': test_evidence, 'timing': timing}
    (output / 'numeric_summary.json').write_text(json.dumps(numeric, indent=2, sort_keys=True, allow_nan=False) + '\n')
    report.write_text('\n'.join(lines))
    for target in re.findall(r'\]\(([^)]+)\)', report.read_text()):
        if not (output / target).is_file():
            raise ValueError('report relative link does not resolve: ' + target)
    subprocess.run(['pandoc', str(report), '-o', str(output / 'REPORT.html'), '--standalone'], check=True, cwd=output)
    subprocess.run(['pandoc', str(report), '-o', str(output / 'REPORT.pdf'), '--pdf-engine=pdflatex'], check=True, cwd=output)
    subprocess.run(['pdftoppm', '-png', '-r', '100', str(output / 'REPORT.pdf'), str(output / 'rendered_page')], check=True)
    from PIL import Image
    for path in list(output.glob('rendered_page-*.png')) + list(figure_dir.glob('*.png')):
        with Image.open(path) as image:
            image.verify()
    names = list(('family_training_losses', 'actual_step_projections') if arms else ())
    if has_comparison_figures:
        names.append('validation_trajectory')
        if selected:
            names.append('matched_components_fidelity')
    names.append('compute_coverage')
    for name in names:
        for suffix in ('.png', '.pdf'):
            if not (figure_dir / (name + suffix)).is_file():
                raise ValueError('report figure link is missing')
    if not list(output.glob('rendered_page-*.png')):
        raise ValueError('report did not render')
    timing['cpu_report_seconds'] = time.perf_counter() - report_started
    (output / 'numeric_summary.json').write_text(json.dumps(numeric, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return {'cpu_report_seconds': timing['cpu_report_seconds'], 'report': str(report), 'report_pdf': str(output / 'REPORT.pdf'),
            'disposition': disposition, 'test_q_ratios': q_ratios, 'test_native_ratios': native_ratios}
