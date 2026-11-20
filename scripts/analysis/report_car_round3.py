#!/usr/bin/env python3
"""Render a saved-evidence Round 3 report; never run model inference or scoring.

Inputs are a JSON manifest, car_round3_report_inputs_v1. Each evidence value is
null or {"path": "...", "sha256": "..."}; histories maps the two full arm
names to the same binding type. The decision, exact-compute and cost receipts use
car_round3_final_decision_v1, car_round3_exact_compute_disposition_v1, and
car_round3_costs_v1 respectively. A cost receipt has training keyed by arm with
seconds_per_epoch and occupied_gpu_seconds, assessment with gpu_seconds,
cpu_seconds and wall_seconds, plus failed_attempt_gpu_seconds; unknown values
are null, not zero. Decision has recommendation, reason, and arm_dispositions
keyed by both names. Exact-compute has disposition and reason, plus optional
baseline_epoch_seconds and reused_epoch_seconds. A missing evidence binding is
reported as pending and prevents a complete status.

Optional costs.training[arm].runtime_device is the recorded device label for
that arm's training; it is descriptive, not evidence of a same-device comparison.
Optional costs.campaign_gpu_accounting fields are additive GPU occupancy in
seconds: setup_upper_bound_gpu_seconds is a conservative bound for setup and
qualification, measured_training_gpu_seconds and measured_assessment_gpu_seconds
are measured occupation outside that setup bound, other_measured_gpu_seconds is
any remaining disjoint measured occupation, and accounted_upper_bound_gpu_seconds
is the supplied overall conservative bound across the campaign, including failed
attempts. Components must be disjoint; omit fields when unknown. The report
does not infer a total or remaining budget from incomplete components.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import re
import subprocess
import time

ARMS = ('FFM_RawRisk_Uniform_Repair', 'FFM_SourceScaledRisk_Uniform_Repair')
SHORT = {ARMS[0]: 'Raw-risk control', ARMS[1]: 'SOURCE-scaled feedback'}
FAMILIES = (('global_distribution', 'State'), ('multiscale_spectrum', 'Scale'), ('topology', 'Shape'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bound(ref):
    if ref is None:
        return None, None
    path = Path(ref['path']).resolve(strict=True)
    if digest(path) != ref['sha256']:
        raise ValueError(f'evidence hash mismatch: {path}')
    return path, path.read_text(encoding='utf-8')


def doc(ref, schema=None):
    path, content = bound(ref)
    if path is None:
        return None
    result = json.loads(content)
    if schema is not None and result.get('schema') != schema:
        raise ValueError(f'{path}: expected {schema}')
    return result


def fmt(value, suffix=''):
    if value is None:
        return 'unavailable'
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError('costs and measurements must be finite numbers or null')
    return f'{value:.5g}{suffix}'


def histories(inputs, epoch):
    result = {}
    for arm in ARMS:
        ref = inputs.get('histories', {}).get(arm)
        path, content = bound(ref)
        if path is None:
            continue
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        epochs = [row.get('epoch') for row in rows]
        if not rows or any(not isinstance(e, (int,float)) or int(e) != e for e in epochs):
            raise ValueError(f'{arm}: integral reporting epochs required')
        if epochs != list(range(1, len(rows)+1)):
            raise ValueError(f'{arm}: history must contain every completed epoch in order')
        if epoch is not None and len(rows) != epoch:
            raise ValueError(f'{arm}: history must end at the declared endpoint')
        if any('step' in row and row['step'] != 12 * int(row['epoch']) for row in rows):
            raise ValueError(f'{arm}: history disagrees with 12 updates per reporting epoch')
        result[arm] = rows
    return result


def save_figure(fig, directory, name):
    import matplotlib.pyplot as plt
    fig.savefig(directory / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(directory / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_histories(rows, figdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), constrained_layout=True)
    colors = {ARMS[0]:'#2165a4', ARMS[1]:'#b84d2d'}
    panels = [(f'coherence_family/{key}/weighted_contribution', f'{label} sampled training contribution') for key,label in FAMILIES]
    panels.append(('data_loss', 'Sampled signed mean-risk proxy'))
    for axis, (key, title) in zip(axes.flat, panels, strict=True):
        for arm in ARMS:
            history = rows.get(arm)
            if not history:
                continue
            series = [(r['epoch'], r.get(key)) for r in history]
            if any(not isinstance(y,(float,int)) or isinstance(y,bool) or not math.isfinite(y) for _,y in series):
                raise ValueError(f'{arm}: missing or nonfinite sampled history {key}')
            axis.plot([x for x,_ in series], [y for _,y in series], color=colors[arm], label=SHORT[arm], linewidth=1.1)
        axis.set_title(title); axis.set_xlabel('Reporting epoch'); axis.set_ylabel('Recorded sampled value')
        axis.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    save_figure(fig, figdir, 'sampled_training_histories')


def plot_scores(assessment, figdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axis = plt.subplots(figsize=(8,4.2), constrained_layout=True)
    x = np.arange(3); width = .34
    for j,arm in enumerate(ARMS):
        row = assessment['candidates'][arm]
        values = [row['normalized_core_differences'][k] for k in ('state','scale','shape')]
        axis.bar(x+(j-.5)*width, values, width, label=SHORT[arm], color=('#2165a4','#b84d2d')[j])
    axis.axhline(0, color='black', linewidth=.8)
    axis.axhline(.02, color='#777777', linestyle=':', linewidth=1, label='Maximum allowed regression')
    axis.axhline(-.05, color='#777777', linestyle='--', linewidth=1, label='Improvement margin')
    axis.set_xticks(x, ['State','Scale','Shape'])
    axis.set_ylabel('Difference from SOURCE / frozen TRAIN SOURCE unit')
    axis.set_title(f"Matched VALIDATION100, epoch {assessment['epoch']}")
    axis.legend(fontsize=8, ncol=2); axis.grid(axis='y',alpha=.2)
    save_figure(fig, figdir, 'matched_core_differences')


def paired_shape_values(shape):
    """Extract recorded paired scores without running a Shape scorer."""
    seeds = shape.get('seeds', []) if shape else []
    if not seeds or not all(isinstance(seed, dict) for seed in seeds):
        return []
    pairs = []
    for entry in seeds:
        seed = entry['seed']
        shared = entry['shared']['scores']['physical_weighted_total']
        specific = entry['condition_specific']['scores']['physical_weighted_total']
        if (isinstance(seed, bool) or not isinstance(seed, int)
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                       for x in (shared, specific))):
            raise ValueError('paired Shape query seeds and scores must be finite')
        pairs.append((seed, shared, specific))
    if len({seed for seed, _, _ in pairs}) != len(pairs):
        raise ValueError('paired Shape query seeds must be unique')
    return pairs


def plot_shape_queries(pairs, figdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    seed, shared, specific = zip(*pairs, strict=True)
    axis.plot(seed, shared, marker='o', linewidth=1.3, color='#2165a4', label='Shared query')
    axis.plot(seed, specific, marker='s', linewidth=1.3, color='#b84d2d', label='Condition-specific queries')
    axis.set_xticks(seed)
    axis.set_xlabel('Paired query seed')
    axis.set_ylabel('Physical-weighted Shape score')
    axis.set_title('Sampled query variation on fixed SOURCE endpoints')
    axis.grid(alpha=.2)
    axis.legend(fontsize=8)
    save_figure(fig, figdir, 'shape_query_variation')


def validate_assessment(value, label):
    if value is None:
        return
    if (value.get('schema') != 'car_round3_matched_validation_v1' or value.get('split') != 'validation'
            or value.get('epoch') not in (100,500) or value.get('members') != 4
            or len(value.get('sample_ids', [])) != 100 or len(set(value['sample_ids'])) != 100
            or set(value.get('candidates', {})) != set(ARMS)):
        raise ValueError(f'{label}: matched VALIDATION100 J4 assessment required')
    if label == 'review' and (value['epoch'] != 100 or value.get('task_scope') != 'both tasks excluded'
                              or any('reused_tasks' in row for row in value['candidates'].values())):
        raise ValueError('review must be epoch 100 with both tasks excluded')


def shape_evidence_complete(shape):
    if shape is None:
        return False
    dispersion = (bool(shape.get('seeds')) and 'shared_dispersion' in shape
                  and 'condition_specific_dispersion' in shape)
    terminal = (str(shape.get('disposition', '')).strip().lower() in {
        'inconclusive', 'budget-exhausted', 'budget_exhausted', 'completed',
        'closed', 'rejected', 'supported'} and isinstance(shape.get('reason'), str)
        and bool(shape['reason'].strip()))
    return dispersion or terminal


def pending_items(review, endpoint, decision, qualification, exact, shape, costs, rows):
    pending = []
    if review is None: pending.append('epoch-100 matched review')
    if endpoint is None: pending.append('matched endpoint assessment')
    elif any('reused_tasks' not in endpoint['candidates'][arm] for arm in ARMS):
        pending.append('both endpoint task scores')
    if decision is None: pending.append('final decision and arm dispositions')
    if qualification is None or qualification.get('status') not in ('passed', 'completed'):
        pending.append('resume qualification disposition')
    if exact is None or not exact.get('disposition') or str(exact['disposition']).lower() in ('pending', 'unknown') or not exact.get('reason'):
        pending.append('exact-compute disposition')
    if not shape_evidence_complete(shape):
        pending.append('cached Shape diagnostic result or disposition')
    if costs is None or any(arm not in costs.get('training', {}) for arm in ARMS) or not isinstance(costs.get('assessment'), dict):
        pending.append('training and assessment cost receipt')
    for arm in ARMS:
        if arm not in rows: pending.append(f'{SHORT[arm]} complete history')
    return pending


def produce(inputs_path, campaign, output):
    started, cpu_started = time.time(), time.process_time()
    inputs_path = inputs_path.resolve(strict=True); campaign = campaign.resolve(strict=True); output = output.resolve()
    if (campaign / 'reports') not in output.parents:
        raise ValueError('new report output must reside under campaign/reports')
    inputs = json.loads(inputs_path.read_text())
    if inputs.get('schema') != 'car_round3_report_inputs_v1':
        raise ValueError('car_round3_report_inputs_v1 manifest required')
    review = doc(inputs.get('review')); endpoint = doc(inputs.get('endpoint'))
    validate_assessment(review, 'review'); validate_assessment(endpoint, 'endpoint')
    if review is not None and endpoint is not None:
        for key in ('sample_ids','panel_sha256','source_bank_sha256','contract_sha256','core_sha256'):
            if review.get(key) != endpoint.get(key): raise ValueError(f'review/endpoint identity mismatch: {key}')
        if endpoint['epoch'] == 100 and (endpoint.get('task_free_review_sha256') != inputs['review']['sha256']
                                         or not endpoint.get('terminal_decision_sha256')):
            raise ValueError('epoch-100 endpoint tasks require the bound terminal decision and task-free review')
    decision = doc(inputs.get('decision'), 'car_round3_final_decision_v1')
    qualification = doc(inputs.get('qualification'))
    exact = doc(inputs.get('exact_compute'), 'car_round3_exact_compute_disposition_v1')
    shape = doc(inputs.get('shape'))
    costs = doc(inputs.get('costs'), 'car_round3_costs_v1')
    epoch = None if endpoint is None else endpoint['epoch']
    rows = histories(inputs, epoch)
    pending = pending_items(review, endpoint, decision, qualification, exact, shape, costs, rows)
    if decision is not None and (set(decision.get('arm_dispositions', {})) != set(ARMS)
                                 or not decision.get('recommendation') or not decision.get('reason')):
        raise ValueError('final decision must state both arm dispositions, recommendation and reason')
    if endpoint is not None and decision is not None:
        if decision.get('epoch') != epoch or decision.get('assessment_sha256') != inputs['endpoint']['sha256']:
            raise ValueError('final decision does not bind the declared endpoint')
        if not pending and decision['recommendation'] != endpoint.get('predeclared_recommendation'):
            raise ValueError('decision conflicts with the predeclared recommendation')
    output.mkdir(parents=True, exist_ok=False); figdir = output/'figures'; figdir.mkdir()
    captions = {}
    if rows:
        plot_histories(rows, figdir)
        captions['sampled_training_histories'] = 'Sampled TRAIN contributions and signed mean-risk proxy by reporting epoch; these traces are not native validation errors or a scalar ConFIG objective.'
    assessment = endpoint or review
    if assessment is not None:
        plot_scores(assessment, figdir)
        captions['matched_core_differences'] = 'State, Scale and Shape score changes on the matched VALIDATION100 panel, normalized by the frozen TRAIN SOURCE score units; negative values are improvements.'
    shape_pairs = paired_shape_values(shape)
    if shape_pairs:
        plot_shape_queries(shape_pairs, figdir)
        captions['shape_query_variation'] = 'Paired shared and condition-specific physical-weighted Shape scores by query seed on fixed SOURCE endpoints; these show sampled query variation, not learning or gradient variance.'
    status = 'INCOMPLETE — required evidence remains pending' if pending else 'COMPLETE — bounded endpoint decision'
    lines = ['# Coherence Adaptive Repair: Round 3', '', f'**Status:** {status}.', '']
    if pending:
        lines += ['Pending: ' + '; '.join(pending) + '.', '']
    if decision is not None:
        lines += [f"**Recommendation:** {decision['recommendation']}. {decision['reason']}", '']
        for arm in ARMS:
            lines += [f"**{SHORT[arm]}:** {decision['arm_dispositions'][arm]}", '']
    else:
        lines += ['The learned-output decision remains pending.', '']
    lines += ['## Evidence and scientific scope', '',
        'The compared paths use one FFM SOURCE, uniform allocation, a 50-epoch warmup and matched reporting epochs. This previously accessed same-trajectory VALIDATION100 panel is development evidence, not independent confirmation; TEST remains locked.', '',
        'RawRisk uses unit risk scales and a dual step of 0.01; SourceScaledRisk uses fieldwise max(TRAIN SOURCE member MSE, 1e-8) risk scales and a dual step of 0.0001. Both use identical fixed allowances and the same feasible set. This compares two feedback rules and does not causally isolate denominator scaling.', '']
    if assessment is not None:
        lines += [f"The latest matched assessment is epoch {assessment['epoch']} with J4 members and {len(assessment['sample_ids'])} conditions. SOURCE checkpoint `{assessment.get('source_checkpoint_sha256', 'see source-bank receipt')}`, panel SHA-256 `{assessment['panel_sha256']}` and SOURCE bank receipt SHA-256 `{assessment['source_bank_sha256']}` bind the comparison.", '',
            '| Field | SOURCE member MSE | Raw-risk member MSE | SOURCE-scaled member MSE | Raw signed-risk difference | Scaled signed-risk difference | Fixed TRAIN allowance |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        source = assessment['source']; raw, scaled = (assessment['candidates'][arm] for arm in ARMS)
        if raw['fidelity']['field_allowance'] != scaled['fidelity']['field_allowance']:
            raise ValueError('candidate native TRAIN allowances differ')
        for i, field in enumerate(assessment['field_order']):
            vals = (source['risk']['member_mse'][i], raw['risk']['member_mse'][i], scaled['risk']['member_mse'][i],
                    raw['fidelity']['field_difference'][i], scaled['fidelity']['field_difference'][i],
                    raw['fidelity']['field_allowance'][i])
            lines.append('| ' + str(field) + ' | ' + ' | '.join(fmt(v) for v in vals) + ' |')
        lines += ['', 'Field values are mass-weighted MSE in the respective native field units squared; signed-risk differences use the matched SOURCE baseline and the fixed TRAIN allowance converted fieldwise to native units. Member MSE, ensemble mean MSE, unbiased member variance and sensor results are saved in the linked assessment evidence.', '']
        for arm in ARMS:
            row = assessment['candidates'][arm]
            lines += [f"{SHORT[arm]}: normalized State/Scale/Shape changes {row['normalized_core_differences']}; signed-field pass {row['fidelity']['primary_field_pass']}, sensor pass {row['fidelity']['sensor_pass']}, separate legacy member-risk pass {row['fidelity']['legacy_member_pass']}.", '']
            lines += [f"{SHORT[arm]} native ensemble-mean MSE {row['risk'].get('ensemble_mean_mse', 'unavailable')}, unbiased member variance {row['risk'].get('unbiased_member_variance', 'unavailable')}, and sensor MSE {row.get('sensor_mse', 'unavailable')}; SOURCE sensor MSE {source.get('sensor_mse', 'unavailable')}. Field order follows the table above; null sensors indicate unobserved fields.", '']
            if 'reused_tasks' in row:
                task = row['reused_tasks']
                lines += [f"{SHORT[arm]} reused tasks: event normalized change {fmt(task['event_normalized_difference'])}; regional normalized improvement {fmt(task['regional_normalized_improvement'])}.", '']
        if assessment.get('paired_scaled_minus_raw'):
            pair = assessment['paired_scaled_minus_raw']
            lines += [f"Paired SOURCE-scaled minus raw-risk effects: regional normalized improvement {fmt(pair['regional_normalized_improvement'])}, event score {fmt(pair['event_score'])}; signed native risk differences {pair['signed_mean_risk']} and normalized State/Scale/Shape differences {pair['normalized_core_differences']}.", '']
        elif assessment['epoch'] == 100:
            lines += ['Both event and regional task scores were excluded from the learning review.', '']
        for arm in ARMS:
            lines += [f"{SHORT[arm]} epoch-{assessment['epoch']} checkpoint SHA-256: `{assessment.get('checkpoint_sha256', {}).get(arm, 'unavailable')}`.", '']
    if review is not None and endpoint is not None and endpoint['epoch'] != review['epoch']:
        lines += ['The epoch-100 review used the same matched panel and excluded both task scores.', '']
        for arm in ARMS:
            first = review['candidates'][arm]
            lines += [f"At the review, {SHORT[arm]} normalized State/Scale/Shape changes were {first['normalized_core_differences']}; signed-field pass {first['fidelity']['primary_field_pass']}, sensor pass {first['fidelity']['sensor_pass']}, and legacy member-risk pass {first['fidelity']['legacy_member_pass']}.", '']
    if captions:
        lines += ['## Figure index', '']
        for name, caption in captions.items():
            lines.append(f"- [Figure: {caption.split(';')[0]}](#figure-{name.replace('_','-')})")
        lines.append('')
        for name, caption in captions.items():
            lines += [f"## Figure {name.replace('_','-')}", '', f'![{caption}](figures/{name}.png)', '', f'{caption} [PDF master](figures/{name}.pdf)', '']
    lines += ['## Implementation and diagnostic decisions', '']
    if exact:
        lines += [f"Exact State/Scale reuse: **{exact.get('disposition', 'pending')}**. {exact.get('reason', '')} Historical and reused complete-epoch measurements: {fmt(exact.get('baseline_epoch_seconds'), ' s')} and {fmt(exact.get('reused_epoch_seconds'), ' s')}.", '']
    else:
        lines += ['Exact State/Scale reuse disposition is pending.', '']
    if qualification:
        lines += [f"Resume qualification status: {qualification.get('status', 'unavailable')}. Qualification establishes execution continuity, not learning efficacy.", '']
    else:
        lines += ['Resume qualification disposition is pending.', '']
    if shape and shape.get('shared_dispersion') and shape.get('condition_specific_dispersion'):
        shared = shape['shared_dispersion']['physical_weighted_total']; diverse = shape['condition_specific_dispersion']['physical_weighted_total']
        lines += [f"Cached TRAIN Shape query replay: {len(shape['seeds'])} paired seeds; shared-query mean {fmt(shared['mean'])}, sample SD {fmt(shared['sample_sd'])}; condition-specific mean {fmt(diverse['mean'])}, sample SD {fmt(diverse['sample_sd'])}; recovery-attempt scorer worker time {fmt(shape.get('scorer_worker_seconds'), ' s')}.", '',
            'This fixed-endpoint scalar dispersion does not establish parameter-gradient variance, a better ConFIG direction or model improvement.', '']
        if shape.get('conclusion') or shape.get('reason'):
            explanation = ' '.join(str(shape.get(key, '')).strip() for key in ('conclusion', 'reason') if shape.get(key))
            lines += [f"Shape diagnostic conclusion: {' '.join(explanation.split())}", '']
    elif shape_evidence_complete(shape):
        explanation = ' '.join(str(shape.get(key, '')).strip() for key in ('conclusion', 'reason') if shape.get(key))
        lines += [f"Cached Shape-query diagnostic closed as {shape['disposition']}: {' '.join(explanation.split())}", '',
            'No complete dispersion comparison is available; this diagnostic does not establish parameter-gradient variance, a better ConFIG direction or model improvement.', '']
    else:
        lines += ['The cached Shape-query diagnostic remains pending; it is independent of the learned-output decision.', '']
    if shape:
        details = []
        if shape.get('sample_count') is not None:
            details.append(f"{shape['sample_count']} cached TRAIN conditions")
        if shape.get('members') is not None:
            details.append(f"SOURCE member pair {shape['members']}")
        if shape.get('elapsed_worker_seconds') is not None:
            details.append(f"recovery-attempt elapsed worker time {fmt(shape['elapsed_worker_seconds'], ' s')}")
        if shape.get('worker_seconds_limit') is not None:
            details.append(f"recovery-attempt worker limit {fmt(shape['worker_seconds_limit'], ' s')}")
        if shape.get('prior_conservative_worker_seconds') is not None:
            details.append(f"prior-attempt conservative worker bound {fmt(shape['prior_conservative_worker_seconds'], ' s')}")
        if shape.get('accounted_upper_bound_worker_seconds') is not None:
            details.append(f"all-attempt accounted worker upper bound {fmt(shape['accounted_upper_bound_worker_seconds'], ' s')}")
        if shape.get('total_worker_seconds_ceiling') is not None:
            details.append(f"total worker ceiling {fmt(shape['total_worker_seconds_ceiling'], ' s')}")
        if details:
            lines += ['Shape replay panel and budget: ' + '; '.join(details) + '.', '']
        if shape.get('prior_conservative_worker_seconds') is not None:
            lines += ['Exact prior-attempt CPU costs are unavailable; the supplied prior-attempt figure is a conservative bound.', '']
    lines += ['## Cost and provenance', '']
    if costs:
        for arm in ARMS:
            item = costs.get('training', {}).get(arm, {})
            device = item.get('runtime_device')
            if device is not None and (not isinstance(device, str) or not device.strip() or '\n' in device or '\r' in device):
                raise ValueError('training runtime_device must be a nonempty single-line label')
            label = f' on {device}' if device is not None else ''
            lines += [f"{SHORT[arm]}{label}: {fmt(item.get('seconds_per_epoch'), ' s/epoch')} and {fmt(item.get('occupied_gpu_seconds'), ' occupied GPU s')}.", '']
        item = costs.get('assessment', {})
        lines += [f"Assessment: {fmt(item.get('gpu_seconds'), ' GPU s')}, {fmt(item.get('cpu_seconds'), ' CPU s')}, {fmt(item.get('wall_seconds'), ' wall s')}; failed attempts {fmt(costs.get('failed_attempt_gpu_seconds'), ' GPU s')}. Unknown values remain unavailable; overlapping wall and CPU/GPU times must not be added as independent costs.", '']
        accounting = costs.get('campaign_gpu_accounting')
        if accounting is not None:
            if not isinstance(accounting, dict):
                raise ValueError('campaign_gpu_accounting must be a mapping')
            labels = (
                ('setup_upper_bound_gpu_seconds', 'conservative setup and qualification upper bound'),
                ('measured_training_gpu_seconds', 'measured training occupancy'),
                ('measured_assessment_gpu_seconds', 'measured assessment occupancy'),
                ('other_measured_gpu_seconds', 'other disjoint measured occupancy'),
                ('accounted_upper_bound_gpu_seconds', 'supplied total campaign upper bound'),
            )
            for key, _ in labels:
                value = accounting.get(key)
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                    raise ValueError(f'{key} must be finite and nonnegative')
            supplied = [f'{label}: {fmt(accounting[key], " GPU s")}' for key, label in labels if accounting.get(key) is not None]
            if supplied:
                lines += ['Campaign GPU accounting against the 45 GPU-hour ceiling: ' + '; '.join(supplied) + '. Setup, training, assessment and other occupancy are disjoint components; the total summarizes them. The setup value is a conservative bound, while training and assessment occupancy are measured.', '']
    else:
        lines += ['Measured training and assessment costs are pending.', '']
    lines += ['Evidence identities below refer to the literal supplied artifacts; saved predictions and histories were read only. No inference, task scoring or new model evaluation was run while producing this report.', '']
    for key in ('review','endpoint','decision','qualification','exact_compute','shape','costs'):
        ref=inputs.get(key)
        if ref:
            lines.append(f"- {key}: `{ref['path']}`; SHA-256 `{ref['sha256']}`.")
    for arm,ref in inputs.get('histories', {}).items():
        if ref: lines.append(f"- {SHORT.get(arm,arm)} history: `{ref['path']}`; SHA-256 `{ref['sha256']}`.")
    lines.append('')
    markdown='\n'.join(lines); (output/'report.md').write_text(markdown,encoding='utf-8')
    render_and_validate(output, captions)
    receipt = dict(schema='car_round3_report_v1', status='incomplete' if pending else 'completed', pending=pending,
        report_sha256=digest(output/'report.md'), html_sha256=digest(output/'report.html'),
        inputs_sha256=digest(inputs_path), script_sha256=digest(Path(__file__)),
        figures={name:{'png_sha256':digest(figdir/f'{name}.png'),'pdf_sha256':digest(figdir/f'{name}.pdf')} for name in captions},
        start_unix=started, end_unix=time.time(), wall_seconds=time.time()-started, cpu_seconds=time.process_time()-cpu_started)
    (output/'report_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    return receipt


def render_and_validate(output, captions):
    import mistune
    from PIL import Image
    class Renderer(mistune.HTMLRenderer):
        def heading(self, text, level, **attrs):
            anchor = re.sub(r'[^a-z0-9 -]', '', re.sub(r'<[^>]*>', '', text).lower()).replace(' ', '-')
            return f'<h{level} id="{anchor}">{text}</h{level}>\n'
    markdown=(output/'report.md').read_text(encoding='utf-8')
    rendered=mistune.create_markdown(renderer=Renderer(),plugins=['table'])(markdown)
    (output/'report.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Round 3 report</title><style>img { max-width: 100%; height: auto; }</style></head><body>'+rendered+'</body></html>',encoding='utf-8')
    for name in captions:
        png, pdf=output/'figures'/f'{name}.png', output/'figures'/f'{name}.pdf'
        with Image.open(png) as image: image.verify()
        info=subprocess.run(['pdfinfo',str(pdf)],check=True,capture_output=True,text=True).stdout
        anchor=f'figure-{name.replace("_","-")}'
        if (not re.search(r'^Pages:\s+[1-9][0-9]*$',info,re.MULTILINE)
                or f'href="#{anchor}"' not in rendered or f'id="{anchor}"' not in rendered
                or html.escape(f'figures/{name}.png') not in rendered or html.escape(f'figures/{name}.pdf') not in rendered):
            raise ValueError(f'figure links or rendering invalid: {name}')
    if re.search(r'!\[[^\]]*\]\([^)]*\.pdf\)',markdown): raise ValueError('PDF masters must be linked, not embedded')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(produce(args.inputs,args.campaign,args.output),indent=2))


if __name__ == '__main__':
    main()
