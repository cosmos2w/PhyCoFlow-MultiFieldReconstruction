#!/usr/bin/env python3
"""Render a final Round 2 report from complete saved artifacts, without inference.

Input JSON schema car_round2_report_inputs_v1 names campaign-owned paths:
`runs` maps the four arm IDs to run directories; `trajectories` maps each arm
and split (train, validation) to six epoch-keyed assessment JSON paths;
`decisions`, `costs`, `bank_decisions`, and `source_banks` map ffm/senseiver
to paths; `allocation_disposition` names its JSON and `allocator_audits` maps
families to lists of audit receipt paths (empty when none were run). Each
source bank directory contains its native receipt and matched saved rows.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import subprocess
import time
from pathlib import Path

EPOCHS = (0, 100, 250, 500, 750, 1000)
ARMS = ('FFM_Selected_Bank_Uniform_FullModel', 'FFM_Selected_Bank_Residual_Repair',
        'Senseiver_Selected_Bank_Uniform_FullModel', 'Senseiver_Selected_Bank_Residual_Repair')
TRAINING_FAMILIES = (('global_distribution', 'State'), ('multiscale_spectrum', 'Scale'), ('topology', 'Shape'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inside(path, campaign):
    result = Path(path).resolve(strict=True)
    if campaign not in result.parents:
        raise ValueError(f'artifact must be inside campaign: {result}')
    return result


def read(path, campaign, schema=None):
    path = inside(path, campaign)
    value = json.loads(path.read_text())
    if schema is not None and value.get('schema') != schema:
        raise ValueError(f'unexpected schema: {path}')
    return value


def selected(doc, alpha=None):
    value = doc['selected_alpha'] if alpha is None else alpha
    matches = [row for row in doc['rows'] if math.isclose(float(row['alpha']), float(value), abs_tol=1e-12)]
    if len(matches) != 1:
        raise ValueError('assessment does not contain the requested unique amplitude')
    return matches[0]


def validate_cost_evidence(cost, decision, family_arms, campaign):
    """Keep measured science reportable while withholding missing cost claims."""
    complete = cost.get('complete') is True
    if decision.get('costs_complete') is not complete or (complete and cost.get('missing')):
        raise ValueError('decision must preserve the cost reconciliation completeness')
    if set(cost.get('arms', {})) != set(family_arms) or (complete and not cost.get('source_receipts')):
        raise ValueError('matching per-arm cost evidence required')
    for ref in cost.get('source_receipts', []):
        if sha(inside(ref['path'], campaign)) != ref['sha256']:
            raise ValueError('cost evidence changed')
    for arm in family_arms:
        item = cost['arms'][arm]
        serial = item.get('serial_cost_seconds')
        if complete:
            if (not isinstance(serial, (int, float)) or not math.isfinite(serial) or serial <= 0
                    or set(item.get('checkpoint_cost_seconds', {})) != {str(x) for x in EPOCHS}):
                raise ValueError(f'{arm}: complete serial and six checkpoint costs required')
        elif serial is not None:
            raise ValueError('incomplete costs must withhold the complete serial cost')
    if not complete:
        if any(decision.get(key) is not None for key in
               ('quality_route', 'learning_efficiency_route', 'deployment_comparison_route')):
            raise ValueError('incomplete costs require unavailable cost-dependent routes')
        disposition = decision.get('deployment_disposition', {})
        if (disposition.get('opt_in_function') is not None or disposition.get('operational_function') is not None
                or any(row.get(key) is not None for row in disposition.get('decisions', {}).values()
                       for key in ('source_deployment_route', 'operational_eligible'))):
            raise ValueError('incomplete costs cannot qualify a deployment adoption')


def route_label(value):
    return 'unavailable (incomplete costs)' if value is None else str(value)


def cost_hours(value):
    return 'unavailable' if value is None else f'{value / 3600:.4g} h'


def validate_inputs(inputs, campaign):
    if inputs.get('schema') != 'car_round2_report_inputs_v1' or set(inputs['runs']) != set(ARMS):
        raise ValueError('four named mature runs and report input schema required')
    freeze = read(campaign / 'configs/formal_freeze.json', campaign, 'car_round2_freeze_receipt_v1')
    if freeze.get('manifest', {}).get('status') != 'frozen':
        raise ValueError('formal frozen manifest required')
    docs, histories = {}, {}
    for arm in ARMS:
        run = inside(inputs['runs'][arm], campaign)
        status = read(run / 'status.json', campaign)
        if status.get('status') != 'completed' or status.get('global_step') != 12000:
            raise ValueError(f'{arm}: completed 1000-epoch mature run required')
        hist = [json.loads(line) for line in (run / 'logs/history.jsonl').read_text().splitlines() if line.strip()]
        if len(hist) != 1000 or hist[-1].get('step') != 12000:
            raise ValueError(f'{arm}: complete 1000-epoch loss history required')
        training_series(hist)
        for name in ('loss_history.png', 'loss_history.pdf', 'recording_definitions.json'):
            if not (run / 'logs' / name).is_file():
                raise ValueError(f'{arm}: missing saved loss recording {name}')
        histories[arm] = hist
        docs[arm] = {}
        for split in ('train', 'validation'):
            paths = inputs['trajectories'][arm][split]
            if set(paths) != {str(epoch) for epoch in EPOCHS}:
                raise ValueError(f'{arm}: all six {split} checkpoint readings required')
            docs[arm][split] = {}
            for epoch in EPOCHS:
                path = inside(paths[str(epoch)], campaign)
                doc = read(path, campaign, 'car_round2_train_nomination_v1' if split == 'train' else 'car_round2_validation_assessment_v1')
                checkpoint = run / 'checkpoints' / f'epoch_{epoch:03d}.pt'
                if (doc.get('arm') != arm or doc.get('split') != split or doc.get('epoch') != epoch
                        or doc.get('child_checkpoint_sha256') != sha(checkpoint)):
                    raise ValueError(f'{arm}: {split} epoch {epoch} mismatches mature checkpoint')
                selected(doc)
                selected(doc, 0.)
                selected(doc, 1.)
                docs[arm][split][epoch] = (path, doc)
        for epoch in EPOCHS:
            train = docs[arm]['train'][epoch][1]
            val = docs[arm]['validation'][epoch][1]
            if train['selected_alpha'] != val['selected_alpha'] or train['child_checkpoint_sha256'] != val['child_checkpoint_sha256']:
                raise ValueError(f'{arm}: validation does not use frozen TRAIN nomination')
        if 'reserved_task' not in docs[arm]['validation'][1000][1]:
            raise ValueError(f'{arm}: final reserved-task assessment required')
    decisions, costs, banks = {}, {}, {}
    allocation = read(inputs['allocation_disposition'], campaign, 'car_round2_allocation_disposition_v1')
    if allocation.get('mode') not in {'uniform', 'qualified_fixed'}:
        raise ValueError('final allocation disposition missing')
    frozen_allocation = freeze['manifest']['artifacts']['allocation_disposition']
    if sha(inside(inputs['allocation_disposition'], campaign)) != frozen_allocation['sha256']:
        raise ValueError('allocation differs from the formal freeze')
    for family, prefix in (('ffm', 'FFM'), ('senseiver', 'Senseiver')):
        decision = read(inputs['decisions'][family], campaign, 'car_round2_final_decision_v1')
        cost_path = inside(inputs['costs'][family], campaign)
        cost = read(cost_path, campaign, 'car_round2_cost_reconciliation_v1')
        if (decision.get('family') != family or decision.get('cost_reconciliation_sha256') != sha(cost_path)
                or cost.get('family') != family):
            raise ValueError(f'{family}: final decision and matching cost evidence required')
        family_arms = [arm for arm in ARMS if arm.startswith(prefix + '_')]
        validate_cost_evidence(cost, decision, family_arms, campaign)
        final_allocation = decision.get('allocation_decision', {})
        if (final_allocation.get('training_mode') != allocation['mode']
                or final_allocation.get('final_recheck') != cost.get('allocator_final')
                or (final_allocation.get('adopt') is True and
                    (cost.get('complete') is not True or cost.get('allocator_final', {}).get('qualified') is not True))):
            raise ValueError('final allocation decision must consume its actual cost recheck')
        bank = read(inputs['bank_decisions'][family], campaign, 'car_round2_bank_decision_v1')
        bank_binding = freeze['manifest']['artifacts'][f'bank_{family}']
        if sha(inside(bank_binding['path'], campaign)) != bank_binding['sha256']:
            raise ValueError('frozen source-specific selected bank changed')
        final_bank = read(bank_binding['path'], campaign)
        if bank['selected_count'] != len(final_bank['selected']):
            raise ValueError('bank decision selected count differs from the frozen bank')
        priority = read(campaign / 'assessment/train' / f'method_priority_{family}.json', campaign,
                        'car_round2_assessment_freeze_v1')
        payload = {key: value for key, value in priority.items() if key != 'sha256'}
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if (priority.get('sha256') != expected or priority.get('formal_freeze_sha256') != sha(campaign / 'configs/formal_freeze.json')
                or priority.get('method_priority') != decision.get('frozen_priority')):
            raise ValueError('TRAIN-frozen method priority or formal identity differs')
        for arm in family_arms:
            for epoch in EPOCHS:
                path = docs[arm]['train'][epoch][0]
                item = priority['arms'][arm][str(epoch)]
                if Path(item['nomination']).resolve() != path or sha(path) != item['sha256']:
                    raise ValueError('TRAIN nomination differs from frozen trajectory')
        for path in inputs['allocator_audits'][family]:
            audit = read(path, campaign, 'car_round2_allocator_job_v1')
            if audit.get('status') != 'completed' or audit.get('source_family') != family:
                raise ValueError('allocator audit is incomplete or belongs to another family')
        if allocation['mode'] == 'qualified_fixed' and not inputs['allocator_audits'][family]:
            raise ValueError('qualified allocation requires its actual audit evidence')
        full = family_arms[0]
        for split in ('train', 'validation'):
            reference = docs[full][split][0][1]
            for arm in family_arms:
                for epoch in EPOCHS:
                    other = docs[arm][split][epoch][1]
                    if (other['sample_ids'] != reference['sample_ids'] or other['panel_sha256'] != reference['panel_sha256']
                            or other['source_checkpoint_sha256'] != reference['source_checkpoint_sha256']
                            or other['core_sha256'] != reference['core_sha256']):
                        raise ValueError('paired assessment IDs, panel, SOURCE or observer differ')
        decisions[family], costs[family], banks[family] = decision, cost, bank
    return docs, histories, decisions, costs, banks, allocation


def adjacent_blocks(ids, values):
    """Means for up to ten consecutive frame IDs within each trajectory."""
    groups = {}
    for sample_id, value in zip(ids, values, strict=True):
        trajectory, frame = sample_id.rsplit(':', 1)
        groups.setdefault(trajectory, []).append((int(frame), float(value)))
    blocks = []
    for trajectory, frames in sorted(groups.items()):
        block = []
        for frame, value in sorted(frames):
            if block and (frame != block[-1][0] + 1 or len(block) == 10):
                blocks.append({'trajectory': trajectory, 'first': block[0][0], 'last': block[-1][0],
                               'count': len(block), 'mean': sum(x[1] for x in block) / len(block)})
                block = []
            block.append((frame, value))
        if block:
            blocks.append({'trajectory': trajectory, 'first': block[0][0], 'last': block[-1][0],
                           'count': len(block), 'mean': sum(x[1] for x in block) / len(block)})
    return blocks


def training_series(history):
    result = {}
    for family, _label in TRAINING_FAMILIES:
        key = f'coherence_family/{family}/raw'
        values = [row.get(key) for row in history]
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            raise ValueError(f'complete finite sampled-training family history required: {key}')
        result[family] = values
    return result


def plot_native(figdir, family, source_dir, assessment_path, assessment):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    receipt = json.loads((source_dir / 'receipt.json').read_text())
    if (receipt.get('schema') != 'car_round2_native_bank_v1' or receipt.get('status') != 'completed'
            or receipt.get('split') != 'validation' or receipt.get('kind') != 'source'
            or sha(source_dir / 'receipt.json') != assessment['source_bank_sha256']):
        raise ValueError('matched completed native SOURCE bank required for reconstruction')
    source_path = source_dir / '0000.pt'
    selected_path = assessment_path.parent / 'selected/0000.pt'
    if (sha(source_path) != receipt['files_sha256'][0]
            or sha(selected_path) != assessment['selected_files_sha256'][0]):
        raise ValueError('native reconstruction row identity differs')
    source = torch.load(source_path, map_location='cpu', weights_only=True)
    row = torch.load(selected_path, map_location='cpu', weights_only=True)
    if (source['sample_id'] != row['sample_id'] or row['sample_id'] != assessment['sample_ids'][0]
            or not torch.equal(source['truth_model'], row['truth_model']) or source['member_seeds'] != row['member_seeds']):
        raise ValueError('reconstruction sample identity differs')
    mass = source['native_masses'].double().reshape(-1).numpy()
    mask = mass > 0
    if not mask.any() or not np.isfinite(mass).all():
        raise ValueError('positive finite native physical mask required')
    coords = source['coordinates_raw'].reshape(-1, source['coordinates_raw'].shape[-1]).numpy()
    offset, scale = source['normalizer_offset'], source['normalizer_scale']
    truth = (source['truth_model'] * scale + offset).reshape(-1, len(receipt['field_order'])).numpy()
    prediction = (row['members_model'].mean(0) * scale + offset).reshape(-1, len(receipt['field_order'])).numpy()
    fields = receipt['field_order']
    fig, axes = plt.subplots(2, len(fields), figsize=(3.0 * len(fields), 6), squeeze=False, constrained_layout=True)
    for col, field in enumerate(fields):
        z, p = truth[mask, col], prediction[mask, col]
        weights = mass[mask]
        denominator = float(np.sum(weights * z * z))
        if denominator > 0:
            error = math.sqrt(float(np.sum(weights * (p-z)**2)) / denominator)
            annotation = f'physical relative L2 = {error:.4g}'
        else:
            error = math.sqrt(float(np.sum(weights * (p-z)**2) / weights.sum()))
            annotation = f'physical RMSE = {error:.4g}'
        low, high = min(z.min(), p.min()), max(z.max(), p.max())
        for ax, values, title in ((axes[0, col], z, f'{field}: reference'), (axes[1, col], p, f'{field}: nominated mean')):
            plot = ax.scatter(coords[mask, 0], coords[mask, 1], c=values, s=.5, vmin=low, vmax=high, cmap='viridis', rasterized=True)
            ax.set_title(title, fontsize=9); ax.set_aspect('equal'); fig.colorbar(plot, ax=ax, fraction=.04)
        axes[1, col].text(.02, .02, annotation, transform=axes[1, col].transAxes, fontsize=7,
                          bbox={'facecolor': 'white', 'alpha': .85, 'edgecolor': 'none'})
    fig.suptitle(f'{family.upper()} VALIDATION {row["sample_id"]}: physical positive-mass support')
    save_figure(fig, figdir, f'native_{family}')


def save_figure(fig, directory, name):
    import matplotlib.pyplot as plt
    fig.savefig(directory / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(directory / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def deployment_memory_table(docs):
    """Render measured final selected/raw allocated memory without inventing missing values."""
    lines = ['## Deployment allocated memory', '',
             'Per-amplitude maxima across measured conditions in the profiling harness, with SOURCE and child models resident. These are not isolated standalone serving memory.', '',
             '| Arm | Output | Amplitude | Baseline (MiB) | Peak (MiB) |',
             '| --- | --- | ---: | ---: | ---: |']
    for arm in ARMS:
        doc = docs[arm]['validation'][1000][1]
        for label, row in (('Nominated', selected(doc)), ('Raw', selected(doc, 1.))):
            memory = row.get('deployment_memory')
            baseline = f"{memory['baseline_allocated_bytes']/2**20:.2f}" if memory else 'unavailable'
            peak = f"{memory['peak_allocated_bytes']/2**20:.2f}" if memory else 'unavailable'
            lines.append(f"| {arm} | {label} | {row['alpha']:g} | {baseline} | {peak} |")
    lines.append('')
    return lines


def produce(inputs, campaign, output):
    start, cpu = time.time(), time.process_time()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    docs, histories, decisions, costs, banks, allocation = validate_inputs(inputs, campaign)
    if output.exists() or campaign not in output.parents:
        raise ValueError('report output must be a new campaign-owned directory')
    output.mkdir(parents=True)
    inputs_path = output / 'report_inputs.json'
    inputs_path.write_text(json.dumps(inputs, indent=2, allow_nan=False) + '\n')
    figures = output / 'figures'; figures.mkdir()
    fig, axes = plt.subplots(4, 3, figsize=(13, 11), constrained_layout=True)
    for axes_row, arm in zip(axes, ARMS, strict=True):
        rows = histories[arm]
        series = training_series(rows)
        for ax, (family, label) in zip(axes_row, TRAINING_FAMILIES, strict=True):
            ax.plot([row['step']/12 for row in rows], series[family], lw=.8)
            ax.set_title(f'{arm.replace("_", " ")} — {label}', fontsize=8)
            ax.set_xlabel('Reporting epoch'); ax.set_ylabel('Sampled raw family loss')
    save_figure(fig, figures, 'training_losses')
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, arm in zip(axes.flat, ARMS, strict=True):
        for split, style in (('train', '--'), ('validation', '-')):
            ax.plot(EPOCHS, [docs[arm][split][e][1]['selected_alpha'] for e in EPOCHS], style, label=split)
        ax.set_title(arm.replace('_', ' '), fontsize=9); ax.set_xlabel('Epoch'); ax.set_ylabel('TRAIN-nominated amplitude'); ax.legend()
    save_figure(fig, figures, 'nominations')
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for axes_row, prefix in zip(axes, ('FFM', 'Senseiver'), strict=True):
        for ax, key in zip(axes_row, ('state', 'scale', 'shape', 'event'), strict=True):
            for arm in (arm for arm in ARMS if arm.startswith(prefix + '_')):
                values = []
                for epoch in EPOCHS:
                    doc = docs[arm]['validation'][epoch][1]
                    values.append((selected(doc)['scores'][key] - selected(doc, 0.)['scores'][key]) / doc['normalizers'][key])
                ax.plot(EPOCHS, values, marker='.', label=arm.replace('_Selected_Bank_', ' '))
            ax.axhline(0, color='black', lw=.5); ax.set_title(f'{prefix}: {key}')
            ax.set_xlabel('Epoch'); ax.set_ylabel('Nominated minus SOURCE / TRAIN unit')
        axes_row[0].legend(fontsize=6)
    save_figure(fig, figures, 'core_trajectory')
    field_orders = {family: read(inside(inputs['source_banks'][family], campaign) / 'receipt.json', campaign)['field_order']
                    for family in ('ffm', 'senseiver')}
    if field_orders['ffm'] != field_orders['senseiver']:
        raise ValueError('source family field orders differ')
    fields = field_orders['ffm']
    fig, axes = plt.subplots(2, len(fields), figsize=(3 * len(fields), 7), squeeze=False, constrained_layout=True)
    for axes_row, family, prefix in zip(axes, ('ffm', 'senseiver'), ('FFM', 'Senseiver'), strict=True):
        arms = [arm for arm in ARMS if arm.startswith(prefix + '_')]
        labels = ['full model', 'residual']
        rows = [selected(docs[a]['validation'][1000][1]) for a in arms]
        for index, (ax, field) in enumerate(zip(axes_row, fields, strict=True)):
            for metric, label in (('member_mse', 'legacy member MSE'), ('ensemble_mean_mse', 'ensemble mean MSE'),
                                  ('signed_mean_risk', 'signed mean risk'), ('unbiased_member_variance', 'unbiased member spread')):
                if metric in rows[0]['risk'] and metric in rows[1]['risk']:
                    ax.plot(labels, [float(row['risk'][metric][index]) for row in rows], marker='o', label=label)
            ax.set_title(f'{family.upper()} {field}'); ax.set_ylabel('Physical field units squared'); ax.tick_params(axis='x', labelrotation=20)
            if index == 0:
                ax.legend(fontsize=6)
    save_figure(fig, figures, 'fidelity')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    paired_notes, block_records = [], {}
    for ax, family, prefix in zip(axes, ('ffm', 'senseiver'), ('FFM', 'Senseiver'), strict=True):
        full, residual = [a for a in ARMS if a.startswith(prefix + '_')]
        a, b = (docs[arm]['validation'][1000][1] for arm in (full, residual))
        differences = [y['state']-x['state'] for x,y in zip(selected(a)['per_condition_scores'], selected(b)['per_condition_scores'], strict=True)]
        blocks = adjacent_blocks(a['sample_ids'], differences)
        block_records[family] = {'sample_ids': a['sample_ids'], 'residual_minus_full_state': differences, 'blocks': blocks}
        ax.axhline(0, color='black', lw=.5); ax.plot(range(len(blocks)), [row['mean'] for row in blocks], '.', ms=3)
        ax.set_title(f'{family.upper()}: residual minus full model'); ax.set_xlabel('Consecutive-frame block'); ax.set_ylabel('State score difference')
        paired_notes.append(f'{family.upper()}: {len(differences)} matched conditions, mean difference {np.mean(differences):.6g}, {len(blocks)} consecutive-frame blocks (up to 10 frames).')
    save_figure(fig, figures, 'paired_blocks')
    (output / 'paired_condition_blocks.json').write_text(json.dumps(block_records, indent=2, allow_nan=False) + '\n')
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axes_row, family, prefix in zip(axes, ('ffm', 'senseiver'), ('FFM', 'Senseiver'), strict=True):
        for arm in (arm for arm in ARMS if arm.startswith(prefix + '_')):
            doc = docs[arm]['validation'][1000][1]; row = selected(doc)
            label = 'Residual' if arm.endswith('Residual_Repair') else 'Full model'
            axes_row[0].bar(label, doc['reserved_task']['normalized_improvement'])
            axes_row[1].bar(label, row['latency_seconds']/doc['source_latency_seconds'])
            serial = costs[family]['arms'][arm]['serial_cost_seconds']
            if serial is None:
                axes_row[2].text(.5, .5 if label == 'Full model' else .35, f'{label}: unavailable',
                                 ha='center', transform=axes_row[2].transAxes)
            else:
                axes_row[2].bar(label, serial/3600)
        for ax, title in zip(axes_row, ('Reserved task: SOURCE-normalized gain', 'Nominated / SOURCE latency', 'Complete serial research cost (hours)'), strict=True):
            ax.set_title(f'{prefix}: {title}', fontsize=9)
    save_figure(fig, figures, 'task_deployment_cost')
    for family, prefix in (('ffm', 'FFM'), ('senseiver', 'Senseiver')):
        arm = f'{prefix}_Selected_Bank_Residual_Repair'
        path, doc = docs[arm]['validation'][1000]
        plot_native(figures, family, inside(inputs['source_banks'][family], campaign), path, doc)
    captions = {
        'training_losses': 'Recorded sampled-training raw State, Scale and Shape family losses from the four complete run-owned histories, shown separately; ConFIG has no invented scalar objective.',
        'nominations': 'Amplitudes selected on TRAIN and carried unchanged to each matched VALIDATION checkpoint.',
        'core_trajectory': 'Fixed common-core and reused-event readings on VALIDATION; lower differences indicate lower score than SOURCE.',
        'fidelity': 'Native field-risk summaries; FFM reports ensemble mean, legacy member and unbiased member spread separately, while Senseiver is a point predictor.',
        'paired_blocks': 'Paired residual-minus-full-model State scores, grouped into up to ten consecutive frames within each trajectory; descriptive only, not independent trajectories.',
        'task_deployment_cost': 'Final reserved-task gain, measured nominated-function latency relative to SOURCE, and complete serial research cost where available; missing cost evidence is labelled unavailable.',
        'native_ffm': 'FFM nominated residual ensemble mean and physical reference on the positive-mass native mask; each field shows its own physical weighted error.',
        'native_senseiver': 'Senseiver nominated residual point prediction and physical reference on the positive-mass native mask; each field shows its own physical weighted error.',
    }
    lines = ['# Coherence Adaptive Repair: Round 2 final report', '',
             'The four mature comparisons completed 1,000 reporting epochs each. VALIDATION uses matched approximately 10% condition panels; TEST remains locked.', '',
             '## Decisions', '']
    for family in ('ffm', 'senseiver'):
        decision, bank, cost = decisions[family], banks[family], costs[family]
        disposition = decision['deployment_disposition']
        lines.append(f"{family.upper()}: operational choice is **{disposition['operational_recommendation']}**; opt-in function is {disposition['opt_in_function']}. Quality route: {route_label(decision['quality_route'])}; learning-efficiency route: {route_label(decision['learning_efficiency_route'])}; comparative deployment route: {route_label(decision['deployment_comparison_route'])}.")
        final_allocation = decision['allocation_decision']
        lines.append(f"{family.upper()} bank selected {bank['selected_count']} of {bank['candidate_count']} candidates; historical training allocation mode is {allocation['mode']}; final allocation decision is {final_allocation['status']} (adopt: {final_allocation['adopt']}). Actual final accounting is {cost['allocator_final']['status']} (qualified: {cost['allocator_final'].get('qualified')}).")
        lines.append(f"{family.upper()} selected-bank decision: retain the documented source-specific target for task-conditioned use. This comparison does not qualify an isolated learned-selection advantage or complete-law sufficiency.")
        for arm, verdict in disposition.get('decisions', {}).items():
            lines.append(f"{arm} learned-output eligibility: {verdict['task_useful']}; SOURCE-reference deployment route: {route_label(verdict['source_deployment_route'])}; default operational eligibility: {route_label(verdict['operational_eligible'])}.")
        if cost.get('complete') is not True:
            lines.append(f"{family.upper()} cost evidence is incomplete: {'; '.join(cost.get('missing', [])) or 'required cost evidence missing'}. Measured scientific and latency results remain reportable; cost-dependent adoption is unavailable.")
        for path in inputs['allocator_audits'][family]:
            audit = read(path, campaign, 'car_round2_allocator_job_v1')
            lines.append(f"{family.upper()} allocator audit: gradient MSE-times-cost ratio {audit.get('gradient_cost_uniform_ratio')}, provisional overhead fraction {audit.get('overhead_fraction')}, one-path qualification {audit.get('qualified_one_path')}; final status follows the complete cost reconciliation.")
        lines.append('')
    allocation_states = [decisions[family]['allocation_decision']['adopt'] for family in ('ffm', 'senseiver')]
    allocation_verdict = ('adopt' if all(value is True for value in allocation_states) else
                          'reject' if any(value is False for value in allocation_states) else 'unavailable')
    lines.extend([f'Two-family nonuniform allocation decision: **{allocation_verdict}**. Family-specific rechecks above retain their own evidence and do not change the recorded training recipe.', ''])
    lines.extend(['The selected amplitude and operational method order were frozen on TRAIN. Raw amplitude 1 remains a separate diagnostic; neither VALIDATION nor the reserved task chooses a new amplitude or checkpoint.',
                  'Both learner comparisons use the same source-specific selected bank, observer and fidelity target. These integrated comparisons do not isolate the effects of bank selection, learner parameterization and sampling policy.', '',
                  '## Fidelity and numerical contract', '',
                  'FFM learning constrains the signed ensemble-mean risk increase with a fieldwise absolute allowance equal to 10% of TRAIN SOURCE member MSE; it does not constrain a mean-risk ratio. The legacy member-MSE gate remains a separate default-deployment requirement. Senseiver is evaluated as a genuine J1 point predictor and retains its native sensor predictions; its residual correction is zero at observed field/site IDs.',
                  'FFM qualification concerns the stated mean-risk contract and measured finite conditional-member observables on this cohort; passing those gates does not establish calibrated complete conditional laws. Native fidelity is reported separately for raw amplitude 1 and the TRAIN-nominated function, including a possible SOURCE fallback.',
                  'Before formal admission, the Shape implementation was corrected so tied projection gradients do not depend on unrelated diagrams in a padded batch. Stable sorting and pointwise two-component projection arithmetic preserve the mathematical distance while choosing a consistent tie subgradient; floating values can differ by roundoff. Earlier allocation failures and pre-correction execution qualifications are preserved as historical attempts. Affected coefficient fitting, optimizer probes, bank selection, final-target execution qualifications and complete allocation audits were revisited on the corrected path. Exact saved-panel comparisons verified unchanged SOURCE outputs. These correctness checks do not establish a learned repair benefit.', '',
                  '## Figure index', ''])
    for name, caption in captions.items():
        lines.append(f'- [{caption}](#figure-{name.replace("_", "-")})')
    lines.extend(['', '## Paired VALIDATION summaries', '', *paired_notes,
                  'Blocks contain only consecutive observed frames within one trajectory and have at most ten frames. These are descriptive summaries; members and adjacent frames are not independent replications.', ''])
    for name, caption in captions.items():
        lines.extend([f'## Figure {name.replace("_", "-")}', '', f'![{caption}](figures/{name}.png)', '',
                      f'{caption} [PDF master](figures/{name}.pdf)', ''])
    lines.extend(deployment_memory_table(docs))
    lines.extend(['## Final raw and nominated results', ''])
    for arm in ARMS:
        doc = docs[arm]['validation'][1000][1]
        raw, nominee = selected(doc, 1.), selected(doc)
        lines.append(f"{arm}: TRAIN-nominated amplitude {doc['selected_alpha']}; raw State/Scale/Shape/event {raw['scores']}; nominated State/Scale/Shape/event {nominee['scores']}; raw primary field/sensor gate {raw['fidelity']['primary_pass']}, nominated primary field/sensor gate {nominee['fidelity']['primary_pass']}; raw legacy member gate {raw['fidelity'].get('legacy_member_pass', 'not applicable')}, nominated legacy member gate {nominee['fidelity'].get('legacy_member_pass', 'not applicable')}; reserved-task normalized gain {doc['reserved_task']['normalized_improvement']:.6g}.")
        family = 'ffm' if arm.startswith('FFM') else 'senseiver'
        item = costs[family]['arms'][arm]
        lines.append(f"{arm} new research cost: complete serial {cost_hours(item.get('serial_cost_seconds'))}; recorded occupied GPU {cost_hours(item.get('gpu_seconds'))}, independent CPU {cost_hours(item.get('independent_cpu_seconds'))}, failed attempts {cost_hours(item.get('failed_attempt_seconds'))}, planned pause segments {cost_hours(item.get('paused_attempt_seconds'))}.")
    lines.extend(['', 'Where complete, serial costs include occupied GPU wall and independent CPU-only wall outside the enclosing GPU lease; group wall is not added again. Recorded subtotals do not replace missing complete costs. Historical sunk costs remain outside the matched new-cost comparison. This report rendering is separately timed in report_receipt.json.',
                  'The reserved task is evaluated at epoch 1000 only. The reused event is a development/continuity endpoint, not an untouched task. No independent-trajectory confirmation or causal attribution to an individual package component is claimed.', ''])
    markdown = '\n'.join(lines)
    (output / 'report.md').write_text(markdown)
    import mistune
    class Renderer(mistune.HTMLRenderer):
        def heading(self, text, level, **attrs):
            anchor = re.sub(r'[^a-z0-9 -]', '', re.sub(r'<[^>]*>', '', text).lower()).replace(' ', '-')
            return f'<h{level} id="{anchor}">{text}</h{level}>\n'
    rendered = mistune.create_markdown(renderer=Renderer())(markdown)
    (output / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Round 2 report</title>' + rendered)
    validate_rendered(output, captions)
    end = time.time()
    (output / 'report_receipt.json').write_text(json.dumps({'schema': 'car_round2_report_job_v1', 'status': 'completed',
        'start_unix': start, 'end_unix': end, 'wall_seconds': end - start, 'cpu_seconds': time.process_time() - cpu,
        'scope': 'independent CPU-only report rendering; not included in the preceding cost reconciliation',
        'report_sha256': sha(output / 'report.md'), 'html_sha256': sha(output / 'report.html'),
        'report_inputs_sha256': sha(inputs_path), 'script_sha256': sha(Path(__file__).resolve())}, indent=2) + '\n')


def validate_rendered(output, captions):
    from PIL import Image

    text = (output / 'report.html').read_text()
    for name in captions:
        png, pdf = output / 'figures' / f'{name}.png', output / 'figures' / f'{name}.pdf'
        with Image.open(png) as image:
            image.verify()
        info = subprocess.run(['pdfinfo', str(pdf)], check=True, capture_output=True, text=True).stdout
        if not re.search(r'^Pages:\s+[1-9][0-9]*$', info, re.MULTILINE) or html.escape(f'figures/{name}.png') not in text or html.escape(f'figures/{name}.pdf') not in text:
            raise ValueError(f'rendered image or PDF link missing: {name}')
        anchor = f'figure-{name.replace("_", "-")}'
        if f'href="#{anchor}"' not in text or f'id="{anchor}"' not in text:
            raise ValueError(f'rendered figure index link is broken: {name}')
    if re.search(r'!\[[^\]]*\]\([^)]*\.pdf\)', (output / 'report.md').read_text()):
        raise ValueError('PDF must be linked as a master, not embedded')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    inputs = read(args.inputs, campaign, 'car_round2_report_inputs_v1')
    produce(inputs, campaign, args.output_dir.resolve())


if __name__ == '__main__':
    main()
