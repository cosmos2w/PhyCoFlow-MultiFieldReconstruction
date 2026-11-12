"""Refresh sampled Round 8 curves from persisted updates only.

No model, device or evaluation is accessed. Metadata makes routine calls cheap:
only the final JSONL line is read unless an epoch boundary is due. The files in
logs/ are inherited across a warmup fork and refreshed from each child's history.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

VERSION = 'round8_training_history_v1'
_FAMILIES = {'state': 'global_distribution', 'scale': 'multiscale_spectrum', 'shape': 'topology'}
_SCALE = {
    'signed': 'signed_band_energy_score.conditional',
    'joint': 'joint_spectral_mean_energy_score.conditional',
    'log_power_self': 'relation_self_energy_score.conditional',
    'rho': 'relation_rho_crps.conditional',
}
_SHAPE = {'self': 'self.persistence', 'mutual': 'mutual.persistence'}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _tail(path):
    with path.open('rb') as stream:
        stream.seek(0, os.SEEK_END)
        position = stream.tell()
        if not position:
            return None
        chunks = []
        while position:
            size = min(position, 8192)
            position -= size
            stream.seek(position)
            chunk = stream.read(size)
            chunks.append(chunk)
            if b'\n' in chunk.rstrip(b'\r\n'):
                break
        return json.loads(b''.join(reversed(chunks)).splitlines()[-1])


def _collect(rows, steps_per_epoch):
    """Prepare named, sparse time series; never interpolate missing values."""
    data = {'family': {}, 'scale': {}, 'shape': {}, 'state_channels': {}}
    last_step = 0
    def add(panel, label, step, value):
        value = _number(value)
        if value is not None:
            series = data[panel].setdefault(label, ([], []))
            series[0].append(step / steps_per_epoch); series[1].append(value)
    for row in rows:
        step = row.get('step')
        if isinstance(step, bool) or not isinstance(step, int) or step <= last_step:
            raise ValueError('full update history must have strictly increasing positive steps')
        last_step = step
        r8 = row.get('round8') or {}
        scores = r8.get('scores') or {}
        for short, name in _FAMILIES.items():
            # The candidate's raw score precedes its internal coefficients;
            # retained controls instead use the original family scalar.
            item = scores.get(short) or {}
            raw = item.get('raw', row.get(f'coherence_family/{name}/raw'))
            calibrated = row.get(f'coherence_family/{name}/weighted_contribution', item.get('calibrated'))
            add('family', f'{short} raw', step, raw)
            add('family', f'{short} calibrated', step, calibrated)
        blocks = r8.get('blocks_raw') or {}
        calibrated_blocks = r8.get('blocks_calibrated') or {}
        for panel, mapping in (('scale', _SCALE), ('shape', _SHAPE)):
            name = _FAMILIES[panel]
            outer = _number(row.get(f'coherence_family/{name}/outer_weight'))
            outer_scale = _number(row.get(f'coherence_family/{name}/calibration_scale'))
            factor = (outer if outer is not None else 1.) * (outer_scale if outer_scale is not None else 1.)
            for short, legacy_path in mapping.items():
                key = f'{panel}.{short}'
                if key in blocks:
                    add(panel, f'{short} raw weighted', step, factor * blocks[key])
                    if key in calibrated_blocks:
                        add(panel, f'{short} calibrated weighted', step, factor * calibrated_blocks[key])
                else:
                    legacy = row.get(f'coherence_component/{name}/{legacy_path}/weighted_contribution')
                    add(panel, f'{short} retained weighted', step, legacy)
        for channel in ('marginal', 'mixed'):
            key = f'state.{channel}'
            if key in blocks:
                add('state_channels', f'{channel} raw chain-rule surrogate', step, blocks[key])
            if key in calibrated_blocks:
                add('state_channels', f'{channel} calibrated chain-rule surrogate', step, calibrated_blocks[key])
    return data


def refresh(run_dir, steps_per_epoch, force=False):
    """Render first available history, each 10 completed epochs, or on force.

    Returns a compact status dictionary. Routine calls are O(1) in history size;
    a scheduled render reads the complete JSONL. Figures use linear axes so
    negative realized fair scores remain visible. ``force`` also covers pause,
    validation and completion. No curve is created before a row is available.
    """
    if isinstance(steps_per_epoch, bool) or not isinstance(steps_per_epoch, int) or steps_per_epoch < 1:
        raise ValueError('steps_per_epoch must be a positive integer')
    logs = Path(run_dir) / 'logs'
    source = logs / 'full_updates.jsonl'
    png, pdf, state_file = (logs / 'round8_history.png', logs / 'round8_history.pdf', logs / 'round8_history_state.json')
    if not source.exists() or source.stat().st_size == 0:
        return {'rendered': False, 'reason': 'no_history'}
    last = _tail(source)
    if not isinstance(last, dict) or isinstance(last.get('step'), bool) or not isinstance(last.get('step'), int) or last['step'] < 1:
        raise ValueError('invalid final update history row')
    step = last['step']
    previous = {}
    if state_file.exists():
        previous = json.loads(state_file.read_text())
        if previous.get('version') != VERSION or previous.get('steps_per_epoch') != steps_per_epoch:
            raise ValueError('Round 8 history state version or epoch size differs')
    complete_epoch = step // steps_per_epoch
    boundary = step % steps_per_epoch == 0 and complete_epoch % 10 == 0
    first = not previous or not png.is_file() or not pdf.is_file()
    due = boundary and complete_epoch > int(previous.get('completed_epoch', 0))
    if not (force or first or due):
        return {'rendered': False, 'reason': 'not_due', 'step': step}
    with source.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    series = _collect(rows, steps_per_epoch)
    if not any(series.values()):
        return {'rendered': False, 'reason': 'no_recorded_scores', 'step': step}
    import matplotlib
    matplotlib.use('Agg', force=True)
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, constrained_layout=True)
    panels = (
        ('family', 'Sampled TRAIN family scores: raw and calibrated', 'Family score'),
        ('scale', 'Sampled Scale: weighted components', 'Weighted score'),
        ('shape', 'Sampled Shape: weighted components', 'Weighted score'),
        ('state_channels', 'State gradient-channel chain-rule surrogates (not additive losses)', 'Gradient-channel surrogate'),
    )
    for ax, (key, title, ylabel) in zip(axes, panels):
        ax.set_title(title, loc='left', fontsize=11); ax.set_ylabel(ylabel)
        ax.axhline(0, color='0.75', linewidth=.6)
        if step / steps_per_epoch >= 10:
            ax.axvline(10, color='0.4', linestyle=':', linewidth=.8)
        for label, (x, y) in series[key].items():
            ax.plot(x, y, label=label, linewidth=1., linestyle='--' if 'raw' in label else '-')
        if series[key]: ax.legend(fontsize=10, ncol=2, loc='best')
        else: ax.text(.5, .5, 'No recorded values', ha='center', va='center', transform=ax.transAxes, color='0.45')
        ax.grid(alpha=.15)
    axes[-1].set_xlabel('Completed epoch equivalent (updates / steps per epoch)')
    if step / steps_per_epoch >= 10:
        axes[0].text(10, .98, 'Epoch 10: calibration freezes / common fork', fontsize=10,
                     transform=axes[0].get_xaxis_transform(), ha='left', va='top',
                     bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':.85})
    fig.suptitle('Round 8 sampled training history; curves are not a ConFIG scalar objective', fontsize=11)
    try:
        for target, fmt in ((png, 'png'), (pdf, 'pdf')):
            temp = target.with_suffix(target.suffix + '.tmp')
            fig.savefig(temp, format=fmt, dpi=130)
            os.replace(temp, target)
    finally:
        plt.close(fig)
    state = {'version': VERSION, 'steps_per_epoch': steps_per_epoch, 'last_step': step,
             'completed_epoch': complete_epoch, 'history_rows': len(rows)}
    temp = state_file.with_suffix('.json.tmp')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    os.replace(temp, state_file)
    return {'rendered': True, 'step': step, 'completed_epoch': complete_epoch,
            'png': str(png), 'pdf': str(pdf), 'history_rows': len(rows)}


def restore_history(run_dir, start_step):
    """Archive complete uncheckpointed rows and atomically restore active history.

    The archive contains the original bytes after the checkpoint and is named
    by their SHA-256. Invalid or incomplete histories are rejected before any
    write. Repeated restores of the same tail reuse its verified archive.
    """
    import hashlib
    import tempfile
    if isinstance(start_step, bool) or not isinstance(start_step, int) or start_step < 0:
        raise ValueError('checkpoint step must be a nonnegative integer')
    logs = Path(run_dir) / 'logs'
    source = logs / 'full_updates.jsonl'
    if not source.exists():
        if start_step:
            raise ValueError('Round 8 checkpoint has no update history')
        return {'restored_step': 0, 'archived_rows': 0, 'archive': None}
    content = source.read_bytes()
    lines = content.splitlines(keepends=True)
    last_step, cutoff = 0, 0
    def invalid_constant(value):
        raise ValueError(f'nonfinite JSON value: {value}')
    for line in lines:
        if not line.endswith(b'\n'):
            raise ValueError('Round 8 update history has an incomplete final row')
        try:
            row = json.loads(line, parse_constant=invalid_constant)
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError('Round 8 update history contains malformed JSON') from error
        if not isinstance(row, dict):
            raise ValueError('Round 8 update history rows must be objects')
        step = row.get('step')
        if isinstance(step, bool) or not isinstance(step, int) or step != last_step + 1:
            raise ValueError('Round 8 update history steps must be contiguous from one')
        last_step = step
        if step <= start_step:
            cutoff += len(line)
    if last_step < start_step:
        raise ValueError('Round 8 update history ends before the checkpoint')
    tail = content[cutoff:]
    if not tail:
        return {'restored_step': start_step, 'archived_rows': 0, 'archive': None}
    digest = hashlib.sha256(tail).hexdigest()
    archive = logs / f'full_updates.after_step_{start_step}_{digest}.jsonl'
    if archive.exists():
        if archive.read_bytes() != tail:
            raise ValueError('existing Round 8 tail archive differs')
    else:
        # Publish a fully flushed archive atomically without overwriting another.
        with tempfile.NamedTemporaryFile(dir=logs, prefix='.round8-tail-', delete=False) as stream:
            temp_archive = Path(stream.name)
            try:
                stream.write(tail); stream.flush(); os.fsync(stream.fileno())
            except BaseException:
                temp_archive.unlink(missing_ok=True)
                raise
        try:
            try:
                os.link(temp_archive, archive)
            except FileExistsError:
                if archive.read_bytes() != tail:
                    raise ValueError('existing Round 8 tail archive differs')
        finally:
            temp_archive.unlink(missing_ok=True)
    with tempfile.NamedTemporaryFile(dir=logs, prefix='.round8-history-', delete=False) as stream:
        temp_history = Path(stream.name)
        try:
            stream.write(content[:cutoff]); stream.flush(); os.fsync(stream.fileno())
        except BaseException:
            temp_history.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp_history, source)
    finally:
        temp_history.unlink(missing_ok=True)
    # Existing plots may include the abandoned tail; trigger regeneration from
    # the restored history before reporting or on the next recorded update.
    (logs / 'round8_history_state.json').unlink(missing_ok=True)
    return {'restored_step': start_step, 'archived_rows': last_step - start_step,
            'archive': str(archive), 'sha256': digest}
