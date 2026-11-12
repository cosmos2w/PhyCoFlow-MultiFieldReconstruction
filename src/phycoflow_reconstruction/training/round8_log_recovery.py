"""Restore checkpoint-bound Round 8 calibration and query-exposure journals."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from ..coherence.round8_queries import pair_count, VERSION, SHAPE_SAMPLING_VERSION, LINE_WEIGHTS, PAIR_SIGNS


def _read(path):
    if not path.exists(): return b'', []
    raw = path.read_bytes(); rows=[]; offset=0
    def invalid(value): raise ValueError(f'nonfinite JSON value: {value}')
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b'\n'): raise ValueError(f'incomplete Round 8 journal: {path.name}')
        try: row = json.loads(line, parse_constant=invalid)
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError(f'malformed Round 8 journal: {path.name}') from error
        if not isinstance(row, dict): raise ValueError(f'invalid Round 8 journal row: {path.name}')
        rows.append((row,offset,offset+len(line))); offset += len(line)
    return raw, rows


def _calibration_boundary(rows, calibration):
    expected = list(calibration.records)
    frozen = calibration.coefficients is not None
    size = len(expected) + int(frozen)
    if len(rows) < size: raise ValueError('calibration journal ends before checkpoint')
    previous = 0; seen_frozen = False
    for index, (row, _, _) in enumerate(rows):
        epoch = row.get('epoch')
        if isinstance(epoch,bool) or not isinstance(epoch,int) or epoch not in (1,3,6,8,10):
            raise ValueError('invalid calibration journal epoch')
        if row.get('event') == 'frozen':
            if seen_frozen or epoch != 10 or previous != 10:
                raise ValueError('invalid calibration journal freeze')
            seen_frozen = True
        else:
            if 'event' in row or seen_frozen or epoch <= previous or not isinstance(row.get('norms'),dict):
                raise ValueError('invalid calibration journal record')
            if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or v < 0 for v in row['norms'].values()):
                raise ValueError('invalid calibration journal norms')
            previous = epoch
        if index < len(expected):
            if row.get('epoch') != expected[index]['epoch'] or row.get('norms') != expected[index]['norms'] or 'event' in row:
                raise ValueError('calibration journal differs from checkpoint')
        elif index == len(expected) and frozen:
            if (row.get('event') != 'frozen' or row.get('coefficients') != calibration.coefficients
                    or row.get('priorities') != calibration.priorities or row.get('units') != calibration.units):
                raise ValueError('calibration freeze differs from checkpoint')
    return rows[size-1][2] if size else 0


def _exposure_boundary(rows, sampler, mode):
    cursor = sampler.counter if mode == 'candidate' else 0
    last_batch=0; seen=set(); groups={}; cutoff=0
    for row, _, end in rows:
        batch,kind,index,count,prob = (row.get(k) for k in ('batch','kind','index','count','probability'))
        if (isinstance(batch,bool) or not isinstance(batch,int) or batch < 1 or
            kind not in ('self','pair') or isinstance(index,bool) or not isinstance(index,int) or
            isinstance(count,bool) or not isinstance(count,int) or count < 1 or
            isinstance(prob,bool) or not isinstance(prob,(int,float)) or not math.isfinite(prob) or not 0 < prob <= 1):
            raise ValueError('invalid query exposure journal row')
        sign, line, inner_probability = row.get('inner_sign'), row.get('inner_line'), row.get('inner_probability')
        if (row.get('query_protocol') != VERSION or row.get('shape_sampling') != SHAPE_SAMPLING_VERSION
                or row.get('inner_seed') != sampler.inner_seed
                or not isinstance(sign, list) or isinstance(inner_probability, bool)
                or not isinstance(inner_probability, (int, float)) or not math.isfinite(inner_probability)):
            raise ValueError('invalid query exposure inner sampling identity')
        if kind == 'self':
            if tuple(sign) not in ((1,), (-1,)) or line is not None or inner_probability != .5:
                raise ValueError('invalid query exposure self sign')
        elif (tuple(sign) not in PAIR_SIGNS or isinstance(line, bool) or not isinstance(line, int)
              or not 0 <= line < len(LINE_WEIGHTS)
              or not math.isclose(inner_probability, LINE_WEIGHTS[line] / 4, rel_tol=1e-12)):
            raise ValueError('invalid query exposure pair sign or line')
        population = len(sampler.fields) if kind == 'self' else pair_count(len(sampler.fields))
        if not 0 <= index < population or count > (1 if kind == 'self' else sampler.mutual_draws):
            raise ValueError('query exposure outside configured population')
        if batch < last_batch or batch > last_batch + 1:
            raise ValueError('query exposure batches must be contiguous')
        if batch != last_batch:
            seen=set(); last_batch=batch
        if (kind,index) in seen: raise ValueError('duplicate query exposure in batch')
        seen.add((kind,index))
        group=groups.setdefault(batch,{'self':0,'pair':0}); group[kind]+=count
        if batch <= cursor: cutoff=end
    if last_batch < cursor: raise ValueError('query exposure journal ends before checkpoint')
    for batch in range(1,cursor+1):
        expected_pairs = sampler.mutual_draws if len(sampler.fields)>1 else 0
        if groups[batch] != {'self':1,'pair':expected_pairs}:
            raise ValueError('checkpointed query exposure batch is incomplete')
    return cutoff


def _commit(path, raw, cutoff):
    tail=raw[cutoff:]
    if not tail: return None
    digest=hashlib.sha256(tail).hexdigest()
    archive=path.with_name(f'{path.stem}.uncheckpointed_{digest}.jsonl')
    if archive.exists():
        if archive.read_bytes()!=tail: raise ValueError('existing Round 8 journal archive differs')
    else:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.round8-journal-tail-',delete=False) as stream:
            temporary=Path(stream.name)
            try: stream.write(tail); stream.flush(); os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True); raise
        try:
            try: os.link(temporary,archive)
            except FileExistsError:
                if archive.read_bytes()!=tail: raise ValueError('existing Round 8 journal archive differs')
        finally: temporary.unlink(missing_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.round8-journal-',delete=False) as stream:
        temporary=Path(stream.name)
        try: stream.write(raw[:cutoff]); stream.flush(); os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True); raise
    try: os.replace(temporary,path)
    finally: temporary.unlink(missing_ok=True)
    return str(archive)


def restore_journals(run_dir, calibration, sampler, mode):
    """Validate both journals, then preserve/truncate tails after saved state."""
    if mode not in {'warmup','control','candidate'} or sampler.pending is not None:
        raise ValueError('Round 8 journal recovery requires a checkpoint boundary')
    logs=Path(run_dir)/'logs'
    cal_path=logs/'round8_calibration.jsonl'; exp_path=logs/'round8_query_exposure.jsonl'
    cal_raw, cal_rows=_read(cal_path); exp_raw, exp_rows=_read(exp_path)
    cal_cutoff=_calibration_boundary(cal_rows,calibration)
    exp_cutoff=_exposure_boundary(exp_rows,sampler,mode)
    return {'calibration_archive':_commit(cal_path,cal_raw,cal_cutoff),
            'exposure_archive':_commit(exp_path,exp_raw,exp_cutoff)}
