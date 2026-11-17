"""Evidence-bound serial cost reconciliation for the Round 2 comparison.

The caller supplies explicit attribution; no missing clock is inferred as zero.
GPU wall and independent CPU-only wall are additive, including failed attempts.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

from .adaptive_campaign import digest

EPOCHS = (0, 100, 250, 500, 750, 1000)
ARMS = tuple(f'{family}_Selected_Bank_{method}' for family in ('FFM','Senseiver')
             for method in ('Uniform_FullModel','Residual_Repair'))
REQUIRED_SCOPES = frozenset(('source_preparation','bank_selection','qualification','training','assessment','deployment'))
POLICY_SCOPES = frozenset(('allocation','policy_qualification','policy_refresh'))
RESIDUAL_ONLY = POLICY_SCOPES | {'residual_probe'}
CAP_SECONDS = 130 * 3600


def validate_excluded_enclosure(doc):
    """Recognize only actual sequence wrappers whose wall includes child work."""
    allowed = {
        'car_round2_qualification_group_v1': 'group wall encloses child work and is not an additional GPU cost',
        'car_round2_qualification_child_v1': 'encloses the child GPU lease; do not add both',
        'car_round2_formal_group_v1': 'encloses child leases; do not add as independent GPU cost',
        'car_round2_formal_child_v1': 'encloses the child GPU lease; do not add both',
        'car_round2_assessment_sequence_v1': 'group wall encloses child costs; do not add both',
        'car_round2_assessment_sequence_child_v1': 'encloses the GPU lease; do not add both',
    }
    variants = {
        ('car_round2_assessment_sequence_v1', 'group encloses GPU leases and independent CPU score; nonadditive'),
        ('car_round2_validation_successor_v11', 'Encloses GPU lease and independent CPU scores; group wall is nonadditive.'),
    }
    identity = (doc.get('schema'), doc.get('wall_accounting'))
    if identity in variants:
        children = doc.get('children')
        if doc.get('status') != 'completed' or not isinstance(children, list) or not children:
            raise ValueError('terminal enclosure must bind completed children')
        expected = 3 if identity[0] == 'car_round2_assessment_sequence_v1' else 26
        if len(children) != expected or len({ref.get('path') for ref in children}) != expected:
            raise ValueError('terminal enclosure child coverage differs')
        refs = [{'path': doc.get('launch_receipt'), 'sha256': doc.get('launch_sha256')}, *children]
        docs = []
        for ref in refs:
            path = Path(ref.get('path') or '')
            if not path.is_file() or digest(path) != ref.get('sha256'):
                raise ValueError('terminal enclosure child or launch binding changed')
            docs.append(json.loads(path.read_text()))
        launch_schema = ('car_round2_assessment_sequence_launch_v1' if expected == 3
                         else 'car_round2_validation_successor_launch_v1')
        if docs[0].get('schema') != launch_schema or any(
                child.get('schema') != 'car_round2_assessment_sequence_child_v1' or child.get('returncode') != 0
                for child in docs[1:]):
            raise ValueError('terminal enclosure launch or child failed')
        return
    if (doc.get('schema') not in allowed or doc.get('wall_accounting') != allowed[doc['schema']]
            or (doc.get('schema') == 'car_round2_assessment_sequence_child_v1' and doc.get('kind') != 'gpu')):
        raise ValueError('excluded enclosure must be a recognized measured sequence GPU/group wrapper')


def _number(value, *, positive=False):
    return (isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def _interval(row):
    if (row.get('schema') in {'car_round2_source_residual_microcase_v1', 'car_round2_native_head_chunk_microcase_v1'}
            and 'start_unix' not in row):
        row = row.get('timer', {})
    start = row.get('start_unix',row.get('started_unix'))
    end = row.get('end_unix',row.get('finished_unix'))
    if not _number(start) or not _number(end) or end <= start:
        raise ValueError('missing or nonpositive chronological job interval')
    wall = row.get('seconds',row.get('wall_seconds'))
    if wall is not None and (not _number(wall,positive=True) or not math.isclose(wall,end-start,abs_tol=1e-3,rel_tol=1e-9)):
        raise ValueError('job wall duration disagrees with interval')
    return float(start),float(end)


def _bound_file(binding, campaign, references):
    path = Path(binding['path']).resolve()
    if campaign not in path.parents:
        raise ValueError(f'external receipt: {path}')
    if not path.is_file():
        raise FileNotFoundError(path)
    if digest(path) != binding['sha256']:
        raise ValueError(f'changed receipt: {path}')
    references[str(path)] = binding['sha256']
    return json.loads(path.read_text())


def _weights(job):
    weights = job.get('allocations')
    if (not isinstance(weights,dict) or not weights or set(weights)-set(ARMS)
            or any(not _number(x,positive=True) for x in weights.values())
            or not math.isclose(sum(weights.values()),1.,rel_tol=1e-10,abs_tol=1e-10)):
        raise ValueError('every job requires explicit positive fractions summing to one across known arms')
    scope = job.get('scope')
    if not isinstance(scope,str) or not scope:
        raise ValueError('job requires a named scope')
    if scope in RESIDUAL_ONLY and any(not arm.endswith('Residual_Repair') for arm in weights):
        raise ValueError('allocation/policy/residual-probe costs belong to residual candidates')
    if scope == 'training' and (len(weights) != 1 or next(iter(weights.values())) != 1):
        raise ValueError('each formal training attempt belongs to exactly one arm')
    if scope in {'source_preparation','bank_selection','common_observer'}:
        for prefix in ('FFM_','Senseiver_'):
            pair = [weights.get(arm,0.) for arm in ARMS if arm.startswith(prefix)]
            if any(pair) and not math.isclose(pair[0],pair[1],rel_tol=1e-10,abs_tol=1e-10):
                raise ValueError('source/common preparation must be shared equally within a source pair')
    return weights


def _overlap(start, end, intervals):
    points = sorted((max(start,a),min(end,b)) for a,b in intervals if max(start,a) < min(end,b))
    total = 0.; last_a = last_b = None
    for a,b in points:
        if last_a is None:
            last_a,last_b = a,b
        elif a <= last_b:
            last_b = max(last_b,b)
        else:
            total += last_b-last_a
            last_a,last_b = a,b
    return total + (last_b-last_a if last_a is not None else 0.)


class _UnknownActivePrefix(ValueError):
    """An aggregate active timer does not locate work within its enclosure."""


def _active_wall(receipt, field):
    """Validate an explicitly independent disjoint controller timer."""
    schema = receipt.get('schema')
    if field == 'active_wall_seconds' and receipt.get('independent_cpu') is True:
        if schema == 'car_round2_independent_cpu_job_v1' and 'nested_child_wall_seconds' in receipt:
            waits = ('blocked_wait_seconds', 'nested_child_wall_seconds')
        elif schema == 'car_round2_independent_cpu_job_v1':
            waits = ('blocked_wait_seconds', 'gpu_child_wall_seconds', 'score_wait_seconds')
        elif schema == 'car_round2_cpu_coordination_v1':
            waits = ('normalizer_child_wait_seconds', 'gpu_child_wait_seconds', 'score_parent_wait_seconds')
        else:
            raise ValueError('unsupported independent active-wall receipt schema')
    elif (field == 'controller_active_wall_seconds' and schema == 'car_round2_cpu_coordination_v1'
          and receipt.get('controller_active_wall_independent_cpu') is True):
        waits = ('blocked_wait_seconds', 'nomination_child_wall_seconds', 'parallel_family_enclosure_seconds')
    else:
        raise ValueError('active wall requires an explicit independent CPU declaration')
    start, end = _interval(receipt)
    values = [receipt.get(name) for name in waits]
    value = receipt.get(field)
    if not _number(value) or any(not _number(item) for item in values):
        raise ValueError('active wall and disjoint waits must be finite and nonnegative')
    expected = end - start - sum(values)
    if expected < -1e-3 or not math.isclose(value, max(0., expected), abs_tol=1e-3, rel_tol=1e-9):
        raise ValueError('active wall does not equal enclosure minus disjoint waits')
    return float(value)


def _prefix_cost(job, cutoff):
    start,end = job['interval']
    stop = min(end,cutoff)
    if stop <= start:
        return 0.
    if 'active_wall_seconds' in job:
        if stop < end:
            raise _UnknownActivePrefix('aggregate active timer has no checkpoint-time placement')
        return job['active_wall_seconds']
    return stop-start-_overlap(start,stop,job.get('excluded_intervals',[]))


def _prefix_bounds(job, cutoff):
    """Return honest bounds when an aggregate active interval crosses cutoff."""
    start, end = job['interval']
    if 'active_wall_seconds' in job and start < cutoff < end:
        return 0., job['active_wall_seconds']
    value = _prefix_cost(job, cutoff)
    return value, value


def _checkpoint_cost_bounds(jobs, extras, arm, cutoff):
    lower, upper = 0., 0.
    for row in jobs:
        # Every declared assessment extra is counted in full even if its
        # underlying work occurred after the training checkpoint.
        lo, hi = ((row['seconds'], row['seconds']) if row['id'] in extras
                  else _prefix_bounds(row, cutoff))
        lower += lo * row['weights'][arm]
        upper += hi * row['weights'][arm]
    return lower, upper


def _default_checkpoint_loader(path):
    from ..training.run_store import load_project_checkpoint
    return load_project_checkpoint(path)


def index_checkpoint_clock(*, campaign, run, arm):
    campaign, run = Path(campaign).resolve(),Path(run).resolve()
    if arm not in ARMS or campaign not in run.parents:
        raise ValueError('named campaign arm and campaign-owned run required')
    status_path = run/'status.json'
    status = json.loads(status_path.read_text())
    if status.get('status') != 'completed' or status.get('global_step') != 12000:
        raise ValueError('complete mature 12000-update run required')
    journal = run/'logs/checkpoint_timing.jsonl'
    rows = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    epochs = {}
    for epoch in EPOCHS:
        path = run/'checkpoints'/f'epoch_{epoch:03d}.pt'
        checkpoint_sha = digest(path)
        matches = [row for row in rows if row.get('schema') == 'checkpoint_clock_v1'
                   and row.get('epoch') == epoch and row.get('global_step') == epoch*12
                   and row.get('checkpoint_sha256') == checkpoint_sha
                   and Path(row.get('checkpoint_path','')).resolve() == path.resolve()]
        if not matches:
            raise ValueError(f'epoch {epoch}: no clock matching actual checkpoint')
        epochs[str(epoch)] = max(matches,key=lambda row: row['recorded_unix'])
    return {'schema':'car_round2_checkpoint_clock_index_v1','arm':arm,'run':str(run),
            'journal_path':str(journal),'journal_sha256':digest(journal),
            'status_path':str(status_path),'status_sha256':digest(status_path),'epochs':epochs}


def reconcile(*, campaign, ledger, manifest, family, checkpoint_loader=_default_checkpoint_loader):
    campaign = Path(campaign).resolve()
    if family not in ('ffm','senseiver') or manifest.get('schema') != 'car_round2_cost_allocation_v1':
        raise ValueError('Round 2 allocation manifest and source family required')
    arms = [arm for arm in ARMS if arm.lower().startswith(family)]
    references: dict[str,str] = {}
    missing: list[str] = list(manifest.get('assembly_missing', []))
    for key in ('registry_binding', 'ledger_binding'):
        if key in manifest:
            try:
                _bound_file(manifest[key], campaign, references)
            except (KeyError, FileNotFoundError, ValueError) as exc:
                missing.append(f'{key} missing or changed: {exc}')
    for binding in manifest.get('excluded_enclosures', []):
        try:
            validate_excluded_enclosure(_bound_file(binding, campaign, references))
        except (KeyError, FileNotFoundError, ValueError) as exc:
            missing.append(f'excluded enclosure missing or changed: {exc}')
    if ledger.get('schema') not in ('car_gpu_ledger_v1','car_round2_gpu_ledger_v1'):
        raise ValueError('invalid campaign GPU ledger')
    if ledger.get('active'):
        missing.append('campaign has active GPU leases; reconciliation is not terminal')
    closed = ledger.get('closed')
    if not isinstance(closed,list):
        raise TypeError('closed GPU lease list missing')
    by_token = {}
    for entry in closed:
        token = entry.get('token')
        if not token or token in by_token:
            raise ValueError('missing or duplicate GPU lease token')
        _interval(entry)
        if not entry.get('gpu_uuid'):
            raise ValueError('GPU lease lacks physical device identity')
        by_token[token] = entry
    for gpu in {row['gpu_uuid'] for row in closed}:
        spans = sorted((_interval(row),row['token']) for row in closed if row['gpu_uuid'] == gpu)
        if any(a[0][1] > b[0][0] for a,b in itertools.pairwise(spans)):
            raise ValueError('overlapping closed leases on one physical GPU')
    jobs, seen_tokens, by_id = [], set(), {}
    for job in manifest.get('gpu_jobs',[]):
        identifier, token = job.get('id'), job.get('token')
        if not identifier or identifier in by_id or token in seen_tokens:
            raise ValueError('GPU job IDs and lease tokens must be unique')
        seen_tokens.add(token)
        if token not in by_token:
            missing.append(f'GPU job {identifier}: closed lease {token} missing')
            continue
        lease = by_token[token]; start,end = _interval(lease)
        weights = _weights(job)
        try:
            bindings = [job['receipt']] if 'receipt' in job else [job['launch'], job['completion']]
            for binding in bindings:
                _bound_file(binding, campaign, references)
        except (KeyError, FileNotFoundError) as exc:
            missing.append(f'GPU job {identifier}: receipt missing: {exc}')
            continue
        if 'receipt' in job:
            receipt = _bound_file(job['receipt'],campaign,references)
            receipt_start,receipt_end = _interval(receipt)
            if (receipt.get('token') != token or receipt.get('gpu_uuid') != lease['gpu_uuid'] or not math.isclose(receipt_start,start,rel_tol=0.,abs_tol=1e-3) or not math.isclose(receipt_end,end,rel_tol=0.,abs_tol=1e-3)):
                raise ValueError(f'GPU job {identifier}: receipt does not match closed lease')
            status = receipt.get('status')
        else:
            launch = _bound_file(job['launch'],campaign,references)
            completion = _bound_file(job['completion'],campaign,references)
            if (launch.get('token') != token or completion.get('token') != token
                    or launch.get('gpu_uuid') != lease['gpu_uuid']
                    or not math.isclose(launch.get('started_unix',-1),start,rel_tol=0.,abs_tol=1e-3)
                    or not math.isclose(completion.get('finished_unix',-1),end,rel_tol=0.,abs_tol=1e-3)):
                raise ValueError(f'GPU job {identifier}: launch/completion do not match closed lease')
            status = completion.get('status')
        if not status or status in {'running','created'}:
            raise ValueError(f'GPU job {identifier}: terminal status required')
        value = {'id':identifier, 'kind':'gpu','scope':job['scope'], 'token':token, 'gpu_uuid':lease['gpu_uuid'],
                 'status':status,'interval':(start,end),'seconds':end-start,'weights':weights,
                 'checkpoint_epochs':job.get('checkpoint_epochs',[])}
        jobs.append(value); by_id[identifier] = value
    for token in sorted(set(by_token)-seen_tokens):
        missing.append(f'closed GPU lease {token}: no allocation and receipt')
    cpu_pending = []
    cpu_jobs = manifest.get('cpu_jobs', [])
    registered_receipts = [str(Path(job['receipt']['path']).resolve()) for job in cpu_jobs if 'receipt' in job]
    unplaced_sources = set()
    for job in cpu_jobs:
        identifier = job.get('id')
        if not identifier or identifier in by_id or any(x['id']==identifier for x in cpu_pending):
            raise ValueError('job IDs must be unique')
        try:
            receipt = _bound_file(job['receipt'],campaign,references)
        except (KeyError, FileNotFoundError) as exc:
            missing.append(f'CPU job {identifier}: receipt missing: {exc}')
            continue
        unplaced = receipt.get('schema') == 'car_round2_unplaced_cpu_duration_v1'
        if unplaced:
            if job.get('independent') is not True or job.get('parent_gpu_token') or job.get('parent_cpu_job') or 'active_wall_field' in job:
                raise ValueError('unplaced duration requires independent CPU without parent or active timer')
            source_binding = receipt['source_receipt']
            source_path = str(Path(source_binding['path']).resolve())
            if source_path in registered_receipts or source_path in unplaced_sources:
                raise ValueError('unplaced source registered more than once')
            source = _bound_file(source_binding, campaign, references)
            unplaced_sources.add(source_path)
            start, end = _interval(source)
            if (receipt.get('envelope_start_unix') != start or receipt.get('envelope_end_unix') != end
                    or source.get('independent_cpu') is not True or receipt.get('independent_cpu') is not True
                    or source.get('status') != 'completed' or receipt.get('status') != 'completed'):
                raise ValueError('unplaced duration source or envelope mismatch')
            tests = source.get('tests')
            if not isinstance(tests, list) or not tests or any(t.get('returncode') != 0 or not _number(t.get('wall_seconds')) for t in tests):
                raise ValueError('unplaced duration requires measured successful test walls')
            duration = receipt.get('measured_duration_seconds')
            if not _number(duration) or duration > end-start or not math.isclose(duration, sum(t['wall_seconds'] for t in tests), rel_tol=0., abs_tol=1e-9):
                raise ValueError('unplaced duration disagrees with source or envelope')
        else:
            start,end = _interval(receipt)
        if receipt.get('status') in {'running','created'}:
            raise ValueError('CPU job receipt must be terminal')
        cpu_status = receipt.get('status', 'completed' if receipt.get('returncode', 0) == 0 else 'failed')
        row = {'id':identifier,'kind':'cpu','scope':job['scope'],'status':cpu_status,
               'interval':(start,end),'weights':_weights(job),'parent_gpu_token':job.get('parent_gpu_token'),
               'parent_cpu_job':job.get('parent_cpu_job'),'independent':job.get('independent'),
               'checkpoint_epochs':job.get('checkpoint_epochs',[])}
        if unplaced:
            row['active_wall_seconds'] = float(duration)
            row['active_wall_field'] = 'measured_duration_seconds'
        if (job.get('independent') is True and 'active_wall_field' not in job
                and ('active_wall_seconds' in receipt or 'controller_active_wall_seconds' in receipt)):
            raise ValueError('independent controller receipt must explicitly select its active wall field')
        if 'active_wall_field' in job:
            if job.get('independent') is not True or job.get('parent_gpu_token') or job.get('parent_cpu_job'):
                raise ValueError('aggregate active wall must be declared independent and have no parent')
            row['active_wall_seconds'] = _active_wall(receipt, job['active_wall_field'])
            row['active_wall_field'] = job['active_wall_field']
        cpu_pending.append(row)
    cpu_index = {row['id']:row for row in cpu_pending}
    for row in cpu_pending:
        excluded = []
        parent_gpu, parent_cpu = row['parent_gpu_token'],row['parent_cpu_job']
        if row['independent'] is not True and not parent_gpu and not parent_cpu:
            raise ValueError(f"CPU job {row['id']}: declare independent or the containing parent")
        if row['independent'] is True and (parent_gpu or parent_cpu):
            raise ValueError('independent CPU work cannot also declare a parent')
        if parent_gpu:
            if parent_gpu not in by_token:
                missing.append(f"CPU job {row['id']}: parent GPU lease missing")
                continue
            parent = next((item for item in jobs if item.get('token') == parent_gpu),None)
            if parent is None or parent['weights'] != row['weights']:
                raise ValueError('nested CPU work must have the same allocation as its GPU parent')
            excluded.append(_interval(by_token[parent_gpu]))
        if parent_cpu:
            if parent_cpu == row['id'] or parent_cpu not in cpu_index:
                raise ValueError('CPU parent must name a distinct declared CPU job')
            parent = cpu_index[parent_cpu]
            if parent['parent_cpu_job']:
                raise ValueError('nested CPU accounting supports one explicit parent level')
            if parent['weights'] != row['weights']:
                raise ValueError('nested CPU work must have the same allocation as its CPU parent')
            excluded.append(parent['interval'])
        row['excluded_intervals'] = excluded
        row['seconds'] = _prefix_cost(row,row['interval'][1])
        jobs.append(row); by_id[row['id']] = row
    for arm in arms:
        present = {job['scope'] for job in jobs if job['weights'].get(arm,0.) > 0}
        for scope in sorted(REQUIRED_SCOPES-present):
            missing.append(f'{arm}: missing measured {scope} scope')
        if arm.endswith('Residual_Repair') and not (present & POLICY_SCOPES):
            missing.append(f'{arm}: missing allocation audit or explicit failed allocation cost')
    checkpoint_clocks: dict[str,dict[int,float]] = {}
    for arm in arms:
        declaration = manifest.get('checkpoint_runs',{}).get(arm)
        if not declaration:
            missing.append(f'{arm}: checkpoint clock declaration missing'); continue
        try:
            clock = _bound_file(declaration['receipt'],campaign,references)
            if clock.get('schema') != 'car_round2_checkpoint_clock_index_v1' or clock.get('arm') != arm:
                raise ValueError('invalid checkpoint clock index')
            journal = Path(clock['journal_path']).resolve()
            status_path = Path(clock['status_path']).resolve()
            if (campaign not in journal.parents or campaign not in status_path.parents
                    or digest(journal) != clock['journal_sha256'] or digest(status_path) != clock['status_sha256']):
                raise ValueError('checkpoint journal or terminal status changed')
            references[str(journal)] = clock['journal_sha256']
            references[str(status_path)] = clock['status_sha256']
            status = json.loads(status_path.read_text())
            if status.get('status') != 'completed' or status.get('global_step') != 12000:
                raise ValueError('checkpoint run is not a completed mature attempt')
            journal_rows = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
            training_intervals = [j['interval'] for j in jobs if j['scope']=='training' and j['weights'].get(arm)==1]
            clocks = {}
            for epoch in EPOCHS:
                entry = clock['epochs'][str(epoch)]
                if entry not in journal_rows:
                    raise ValueError(f'epoch {epoch} clock is absent from recorded journal')
                checkpoint = Path(entry['checkpoint_path']).resolve()
                if campaign not in checkpoint.parents or digest(checkpoint) != entry['checkpoint_sha256']:
                    raise ValueError(f'epoch {epoch} checkpoint hash mismatch')
                references[str(checkpoint)] = entry['checkpoint_sha256']
                when = entry['recorded_unix']
                if not _number(when) or entry['global_step'] != 12*epoch or not any(a <= when <= b for a,b in training_intervals):
                    raise ValueError(f'epoch {epoch} clock outside its formal training lease')
                clocks[epoch] = float(when)
            if any(clocks[a] > clocks[b] for a,b in itertools.pairwise(EPOCHS)):
                raise ValueError('checkpoint times are not chronological')
            checkpoint_clocks[arm] = clocks
        except (KeyError,FileNotFoundError,ValueError) as exc:
            missing.append(f'{arm}: checkpoint evidence incomplete: {exc}')
    arm_results = {}
    for arm in arms:
        own = [row for row in jobs if row['weights'].get(arm,0.) > 0]
        measured = sum(row['seconds']*row['weights'][arm] for row in own)
        checkpoints, checkpoint_bounds = {}, {}
        if arm in checkpoint_clocks:
            for epoch,cutoff in checkpoint_clocks[arm].items():
                extras = manifest.get('checkpoint_extras',{}).get(arm,{}).get(str(epoch))
                if not isinstance(extras,list):
                    missing.append(f'{arm}: epoch {epoch} post-checkpoint assessment allocation unspecified')
                    continue
                if not extras:
                    missing.append(f'{arm}: epoch {epoch} assessment costs not supplied')
                    continue
                if len(extras) != len(set(extras)) or any(job_id not in by_id for job_id in extras):
                    raise ValueError('checkpoint extra job IDs must be unique and declared')
                if any(by_id[j]['scope'] not in {'assessment', 'deployment'} or by_id[j]['weights'].get(arm,0.) <= 0
                       or epoch not in by_id[j]['checkpoint_epochs'] for j in extras):
                    raise ValueError('checkpoint extras must be attributable assessment/deployment jobs')
                lower, upper = _checkpoint_cost_bounds(own, extras, arm, cutoff)
                checkpoint_bounds[str(epoch)] = [lower, upper]
                checkpoints[str(epoch)] = lower if lower == upper else None
        arm_results[arm] = {'serial_cost_seconds':measured if own else None,
                            'observed_serial_cost_seconds':measured if own else None,
                            'checkpoint_cost_seconds':checkpoints,
                            'checkpoint_cost_bounds_seconds':checkpoint_bounds,
                            'gpu_seconds':sum(row['seconds']*row['weights'][arm] for row in own if row['kind']=='gpu'),
                            'independent_cpu_seconds':sum(row['seconds']*row['weights'][arm] for row in own if row['kind']=='cpu'),
                            'paused_attempt_seconds':sum(row['seconds']*row['weights'][arm] for row in own if row['status'] == 'budget_paused'),
                            'failed_attempt_seconds':sum(row['seconds']*row['weights'][arm] for row in own if row['status'] not in {'completed','prepared','budget_paused'})}
    allocation = manifest.get('allocator_disposition',{}).get(family)
    allocator_result: dict[str,Any] = {'status':'incomplete'}
    if allocation == 'uniform':
        allocator_result = {'status':'uniform_fallback','qualified':False,'scope':'no nonuniform policy deployed'}
    elif allocation == 'qualified_policy':
        full = next(arm for arm in arms if arm.endswith('Uniform_FullModel'))
        residual = next(arm for arm in arms if arm.endswith('Residual_Repair'))
        try:
            payloads = {}
            for arm in arms:
                binding = manifest['final_checkpoints'][arm]
                path = Path(binding['path']).resolve()
                if campaign not in path.parents or digest(path) != binding['sha256']:
                    raise ValueError(f'{arm}: final checkpoint hash mismatch')
                final_clock = manifest['checkpoint_runs'][arm]['receipt']
                clock = _bound_file(final_clock, campaign, references)
                if clock['epochs']['1000']['checkpoint_sha256'] != binding['sha256'] or Path(clock['epochs']['1000']['checkpoint_path']).resolve() != path:
                    raise ValueError(f'{arm}: final checkpoint differs from clocked epoch 1000')
                references[str(path)] = binding['sha256']
                payload = checkpoint_loader(path)
                runtime = payload['scalable_runtime']
                if (payload.get('global_step') != 12000 or runtime.get('sampler_timer_complete') is not True
                        or runtime.get('sampler_completed_updates') != 12000
                        or not _number(runtime.get('sampler_seconds')) or not _number(runtime.get('policy_seconds'))):
                    raise ValueError(f'{arm}: complete measured final sampler/policy telemetry missing')
                payloads[arm] = runtime
            measured_delta = payloads[residual]['sampler_seconds'] - payloads[full]['sampler_seconds']
            policy_cpu = payloads[residual]['policy_seconds']
            research = sum(row['seconds']*row['weights'].get(residual,0.) for row in jobs if row['scope'] in POLICY_SCOPES)
            training = sum(row['seconds']*row['weights'].get(residual,0.) for row in jobs if row['scope']=='training')
            if training <= 0:
                raise ValueError('actual candidate training cost missing')
            total = research + policy_cpu + max(0.,measured_delta)
            allocator_result = {'status':'measured','qualified':total/training <= .05,'overhead_seconds':total,
                                'overhead_fraction':total/training,'candidate_training_seconds':training,
                                'allocated_policy_research_seconds':research,'online_policy_seconds':policy_cpu,
                                'candidate_sampler_seconds':payloads[residual]['sampler_seconds'],
                                'control_sampler_seconds':payloads[full]['sampler_seconds'],
                                'signed_sampler_difference_seconds':measured_delta,
                                'scope':'measured candidate sampler excess plus online policy and allocated policy-only research over actual candidate training cost'}
        except (KeyError,FileNotFoundError,ValueError) as exc:
            missing.append(f'{residual}: actual allocator overhead incomplete: {exc}')
    else:
        missing.append(f'{family}: final allocator disposition missing')
    total_gpu = sum(_interval(row)[1]-_interval(row)[0] for row in closed)
    if missing:
        for value in arm_results.values():
            value['serial_cost_seconds'] = None
            value['checkpoint_cost_seconds'] = {}
            value['checkpoint_cost_bounds_seconds'] = {}
        if allocator_result.get('status') == 'measured':
            allocator_result['qualified'] = False
            allocator_result['status'] = 'incomplete_campaign_costs'
    all_sources = [{'path':path,'sha256':sha} for path,sha in sorted(references.items())]
    return {'schema':'car_round2_cost_reconciliation_v1','family':family,
            'complete':not missing,'missing':missing,'within_budget':not ledger.get('active') and total_gpu <= CAP_SECONDS,
            'total_occupied_gpu_seconds':total_gpu, 'arms':arm_results, 'source_receipts':all_sources,
            'allocator_final':allocator_result, 'allocated_jobs':jobs,
            'historical_sunk':manifest.get('historical_sunk',[]),
            'semantics':'additive occupied GPU wall plus independent CPU wall outside the same explicitly declared parent lease/job; common costs split by explicit weights'}
