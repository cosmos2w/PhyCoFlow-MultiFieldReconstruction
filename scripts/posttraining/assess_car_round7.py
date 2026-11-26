#!/usr/bin/env python3
"""Round 7 matched bank generation and scoring. TEST is intentionally unavailable.

All file inputs in --spec are hash-bound {path, sha256} references. See
`--help` and `spec-template` for the complete configuration contract.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def checked(ref):
    path = Path(ref['path']).resolve(strict=True)
    if digest(path) != ref['sha256']:
        raise ValueError(f'hash mismatch: {path}')
    return path


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def ref(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': digest(path)}


def validate_spec(spec):
    from scripts.posttraining.assess_car_round2_checkpoints import panel_contract
    if spec.get('schema') != 'car_round7_assessment_spec_v1' or spec.get('split') not in ('train', 'validation'):
        raise ValueError('Round 7 TRAIN or VALIDATION specification required; TEST is locked')
    campaign = Path(spec['campaign_root']).resolve(strict=True)
    data_root = Path('/data2/wanglz/PhyCoFlow-MultiFieldReconstruction').resolve(strict=True)
    if data_root not in campaign.parents:
        raise ValueError('campaign must resolve within the DATA2 project root')
    panel = load(checked(spec['panel']))
    panel_contract(panel, spec['split'])
    query = panel['query_indices']
    if len(query) != 4096 or len(set(query)) != 4096 or min(query) < 0 or max(query) >= panel['native_points']:
        raise ValueError('fixed Q4096 panel required')
    for name in ('source_native_receipt', 'run_config', 'core', 'core_config', 'core_manifest',
                 'expanded_bank', 'prefit', 'assessment_contract', 'source_checkpoint'):
        checked(spec[name])
    source = load(checked(spec['source_native_receipt']))
    if (source.get('status') != 'completed' or source.get('split') != spec['split'] or
        source.get('sample_ids') != panel['sample_ids'] or source.get('members') != 4 or
        source.get('panel_sha256') != spec['panel']['sha256'] or
        source.get('source_sha256') != spec['source_checkpoint']['sha256']):
        raise ValueError('native SOURCE, panel or checkpoint identities do not match')
    if len(source.get('files_sha256', [])) != len(panel['sample_ids']):
        raise ValueError('SOURCE receipt must bind every row')
    if not isinstance(spec.get('native_allowance_model'), list) or len(spec['native_allowance_model']) != len(source['field_order']) or any(not math.isfinite(x) or x <= 0 for x in spec['native_allowance_model']):
        raise ValueError('positive ordered Round 7 training-law allowance required')
    if len(spec.get('field_units', [])) != len(source['field_order']) or not spec.get('physical_mask'):
        raise ValueError('ordered field units and physical mass/mask declaration required')
    return panel, source


def assessment_output(spec, output):
    output = Path(output).resolve()
    if (Path(spec['campaign_root']).resolve(strict=True) / 'assessment') not in output.parents:
        raise ValueError('new output must be inside the DATA2 campaign assessment directory')
    return output


def verify_bank(path, spec, panel, source, support):
    path = Path(path).resolve(strict=True)
    receipt = load(path)
    if (receipt.get('status') != 'completed' or receipt.get('split') != spec['split'] or
        receipt.get('sample_ids') != panel['sample_ids'] or receipt.get('members') != 4 or
        receipt.get('panel_sha256') != spec['panel']['sha256'] or
        receipt.get('source_sha256') != spec['source_checkpoint']['sha256'] or
        receipt.get('source_bank_sha256') != spec['source_native_receipt']['sha256'] or
        receipt.get('support') != support or len(receipt.get('files_sha256', [])) != len(panel['sample_ids'])):
        raise ValueError(f'incompatible {support} bank: {path}')
    return path, receipt


def source_native_bank(spec, panel, source):
    # The immutable Round 2 SOURCE is native even though its old receipt predates a support key.
    path = checked(spec['source_native_receipt'])
    return path, source


def read_row(path, receipt, ordinal):
    import torch
    rowpath = path.parent / f'{ordinal:04d}.pt'
    if digest(rowpath) != receipt['files_sha256'][ordinal]:
        raise ValueError(f'row changed: {rowpath}')
    row = torch.load(rowpath, map_location='cpu', weights_only=True)
    if row['sample_id'] != receipt['sample_ids'][ordinal] or len(row['member_seeds']) != 4:
        raise ValueError('row sample/member identity differs')
    return row


def generate(args, spec, panel, source):
    import torch
    import yaml
    from phycoflow_reconstruction.data.factory import open_field_dataset
    from phycoflow_reconstruction.data.sensor_protocols import SensorProtocol, build_observation_batch
    from phycoflow_reconstruction.training.model_lifecycle import load_training_aux_state
    from phycoflow_reconstruction.training.rollout import differentiable_reconstruction
    from phycoflow_reconstruction.training.run_store import load_model_state_strict, load_project_checkpoint
    from phycoflow_reconstruction.training.source import load_source_model
    from phycoflow_reconstruction.training.scalable_fitting import _digest
    from scripts.posttraining.assess_car_round2_checkpoints import matched_rows
    token = os.environ.get('CAR_ROUND7_LEASE_TOKEN')
    if not token:
        raise ValueError('enclosing Round 7 assessment lease token required')
    checkpoint = checked(spec['checkpoint']) if spec.get('checkpoint') else None
    cfg = yaml.safe_load(checked(spec['run_config']).read_text())
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('prediction generation requires an explicitly selected CUDA device')
    output = assessment_output(spec, args.output)
    output.mkdir(parents=True, exist_ok=False)
    start, cpu = time.time(), time.process_time()
    receipt = {'schema': 'car_round7_prediction_bank_v1', 'status': 'failed', 'split': spec['split'],
               'support': args.support, 'sample_ids': panel['sample_ids'], 'members': 4,
               'field_order': source['field_order'], 'panel_sha256': spec['panel']['sha256'],
               'source_sha256': spec['source_checkpoint']['sha256'], 'source_bank_sha256': spec['source_native_receipt']['sha256'],
               'checkpoint_sha256': digest(checkpoint) if checkpoint else spec['source_checkpoint']['sha256'],
               'run_config_sha256': spec['run_config']['sha256'], 'spec_sha256': digest(args.spec),
               'test_locked': True, 'start_unix': start, 'device': str(device), 'lease_token': token}
    dataset = train = None
    try:
        model, train, _ = load_source_model(cfg, device)
        if train.normalizer.digest() != source['normalizer_sha256'] or _digest(model.state_dict()) != source['source_parameter_buffer_sha256']:
            raise ValueError('SOURCE parameters or normalizer differ from cached bank')
        dataset = train if spec['split'] == 'train' else open_field_dataset(cfg['dataset'], split='validation', normalizer=train.normalizer)
        if checkpoint:
            state = load_project_checkpoint(checkpoint)
            if state.get('source_hashes', {}).get('checkpoint') != spec['source_checkpoint']['sha256']:
                raise ValueError('checkpoint SOURCE identity differs')
            load_model_state_strict(model, state['model'])
            load_training_aux_state(model, state)
        model.eval()
        receipt['device_name'] = torch.cuda.get_device_name(device)
        manifest = checked(spec['source_native_receipt'])
        files = []
        protocol = SensorProtocol(field_counts={'T': 256}, seed=2047)
        with torch.no_grad():
            for ordinal, sid in enumerate(panel['sample_ids']):
                base = read_row(manifest, source, ordinal)
                index = int(base['sample_index'])
                sample = dataset[index]
                actual = f'{dataset.trajectory_ids[dataset._items[index][0]]}:{dataset._items[index][1]}'
                if sid != actual or len(base['obs_indices'][0]) != 256:
                    raise ValueError('dataset or observation identity mismatch')
                obs = {sid: list(zip(base['obs_indices'][0].tolist(), base['obs_fields'][0].tolist(), strict=True))}
                query = panel['query_indices'] if args.support == 'selected' else None
                batch = build_observation_batch([sample], protocol, query_indices=query, manifest_indices=obs).to(device)
                members = [differentiable_reconstruction(model, replace(batch, target_fields=None),
                    steps=int(cfg['rollout']['steps']), solver=cfg['rollout']['solver'],
                    generator=torch.Generator(device=device).manual_seed(seed),
                    observation_config=cfg['observation_consistency']).detach().cpu() for seed in base['member_seeds']]
                indices = torch.as_tensor(query if query is not None else list(range(len(sample.values))), dtype=torch.long)
                row = dict(sample_id=sid, sample_index=index, members_model=torch.stack(members),
                    truth_model=batch.target_fields.detach().cpu(), coordinates_model=batch.query_coords.detach().cpu(),
                    coordinates_raw=sample.coordinates_raw[indices].cpu(), native_masses=dataset.physical_weights[indices].cpu(),
                    obs_indices=batch.obs_indices.detach().cpu(), obs_values=batch.obs_values.detach().cpu(),
                    obs_fields=batch.obs_field_ids.detach().cpu(), member_seeds=base['member_seeds'],
                    normalizer_offset=dataset.normalizer.offset.cpu(), normalizer_scale=dataset.normalizer.scale.cpu())
                if args.support == 'native':
                    matched_rows(base, row)
                else:
                    for name in ('sample_id','sample_index','member_seeds','obs_indices','obs_values','obs_fields','normalizer_offset','normalizer_scale'):
                        left, right = base[name], row[name]
                        if not (torch.equal(left, right) if torch.is_tensor(left) else left == right):
                            raise ValueError(f'selected observation/member mismatch: {name}')
                    for name in ('truth_model', 'coordinates_model'):
                        if not torch.equal(base[name][:, indices], row[name]):
                            raise ValueError(f'selected {name} mismatch')
                    if not torch.equal(base['native_masses'][indices], row['native_masses']):
                        raise ValueError('selected physical mass mismatch')
                path = output / f'{ordinal:04d}.pt'
                torch.save(row, path); files.append(digest(path))
        receipt.update(status='completed', files_sha256=files)
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if dataset is not None and dataset is not train: dataset.close()
        if train is not None: train.close()
        receipt.update(end_unix=time.time(), wall_seconds=time.time()-start, cpu_seconds=time.process_time()-cpu)
        write(output / 'receipt.json', receipt)


def build_observer(spec, source):
    import torch
    import yaml
    from phycoflow_reconstruction.coherence import build_enabled_families
    from phycoflow_reconstruction.coherence.adaptive_fitting import validate_bank, validate_bank_target_binding
    from phycoflow_reconstruction.data.factory import open_field_dataset
    cfg = yaml.safe_load(checked(spec['core_config']).read_text())
    dataset = open_field_dataset(cfg['dataset'], split='train')
    try:
        if dataset.normalizer.digest() != source['normalizer_sha256'] or list(dataset.field_names) != source['field_order']:
            raise ValueError('frozen normalizer or field order differs')
        families = build_enabled_families(cfg['coherence'], dataset.data_spec, dataset.normalizer)
        manifest = load(checked(spec['core_manifest']))
        root = Path(spec['core_config']['path']).resolve().parent
        for name, family in families.items():
            path = root / 'artifacts' / f'{name}_family.pt'
            if digest(path) != manifest['coherence_family_state_sha256s'][name]:
                raise ValueError(f'frozen family changed: {name}')
            family.load_state_artifact(torch.load(path, map_location='cpu', weights_only=True))
        core = torch.load(checked(spec['core']), map_location='cpu', weights_only=True)
        if core.get('adaptive_bank') is not None or core['provenance']['source_checkpoint_sha256'] != spec['source_checkpoint']['sha256']:
            raise ValueError('SOURCE-bound frozen core observer required')
        bank = load(checked(spec['expanded_bank']))
        prefit = load(checked(spec['prefit']))
        binding = prefit['binding']
        validate_bank(bank, field_order=source['field_order'], support_identity=binding['support_sha256'], source_identity=source['source_sha256'])
        validate_bank_target_binding(binding, bank)
        if (binding['source_checkpoint_sha256'] != source['source_sha256'] or binding['normalizer_sha256'] != source['normalizer_sha256'] or binding['field_order'] != source['field_order']):
            raise ValueError('expanded bank, prefit and SOURCE identities differ')
        weights = prefit['coefficients']
        expected = {'scale.signed':2.0, 'scale.joint':1.2835023021293974,
                    'scale.log_power_self':0.5, 'scale.rho':0.8190801090444172, 'scale.third_order':0.5}
        family = families['multiscale_spectrum']
        if (any(not math.isclose(weights.get(k,float('nan')),v,rel_tol=1e-12) for k,v in expected.items()) or
            family.settings.joint_energy_score_weight != 0.25 or family.relation_extension['self_weight'] != 1 or
            family.relation_extension['rho_weight'] != 1):
            raise ValueError('frozen complete Scale definitions differ')
        return dataset, families, core, bank, prefit['coefficients']
    except BaseException:
        dataset.close()
        raise



def baseline_binding(spec):
    names = ('panel','source_native_receipt','core','core_config','core_manifest','expanded_bank',
             'prefit','assessment_contract','source_checkpoint')
    return {'split':spec['split'], **{name:spec[name]['sha256'] for name in names},
            'native_allowance_model':spec['native_allowance_model'], 'field_units':spec['field_units'],
            'physical_mask':spec['physical_mask']}


def native_field_row(row, observation_row, field_count):
    import torch
    from phycoflow_reconstruction.evaluation.round2_assessment import native_risks
    from scripts.posttraining.assess_car_round2_checkpoints import sensor_mse
    scale, offset = row['normalizer_scale'], row['normalizer_offset']
    risk = native_risks(row['members_model'] * scale + offset, row['truth_model'] * scale + offset,
                        row['native_masses'].double(), model_family='ffm')
    sensor, counts = sensor_mse(row['members_model'], observation_row, field_count)
    return {**{key:risk[key].tolist() for key in ('member_mse','ensemble_mean_mse','signed_mean_risk','unbiased_member_variance')},
            'sensor_mse':(sensor * scale.double().square()).tolist(), 'sensor_counts':counts.tolist()}


def score_selected(observer, members, sid, label):
    import torch
    with torch.no_grad():
        values = observer.score_tensors(members, include_components=True)
    shape_keys = ['shape.self', 'shape.mutual'] + [
        f'shape.{kind}.{category}.{term}' for kind in ('self','mutual')
        for category in ('finite','essential') for term in ('attraction','separation')]
    result = {key:float(values[key].mean()) for key in ('state','scale','shape',*shape_keys)}
    if not math.isclose(result['shape.self'] + result['shape.mutual'], result['shape'], rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError(f'Shape self/mutual reconstruction mismatch: {sid} {label}')
    for kind in ('self','mutual'):
        terms = [result[f'shape.{kind}.{category}.{term}'] for category in ('finite','essential')
                 for term in ('attraction','separation')]
        reconstructed = terms[0] - terms[1] + terms[2] - terms[3]
        tolerance = 1e-8 + 1e-6 * sum(abs(term) for term in terms)
        if not math.isclose(reconstructed, result[f'shape.{kind}'], rel_tol=1e-6, abs_tol=tolerance):
            raise ValueError(f'Shape finite/essential reconstruction mismatch: {sid} {label} {kind}')
    return result, shape_keys


def verify_selected_reference(native, selected, query):
    import torch
    for key in ('sample_id','sample_index','member_seeds','obs_indices','obs_values','obs_fields','normalizer_offset','normalizer_scale'):
        left,right = native[key], selected[key]
        if not (torch.equal(left,right) if torch.is_tensor(left) else left == right):
            raise ValueError(f'native/selected manifest mismatch: {key}')
    for key in ('truth_model','coordinates_model'):
        if not torch.equal(native[key][:,query],selected[key]):
            raise ValueError(f'native/selected support mismatch: {key}')
    if not torch.equal(native['native_masses'][query], selected['native_masses']):
        raise ValueError('native/selected support mass mismatch')


def baseline(args, spec, panel, source):
    import torch
    from phycoflow_reconstruction.coherence.adaptive_fitting import spectral_filter
    from phycoflow_reconstruction.evaluation.frozen_coherence import FrozenCoherenceObserver
    from scripts.analysis.car_round7_scale_decompose import decompose
    if spec.get('checkpoint') is not None:
        raise ValueError('SOURCE baseline requires checkpoint: null')
    source_path, source_receipt = source_native_bank(spec, panel, source)
    selected_path, selected_receipt = verify_bank(args.source_selected, spec, panel, source, 'selected')
    if selected_receipt['checkpoint_sha256'] != spec['source_checkpoint']['sha256']:
        raise ValueError('SOURCE selected bank checkpoint mismatch')
    output = assessment_output(spec,args.output); output.mkdir(parents=True,exist_ok=False)
    start,cpu=time.time(),time.process_time()
    status={'schema':'car_round7_baseline_job_v1','status':'failed','split':spec['split'],'test_locked':True}
    dataset=None
    try:
        torch.set_num_threads(args.threads)
        dataset,families,core,bank,prefit=build_observer(spec,source)
        contract=load(checked(spec['assessment_contract']))
        query=torch.tensor(panel['query_indices'],dtype=torch.long)
        files=[]; timings={'read_seconds':0.,'native_field_seconds':0.,'observer_and_scale_seconds':0.,'write_seconds':0.}
        for ordinal,sid in enumerate(panel['sample_ids']):
            stage=time.perf_counter()
            native=read_row(source_path,source_receipt,ordinal); selected=read_row(selected_path,selected_receipt,ordinal)
            verify_selected_reference(native,selected,query)
            timings['read_seconds'] += time.perf_counter()-stage
            stage=time.perf_counter(); fields=native_field_row(native,native,len(source['field_order']))
            timings['native_field_seconds'] += time.perf_counter()-stage
            stage=time.perf_counter()
            observer=FrozenCoherenceObserver(core,families,expected_provenance=core['provenance'],
                reference=selected['truth_model'], coordinates=selected['coordinates_model'], sample_ids=[sid],
                member_weights=selected['native_masses'].unsqueeze(0),reference_weights=selected['native_masses'].unsqueeze(0),
                mode='conditional',query_seed=int(contract['shape_query_seed']),complete_scale=True)
            values,_=score_selected(observer,selected['members_model'],sid,'SOURCE')
            with torch.no_grad():
                filtered=spectral_filter(families['multiscale_spectrum'],selected['truth_model'])
                decomposition=decompose(families['multiscale_spectrum'],selected['members_model'],selected['truth_model'],bank,prefit,
                                        target_statistics=observer.scale_reference,target_filtered=filtered)
            if not math.isclose(decomposition['definitions']['Published_Core_Default'],values['scale'],rel_tol=1e-5,abs_tol=1e-7):
                raise ValueError(f'published Scale reconstruction mismatch: {sid} SOURCE')
            saved={'sample_id':sid,'native_fields':fields,'scores':values,'decomposition':decomposition,
                   'compiled_target':observer.export_compiled_target(),'target_filtered':filtered}
            timings['observer_and_scale_seconds'] += time.perf_counter()-stage
            stage=time.perf_counter(); path=output/f'{ordinal:04d}.pt';torch.save(saved,path);files.append(digest(path))
            timings['write_seconds'] += time.perf_counter()-stage
        write(output/'baseline.json',{'schema':'car_round7_source_baseline_v1','status':'completed','binding':baseline_binding(spec),
            'sample_ids':panel['sample_ids'],'source_selected':ref(selected_path),'source_native':ref(source_path),
            'files_sha256':files,'members':4,'timing_seconds':timings,'test_locked':True})
        status.update(status='completed',conditions=len(files),timing_seconds=timings,threads=args.threads)
    except BaseException as error:
        status.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        if dataset is not None:dataset.close()
        status.update(wall_seconds=time.time()-start,cpu_seconds=time.process_time()-cpu)
        write(output/'receipt.json',status)

def score(args, spec, panel, source):
    import numpy as np
    import torch
    from phycoflow_reconstruction.evaluation.frozen_coherence import FrozenCoherenceObserver
    from phycoflow_reconstruction.evaluation.round2_assessment import native_risks, field_sensor_eligibility
    from scripts.analysis.car_round7_scale_decompose import decompose
    from scripts.analysis.car_round7_paired_summary import paired_summary, summarize_scale
    from scripts.posttraining.assess_car_round2_checkpoints import matched_rows, sensor_mse
    source_path, source_receipt = source_native_bank(spec, panel, source)
    native_path, native_receipt = verify_bank(args.child_native, spec, panel, source, 'native')
    selected_source_path, selected_source_receipt = verify_bank(args.source_selected, spec, panel, source, 'selected')
    selected_child_path, selected_child_receipt = verify_bank(args.child_selected, spec, panel, source, 'selected')
    if (selected_source_receipt['checkpoint_sha256'] != spec['source_checkpoint']['sha256'] or
        native_receipt['checkpoint_sha256'] != selected_child_receipt['checkpoint_sha256'] or
        not spec.get('checkpoint') or native_receipt['checkpoint_sha256'] != spec['checkpoint']['sha256'] or
        native_receipt['run_config_sha256'] != spec['run_config']['sha256'] or
        selected_child_receipt['run_config_sha256'] != spec['run_config']['sha256']):
        raise ValueError('SOURCE and child selected/native checkpoint mismatch')
    baseline_path = args.baseline.resolve(strict=True)
    baseline_doc = load(baseline_path)
    if (baseline_doc.get('schema') != 'car_round7_source_baseline_v1' or baseline_doc.get('status') != 'completed' or
        baseline_doc.get('binding') != baseline_binding(spec) or baseline_doc.get('sample_ids') != panel['sample_ids'] or
        baseline_doc.get('source_selected') != ref(selected_source_path) or baseline_doc.get('source_native') != ref(source_path) or
        len(baseline_doc.get('files_sha256', [])) != len(panel['sample_ids'])):
        raise ValueError('SOURCE baseline identity differs from this matched assessment')
    contract = load(checked(spec['assessment_contract']))
    normalizers = contract['normalizers']
    output = assessment_output(spec, args.output); output.mkdir(parents=True, exist_ok=False)
    start, cpu = time.time(), time.process_time()
    status = {'schema': 'car_round7_score_job_v1', 'status': 'failed', 'split': spec['split'], 'test_locked': True}
    dataset = None
    try:
        torch.set_num_threads(args.threads)
        dataset, families, core, bank, prefit = build_observer(spec, source)
        allowance = (torch.as_tensor(spec['native_allowance_model'], dtype=torch.float64) * dataset.normalizer.scale.double().square()).tolist()
        records = []; fields = {'SOURCE': [], 'candidate': []}; scores = {'SOURCE': [], 'candidate': []}
        query = torch.tensor(panel['query_indices'], dtype=torch.long)
        timings = {'read_native_seconds': 0., 'native_field_seconds': 0., 'read_selected_seconds': 0., 'coherence_scale_seconds': 0.}
        for index, sid in enumerate(panel['sample_ids']):
            stage = time.perf_counter()
            s_native = read_row(source_path, source_receipt, index)
            c_native = read_row(native_path, native_receipt, index)
            matched_rows(s_native, c_native)
            cache_path=baseline_path.parent / f'{index:04d}.pt'
            if digest(cache_path) != baseline_doc['files_sha256'][index]:
                raise ValueError(f'SOURCE baseline row changed: {cache_path}')
            cached=torch.load(cache_path,map_location='cpu',weights_only=True)
            if cached['sample_id'] != sid: raise ValueError('SOURCE baseline condition differs')
            fields['SOURCE'].append(cached['native_fields']); scores['SOURCE'].append(cached['scores'])
            timings['read_native_seconds'] += time.perf_counter()-stage
            stage=time.perf_counter()
            fields['candidate'].append(native_field_row(c_native,s_native,len(source['field_order'])))
            timings['native_field_seconds'] += time.perf_counter()-stage
            stage=time.perf_counter()
            s_selected=read_row(selected_source_path,selected_source_receipt,index)
            c_selected=read_row(selected_child_path,selected_child_receipt,index)
            matched_rows(s_selected,c_selected)
            verify_selected_reference(s_native,s_selected,query)
            verify_selected_reference(c_native,c_selected,query)
            timings['read_selected_seconds'] += time.perf_counter()-stage
            stage=time.perf_counter()
            observer=FrozenCoherenceObserver(core,families,expected_provenance=core['provenance'],
                reference=s_selected['truth_model'],coordinates=s_selected['coordinates_model'],sample_ids=[sid],
                member_weights=s_selected['native_masses'].unsqueeze(0),reference_weights=s_selected['native_masses'].unsqueeze(0),
                mode='conditional',query_seed=int(contract['shape_query_seed']),complete_scale=True,
                compiled_target=cached['compiled_target'])
            child_scores,shape_keys=score_selected(observer,c_selected['members_model'],sid,'candidate')
            scores['candidate'].append(child_scores)
            with torch.no_grad():
                child_decomposition=decompose(families['multiscale_spectrum'],c_selected['members_model'],
                    s_selected['truth_model'],bank,prefit,target_statistics=observer.scale_reference,
                    target_filtered=cached['target_filtered'])
            if not math.isclose(child_decomposition['definitions']['Published_Core_Default'],child_scores['scale'],rel_tol=1e-5,abs_tol=1e-7):
                raise ValueError(f'published Scale reconstruction mismatch: {sid} candidate')
            records.append({'sample_id':sid,'member_seeds':s_selected['member_seeds'],
                            'models':{'SOURCE':cached['decomposition'],'candidate':child_decomposition}})
            timings['coherence_scale_seconds'] += time.perf_counter()-stage
        count = len(records)
        index = np.random.default_rng(6042026).integers(0,count,size=(2000,count))
        def mean_fields(name):
            return {key: np.mean([row[key] for row in fields[name]], axis=0).tolist() for key in
                    ('member_mse','ensemble_mean_mse','signed_mean_risk','unbiased_member_variance','sensor_mse')}
        source_means, child_means = mean_fields('SOURCE'), mean_fields('candidate')
        fidelity = field_sensor_eligibility(source_means, child_means, model_family='ffm', absolute_mean_allowance=allowance,
            source_sensor_mse=source_means['sensor_mse'], child_sensor_mse=child_means['sensor_mse'],
            sensor_equality_tolerance=contract['sensor_equality_tolerance'])
        paired_fields = {key: paired_summary([x[key] for x in fields['SOURCE']], [x[key] for x in fields['candidate']], index)
                         for key in ('member_mse','ensemble_mean_mse','signed_mean_risk','unbiased_member_variance','sensor_mse')}
        paired_fields['signed_risk_difference_over_allowance'] = paired_summary(
            [x['signed_mean_risk'] for x in fields['SOURCE']], [x['signed_mean_risk'] for x in fields['candidate']],
            index, np.asarray(allowance))
        if any(x <= 0 for x in source_means['member_mse']):
            raise ValueError('positive SOURCE member MSE needed for ratio')
        sr = np.asarray([r['member_mse'] for r in fields['SOURCE']]); cr = np.asarray([r['member_mse'] for r in fields['candidate']])
        ratio = cr.mean(0)/sr.mean(0); boots = cr[index].mean(1)/sr[index].mean(1)
        paired_fields['member_mse_ratio'] = {'mean': ratio.tolist(), 'interval_95': np.quantile(boots,[.025,.975],axis=0).tolist()}
        paired_core = {key: paired_summary([r[key] for r in scores['SOURCE']], [r[key] for r in scores['candidate']], index, normalizers[key])
                       for key in ('state','scale','shape')}
        paired_shape = {key: paired_summary([r[key] for r in scores['SOURCE']], [r[key] for r in scores['candidate']], index, normalizers['shape'])
                        for key in shape_keys}
        decomposition = {'schema':'car_scale_cached_decomposition_v1', 'status':'completed', 'split':spec['split'],
            'sample_ids':panel['sample_ids'], 'sample_count':count, 'members':4, 'field_order':source['field_order'],
            'support':'Q4096 separately reconstructed, fixed T256, J4 independent member streams',
            'prefit_coefficients':prefit, 'records':records,
            'source_receipt':ref(selected_source_path), 'child_receipt':ref(selected_child_path),
            'spec':ref(args.spec), 'source_baseline':ref(baseline_path)}
        scale_result = summarize_scale(decomposition,'candidate',float(normalizers['scale']))
        write(output/'decomposition.json', decomposition)
        write(output/'scale_summary.json', scale_result)
        result = {'schema':'car_round7_matched_assessment_v1', 'status':'completed', 'split':spec['split'],
            'sample_ids':panel['sample_ids'], 'members':4, 'field_order':source['field_order'],
            'field_units':spec['field_units'], 'physical_mask':spec['physical_mask'],
            'selection_use':'reused development validation' if spec['split']=='validation' else 'training diagnostic',
            'native_allowance_physical':allowance, 'source_fields':source_means, 'candidate_fields':child_means,
            'native_safeguards':fidelity, 'paired_fields':paired_fields, 'per_condition_fields':fields,
            'paired_core':paired_core, 'paired_shape_components':paired_shape, 'per_condition_scores':scores,
            'shape_components':'weighted self/mutual and finite/essential attraction and positive separation; score is attraction minus separation',
            'deferred_tasks':'not computed before sealed decision', 'spec':ref(args.spec),
            'source_native':ref(source_path), 'child_native':ref(native_path), 'source_baseline':ref(baseline_path),
            'source_selected':ref(selected_source_path), 'child_selected':ref(selected_child_path),
            'timing_seconds':timings, 'threads':args.threads, 'test_locked':True}
        write(output/'assessment.json', result)
        status.update(status='completed', conditions=count, timing_seconds=timings, threads=args.threads)
    except BaseException as error:
        status.update(error_type=type(error).__name__, error=str(error)); raise
    finally:
        if dataset is not None: dataset.close()
        status.update(wall_seconds=time.time()-start, cpu_seconds=time.process_time()-cpu)
        write(output/'receipt.json', status)


def tasks(args, spec, panel, source):
    import numpy as np
    import torch
    from phycoflow_reconstruction.evaluation.round2_assessment import scalar_task_score
    from phycoflow_reconstruction.evaluation.round2_task import regional_covariance
    from scripts.analysis.car_round7_paired_summary import paired_summary
    from scripts.posttraining.assess_car_round2_checkpoints import matched_rows, score_event
    from scripts.posttraining.freeze_car_event import FrozenRedesignedEvent
    source_path, source_receipt = source_native_bank(spec, panel, source)
    child_path, child_receipt = verify_bank(args.child_native, spec, panel, source, 'native')
    decision_path = args.decision.resolve(strict=True)
    decision = load(decision_path)
    if (decision.get('schema') != 'car_round7_task_release_v1' or decision.get('status') != 'sealed' or
        decision.get('child_checkpoint_sha256') != child_receipt['checkpoint_sha256'] or
        decision.get('child_native_receipt_sha256') != digest(child_path) or
        decision.get('split') != spec['split'] or not isinstance(decision.get('reason'), str) or not decision['reason'].strip()):
        raise ValueError('sealed checkpoint- and bank-bound endpoint-task release required')
    event = FrozenRedesignedEvent(load(checked(spec['event'])))
    regional = load(checked(spec['regional_task']))
    output = assessment_output(spec, args.output); output.mkdir(parents=True, exist_ok=False)
    start, cpu = time.time(), time.process_time()
    status = {'schema':'car_round7_task_job_v1', 'status':'failed', 'test_locked':True}
    try:
        torch.set_num_threads(args.threads)
        rows = {'SOURCE':[], 'candidate':[]}
        for i, sid in enumerate(panel['sample_ids']):
            source_row = read_row(source_path, source_receipt, i)
            child_row = read_row(child_path, child_receipt, i)
            matched_rows(source_row, child_row)
            scale, offset = source_row['normalizer_scale'], source_row['normalizer_offset']
            truth = source_row['truth_model'] * scale + offset
            mass = source_row['native_masses'].double()
            for name, row in (('SOURCE',source_row), ('candidate',child_row)):
                physical = row['members_model'] * scale + offset
                event_score, area, target = score_event(event, physical, truth, mass)
                regional_members = torch.stack([regional_covariance(member, mass, regional, tuple(source['field_order'])) for member in physical])
                regional_truth = regional_covariance(truth, mass, regional, tuple(source['field_order']))
                rows[name].append({'sample_id':sid, 'event_score':float(event_score.mean()),
                    'event_member_area':area.reshape(-1).tolist(), 'event_target_area':float(target.mean()),
                    'regional_score':float(scalar_task_score(regional_members, regional_truth, model_family='ffm').mean()),
                    'regional_members':regional_members.reshape(-1).tolist(), 'regional_truth':float(regional_truth.mean())})
        n = len(panel['sample_ids']); idx = np.random.default_rng(6042026).integers(0,n,size=(2000,n))
        units = spec['task_normalizers']
        if any(not math.isfinite(units[k]) or units[k] <= 0 for k in ('event','regional')):
            raise ValueError('positive frozen task normalizers required')
        summary = {key:paired_summary([r[key+'_score'] for r in rows['SOURCE']],
                                      [r[key+'_score'] for r in rows['candidate']],idx,units[key])
                   for key in ('event','regional')}
        write(output/'tasks.json', {'schema':'car_round7_deferred_tasks_v1','status':'completed','split':spec['split'],
            'sample_ids':panel['sample_ids'], 'members':4, 'rows':rows, 'paired':summary,
            'decision':ref(decision_path), 'spec':ref(args.spec), 'event':spec['event'],
            'regional_task':spec['regional_task'], 'source_native':ref(source_path), 'child_native':ref(child_path),
            'interpretation':'Lower event and regional scores are better; paired condition bootstrap on reused development panel.',
            'test_locked':True})
        status.update(status='completed',conditions=n,threads=args.threads)
    except BaseException as error:
        status.update(error_type=type(error).__name__, error=str(error)); raise
    finally:
        status.update(wall_seconds=time.time()-start,cpu_seconds=time.process_time()-cpu)
        write(output/'receipt.json',status)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('spec-template', help='print the required JSON fields; fill all references from the frozen campaign')
    f = sub.add_parser('forecast', help='scale measured bank/scoring receipts to a declared remaining workload')
    f.add_argument('--scoring-receipt',type=Path,required=True); f.add_argument('--prediction-receipt',type=Path,required=True)
    f.add_argument('--conditions',type=int,required=True); f.add_argument('--remaining-pairs',type=int,required=True)
    f.add_argument('--remaining-banks',type=int,required=True); f.add_argument('--training-seconds',type=float,required=True)
    b = sub.add_parser('baseline'); b.add_argument('--spec',type=Path,required=True); b.add_argument('--source-selected',type=Path,required=True)
    b.add_argument('--threads',type=int,default=2); b.add_argument('--output',type=Path,required=True)
    g = sub.add_parser('generate'); g.add_argument('--spec',type=Path,required=True); g.add_argument('--output',type=Path,required=True)
    g.add_argument('--support',choices=('native','selected'),required=True); g.add_argument('--device',required=True)
    s = sub.add_parser('score'); s.add_argument('--spec',type=Path,required=True); s.add_argument('--child-native',type=Path,required=True)
    s.add_argument('--source-selected',type=Path,required=True); s.add_argument('--child-selected',type=Path,required=True); s.add_argument('--baseline',type=Path,required=True)
    s.add_argument('--threads',type=int,default=2); s.add_argument('--output',type=Path,required=True)
    e = sub.add_parser('tasks'); e.add_argument('--spec',type=Path,required=True); e.add_argument('--child-native',type=Path,required=True)
    e.add_argument('--decision',type=Path,required=True); e.add_argument('--threads',type=int,default=2); e.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    if a.command == 'spec-template':
        reference = {'path':'REPLACE_WITH_ABSOLUTE_EXISTING_FILE','sha256':'REPLACE_WITH_SHA256'}
        print(json.dumps({'schema':'car_round7_assessment_spec_v1','split':'train OR validation',
            'campaign_root':'EXISTING_DATA2_CAMPAIGN_DIRECTORY',
            **{name:dict(reference) for name in ('panel','source_native_receipt','run_config','core','core_config','core_manifest',
                  'expanded_bank','prefit','assessment_contract','source_checkpoint','event','regional_task')},
            'checkpoint':None, 'native_allowance_model':['ROUND7_TRAIN_LAW_ORDERED_FIELD_ALLOWANCES'],
            'field_units':['ORDERED_FIELD_UNITS'], 'physical_mask':'DECLARE_NATIVE_PHYSICAL_MASS_AND_MASK',
            'task_normalizers':{'event':'FROZEN_EVENT_UNIT','regional':'FROZEN_REGIONAL_UNIT'}},indent=2))
        return
    if a.command == 'forecast':
        score_receipt, prediction_receipt = load(a.scoring_receipt), load(a.prediction_receipt)
        if (score_receipt.get('status') != 'completed' or prediction_receipt.get('status') != 'completed' or
            a.conditions < 1 or a.remaining_pairs < 0 or a.remaining_banks < 0 or a.training_seconds <= 0):
            raise ValueError('completed timed receipts and nonnegative declared workload required')
        n_score = score_receipt['conditions']; n_prediction = len(prediction_receipt['sample_ids'])
        if n_score < 1 or n_prediction < 1: raise ValueError('receipts must have sample counts')
        cpu_wall = score_receipt['wall_seconds'] / n_score * a.conditions * a.remaining_pairs
        cpu_work = score_receipt['cpu_seconds'] / n_score * a.conditions * a.remaining_pairs
        gpu_wall = prediction_receipt['wall_seconds'] / n_prediction * a.conditions * a.remaining_banks
        print(json.dumps({'scoring_receipt':ref(a.scoring_receipt),'prediction_receipt':ref(a.prediction_receipt),
            'assumed_conditions_per_bank':a.conditions,'remaining_scoring_pairs':a.remaining_pairs,
            'remaining_prediction_banks':a.remaining_banks,'estimated_scoring_wall_seconds':cpu_wall,
            'estimated_scoring_cpu_seconds':cpu_work,'estimated_prediction_occupied_seconds':gpu_wall,
            'serial_cpu_tail_seconds':cpu_wall,'scoring_wall_over_training':cpu_wall/a.training_seconds,
            'assumption':'Linear per-condition scaling from the referenced device, support and scorer; serial CPU tail is a conservative no-overlap estimate. Include separate I/O and delivery overhead from the recorded timing receipts.'},indent=2))
        return
    spec = load(a.spec); panel, source = validate_spec(spec)
    if getattr(a,'threads',1) < 1: raise ValueError('positive bounded thread count required')
    {'generate':generate,'baseline':baseline,'score':score,'tasks':tasks}[a.command](a,spec,panel,source)


if __name__ == '__main__': main()
