"""Fit frozen Round 8 artifacts from a replayable TRAIN/SOURCE cache stream.

No model is called here. Callers provide already-computed common observer and
semantic-block SOURCE scores; these must be produced using the same cache and
fitted observers, with their identity bound in provenance.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
import torch
from .fidelity_budget import FidelityBudget
from .round8_calibration import BLOCKS
from .round8_policy import physical_risk
from .round8_query_runtime import fit_pooled_shape
from ..coherence.round8_state import fit_source_hardness, select_directions

VERSION = 'round8_train_artifacts_v1'


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def fit_artifacts(*, cache_factory, state_family, shape_family, candidates, field_order,
                  manifest_sha256, source_checkpoint_sha256, normalizer_sha256, score_provenance,
                  block_observations=None, common_scores=None, score_observer=None, score_floors, field_scale,
                  shape_sample_ids, query_batches, probe_seed=800047,
                  risk_floor=1e-8, c0=None, family_weights=None, transform_floor=1e-6):
    """Fit from a factory returning fresh iterators of one-condition cache records.

    Record keys: id, split, predictions_model [J,N,C], target_model [N,C],
    physical_weights [N] (selected native normalized masses, *not* HT weights),
    inclusion_probability, coordinates [N,D]. IDs are unique and frozen.
    The score records are precomputed on precisely this TRAIN panel and contain
    all eight semantic keys (None only for structural absence), while common
    scores contain scalar SOURCE scale/shape/state per condition. Their producer
    and observer identities are pinned in score_provenance.

    The bounded shape sample must share the fixed Q spatial geometry and use
    fixed query batches. Its raster transform comes from TRAIN target values in
    the family-declared units. Probe block values are already multiplied by
    retained physical priorities; priorities are metadata and must not be
    multiplied a second time by the calibration coordinator.
    """
    fields = tuple(field_order)
    if not fields or len(set(fields)) != len(fields) or not _digest(manifest_sha256) or not _digest(source_checkpoint_sha256) or not _digest(normalizer_sha256):
        raise ValueError('unique fields and SHA256 TRAIN/SOURCE identities required')
    if not isinstance(score_provenance, Mapping) or score_provenance.get('split') != 'train' or score_provenance.get('manifest_sha256') != manifest_sha256 or score_provenance.get('source_checkpoint_sha256') != source_checkpoint_sha256 or not score_provenance.get('observer_identity'):
        raise ValueError('frozen TRAIN observer score provenance required')
    if tuple(state_family.field_names) != fields or tuple(shape_family.field_names) != fields or tuple(state_family.field_ids) != tuple(range(len(fields))):
        raise ValueError('State and Shape must use all fields in the frozen order')
    source_component = state_family.components_by_key['cross_joint_topk_swd'] if 'cross_joint_topk_swd' in state_family.components_by_key else None
    if len(fields) > 1 and (source_component is None or not torch.allclose(torch.as_tensor(candidates).cpu().double(), source_component.directions.detach().cpu().double())):
        raise ValueError('candidates must be the unchanged SOURCE State direction bank')
    sample_ids = tuple(str(x) for x in shape_sample_ids)
    if not sample_ids or len(set(sample_ids)) != len(sample_ids) or not query_batches:
        raise ValueError('nonempty unique bounded Shape IDs and query batches required')
    sample_set = set(sample_ids)
    ids, shape_rows, shapes, risk_sum, total = [], {}, None, None, 0
    mean_sum, second_sum = None, None
    expected_coords = None
    for record in cache_factory():
        ident = str(record['id'])
        if record.get('split') != 'train' or ident in ids:
            raise ValueError('unique TRAIN condition identities required')
        ids.append(ident)
        pred = torch.as_tensor(record['predictions_model']).double()
        target = torch.as_tensor(record['target_model']).double()
        weights = torch.as_tensor(record['physical_weights']).double()
        coords = torch.as_tensor(record['coordinates']).double()
        p = float(record['inclusion_probability'])
        if pred.ndim != 3 or pred.shape[0] != 2 or pred.shape[1:] != target.shape or target.shape[-1] != len(fields) or weights.shape != target.shape[:1] or coords.shape[0] != len(weights) or not 0 < p <= 1:
            raise ValueError('invalid frozen J2 TRAIN cache dimensions or inclusion probability')
        if not all(torch.isfinite(x).all() for x in (pred,target,weights,coords)) or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError('cache fields, weights and coordinates must be finite')
        if shapes is None: shapes = target.shape
        if target.shape != shapes: raise ValueError('TRAIN Q support must be fixed')
        risk = physical_risk(pred.permute(0,2,1)[None],target.T[None],weights,design='without_replacement',probability=p)
        risk_sum = risk if risk_sum is None else risk_sum + risk
        # Fixed physical field scales in the Shape family's declared units.
        _, y = shape_family._units_and_fields(pred[0:1],target[None])
        selected = y[0].double()
        mass = weights / p
        first = (selected * mass[:,None]).sum(0)
        second = (selected.square() * mass[:,None]).sum(0)
        mean_sum = first if mean_sum is None else mean_sum + first
        second_sum = second if second_sum is None else second_sum + second
        if ident in sample_set:
            if expected_coords is None: expected_coords = coords
            elif not torch.equal(coords,expected_coords): raise ValueError('Shape fit requires a common frozen coordinate geometry')
            shape_rows[ident] = (pred[0],target)
        total += 1
    if not total or set(shape_rows) != sample_set:
        raise ValueError('TRAIN stream missing selected Shape observations')
    means = mean_sum / total
    variance = (second_sum / total - means.square()).clamp_min(0)
    scales = variance.sqrt().clamp_min(transform_floor)

    def hardness_stream():
        for record in cache_factory():
            pred = torch.as_tensor(record['predictions_model']).double()
            target = torch.as_tensor(record['target_model']).double()
            converted = [state_family._in_declared_units(x[None],target[None])[0][0] for x in pred]
            _, truth = state_family._in_declared_units(pred[0:1],target[None])
            weights = torch.as_tensor(record['physical_weights']).double()[None]
            yield torch.stack(converted)[None], truth, weights, weights
    if len(fields) == 1:
        hardness = torch.zeros(len(candidates),dtype=torch.float64)
    else:
        hardness = fit_source_hardness(hardness_stream(), candidates)
    bank = select_directions(candidates,hardness,fields,seed=probe_seed)
    provenance = {'split':'train','manifest_sha256':manifest_sha256,'source_checkpoint_sha256':source_checkpoint_sha256,'normalizer_sha256':normalizer_sha256,'field_order':fields,'sample_ids':tuple(ids),'shape_sample_ids':sample_ids,'spatial_points':shapes[0], 'score_observer_identity':score_provenance['observer_identity']}
    shape_provenance = {'split':'train','source_checkpoint_sha256':source_checkpoint_sha256,'sample_ids':sample_ids,'manifest_sha256':manifest_sha256}
    shape_artifact = fit_pooled_shape(shape_family,torch.stack([shape_rows[i][0] for i in sample_ids]),torch.stack([shape_rows[i][1] for i in sample_ids]),expected_coords[None],query_batches,field_mean=means,field_scale=scales,provenance=shape_provenance)
    keys = tuple(f'{f}.{b}' for f, blocks in BLOCKS.items() for b in blocks)
    if score_observer is not None:
        if block_observations is not None or common_scores is not None:
            raise ValueError('provide a score observer or precomputed scores, not both')
        block_observations = {k: [] for k in keys}
        common_scores = {k: [] for k in ('scale','shape','state')}
        scored_ids = []
        for record in cache_factory():
            scored_ids.append(str(record['id']))
            blocks, common = score_observer(record, bank, shape_artifact)
            if set(blocks) != set(keys) or set(common) != set(common_scores):
                raise ValueError('score observer must return all semantic and common scores')
            for k in keys:
                block_observations[k].append(blocks[k])
            for k in common_scores:
                common_scores[k].append(common[k])
        if scored_ids != ids:
            raise ValueError('cache stream identities changed during scoring')
        for k, values in block_observations.items():
            if all(v is None for v in values): block_observations[k] = None
            elif any(v is None for v in values): raise ValueError('inconsistent structural block activity')
    if block_observations is None or common_scores is None:
        raise ValueError('TRAIN score observer or precomputed score records are required')
    if set(block_observations) != set(keys): raise ValueError('all eight SOURCE block observations required')
    priorities = {
        'scale.signed':1., 'scale.joint':float(score_provenance['retained_priorities']['scale.joint']),
        'scale.log_power_self':float(score_provenance['retained_priorities']['scale.log_power_self']),
        'scale.rho':float(score_provenance['retained_priorities']['scale.rho']),
        'state.marginal':float(state_family.component_weights['self_marginal_w2']),
        'state.mixed':float(state_family.component_weights.get('cross_joint_topk_swd',0.)),
        'shape.self':float(shape_family.spatial_objective.weights.get('self.persistence',0.)),
        'shape.mutual':float(shape_family.spatial_objective.weights.get('mutual.persistence',0.)),
    }
    units, active = {}, {}
    for k in keys:
        values = block_observations[k]
        active[k] = values is not None
        if not active[k]:
            if priorities[k] != 0 and not (len(fields) == 1 and k in ('scale.rho','state.mixed','shape.mutual')): raise ValueError('unexpected inactive physical block')
            priorities[k] = 1.; units[k] = 1.; continue
        if priorities[k] <= 0: raise ValueError('active block requires positive retained physical priority')
        v = torch.as_tensor(values,dtype=torch.float64).flatten()
        if len(v) != total or not torch.isfinite(v).all(): raise ValueError('TRAIN block observations must match every condition')
        units[k] = max(float(v.abs().mean()),transform_floor)
    if set(common_scores) != {'scale','shape','state'} or set(score_floors) != set(common_scores): raise ValueError('three common SOURCE scores and frozen floors required')
    denominators = {}
    for k in common_scores:
        v = torch.as_tensor(common_scores[k],dtype=torch.float64).flatten()
        floor = float(score_floors[k])
        if len(v) != total or not torch.isfinite(v).all() or not math.isfinite(floor) or floor <= 0: raise ValueError('common scores must match TRAIN conditions with positive floors')
        denominators[k] = max(abs(float(v.mean())),floor)
    field_scale = torch.as_tensor(field_scale,dtype=torch.float64)
    if field_scale.shape != (len(fields),) or not torch.isfinite(field_scale).all() or (field_scale <= 0).any(): raise ValueError('positive frozen field scales required')
    if c0 is None:
        if family_weights is None or set(family_weights) != {'scale','shape','state'}:
            raise ValueError('C0 requires declared common family weights')
        c0 = sum(float(family_weights[k]) * float(torch.as_tensor(common_scores[k], dtype=torch.float64).mean()) for k in family_weights)
        if not math.isfinite(c0) or c0 < 0:
            raise ValueError('SOURCE weighted common objective C0 must be nonnegative')
    budget = FidelityBudget.from_reference(fields, risk_sum/total, c0, relative_increase=.10, absolute_allowance=0., denominator_floor=risk_floor)
    return {'version':VERSION,'state_bank':bank,'shape_artifact':shape_artifact,'priorities':priorities,'units':units,'active':active,'probe_seed':int(probe_seed),'provenance':provenance,'fidelity_reference':budget.state_dict(),'source_risk_model':risk_sum/total,'source_risk_physical':risk_sum/total * field_scale.square(),'risk_measure':{'version':'round8_member_ht_without_replacement_v1','member_count':2,'condition_count':total,'spatial_points':shapes[0],'support':'fixed TRAIN spatial proxy','normalizer_sha256':normalizer_sha256},'field_scale':field_scale,'common_score_denominators':denominators,'score_floors':dict(score_floors),'score_provenance':dict(score_provenance)}


def source_panel_cache_factory(panel_path, manifest_path):
    """Verify and stream the concrete round8_inference SOURCE TRAIN NPZ panel.

    Returns (fresh iterator factory, validated panel metadata). Every iterator
    verifies each immutable shard before reading; no full panel is loaded.
    """
    import json
    import numpy as np
    from pathlib import Path
    from ..evaluation.round8_campaign import file_hash
    panel_path, manifest_path = Path(panel_path), Path(manifest_path)
    panel = json.loads(panel_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if panel.get('schema') != 'round8_endpoint_panel_v1' or panel.get('status') != 'completed' or panel.get('population') != 'TRAIN' or panel.get('endpoint') != 'SOURCE':
        raise ValueError('a completed SOURCE TRAIN panel is required')
    if panel.get('manifest_sha256') != file_hash(manifest_path) or panel.get('checkpoint_sha256') != panel.get('source_checkpoint_sha256'):
        raise ValueError('panel manifest or SOURCE checkpoint identity mismatch')
    expected = [str(x['id']) for x in manifest['TRAIN']]
    rows = panel['conditions']
    if len(expected) != len(set(expected)) or [str(x['sample_id']) for x in rows] != expected:
        raise ValueError('panel condition identities differ from frozen TRAIN manifest')
    fields = tuple(panel['field_names'])
    p = float(panel['inclusion_probability'])
    def iterate():
        for row in rows:
            path = panel_path.parent / row['path']
            if file_hash(path) != row['sha256']:
                raise ValueError('SOURCE panel cache shard changed')
            with np.load(path,allow_pickle=False) as data:
                if str(data['sample_id']) != str(row['sample_id']) or tuple(data['field_names'].tolist()) != fields or not np.isclose(float(data['inclusion_probability']),p) or not data['physical_mask'].all():
                    raise ValueError('SOURCE shard identities, support, or field order differ')
                yield {'id':str(row['sample_id']), 'split':'train', 'predictions_model':data['draws_model'].copy(), 'target_model':data['target_model'].copy(), 'physical_weights':data['physical_weights'].copy(), 'inclusion_probability':p, 'coordinates':data['model_coordinates'].copy(), 'native_ids':data['native_ids'].copy(), 'dataset_index':int(row['dataset_index'])}
    return iterate, {'panel_sha256':file_hash(panel_path),'manifest_sha256':panel['manifest_sha256'],'source_checkpoint_sha256':panel['source_checkpoint_sha256'],'normalizer_sha256':panel['normalizer_sha256'],'field_order':fields,'sample_ids':tuple(expected),'spatial_inclusion_probability':p}


def prepare_from_panel(config_path, manifest_path, panel_path, *, shape_samples=16, shape_queries=4,
                       probe_seed=800047, score_floor=1e-8):
    """Build frozen R7 observers and derive all artifacts directly from TRAIN cache.

    The source panel is already generated; this function does no model inference
    or optimization. All heavy common-observer scores are per condition, streamed.
    """
    from ..config import load_config
    from ..data.factory import open_field_dataset
    from ..coherence import build_enabled_families
    from ..coherence.round8_queries import SharedQuerySampler
    from ..coherence.round8_state import score_family as state_score
    from ..evaluation.round8_campaign import file_hash
    from .coherence_sampling import _fit_training_references
    from .round8_query_runtime import Round8QueryAdapter
    from .post_training import _configure_persistence_workers
    config = load_config(config_path)
    _configure_persistence_workers(config)
    factory, identity = source_panel_cache_factory(panel_path,manifest_path)
    fields = identity['field_order']
    if shape_samples < 1 or shape_queries < 1 or score_floor <= 0:
        raise ValueError('positive bounded Shape sample, query count and score floor required')
    dataset = open_field_dataset(config['dataset'],split='train')
    try:
        if tuple(dataset.field_names) != fields or dataset.normalizer.digest() != identity['normalizer_sha256']:
            raise ValueError('TRAIN dataset field order or normalizer mismatch')
        families = build_enabled_families(config['coherence'],dataset.data_spec,dataset.normalizer)
        required = {'global_distribution','multiscale_spectrum','topology'}
        if set(families) != required or not required.issubset(config['coherence'].get('frozen_family_artifacts',{})):
            raise ValueError('three frozen R7 families are required')
        _fit_training_references(families,dataset,config,device=torch.device('cpu'),structured_grid_required=False)
        state, scale, shape = (families[k] for k in ('global_distribution','multiscale_spectrum','topology'))
        # Common complete Scale scores do not inherit a bounded coverage bank.
        if hasattr(scale,'coverage_pairs'): delattr(scale,'coverage_pairs')
        candidates = state.components_by_key['cross_joint_topk_swd'].directions if len(fields)>1 else torch.empty(0,1)
        sample_ids = identity['sample_ids'][:min(shape_samples,len(identity['sample_ids']))]
        query_sampler = SharedQuerySampler(fields,seed=probe_seed)
        query_batches = []
        for _ in range(shape_queries):
            query_batches.append(query_sampler.draw()); query_sampler.finish()
        scoring_sampler = SharedQuerySampler(fields,seed=probe_seed+1)
        adapter = None
        def scorer(record,bank,shape_artifact):
            nonlocal adapter
            if adapter is None: adapter = Round8QueryAdapter(families,shape_artifact)
            members = torch.as_tensor(record['predictions_model']).float()
            ref = torch.as_tensor(record['target_model']).float()[None]
            coords = torch.as_tensor(record['coordinates']).float()[None]
            mass = torch.as_tensor(record['physical_weights']).float()[None]
            # Occupation measures are normalized by the original family; HT
            # applies only to target risk, not to this empirical distribution.
            context={'generated_weights':mass,'reference_weights':mass,'conditional_draws':(members[1:2],)}
            with torch.no_grad():
                common = {key: float(families[name](members[0:1],ref,coordinates=coords,context=context).scalar_loss) for key,name in [('state','global_distribution'),('scale','multiscale_spectrum'),('shape','topology')]}
                transformed = [state._in_declared_units(x[None],ref)[0] for x in members]
                _, truth = state._in_declared_units(members[0:1],ref)
                ids = list(state.field_ids)
                _, blocks = state_score(torch.stack([x[...,ids] for x in transformed]),truth[...,ids],mass,mass,bank,channel_weights=state.components_by_key['self_marginal_w2'].channel_weights,marginal_weight=state.component_weights['self_marginal_w2'],mixed_weight=state.component_weights.get('cross_joint_topk_swd',0.))
                batch = scoring_sampler.draw()
                for name in ('multiscale_spectrum','topology'):
                    _, extra, _ = adapter.score(name,members[0:1],ref,coordinates=coords,context=context,batch=batch,coefficients={})
                    blocks.update(extra)
                scoring_sampler.finish()
            for key in (f'{f}.{b}' for f,bs in BLOCKS.items() for b in bs):
                blocks.setdefault(key,None)
            return {k: None if v is None else float(v) for k,v in blocks.items()}, common
        retained = {'scale.joint':scale.settings.joint_energy_score_weight,'scale.log_power_self':scale.relation_extension['self_weight'],'scale.rho':scale.relation_extension['rho_weight']}
        observer_identity = {k:v['sha256'] for k,v in config['coherence']['frozen_family_artifacts'].items()}
        observer_identity.update({'config_sha256':file_hash(config_path),'source_panel_sha256':identity['panel_sha256'],'common_scale':'full_population','candidate_query_seed':probe_seed+1})
        score_provenance={'split':'train','manifest_sha256':identity['manifest_sha256'],'source_checkpoint_sha256':identity['source_checkpoint_sha256'],'observer_identity':observer_identity,'retained_priorities':retained}
        family_weights = {key:float(config['coherence']['families'][name].get('weight',1.)) for key,name in [('state','global_distribution'),('scale','multiscale_spectrum'),('shape','topology')]}
        artifacts = fit_artifacts(cache_factory=factory,state_family=state,shape_family=shape,candidates=candidates,field_order=fields,manifest_sha256=identity['manifest_sha256'],source_checkpoint_sha256=identity['source_checkpoint_sha256'],normalizer_sha256=identity['normalizer_sha256'],score_provenance=score_provenance,score_observer=scorer,score_floors=dict.fromkeys(('state','scale','shape'),score_floor),field_scale=dataset.normalizer.scale,shape_sample_ids=sample_ids,query_batches=query_batches,probe_seed=probe_seed,family_weights=family_weights)
        probe_records = []
        for record in factory():
            probe_records.append(record)
            if len(probe_records) == 32: break
        if len(probe_records) < 32:
            raise ValueError('frozen TRAIN panel needs at least 32 conditions for calibration probe')
        artifacts['provenance']['shape_query_seed'] = int(probe_seed)
        artifacts['provenance']['shape_query_batches'] = int(shape_queries)
        artifacts['provenance']['score_query_seed'] = int(probe_seed + 1)
        artifacts['provenance']['spatial_inclusion_probability'] = identity['spatial_inclusion_probability']
        artifacts['probe_dataset_indices'] = [r['dataset_index'] for r in probe_records]
        artifacts['probe_sample_ids'] = [r['id'] for r in probe_records]
        artifacts['probe_query_indices'] = torch.stack([torch.as_tensor(r['native_ids'],dtype=torch.int64) for r in probe_records])
        if not torch.all(artifacts['probe_query_indices'] == artifacts['probe_query_indices'][0]):
            raise ValueError('calibration probe must use the shared frozen spatial support')
        return artifacts
    finally:
        dataset.close()
