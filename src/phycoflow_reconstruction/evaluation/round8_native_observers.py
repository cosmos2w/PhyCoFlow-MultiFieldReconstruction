"""Versioned native-support bridge for the frozen Round 8 common observers.

Uses historical native geometry and TRAIN calibrations as data, not as a claim
that an older implementation artifact has current-code identity. The native
and Q4096 observers remain distinct resolution-specific finite observables.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import torch

from .canonical_coherence import content_sha256
from .round8_campaign import file_hash
from ..coherence.registry import build_coherence_family
from ..coherence.families.multiscale_spectrum.features import descriptor_layout
from ..coherence.families.topology.geometry import build_physical_cell_overlap_map


def native_common_families(config, train, sparse_families, fair, native_bundle_path, *, device='cpu'):
    """Bind native FEM and exact native-to-existing-raster overlap, without refitting.

    The signed spectral anchor and its normalization come from the frozen
    native TRAIN bundle. The R7 relation extension retains its sparse TRAIN
    self-power units and fixed weights. Fair Shape keeps its exact R7 TRAIN
    descriptor transform and leaf scales, while evaluating an exact native
    physical cell overlap on the same raster. This mixed provenance is explicit.
    """
    path = Path(native_bundle_path)
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    expected = bundle.pop('sha256')
    if content_sha256(bundle) != expected:
        raise ValueError('historical native bundle content changed')
    bundle['sha256'] = expected
    from ..training.source import source_checkpoint_path
    if bundle['source_checkpoint_sha256'] != file_hash(source_checkpoint_path(config)):
        raise ValueError('native bundle SOURCE identity differs')
    if bundle['normalizer_digest'] != train.normalizer.digest():
        raise ValueError('native bundle normalizer differs')
    sample = train[0]
    if (tuple(bundle['data_spec']['field_names']) != tuple(train.field_names)
            or not torch.equal(bundle['coordinates'][0], sample.coordinates)
            or not torch.equal(bundle['raw_coordinates'], sample.coordinates_raw)
            or not torch.equal(bundle['native_ids'], torch.arange(sample.values.shape[0]))
            or not torch.all(bundle['native_mask'])):
        raise ValueError('native bundle field order or physical support differs')
    expected_weights = train.physical_weights.double()
    expected_weights = expected_weights / expected_weights.sum()
    native_weights = bundle['native_weights'].double()
    native_weights = native_weights / native_weights.sum()
    if not torch.allclose(native_weights, expected_weights, rtol=1e-12, atol=1e-15):
        raise ValueError('native bundle physical measure differs')
    sparse_scale = sparse_families['multiscale_spectrum']
    native_anchor_config = deepcopy(config['coherence']['families']['multiscale_spectrum'])
    extension = native_anchor_config.pop('relation_extension')
    if native_anchor_config != bundle['configs']['multiscale_spectrum']:
        raise ValueError('native anchor mathematical config differs from common Scale')
    family = build_coherence_family('multiscale_spectrum',
        config['coherence']['families']['multiscale_spectrum'], train.data_spec, train.normalizer).to(device)
    reference = bundle['states']['multiscale_spectrum']
    old_state = reference['state_dict']
    # Transfer immutable numerical operator and calibration as inputs to a new
    # observer, without replacing or relabeling the old implementation hash.
    for key, value in old_state.items():
        if not hasattr(family, key):
            raise ValueError(f'unknown native Scale buffer {key}')
        setattr(family, key, value.to(device))
    scale = sparse_scale.relation_self_scale.detach().clone().to(device)
    if scale.numel() != len(family.settings.bands.names) * len(family.field_names) or not torch.all(scale > 0):
        raise ValueError('invalid TRAIN relation self units')
    family.relation_self_scale = scale
    family.geometry_record = deepcopy(reference['geometry'])
    family.calibration_record = deepcopy(reference['calibration'])
    family.calibration_record['relation_extension'] = {
        'definition': extension['definition'], 'source': 'frozen_round6_TRAIN_Q4096_units',
        'source_artifact_sha256': config['coherence']['frozen_family_artifacts']['multiscale_spectrum']['sha256'],
        'scale': scale.cpu().tolist()}
    family.layout = descriptor_layout(family.settings, tuple(family.geometry_record['region_names']))
    if not family.fitted or family.basis.shape[0] != sample.values.shape[0]:
        raise ValueError('native Scale binding incomplete')
    if fair.raster_mapping != 'physical_cell_overlap_v1' or fair.native_masked:
        raise ValueError('fair Shape must use the frozen physical overlap raster')
    mapping = build_physical_cell_overlap_map(sample.coordinates_raw, torch.arange(len(sample.values)),
        grid_shape=fair.grid_shape, query_coordinates=sample.coordinates, axes=fair.axes,
        neighbors=fair.raster_neighbors, power=fair.raster_power)
    if mapping.diagnostics['sparse_interpolation'] != 0:
        raise ValueError('native Shape must use all physical cells')
    fair.neighbor_indices = mapping.neighbor_indices.to(device)
    fair.neighbor_weights = mapping.neighbor_weights.to(device)
    fair.grid_coordinates = mapping.grid_coordinates.to(device)
    fair.geometry_sha256 = mapping.coordinate_sha256
    fair.geometry_diagnostics = dict(mapping.diagnostics)
    identity = {'schema': 'round8_common_native_geometry_v1',
                'native_bundle_sha256': file_hash(path), 'native_bundle_content_sha256': expected,
                'native_scale_operator_coordinate_sha256': reference['geometry']['coordinate_sha256'],
                'native_scale_reference': 'Round2_TRAIN_native_v4',
                'relation_self_units': 'Round6_TRAIN_Q4096',
                'shape_mapping': mapping.diagnostics,
                'shape_units': 'Round7_TRAIN_Q4096_fixed_transform_and_leaf_scales',
                'scope': 'native support observer; not numerical identity with Q4096 observer'}
    return {'global_distribution': sparse_families['global_distribution'],
            'multiscale_spectrum': family, 'topology': fair}, identity
