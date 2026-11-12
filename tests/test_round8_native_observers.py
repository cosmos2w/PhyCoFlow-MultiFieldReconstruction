"""CPU numerical qualification against the retained physical native operator."""
from copy import deepcopy
from pathlib import Path
import math

import pytest
import torch
import yaml


def test_native_bridge_full_support_and_direct_score():
    from phycoflow_reconstruction.data.factory import open_field_dataset
    from phycoflow_reconstruction.coherence.compose import build_enabled_families
    from phycoflow_reconstruction.training.coherence_sampling import _fit_training_references, _family_context
    from phycoflow_reconstruction.evaluation.round8_native_observers import native_common_families
    from phycoflow_reconstruction.coherence.families.multiscale_spectrum.losses import (
        selected_band_coherence, signed_band_energy_score, vector_energy_score)
    from phycoflow_reconstruction.coherence.families.multiscale_relation import ANCHOR_PATH, SELF_PATH, RHO_PATH

    root = Path(__file__).resolve().parents[1]
    config_path = root / 'cases/turbulent_combustion/runs/coherence_compute_round6/configs/U6.yaml'
    fair_path = root / 'cases/turbulent_combustion/runs/coherence_compute_round7/artifacts/conditional_fair_topology.pt'
    bundle_path = root / 'UpdatePlans/CoherenceComputeEnhancement_20261106/Round_2/artifacts/native_metric_bundle.pt'
    if not all(p.exists() for p in (config_path, fair_path, bundle_path)):
        pytest.skip('case-owned frozen inputs unavailable')
    config = yaml.safe_load(config_path.read_text())
    if not Path(config['dataset']['path']).exists():
        pytest.skip('case dataset unavailable')
    train = open_field_dataset(config['dataset'], split='train')
    try:
        families = build_enabled_families(config['coherence'], train.data_spec, train.normalizer)
        _fit_training_references(families, train, config, device=torch.device('cpu'), structured_grid_required=False)
        settings = deepcopy(config['coherence'])
        settings['families']['topology']['persistence']['definition'] = 'conditional_fixed_transform_fair_v1'
        fair = build_enabled_families(settings, train.data_spec, train.normalizer)['topology']
        fair.load_state_artifact(torch.load(fair_path, map_location='cpu', weights_only=True))
        native, identity = native_common_families(config, train, families, fair, bundle_path)
        assert identity['shape_mapping']['query_points'] == 40300
        assert identity['shape_mapping']['sparse_interpolation'] == 0
        family = native['multiscale_spectrum']
        assert family.basis.shape == (40300, 86)
        sample = train[0]
        target = sample.values[None]
        generator = torch.Generator().manual_seed(812)
        x = target + .03 * torch.randn(target.shape, generator=generator)
        z = target + .04 * torch.randn(target.shape, generator=generator)
        coordinates = sample.coordinates[None]
        with torch.no_grad():
            result = family(x, target, coordinates=coordinates, context={'conditional_draws': (z,)})
            stats = [family._statistics(v) for v in (target, x, z)]
            descriptor = [family._descriptors(v) for v in stats]
            den = family.relation_self_scale * math.sqrt(family.relation_self_scale.numel())
            self_score = vector_energy_score(torch.stack([descriptor[1]['log_band_power']/den,
                descriptor[2]['log_band_power']/den]), descriptor[0]['log_band_power']/den, alpha=1)[0]
            rho = [selected_band_coherence(s, family.power_floor) for s in stats]
            rho_score = vector_energy_score(torch.stack([rho[1], rho[2]])[..., None], rho[0][..., None], alpha=1)[0].mean((1, 2))
            assert torch.allclose(result.component_results[SELF_PATH].per_sample_cost, self_score)
            assert torch.allclose(result.component_results[RHO_PATH].per_sample_cost, rho_score)
            # Reconstruct retained native modes explicitly and evaluate spatial
            # signed projections, independently of coefficient-space reduction.
            coeff = torch.stack([stats[1]['coefficients'], stats[2]['coefficients']])
            target_coeff = stats[0]['coefficients']
            draws = torch.einsum('nk,sk,jbkc->jbsnc', family.basis, family.filters, coeff)
            truth = torch.einsum('nk,sk,bkc->bsnc', family.basis, family.filters, target_coeff)
            signed = signed_band_energy_score(draws, truth, mass=family.mass, scales=family.band_scale,
                projections=family._projections(family.compute_dtype, torch.device('cpu')), alpha=1)[0]
            signed = (signed * family._band_weights(family.compute_dtype, torch.device('cpu'))).sum(1)
            mass = family.mass / family.mass.sum()
            def embed(field, means):
                weighted = field / family.band_scale[None, :, None, :] * mass.sqrt()[None, None, :, None]
                pieces = family.piece_measures.sum(1)
                pieces = pieces / pieces.sum()
                mean = means * (family.settings.mean_weight * pieces).sqrt()[None, :, None] / family.piece_mean_scale
                return torch.cat((weighted.flatten(1), mean.flatten(1)), dim=1)
            draws_embedded = torch.stack([embed(draws[0], stats[1]['piece_means']), embed(draws[1], stats[2]['piece_means'])])
            truth_embedded = embed(truth, stats[0]['piece_means'])
            joint = vector_energy_score(draws_embedded, truth_embedded, alpha=1)[0]
            assert torch.allclose(result.component_results['multiscale_spectrum.signed_band_energy_score.conditional'].per_sample_cost, signed, rtol=1e-6, atol=1e-8)
            assert torch.allclose(result.component_results['multiscale_spectrum.joint_spectral_mean_energy_score.conditional'].per_sample_cost, joint, rtol=1e-6, atol=1e-8)
            direct_anchor = signed + family.settings.joint_energy_score_weight * joint
            assert torch.allclose(result.component_results[ANCHOR_PATH].per_sample_cost, direct_anchor, rtol=1e-6, atol=1e-8)
            weights = train.physical_weights[None]
            mask = torch.ones_like(weights, dtype=torch.bool)
            base = {'sample_ids': ('example',), 'reference_ids': ('example',),
                    'generated_weights': weights, 'reference_weights': weights,
                    'generated_mask': mask, 'reference_mask': mask,
                    'measure_metadata': {'native_measure_sha256': train.physical_measure_receipt['sha256']}}
            for name in ('global_distribution', 'topology'):
                observer = native[name]
                score = observer(x, target, coordinates=coordinates,
                                 context=_family_context(observer, base, [z], report_detail=True))
                assert torch.isfinite(score.scalar_loss)
    finally:
        train.close()
