"""Frozen descriptor calibration cannot perturb streams or be overwritten on resume."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from helpers.coherence import _base_config, _write_fixture

from phycoflow_reconstruction.contracts import ModelCapabilities
from phycoflow_reconstruction.data.factory import open_field_dataset
from phycoflow_reconstruction.training.coherence_sampling import (
    _conditional_draw_count,
    _conditional_draw_seed,
    _family_context,
    _fit_source_calibrations,
    _load_checkpoint_family_state,
)


def test_state_reuses_scale_draw_without_advancing_inference_rng():
    paired = SimpleNamespace(conditional_draws=1)
    state = SimpleNamespace(conditional_draws=2)
    scale = SimpleNamespace(conditional_draws=2)
    assert _conditional_draw_count({'state': paired, 'scale': scale}) == 2
    assert _conditional_draw_count({'state': state, 'scale': scale}) == 2
    primary = torch.Generator().manual_seed(43)
    primary_state = primary.get_state().clone()
    extra = torch.randn(4, generator=torch.Generator().manual_seed(_conditional_draw_seed(43, 1)))
    context = _family_context(state, {}, [extra])
    assert context['conditional_draws'][0] is extra
    assert torch.equal(primary.get_state(), primary_state)
    assert _family_context(paired, {}, [extra]) == {}


def test_conditional_state_frozen_checkpoint_is_strict_and_legacy_restore_unchanged():
    class Family(torch.nn.Module):
        family_name = 'global_distribution'; version = '1'
        def __init__(self, definition):
            super().__init__()
            self.score_definition = definition
            self.register_buffer('directions', torch.ones(2))
    for definition in ('marginal_mixed_weighted_v3', 'conditional_quantile_fair_v1'):
        family = Family(definition)
        state = deepcopy(family.state_dict())
        _load_checkpoint_family_state(family, state)
        state['directions'][0] = 2
        with pytest.raises(ValueError, match='frozen family artifact'):
            _load_checkpoint_family_state(family, state)
    legacy = Family('global_distribution_v1')
    _load_checkpoint_family_state(legacy, {'directions': torch.tensor([2., 1.])})
    assert legacy.directions[0] == 2


def test_source_descriptor_calibration_is_train_only_target_free_and_rng_isolated(tmp_path):
    path = tmp_path / 'data.h5'; _write_fixture(path)
    config = _base_config(path)
    config.update(coherence={'compute_budget':{'point_count':8,'query_seed':100045}},
                  rollout={'steps':2,'solver':'euler'},
                  observation_consistency={'mode':'none','final_clamp':False})
    class Model(torch.nn.Module):
        capabilities = ModelCapabilities('point', True, True, False, True)
        def differentiable_reconstruct(self, batch, *, steps, generator):
            assert batch.target_fields is None
            assert not torch.is_grad_enabled()
            return torch.randn((*batch.query_coords.shape[:2],2), generator=generator)
    class Family:
        needs_source_calibration = True
        def fit_calibration(self, pred, target, *, coordinates, provenance):
            assert pred.shape == target.shape == (2,8,2)
            assert coordinates.shape == (2,8,2)
            assert provenance['split'] == 'train'
            assert provenance['source_checkpoint_sha256'] == 'a'*64
            assert len(set(provenance['sample_ids'])) == 2
            torch.rand(3)  # Deliberately exercise the preserved global RNG scope.
            self.needs_source_calibration = False
    model, family = Model().train(), Family()
    dataset = open_field_dataset(config['dataset'],split='train')
    try:
        state = torch.get_rng_state().clone()
        _fit_source_calibrations({'topology':family}, model,dataset,config,
                                 device=torch.device('cpu'),source_checkpoint_sha256='a'*64)
        assert model.training and not family.needs_source_calibration
        assert torch.equal(state,torch.get_rng_state())
    finally: dataset.close()


def test_checkpoint_cannot_replace_frozen_new_family_scales_or_source():
    class Family(torch.nn.Module):
        family_name = 'topology'; version = '4'
        def __init__(self):
            super().__init__();self.register_buffer('scale',torch.ones(2))
            self.spatial_objective = SimpleNamespace(calibration_record={
                'provenance':{'source_checkpoint_sha256':'a'*64}})
    family = Family()
    state = deepcopy(family.state_dict())
    _load_checkpoint_family_state(family,state,source_checkpoint_sha256='a'*64)
    with pytest.raises(ValueError,match='SOURCE'):
        _load_checkpoint_family_state(family,state,source_checkpoint_sha256='b'*64)
    state['scale'][0] += 1
    with pytest.raises(ValueError,match='frozen family artifact'):
        _load_checkpoint_family_state(family,state,source_checkpoint_sha256='a'*64)


def test_saved_reconstruction_contract_retains_nonuniform_masked_support():
    from phycoflow_reconstruction.evaluation.checkpoint import physical_plot_contract
    sample = SimpleNamespace(values=torch.zeros(4,2), field_names=("u","v"),
        valid_points=torch.tensor([True,False,True,True]),
        metadata={"auxiliary":{"physical_weights":torch.tensor([1.,-1.,4.,2.])},
                  "physical_measure":"native_cell_area"})
    result=physical_plot_contract(sample,torch.tensor([0,1,2]))
    assert result["physical_mask"].tolist()==[[True,False,True]]
    assert result["physical_weights"].tolist()==[[1.,-1.,4.]]
    assert str(result["physical_measure"])=="native_cell_area"
    sample.metadata={}
    result=physical_plot_contract(sample,torch.tensor([0,2,3]))
    assert "physical_weights" not in result
    assert str(result["physical_measure"])=="equal_valid_nodes"
