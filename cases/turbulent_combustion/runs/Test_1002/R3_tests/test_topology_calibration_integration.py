"""CPU integration of fixed TRAIN panel, ordering, and frozen artifact identities.

Rollout and dataset I/O are mocked; actual GUDHI distances and model gradients
remain live. This validates plumbing, not measured scientific performance.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from phycoflow_reconstruction.coherence.families.topology.persistence_objective import PersistenceTopologyObjective
from phycoflow_reconstruction.training import post_training as module


@pytest.mark.parametrize("mode", ["legacy", "finite_primary_v1"])
def test_same_train_panel_frozen_before_gradient_calibration_and_resume(mode, monkeypatch):
    topology_config = {"strategy":"cubical_persistence", "target_use":"paired_supervised",
                       "fields":["a","b"], "geometry":{"grid_shape":[8,8],"periodic":False},
                       "filtration":{"smoothing_sigma":0},
                       "persistence":{"aggregation":mode,"projections":8},
                       "components":{"self":{"enabled":True},"mutual":{"enabled":True,
                           "groups":[["a","b"]],"line_bank_size":16,"training_subset_size":4}}}
    objective = PersistenceTopologyObjective(topology_config,("a","b"))
    topology = SimpleNamespace(spatial_objective=objective,family_weight=1.0)
    families = {"topology":topology}; banks = {"topology":None}
    config = {"runtime":{"seed":7},"optimization":{"batch_size":1},
              "checkpointing":{"exploratory_policy":{"version":"r3_endpoint_corridor_v1"}},
              "coherence":{"compute_budget":{"batch_size":1,"point_count":64},
                           "family_balance":{"mode":"initial_grad_norm","calibration_batches":2,
                                             "seed":19,"max_batch_ratio":1000}}}
    class Model(torch.nn.Module):
        capabilities = SimpleNamespace(structured_grid_required=False)
        def __init__(self):
            super().__init__(); self.amplitude = torch.nn.Parameter(torch.tensor(0.2))
        def training_loss(self,batch):
            return SimpleNamespace(total=self.amplitude.square()+1)
    model = Model(); chosen = []; calls = []; sources = []
    targets = [torch.randn(1,2,8,8,generator=torch.Generator().manual_seed(13+i)) for i in range(4)]
    class Source(list):
        closed = False
        def close(self): self.closed = True
    def source_builder(dataset,indices,config,**kwargs):
        selected = [list(map(int,row)) for row in indices]; chosen.append(selected)
        source = Source(SimpleNamespace(sample_ids=(f"train:{row[0]}",),
                            metadata={"query_indices":torch.arange(64)[None]},
                            image=targets[row[0]]) for row in selected)
        sources.append(source); return source
    def coherence(model,batch,families,banks,config,*,step,generator,phase,
                  step_context=None,source_anchor_model=None):
        # Same inputs/seeds are checked; the actual scalar contains live PH grads.
        calls.append((batch.sample_ids,generator.initial_seed(),objective.source_calibration is not None))
        target = batch.image
        prediction = target + model.amplitude * target.roll(1,-1)
        result = objective(prediction,target,phase=phase)
        if step_context is not None:
            assert source_anchor_model is model
            step_context.update(prediction=prediction.flatten(2).transpose(1,2),
                                reference=target.flatten(2).transpose(1,2),
                                family_results={"topology":result}, batch=batch)
        return result, batch.sample_ids
    monkeypatch.setattr(module,"build_training_batch_source",source_builder)
    monkeypatch.setattr(module,"_coherence_objective",coherence)
    monkeypatch.setattr(module,"subset_query_batch",lambda batch,*args,**kwargs:batch)
    monkeypatch.setattr(module,"_slice_batch",lambda batch,*args,**kwargs:batch)
    monkeypatch.setattr(module,"post_training_mode",lambda *args:None)
    before = torch.get_rng_state().clone()
    calibration = module._calibrate_finite_primary(model,list(range(4)),families,banks,
                                                  config,torch.device("cpu"),"immutable-source")
    assert all(not frozen for _,_,frozen in calls)
    gradient_calibration = module._calibrate_family_balance(
        model,list(range(4)),families,banks,config,device=torch.device("cpu"),
        source_hashes_before={"checkpoint":"immutable-source"})
    assert all(frozen for _,_,frozen in calls[2:])
    assert chosen[0] == chosen[1]
    assert [(ids,seed) for ids,seed,_ in calls[:2]] == [(ids,seed) for ids,seed,_ in calls[2:]]
    assert gradient_calibration["sample_ids"] == [p["sample_ids"] for p in calibration["provenance"]["panels"]]
    assert all(source.closed for source in sources)
    assert torch.equal(before,torch.get_rng_state())
    assert model.amplitude.item() == pytest.approx(0.2)
    assert gradient_calibration["raw_family_gradient_norms"]["topology"] > 0
    for record in gradient_calibration["calibration_batches"]:
        diagnosis = record["R3_component_diagnostics"]["topology"]
        assert diagnosis["components"]["native.raw"]["raw_parameter_gradient_norm"] > 0
        assert diagnosis["components"]["endpoint.raw"]["raw_parameter_gradient_norm"] > 0
        assert diagnosis["components"]["native.pressure"]["raw_parameter_gradient_norm"] == 0
        assert diagnosis["components"]["endpoint.pressure"]["raw_parameter_gradient_norm"] == 0
        assert diagnosis["ABC_norm"] > 0
        assert diagnosis["components"]["topology.finite_primary_finite"]["weighted_parameter_gradient_norm"] > 0
    restored = PersistenceTopologyObjective(topology_config,("a","b"))
    restored.load_source_calibration(calibration)
    assert restored.source_calibration == objective.source_calibration
    assert restored.selected_line_indices(23,"train") == objective.selected_line_indices(23,"train")
    corrupt = deepcopy(calibration); group = next(iter(corrupt["weights"]))
    corrupt["scales"]["finite"][group] *= 2
    with pytest.raises(ValueError,match="cannot replace frozen"):
        restored.load_source_calibration(corrupt)
    corrupt = deepcopy(calibration); corrupt["provenance"]["split"] = "test"
    with pytest.raises(ValueError,match="identity/provenance"):
        restored.load_source_calibration(corrupt)
