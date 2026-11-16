import pytest
import torch

from phycoflow_reconstruction.training.internal_calibration import (
    sample_weighted_parameter_gradients,
)
from phycoflow_reconstruction.training.round2_virtual_probe import (
    accumulated_coherence_update,
    choose_scale_ratio,
    pooled_probe_feasibility,
    probe_fidelity,
    virtual_optimizer_step,
    virtual_scale_response,
)


def test_uneven_microbatch_gradients_equal_full_batch():
    w = torch.nn.Parameter(torch.tensor([0.4, -0.2], dtype=torch.float64))
    x = torch.tensor([[1., 2.], [3., -2.], [2., 1.], [0., 3.], [2., -1.]], dtype=torch.float64)
    y = torch.tensor([1., 0., -1., 2., 3.], dtype=torch.float64)
    def losses(a, b):
        pred = x[a:b] @ w
        return {"family": (pred - y[a:b]).square().mean(), "risk": pred.square().mean()}
    full = losses(0, 5)
    expected = {name: torch.autograd.grad(value, (w,), retain_graph=True)[0] for name, value in full.items()}
    pieces = (([str(i) for i in range(a, b)], losses(a, b)) for a, b in ((0, 2), (2, 5)))
    actual, values, ids = sample_weighted_parameter_gradients(pieces, (w,))
    assert ids == [str(i) for i in range(5)]
    for name, expected_gradient in expected.items():
        torch.testing.assert_close(actual[name][0], expected_gradient)
        torch.testing.assert_close(torch.tensor(values[name]), full[name].detach().float())


def test_virtual_adam_step_restores_parameters_optimizer_rng_and_gradients():
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model.weight.grad = torch.ones_like(model.weight)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    def measure():
        return model(torch.tensor([[1., 2.]])).detach().clone()
    def step():
        optimizer.zero_grad()
        model(torch.rand(1, 2)).square().sum().backward()
        optimizer.step()
    original, changed = virtual_optimizer_step(model, optimizer, propose_step=step, assess=measure)
    assert not torch.equal(original, changed)
    assert torch.equal(torch.get_rng_state(), rng)
    assert optimizer.state_dict()["state"] == {}
    assert torch.equal(model.weight.grad, torch.ones_like(model.weight))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_virtual_scale_response_runs_real_policy_and_restores_it():
    from phycoflow_reconstruction.training.fidelity_budget import ExternalPolicy, FidelityBudget
    from phycoflow_reconstruction.training.multitask import OuterAggregator

    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    reference = FidelityBudget.from_reference(("field",), (100.,), 1.)
    controller = ExternalPolicy({"method": "rectify"}, reference, "cpu")
    initial = model.weight.detach().clone()
    policy_state = controller.state_dict()
    def build(ratio):
        w = model.weight.flatten()
        return {"state": w[0].square() + w[1], "scale": w[0] + ratio * w[1].square(),
                "shape": w[0] + w[1]}, w.square().sum().reshape(1)
    before, after = virtual_scale_response(
        model, optimizer, controller, OuterAggregator({"method": "torchjd_config"}),
        build_losses=build, assess=lambda: model.weight.detach().clone(), ratio=2, grad_clip=None,
    )
    assert not torch.equal(before, after)
    torch.testing.assert_close(model.weight, initial, rtol=0, atol=0)
    assert controller.state_dict() == policy_state and not optimizer.state


def test_accumulated_update_matches_whole_batch_config_and_adamw():
    from phycoflow_reconstruction.training.fidelity_budget import ExternalPolicy, FidelityBudget
    from phycoflow_reconstruction.training.multitask import OuterAggregator
    torch.manual_seed(42)
    first = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    second = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    second.load_state_dict(first.state_dict())
    optimizers = [torch.optim.AdamW(m.parameters(), lr=.01) for m in (first, second)]
    policies = [ExternalPolicy({"method": "rectify"}, FidelityBudget.from_reference(("field",), (100.,), 1.), "cpu") for _ in range(2)]
    x = torch.tensor([[1., 2.], [2., 1.], [3., -1.], [1., 3.], [-1., 2.]], dtype=torch.float64)
    def losses(model, a, b):
        z = model(x[a:b]).flatten()
        return {"state": (z - 1).square().mean(), "scale": (z + 2).square().mean(),
                "shape": (z - 3).square().mean(), "risk": z.square().mean()}
    full = losses(first, 0, 5)
    aggregator = OuterAggregator({"method": "torchjd_config"})
    policies[0].update_with_coherence_aggregation(first, optimizers[0], {k: full[k] for k in ("state", "scale", "shape")}, full["risk"].reshape(1), None, aggregator)
    chunks = (([str(i) for i in range(a, b)], losses(second, a, b)) for a, b in ((0, 2), (2, 5)))
    _, ids = accumulated_coherence_update(second, optimizers[1], policies[1], aggregator, chunks,
                                           family_names=("state", "scale", "shape"), risk_names=("risk",), grad_clip=None)
    assert len(ids) == 5
    torch.testing.assert_close(first.weight, second.weight, rtol=1e-12, atol=1e-12)
    assert policies[0].completed_updates == policies[1].completed_updates == 1


def test_scale_ratio_rule_uses_both_parameterizations_and_all_seed_manifests():
    rows = []
    for kind in ("full_model", "residual_head"):
        for ratio in (1, 2, 4):
            for seed in (100, 200, 300):
                for batch in range(4):
                    rows.append({"parameterization": kind, "ratio": ratio, "seed": seed,
                                 "sample_count": 1, "sample_ids": [str(batch)],
                                 "scale_before": 1., "scale_after": .99 if ratio >= 2 else 1.,
                                 "field_and_sensor_feasible": True})
    assert choose_scale_ratio(rows, source_scale=1, deterministic_replay_error=0)["chosen_ratio"] == 2
    for row in rows:
        if row["parameterization"] == "residual_head" and row["ratio"] == 2 and row["seed"] == 200:
            row["field_and_sensor_feasible"] = False
    assert choose_scale_ratio(rows, source_scale=1, deterministic_replay_error=0)["chosen_ratio"] == 4


def test_point_probe_sensor_uses_exact_zero_source_allowance():
    target = torch.zeros(1, 2, 1)
    source = torch.tensor([[[[1.], [1.]]]])
    child = source.clone()
    settings = {"generative": False, "sensor_source": torch.zeros(1, 1, 1),
                "observation_values": torch.zeros(1, 1, 1), "observation_fields": torch.zeros(1, 1, dtype=torch.long),
                "observation_mask": torch.ones(1, 1, dtype=torch.bool), "zero_tolerance": 0., "sampling": {"sampling": "full"}}
    good = probe_fidelity(child, source, target, torch.ones(1, 2), None,
                          sensor_child=torch.zeros(1, 1, 1), **settings)
    assert good["field_passed"] and good["sensor_passed"]
    bad = probe_fidelity(child, source, target, torch.ones(1, 2), None,
                         sensor_child=torch.full((1, 1, 1), .01), **settings)
    assert bad["field_passed"] and not bad["sensor_passed"]


def test_probe_feasibility_is_pooled_across_unequal_batches():
    first = {"risk": [2.], "source_risk": [1.],
             "sensor_by_field": {"0": {"child_mse": 2., "source_mse": 1., "count": 1}}}
    second = {"risk": [.5], "source_risk": [1.],
              "sensor_by_field": {"0": {"child_mse": .5, "source_mse": 1., "count": 3}}}
    pooled = pooled_probe_feasibility([first, second], [1, 3], None, generative=False, zero_tolerance=0.)
    assert pooled["field_and_sensor_feasible"]


@pytest.mark.parametrize("generative", [False, True])
def test_actual_scale_panel_runs_and_caches_production_steps(tmp_path, monkeypatch, generative):
    from types import SimpleNamespace

    import phycoflow_reconstruction.training.round2_prefit as prefit
    from phycoflow_reconstruction.contracts import ModelCapabilities, ObservationBatch
    from phycoflow_reconstruction.models.source_residual import FrozenSourceResidual
    from phycoflow_reconstruction.training.fidelity_budget import (
        ExternalPolicy,
        FidelityBudget,
        MeanRiskDifferenceBudget,
    )
    from phycoflow_reconstruction.training.multitask import OuterAggregator
    from phycoflow_reconstruction.training.round2_virtual_probe import actual_scale_panel

    class Toy(torch.nn.Module):
        capabilities = ModelCapabilities("point", generative, True, False, generative, ("post_training",))
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([.5, .2]))
        def differentiable_reconstruct(self, batch, *, steps, generator):
            x = batch.query_coords
            noise = torch.randn(x.shape, generator=generator) * .1 if generative else 0.
            return self.w[0] * x + self.w[1] * x.square() + noise

    class Runtime:
        fields = ("field",)
        scoring_mode = "conditional" if generative else "deterministic"
        def __init__(self):
            self.families = {name: SimpleNamespace(family_weight=1.) for name in ("global_distribution", "multiscale_spectrum", "topology")}
            self.calibration = SimpleNamespace(coefficients={"scale.joint": 1.})
        class Adapter:
            def score(self, name, prediction, reference, *, coordinates, context, batch, coefficients):
                return SimpleNamespace(scalar_loss=((prediction - reference) * coordinates).square().mean()), {}, {}
        adapter = Adapter()
        def _score(self, name, family, prediction, reference, coords, context, query):
            residual = prediction - reference
            value = {"global_distribution": residual.square().mean(),
                     "multiscale_spectrum": (residual * coords).square().mean() * self.calibration.coefficients["scale.joint"],
                     "topology": (residual * (1 + coords)).square().mean()}[name]
            return SimpleNamespace(scalar_loss=value), {}, {}

    batch = ObservationBatch(obs_coords=torch.tensor([[[0.], [.5]]]), obs_values=torch.zeros(1, 2, 1),
                             obs_field_ids=torch.zeros(1, 2, dtype=torch.long), obs_valid_mask=torch.ones(1, 2, dtype=torch.bool),
                             query_coords=torch.tensor([[[0.], [.25], [.5], [.75]]]), query_valid_mask=torch.ones(1, 4, dtype=torch.bool),
                             target_fields=torch.zeros(1, 4, 1), sample_ids=["sample"], obs_indices=torch.tensor([[0, 2]]),
                             metadata={"query_indices": torch.arange(4)[None], "sample_context": {"auxiliary": {"physical_weights": torch.ones(1, 4)}}})
    monkeypatch.setattr(prefit, "matched_panel_batches", lambda *a, **k: iter([(0, batch)]))
    source, full = Toy(), Toy()
    full.load_state_dict(source.state_dict())
    residual = FrozenSourceResidual(Toy(), coordinate_dim=1, field_scales=[1.], width=4)
    residual.source.load_state_dict(source.state_dict())
    ref = (MeanRiskDifferenceBudget.from_reference(("field",), (100.,), (10.,), (1.,))
           if generative else FidelityBudget.from_reference(("field",), (100.,), 1.))
    config = {"rollout": {"steps": 1, "solver": "euler"}, "observation_consistency": {"mode": "none", "final_clamp": False}, "optimization": {"grad_clip": None}}
    runtime = Runtime()
    result = actual_scale_panel(runtime=runtime, source=source, full_model=full, residual_model=residual,
                                dataset=SimpleNamespace(physical_weights=torch.ones(4)), config=config, panel={}, reference=ref,
                                controller_factory=lambda: ExternalPolicy({"method": "rectify"}, ref, "cpu"),
                                optimizer_factory=lambda kind, model: torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001),
                                aggregator_factory=lambda: OuterAggregator({"method": "torchjd_config"}),
                                output_dir=tmp_path / "cache", seeds=(1, 2, 3), zero_tolerance=0., device="cpu",
                                family_scales={name: 1. for name in runtime.families})
    assert len(result["rows"]) == 18
    saved = torch.load(tmp_path / "cache" / result["rows"][0]["cache"], weights_only=True)
    assert saved["schema"] == "car_round2_virtual_endpoints_v1"
    assert saved["source"].shape == saved["child"].shape == (2 if generative else 1, 1, 4, 1)
