"""Production caller diagnostics preserve tiny CPU updates and supported weights."""
from copy import deepcopy
import random
import sys

import numpy as np
import pytest
import torch

from helpers.coherence import _base_config, _write_fixture
from test_upgrade_1002_lifecycle import upgraded_config
from phycoflow_reconstruction.config.validate import validate_config
from phycoflow_reconstruction.training import gradient_balance as balance
from phycoflow_reconstruction.training import post_training as post
from phycoflow_reconstruction.training.base_training import run_base_training
from phycoflow_reconstruction.training.run_store import file_sha256, load_project_checkpoint


def _freeze(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: _freeze(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_freeze(v) for v in value)
    return deepcopy(value)


def _assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_equal(a, b)
    else:
        assert left == right


@pytest.fixture(scope="module")
def tiny_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("r4_diagnostic_caller_source")
    dataset = root / "fixture.h5"
    _write_fixture(dataset)
    config = _base_config(dataset)
    config["runtime"].update(progress=False, plot_format="pdf")
    source = run_base_training(config, case_dir=root / "source_case")
    return dataset, source


def _run(config, root, monkeypatch, *, legacy_oracle=False):
    counts = {"shared": 0, "sparse": 0, "callback": 0, "grad": 0}
    captured = {}
    coherence = post._coherence_objective
    sparse = post._sparse_family_gradient_diagnostics
    callback = balance.make_shared_family_diagnostic_callback
    updater = post.two_objective_update
    assign = balance._assign_flat_gradient
    grad = torch.autograd.grad

    def shared(*args, **kwargs):
        if "committed_step" in kwargs:
            counts["shared"] += 1
        return coherence(*args, **kwargs)

    def old_diagnostic(*args, **kwargs):
        counts["sparse"] += 1
        return sparse(*args, **kwargs)

    def factory(*args, **kwargs):
        counts["callback"] += 1
        return callback(*args, **kwargs)

    def update(model, optimizer, *args, **kwargs):
        caller = sys._getframe(1).f_locals
        captured["callback_supplied"] = kwargs.get("diagnostic_callback") is not None
        captured["weights"] = (caller["data_update_weight"], caller["coherence_update_weight"])
        captured["raw_family_losses"] = {name: float(result.scalar_loss.detach())
                                        for name, result in caller["step_context"]["family_results"].items()}
        if legacy_oracle and captured["callback_supplied"]:
            # Actual historical diagnostic traversals, same existing raw graphs,
            # original private rollout seed; updater mathematics stays unchanged.
            captured["family_diagnostics"] = old_diagnostic(
                model, caller["batch"], caller["data_loss"], caller["coherence_loss"],
                caller["families"], caller["banks"], caller["config"],
                step=caller["global_step"],
                rollout_seed=caller["seed"] + 1_000_003 + caller["global_step"],
                device=caller["device"])
            kwargs["diagnostic_callback"] = None
        result = updater(model, optimizer, *args, **kwargs)
        captured.setdefault("family_diagnostics", result.get("family_gradient_diagnostics")
                            or caller["row"].get("family_gradient_diagnostics"))
        captured["rng_after_update"] = _freeze((torch.get_rng_state(), np.random.get_state(), random.getstate()))
        return result

    def assigned(parameters, vector):
        captured["preclip"] = _freeze(vector)
        return assign(parameters, vector)

    def counted(*args, **kwargs):
        counts["grad"] += 1
        return grad(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(post, "_coherence_objective", shared)
        patch.setattr(post, "_sparse_family_gradient_diagnostics", old_diagnostic)
        patch.setattr(balance, "make_shared_family_diagnostic_callback", factory)
        patch.setattr(post, "two_objective_update", update)
        patch.setattr(balance, "_assign_flat_gradient", assigned)
        patch.setattr(torch.autograd, "grad", counted)
        child = post.run_post_training(deepcopy(config), case_dir=root, max_steps=1)
    assert captured["family_diagnostics"] is not None
    captured["counts"] = counts
    captured["checkpoint"] = load_project_checkpoint(child / "checkpoints/last.pt")
    return captured


@pytest.mark.parametrize("variant", ["positive", "family_only", "disabled_native", "zero_config_scale"])
def test_production_simple_diagnostics_preserve_updates_and_supported_zero_weights(
        tmp_path, tiny_source, monkeypatch, variant):
    dataset, source = tiny_source
    source_hash = file_sha256(source / "checkpoints/last.pt")
    config = upgraded_config(dataset, source)
    config.pop("fidelity_controller")
    config["optimization"].update(epochs=1, steps_per_epoch=1, sampling="full_pass", update_policy="legacy",
                                  gradient_balance="config", config_data_grad_scale=1.)
    config["optimization"].pop("coherence_gradient_method")
    config["objectives"]["data_retention"].update(enabled=True, weight=.1)
    config["runtime"].update(progress=False, plot_format="pdf", execution_mode="r4_exact",
                             diagnostics_every_epochs=2 if variant == "family_only" else 1)
    config["coherence"]["family_balance"]["gradient_diagnostics_every_epochs"] = 1
    config["coherence"]["families"]["global_distribution"]["weight"] = .3
    config["coherence"]["families"]["topology"]["weight"] = 2.7
    if variant == "disabled_native":
        config["objectives"]["data_retention"].update(enabled=False, weight=0.)
    if variant == "zero_config_scale":
        config["optimization"]["config_data_grad_scale"] = 0.
    validate_config(config)
    actual = _run(config, tmp_path / "actual", monkeypatch)
    oracle = _run(config, tmp_path / "legacy_diagnostic_oracle", monkeypatch, legacy_oracle=True)
    for key in ["preclip", "rng_after_update", "raw_family_losses"]:
        _assert_equal(actual[key], oracle[key])
    for key in ["model", "optimizer", "rng_state", "family_states", "family_scales"]:
        _assert_equal(actual["checkpoint"][key], oracle["checkpoint"][key])
    assert actual["checkpoint"]["global_step"] == oracle["checkpoint"]["global_step"] == 1
    assert actual["counts"]["shared"] == oracle["counts"]["shared"] == 1
    for key in ["family_losses", "family_gradient_norms", "total_coherence_gradient_norm"]:
        assert actual["family_diagnostics"][key] == pytest.approx(oracle["family_diagnostics"][key])
    if variant in {"disabled_native", "zero_config_scale"}:
        assert actual["weights"][0] == 0.
        assert not actual["callback_supplied"]
        assert actual["counts"]["callback"] == 0 and actual["counts"]["sparse"] == 1
    else:
        assert actual["callback_supplied"]
        assert actual["counts"]["sparse"] == 0 and oracle["counts"]["sparse"] == 1
        assert actual["counts"]["grad"] < oracle["counts"]["grad"]
        assert actual["family_diagnostics"]["family_losses"] == pytest.approx(actual["raw_family_losses"])
    assert file_sha256(source / "checkpoints/last.pt") == source_hash
