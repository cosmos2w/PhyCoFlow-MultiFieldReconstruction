"""The resume observer sees live recovery state without changing it."""

import random
from copy import deepcopy

import numpy as np
import torch

from phycoflow_reconstruction.training.post_training import _observe_resume_state


def test_resume_observer_receives_live_state_and_cannot_change_training_streams():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    checkpoint = {"model": deepcopy(model.state_dict()), "optimizer": deepcopy(optimizer.state_dict()),
                  "rng_state": {"torch_cpu": torch.get_rng_state().clone()}}
    first_param = next(iter(checkpoint["optimizer"]["state"]))
    # Simulate a bad restoration; the observer must receive actual live values.
    optimizer.state[next(iter(model.parameters()))]["exp_avg"].add_(1)
    expected_model = deepcopy(model.state_dict())
    expected_optimizer = deepcopy(optimizer.state_dict())
    cpu_rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    calls = []

    def make_payload():
        return {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "rng_state": {"torch_cpu": torch.get_rng_state()}}

    def observe(saved, actual):
        calls.append(True)
        assert not torch.equal(saved["optimizer"]["state"][first_param]["exp_avg"],
                               actual["optimizer"]["state"][first_param]["exp_avg"])
        assert torch.equal(saved["rng_state"]["torch_cpu"], actual["rng_state"]["torch_cpu"])
        actual["model"]["weight"].zero_()
        saved["model"]["weight"].zero_()
        actual["optimizer"]["state"][first_param]["exp_avg"].zero_()
        torch.rand(5)
        random.random()
        np.random.random()

    _observe_resume_state(observe, checkpoint, make_payload, torch.device("cpu"))
    assert calls == [True]
    assert torch.equal(model.weight, expected_model["weight"])
    assert torch.equal(checkpoint["model"]["weight"], expected_model["weight"])
    assert torch.equal(optimizer.state_dict()["state"][first_param]["exp_avg"],
                       expected_optimizer["state"][first_param]["exp_avg"])
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert random.getstate() == python_rng
    restored_numpy_rng = np.random.get_state()
    assert restored_numpy_rng[0] == numpy_rng[0]
    assert np.array_equal(restored_numpy_rng[1], numpy_rng[1])
    assert restored_numpy_rng[2:] == numpy_rng[2:]


def test_resume_observer_runs_only_on_resume_before_update(tmp_path, monkeypatch):
    from helpers.coherence import _base_config, _post_config, _write_fixture
    from phycoflow_reconstruction.data import training_batches
    from phycoflow_reconstruction.training.base_training import run_base_training
    from phycoflow_reconstruction.training.post_training import run_post_training

    loader = training_batches.DataLoader

    def in_process(*args, **kwargs):
        kwargs.update(num_workers=0, persistent_workers=False)
        kwargs.pop("prefetch_factor", None)
        return loader(*args, **kwargs)

    monkeypatch.setattr(training_batches, "DataLoader", in_process)
    dataset = tmp_path / "data.h5"
    _write_fixture(dataset)
    source = run_base_training(_base_config(dataset), case_dir=tmp_path / "source")
    config = _post_config(dataset, source)
    config["optimization"].update(epochs=2, steps_per_epoch=1, sampling="full_pass")
    config["checkpointing"] = {"every_epochs": 1, "selection_metric": "reconstruction_mse"}
    config["runtime"].update(progress=False, plot_every_steps=1000)
    calls = []

    def observe(saved, actual):
        calls.append((saved["global_step"], actual["global_step"]))
        for key in saved["model"]:
            assert torch.equal(saved["model"][key], actual["model"][key])
        for key, state in saved["optimizer"]["state"].items():
            for name, value in state.items():
                assert torch.equal(value, actual["optimizer"]["state"][key][name])
        assert torch.equal(saved["rng_state"]["torch_cpu"], actual["rng_state"]["torch_cpu"])

    child = run_post_training(config, case_dir=tmp_path / "child", max_steps=1,
                              resume_state_callback=observe)
    assert not calls
    run_post_training(config, case_dir=tmp_path / "child", max_steps=1, resume=child,
                      resume_state_callback=observe)
    assert calls == [(1, 1)]
