"""Atomic recovery and implementation bindings for opt-in FAST execution."""

import json
from copy import deepcopy

import pytest
import torch
from helpers.coherence import _base_config, _family_config, _post_config, _write_fixture

from phycoflow_reconstruction.training.base_training import run_base_training
from phycoflow_reconstruction.training.fast_lagrangian import implementation_binding
from phycoflow_reconstruction.training.post_training import run_post_training
from phycoflow_reconstruction.training.run_store import RunStore


def test_link_runstore_atomic_checkpoint_and_history_recovery(tmp_path):
    payload = tmp_path / "payload"
    payload.mkdir()
    case = tmp_path / "case"
    (case / "runs").mkdir(parents=True)
    (case / "runs" / "campaign").symlink_to(payload, target_is_directory=True)
    config = {"stage": "post_training", "optimization": {"epochs": 10}}
    store = RunStore.create(case, "campaign/T0", config)
    assert store.run_dir.is_relative_to(payload)
    store.save_checkpoint("last", {"global_step": 2, "state": torch.tensor([3.0])})
    history = store.run_dir / "metrics/fast_lagrangian_updates.jsonl"
    history.write_text("".join(json.dumps({"step": i}) + "\n" for i in [1, 2, 3]) + "{")
    restored = RunStore.resume(store.run_dir, config)
    assert restored.load_checkpoint()["global_step"] == 2
    restored.recover_metric_histories(2)
    assert [json.loads(x)["step"] for x in history.read_text().splitlines()] == [1, 2]
    assert not list(store.run_dir.rglob("*.tmp"))
    with pytest.raises(ValueError, match="config hash"):
        RunStore.resume(store.run_dir, {"stage": "post_training", "optimization": {"epochs": 100}})


@pytest.fixture
def toy_post_config(tmp_path):
    path = tmp_path / "fixture.h5"
    _write_fixture(path)
    base = _base_config(path)
    base["model"] = {
        "name": "senseiver", "width": 16, "num_latents": 4, "heads": 2, "depth": 1,
    }
    # Exercise recovery with the supported in-process observation loader.
    base["observations"]["protocol"] = "structured_stride"
    base["evaluation"]["preview"] = {"enabled": False}
    base["runtime"]["progress"] = False
    case = tmp_path / "case"
    source = run_base_training(base, case_dir=case)
    post = _post_config(path, source)
    post["model"] = base["model"]
    post["observations"] = deepcopy(base["observations"])
    post["optimization"]["epochs"] = 3
    post["evaluation"]["preview"] = {"enabled": False}
    post["runtime"]["progress"] = False
    return post, case


def test_empty_fast_settings_enforce_binding_and_resume_with_matching_code(toy_post_config):
    pytest.importorskip("cooper")
    post, case = toy_post_config
    post["optimization"].update(
        fast_lagrangian={},
        multitask={
            "family_scalarization": {"method": "fixed_sum"},
            "outer_aggregation": None,
            "external_policy": {"method": "lagrangian", "vector_reduction": "sum"},
            "fidelity_budget": {"calibration_batches": 1, "calibration_draws": 1},
        },
    )
    post["objectives"]["data_retention"] = {"enabled": False, "weight": 0.0}
    post["coherence"]["compute_budget"]["batch_size"] = 2
    post["coherence"]["families"]["global_distribution"] = _family_config("paired_supervised")
    post["checkpointing"] = {"selection_metric": "reconstruction_mse"}
    child = run_post_training(post, case_dir=case, max_steps=1)
    store = RunStore.resume(child, post)
    saved = store.load_checkpoint("last")
    assert saved["fast_lagrangian_implementation"] == implementation_binding()

    for missing in (True, False):
        changed = deepcopy(saved)
        if missing:
            changed.pop("fast_lagrangian_implementation")
        else:
            changed["fast_lagrangian_implementation"]["training/fast_lagrangian.py"] = "changed"
        store.save_checkpoint("last", changed)
        with pytest.raises(ValueError, match="fast Lagrangian implementation changed"):
            run_post_training(post, case_dir=case, resume=child, max_steps=1)

    store.save_checkpoint("last", saved)
    assert run_post_training(post, case_dir=case, resume=child, max_steps=1) == child
    resumed = store.load_checkpoint("last")
    assert resumed["global_step"] == saved["global_step"] + 1
    assert resumed["fast_lagrangian_implementation"] == saved["fast_lagrangian_implementation"]


@pytest.mark.parametrize("fast_setting", ["absent", "none"])
def test_legacy_nonfast_resume_does_not_require_fast_binding(toy_post_config, fast_setting):
    post, case = toy_post_config
    if fast_setting == "none":
        post["optimization"]["fast_lagrangian"] = None
    child = run_post_training(post, case_dir=case, max_steps=1)
    store = RunStore.resume(child, post)
    saved = store.load_checkpoint("last")
    saved.pop("fast_lagrangian_implementation", None)
    store.save_checkpoint("last", saved)

    assert run_post_training(post, case_dir=case, resume=child, max_steps=1) == child
    assert store.load_checkpoint("last")["global_step"] == saved["global_step"] + 1
