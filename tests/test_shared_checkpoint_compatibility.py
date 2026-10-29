"""Saved base/child schemas remain usable without opting into new methods.

These are synthetic compatibility fixtures, not trained scientific results.
They exercise the public evaluator with the established checkpoint envelope,
including NeuralOperator metadata and GL-RBF/CQ's required EMA lifecycle state.
"""

from copy import deepcopy

import pytest
import torch
import yaml
from helpers.coherence import _base_config, _write_fixture
from helpers.pointcloud import _small_core_config

from phycoflow_reconstruction.data.factory import open_field_dataset
from phycoflow_reconstruction.data.sensor_protocols import (
    SensorProtocol,
    build_observation_batch,
)
from phycoflow_reconstruction.evaluation.checkpoint import load_evaluation_runtime
from phycoflow_reconstruction.models import build_model
from phycoflow_reconstruction.training.model_lifecycle import (
    add_training_aux_state,
    selected_evaluation_weight_context,
)


def _cq_config():
    config = _small_core_config(2)
    config.pop("model_name")
    config.pop("coord_dim")
    return {"name": "gl_rbf_cq", **config}


_MODELS = [
    pytest.param({"name": "senseiver", "width": 16, "num_latents": 4,
                  "heads": 2, "depth": 1}, id="senseiver"),
    pytest.param({"name": "mimonet", "conditioning_fields": ["u"],
                  "sensor_capacities": [4], "basis_dim": 4,
                  "branch_hidden_dim": 8, "trunk_hidden_dim": 8}, id="mimonet-input-order"),
    pytest.param({"name": "geofno", "hidden_channels": 8, "layers": 1,
                  "modes": [2, 2]}, id="geofno"),
    pytest.param({"name": "pointcloud_ffm", "backbone": "gl_rbf_enh",
                  "prior": "iid", "hidden_dim": 16, "latent_dim": 16,
                  "num_latents": 4, "heads": 2, "latent_blocks": 1,
                  "gather_topk": 2}, id="pointcloud-ffm"),
    pytest.param(_cq_config(), id="gl-rbf-cq-ema"),
]


def _saved_run(tmp_path, model_config, stage):
    path = tmp_path / "fixture.h5"
    _write_fixture(path)
    config = _base_config(path)
    config["model"] = deepcopy(model_config)
    config["stage"] = stage
    config["runtime"]["device"] = "cpu"
    if stage == "post_training":
        config["source_run"] = str(tmp_path / "immutable-source")
        config["source_checkpoint"] = "last.pt"

    dataset = open_field_dataset(config["dataset"], split="validation")
    try:
        model = build_model(config["model"], dataset.data_spec).eval()
        # Distinct non-default weights make a skipped load observable. No training.
        with torch.no_grad():
            next(model.parameters()).add_(0.125)
        update_hook = getattr(model, "after_optimizer_step", None)
        if callable(update_hook):
            update_hook()
        batch = build_observation_batch(
            [dataset[0]], SensorProtocol(field_counts={"u": 4}, seed=19)
        )
        with selected_evaluation_weight_context(model, "configured"), torch.no_grad():
            expected = model.reconstruct(
                batch, steps=1, generator=torch.Generator().manual_seed(29)
            ).prediction.clone()

        # Preserve the original adapter state, including NeuralOperator's known
        # non-tensor _metadata pseudo-key, instead of rewriting old checkpoints.
        payload = {
            "model": model.state_dict(),
            "normalization": dataset.normalizer.state_dict(),
            "global_step": 7,
        }
        if stage == "post_training":
            payload.update(source_run=config["source_run"], family_states={})
        add_training_aux_state(payload, model)
    finally:
        dataset.close()

    run = tmp_path / "saved-run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    torch.save(payload, run / "checkpoints/last.pt")
    return run, payload, batch, expected


@pytest.mark.parametrize("model_config", _MODELS)
@pytest.mark.parametrize("stage", ["base_training", "post_training"])
def test_established_saved_run_loads_without_experimental_state(tmp_path, model_config, stage):
    if model_config["name"] == "geofno":
        pytest.importorskip("neuralop")
    run, payload, batch, expected = _saved_run(tmp_path, model_config, stage)
    assert not {"fast_lagrangian_implementation", "lpc_ordered_state",
                "execution_strategy"} & payload.keys()

    runtime = load_evaluation_runtime(
        run, split="validation", checkpoint="last", sensor_config=None,
        generation_steps=1, device_name="cpu", include_temporal_derivative=False,
    )
    try:
        with selected_evaluation_weight_context(runtime.model, "configured"), torch.no_grad():
            actual = runtime.model.reconstruct(
                batch, steps=1, generator=torch.Generator().manual_seed(29)
            ).prediction
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if "training_aux_state" in payload:
            actual_aux = runtime.model.training_aux_state_dict()
            assert actual_aux["model_ema"]["num_updates"] == 1
    finally:
        runtime.dataset.close()


def test_saved_run_loading_remains_strict_about_missing_weights(tmp_path):
    run, payload, _, _ = _saved_run(
        tmp_path, {"name": "coordinate_mlp", "hidden_dim": 8, "fourier_bands": 2},
        "base_training",
    )
    payload["model"].pop(next(iter(payload["model"])))
    torch.save(payload, run / "checkpoints/last.pt")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_evaluation_runtime(
            run, split="validation", checkpoint="last", sensor_config=None,
            generation_steps=1, device_name="cpu", include_temporal_derivative=False,
        )
