"""End-to-end small CPU profile with artifact reuse at a checkpoint boundary."""

import json
from pathlib import Path

import pytest
import torch
import yaml
from helpers.coherence import _base_config, _post_config, _write_fixture

from phycoflow_reconstruction.cli import _load_case_config
from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.config.load import deep_merge
from phycoflow_reconstruction.training.base_training import run_base_training
from phycoflow_reconstruction.training.post_training import run_post_training
from phycoflow_reconstruction.training.run_store import load_project_checkpoint


@pytest.mark.parametrize("deterministic", [False, True])
def test_tiny_scalable_profile_prefit_train_and_resume(tmp_path, deterministic):
    pytest.importorskip("gudhi")
    path = tmp_path / "fixture.h5"
    _write_fixture(path)
    base = _base_config(path)
    if deterministic:
        base["model"] = {"name": "coordinate_mlp", "hidden_dim": 8, "fourier_bands": 2}
    # An existing SOURCE checkpoint need not contain physical-measure metadata.
    base["dataset"].update(coordinate_dim=2)
    source = run_base_training(base, case_dir=tmp_path / "case")
    post = _post_config(path, source)
    post["model"] = base["model"]
    post["dataset"].update(coordinate_dim=2, physical_measure="native_rectilinear_area")
    post["optimization"].update(epochs=12, steps_per_epoch=1, sampling="full_pass")
    post["coherence"]["families"] = {}
    post = deep_merge(post, load_config(Path("configs/defaults/scalable_anchored_coherence.yaml")))
    post["coherence"]["compute_budget"].update(point_count=16, batch_size=1)
    post["coherence"]["families"]["multiscale_spectrum"]["bands"]["count"] = 1
    post["coherence"]["families"]["topology"]["geometry"]["grid_shape"] = [4, 4]
    post["coherence"]["families"]["topology"]["persistence"]["projections"] = 4
    post["evaluation"].update(query_points=16, preview={"enabled": False})
    post["observation_consistency"] = {"mode": "hard", "final_clamp": True}
    post["checkpointing"] = {"every_epochs": 1, "selection_metric": "reconstruction_mse"}
    post["runtime"]["progress"] = False
    post_path = tmp_path / "posttrain.yaml"
    post_path.write_text(yaml.safe_dump(post))
    post = _load_case_config(post_path, tmp_path / "case", "fixture", [])
    assert post["dataset"]["physical_measure"] == "native_rectilinear_area"
    assert post["source"]["config_origins"]["dataset.physical_measure"] == "child_config"
    child = run_post_training(post, case_dir=tmp_path / "case", max_steps=9)
    first = load_project_checkpoint(child / "checkpoints/last.pt")
    assert first["global_step"] == 9
    assert first["scalable_runtime"]["calibration"]["coefficients"] is None
    artifact = child / "artifacts/scalable_coherence.pt"
    assert artifact.is_file()
    original = artifact.read_bytes()
    run_post_training(post, case_dir=tmp_path / "case", resume=child, max_steps=3)
    second = load_project_checkpoint(child / "checkpoints/last.pt")
    assert second["global_step"] == 12
    assert second["scalable_runtime"]["calibration"]["coefficients"] is not None
    assert artifact.read_bytes() == original
    after = json.loads((child / "evaluation/after.json").read_text())
    assert after["physical_member_mse_model_units"]
    assert after["validation_physical_member_source_ratios"]
    assert any(not torch.equal(first["model"][k], v) for k, v in second["model"].items())

    other = run_post_training(post, case_dir=tmp_path / "case", max_steps=12)
    uninterrupted = load_project_checkpoint(other / "checkpoints/last.pt")

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                equal(x, y)
        else:
            assert a == b

    for key in ("model", "optimizer", "rng_state", "scalable_runtime", "external_policy"):
        equal(second[key], uninterrupted[key])
