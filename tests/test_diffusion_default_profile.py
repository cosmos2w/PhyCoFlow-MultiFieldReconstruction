"""New source launch profiles select EDM; archived weights keep their layout."""

from pathlib import Path

import pytest

from phycoflow_reconstruction.config import load_config, validate_config
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.models import build_model
from phycoflow_reconstruction.models.generative.edm import EDMDiffusionPDEModel
from phycoflow_reconstruction.training.base_training import run_base_training

ROOT = Path(__file__).resolve().parents[1]
PROFILES = tuple(sorted((ROOT / "cases").glob("*/configs/base/diffusion_pde*.yaml")))


@pytest.mark.parametrize("path", PROFILES, ids=lambda path: f"{path.parents[2].name}-{path.stem}")
def test_source_launch_profiles_select_edm_and_full_grid_heun(path):
    config = load_config(path)
    validate_config(config)
    assert config["model"]["formulation"] == "edm"
    assert config["model"]["model_ema_eval"] is True
    assert not {"hidden_channels", "training_timesteps"}.intersection(config["model"])
    for settings in (config["evaluation"], config["evaluation"]["preview"]):
        assert settings["generation_steps"] == 32
        assert settings["query_points"] is None
    assert config["evaluation"]["reconstruction_panel"]["generation_steps"] == 32
    assert config["checkpointing"]["selection_metric"] == "reconstruction_mse"


def test_default_and_existing_edm_imports_build_the_same_model():
    model_config = load_config(ROOT / "configs/models/diffusion_pde.yaml")["model"]
    assert load_config(ROOT / "configs/models/diffusion_pde_edm.yaml")["model"] == model_config
    model_config.update(base_channels=8, channel_multipliers=[1, 2],
                        num_res_blocks=1, time_embed_dim=16, attention_levels=[1],
                        attention_heads=2)
    model = build_model(model_config, DataSpec(("a", "b"), ("1", "1"), 2, (5, 7)))
    assert isinstance(model, EDMDiffusionPDEModel)


@pytest.mark.parametrize("formulation", [None, "ddpm"])
def test_new_source_training_rejects_archived_ddpm_settings_before_run_creation(tmp_path, formulation):
    model = {"name": "diffusion_pde", "hidden_channels": 8}
    if formulation is not None:
        model["formulation"] = formulation
    with pytest.raises(ValueError, match="New DiffusionPDE training requires the EDM profile"):
        run_base_training({"stage": "base_training", "model": model}, case_dir=tmp_path)
    assert not (tmp_path / "runs").exists()


def test_existing_edm_case_path_resolves_to_the_canonical_profile():
    base = ROOT / "cases/turbulent_combustion/configs/base"
    canonical = load_config(base / "diffusion_pde.yaml")
    assert load_config(base / "diffusion_pde_edm_5000ep.yaml") == canonical
    assert load_config(base / "diffusion_pde_5000ep.yaml") == canonical
