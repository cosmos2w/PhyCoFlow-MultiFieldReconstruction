"""Configuration contracts for opt-in external fidelity-budget policies."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from helpers.coherence import _post_config

from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.config.validate import validate_config


def _config() -> dict:
    config = _post_config(Path("/tmp/fidelity-budget-unused.h5"), Path("/tmp/source"))
    config["objectives"] = {
        "data_retention": {"enabled": False, "weight": 0.0},
        "coherence": {"enabled": True, "weight": 1.0},
    }
    config["coherence"]["family_balance"] = {"mode": "none"}
    config["evaluation"]["preview"] = {"enabled": False}
    config["checkpointing"] = {
        "enabled": True,
        "selection_metric": "reconstruction_mse",
    }
    config["optimization"]["multitask"] = {
        "family_scalarization": {"method": "fixed_sum"},
        "outer_aggregation": None,
        "external_policy": {
            "method": "lagrangian",
            "vector_reduction": "sum",
        },
    }
    return config


@pytest.mark.parametrize("method", ["lagrangian", "rectify", "extended_log_barrier"])
def test_external_policy_methods_use_common_defaults(method):
    config = _config()
    config["optimization"]["multitask"]["external_policy"]["method"] = method
    validate_config(config)


def test_external_policy_none_family_scalarization_requires_explicit_sum_and_clears_outer():
    config = _config()
    multitask = config["optimization"]["multitask"]
    multitask["family_scalarization"] = {"method": "none"}
    validate_config(config)

    multitask["outer_aggregation"] = {"method": "sum"}
    with pytest.raises(ValueError, match="outer_aggregation must be null or omitted"):
        validate_config(config)

    multitask["outer_aggregation"] = None
    multitask["external_policy"].pop("vector_reduction")
    with pytest.raises(ValueError, match="vector_reduction must be sum"):
        validate_config(config)


def test_external_null_outer_aggregation_clears_inherited_choice(tmp_path):
    parent = _config()
    parent["optimization"]["multitask"]["outer_aggregation"] = {"method": "torchjd_config"}
    parent_path = tmp_path / "parent.yaml"
    parent_path.write_text(yaml.safe_dump(parent), encoding="utf-8")
    child_path = tmp_path / "child.yaml"
    child_path.write_text(
        "defaults: [parent.yaml]\n"
        "optimization:\n"
        "  multitask:\n"
        "    outer_aggregation: null\n",
        encoding="utf-8",
    )

    config = load_config(child_path)
    validate_config(config)
    assert config["optimization"]["multitask"]["outer_aggregation"] is None


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("relative_increase", True, "relative_increase"),
        ("relative_increase", float("nan"), "relative_increase"),
        ("absolute_allowance", -0.1, "absolute_allowance"),
        ("denominator_floor", 0.0, "denominator_floor"),
        ("calibration_batches", 0, "calibration_batches"),
        ("calibration_draws", 1.5, "calibration_draws"),
        ("calibration_seed", -1, "calibration_seed"),
        ("metric", "relative_l2", "metric"),
    ],
)
def test_external_policy_rejects_invalid_fidelity_budget(path, value, message):
    config = _config()
    budget = config["optimization"]["multitask"].setdefault("fidelity_budget", {})
    budget[path] = value
    with pytest.raises(ValueError, match=message):
        validate_config(config)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda c: c["objectives"]["data_retention"].update(enabled=True, weight=0.1), "data_retention disabled"),
        (lambda c: c["objectives"]["coherence"].update(weight=0.5), "coherence enabled with weight 1"),
        (
            lambda c: c["objectives"].update(
                data_retention={"enabled": True, "weight": 0.1},
                endpoint={"enabled": True, "weight": 1.0},
            ),
            "objectives.endpoint",
        ),
        (
            lambda c: c["objectives"].update(
                data_retention={"enabled": True, "weight": 0.1},
                source_anchor={"enabled": True, "weight": 1.0},
            ),
            "objectives.source_anchor",
        ),
        (lambda c: c["trainable"].update(scope="named_modules", modules=["encoder"]), "scope=full_model"),
        (
            lambda c: c["coherence"]["schedule"].update(every_n_steps=2),
            "every step",
        ),
        (
            lambda c: c["coherence"].update(
                family_balance={"mode": "none", "gradient_diagnostics_every_epochs": 1}
            ),
            "gradient_diagnostics_every_epochs=0",
        ),
        (
            lambda c: c["checkpointing"].update(selection_metric="native_validation_loss"),
            "selection_metric=reconstruction_mse",
        ),
    ],
)
def test_external_policy_rejects_incompatible_legacy_settings(mutation, message):
    config = _config()
    mutation(config)
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_fidelity_budget_requires_external_policy():
    config = _config()
    config["optimization"]["multitask"].pop("external_policy")
    config["optimization"]["multitask"]["fidelity_budget"] = {}
    config["evaluation"]["preview"] = {"enabled": True}
    with pytest.raises(ValueError, match="fidelity_budget requires external_policy"):
        validate_config(config)


@pytest.mark.parametrize(
    "filename",
    [
        "lagrangian.yaml",
        "rectify.yaml",
        "extended_log_barrier.yaml",
        "smoke_flow.yaml",
        "smoke_senseiver.yaml",
    ],
)
def test_portable_turbulent_combustion_smoke_configs_validate(filename):
    path = (
        Path(__file__).resolve().parents[1]
        / "cases/turbulent_combustion/configs/posttrain/fidelity_budget"
        / filename
    )
    config = load_config(path, overrides=["source_run=/tmp/completed-source"])
    validate_config(config)

    assert config["optimization"]["epochs"] == 1
    assert config["optimization"]["steps_per_epoch"] == 2
    assert config["source"]["allow_integration_source"] is False
    assert config["trainable"]["scope"] == "full_model"
    assert config["coherence"]["compute_budget"] == {
        "batch_size": 4,
        "point_count": 4096,
        "query_policy": "fixed_shared",
        "query_seed": 100045,
    }
    topology = config["coherence"]["families"]["topology"]
    assert topology["fields"] == ["CO"]
    assert topology["geometry"]["grid_shape"] == [16, 32]
    assert topology["filtration"]["dimensions"] == [0]
    assert topology["filtration"]["directions"] == ["sublevel"]
    assert topology["persistence"]["projections"] == 2
    assert topology["components"]["mutual"]["enabled"] is False
    assert config["evaluation"]["preview"]["enabled"] is False


def test_legacy_multitask_outer_aggregation_still_validates_unchanged():
    config = _post_config(Path("/tmp/legacy-unused.h5"), Path("/tmp/source"))
    config["optimization"]["multitask"] = {
        "family_scalarization": {"method": "none"},
        "outer_aggregation": {"method": "sum"},
    }
    validate_config(config)


@pytest.mark.parametrize("entrypoint", ["smoke_senseiver.yaml", "rectify.yaml"])
def test_deterministic_source_inheritance_does_not_receive_flow_ema_option(tmp_path, entrypoint):
    from phycoflow_reconstruction.cli import _load_case_config

    root = Path(__file__).resolve().parents[1]
    case_dir = root / "cases/turbulent_combustion"
    path = case_dir / "configs/posttrain/fidelity_budget" / entrypoint
    template = load_config(path)
    source = tmp_path / "source"
    source.mkdir()
    source_model = {"name": "senseiver", "width": 16, "heads": 2, "num_latents": 4, "depth": 1}
    (source / "resolved_config.yaml").write_text(yaml.safe_dump({
        "case": "turbulent_combustion", "dataset": template["dataset"],
        "model": source_model, "observations": template["observations"],
    }))
    config = _load_case_config(
        path, case_dir, "turbulent_combustion", [f"source_run={source}"]
    )
    assert config["model"] == source_model
    assert "model_ema_eval" not in config["model"]
