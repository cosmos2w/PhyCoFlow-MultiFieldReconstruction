"""Bounded Test_1002 invocations retain the long recipe and fail before setup."""

import copy
import sys

import pytest
import yaml

from helpers.coherence import _post_config
from phycoflow_reconstruction.cli import run_case_cli
from phycoflow_reconstruction.config import validate_config
from phycoflow_reconstruction.training import post_training


@pytest.fixture
def long_test_config(tmp_path):
    config = _post_config(tmp_path / "unopened.h5", tmp_path / "unopened_source")
    config["inherit_base_config"] = False
    config["optimization"]["epochs"] = 5000
    config["output"]["experiment_name"] = "Test_1002/fixture_sibling"
    return config


def test_bare_long_test_recipe_remains_invalid(long_test_config):
    with pytest.raises(ValueError, match="strictly fewer than 250"):
        validate_config(long_test_config)


def test_until_two_context_keeps_the_entire_saved_recipe(long_test_config):
    before = copy.deepcopy(long_test_config)
    validate_config(long_test_config, invocation_until_epoch=2)
    assert long_test_config == before
    assert long_test_config["optimization"]["epochs"] == 5000


@pytest.mark.parametrize("target", [0, -1, True, False, 1.5, "2", 250, 5000])
def test_invalid_test_epoch_context_is_rejected(long_test_config, target):
    with pytest.raises(ValueError, match="strictly fewer than 250"):
        validate_config(long_test_config, invocation_until_epoch=target)


def test_context_cannot_exceed_the_configured_horizon(long_test_config):
    long_test_config["optimization"]["epochs"] = 1
    with pytest.raises(ValueError, match="strictly fewer than 250"):
        validate_config(long_test_config, invocation_until_epoch=2)


@pytest.mark.parametrize("epochs,output", [(200, "Test_1002/fixture"), (5000, "formal_fixture")])
def test_existing_short_and_formal_recipe_validation_is_unchanged(long_test_config, epochs, output):
    long_test_config["optimization"]["epochs"] = epochs
    long_test_config["output"]["experiment_name"] = output
    validate_config(long_test_config)


def test_normal_cli_passes_until_two_without_rewriting_recipe(long_test_config, tmp_path, monkeypatch):
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(long_test_config))
    captured = {}

    def receive(config, **kwargs):
        validate_config(config, invocation_until_epoch=kwargs["until_epoch"])
        captured.update(config=copy.deepcopy(config), kwargs=kwargs)
        return tmp_path / "unstarted_child"

    monkeypatch.setattr(post_training, "run_post_training", receive)
    monkeypatch.setattr(sys, "argv", ["run.py", "post-train", "--config", str(path), "--until-epoch", "2"])
    assert run_case_cli("fixture", tmp_path) == 0
    assert captured["config"] == long_test_config
    assert captured["kwargs"]["until_epoch"] == 2
    assert captured["kwargs"]["max_steps"] is None
    assert captured["kwargs"]["additional_epochs"] is None
    assert captured["kwargs"]["resume"] is None


def test_normal_cli_bare_validate_rejects_before_dataset_access(long_test_config, tmp_path, monkeypatch):
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(long_test_config))
    monkeypatch.setattr(sys, "argv", ["run.py", "validate", "--config", str(path)])
    with pytest.raises(ValueError, match="strictly fewer than 250"):
        run_case_cli("fixture", tmp_path)


@pytest.mark.parametrize("other", [["--max-steps", "2"], ["--additional-epochs", "2"]])
def test_normal_cli_conflicting_limits_fail_before_routing(long_test_config, tmp_path, monkeypatch, other):
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(long_test_config))
    monkeypatch.setattr(sys, "argv", ["run.py", "post-train", "--config", str(path), "--until-epoch", "2", *other])
    with pytest.raises(SystemExit) as error:
        run_case_cli("fixture", tmp_path)
    assert error.value.code == 2


@pytest.mark.parametrize("limits", [
    {}, {"max_steps": 2}, {"additional_epochs": 2}, {"until_epoch": 0},
    {"until_epoch": True}, {"until_epoch": 1.5}, {"until_epoch": 250},
])
def test_direct_plain_legacy_trainer_rejects_before_source_device_or_setup(long_test_config, tmp_path, monkeypatch, limits):
    # This ordinary fixture has no matched-stream policy, parameter penalty,
    # controller or native audit. The Test_1002 guard must still be mandatory.
    def forbidden(*args, **kwargs):
        pytest.fail("invalid invocation reached SOURCE/device/trainer setup")

    monkeypatch.setattr(post_training, "_configure_persistence_workers", forbidden)
    monkeypatch.setattr(post_training, "seed_everything", forbidden)
    monkeypatch.setattr(post_training, "load_source_model", forbidden)
    monkeypatch.setattr(post_training.torch.cuda, "is_available", forbidden)
    with pytest.raises(ValueError, match="strictly fewer than 250"):
        post_training.run_post_training(long_test_config, case_dir=tmp_path, **limits)


@pytest.mark.parametrize("limits", [
    {"until_epoch": 2, "max_steps": 2},
    {"until_epoch": 2, "additional_epochs": 2},
    {"max_steps": 2, "additional_epochs": 2},
])
def test_direct_conflicting_limits_fail_before_setup(long_test_config, tmp_path, monkeypatch, limits):
    monkeypatch.setattr(post_training, "_configure_persistence_workers", lambda *args: pytest.fail("conflicting invocation reached setup"))
    with pytest.raises(ValueError, match="choose only one"):
        post_training.run_post_training(long_test_config, case_dir=tmp_path, **limits)


def test_direct_until_two_validates_without_constructing_source_or_device(long_test_config, tmp_path, monkeypatch):
    class ValidationPassed(Exception):
        pass

    def stop_before_setup(*args, **kwargs):
        raise ValidationPassed

    monkeypatch.setattr(post_training, "_configure_persistence_workers", stop_before_setup)
    before = copy.deepcopy(long_test_config)
    with pytest.raises(ValidationPassed):
        post_training.run_post_training(long_test_config, case_dir=tmp_path, until_epoch=2)
    assert long_test_config == before
