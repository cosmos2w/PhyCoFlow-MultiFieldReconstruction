"""Boundary tests for R4's independent pilot authorization envelope."""
import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("r4_launcher", Path(__file__).resolve().parents[1] / "scripts/training/run_upgrade_1002_r4.py")
R4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R4)


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(R4, "sha", lambda _: R4.SOURCE_SHA)
    return {"stage":"post_training","case":"turbulent_combustion",
            "output":{"experiment_name":"Test_1002/R4_20_fast_adaptive"},
            "optimization":{"epochs":200,"batch_size":32,"train_fraction":.15},
            "runtime":{"device":"cuda:0"},"evaluation":{"split":"validation"},
            "source_run":str(R4.SOURCE),"source_checkpoint":"last.pt","model":{"model_ema_eval":False}}


@pytest.mark.parametrize("section,key,value", [
    ("optimization","epochs",250),("optimization","epochs",5000),
    ("optimization","batch_size",16),("optimization","train_fraction",.10),
    ("optimization","steps_per_epoch",38),("evaluation","split","test"),
    ("runtime","device","cuda:1"),("model","model_ema_eval",True),
    ("output","experiment_name","Test_1002/R3_old"),
    ("output","experiment_name","Test_1002/R4_formal_5000ep"),
    ("output","experiment_name","Test_1002/R4_20/../../SOURCE")])
def test_boundary_refusal(config,section,key,value):
    config[section][key]=value
    with pytest.raises(ValueError):R4.validate(config)


def test_partial_resume_charges_exact_additional_exposure(config):
    recovered_updates=1234
    actual=R4.validate(config,resume_epoch=recovered_updates/38,max_steps=38)
    assert actual["reserved_exposures"] == 1
    remainder=R4.validate(config,resume_epoch=recovered_updates/38)
    assert remainder["reserved_exposures"] == (200*38-recovered_updates)/38


def test_no_new_exposure_after_completed_horizon(config):
    with pytest.raises(ValueError):R4.validate(config,resume_epoch=200)


def test_source_hash_failure(config,monkeypatch):
    monkeypatch.setattr(R4,"sha",lambda _:"changed")
    with pytest.raises(ValueError,match="SOURCE checksum"):R4.validate(config)
