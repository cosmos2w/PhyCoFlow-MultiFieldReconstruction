"""Synthetic coverage-mode external policy, including exact recovery."""
import pytest
import torch
from helpers.coherence import _base_config, _post_config, _write_fixture
from test_multiscale_gram_post_training import _make_shared_reference
from test_new_coherence_resume import _assert_exact, _new_families

from phycoflow_reconstruction.data import training_batches
from phycoflow_reconstruction.training.base_training import run_base_training
from phycoflow_reconstruction.training.post_training import run_post_training
from phycoflow_reconstruction.training.run_store import file_sha256, load_project_checkpoint


def test_coverage_external_policy_resume_without_native_loss(tmp_path, monkeypatch):
    pytest.importorskip("torchjd")
    pytest.importorskip("cooper")
    pytest.importorskip("gudhi")
    torch.set_num_threads(1)
    loader = training_batches.DataLoader
    def in_process(*args, **kwargs):
        kwargs.update(num_workers=0, persistent_workers=False)
        kwargs.pop("prefetch_factor", None)
        return loader(*args, **kwargs)
    monkeypatch.setattr(training_batches, "DataLoader", in_process)
    path = tmp_path / "fixture.h5"
    _write_fixture(path)
    source = run_base_training(_base_config(path), case_dir=tmp_path / "source")
    from phycoflow_reconstruction.models.flows.pointcloud.adapters.pointcloud_ffm_adapter import (
        PointCloudFFM,
    )
    def forbidden(*args, **kwargs):
        raise AssertionError("native training loss was replayed")
    monkeypatch.setattr(PointCloudFFM, "training_loss", forbidden)
    cfg = _post_config(path, source)
    cfg["dataset"]["grid_shape"] = [4, 4]
    cfg["objectives"]["data_retention"] = {"enabled": False, "weight": 0.0}
    cfg["optimization"].update(epochs=2, steps_per_epoch=1, sampling="full_pass", multitask={
        "family_scalarization": {"method": "none"}, "outer_aggregation": None,
        "coherence_aggregation": {"method": "torchjd_config"},
        "external_policy": {"method": "lagrangian", "vector_reduction": "sum",
                            "lagrangian": {"penalty_interval_updates": 1}},
        "fidelity_budget": {"calibration_batches": 2, "calibration_draws": 2}})
    families = _new_families()
    cfg["coherence"].update(compute_budget={"batch_size": 2, "point_count": 16,
        "query_policy": "fixed_shared"}, families=families)
    cfg["checkpointing"] = {"every_epochs": 1, "selection_metric": "reconstruction_mse"}
    cfg["evaluation"].update(max_samples=2, query_points=16, preview={"enabled": False})
    cfg["runtime"].update(progress=False, plot_every_steps=1000, num_workers=0)
    kwargs = {"evaluation_callback": lambda **_: None, "qualification_only": True}
    baseline = run_post_training(cfg, case_dir=tmp_path / "baseline", max_steps=1, **kwargs)
    reference = tmp_path / "reference.json"
    _make_shared_reference(cfg, baseline, reference)
    cfg["optimization"]["multitask"]["fidelity_budget"].update(reference_artifact=str(reference), reference_sha256=file_sha256(reference))
    families["multiscale_spectrum"]["relation_extension"] = {
        "definition": "self_rho_fair_v1", "self_weight": 0.2, "rho_weight": 0.3,
        "std_floor": 0.001, "relative_std_floor": 0.05}
    cfg["coherence"].update(compute_budget={"batch_size": 2, "point_count": 16,
        "query_policy": "fixed_shared"}, families=families,
        query_coverage={"enabled": True, "mode": "adaptive", "seed": 7})
    cfg["checkpointing"] = {"every_epochs": 1, "selection_metric": "reconstruction_mse"}
    cfg["evaluation"].update(max_samples=2, query_points=16, preview={"enabled": False})
    cfg["runtime"].update(progress=False, plot_every_steps=1000, num_workers=0)
    kwargs = {"evaluation_callback": lambda **_: None, "qualification_only": True}
    child = run_post_training(cfg, case_dir=tmp_path / "resumed", max_steps=1, **kwargs)
    run_post_training(cfg, case_dir=tmp_path / "resumed", resume=child, max_steps=1, **kwargs)
    full = run_post_training(cfg, case_dir=tmp_path / "full", max_steps=2, **kwargs)
    a = load_project_checkpoint(child / "checkpoints/last.pt")
    b = load_project_checkpoint(full / "checkpoints/last.pt")
    for key in ("model", "optimizer", "rng_state", "external_policy", "query_coverage"):
        _assert_exact(a[key], b[key])
    assert len((child / "logs/query_coverage.jsonl").read_text().splitlines()) == 4
    assert not (child / "evaluation/after.json").exists()
