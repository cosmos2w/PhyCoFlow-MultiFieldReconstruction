import copy

import numpy as np
import pytest
import torch

from phycoflow_reconstruction.coherence.families.multiscale_spectrum.features import (
    FactorManifest,
    raw_statistics,
)
from phycoflow_reconstruction.coherence.families.multiscale_spectrum.losses import (
    joint_spectral_energy_score,
)
from phycoflow_reconstruction.coherence.families.native_quadrature import NativeQuadrature


def geometry(query=(10, 20, 30, 40, 50, 60)):
    # Two disconnected pieces, nonuniform masses and globally M-orthonormal modes.
    m = np.array([.05, .10, .15, .20, .20, .30])
    p = np.array([0, 0, 0, 1, 1, 1])
    seed = np.array([[2., 0.], [-1., 1.], [0., -1.], [1., 2.], [-1., 0.], [0., -1.]])
    seed -= np.stack([np.average(seed[p == k], axis=0, weights=m[p == k]) for k in p])
    gram = seed.T @ (m[:, None] * seed)
    eigenvalues, vectors = np.linalg.eigh(gram)
    basis = seed @ vectors @ np.diag(1 / np.sqrt(eigenvalues)) @ vectors.T
    return {"coordinates": np.array([[0.], [1.], [2.], [10.], [11.], [12.]]),
            "model_coordinates": np.array([[0.], [.1], [.2], [1.], [1.1], [1.2]]),
            "node_ids": np.array([10, 20, 30, 40, 50, 60]), "mass": m, "pieces": p,
            "basis": basis, "filters": np.array([[1., .4], [.2, 1.]]), "query_ids": np.array(query),
            "metric_identity": "physical-x-v1", "calibration_identity": "train-frozen-v1"}


def native_stats(x, g):
    t = lambda v: torch.tensor(v, dtype=x.dtype)
    p = g["pieces"]
    return raw_statistics(x, mass=t(g["mass"]), basis=t(g["basis"]),
                          filters=t(g["filters"]),
                          piece_measures=t(np.stack([g["mass"] * (p == i) for i in range(2)])),
                          pieces=torch.tensor(p), manifest=FactorManifest((), ()),
                          with_piece_means=True)


def test_full_support_values_and_every_draw_gradients_match_native():
    g = geometry(query=(60, 20, 40, 10, 50, 30))
    adapter = NativeQuadrature(**g)
    torch.manual_seed(4)
    draws = [torch.randn(2, 6, 2, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    targets = torch.randn(2, 6, 2, dtype=torch.float64)
    indices = torch.tensor(adapter.query_index)
    sparse = [adapter.statistics(x[:, indices]) for x in draws]
    native = [native_stats(x, g) for x in draws]
    for a, b in zip(sparse, native):
        for key in ("coefficients", "gram", "piece_means", "second_moment", "fluctuation_power"):
            torch.testing.assert_close(a[key], b[key], atol=1e-12, rtol=1e-12)
    mean_ga = torch.autograd.grad(sparse[0]["piece_means"].sum(), draws[0], retain_graph=True)[0]
    mean_gb = torch.autograd.grad(native[0]["piece_means"].sum(), draws[0], retain_graph=True)[0]
    torch.testing.assert_close(mean_ga, mean_gb, atol=1e-12, rtol=1e-12)
    target_a = adapter.statistics(targets[:, indices])
    target_b = native_stats(targets, g)
    def score(stats, target):
        return joint_spectral_energy_score(
            torch.stack([s["coefficients"] for s in stats]), target["coefficients"],
            filters=torch.tensor(g["filters"]), scales=torch.ones(2, 2, dtype=torch.float64),
            mass=torch.tensor(g["mass"]), draw_piece_means=torch.stack([s["piece_means"] for s in stats]),
            target_piece_means=target["piece_means"], piece_weights=torch.tensor(adapter.piece_mass),
            mean_scales=torch.ones(2, 2, dtype=torch.float64), mean_weight=.25)[0].sum()
    a, b = score(sparse, target_a), score(native, target_b)
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    ga = torch.autograd.grad(a, draws, retain_graph=True)
    gb = torch.autograd.grad(b, draws)
    for left, right in zip(ga, gb):
        assert left.norm() > 0
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)
    d = adapter.diagnostics()
    np.testing.assert_allclose(d["restricted_basis_gram"], np.eye(2), atol=1e-12)
    assert d["rank"] == 2


def test_sparse_mass_means_approximation_and_deterministic_ties():
    g = geometry(query=(30, 10, 40, 60))
    adapter = NativeQuadrature(**g)
    np.testing.assert_allclose(adapter.query_weights, [.15, .15, .4, .3])
    np.testing.assert_allclose(adapter.diagnostics()["piece_mass_error"], 0, atol=1e-15)
    # A piecewise-constant input has exactly zero coefficients, means retain gradients.
    x = torch.tensor([[[2.], [2.], [5.], [5.]]], dtype=torch.float64, requires_grad=True)
    s = adapter.statistics(x)
    torch.testing.assert_close(s["piece_means"], torch.tensor([[[2.], [5.]]], dtype=x.dtype))
    torch.testing.assert_close(s["coefficients"], torch.zeros_like(s["coefficients"]), atol=1e-14, rtol=0)
    grad, = torch.autograd.grad(s["piece_means"].sum(), x)
    torch.testing.assert_close(grad.flatten(), torch.tensor([.5, .5, 4/7, 3/7], dtype=x.dtype))
    # Low native modes are approximations under lumping; report actual error, not exact recovery.
    full = torch.tensor(g["basis"][None, :, :], dtype=x.dtype)
    estimated = adapter.statistics(full[:, adapter.query_index])
    true = native_stats(full, g)
    assert (estimated["coefficients"] - true["coefficients"]).norm() > 1e-3
    np.testing.assert_allclose(estimated["coefficients"][0].numpy(),
                               adapter.diagnostics()["centered_restricted_basis_gram"], atol=1e-12)
    assert np.linalg.norm(adapter.diagnostics()["centered_restricted_basis_gram"] - np.eye(2)) > 1e-3


def test_strict_support_and_artifact_identity():
    with pytest.raises(ValueError, match="every native piece"):
        NativeQuadrature(**geometry(query=(10, 20)))
    with pytest.raises(ValueError, match="not a native"):
        NativeQuadrature(**geometry(query=(10, 40, 99)))
    with pytest.raises(ValueError, match="unique"):
        NativeQuadrature(**geometry(query=(10, 10, 40)))
    adapter = NativeQuadrature(**geometry(query=(10, 30, 40, 60)))
    adapter.validate_coordinates(adapter.query_ids, adapter.model_coordinates[adapter.query_index])
    with pytest.raises(ValueError, match="coordinates"):
        adapter.validate_coordinates(adapter.query_ids, adapter.coordinates[adapter.query_index])
    artifact = adapter.state_artifact()
    restored = NativeQuadrature.from_artifact(artifact, metric_identity="physical-x-v1",
                                              calibration_identity="train-frozen-v1")
    np.testing.assert_array_equal(restored.query_weights, adapter.query_weights)
    with pytest.raises(ValueError, match="contract"):
        NativeQuadrature.from_artifact(artifact, metric_identity="wrong", calibration_identity="train-frozen-v1")
    broken = copy.deepcopy(artifact)
    broken["basis"][0][0] += .1
    with pytest.raises(ValueError, match="digest"):
        NativeQuadrature.from_artifact(broken, metric_identity="physical-x-v1", calibration_identity="train-frozen-v1")
