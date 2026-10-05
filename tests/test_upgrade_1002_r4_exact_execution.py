"""R4 exact execution parity: gradients, optimizer state and degeneracy."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from phycoflow_reconstruction.training.gradient_balance import (
    coherence_primal_dual_update,
    combine_coherence_gradients,
    two_objective_update,
)
from phycoflow_reconstruction.coherence.families.global_distribution.components.tail_risk import (
    identity_calibrated_smooth_cvar, batched_identity_calibrated_smooth_cvar,
)
from phycoflow_reconstruction.coherence.families.global_distribution.components.cross_copula import CrossJointCopulaCVaR
from phycoflow_reconstruction.coherence.families.cross_spectrum.covariance_blocks import second_order_covariance_block_losses


class MixedFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.real = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
        self.complex = torch.nn.Parameter(torch.tensor([1. + 2j, -3j], dtype=torch.complex128))
        self.unused = torch.nn.Parameter(torch.tensor([2., 3.], dtype=torch.float64))

    def losses(self):
        shared = self.real.square().sum() + self.complex.abs().square().sum()
        return {"A": shared * .2, "B": self.real.sin().sum(),
                "C": self.complex.real.square().sum() * .4}, shared * .1


@pytest.mark.parametrize("grad_clip", [None, .5])
@pytest.mark.parametrize("adaptive", [False, True])
def test_scalar_matches_component_gradient_adamw_and_unused_decay(grad_clip, adaptive):
    old = MixedFixture()
    fast = deepcopy(old)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=.002, weight_decay=.07) for m in (old, fast)]
    for _ in range(3):
        for model, optimizer, execution in zip((old, fast), optimizers, ("legacy", "scalar")):
            families, fidelity = model.losses()
            if adaptive:
                coherence_primal_dual_update(
                    model, optimizer, families, fidelity, method="weighted_sum",
                    coherence_direction_scale=.37, grad_clip=grad_clip,
                    diagnostics=False, execution=execution)
            else:
                two_objective_update(
                    model, optimizer, fidelity, sum(families.values()), mode="weighted_sum",
                    data_weight=.7, coherence_weight=.3, grad_clip=grad_clip,
                    diagnostics=False, execution=execution)
        for left, right in zip(old.parameters(), fast.parameters()):
            torch.testing.assert_close(left, right, atol=2e-14, rtol=2e-14)
            torch.testing.assert_close(left.grad, right.grad, atol=2e-14, rtol=2e-14)
            for key in optimizers[0].state[left]:
                torch.testing.assert_close(optimizers[0].state[left][key], optimizers[1].state[right][key],
                                           atol=2e-14, rtol=2e-14)
    assert not torch.equal(fast.unused, torch.tensor([2., 3.], dtype=torch.float64))


def test_scalar_backward_count_and_diagnostic_oracle():
    model = MixedFixture()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    families, fidelity = model.losses()
    original = torch.autograd.grad
    with patch("torch.autograd.grad", wraps=original) as gradients:
        result = coherence_primal_dual_update(
            model, optimizer, families, fidelity, method="weighted_sum", grad_clip=None,
            diagnostics=False, execution="scalar")
        assert gradients.call_count == 0
    assert result["gradient/component_diagnostics_sampled"] is False
    families, fidelity = model.losses()
    with patch("torch.autograd.grad", wraps=original) as gradients:
        coherence_primal_dual_update(
            model, optimizer, families, fidelity, method="weighted_sum", grad_clip=None,
            diagnostics=True, execution="scalar")
        assert gradients.call_count == 4


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("case", ["random", "duplicate", "opposing", "zero", "nearly_dependent"])
def test_small_gram_matches_installed_config_pinv_oracle(dtype, case):
    pytest.importorskip("conflictfree")
    generator = torch.Generator().manual_seed(13)
    matrix = torch.randn(3, 257, generator=generator, dtype=dtype)
    if case == "duplicate":
        matrix[1] = matrix[0] * 2
    elif case == "opposing":
        matrix[1] = -matrix[0]
        matrix[2] = 0
    elif case == "zero":
        matrix.zero_()
    elif case == "nearly_dependent":
        matrix[1] = matrix[0] + 1e-5 * matrix[1]
    bank = dict(zip("ABC", matrix))
    expected, original = combine_coherence_gradients(bank, config_backend="legacy")
    actual, report = combine_coherence_gradients(bank, config_backend="gram")
    tolerance = 1e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert original["fallback_reason"] == report["fallback_reason"]
    assert original["active_count"] == report["active_count"]
    assert original["strict_common_descent"] == report["strict_common_descent"]
    for name in bank:
        assert report["combined_dot"][name] == pytest.approx(original["combined_dot"][name], abs=tolerance,
                                                            rel=tolerance)


def test_scalar_detached_disconnected_and_nonfinite_guard():
    model = MixedFixture()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    before = deepcopy(model.state_dict())
    coherence_primal_dual_update(
        model, optimizer, {"A": torch.tensor(0.), "B": torch.tensor(2., requires_grad=True)},
        torch.tensor(0.), method="weighted_sum", grad_clip=1., diagnostics=False, execution="scalar")
    for name, value in model.state_dict().items():
        assert torch.equal(before[name], value)
    with pytest.raises(FloatingPointError):
        coherence_primal_dual_update(
            model, optimizer, {"A": model.real.sum() * float("inf")}, torch.tensor(0.),
            method="weighted_sum", grad_clip=None, diagnostics=False, execution="scalar")


def test_two_objective_config_gram_one_update_parity():
    old = MixedFixture()
    fast = deepcopy(old)
    for model, backend in ((old, "legacy"), (fast, "gram")):
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        two_objective_update(model, optimizer, model.real[0] + model.complex.real.sum(),
                             -2 * model.real[0] + 3 * model.real[1], mode="config",
                             data_weight=1., coherence_weight=1., grad_clip=1., config_backend=backend)
    for left, right in zip(old.parameters(), fast.parameters()):
        torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("extreme", [False, True])
def test_scalar_float32_complex64_clipping_preserves_wide_oracle_semantics(extreme):
    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.real = torch.nn.Parameter(torch.zeros(3))
            self.complex = torch.nn.Parameter(torch.zeros(2, dtype=torch.complex64))
            self.unused = torch.nn.Parameter(torch.ones(1))
    old = Fixture()
    fast = deepcopy(old)
    scale = 1e38 if extreme else 3.
    limit = 1e-20 if extreme else .5
    for model, execution in ((old,"legacy"),(fast,"scalar")):
        optimizer = torch.optim.SGD(model.parameters(), lr=1.)
        # Zero parameter values keep the forward scalar finite even for the
        # large gradient. The norm and clip product must remain float64.
        loss = scale * (model.real.sum() + model.complex.real.sum())
        coherence_primal_dual_update(model,optimizer,{"A":loss},loss * 0,
                                     method="weighted_sum",grad_clip=limit,
                                     diagnostics=False,execution=execution)
    for left,right in zip(old.parameters(),fast.parameters()):
        torch.testing.assert_close(right.grad,left.grad,rtol=0,atol=0)
        torch.testing.assert_close(right,left,rtol=0,atol=0)
    assert float(fast.real.grad.abs().max()) > 0


@pytest.mark.parametrize("rho", [.1, .5, 1.])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batched_tail_value_root_envelope_gradient_identity_and_ties(rho, dtype):
    torch.manual_seed(45)
    costs = torch.rand(4, 17, dtype=dtype)
    costs[0].zero_()
    costs[1] = .2
    original = costs.clone().requires_grad_()
    fast = costs.clone().requires_grad_()
    tails = [identity_calibrated_smooth_cvar(row, rho=rho, temperature=.02, tolerance=1e-10)
             for row in original]
    packed = batched_identity_calibrated_smooth_cvar(fast, rho=rho, temperature=.02, tolerance=1e-10)
    expected = torch.stack([tail.value for tail in tails])
    torch.testing.assert_close(packed.value, expected, rtol=1e-7, atol=1e-7)
    if rho < 1:
        torch.testing.assert_close(packed.eta, torch.stack([tail.eta for tail in tails]), rtol=1e-10, atol=1e-10)
    expected.sum().backward(); packed.value.sum().backward()
    torch.testing.assert_close(original.grad, fast.grad, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(fast.grad, packed.direction_weights, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("masked", [False, True])
def test_fast_copula_preserves_own_cdf_mask_tie_value_gradient_and_rng(masked):
    torch.manual_seed(11)
    generated = torch.randn(3, 33, 3, dtype=torch.float64)
    generated[0, :, 0] = .4
    reference = torch.randn_like(generated)
    point_mask = torch.rand(3,33) > .2 if masked else None
    component = CrossJointCopulaCVaR((0,1,2), 8, 19, landmarks=8, chunk_size=16,
                                     temperature=.02, eta_tolerance=1e-10)
    old = generated.clone().requires_grad_()
    fast = generated.clone().requires_grad_()
    rng = torch.get_rng_state()
    original = component(old, reference, point_mask=point_mask)
    packed = component(fast, reference, point_mask=point_mask, batch_tail_roots=True, diagnostics=False)
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(packed.scalar_loss, original.scalar_loss, rtol=1e-10, atol=1e-10)
    original.scalar_loss.backward(); packed.scalar_loss.backward()
    torch.testing.assert_close(fast.grad, old.grad, rtol=1e-10, atol=1e-10)
    assert packed.diagnostics["diagnostics_deferred"]
    assert "exact_midranks_projected_w2_diagnostic" not in packed.diagnostics


@pytest.mark.parametrize("complex_coefficients", [False, True])
@pytest.mark.parametrize("collapse", [False, True])
def test_fast_signed_covariance_floor_mask_value_gradient_parity(complex_coefficients, collapse):
    torch.manual_seed(21)
    dtype = torch.complex128 if complex_coefficients else torch.float64
    coefficients = torch.randn(5, 9, 3, dtype=dtype)
    if collapse:
        coefficients.zero_()
    reference = torch.randn_like(coefficients)
    reference[:, :3, 0] *= 1e-12
    bands = torch.arange(9) // 3
    old = coefficients.clone().requires_grad_()
    fast = coefficients.clone().requires_grad_()
    settings = dict(minimum_reference_band_fraction=.01)
    original = second_order_covariance_block_losses(old,reference,bands,((0,1),(1,2)),**settings)
    packed = second_order_covariance_block_losses(fast,reference,bands,((0,1),(1,2)),
                                                 execution="exact",defer_diagnostics=True,**settings)
    for key in ("same_frequency", "cross_frequency"):
        torch.testing.assert_close(getattr(packed,key),getattr(original,key),rtol=1e-12,atol=1e-12)
        assert packed.diagnostics[key+"_eligible_blocks"] == original.diagnostics[key+"_eligible_blocks"]
        assert packed.diagnostics[key+"_skipped_blocks"] == original.diagnostics[key+"_skipped_blocks"]
    (original.same_frequency+original.cross_frequency).backward()
    (packed.same_frequency+packed.cross_frequency).backward()
    torch.testing.assert_close(fast.grad,old.grad,rtol=1e-12,atol=1e-12)
    assert torch.isfinite(fast.grad).all()
    assert packed.diagnostics["same_frequency_blocks"] == []
