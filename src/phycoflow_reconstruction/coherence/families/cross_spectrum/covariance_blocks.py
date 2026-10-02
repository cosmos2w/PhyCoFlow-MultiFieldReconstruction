"""Second-order graph-spectral covariance blocks.

All covariance matrices live in retained graph-coefficient space.  No spatial
``[N, N]`` covariance or projector is formed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch


def _check_coefficients(coefficients: torch.Tensor) -> None:
    if coefficients.ndim != 3:
        raise ValueError("graph coefficients must have shape [B,K,C]")
    if coefficients.shape[0] < 2 or coefficients.shape[1] < 1 or coefficients.shape[2] < 1:
        raise ValueError("graph coefficients require B>=2, K>=1, and C>=1")
    if not torch.isfinite(coefficients).all():
        raise FloatingPointError("graph coefficients contain non-finite values")


def linear_coefficient_covariance(
    coefficients: torch.Tensor, *, ddof: int = 1
) -> torch.Tensor:
    """Return centered sample covariance with axes ``[K,K,C,C]``.

    Entry ``[k,q,i,j]`` is ``sum_b (a[b,k,i]-mean[k,i]) *
    conj(a[b,q,j]-mean[q,j]) / (B-ddof)``.  The unbiased sample estimator
    (``ddof=1``) is the production convention.
    """
    _check_coefficients(coefficients)
    if ddof < 0 or coefficients.shape[0] <= ddof:
        raise ValueError("covariance requires batch_size > ddof >= 0")
    centered = coefficients - coefficients.mean(dim=0, keepdim=True)
    return torch.einsum("bki,bqj->kqij", centered, centered.conj()) / (
        coefficients.shape[0] - ddof
    )


def covariance_band_energies(
    covariance: torch.Tensor, band_ids: torch.Tensor
) -> torch.Tensor:
    """Trace each per-field auto-covariance block, yielding ``[L,C]``."""
    if covariance.ndim != 4 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("coefficient covariance must have shape [K,K,C,C]")
    if covariance.shape[2] != covariance.shape[3]:
        raise ValueError("coefficient covariance field axes must be square")
    if band_ids.ndim != 1 or band_ids.numel() != covariance.shape[0]:
        raise ValueError("band_ids must align with the covariance mode axes")
    if band_ids.numel() == 0 or int(band_ids.min()) != 0:
        raise ValueError("band_ids must use contiguous ids starting at zero")
    band_count = int(band_ids.max()) + 1
    energies = []
    for band in range(band_count):
        modes = torch.nonzero(band_ids == band, as_tuple=False).flatten().to(covariance.device)
        if modes.numel() == 0:
            raise ValueError("every spectral band must contain at least one retained mode")
        # A trace over the mode block selects equal mode indices and then the
        # field diagonal.  ``real`` removes roundoff imaginary parts from a
        # Hermitian auto-covariance.
        mode_diagonal = covariance[modes, modes]
        energies.append(torch.diagonal(mode_diagonal, dim1=-2, dim2=-1).real.sum(dim=0))
    return torch.stack(energies, dim=0)


def covariance_band_block(
    covariance: torch.Tensor,
    band_ids: torch.Tensor,
    left_band: int,
    right_band: int,
    left_field: int,
    right_field: int,
) -> torch.Tensor:
    """Extract ``U_l* C_ij U_m`` in the retained coefficient basis."""
    if covariance.ndim != 4 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("coefficient covariance must have shape [K,K,C,C]")
    if band_ids.ndim != 1 or band_ids.numel() != covariance.shape[0]:
        raise ValueError("band_ids must align with the covariance mode axes")
    if not (0 <= left_field < covariance.shape[2] and 0 <= right_field < covariance.shape[3]):
        raise IndexError("field index is outside the covariance axes")
    left_modes = torch.nonzero(band_ids == left_band, as_tuple=False).flatten().to(covariance.device)
    right_modes = torch.nonzero(band_ids == right_band, as_tuple=False).flatten().to(covariance.device)
    if left_modes.numel() == 0 or right_modes.numel() == 0:
        raise ValueError("covariance block references an empty or unknown band")
    return covariance.index_select(0, left_modes).index_select(1, right_modes)[
        :, :, left_field, right_field
    ]


@dataclass(frozen=True)
class BlockLosses:
    same_frequency: torch.Tensor
    cross_frequency: torch.Tensor
    diagnostics: dict[str, object]


def second_order_covariance_block_losses(
    generated_coefficients: torch.Tensor,
    reference_coefficients: torch.Tensor,
    band_ids: torch.Tensor,
    pairs: tuple[tuple[int, int], ...],
    *,
    relative_floor: float = 1.0e-6,
    absolute_floor: float = 1.0e-12,
    minimum_reference_band_fraction: float = 1.0e-8,
    include_block_values: bool = False,
    calibration_reference_energies: torch.Tensor | None = None,
    calibration_ensemble_size: int | None = None,
    ddof: int = 1,
    generated_covariance: torch.Tensor | None = None,
    reference_covariance: torch.Tensor | None = None,
) -> BlockLosses:
    """Compare signed/complex normalized covariance blocks by Frobenius loss.

    The same reference-calibrated additive floor is used for generated and
    reference blocks.  Eligibility depends only on reference band energies;
    a generated field therefore cannot mask a low-power discrepancy by
    collapsing its own spectrum.  If a frozen calibration panel is supplied,
    its energies set both the eligibility mask and the additive floor; the
    current reference batch still defines the target covariance and its
    normalization scale.  Generated energies are clamped below by a positive
    calibration-derived value before the square root, preventing a singular
    derivative at exact collapse.  Frobenius norms are squared sums, with no
    division by block size, then averaged over eligible blocks.
    """
    _check_coefficients(generated_coefficients)
    _check_coefficients(reference_coefficients)
    if generated_coefficients.shape != reference_coefficients.shape:
        raise ValueError("generated and reference coefficients must have equal shapes")
    if (
        not isfinite(relative_floor)
        or not isfinite(absolute_floor)
        or relative_floor < 0
        or absolute_floor < 0
    ):
        raise ValueError("covariance stabilization floors must be finite and non-negative")
    if relative_floor == 0 and absolute_floor == 0:
        raise ValueError("at least one covariance stabilization floor must be positive")
    if not 0.0 <= minimum_reference_band_fraction < 1.0:
        raise ValueError("minimum_reference_band_fraction must be in [0,1)")
    if ddof < 0 or generated_coefficients.shape[0] <= ddof:
        raise ValueError("covariance requires batch_size > ddof >= 0")
    if band_ids.ndim != 1 or band_ids.numel() != generated_coefficients.shape[1]:
        raise ValueError("band_ids must align with the coefficient mode axis")
    field_count = generated_coefficients.shape[2]
    if any(i == j or not (0 <= i < field_count and 0 <= j < field_count) for i, j in pairs):
        raise ValueError("field pairs must contain distinct valid indices")
    if not pairs:
        raise ValueError("at least one cross-field pair is required")

    if (generated_covariance is None) != (reference_covariance is None):
        raise ValueError("generated and reference precomputed covariances must be provided together")
    if generated_covariance is None:
        covariance_generated = linear_coefficient_covariance(generated_coefficients, ddof=ddof)
        covariance_reference = linear_coefficient_covariance(reference_coefficients, ddof=ddof)
        covariance_estimator = "centered_coefficients"
    else:
        assert reference_covariance is not None
        expected_shape = (
            generated_coefficients.shape[1],
            generated_coefficients.shape[1],
            generated_coefficients.shape[2],
            generated_coefficients.shape[2],
        )
        if generated_covariance.shape != expected_shape or reference_covariance.shape != expected_shape:
            raise ValueError("precomputed covariance tensors do not align with coefficient axes")
        if not torch.isfinite(generated_covariance).all() or not torch.isfinite(
            reference_covariance
        ).all():
            raise FloatingPointError("precomputed coefficient covariance contains non-finite values")
        if generated_covariance.device != reference_covariance.device:
            raise ValueError("precomputed covariance tensors must be on the same device")
        covariance_generated = generated_covariance
        covariance_reference = reference_covariance
        covariance_estimator = "precomputed_centered_covariance"
    energy_generated = covariance_band_energies(covariance_generated, band_ids)
    energy_reference = covariance_band_energies(covariance_reference, band_ids)
    band_count = energy_reference.shape[0]
    if calibration_reference_energies is None:
        energy_calibration = energy_reference.detach()
        calibration_source = "current_reference_batch"
        effective_calibration_ensemble_size = int(reference_coefficients.shape[0])
    else:
        if calibration_reference_energies.shape != energy_reference.shape:
            raise ValueError("calibration reference energies must have shape [L,C]")
        if not torch.isfinite(calibration_reference_energies).all():
            raise FloatingPointError("calibration reference energies are non-finite")
        if torch.any(calibration_reference_energies < 0):
            raise ValueError("calibration reference energies must be non-negative")
        energy_calibration = calibration_reference_energies.detach().to(energy_reference)
        calibration_source = "frozen_reference_panel"
        effective_calibration_ensemble_size = calibration_ensemble_size

    tiny = torch.finfo(energy_reference.dtype).tiny

    def eligible_fraction(energy: torch.Tensor) -> torch.Tensor:
        totals = energy.sum(dim=0, keepdim=True)
        return torch.where(
            totals > 0,
            energy / totals.clamp_min(torch.finfo(energy.dtype).tiny),
            torch.zeros_like(energy),
        )

    fractions_calibration = eligible_fraction(energy_calibration)
    same_values: list[torch.Tensor] = []
    cross_values: list[torch.Tensor] = []
    same_records: list[dict[str, object]] = []
    cross_records: list[dict[str, object]] = []
    skipped_same = 0
    skipped_cross = 0

    for left_field, right_field in pairs:
        for left_band in range(band_count):
            for right_band in range(band_count):
                is_same = left_band == right_band
                eligible = bool(
                    (fractions_calibration[left_band, left_field] >= minimum_reference_band_fraction)
                    and (fractions_calibration[right_band, right_field] >= minimum_reference_band_fraction)
                )
                if not eligible:
                    if is_same:
                        skipped_same += 1
                    else:
                        skipped_cross += 1
                    continue

                ref_scale = torch.sqrt(
                    energy_reference[left_band, left_field].clamp_min(0)
                    * energy_reference[right_band, right_field].clamp_min(0)
                )
                calibration_scale = torch.sqrt(
                    energy_calibration[left_band, left_field].clamp_min(0)
                    * energy_calibration[right_band, right_field].clamp_min(0)
                )
                floor = absolute_floor + relative_floor * calibration_scale
                left_generated_floor = (
                    relative_floor * energy_calibration[left_band, left_field]
                ).clamp_min(tiny)
                right_generated_floor = (
                    relative_floor * energy_calibration[right_band, right_field]
                ).clamp_min(tiny)
                gen_scale = torch.sqrt(
                    torch.maximum(
                        energy_generated[left_band, left_field].clamp_min(0),
                        left_generated_floor,
                    )
                    * torch.maximum(
                        energy_generated[right_band, right_field].clamp_min(0),
                        right_generated_floor,
                    )
                )
                ref_denominator = torch.sqrt(
                    energy_reference[left_band, left_field].clamp_min(tiny)
                    * energy_reference[right_band, right_field].clamp_min(tiny)
                ) + floor
                gen_denominator = gen_scale + floor
                if bool(ref_denominator <= 0) or bool(gen_denominator <= 0):
                    raise FloatingPointError("covariance block denominator is not positive")

                gen_block = covariance_band_block(
                    covariance_generated,
                    band_ids,
                    left_band,
                    right_band,
                    left_field,
                    right_field,
                ) / gen_denominator
                ref_block = covariance_band_block(
                    covariance_reference,
                    band_ids,
                    left_band,
                    right_band,
                    left_field,
                    right_field,
                ) / ref_denominator
                difference = gen_block - ref_block
                loss = difference.abs().square().sum()
                record = {
                    "fields": (left_field, right_field),
                    "bands": (left_band, right_band),
                    "frobenius_squared": float(loss.detach()),
                    "reference_scale": float(ref_scale.detach()),
                    "shared_floor": float(floor.detach()),
                    "reference_energy_fractions": (
                        float(fractions_calibration[left_band, left_field].detach()),
                        float(fractions_calibration[right_band, right_field].detach()),
                    ),
                }
                if include_block_values:
                    def payload(value: torch.Tensor) -> object:
                        value = value.detach().cpu()
                        if value.is_complex():
                            return {
                                "real": value.real.tolist(),
                                "imag": value.imag.tolist(),
                            }
                        return value.tolist()

                    record["generated_normalized_block"] = payload(gen_block)
                    record["reference_normalized_block"] = payload(ref_block)
                if is_same:
                    same_values.append(loss)
                    same_records.append(record)
                else:
                    cross_values.append(loss)
                    cross_records.append(record)

    zero = generated_coefficients.real.sum() * 0.0
    same_loss = torch.stack(same_values).mean() if same_values else zero
    cross_loss = torch.stack(cross_values).mean() if cross_values else zero
    return BlockLosses(
        same_frequency=same_loss,
        cross_frequency=cross_loss,
        diagnostics={
            "covariance_ddof": ddof,
            "ensemble_size": int(generated_coefficients.shape[0]),
            "relative_floor": relative_floor,
            "absolute_floor": absolute_floor,
            "minimum_reference_band_fraction": minimum_reference_band_fraction,
            "calibration_source": calibration_source,
            "calibration_ensemble_size": effective_calibration_ensemble_size,
            "covariance_estimator": covariance_estimator,
            "same_frequency_eligible_blocks": len(same_values),
            "same_frequency_skipped_blocks": skipped_same,
            "cross_frequency_eligible_blocks": len(cross_values),
            "cross_frequency_skipped_blocks": skipped_cross,
            "same_frequency_blocks": same_records,
            "cross_frequency_blocks": cross_records,
        },
    )


class LinearCoefficientCovarianceAccumulator:
    """Mergeable pooled sufficient statistics for graph coefficients.

    Stores ``sum(a)`` and ``sum(a_k_i conj(a_q_j))``.  Its covariance matches
    one pooled centered sample covariance and does not average per-batch
    normalized scores.
    """

    def __init__(self) -> None:
        self.sample_count = 0
        self.coefficient_sum: torch.Tensor | None = None
        self.cross_product_sum: torch.Tensor | None = None

    def update(self, coefficients: torch.Tensor) -> None:
        _check_coefficients(coefficients)
        values = coefficients.detach()
        accumulation_dtype = (
            torch.complex128 if values.is_complex() else torch.float64
        )
        values = values.to(dtype=accumulation_dtype)
        current_sum = values.sum(dim=0)
        current_cross_product = torch.einsum("bki,bqj->kqij", values, values.conj())
        if self.coefficient_sum is None:
            self.coefficient_sum = current_sum
            self.cross_product_sum = current_cross_product
        else:
            if self.coefficient_sum.shape != current_sum.shape:
                raise ValueError("pooled coefficient shapes changed between updates")
            if self.coefficient_sum.device != current_sum.device:
                raise ValueError("pooled coefficients changed device between updates")
            self.coefficient_sum = self.coefficient_sum + current_sum
            assert self.cross_product_sum is not None
            self.cross_product_sum = self.cross_product_sum + current_cross_product
        self.sample_count += int(values.shape[0])

    def merge(self, other: LinearCoefficientCovarianceAccumulator) -> None:
        if other.sample_count == 0:
            return
        if self.sample_count == 0:
            self.sample_count = other.sample_count
            assert other.coefficient_sum is not None
            assert other.cross_product_sum is not None
            self.coefficient_sum = other.coefficient_sum.clone()
            self.cross_product_sum = other.cross_product_sum.clone()
            return
        assert self.coefficient_sum is not None
        assert self.cross_product_sum is not None
        assert other.coefficient_sum is not None
        assert other.cross_product_sum is not None
        if self.coefficient_sum.shape != other.coefficient_sum.shape:
            raise ValueError("cannot pool accumulators with different coefficient shapes")
        if self.coefficient_sum.device != other.coefficient_sum.device:
            raise ValueError("cannot pool accumulators on different devices")
        self.coefficient_sum = self.coefficient_sum + other.coefficient_sum
        self.cross_product_sum = self.cross_product_sum + other.cross_product_sum
        self.sample_count += other.sample_count

    def covariance(self, *, ddof: int = 1) -> torch.Tensor:
        if self.sample_count <= ddof or ddof < 0:
            raise ValueError("pooled covariance requires sample_count > ddof >= 0")
        if self.coefficient_sum is None or self.cross_product_sum is None:
            raise ValueError("pooled covariance accumulator is empty")
        centered_cross_product = self.cross_product_sum - torch.einsum(
            "ki,qj->kqij", self.coefficient_sum, self.coefficient_sum.conj()
        ) / self.sample_count
        return centered_cross_product / (self.sample_count - ddof)
