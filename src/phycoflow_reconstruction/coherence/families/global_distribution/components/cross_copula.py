"""Joint copula dependence discrepancy with a smooth tail objective."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations
from typing import Any

import torch
from torch import nn

from .....contracts import CoherenceComponentSpec, TermResult
from ....base import empirical_w2_columns, projection_bank
from .copula import exact_midrank_uniform, quantile_landmark_smooth_cdf
from .point_masks import point_mask_for_batch, select_valid_points
from .tail_risk import identity_calibrated_smooth_cvar


class CrossJointCopulaCVaR(nn.Module):
    """Compare separately ranked multivariate snapshots by sliced W2.

    The training value uses per-channel quantile-landmark smooth-CDF
    coordinates and a mean/smooth-CVaR mixture. Exact empirical midranks are
    included as detached diagnostics. The projection bank is deterministic
    from its seed and serialized as a module buffer.
    """

    def __init__(
        self,
        field_ids: Sequence[int],
        directions: int,
        seed: int,
        *,
        include_axes: bool = False,
        qmc: bool = True,
        canonicalizer: str = "quantile_landmark_smooth_cdf",
        landmarks: int = 64,
        bandwidth: float = 0.1,
        scale_floor: float = 1e-6,
        chunk_size: int = 512,
        alpha: float = 0.25,
        rho: float = 0.1,
        temperature: float = 1e-3,
        eta_tolerance: float = 1e-6,
        eta_max_iterations: int = 80,
        pairwise_diagnostics: bool = False,
        pairwise_directions: int = 16,
        target_use: str = "training_reference",
        units: str = "model_units",
    ) -> None:
        super().__init__()
        if canonicalizer != "quantile_landmark_smooth_cdf":
            raise ValueError(
                "marginal_copula_v2 currently supports only "
                "quantile_landmark_smooth_cdf"
            )
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("cross.tail.alpha must lie in [0,1]")
        if not 0.0 < float(rho) <= 1.0:
            raise ValueError("cross.tail.rho must lie in (0,1]")
        if float(temperature) <= 0.0:
            raise ValueError("cross.tail.temperature must be positive")
        if float(eta_tolerance) <= 0.0 or int(eta_max_iterations) < 1:
            raise ValueError("cross.tail eta tolerance and iteration limit must be positive")
        if int(directions) < 1:
            raise ValueError("cross.directions must be positive")
        if int(pairwise_directions) < 1:
            raise ValueError("pairwise diagnostic directions must be positive")

        self.field_ids = tuple(int(value) for value in field_ids)
        if not self.field_ids:
            raise ValueError("cross joint copula requires at least one field")
        self.spec = CoherenceComponentSpec(
            "cross.joint_copula_cvar",
            target_use,
            units,
            True,
            metadata={
                "definition": "marginal_copula_v2",
                "canonicalizer": canonicalizer,
                "tail": "identity_calibrated_smooth_cvar",
            },
        )
        self.spec.validate()
        if len(self.field_ids) >= 2:
            mixed_directions = projection_bank(
                len(self.field_ids),
                int(directions),
                seed=int(seed),
                include_axes=bool(include_axes),
                qmc=bool(qmc),
            )
        else:
            mixed_directions = torch.empty(0, len(self.field_ids))
        self.register_buffer("directions", mixed_directions)

        self.landmarks = int(landmarks)
        self.bandwidth = float(bandwidth)
        self.scale_floor = float(scale_floor)
        self.chunk_size = int(chunk_size)
        self.alpha = float(alpha)
        self.rho = float(rho)
        self.temperature = float(temperature)
        self.eta_tolerance = float(eta_tolerance)
        self.eta_max_iterations = int(eta_max_iterations)
        self.pairwise_diagnostics_enabled = bool(pairwise_diagnostics)
        self.pairwise_direction_count = int(pairwise_directions)

        self.pairs = tuple(combinations(range(len(self.field_ids)), 2))
        if self.pairs:
            pair_banks = [
                projection_bank(
                    2,
                    self.pairwise_direction_count,
                    seed=int(seed)
                    + self.field_ids[left] * 1009
                    + self.field_ids[right],
                    qmc=False,
                )
                for left, right in self.pairs
            ]
            pair_directions = torch.stack(pair_banks)
        else:
            pair_directions = torch.empty(0, self.pairwise_direction_count, 2)
        self.register_buffer("pairwise_directions", pair_directions)

    def _copula(self, values: torch.Tensor) -> torch.Tensor:
        return quantile_landmark_smooth_cdf(
            values,
            landmarks=self.landmarks,
            bandwidth=self.bandwidth,
            scale_floor=self.scale_floor,
            chunk_size=self.chunk_size,
        )

    @staticmethod
    def _hard_tail(values: torch.Tensor, rho: float) -> torch.Tensor:
        tail_mass = float(rho) * values.numel()
        whole_directions = math.floor(tail_mass)
        fractional_direction = tail_mass - whole_directions
        descending = values.sort(descending=True).values
        numerator = descending[:whole_directions].sum()
        if fractional_direction > 0.0:
            numerator = numerator + fractional_direction * descending[whole_directions]
        return numerator / tail_mass

    @torch.no_grad()
    def pairwise_diagnostics(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
        *,
        point_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return raw-state, smooth-copula, and exact-copula pairwise SWD.

        This audit path is opt-in because it evaluates 16 directions for every
        selected field pair by default. It never contributes to the principal
        A objective.
        """
        if generated.shape != reference.shape or generated.ndim != 3:
            raise ValueError("pairwise diagnostics require matching [B,N,C] tensors")
        if not self.pairs:
            empty = generated.new_empty(generated.shape[0], 0)
            return {
                "pair_indices": torch.empty(0, 2, dtype=torch.long, device=generated.device),
                "pairwise_raw_state_swd": empty,
                "pairwise_soft_copula_swd": empty,
                "pairwise_exact_copula_swd": empty,
            }
        validate_mask = point_mask
        if validate_mask is not None and validate_mask.dtype != torch.bool:
            raise TypeError("point_mask must use boolean dtype")

        pair_raw_batches = []
        pair_soft_batches = []
        pair_exact_batches = []
        banks = self.pairwise_directions.to(
            device=generated.device, dtype=generated.dtype
        )
        for batch_index, (generated_item, reference_item) in enumerate(
            zip(generated, reference)
        ):
            mask = point_mask_for_batch(
                validate_mask,
                batch_index,
                batch_size=generated.shape[0],
                point_count=generated.shape[1],
                device=generated.device,
            )
            generated_item = select_valid_points(
                generated_item[:, self.field_ids], mask
            )
            reference_item = select_valid_points(
                reference_item[:, self.field_ids], mask
            )
            generated_copula = self._copula(generated_item)
            reference_copula = self._copula(reference_item)
            generated_exact = exact_midrank_uniform(generated_item)
            reference_exact = exact_midrank_uniform(reference_item)
            raw_per_pair = []
            soft_per_pair = []
            exact_per_pair = []
            for pair_index, pair in enumerate(self.pairs):
                pair_bank = banks[pair_index]
                raw_per_pair.append(
                    empirical_w2_columns(
                        generated_item[:, pair] @ pair_bank.T,
                        reference_item[:, pair] @ pair_bank.T,
                    ).mean()
                )
                soft_per_pair.append(
                    empirical_w2_columns(
                        generated_copula[:, pair] @ pair_bank.T,
                        reference_copula[:, pair] @ pair_bank.T,
                    ).mean()
                )
                exact_per_pair.append(
                    empirical_w2_columns(
                        generated_exact[:, pair] @ pair_bank.T,
                        reference_exact[:, pair] @ pair_bank.T,
                    ).mean()
                )
            pair_raw_batches.append(torch.stack(raw_per_pair))
            pair_soft_batches.append(torch.stack(soft_per_pair))
            pair_exact_batches.append(torch.stack(exact_per_pair))

        pair_ids = torch.tensor(self.pairs, dtype=torch.long, device=generated.device)
        return {
            "pair_indices": pair_ids,
            "pairwise_raw_state_swd": torch.stack(pair_raw_batches),
            "pairwise_soft_copula_swd": torch.stack(pair_soft_batches),
            "pairwise_exact_copula_swd": torch.stack(pair_exact_batches),
        }

    def forward(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
        *,
        point_mask: torch.Tensor | None = None,
    ) -> TermResult:
        if generated.shape != reference.shape or generated.ndim != 3:
            raise ValueError("cross joint copula inputs must share shape [B,N,C]")
        if len(self.field_ids) < 2:
            zero = generated.sum(dim=(1, 2)) * 0.0
            return TermResult(
                zero,
                zero.mean(),
                reason="fewer than two configured fields",
                diagnostics={"definition": "marginal_copula_v2"},
            )

        directions = self.directions.to(device=generated.device, dtype=generated.dtype)
        temperature = self.temperature
        per_sample = []
        direction_costs = []
        raw_means = []
        smooth_tails = []
        hard_tails = []
        eta_values = []
        eta_residuals = []
        effective_weights = []
        exact_means = []

        for batch_index, (generated_item, reference_item) in enumerate(
            zip(generated, reference)
        ):
            mask = point_mask_for_batch(
                point_mask,
                batch_index,
                batch_size=generated.shape[0],
                point_count=generated.shape[1],
                device=generated.device,
            )
            generated_item = select_valid_points(
                generated_item[:, self.field_ids], mask
            )
            reference_item = select_valid_points(
                reference_item[:, self.field_ids], mask
            )
            generated_copula = self._copula(generated_item)
            reference_copula = self._copula(reference_item.detach())
            generated_projection = generated_copula @ directions.T
            reference_projection = reference_copula @ directions.T
            costs = empirical_w2_columns(generated_projection, reference_projection)
            tail = identity_calibrated_smooth_cvar(
                costs,
                rho=self.rho,
                temperature=temperature,
                tolerance=self.eta_tolerance,
                max_iterations=self.eta_max_iterations,
            )
            raw_mean = costs.mean()
            score = (1.0 - self.alpha) * raw_mean + self.alpha * tail.value
            per_sample.append(score)
            direction_costs.append(costs)
            raw_means.append(raw_mean)
            smooth_tails.append(tail.value)
            hard_tails.append(self._hard_tail(costs.detach(), self.rho))
            eta_values.append(tail.eta)
            eta_residuals.append(tail.eta_residual)
            effective_weights.append(
                (1.0 - self.alpha) / costs.numel()
                + self.alpha * tail.direction_weights
            )
            with torch.no_grad():
                exact_generated = exact_midrank_uniform(generated_item)
                exact_reference = exact_midrank_uniform(reference_item)
                exact_costs = empirical_w2_columns(
                    exact_generated @ directions.T,
                    exact_reference @ directions.T,
                )
                exact_means.append(exact_costs.mean())

        per_sample_tensor = torch.stack(per_sample)
        diagnostics: dict[str, Any] = {
            "definition": "marginal_copula_v2",
            "canonicalizer": "quantile_landmark_smooth_cdf",
            "landmarks": self.landmarks,
            "bandwidth": self.bandwidth,
            "scale_floor": self.scale_floor,
            "chunk_size": self.chunk_size,
            "tail_alpha": self.alpha,
            "tail_rho": self.rho,
            "tail_temperature": temperature,
            "tail_temperature_policy": "fixed_configured_value",
            "per_direction_w2": torch.stack(direction_costs),
            "mean_directional_w2": torch.stack(raw_means),
            "hard_cvar_diagnostic": torch.stack(hard_tails),
            "identity_calibrated_smooth_cvar": torch.stack(smooth_tails),
            "eta": torch.stack(eta_values),
            "eta_residual": torch.stack(eta_residuals),
            "effective_direction_weights": torch.stack(effective_weights),
            "exact_midranks_projected_w2_diagnostic": torch.stack(exact_means),
        }
        if self.pairwise_diagnostics_enabled:
            diagnostics.update(
                self.pairwise_diagnostics(
                    generated.detach(),
                    reference.detach(),
                    point_mask=point_mask,
                )
            )
        return TermResult(
            per_sample_cost=per_sample_tensor,
            scalar_loss=per_sample_tensor.mean(),
            diagnostics=diagnostics,
        )
