"""Self-contained CPU parity gates for the absorbed coherence equations.

The compact reference calculations are independent of production primitives.
They preserve the archived equations without requiring an optional sibling
checkout, so every essential A/B/C parity gate runs in a clean CI clone.
"""

from __future__ import annotations

import math

import pytest
import torch

from phycoflow_reconstruction.coherence import build_coherence_family
from phycoflow_reconstruction.coherence.families.cross_spectrum.basis import (
    coordinate_digest,
)
from phycoflow_reconstruction.coherence.families.topology.betti_curves import (
    betti_curves,
)
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer


def _sorted_w2_columns(generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    generated = torch.sort(generated, dim=0).values
    reference = torch.sort(reference, dim=0).values
    return (generated - reference).square().mean(dim=0)


def _global_component_oracle(
    generated: torch.Tensor,
    reference: torch.Tensor,
    *,
    field_ids: tuple[int, ...],
    pairs: tuple[tuple[int, int], ...],
    pair_directions: torch.Tensor,
    cross_directions: torch.Tensor,
    cross_top_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    self_costs = []
    mutual_costs = []
    cross_costs = []
    for generated_item, reference_item in zip(generated, reference):
        self_costs.append(
            _sorted_w2_columns(
                generated_item[:, field_ids], reference_item[:, field_ids]
            ).mean()
        )
        pair_costs = []
        for pair_index, pair in enumerate(pairs):
            generated_projection = generated_item[:, pair] @ pair_directions[pair_index].T
            reference_projection = reference_item[:, pair] @ pair_directions[pair_index].T
            pair_costs.append(
                _sorted_w2_columns(generated_projection, reference_projection).mean()
            )
        mutual_costs.append(torch.stack(pair_costs).mean())
        generated_projection = generated_item[:, field_ids] @ cross_directions.T
        reference_projection = reference_item[:, field_ids] @ cross_directions.T
        direction_costs = _sorted_w2_columns(generated_projection, reference_projection)
        top_count = math.ceil(cross_top_fraction * direction_costs.numel())
        cross_costs.append(direction_costs.topk(top_count).values.mean())
    return (
        torch.stack(self_costs).mean(),
        torch.stack(mutual_costs).mean(),
        torch.stack(cross_costs).mean(),
    )


def test_global_distribution_matches_archived_canonical_estimators() -> None:
    generated = torch.tensor(
        [[[0.0, 1.0, 2.0], [2.0, -1.0, 1.0], [1.0, 3.0, -2.0], [4.0, 0.5, 0.0]]]
    )
    reference = torch.tensor(
        [[[1.0, 0.0, 2.0], [3.0, -2.0, 0.0], [0.0, 2.0, -1.0], [2.0, 1.5, 1.0]]]
    )
    names = ("a", "b", "c")
    family = build_coherence_family(
        "global_distribution",
        {
            "fields": names,
            "components": {
                "self": {"weight": 1.0},
                "mutual": {"weight": 0.5, "directions": 4, "seed": 19},
                "cross": {
                    "weight": 0.25,
                    "directions": 6,
                    "top_fraction": 0.5,
                    "seed": 23,
                    "include_axes": True,
                    "qmc": True,
                },
            },
        },
        DataSpec(names, ("1",) * 3, 1, (4,)),
        FieldNormalizer.identity(3),
    )
    pair_directions = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]]
    )
    pair_directions = pair_directions / pair_directions.norm(dim=1, keepdim=True)
    pair_directions = pair_directions.expand(3, -1, -1).clone()
    cross_directions = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, -1.0],
        ]
    )
    cross_directions = cross_directions / cross_directions.norm(dim=1, keepdim=True)
    family.components_by_key["mutual_pairwise_swd"].directions.copy_(pair_directions)
    family.components_by_key["cross_joint_topk_swd"].directions.copy_(cross_directions)

    expected_self, expected_mutual, expected_cross = _global_component_oracle(
        generated,
        reference,
        field_ids=(0, 1, 2),
        pairs=((0, 1), (0, 2), (1, 2)),
        pair_directions=pair_directions,
        cross_directions=cross_directions,
        cross_top_fraction=0.5,
    )
    actual = family(generated, reference)
    expected = {
        "global_distribution.self.marginal_w2": expected_self,
        "global_distribution.mutual.pairwise_swd": expected_mutual,
        "global_distribution.cross.joint_topk_swd": expected_cross,
    }
    for path, expected_loss in expected.items():
        torch.testing.assert_close(actual.component_results[path].scalar_loss, expected_loss)
    torch.testing.assert_close(
        actual.scalar_loss,
        expected_self + 0.5 * expected_mutual + 0.25 * expected_cross,
    )


def _spectral_coherence_oracle(coefficients: torch.Tensor, eps: float) -> torch.Tensor:
    batch, modes, fields = coefficients.shape
    result = coefficients.new_empty(modes, fields, fields)
    for mode in range(modes):
        for left in range(fields):
            left_power = coefficients[:, mode, left].abs().square().sum() / batch
            for right in range(fields):
                right_power = coefficients[:, mode, right].abs().square().sum() / batch
                cross = (
                    coefficients[:, mode, left]
                    * coefficients[:, mode, right].conj()
                ).sum() / batch
                result[mode, left, right] = cross.abs().square() / (
                    left_power * right_power + eps
                )
    return result


def _band_energy_oracle(
    coefficients: torch.Tensor, bands: tuple[tuple[int, ...], ...]
) -> torch.Tensor:
    return torch.stack(
        [coefficients[:, indices].abs().square().sum(dim=1) for indices in bands],
        dim=1,
    )


def _cross_band_oracle(energies: torch.Tensor, eps: float) -> torch.Tensor:
    batch, band_count, field_count = energies.shape
    centered = energies - energies.mean(dim=0, keepdim=True)
    covariance = energies.new_empty(band_count, band_count, field_count, field_count)
    for left_band in range(band_count):
        for right_band in range(band_count):
            for left_field in range(field_count):
                for right_field in range(field_count):
                    covariance[left_band, right_band, left_field, right_field] = (
                        centered[:, left_band, left_field]
                        * centered[:, right_band, right_field]
                    ).sum() / batch
    result = torch.empty_like(covariance)
    for left_band in range(band_count):
        for right_band in range(band_count):
            for left_field in range(field_count):
                left_variance = covariance[left_band, left_band, left_field, left_field]
                for right_field in range(field_count):
                    right_variance = covariance[
                        right_band, right_band, right_field, right_field
                    ]
                    value = covariance[left_band, right_band, left_field, right_field]
                    result[left_band, right_band, left_field, right_field] = (
                        value.abs().square() / (left_variance * right_variance + eps)
                    )
    return result


def test_cross_spectrum_restored_basis_matches_archived_zero_mode_and_bands() -> None:
    eps = 1e-8
    config = {
        "fields": ["u", "v"],
        "pairs": [["u", "v"]],
        "eps": eps,
        "graph": {"num_modes": 4, "exclude_zero": False, "bands": ["low", "high"]},
        "components": {
            "same_frequency": {"weight": 1.0},
            "cross_frequency": {"weight": 1.0},
            "band_energy": {"enabled": True, "weight": 1.0},
        },
    }
    data_spec = DataSpec(("u", "v"), ("1", "1"), 1, (4,))
    family = build_coherence_family(
        "cross_spectrum", config, data_spec, FieldNormalizer.identity(2)
    )
    coordinates = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1).expand(4, -1, -1)
    artifact = family.state_artifact()
    artifact["geometry_sha256"] = coordinate_digest(coordinates[0])
    artifact["resolved_sigma"] = 1.0
    artifact["state_dict"]["eigenvalues"] = torch.arange(4, dtype=torch.float32)
    artifact["state_dict"]["eigenvectors"] = torch.eye(4)
    artifact["state_dict"]["band_ids"] = torch.tensor([0, 0, 1, 1])
    family.load_state_artifact(artifact)

    generated = torch.arange(32, dtype=torch.float32).reshape(4, 4, 2) / 7.0
    reference = generated.flip(0).clone()
    reference[:, 0, 1] += torch.tensor([0.0, 1.0, -0.5, 2.0])
    actual = family(generated, reference, coordinates=coordinates)

    coefficients_generated = torch.stack(
        [torch.eye(4).T @ generated_item for generated_item in generated]
    )
    coefficients_reference = torch.stack(
        [torch.eye(4).T @ reference_item for reference_item in reference]
    )
    coherence_generated = _spectral_coherence_oracle(coefficients_generated, eps)
    coherence_reference = _spectral_coherence_oracle(coefficients_reference, eps)
    expected_same = (
        coherence_generated[:, 0, 1] - coherence_reference[:, 0, 1]
    ).square().mean()
    bands = ((0, 1), (2, 3))
    energy_generated = _band_energy_oracle(coefficients_generated, bands)
    energy_reference = _band_energy_oracle(coefficients_reference, bands)
    coupling_generated = _cross_band_oracle(energy_generated, eps)
    coupling_reference = _cross_band_oracle(energy_reference, eps)
    off_diagonal = ~torch.eye(2, dtype=torch.bool)
    expected_cross = (
        coupling_generated[:, :, 0, 1] - coupling_reference[:, :, 0, 1]
    )[off_diagonal].square().mean()
    expected_energy = (
        (energy_generated.mean(0) + eps).log()
        - (energy_reference.mean(0) + eps).log()
    ).square().mean()

    torch.testing.assert_close(
        actual.component_results[
            "cross_spectrum.same_frequency.magnitude_squared"
        ].scalar_loss,
        expected_same,
    )
    torch.testing.assert_close(
        actual.component_results[
            "cross_spectrum.cross_frequency.band_energy_coupling"
        ].scalar_loss,
        expected_cross,
    )
    torch.testing.assert_close(
        actual.component_results["cross_spectrum.band_energy.log_power"].scalar_loss,
        expected_energy,
    )
    torch.testing.assert_close(actual.scalar_loss, expected_same + expected_cross + expected_energy)


def _neighbor_coordinates(
    row: int, column: int, height: int, width: int, periodic: bool
):
    for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        next_row = row + delta_row
        next_column = column + delta_column
        if periodic:
            yield next_row % height, next_column % width
        elif 0 <= next_row < height and 0 <= next_column < width:
            yield next_row, next_column


def _component_count(mask: torch.Tensor, periodic: bool) -> int:
    height, width = mask.shape
    seen: set[tuple[int, int]] = set()
    components = 0
    for row in range(height):
        for column in range(width):
            if not bool(mask[row, column]) or (row, column) in seen:
                continue
            components += 1
            pending = [(row, column)]
            seen.add((row, column))
            while pending:
                current_row, current_column = pending.pop()
                for neighbor in _neighbor_coordinates(
                    current_row, current_column, height, width, periodic
                ):
                    if bool(mask[neighbor]) and neighbor not in seen:
                        seen.add(neighbor)
                        pending.append(neighbor)
    return components


def _hard_betti_oracle(
    field: torch.Tensor, levels: torch.Tensor, periodic: bool
) -> dict[int, torch.Tensor]:
    b0_values = []
    b1_values = []
    for level in levels:
        mask = field >= level
        b0 = _component_count(mask, periodic)
        vertices = int(mask.sum())
        if periodic:
            horizontal = int((mask & torch.roll(mask, -1, dims=1)).sum())
            vertical = int((mask & torch.roll(mask, -1, dims=0)).sum())
            faces = int(
                (
                    mask
                    & torch.roll(mask, -1, dims=1)
                    & torch.roll(mask, -1, dims=0)
                    & torch.roll(mask, shifts=(-1, -1), dims=(0, 1))
                ).sum()
            )
            b2 = int(bool(mask.all()))
        else:
            horizontal = int((mask[:, :-1] & mask[:, 1:]).sum())
            vertical = int((mask[:-1, :] & mask[1:, :]).sum())
            faces = int(
                (
                    mask[:-1, :-1]
                    & mask[:-1, 1:]
                    & mask[1:, :-1]
                    & mask[1:, 1:]
                ).sum()
            )
            b2 = 0
        euler = vertices - horizontal - vertical + faces
        b0_values.append(b0)
        b1_values.append(b0 - euler + b2)
    return {
        0: torch.tensor(b0_values, dtype=field.dtype),
        1: torch.tensor(b1_values, dtype=field.dtype),
    }


@pytest.mark.parametrize("periodic", [False, True])
def test_topology_hard_forward_betti_v1_matches_archived_kernel(periodic: bool) -> None:
    field = torch.tensor(
        [
            [2.0, 2.0, -1.0, -1.0],
            [2.0, 0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0, 2.0],
            [-1.0, -1.0, 2.0, 2.0],
        ]
    )
    levels = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    actual = betti_curves(field, levels, (0, 1), sharpness=12.0, periodic=periodic)
    expected = _hard_betti_oracle(field, levels, periodic)
    for dimension in (0, 1):
        torch.testing.assert_close(actual[dimension], expected[dimension])
