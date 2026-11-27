#!/usr/bin/env python3
"""Reproduce tiny CPU-only support/Shape calculations; no model or dataset access.

Run from the repository root with the project's topology-enabled Python:
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python \
    UpdatePlans/_Plans/Coherence_Adaptive_Repair/Round_8/evidence/support_shape_cpu_diagnostic.py

The JSON printed to stdout is a diagnostic receipt, not learning evidence. The
regional feature below is a prototype calculation, not a production observer.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import gudhi
import numpy
import scipy
import torch

REPOSITORY = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPOSITORY / "src"))

from phycoflow_reconstruction.coherence.families.topology.persistence import (  # noqa: E402
    cubical_diagrams,
    sliced_diagram_distance,
)
from phycoflow_reconstruction.coherence.physical_risk import physical_ensemble_risks  # noqa: E402


def main() -> None:
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    height, width = 32, 64
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")

    def bowls(shift):
        first = -(1 - ((x - 14 - shift).square() + (y - 12).square()) / 25).clamp_min(0)
        second = -0.7 * (
            1 - ((x - 39 - shift).square() + (y - 22).square()) / 16
        ).clamp_min(0)
        return torch.minimum(first, second)

    windows = torch.stack(
        [
            torch.exp(-0.5 * ((x - cx) / 14).square() - 0.5 * ((y - cy) / 12).square())
            for cy in torch.linspace(0, height - 1, 2)
            for cx in torch.linspace(0, width - 1, 4)
        ]
    )
    thresholds = torch.tensor([-0.6, -0.3, 0.0])

    def features(field, mask=None):
        mass = torch.ones_like(field) if mask is None else mask.to(field)
        soft = torch.sigmoid(8 * (thresholds[:, None, None] - field[None]))
        numerator = (soft[:, None] * windows[None] * mass).sum((-2, -1))
        return numerator / (windows * mass).sum((-2, -1))[None]

    def feature_distance(first, second):
        delta = features(first) - features(second)
        return torch.linalg.vector_norm(delta) / delta.numel() ** 0.5

    truth = bowls(torch.tensor(0.0))
    shifts = []
    for displacement in (0.25, 1.0, 8.0):
        shift = torch.tensor(displacement, requires_grad=True)
        distance = feature_distance(bowls(shift), truth)
        distance.backward()
        shifts.append(
            {
                "displacement_columns": displacement,
                "distance": distance.item(),
                "distance_derivative_wrt_shift": shift.grad.item(),
            }
        )

    translated = bowls(torch.tensor(8.0))
    diagrams = cubical_diagrams(
        torch.stack((truth, translated)), periodic=False, locations=True
    )
    ph_distance = sum(
        sliced_diagram_distance(
            diagrams[0][dim], diagrams[1][dim], normalization=height * width
        )
        for dim in (0, 1)
    )

    def hard_half_features(field):
        soft = torch.sigmoid(8 * (thresholds[:, None, None] - field[None]))
        return torch.stack(
            (soft[:, :, :32].mean((-2, -1)), soft[:, :, 32:].mean((-2, -1)))
        )

    mask = torch.ones(height, width, dtype=torch.bool)
    mask[:, 20:25] = False
    changed = truth.clone()
    changed[:, 20:25] = 1000.0
    equal_feature = torch.tensor([0.2, 0.6, 0.7], requires_grad=True)
    equal_distance = torch.linalg.vector_norm(equal_feature - equal_feature.detach())
    equal_distance.backward()
    saturated = torch.full((height, width), 100.0, requires_grad=True)
    saturated_features = features(saturated).sum()
    saturated_features.backward()

    # Exact design expectation over 15 complement subsets of fixed endpoint
    # values. This cannot establish invariance of a query-dependent model.
    native_mass = torch.tensor([1.0, 2.0, 1.0, 3.0, 4.0, 2.0, 5.0, 1.0])
    reference = torch.arange(16.0).reshape(8, 2) / 10
    spatial_error = torch.sin(torch.arange(16.0)).reshape(8, 2) / 10
    members = torch.stack(
        (
            reference + 0.03 + spatial_error,
            reference - 0.07 + 0.6 * spatial_error,
            reference + 0.09 - 0.8 * spatial_error,
        )
    )
    selected = torch.tensor([0, 1])
    full_member = physical_ensemble_risks(members, reference, native_mass)[0]
    estimates = []
    for chosen in itertools.combinations(range(2, 8), 2):
        indices = torch.cat((selected, torch.tensor(chosen)))
        estimates.append(
            physical_ensemble_risks(
                members[:, indices],
                reference[indices],
                native_mass[indices],
                total_mass=native_mass.sum(),
                sampling="without_replacement",
                probabilities=torch.tensor([1.0, 1.0, 2 / 6, 2 / 6]),
            )[0]
        )
    source_files = (
        "src/phycoflow_reconstruction/coherence/families/topology/persistence.py",
        "src/phycoflow_reconstruction/coherence/physical_risk.py",
    )
    receipt = {
        "schema": "round8_support_shape_cpu_diagnostic_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Synthetic CPU tensor calculations only; no model, dataset, training or GPU access.",
        "dependencies": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "gudhi": gudhi.__version__,
            "torch_cpu_threads": torch.get_num_threads(),
        },
        "source_sha256": {
            path: hashlib.sha256((REPOSITORY / path).read_bytes()).hexdigest()
            for path in source_files
        },
        "prototype": {
            "grid_shape": [height, width],
            "windows": "Eight untruncated overlapping Gaussian windows: 4 x 2 centers over grid extent; sigma_x=14 and sigma_y=12 grid units.",
            "thresholds": thresholds.tolist(),
            "sigmoid_inverse_temperature": 8.0,
            "distance": "Euclidean feature norm divided by sqrt(24).",
            "translation_results": shifts,
            "ph_distance_translation_8_columns": ph_distance.item(),
            "finite_bars_h0_h1": [
                [len(item[dim].finite) for dim in (0, 1)] for item in diagrams
            ],
            "hard_left_right_half_translation_difference": (
                hard_half_features(truth) - hard_half_features(translated)
            ).abs().max().item(),
            "masked_values_invariance_max_difference": (
                features(truth, mask) - features(changed, mask)
            ).abs().max().item(),
            "equal_features_distance": equal_distance.item(),
            "equal_features_gradient": equal_feature.grad.tolist(),
            "saturated_features_finite": torch.isfinite(saturated_features).item(),
            "saturated_gradient_max": saturated.grad.abs().max().item(),
        },
        "quadrature": {
            "native_points": 8,
            "selected_points": 2,
            "complement_points": 2,
            "generated_members": 3,
            "complement_subsets_enumerated": len(estimates),
            "member_mse_exact_design_expectation_max_error": (
                full_member - torch.stack(estimates).mean(0)
            ).abs().max().item(),
            "limitation": "Endpoint values are held fixed before selecting support; this is arithmetic evidence, not model support-invariance or physical efficacy.",
        },
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
