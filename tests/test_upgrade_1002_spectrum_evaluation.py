"""Focused evaluator tests for the opt-in second-order covariance definition."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from phycoflow_reconstruction.coherence.families.cross_spectrum.basis import (
    build_graph_basis,
)
from phycoflow_reconstruction.coherence.families.cross_spectrum.family import (
    CrossSpectrumFamily,
)
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.evaluation.coherence_set import (
    CrossSpectrumAccumulator,
    _evaluation_family_config,
)
from phycoflow_reconstruction.training.run_store import file_sha256


def _coordinates(count: int = 24) -> torch.Tensor:
    generator = torch.Generator().manual_seed(10_020)
    coords = torch.rand(count, 2, generator=generator)
    coords[:, 1] = (coords[:, 1] + 0.031 * coords[:, 0].square()).clamp(0, 1)
    return coords


def _raw_config(coordinates: torch.Tensor) -> dict:
    basis = build_graph_basis(
        coordinates,
        k_neighbors=5,
        sigma=None,
        num_modes=6,
        band_names=("low", "high"),
        exclude_zero=True,
    )
    values = basis.eigenvalues.detach().cpu()
    split = int(torch.argmax(torch.diff(values))) + 1
    boundary = float((values[split - 1] + values[split]) / 2)
    intervals = [
        {"name": "low", "lower": float(values[0] - 1.0e-4), "upper": boundary},
        {"name": "high", "lower": boundary, "upper": float(values[-1] + 1.0e-4)},
    ]
    return {
        "definition": "second_order_blocks_v4",
        "weight": 1.0,
        "target_use": "paired_supervised",
        "units": "model_units",
        "fields": ["u", "v"],
        "pairs": [["u", "v"]],
        "reference_bank": {"enabled": False},
        "graph": {
            "k_neighbors": 5,
            "num_modes": 6,
            "exclude_zero": True,
            "bands": ["low", "high"],
            "band_intervals": intervals,
            "degeneracy_tolerance": 1.0e-6,
        },
        "minimum_ensemble_size": 32,
        "stabilization": {
            "relative_floor": 1.0e-6,
            "absolute_floor": 1.0e-12,
            "minimum_reference_band_fraction": 1.0e-8,
        },
        "components": {
            "self_spectrum": {"enabled": False, "weight": 0.0},
            "same_frequency": {"enabled": True, "weight": 1.0},
            "cross_frequency": {"enabled": True, "weight": 1.0},
            "band_energy": {"enabled": False, "weight": 0.0},
        },
    }


def _runtime_with_artifact(tmp_path: Path):
    coordinates = _coordinates()
    raw_config = _raw_config(coordinates)
    run_config = {
        "stage": "supervised",
        "coherence": {
            "families": {"cross_spectrum": raw_config},
            "compute_budget": {"batch_size": 32, "point_count": coordinates.shape[0], "query_seed": 81},
        },
    }
    data_spec = DataSpec(("u", "v"), ("1", "1"), 2, (4, 6))
    normalizer = FieldNormalizer.identity(2)
    settings = _evaluation_family_config(run_config, "cross_spectrum", data_spec.field_names)
    training_family = CrossSpectrumFamily(settings, data_spec, normalizer)
    basis = training_family._basis(coordinates.unsqueeze(0), torch.float32)

    generator = torch.Generator().manual_seed(52)
    reference_coefficients = torch.randn(32, basis.shape[1], 2, generator=generator)
    reference_panel = torch.einsum("nk,bkc->bnc", basis, reference_coefficients)
    panel_coordinates = coordinates.unsqueeze(0).expand(32, -1, -1).contiguous()
    training_family.freeze_reference_calibration(reference_panel, panel_coordinates)

    run_dir = tmp_path / "training_run"
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    artifact_path = run_dir / "artifacts" / "cross_spectrum_family.pt"
    torch.save(training_family.state_artifact(), artifact_path)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "coherence_family_state_sha256s": {
                    "cross_spectrum": file_sha256(artifact_path)
                }
            }
        ),
        encoding="utf-8",
    )

    dataset = SimpleNamespace(
        field_names=data_spec.field_names,
        data_spec=data_spec,
        normalizer=normalizer,
    )
    runtime = SimpleNamespace(
        config=run_config,
        dataset=dataset,
        device=torch.device("cpu"),
        seed=22,
        checkpoint_path=run_dir / "checkpoints" / "best.pt",
        run_dir=run_dir,
        coherence_artifact_run_dir=run_dir,
    )
    return runtime, training_family, coordinates, basis


def _field_snapshots(
    basis: torch.Tensor,
    coordinates: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7_300)
    reference_coefficients = torch.randn(count, basis.shape[1], 2, generator=generator)
    generated_coefficients = reference_coefficients + 0.18 * torch.randn(
        count, basis.shape[1], 2, generator=generator
    )
    reference_fields = torch.einsum("nk,bkc->bnc", basis, reference_coefficients)
    generated_fields = torch.einsum("nk,bkc->bnc", basis, generated_coefficients)
    return generated_fields, reference_fields


def test_v4_accumulator_restores_hashed_basis_and_frozen_calibration(tmp_path: Path) -> None:
    runtime, training_family, coordinates, _ = _runtime_with_artifact(tmp_path)
    accumulator = CrossSpectrumAccumulator.build(runtime, aggregation="pooled")

    assert accumulator.run_dir == runtime.run_dir.resolve()
    assert accumulator.family_artifact_sha256 == file_sha256(
        runtime.run_dir / "artifacts" / "cross_spectrum_family.pt"
    )
    assert accumulator.family.basis_sha256 == training_family.basis_sha256
    assert accumulator.family.calibration_sha256 == training_family.calibration_sha256
    torch.testing.assert_close(
        accumulator.family.calibrated_band_energies,
        training_family.calibrated_band_energies,
    )

    generated, reference = _field_snapshots(training_family.eigenvectors, coordinates, 2)
    accumulator.update(
        generated[:1], reference[:1], coordinates.unsqueeze(0), "snapshot-0"
    )
    changed_coordinates = coordinates.clone()
    changed_coordinates[0, 0] += 0.01
    try:
        accumulator.update(
            generated[1:2], reference[1:2], changed_coordinates.unsqueeze(0), "snapshot-1"
        )
    except ValueError as error:
        assert "coordinates" in str(error)
    else:
        raise AssertionError("evaluation must not replace the training graph basis")


def test_v4_source_comparison_restores_child_artifact_for_both_models(tmp_path: Path) -> None:
    child_runtime, child_family, _, _ = _runtime_with_artifact(tmp_path / "child")
    base_runtime, _, _, _ = _runtime_with_artifact(tmp_path / "base")
    base_runtime.coherence_artifact_run_dir = child_runtime.run_dir

    source_accumulator = CrossSpectrumAccumulator.build(base_runtime, aggregation="pooled")

    assert source_accumulator.run_dir == base_runtime.run_dir.resolve()
    assert source_accumulator.family_artifact_provenance["artifact_run_dir"] == str(
        child_runtime.run_dir.resolve()
    )
    assert source_accumulator.family_artifact_sha256 == file_sha256(
        child_runtime.run_dir / "artifacts" / "cross_spectrum_family.pt"
    )
    torch.testing.assert_close(
        source_accumulator.family.calibrated_band_energies,
        child_family.calibrated_band_energies,
    )


def test_v4_accumulator_reports_pooled_and_training_membership_separately(
    tmp_path: Path,
) -> None:
    runtime, _training_family, coordinates, basis = _runtime_with_artifact(tmp_path)
    accumulator = CrossSpectrumAccumulator.build(runtime, aggregation="training_aligned")
    generated, reference = _field_snapshots(basis, coordinates, 70)
    for index in range(generated.shape[0]):
        accumulator.update(
            generated[index : index + 1],
            reference[index : index + 1],
            coordinates.unsqueeze(0),
            f"sample-{index:03d}",
        )

    result = accumulator.finalize(
        tmp_path / "evaluation",
        split="validation",
        checkpoint_label="best",
        run_label="fixture",
        scale="linear",
    )
    destination = Path(result["directory"])
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    assert report["definition"] == "second_order_blocks_v4"
    assert report["statistic_scale"] == "squared_frobenius_difference_in_fixed_model_units"
    assert "coherence_score" not in report
    assert report["covariance_reports"]["pooled"]["sample_count"] == 70
    aligned = report["covariance_reports"]["training_aligned"]
    assert aligned["ensemble_count"] == 2
    assert aligned["sample_count"] == 64
    assert sum(map(len, aligned["sample_ids_by_ensemble"])) == 64
    block_payload = json.loads(
        (destination / "covariance_blocks_v4.json").read_text(encoding="utf-8")
    )
    dropped_ids = block_payload["reports"]["training_aligned"]["dropped_sample_ids"]
    assert len(dropped_ids) == 6
    assert not set(dropped_ids).intersection(
        sample_id for group in aligned["sample_ids_by_ensemble"] for sample_id in group
    )
    assert len(block_payload["reports"]["pooled"]["ensembles"][0]["sample_ids"]) == 70
    pooled_blocks = block_payload["reports"]["pooled"]["ensembles"][0]["diagnostics"]
    assert pooled_blocks["calibration_source"] == "frozen_reference_panel"
    assert block_payload["reports"]["pooled"]["ensembles"][0]["covariance_estimator"] == (
        "mergeable_sufficient_statistics"
    )
    assert block_payload["reports"]["training_aligned"]["ensembles"][0][
        "covariance_estimator"
    ] == "centered_coefficients_within_training_aligned_ensemble"
    assert pooled_blocks["same_frequency_blocks"][0]["generated_normalized_block"]
    assert pooled_blocks["cross_frequency_blocks"][0]["reference_normalized_block"]
    with (destination / "covariance_metrics_v4.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert {row["definition"] for row in rows} == {"second_order_blocks_v4"}
    assert {row["aggregation"] for row in rows} == {"pooled", "training_aligned"}
    assert (destination / "same_frequency_covariance_block_loss_v4.png").is_file()
    assert (destination / "cross_frequency_covariance_block_loss_v4.pdf").is_file()
    assert (destination / "covariance_band_energy_profile_v4.csv").is_file()


def test_v4_accumulator_rejects_missing_or_tampered_training_artifact(tmp_path: Path) -> None:
    runtime, _, _, _ = _runtime_with_artifact(tmp_path)
    artifact_path = runtime.run_dir / "artifacts" / "cross_spectrum_family.pt"
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tamper")
    try:
        CrossSpectrumAccumulator.build(runtime, aggregation="pooled")
    except ValueError as error:
        assert "artifact hash mismatch" in str(error)
    else:
        raise AssertionError("v4 evaluation must reject an unverified training artifact")


def test_v4_evaluator_does_not_fit_missing_evaluation_calibration(tmp_path: Path) -> None:
    runtime, family, _coordinates, _ = _runtime_with_artifact(tmp_path)
    family.calibrated_band_energies = torch.empty(0)
    family.calibration_ensemble_size = None
    family.calibration_sha256 = None
    artifact_path = runtime.run_dir / "artifacts" / "cross_spectrum_family.pt"
    torch.save(family.state_artifact(), artifact_path)
    manifest_path = runtime.run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["coherence_family_state_sha256s"]["cross_spectrum"] = file_sha256(artifact_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        CrossSpectrumAccumulator.build(runtime, aggregation="pooled")
    except ValueError as error:
        assert "no frozen calibration" in str(error)
    else:
        raise AssertionError("evaluation must not fit calibration from evaluation snapshots")


@pytest.mark.skipif(
    os.environ.get("PYCOFLOW_RUN_CUDA_FIXTURES") != "1" or not torch.cuda.is_available(),
    reason="bounded CUDA fixture is explicitly opt-in",
)
def test_v4_accumulator_cuda_projection_uses_restored_training_basis(tmp_path: Path) -> None:
    runtime, training_family, coordinates, basis = _runtime_with_artifact(tmp_path)
    runtime.device = torch.device("cuda:0")
    accumulator = CrossSpectrumAccumulator.build(runtime, aggregation="pooled")
    generated, reference = _field_snapshots(basis, coordinates, 32)
    device_coordinates = coordinates.to(runtime.device).unsqueeze(0)
    for index in range(generated.shape[0]):
        accumulator.update(
            generated[index : index + 1].to(runtime.device),
            reference[index : index + 1].to(runtime.device),
            device_coordinates,
            f"cuda-{index:02d}",
        )
    result = accumulator.finalize(
        tmp_path / "cuda_evaluation",
        split="validation",
        checkpoint_label="best",
        run_label="cuda_fixture",
        scale="linear",
    )
    torch.cuda.synchronize(runtime.device)
    assert accumulator.family.basis_sha256 == training_family.basis_sha256
    assert Path(result["signed_covariance_blocks_json"]).is_file()
