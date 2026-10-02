"""Public paired-report checks for the opt-in covariance-block evaluator."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from phycoflow_reconstruction.evaluation.explanatory_coherence import (
    render_saved_coherence_explanatory,
)
from phycoflow_reconstruction.evaluation.reconstruction_set import (
    _load_second_order_spectrum_artifacts,
    _render_posttraining_comparison,
    _SetEvaluationResult,
    _validate_second_order_spectrum_pair,
)

FIELD_NAMES = ["u", "v"]
FIELD_PAIRS = ["u–v"]
BANDS = ["low", "high"]
SAMPLE_IDS = ["sample-0", "sample-1", "sample-2", "sample-3"]


def _normalized_block(seed: float, *, complex_value: bool = False) -> dict[str, list[list[float]]]:
    real = [[0.18 + seed, -0.07], [0.11, 0.24 - seed / 2]]
    imaginary = [[0.0, -0.035], [0.035, 0.0]] if complex_value else [[0.0, 0.0], [0.0, 0.0]]
    return {"real": real, "imag": imaginary}


def _block_record(fields: list[int], bands: list[int], *, role: str, index: int) -> dict:
    reference = _normalized_block(0.0, complex_value=bands[0] != bands[1])
    generated = _normalized_block(
        (0.035 if role == "base" else 0.012) * (index + 1),
        complex_value=bands[0] != bands[1],
    )
    ref = np.asarray(reference["real"]) + 1j * np.asarray(reference["imag"])
    gen = np.asarray(generated["real"]) + 1j * np.asarray(generated["imag"])
    diff = gen - ref
    return {
        "fields": fields,
        "bands": bands,
        "frobenius_squared": float(np.square(np.abs(diff)).sum()),
        "reference_normalized_block": reference,
        "generated_normalized_block": generated,
        "reference_energy_fractions": [0.37, 0.41],
        "shared_floor": 2.5e-8,
    }


def _ensemble_record(sample_ids: list[str], *, role: str, ensemble_index: int) -> dict:
    same = [_block_record([0, 1], [band, band], role=role, index=ensemble_index + band) for band in range(2)]
    cross = [
        _block_record([0, 1], [0, 1], role=role, index=ensemble_index + 2),
        _block_record([0, 1], [1, 0], role=role, index=ensemble_index + 3),
    ]
    return {
        "ensemble_index": ensemble_index,
        "sample_count": len(sample_ids),
        "sample_ids": sample_ids,
        "reference_energy_fractions": [[0.7, 0.6], [0.3, 0.4]],
        "generated_energy_fractions": [[0.72, 0.58], [0.28, 0.42]],
        "diagnostics": {
            "same_frequency_eligible_blocks": len(same),
            "same_frequency_skipped_blocks": 0,
            "cross_frequency_eligible_blocks": len(cross),
            "cross_frequency_skipped_blocks": 0,
            "same_frequency_blocks": same,
            "cross_frequency_blocks": cross,
        },
    }


def _summary(records: list[dict]) -> dict:
    def component(name: str) -> dict:
        values = []
        for record in records:
            blocks = record["diagnostics"][f"{name}_blocks"]
            values.append(float(np.mean([block["frobenius_squared"] for block in blocks])))
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        return {
            "raw_block_loss_mean": mean,
            "raw_block_loss_sample_sd": sd,
            "raw_block_loss_by_ensemble": values,
            "per_field_pair": {
                "u–v": {
                    "raw_block_loss_mean": mean,
                    "raw_block_loss_sample_sd": sd,
                    "ensemble_values": values,
                }
            },
        }

    return {
        "ensemble_count": len(records),
        "sample_count": sum(record["sample_count"] for record in records),
        "sample_ids_by_ensemble": [record["sample_ids"] for record in records],
        "components": {
            "same_frequency": component("same_frequency"),
            "cross_frequency": component("cross_frequency"),
        },
        "weighted_raw_family_loss": 0.0,
    }


def _report_and_blocks(*, role: str) -> tuple[dict, dict]:
    pooled = [_ensemble_record(SAMPLE_IDS, role=role, ensemble_index=0)]
    aligned = [
        _ensemble_record(SAMPLE_IDS[:2], role=role, ensemble_index=0),
        _ensemble_record(SAMPLE_IDS[2:], role=role, ensemble_index=1),
    ]
    settings = {"definition": "second_order_blocks_v4", "graph": {"bands": BANDS}}
    graph = {
        "query_policy": "fixed_shared",
        "query_seed": 41,
        "query_point_count": 4,
        "query_indices": [1, 3, 4, 6],
        "geometry_sha256": "geometry-hash",
        "k_neighbors": 2,
        "resolved_sigma": 0.27,
        "num_modes": 4,
        "eigenvalues": [0.1, 0.2, 0.8, 0.9],
        "band_intervals": [{"name": "low"}, {"name": "high"}],
        "band_mode_ids": [0, 0, 1, 1],
    }
    calibration = {
        "source": "frozen_training_reference_panel",
        "sha256": "calibration-hash",
        "ensemble_size": 32,
        "band_energies": [[1.0, 1.2], [0.7, 0.9]],
        "relative_floor": 1.0e-6,
        "absolute_floor": 1.0e-10,
        "minimum_reference_band_fraction": 1.0e-5,
    }
    training_artifact = {
        "basis_sha256": "basis-hash",
        "family_artifact_sha256": "artifact-hash",
        "source_checkpoint_sha256": "source-checkpoint-hash",
        "geometry_sha256": "geometry-hash",
    }
    covariance_estimator = {
        "definition": "centered sample covariance of LINEAR graph coefficients",
        "denominator": "B-1",
        "mask_source": "frozen_training_reference_band_energy_fractions",
        "reference_denominator": "sqrt(reference energy)+floor",
    }
    report = {
        "family": "cross_spectrum",
        "version": "4",
        "definition": "second_order_blocks_v4",
        "target_use": "paired_supervised",
        "units": "model_units",
        "field_names": FIELD_NAMES,
        "field_pairs": FIELD_PAIRS,
        "bands": BANDS,
        "family_weight": 1.0,
        "component_weights": {"same_frequency": 1.0, "cross_frequency": 1.0},
        "settings": settings,
        "aggregation": "training_aligned",
        "selected_sample_ids": SAMPLE_IDS,
        "graph": graph,
        "calibration": calibration,
        "training_artifact": training_artifact,
        "covariance_estimator": covariance_estimator,
        "covariance_reports": {
            "pooled": _summary(pooled),
            "training_aligned": _summary(aligned),
        },
        "artifacts": {
            "signed_covariance_blocks_json": "covariance_blocks_v4.json",
            "covariance_metrics_csv": "covariance_metrics_v4.csv",
            "covariance_band_energy_profile_csv": "covariance_band_energy_profile_v4.csv",
            "figures": {},
        },
    }
    blocks = {
        "definition": "second_order_blocks_v4",
        "field_names": FIELD_NAMES,
        "band_names": BANDS,
        "basis_sha256": "basis-hash",
        "calibration_sha256": "calibration-hash",
        "primary_aggregation": "training_aligned",
        "reports": {
            "pooled": {"summary": _summary(pooled), "ensembles": pooled},
            "training_aligned": {"summary": _summary(aligned), "ensembles": aligned},
        },
    }
    return report, blocks


def _write_cross_spectrum_payload(output_dir: Path, *, role: str) -> None:
    cross_dir = output_dir / "coherence" / "cross_spectrum"
    cross_dir.mkdir(parents=True)
    report, blocks = _report_and_blocks(role=role)
    (cross_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (cross_dir / "covariance_blocks_v4.json").write_text(json.dumps(blocks), encoding="utf-8")
    (cross_dir / "covariance_metrics_v4.csv").write_text("definition,loss\nsecond_order_blocks_v4,0.1\n")
    (cross_dir / "covariance_band_energy_profile_v4.csv").write_text(
        "definition,profile\nsecond_order_blocks_v4,0.1\n"
    )


def _set_result(output_dir: Path, *, role: str, figure_path: Path) -> _SetEvaluationResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "set_metrics.npz"
    np.savez(
        payload_path,
        per_field_relative_l2_physical=np.asarray(
            [[0.1, 0.15], [0.12, 0.19], [0.09, 0.13], [0.11, 0.17]]
        ),
        field_names=np.asarray(FIELD_NAMES),
        sample_ids=np.asarray(SAMPLE_IDS),
        split=np.asarray("test"),
    )
    manifest_path = output_dir / "sensor_manifest.json"
    manifest_path.write_text('{"manifest":"identical"}\n', encoding="utf-8")
    report_path = output_dir / "set_report.json"
    report_path.write_text("{}\n", encoding="utf-8")
    _write_cross_spectrum_payload(output_dir, role=role)
    run_dir = output_dir
    return _SetEvaluationResult(
        run_dir=run_dir,
        output_dir=output_dir,
        figure_path=figure_path,
        payload_path=payload_path,
        report_path=report_path,
        manifest_path=manifest_path,
        checkpoint_path=run_dir / "checkpoints" / "best.pt",
        checkpoint_label="best",
        run_label=role,
        split="test",
        sample_ids=tuple(SAMPLE_IDS),
        field_names=tuple(FIELD_NAMES),
        dataset_fingerprint="dataset-hash",
        generation_steps=1,
        evaluation_seed=42,
        coherence_outputs={},
        coherence_accumulators={},
    )


def test_v4_paired_comparison_preserves_raw_artifacts_and_explains_without_metrics_npz(
    tmp_path: Path,
) -> None:
    current_dir = tmp_path / "current"
    current = _set_result(current_dir, role="post_training", figure_path=current_dir / "set.png")
    with tempfile.TemporaryDirectory(prefix="temporary-source-", dir=current_dir) as temporary:
        base = _set_result(Path(temporary), role="base", figure_path=Path(temporary) / "set.png")
        _render_posttraining_comparison(
            current,
            base,
            coherence_families=("cross_spectrum",),
            statistic_scale="linear",
        )

    cross_dir = current_dir / "coherence" / "cross_spectrum"
    assert (cross_dir / "covariance_blocks_v4-base.json").is_file()
    assert (cross_dir / "covariance_metrics_v4-base.csv").is_file()
    assert (cross_dir / "covariance_band_energy_profile_v4-base.csv").is_file()
    base_report = json.loads((cross_dir / "report-base.json").read_text())
    post_report = json.loads((cross_dir / "report.json").read_text())
    assert base_report["artifacts"]["signed_covariance_blocks_json"] == "covariance_blocks_v4-base.json"
    assert post_report["paired_source_comparison"]["shared_raw_loss_limits"] == base_report[
        "paired_source_comparison"
    ]["shared_raw_loss_limits"]
    assert "coherence_score" not in post_report["paired_source_comparison"]
    comparison = json.loads((current_dir / "comparison_report.json").read_text())
    contract = comparison["matched_coherence_contracts"]["cross_spectrum"]
    assert contract["definition"] == "second_order_blocks_v4"
    assert contract["calibration_sha256"] == "calibration-hash"
    assert contract["reference_mask"]["minimum_reference_band_fraction"] == 1.0e-5
    assert contract["raw_block_loss_by_role_and_aggregation"]["base"]["pooled"][
        "same_frequency"
    ]["raw_block_loss_mean"] > contract["raw_block_loss_by_role_and_aggregation"][
        "post_training"
    ]["pooled"]["same_frequency"]["raw_block_loss_mean"]
    assert comparison["shared_axis_limits"]["cross_spectrum"][
        "raw_covariance_block_loss_by_component"
    ]["same_frequency"][0] == 0.0
    assert (cross_dir / "same_frequency_covariance_block_loss_v4.png").is_file()
    assert (cross_dir / "same_frequency_covariance_block_loss_v4-base.png").is_file()
    assert (cross_dir / "covariance_band_energy_profile_v4.png").is_file()
    assert (cross_dir / "covariance_band_energy_profile_v4-base.png").is_file()
    assert base_report["paired_source_comparison"][
        "shared_covariance_energy_fraction_limits"
    ] == post_report["paired_source_comparison"]["shared_covariance_energy_fraction_limits"] == [
        0.0,
        1.0,
    ]

    assert not (cross_dir / "metrics.npz").exists()
    explanatory_report = render_saved_coherence_explanatory(
        current_dir, families=("cross_spectrum",)
    )
    assert explanatory_report == current_dir / "coherence" / "explanatory_report.json"
    explanatory_path = cross_dir / "signed_covariance_blocks_explanatory_v4.png"
    assert explanatory_path.is_file()
    updated_comparison = json.loads((current_dir / "comparison_report.json").read_text())
    assert updated_comparison["artifacts"]["paired_explanatory_cross_spectrum"][
        "signed_covariance_blocks_v4"
    ] == str(explanatory_path.relative_to(current_dir))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("query", "query_indices"),
        ("basis", "basis hash"),
        ("band", "band names"),
        ("mask", "reference masks"),
        ("normalization", "normalization"),
        ("membership", "memberships"),
    ],
)
def test_v4_paired_comparison_rejects_contract_mismatches(
    mutation: str, message: str
) -> None:
    base_report, base_blocks = _report_and_blocks(role="base")
    post_report, post_blocks = _report_and_blocks(role="post_training")
    if mutation == "query":
        post_report["graph"]["query_indices"] = [1, 2, 4, 6]
    elif mutation == "basis":
        post_blocks["basis_sha256"] = "other-basis"
    elif mutation == "band":
        post_blocks["band_names"] = ["different", "high"]
    elif mutation == "mask":
        post_blocks["reports"]["training_aligned"]["ensembles"][0]["diagnostics"][
            "same_frequency_blocks"
        ][0]["reference_energy_fractions"] = [0.5, 0.5]
    elif mutation == "normalization":
        post_report["covariance_estimator"]["denominator"] = "B"
    elif mutation == "membership":
        post_report["covariance_reports"]["training_aligned"]["sample_ids_by_ensemble"][0][0] = "other"
    with pytest.raises(ValueError, match=message):
        _validate_second_order_spectrum_pair(
            base_report, post_report, base_blocks, post_blocks
        )


def test_v4_artifact_loader_requires_versioned_signed_payload(tmp_path: Path) -> None:
    _write_cross_spectrum_payload(tmp_path, role="base")
    report, blocks, *_ = _load_second_order_spectrum_artifacts(tmp_path)
    assert report["definition"] == blocks["definition"] == "second_order_blocks_v4"
