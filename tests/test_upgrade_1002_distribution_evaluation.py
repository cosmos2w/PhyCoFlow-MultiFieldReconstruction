"""Artifact-grounded evaluation reporting for marginal/copula v2."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from phycoflow_reconstruction.coherence import build_coherence_family
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.evaluation.coherence_set import (
    GlobalDistributionAccumulator,
    _evaluation_family_config,
)
from phycoflow_reconstruction.evaluation.reconstruction_set import (
    _SetEvaluationResult,
    evaluate_reconstruction_set,
)
from phycoflow_reconstruction.training.run_store import file_sha256


def _v2_settings(*, pairwise_diagnostics: bool = False) -> dict:
    return {
        "definition": "marginal_copula_v2",
        "target_use": "paired_supervised",
        "units": "model_units",
        "fields": ["u", "v"],
        "components": {
            "self": {"enabled": True, "weight": 1.0},
            "mutual": {"enabled": False, "weight": 0.0},
            "cross": {
                "enabled": True,
                "weight": 1.0,
                "directions": 8,
                "seed": 21002,
                "include_axes": False,
                "qmc": True,
                "pairwise_diagnostics": pairwise_diagnostics,
                "copula": {
                    "canonicalizer": "quantile_landmark_smooth_cdf",
                    "landmarks": 8,
                    "bandwidth": 0.1,
                    "scale_floor": 1e-6,
                    "chunk_size": 8,
                },
                "tail": {
                    "alpha": 0.25,
                    "rho": 0.25,
                    "temperature": 1e-3,
                    "eta_tolerance": 1e-7,
                    "eta_max_iterations": 80,
                },
            },
        },
    }


def _runtime(tmp_path: Path, *, pairwise_diagnostics: bool = False):
    fields = ("u", "v")
    spec = DataSpec(fields, ("1", "1"), 2, (8,))
    normalizer = FieldNormalizer.identity(2)
    dataset = SimpleNamespace(
        field_names=fields,
        data_spec=spec,
        normalizer=normalizer,
    )
    settings = _v2_settings(pairwise_diagnostics=pairwise_diagnostics)
    config = {
        "coherence": {
            "families": {"global_distribution": settings},
            "compute_budget": {"point_count": 8, "query_seed": 100045},
        }
    }
    resolved = _evaluation_family_config(config, "global_distribution", fields)
    source_family = build_coherence_family(
        "global_distribution", resolved, spec, normalizer
    )
    artifact = source_family.state_artifact()
    directions_key = "components_by_key.cross_joint_copula_cvar.directions"
    artifact["state_dict"][directions_key] = torch.roll(
        artifact["state_dict"][directions_key], shifts=1, dims=0
    )

    run_dir = tmp_path / "source_run"
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    artifact_path = artifact_dir / "global_distribution_family.pt"
    torch.save(artifact, artifact_path)
    artifact_hash = file_sha256(artifact_path)
    manifest = {
        "source_run_id": "source-run-fixture",
        "coherence_family_state_sha256s": {
            "global_distribution": artifact_hash
        },
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    runtime = SimpleNamespace(
        config=config,
        dataset=dataset,
        device=torch.device("cpu"),
        checkpoint_path=run_dir / "checkpoints" / "best.pt",
        run_dir=run_dir,
        coherence_artifact_run_dir=run_dir,
    )
    return runtime, artifact, artifact_path, artifact_hash


def test_v2_accumulator_restores_frozen_artifact_and_reports_separate_exact_diagnostic(
    tmp_path,
):
    runtime, artifact, artifact_path, artifact_hash = _runtime(tmp_path)
    accumulator = GlobalDistributionAccumulator.build(runtime)
    restored = accumulator.family.components_by_key["cross_joint_copula_cvar"]
    expected_directions = artifact["state_dict"][
        "components_by_key.cross_joint_copula_cvar.directions"
    ]
    torch.testing.assert_close(restored.directions, expected_directions)
    assert accumulator.family.version == "2"

    torch.manual_seed(1002)
    for index in range(3):
        prediction = torch.randn(1, 16, 2)
        target = torch.randn(1, 16, 2)
        accumulator.update(
            prediction,
            target,
            torch.randn(1, 16, 2),
            f"snapshot-{index}",
        )

    result = accumulator.finalize(
        tmp_path / "evaluation",
        split="validation",
        checkpoint_label="best",
        run_label="source_run",
        scale="linear",
    )
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    with np.load(Path(result["directory"]) / "metrics.npz", allow_pickle=False) as data:
        assert data["marginal_per_field_w2"].shape == (3, 2)
        assert data["joint_copula_smooth_cvar"].shape == (3, 1)
        assert data["per_direction_w2"].shape == (3, 8)
        assert data["exact_midranks_projected_w2_diagnostic"].shape == (3, 1)
        assert data["pairwise_diagnostic_enabled"].item() is False
        assert data["pairwise_soft_copula_swd_diagnostic"].shape == (3, 0)
        np.testing.assert_allclose(
            data["family_total"][:, 0],
            data["weighted_component_totals"].sum(axis=1),
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            data["effective_direction_weights"].sum(axis=1),
            np.ones(3),
            rtol=0.0,
            atol=2e-5,
        )
        assert data["projection_direction_sha256"].item() == report["projection_bank"]["sha256"]

    assert report["definition"] == "marginal_copula_v2"
    assert report["objective"]["canonicalizer"]["name"] == "quantile_landmark_smooth_cdf"
    assert "not exactly invariant to nonlinear monotone transforms" in report["objective"]["canonicalizer"]["approximation"]
    assert report["exact_rank_diagnostic"]["gradient"].startswith("detached")
    assert report["objective"]["mean_tail_mixture"]["temperature_policy"] == (
        "fixed_configured_value; not source calibrated"
    )
    assert report["pairwise_diagnostics"]["enabled"] is False
    assert report["projection_bank"]["artifact_restored"] is True
    assert report["family_artifact"]["artifact_path"] == str(artifact_path.resolve())
    assert report["family_artifact"]["artifact_sha256"] == artifact_hash
    assert report["family_artifact"]["source_run_id"] == "source-run-fixture"
    assert Path(result["figures"]["direction_spectrum"]).is_file()
    assert Path(result["figures"]["tail_diagnostics"]).is_file()


def test_v2_accumulator_materializes_pairwise_diagnostics_only_when_requested(tmp_path):
    runtime, _, _, _ = _runtime(tmp_path, pairwise_diagnostics=True)
    accumulator = GlobalDistributionAccumulator.build(runtime)
    accumulator.update(
        torch.randn(1, 16, 2),
        torch.randn(1, 16, 2),
        torch.randn(1, 16, 2),
        "snapshot-0",
    )
    accumulator.finalize(
        tmp_path / "evaluation",
        split="test",
        checkpoint_label="best",
        run_label="source_run",
        scale="linear",
    )
    with np.load(
        tmp_path / "evaluation/coherence/global_distribution/metrics.npz",
        allow_pickle=False,
    ) as data:
        assert data["pairwise_diagnostic_enabled"].item() is True
        assert data["pairwise_raw_state_swd_diagnostic"].shape == (1, 1)
        assert data["pairwise_soft_copula_swd_diagnostic"].shape == (1, 1)
        assert data["pairwise_exact_midranks_swd_diagnostic"].shape == (1, 1)


def test_public_paired_v2_report_uses_child_family_for_live_source_and_retains_source_rows(
    tmp_path, monkeypatch
):
    runtime, _artifact, source_artifact_path, artifact_hash = _runtime(tmp_path)
    source_run = runtime.run_dir
    source_checkpoint = source_run / "checkpoints" / "best.pt"
    source_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    source_checkpoint.write_bytes(b"source checkpoint fixture")

    child_run = tmp_path / "child_run"
    child_artifact_dir = child_run / "artifacts"
    (child_run / "checkpoints").mkdir(parents=True)
    child_artifact_dir.mkdir(parents=True)
    (child_run / "checkpoints" / "best.pt").write_bytes(b"child checkpoint fixture")
    child_artifact_path = child_artifact_dir / "global_distribution_family.pt"
    shutil.copy2(source_artifact_path, child_artifact_path)
    child_artifact_hash = file_sha256(child_artifact_path)
    assert child_artifact_hash == artifact_hash

    child_config = {
        "stage": "post_training",
        "source_run": str(source_run),
        "source_checkpoint": str(source_checkpoint),
        "optimization": {"update_policy": "coherence_primal_dual"},
        "coherence": runtime.config["coherence"],
    }
    (child_run / "resolved_config.yaml").write_text(
        yaml.safe_dump(child_config, sort_keys=False), encoding="utf-8"
    )
    child_manifest = {
        "stage": "post_training",
        "parent_run": str(source_run),
        "source_checkpoint": str(source_checkpoint),
        "source_hashes": {"checkpoint": file_sha256(source_checkpoint)},
        "source_run_id": "child-run-fixture",
        "coherence_family_state_sha256s": {
            "global_distribution": child_artifact_hash
        },
    }
    (child_run / "run_manifest.json").write_text(
        json.dumps(child_manifest), encoding="utf-8"
    )

    calls = []
    source_rows = {}
    source_temp_dirs = []

    def fake_evaluate_once(
        run_dir,
        *,
        case_dir,
        split,
        checkpoint="best",
        output_dir_override=None,
        coherence_config_override=None,
        coherence_artifact_run_dir=None,
        evaluation_seed_override=None,
        weight_selection="configured",
        generation_steps=None,
        **kwargs,
    ):
        del case_dir, kwargs
        run_dir = Path(run_dir)
        calls.append(
            {
                "run_dir": run_dir,
                "weight_selection": weight_selection,
                "coherence_config_override": coherence_config_override,
                "coherence_artifact_run_dir": coherence_artifact_run_dir,
                "evaluation_seed_override": evaluation_seed_override,
                "output_dir_override": output_dir_override,
            }
        )
        evaluation_config = (
            {"coherence": coherence_config_override}
            if coherence_config_override is not None
            else child_config
        )
        artifact_run_dir = (
            Path(coherence_artifact_run_dir)
            if coherence_artifact_run_dir is not None
            else run_dir
        )
        evaluation_runtime = SimpleNamespace(
            config=evaluation_config,
            dataset=runtime.dataset,
            device=torch.device("cpu"),
            checkpoint_path=Path(checkpoint)
            if Path(checkpoint).is_absolute()
            else run_dir / "checkpoints" / f"{Path(checkpoint).stem}.pt",
            run_dir=run_dir,
            coherence_artifact_run_dir=artifact_run_dir,
        )
        accumulator = GlobalDistributionAccumulator.build(evaluation_runtime)
        generator = torch.Generator().manual_seed(170046)
        sample_ids = ("sample-0", "sample-1", "sample-2")
        for index, sample_id in enumerate(sample_ids):
            target = torch.randn(1, 16, 2, generator=generator)
            coordinates = torch.linspace(0.0, 1.0, 16).reshape(1, 16, 1).expand(-1, -1, 2)
            offset = 0.025 if run_dir == source_run else 0.04
            prediction = target + offset * (index + 1) + 0.01 * coordinates
            accumulator.update(prediction, target, coordinates, sample_id)

        output_dir = (
            Path(output_dir_override)
            if output_dir_override is not None
            else run_dir / "evaluation" / f"reconstruction_set_{split}_best"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        coherence_outputs = accumulator.finalize(
            output_dir,
            split=split,
            checkpoint_label="best",
            run_label=run_dir.name,
            scale="linear",
        )
        if run_dir == source_run:
            source_temp_dirs.append(Path(output_dir_override))
            with np.load(
                Path(coherence_outputs["directory"]) / "metrics.npz",
                allow_pickle=False,
            ) as payload:
                source_rows.update(
                    sample_ids=payload["sample_ids"].copy(),
                    family_total=payload["family_total"].copy(),
                    projection_directions=payload["projection_directions"].copy(),
                )

        relative_payload = output_dir / "relative_l2.npz"
        np.savez_compressed(
            relative_payload,
            per_field_relative_l2_physical=np.asarray(
                [[0.2, 0.3], [0.1, 0.25], [0.15, 0.22]], dtype=np.float64
            ),
            field_names=np.asarray(("u", "v")),
            sample_ids=np.asarray(sample_ids),
            split=np.asarray(split),
        )
        manifest_path = output_dir / "sensor_manifest.jsonl"
        manifest_path.write_text(
            json.dumps({"sample_ids": sample_ids, "geometry": "shared-grid"}) + "\n",
            encoding="utf-8",
        )
        run_report_path = output_dir / "evaluation_report.json"
        run_report_path.write_text(
            json.dumps({"trace": {"weight_selection": weight_selection}}),
            encoding="utf-8",
        )
        checkpoint_path = evaluation_runtime.checkpoint_path
        return _SetEvaluationResult(
            run_dir=run_dir,
            output_dir=output_dir,
            figure_path=output_dir / "relative_l2_violin.png",
            payload_path=relative_payload,
            report_path=run_report_path,
            manifest_path=manifest_path,
            checkpoint_path=checkpoint_path,
            checkpoint_label="best",
            run_label=run_dir.name,
            split=split,
            sample_ids=sample_ids,
            field_names=("u", "v"),
            dataset_fingerprint="fixture-dataset-geometry-hash",
            generation_steps=generation_steps or 4,
            evaluation_seed=evaluation_seed_override or 2027,
            coherence_outputs={"global_distribution": coherence_outputs},
            coherence_accumulators={"global_distribution": accumulator},
        )

    from phycoflow_reconstruction.evaluation import reconstruction_set

    monkeypatch.setattr(
        reconstruction_set, "_evaluate_reconstruction_set_once", fake_evaluate_once
    )
    output_figure = evaluate_reconstruction_set(
        child_run,
        case_dir=tmp_path,
        split="test",
        max_samples=3,
        coherence_families=("global_distribution",),
    )

    assert output_figure.is_file()
    assert len(calls) == 2
    assert calls[0]["weight_selection"] == calls[1]["weight_selection"] == "live"
    assert calls[1]["run_dir"] == source_run
    assert calls[1]["coherence_config_override"] == child_config["coherence"]
    assert calls[1]["coherence_artifact_run_dir"] == child_run
    assert calls[0]["evaluation_seed_override"] is None
    assert calls[1]["evaluation_seed_override"] == 2027
    assert source_temp_dirs and not source_temp_dirs[0].exists()

    output_dir = output_figure.parent
    global_dir = output_dir / "coherence" / "global_distribution"
    for stem in (
        "marginal_copula_v2_marginal_distributions",
        "marginal_copula_v2_direction_spectrum",
        "marginal_copula_v2_tail_diagnostics",
    ):
        assert (global_dir / f"{stem}.png").is_file()
        assert (global_dir / f"{stem}-base.png").is_file()
        assert (global_dir / f"{stem}-base.pdf").is_file()
        assert (global_dir / f"{stem}-base.svg").is_file()

    with np.load(global_dir / "metrics-base.npz", allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload["sample_ids"], source_rows["sample_ids"])
        np.testing.assert_array_equal(payload["family_total"], source_rows["family_total"])
        np.testing.assert_array_equal(
            payload["projection_directions"], source_rows["projection_directions"]
        )
    source_report = json.loads((global_dir / "report-base.json").read_text())
    assert source_report["artifacts"]["metrics_payload"] == "metrics-base.npz"
    assert source_report["paired_source_comparison"]["role"] == "base_source"
    current_report = json.loads((global_dir / "report.json").read_text())
    assert current_report["paired_source_comparison"]["role"] == "post_training"

    comparison = json.loads((output_dir / "comparison_report.json").read_text())
    contract = comparison["matched_coherence_contracts"]["global_distribution"]
    assert contract["definition"] == "marginal_copula_v2"
    assert contract["weight_selection"] == "live"
    assert contract["family_artifact"]["artifact_run_dir"] == str(child_run.resolve())
    assert contract["tail_parameters"]["temperature"] == runtime.config[
        "coherence"
    ]["families"]["global_distribution"]["components"]["cross"]["tail"]["temperature"]
    assert contract["projection_bank"]["sha256"] == source_report["projection_bank"]["sha256"]
    assert comparison["matched_inputs"]["dataset_fingerprint"] == "fixture-dataset-geometry-hash"
