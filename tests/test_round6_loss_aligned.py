import json
from types import SimpleNamespace

import pytest
import torch

from phycoflow_reconstruction.coherence.relation_queries import QueryBank
from phycoflow_reconstruction.contracts import FamilyResult, TermResult
from phycoflow_reconstruction.evaluation import loss_aligned as aligned


def test_matched_shards_bind_condition_draw_and_weight(monkeypatch, tmp_path):
    monkeypatch.setattr(aligned, "_COUNTS", {"train": 1, "validation": 1, "test": 1})
    bank = QueryBank(("a", "b"))

    def batch(name):
        return SimpleNamespace(
            sample_ids=(name,),
            query_coords=torch.zeros(1, 4096, 2),
            target_fields=torch.zeros(1, 4096, 2),
            query_valid_mask=torch.ones(1, 4096, dtype=torch.bool),
            obs_coords=torch.zeros(1, 1, 2),
            obs_values=torch.zeros(1, 1, 1),
            obs_field_ids=torch.zeros(1, 1, dtype=torch.long),
            obs_valid_mask=torch.ones(1, 1, dtype=torch.bool),
            obs_indices=torch.zeros(1, 1, dtype=torch.long),
            logical_shapes=((4096,),),
            metadata={"query_indices": torch.arange(4096)[None, :]},
        )

    batches = {split: [batch(split)] for split in aligned._COUNTS}
    provenance = {k: "a" * 64 for k in aligned._PROVENANCE}
    manifest = aligned.freeze_manifest(
        batches,
        bank,
        provenance=provenance,
        draw_seed=42,
        expected_sample_ids={split: [split] for split in batches},
    )
    manifest = json.loads(json.dumps(manifest))

    class Family:
        target_use = "paired_supervised"
        family_weight = 1.0

        def __init__(self, name):
            self.name = name
            self.conditional_draws = 1 if name == "topology" else 2

        def __call__(self, pred, target, *, coordinates, context):
            if self.name == "topology":
                selection = context["query_selection"]
                assert (selection.self_draws, selection.mutual_draws) == (4, 12)
            else:
                assert len(context["conditional_draws"]) == 1
            value = (pred - target).square().mean()
            return FamilyResult(
                {self.name + ".component": TermResult(value[None], value)}, value[None], value, {}
            )

    families = {
        name: Family(name) for name in ("global_distribution", "multiscale_spectrum", "topology")
    }
    model = torch.nn.Linear(1, 1)
    model.capabilities = SimpleNamespace(structured_grid_required=False)
    calls = []
    monkeypatch.setattr(aligned, "_without_target", lambda b: b)

    def rollout(model, batch, **kwargs):
        calls.append(kwargs["generator"].initial_seed())
        return torch.ones_like(batch.target_fields)

    monkeypatch.setattr(aligned, "differentiable_reconstruction", rollout)
    kwargs = {
        "model": model,
        "batches": batches["validation"],
        "families": families,
        "config": {"rollout": {"steps": 1, "solver": "x"}, "observation_consistency": {}},
        "manifest": manifest,
        "split": "validation",
        "checkpoint_sha256": "b" * 64,
        "output_dir": tmp_path,
        "query_bank": bank,
        "weight_identity_kind": "model_state_sha256",
    }
    path = aligned.evaluate_split(**kwargs)
    assert len(calls) == 2
    assert aligned.legacy_metrics(path, bank.field_ids)["endpoint_mse_model_units"] == {
        "a": 1,
        "b": 1,
    }
    aligned.evaluate_split(**kwargs)
    assert len(calls) == 2
    with pytest.raises(ValueError, match="incompatible cached"):
        aligned.evaluate_split(**{**kwargs, "checkpoint_sha256": "c" * 64})
    batches["validation"][0].obs_values += 1
    with pytest.raises(ValueError, match="condition identity"):
        aligned.evaluate_split(**kwargs)


def test_offline_report_matched_decision_and_guard(monkeypatch, tmp_path):
    from phycoflow_reconstruction.evaluation import loss_aligned_report as report

    monkeypatch.setattr(report, "_COUNTS", {"train": 1, "validation": 1, "test": 1})
    metrics = {
        name: name
        for name in (
            "scale_rho",
            "shape_mutual",
            "scale_self",
            "shape_self",
            "state",
            "scale",
            "shape",
        )
    }
    inputs = {}
    for split in ("train", "validation", "test"):
        inputs[split] = {}
        for model in ("SOURCE", "U6", "A6"):
            losses = dict.fromkeys(metrics, 1.0)
            if model != "SOURCE":
                losses["scale_rho"] -= 0.03 if model == "U6" else 0.06
            selection = {
                "queries": [
                    {"kind": "self", "multiplicity": 4, "query_id": ["self"]},
                    {"kind": "mutual", "multiplicity": 12, "query_id": ["mutual"]},
                ]
            }
            row = {
                "sample_id": "one",
                "losses": losses,
                "field_error_sums": [1.0],
                "field_error_counts": [1.0],
                "query_selection": selection,
            }
            path = tmp_path / f"{split}_{model}.json"
            path.write_text(
                json.dumps(
                    {
                        "split": split,
                        "manifest_sha256": "x",
                        "weight_identity_kind": "model_state_sha256",
                        "weight_sha256": model,
                        "rows": [row],
                    }
                )
            )
            inputs[split][model] = path.name
    spec = {
        "inputs": inputs,
        "field_names": ["CO"],
        "metadata": {"loss_balance_U100": {
            "families": {"multiscale_spectrum": {"relative_train_change": -0.0146917}},
            "validation": {"multiscale_spectrum.total": {"source": 0.999421, "delta": 0.0144531}},
            "coordination": {"equal_positive_steps_tolerance_1e_10": 1300},
        }},
        "spending": {"qualification_attempts": 24, "formal_updates": {"U6": 2600, "A6": 2600}},
        "decision_metrics": metrics,
        "scales": dict.fromkeys(metrics, 1),
        "costs": {
            "U6": {"update_seconds_per_step": 1, "total_wall_seconds": 10},
            "A6": {"update_seconds_per_step": 3, "total_wall_seconds": 30},
            "pipeline": {
                "reference_preparation_wall_seconds": 4,
                "qualification_wall_seconds": 5,
                "parallel_evaluation_and_validation_freeze_wall_seconds": 6,
                "evaluation_children": {model: {"wall_seconds": 5} for model in ("SOURCE", "U6", "A6")},
                "campaign_elapsed_through_evaluation_seconds": 60,
            },
        },
    }
    receipt = report.freeze_validation_choice(
        {model: tmp_path / inputs["validation"][model] for model in ("SOURCE", "U6", "A6")},
        metrics,
        spec["scales"],
        spec["costs"],
        output_path=tmp_path / "validation_choice.json",
    )
    assert receipt["validation_choice"] == "U6"
    spec["validation_receipt"] = "validation_choice.json"
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    result = report.build_report(spec_path)
    assert result["validation_choice"] == "U6"  # A's gain per update-second is lower.
    assert result["test_confirmation"] is True
    assert result["coverage"]["test"]["mutual_draws"] == 12
    report_text = (tmp_path / "postprocess" / "REPORT.md").read_text()
    assert "CO" in report_text and "2600" in report_text and "A6 − U6" in report_text
    assert "process times overlap" in report_text and "preparation failures" in report_text
    assert "Scale at U100" in report_text and "did not establish useful Scale learning" in report_text
    test_u = tmp_path / inputs["test"]["U6"]
    test_u_payload = json.loads(test_u.read_text())
    test_u_payload["rows"][0]["field_error_sums"] = [2.0]
    test_u.write_text(json.dumps(test_u_payload))
    failed = report.build_report(spec_path)
    assert failed["validation_choice"] == "U6" and failed["operational_choice"] == "SOURCE"
    file = tmp_path / inputs["test"]["A6"]
    payload = json.loads(file.read_text())
    payload["rows"][0]["sample_id"] = "wrong"
    file.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unmatched"):
        report.build_report(spec_path)


def test_real_selected_topology_leaves_are_executed_only_and_reducible(tmp_path):
    pytest.importorskip("gudhi")
    from test_topology_finite_primary import coords, family, provenance, witness

    from phycoflow_reconstruction.training.adaptive_coverage import AdaptiveCoverageSampler

    topology = family()
    prediction, target = witness()
    topology.fit_calibration(prediction, target, coordinates=coords(), provenance=provenance())
    bank = QueryBank(("CO", "T"))
    rows = []
    for seed in (7, 29):
        selection = AdaptiveCoverageSampler(bank, seed=seed).select(4, 12)
        result = topology(
            prediction, target, coordinates=coords(), context={"query_selection": selection}
        )
        losses, diagnostics, leaves = {}, {}, {}
        aligned._collect_family_result("topology", result, losses, diagnostics, leaves)
        assert set(losses) == {
            "topology.total",
            "topology.self.persistence",
            "topology.mutual.persistence",
        }
        assert leaves and all(".h" in key for key in leaves)
        losses.update({"global_distribution.total": 1.0, "multiscale_spectrum.total": 2.0})
        rows.append(
            {
                "sample_id": str(seed),
                "losses": losses,
                "field_error_sums": [1.0, 2.0],
                "field_error_counts": [1.0, 1.0],
                "query_leaf_diagnostics": leaves,
                "family_weights": {
                    name: {"weight": 1.0, "calibration_scale": 1.0}
                    for name in ("topology", "global_distribution", "multiscale_spectrum")
                },
            }
        )
    assert set(rows[0]["query_leaf_diagnostics"]) != set(rows[1]["query_leaf_diagnostics"])
    path = tmp_path / "reductions.json"
    path.write_text(json.dumps({"rows": rows}))
    metrics = aligned.legacy_metrics(path, bank.field_ids)
    assert set(metrics["coherence"]["families"]["topology"]["component_scalars"]) == {
        "topology.self.persistence",
        "topology.mutual.persistence",
    }
    assert metrics["coherence"]["families"]["topology"]["total"] == pytest.approx(
        sum(row["losses"]["topology.total"] for row in rows) / 2
    )
