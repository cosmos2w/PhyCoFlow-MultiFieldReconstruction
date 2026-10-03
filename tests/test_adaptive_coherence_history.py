"""Focused fixtures for opt-in adaptive coherence audit plots."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from phycoflow_reconstruction.training.coherence_history import (
    build_adaptive_coherence_figures,
    extract_adaptive_coherence_history,
    render_adaptive_coherence_history,
)


def _config() -> dict:
    return {
        "checkpointing": {"selection_metric": "coherence_with_fidelity"},
        "fidelity_controller": {
            "relative_budget_total": 0.05,
            "relative_budget_per_field": 0.05,
            "multiplier_max": 8.0,
        },
        "coherence": {
            "families": {
                "A": {"enabled": True},
                "B": {"enabled": True},
            }
        },
    }


def _before() -> dict:
    return {
        "mse_normalized": 1.0,
        "per_field_mse_normalized": {"u": 0.4, "v": 0.6},
        "coherence": {
            "families": {"A": {"total": 2.0}, "B": {"total": 4.0}}
        },
    }


def _training_rows() -> list[dict]:
    common = {
        "fidelity/total/source_risk": 100.0,
        "fidelity/u/source_risk": 40.0,
        "fidelity/v/source_risk": 60.0,
        "fidelity/total/violation": 0.2,
        "fidelity/total/ema_violation": 0.1,
        "fidelity/total/multiplier": 0.5,
        "fidelity/total/effective_primal_coefficient": 0.05,
        "fidelity/total/cap_saturated": False,
        "fidelity/u/violation": 0.1,
        "fidelity/u/ema_violation": 0.05,
        "fidelity/u/multiplier": 8.0,
        "fidelity/u/effective_primal_coefficient": 0.2,
        "fidelity/u/cap_saturated": True,
        "fidelity/u/cap_saturated_fraction": 0.75,
        "fidelity/v/violation": -0.1,
        "fidelity/v/ema_violation": -0.05,
        "fidelity/v/multiplier": 0.0,
        "fidelity/v/effective_primal_coefficient": 0.0,
        "fidelity/v/cap_saturated": False,
    }
    return [
        {
            "step": 10,
            "epoch": 1,
            **common,
            "fidelity/total/live_risk": 104.0,
            "fidelity/u/live_risk": 44.0,
            "fidelity/v/live_risk": 60.0,
            "gradient/A/A/cosine": 1.0,
            "gradient/A/B/cosine": None,
            "gradient/B/B/cosine": 1.0,
            "update/actual_dot/A": -0.3,
            "update/actual_dot/B": 0.2,
            "update/actual_dot/fidelity": -0.1,
            "update/actual_dot/combined_coherence": -0.05,
            "update/actual_dot/final": -0.15,
        },
        {
            "step": 20,
            "epoch": 2,
            **common,
            "fidelity/total/live_risk": 101.0,
            # A zero source-risk must not be turned into an infinite ratio.
            "fidelity/u/source_risk": 0.0,
            "fidelity/u/live_risk": 0.0,
            "fidelity/v/live_risk": 59.0,
            "gradient/A/A/cosine": 1.0,
            "gradient/A/B/cosine": None,
            "gradient/B/B/cosine": 1.0,
            "update/actual_dot/A": -0.1,
            "update/actual_dot/B": 0.1,
            "update/actual_dot/fidelity": -0.02,
            "update/actual_dot/combined_coherence": -0.03,
            "update/actual_dot/final": -0.05,
        },
    ]


def _validation_rows() -> list[dict]:
    return [
        {
            "step": 10,
            "metric": 1.05,
            "eligible": True,
            "failed_fidelity_fields": [],
            "family_source_normalized_scores": {"A": 0.9, "B": 1.1},
        },
        {
            "step": 20,
            "metric": 1.2,
            "eligible": False,
            "failed_fidelity_fields": ["u"],
            "family_source_normalized_scores": {"A": 0.8, "B": 1.2},
        },
    ]


def _selected() -> dict:
    return {
        "global_step": 20,
        "metric": 1.05,
        "eligible": True,
        "family_source_normalized_scores": {"A": 0.85, "B": 1.05},
        "failed_fidelity_fields": [],
    }


def test_adaptive_extraction_aligns_validation_and_keeps_undefined_values_null() -> None:
    training_rows = _training_rows()
    training_rows[1]["gradient/A/B/cosine"] = float("nan")
    after = {
        "mse_normalized": 1.03,
        "per_field_mse_normalized": {"u": float("nan"), "v": 0.61},
    }
    data = extract_adaptive_coherence_history(
        training_rows,
        _validation_rows(),
        before=_before(),
        selected=_selected(),
        after=after,
        config=_config(),
    )

    assert [row["x"] for row in data["training_records"]] == [10.0, 20.0]
    assert [row["x"] for row in data["validation_records"]] == [10.0, 20.0]
    assert data["family_order"] == ["A", "B"]
    assert data["constraints"] == ["total", "u", "v"]
    assert data["training_records"][0]["fidelity"]["total"]["relative_change"] == pytest.approx(0.04)
    assert data["training_records"][1]["fidelity"]["u"]["relative_change"] is None
    assert data["validation_records"][0]["family_source_normalized_scores"] == {"A": 0.9, "B": 1.1}
    assert [row["eligible"] for row in data["validation_records"]] == [True, False]
    assert data["gradient_cosine_matrix"]["values"]["A"]["B"] is None
    assert data["training_records"][0]["fidelity"]["u"]["cap_saturated_fraction"] == 0.75
    assert data["evaluation_endpoints"]["after"]["per_field_mse_normalized"]["u"] is None


def test_partial_epoch_windows_keep_distinct_run_steps_and_selected_marker() -> None:
    training, validation = _training_rows(), _validation_rows()
    for step, train, valid in zip((3, 38), training, validation):
        train.update(step=step, epoch=1)
        valid.update(step=step, epoch=1)
    data = extract_adaptive_coherence_history(
        training, validation, before=_before(),
        selected={**_selected(), "global_step": 38}, config=_config(),
    )
    assert [row["x"] for row in data["training_records"]] == [3., 38.]
    assert [row["x"] for row in data["validation_records"]] == [3., 38.]
    assert data["selected"]["x"] == 38.
    assert data["x_label"] == "Optimizer updates (run step)"


def test_adaptive_figures_show_fidelity_limits_selector_labels_and_adamw_sign() -> None:
    data = extract_adaptive_coherence_history(
        _training_rows(), _validation_rows(), before=_before(), selected=_selected(), config=_config()
    )
    figures = build_adaptive_coherence_figures(data, plt)
    risk_axis = figures["fidelity_risks"].axes[0]
    selector_axis = figures["selector_trajectories"].axes[0]
    matrix_axis, dot_axis = figures["gradient_geometry"].axes[:2]

    risk_labels = {line.get_label() for line in risk_axis.lines}
    assert "Total +5% limit" in risk_labels
    assert "Per-field +5% limit" in risk_labels
    assert "pre-step" in risk_axis.get_title()
    assert selector_axis.get_ylabel() == "Raw family total · candidate / source"
    assert selector_axis.get_xlabel() == "Optimizer updates (run step)"
    assert any(collection.get_offsets().shape[0] for collection in selector_axis.collections)
    assert any(text.get_text() == "—" for text in matrix_axis.texts)
    assert "Negative predicts local first-order decrease" in {
        text.get_text() for text in dot_axis.texts
    }
    assert "post-step" in figures["controller_state"].axes[1].get_title()
    assert "pre-step" in figures["controller_state"].axes[2].get_title()
    assert {line.get_label() for line in dot_axis.lines} >= {
        "A", "B", "fidelity", "combined_coherence", "final"
    }

    for figure in figures.values():
        plt.close(figure)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _write_run(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    config = """checkpointing:\n  selection_metric: coherence_with_fidelity\nfidelity_controller:\n  relative_budget_total: 0.05\n  relative_budget_per_field: 0.05\n  multiplier_max: 8.0\ncoherence:\n  families:\n    A:\n      enabled: true\n    B:\n      enabled: true\n"""
    (root / "resolved_config.yaml").write_text(config, encoding="utf-8")
    _write(root / "evaluation" / "before.json", _before())
    _write(root / "evaluation" / "selected.json", _selected())
    _write(
        root / "evaluation" / "after.json",
        {"mse_normalized": 1.03, "per_field_mse_normalized": {"u": 0.41, "v": 0.62}},
    )
    for filename, rows in (
        ("history.jsonl", _training_rows()),
        ("coherence_validation.jsonl", _validation_rows()),
    ):
        path = root / "metrics" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def test_opt_in_renderer_and_report_entrypoint_write_run_local_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    direct = render_adaptive_coherence_history(run_dir, output_dir=tmp_path / "direct")
    assert direct is not None
    assert len(direct["figures"]) == 4
    for formats in direct["figures"].values():
        assert set(formats) == {"png", "pdf", "svg"}
        assert all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in formats.values())
    summary = json.loads(Path(direct["summary"]).read_text(encoding="utf-8"))
    assert summary["eligible_candidate_count"] == 1
    assert summary["ineligible_candidate_count"] == 1
    assert summary["actual_adamw_gradient_dot"]["sign_interpretation"].startswith("negative dot")

    script = Path(__file__).parents[1] / "scripts" / "visualization" / "coherence_posttraining_report.py"
    spec = importlib.util.spec_from_file_location("coherence_posttraining_report_adaptive", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.generate_report(run_dir, output_dir=tmp_path / "report", include_adaptive_history=True)
    assert report["adaptive_history"] is not None
    assert "adaptive_coherence_history.json" in report["generated"]
    persisted = json.loads((tmp_path / "report" / "coherence_report.json").read_text(encoding="utf-8"))
    assert persisted["adaptive_history"]["summary"] == report["adaptive_history"]["summary"]


def test_historical_topology_selector_does_not_trigger_adaptive_outputs(tmp_path: Path) -> None:
    _write(tmp_path / "evaluation" / "selected.json", {"topology_selection": {"eligible": True}})
    assert render_adaptive_coherence_history(tmp_path, output_dir=tmp_path / "visualization") is None
    assert not (tmp_path / "visualization").exists()
