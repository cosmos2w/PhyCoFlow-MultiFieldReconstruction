import json

from PIL import Image

from phycoflow_reconstruction.training.coverage_history import render_coverage_history


def test_saved_history_plot_with_zero_duals_and_matched_validation(tmp_path):
    logs = tmp_path / "logs"
    artifacts = tmp_path / "artifacts"
    logs.mkdir()
    artifacts.mkdir()
    (artifacts / "fidelity_budget_reference.json").write_text(
        json.dumps({"reference": {"field_names": ["a", "b"], "upper_bounds": [2.0, 4.0]}})
    )
    rows = [
        {"epoch": 1, "field_risk": {"a": 1.0, "b": 4.0}, "multipliers_after": [0.0, 0.0]},
        {"epoch": 2, "field_risk": {"a": 2.0, "b": 2.0}, "multipliers_after": [0.0, 0.2]},
    ]
    (logs / "full_updates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    paths = render_coverage_history(tmp_path)
    assert all(path.is_file() and path.stat().st_size > 100 for path in paths)
    assert paths[0].read_bytes().startswith(b"%PDF")
    with Image.open(paths[1]) as preview:
        preview.verify()
    (logs / "loss_aligned_validation.jsonl").write_text(
        json.dumps(
            {
                "epoch": 2,
                "step": 26,
                "metrics": {"endpoint_mse_model_units": {"a": 1.2, "b": 2.0}},
                "source_ratios": {"a": 1.2, "b": 1.0},
            }
        )
        + "\n"
    )
    paths = render_coverage_history(tmp_path)
    with Image.open(paths[1]) as preview:
        preview.verify()
