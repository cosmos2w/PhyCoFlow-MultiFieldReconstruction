import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from scripts.analysis.car_round8_cached_summary import digest
from scripts.analysis.car_round8_j3_action_report import launch_bindings, summarize
from scripts.analysis.car_round8_position_summary import position_conclusion, summarize as summarize_position


def receipt():
    common = {"members": 3, "sample_ids": ["sample_1", "sample_2"],
              "query_sha256": "a", "selected_sha256": "b", "complement_sha256": "c",
              "first_generator_sha256": "d", "field_risk_before": [1., 2.],
              "multiplier": [.1, .1],
              "family_row_norms": {"topology": 2.5}, "weighted_field_row_norm": 1.5,
              "clip_input_norm": 4., "clip_limit": 1., "adam_displacement_norm": .01,
              "forward_seconds": 2., "parameter_rows_seconds": 3., "optimizer_seconds": 4.,
              "component_diagnostic_seconds": 1., "finite_difference_seconds": 5.,
              "peak_allocated_bytes": 10 * 2**30, "peak_increment_bytes": 6 * 2**30,
              "component_diagnostic_peak_allocated_bytes": 11 * 2**30}
    baseline = {**common, "package": "PH_Only", "shape_component_row_norms": {"ph": 2.5},
                "field_risk_after": [.9, 2.2], "field_risk_change": [-.1, .2]}
    position = {**common, "package": "PH_Plus_Position", "shape_component_row_norms": {"ph": 1.5, "position": .7},
                "field_risk_after": [.95, 1.8], "field_risk_change": [-.05, -.2]}
    return {"schema": "car_round8_j3_position_action_v1", "status": "completed",
            "field_order": ["T", "CO"], "actions": [baseline, position], "learning_rate": 5e-6,
            "metric": "physical_member_mse_model_units", "device": "cuda:0", "device_name": "fixture",
            "scope": "one matched batch", "scale_objective": "joint_anchor", "scale_joint_kappa": 1.0,
            "pi_state": "unchanged", "adam_state": "fresh"}


def test_summary_separates_costs_and_validates_the_matched_measurement():
    source = receipt()
    result = summarize(source)
    actions = result["actions"]
    assert actions["PH_Only"]["main_action_seconds"] == 9.
    assert actions["PH_Only"]["times"]["component_diagnostic_seconds"] == 1.
    assert actions["PH_Only"]["times"]["finite_difference_seconds"] == 5.
    assert actions["PH_Only"]["clip_factor"] == pytest.approx(1 / (4 + 1e-6))
    assert actions["PH_Only"]["field_risk_change_percent"] == pytest.approx([-10., 10.])
    source["actions"][1]["sample_ids"] = ["other"]
    with pytest.raises(ValueError, match="identity"):
        summarize(source)


def test_reference_supplies_field_order_only_with_matching_hash(tmp_path):
    source = receipt()
    source.pop("field_order")
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"reference": {"field_names": ["T", "CO"]}}))
    source["reference_sha256"] = digest(reference)
    assert summarize(source, reference_path=reference)["field_order"] == ["T", "CO"]
    with pytest.raises(ValueError, match="field order"):
        summarize(source)
    reference.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        summarize(source, reference_path=reference)


def test_report_produces_figure_manifest_without_duplicate_section_embeds(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "receipt.json"
    source.write_text(json.dumps(receipt()))
    output = tmp_path / "report"
    env = {**os.environ, "MPLCONFIGDIR": str(tmp_path / "matplotlib")}
    subprocess.run([sys.executable, str(root / "scripts/analysis/car_round8_j3_action_report.py"),
                    "--receipt", str(source), "--output", str(output),
                    "--first-action-cold-start"], cwd=root, env=env, check=True)
    captions = json.loads((output / "figure_captions.json").read_text())
    text = (output / "section.md").read_text()
    assert len(captions) == 3 and all("\n" not in value for value in captions.values())
    for name in captions:
        image, pdf = output / "figures" / f"{name}.png", output / "figures" / f"{name}.pdf"
        with Image.open(image) as decoded:
            decoded.verify()
        assert pdf.read_bytes().startswith(b"%PDF-")
    assert "![" not in text and "(figures/" not in text
    assert "Shape vector row is the sum of the PH and position contribution vectors" in text
    assert "Extra component reverse" in text and "do not measure an epoch" in text and "cold-start" in text
    assert "cold-start" in captions["j3_action_cost"]


def test_launch_binds_frozen_source_and_reference_law(tmp_path):
    source = receipt()
    action_file = tmp_path / "action.json"
    code = tmp_path / "code"
    action_script = code / "scripts/qualification/car_round8_j3_position_action.py"
    action_script.parent.mkdir(parents=True)
    action_script.write_text("# immutable action fixture\n")
    manifest = code / "source_manifest.json"
    manifest.write_text(json.dumps({"scripts/qualification/car_round8_j3_position_action.py": digest(action_script)}))
    config = tmp_path / "config.yaml"
    config.write_text("fixture: true\n")
    position = tmp_path / "position.pt"
    position.write_bytes(b"fixture position")
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"reference": {"field_names": ["T", "CO"]},
        "settings": {"metric": source["metric"], "calibration_draws": 2},
        "risk_reference_panel": {"members": 2, "sample_count": 2,
            "sample_ids": ["sample_1", "sample_2"], "reference_law": "training_selected_complement_v1"}}))
    source.update(config_sha256=digest(config), position_sha256=digest(position), reference_sha256=digest(reference))
    action_file.write_text(json.dumps(source))
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({"result": str(action_file), "code_root": str(code),
        "source_manifest_sha256": digest(manifest), "command": ["python", str(action_script),
        "--config", str(config), "--position-artifact", str(position), "--reference", str(reference)]}))
    _, bound, calibration = launch_bindings(launch, action_file, source)
    assert bound["frozen_action_script"]["sha256"] == digest(action_script)
    assert calibration["members"] == 2 and calibration["train_conditions"] == 2
    position.write_bytes(b"altered")
    with pytest.raises(ValueError, match="position_artifact"):
        launch_bindings(launch, action_file, source)


def test_position_conclusion_follows_actual_signs_and_intervals():
    def row(delta, bounds):
        return {"paired_source_deltas": {"JointAnchor500": {
            "mean": [0., 0., delta], "interval_95": [[0., 0.], [0., 0.], bounds]}}}
    values = {"ph_ARS": row(-.4, [-.5, -.3]),
              "location_unit_matched_ARS": row(.1, [-.08, .27]),
              "native_location_unit_matched_ARS": row(.09, [-.08, .27])}
    first = position_conclusion(values)
    assert "PH fair score decreased" in first and first.count("includes zero") == 2
    values["ph_ARS"] = row(.4, [.3, .5])
    values["location_unit_matched_ARS"] = row(-.2, [-.3, -.1])
    values["native_location_unit_matched_ARS"] = row(.2, [.1, .3])
    second = position_conclusion(values)
    assert "PH fair score increased" in second and "below zero" in second and "above zero" in second


def test_position_summary_rejects_unmatched_record_identity():
    value = {"status": "completed", "split": "validation", "members": 4, "test_locked": True,
             "sample_ids": [f"sample_{i}" for i in range(100)],
             "records": [{"sample_id": f"sample_{i}"} for i in range(100)]}
    value["records"][10]["sample_id"] = value["records"][9]["sample_id"]
    with pytest.raises(ValueError, match="sample IDs"):
        summarize_position(value)


def test_position_ph_interval_matches_direct_paired_frame_interval():
    from scripts.analysis.car_round8_relation_intervals import paired_triplet

    ids = [f"sample_{i}" for i in range(100)]
    records = []
    for i, sid in enumerate(ids):
        models = {}
        for name, factor in (("SOURCE", 1.), ("JointAnchor100", 1.1), ("JointAnchor500", 1.2)):
            attraction = factor * (2. + i / 100.)
            separation = factor * (.4 + i / 200.)
            values = [attraction / 2, separation / 2, (attraction - separation) / 2]
            models[name] = {kind: {observer: values for observer in (
                "ph_ARS", "location_unit_matched_ARS", "combined_ARS", "native_location_unit_matched_ARS")}
                for kind in ("self", "mutual")}
        records.append({"sample_id": sid, "models": models})
    readout = {"status": "completed", "split": "validation", "members": 4, "test_locked": True,
               "sample_ids": ids, "records": records}
    actual = summarize_position(readout)["ph_ARS"]["paired_source_deltas"]["JointAnchor500"]
    def triplets(model):
        return [[sum(r["models"][model][k]["ph_ARS"][j] for k in ("self", "mutual"))
                 for j in range(3)] for r in records]
    direct = paired_triplet(triplets("SOURCE"), triplets("JointAnchor500"))
    np.testing.assert_allclose(actual["mean"], direct["mean"])
    np.testing.assert_allclose(actual["interval_95"], np.asarray(direct["interval_95"]).T)
