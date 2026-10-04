import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
SPEC = importlib.util.spec_from_file_location("r3_report", ROOT / "scripts/visualization/upgrade_1002_r3_campaign_report.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def test_native_windows_pool_losses_instead_of_stochastic_ratios():
    rows = [{"epoch": epoch, "batches": 38, "fidelity/native/live_risk": live,
             "fidelity/native/source_risk": source}
            for epoch, live, source in ((25, .002, .001), (50, 1., 1.))]
    result = REPORT.native_windows(rows)
    assert result["rolling50"]["ratio_of_mean_losses"] == pytest.approx(1.002/1.001)
    assert result["rolling50"]["matched_batch_exposures"] == 76
    assert result["final25_minus_previous25"] == -1
    assert result["rolling50"]["ratio_of_mean_losses"] != pytest.approx(1.5)


def test_calibration_aggregates_only_TRAIN_components(tmp_path):
    path = tmp_path / "coherence_calibration.json"
    payload = {"calibration_batches": [
        {"R3_component_diagnostics": {"topology": {"split": "train", "components": {
            "native.raw": {"raw_value": value, "weighted_parameter_gradient_norm": value}}}}}
        for value in (2., 4.)]}
    path.write_text(json.dumps(payload))
    result = REPORT.load_calibration(path)
    assert result["components"]["native.raw"]["raw_value"] == 3
    assert result["components"]["native.raw"]["observed_batches"] == 2
    payload["calibration_batches"][0]["R3_component_diagnostics"]["topology"]["split"] = "test"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="TRAIN"):
        REPORT.load_calibration(path)
