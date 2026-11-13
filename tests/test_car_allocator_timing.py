import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def driver():
    path = Path(__file__).resolve().parents[1] / "scripts/benchmarks/profile_car_allocator.py"
    spec = importlib.util.spec_from_file_location("allocator_timing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_warmup_and_balancing_remove_cold_and_linear_drift(driver):
    calls = []

    def timed(anchor, extra):
        calls.append((anchor, extra))
        cold = 80 if len(calls) == 1 else 0
        return 100 + cold + len(calls) * .01 + (0 if extra is None else .2 + .1 * extra)

    evidence = {}
    rows = driver.measure_bundle_costs(timed, 3, 3, evidence=evidence)
    assert len(calls) == 2 + 3 * 3 * 4
    assert evidence["warmup"][0]["seconds"] > 180
    assert evidence["status"] == "resolved"
    assert [row["incremental_seconds"] for row in rows] == pytest.approx([.2, .3, .4])
    assert rows[0]["blocks"][0]["samples"][1]["order"] == ["two", "one"]
    assert all(row["fixed_seconds"] > 0 for row in rows)


def test_noisy_positive_difference_rejected_with_all_samples(driver):
    calls = 0

    def timed(anchor, extra):
        nonlocal calls
        calls += 1
        if calls <= 2 or extra is None:
            return 10
        block = ((calls - 3) // 4) % 3
        return 10 + [.001, .01, .1][block]

    evidence = {}
    with pytest.raises(ValueError, match="incremental timing unresolved"):
        driver.measure_bundle_costs(timed, 2, 3, evidence=evidence)
    assert evidence["status"] == "failed"
    assert len(evidence["rows"]) == 2
    assert evidence["rows"][0]["incremental_seconds"] > 0
    assert evidence["elapsed_seconds"] >= 0


def test_budget_failure_preserves_completed_and_partial_samples(driver):
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == 6:
            raise TimeoutError("budget")

    evidence = {}
    with pytest.raises(TimeoutError):
        driver.measure_bundle_costs(lambda anchor, extra: 10., 2, 3,
                                    evidence=evidence, budget_check=check)
    assert len(evidence["warmup"]) == 2
    samples = evidence["rows"][0]["blocks"][0]["samples"]
    assert samples[0]["one"] == samples[0]["two"] == 10
    assert samples[1]["two"] == 10 and "one" not in samples[1]
    assert evidence["active_attempt"] == {"anchor": 1, "extra": None}
    assert evidence["error_type"] == "TimeoutError"


def test_nonpositive_fixed_and_too_few_blocks_are_rejected(driver):
    evidence = {}
    with pytest.raises(ValueError, match="fixed bundle timing unresolved"):
        driver.measure_bundle_costs(lambda anchor, extra: 1 if extra is None else 3,
                                    2, 3, evidence=evidence)
    assert evidence["fixed_seconds"] < 0
    with pytest.raises(ValueError, match="three balanced blocks"):
        driver.measure_bundle_costs(lambda a, b: 1, 2, 2)


def test_nonfinite_samples_keep_json_serializable_failure_evidence(driver):
    import json
    evidence = {}
    with pytest.raises(ValueError, match="positive and finite"):
        driver.measure_bundle_costs(lambda a, b: float("nan"), 2, 3, evidence=evidence)
    assert evidence["active_attempt"]["invalid_seconds"] == "nan"
    assert evidence["status"] == "failed"
    json.dumps(evidence, allow_nan=False)


def test_import_failure_still_writes_failed_receipt(driver, tmp_path, monkeypatch):
    import builtins
    import json
    import sys
    importlib.import_module("phycoflow_reconstruction.evaluation.adaptive_campaign")
    output = tmp_path / "profile.json"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test")
    monkeypatch.setenv("KEOPS_CACHE_FOLDER", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["profile", "--campaign", str(tmp_path), "--run-dir", str(tmp_path),
        "--source-family", "ffm", "--gpu-uuid", "GPU-test", "--output", str(output),
        "--research-costs", str(tmp_path / "costs.json"), "--spill-dir", str(tmp_path),
        "--policy-overhead-fraction", "0.01", "--complete-update-seconds", "1", "--max-seconds", "120"])
    original_import = builtins.__import__
    def fail_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("injected import error")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fail_torch)
    with pytest.raises(ImportError, match="injected"):
        driver.main()
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "failed" and receipt["error_type"] == "ImportError"
    assert receipt["elapsed_seconds"] >= 0
