"""Focused tests for the coherence post-training visualization report."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).parents[1] / "scripts" / "visualization" / "coherence_posttraining_report.py"
_SPEC = importlib.util.spec_from_file_location("coherence_posttraining_report", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
generate_report = _MODULE.generate_report
parse_run_payload = _MODULE.parse_run_payload


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _evaluation(multiplier: float) -> dict[str, object]:
    families = {}
    for index, family in enumerate(("global_distribution", "cross_spectrum", "topology"), 1):
        total = multiplier * index
        families[family] = {
            "total": total,
            "weighted_total": total * 0.5,
            "components": {f"{family}.component": total / 2},
            "diagnostics": {"epsilon": 1e-6, "bands": [1.0, 2.0]},
        }
    return {
        "mse_normalized": multiplier,
        "mean_relative_l2": multiplier * 2,
        "worst_field_relative_l2": multiplier * 3,
        "coherence": {"families": families},
    }


def test_parse_run_payload_extracts_family_ratios_and_history(tmp_path: Path) -> None:
    _write(tmp_path / "evaluation" / "before.json", _evaluation(2.0))
    _write(tmp_path / "evaluation" / "after.json", _evaluation(1.0))
    history = tmp_path / "metrics" / "history.jsonl"
    history.parent.mkdir(parents=True)
    history.write_text('{"data_grad_norm": 2.0}\n{"gradient_cosine": -0.2}\n')

    parsed = parse_run_payload(tmp_path)

    assert parsed["families"]["global_distribution"]["source_normalized"] == 0.5
    assert parsed["families"]["cross_spectrum"]["after_weighted"] == 1.0
    assert len(parsed["history"]) == 2
    assert parsed["families"]["topology"]["after_diagnostics"]["bands.mean"] == 1.5


def test_generate_report_writes_summary_inventory_and_available_family_figures(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "evaluation" / "before.json", _evaluation(2.0))
    _write(tmp_path / "evaluation" / "after.json", _evaluation(1.0))

    report = generate_report(tmp_path)
    output = tmp_path / "visualization"

    assert (output / "coherence_summary.png").stat().st_size > 0
    assert (output / "global_distribution_diagnostics.png").stat().st_size > 0
    assert (output / "cross_spectrum_diagnostics.png").stat().st_size > 0
    assert (output / "topology_diagnostics.png").stat().st_size > 0
    inventory = json.loads((output / "coherence_report.json").read_text())
    assert inventory["generated"] == report["generated"]
    assert inventory["inputs"]["calibration"]["error"] == "missing"


def test_missing_optional_payloads_are_inventoried_without_family_figures(tmp_path: Path) -> None:
    _write(tmp_path / "evaluation" / "before.json", {"mse_normalized": 2.0})

    report = generate_report(tmp_path)

    assert report["missing_families"] == [
        "global_distribution",
        "cross_spectrum",
        "topology",
    ]
    assert report["inputs"]["evaluation_after"]["error"] == "missing"
    assert report["generated"] == ["coherence_summary.png"]


def test_history_inputs_combine_legacy_and_logs_and_deduplicate_alias(tmp_path: Path) -> None:
    legacy = tmp_path / "metrics/history.jsonl"
    current = tmp_path / "logs/history.jsonl"
    legacy.parent.mkdir()
    current.parent.mkdir()
    legacy.write_text('{"step": 1}\n')
    current.write_text('{"step": 2}\n')
    parsed = parse_run_payload(tmp_path)
    assert [row["step"] for row in parsed["history"]] == [1, 2]
    assert parsed["inputs"]["history"] == {
        "path": "metrics/history.jsonl", "paths": ["metrics/history.jsonl", "logs/history.jsonl"],
        "error": None, "rows": 2,
    }
    legacy.unlink()
    legacy.symlink_to("../logs/history.jsonl")
    parsed = parse_run_payload(tmp_path)
    assert [row["step"] for row in parsed["history"]] == [2]
    assert parsed["inputs"]["history"]["paths"] == ["metrics/history.jsonl"]


def test_history_input_reports_canonical_path_and_read_errors(tmp_path: Path) -> None:
    current = tmp_path / "logs/history.jsonl"
    current.parent.mkdir()
    current.write_text('{"step": 1}\n{broken}\n')
    history = parse_run_payload(tmp_path)["inputs"]["history"]
    assert history["path"] == "logs/history.jsonl"
    assert history["paths"] == ["logs/history.jsonl"]
    assert history["rows"] == 1
    assert "logs/history.jsonl: unreadable" in history["error"]


def test_offline_parser_import_does_not_load_training_or_model_dependencies(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    code = """
import runpy
import sys
runpy.run_path(sys.argv[1], run_name='offline_report_import')
blocked = ('phycoflow_reconstruction.training', 'phycoflow_reconstruction.models', 'pykeops')
loaded = [name for name in sys.modules if any(name == p or name.startswith(p + '.') for p in blocked)]
assert not loaded, loaded
"""
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    subprocess.run([sys.executable, "-c", code, str(_SCRIPT)], check=True, env=env)
