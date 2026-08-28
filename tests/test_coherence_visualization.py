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
