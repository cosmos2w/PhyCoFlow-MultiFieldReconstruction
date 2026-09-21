"""Validation-only topology selection with source-relative fidelity constraints."""

import math
from collections.abc import Mapping


def fidelity_eligibility(source: Mapping, candidate: Mapping, settings: Mapping) -> dict:
    total_budget = float(settings["max_relative_mse_increase"])
    field_budget = float(settings["max_relative_field_mse_increase"])
    failures = []
    pairs = {"total": (source["mse_normalized"], candidate["mse_normalized"], total_budget)}
    for name, before in source["per_field_mse_normalized"].items():
        pairs[name] = (before, candidate["per_field_mse_normalized"][name], field_budget)
    ratios = {}
    for name, (before, after, budget) in pairs.items():
        before, after = float(before), float(after)
        ratios[name] = after / before - 1 if before > 0 else (0.0 if after == 0 else None)
        if not math.isfinite(after) or after > before * (1 + budget):
            failures.append(name)
    return {
        "eligible": not failures,
        "failed_fidelity_fields": failures,
        "relative_mse_increase": ratios,
    }


def topology_selection_report(source: Mapping, candidate: Mapping, settings: Mapping) -> dict:
    report = fidelity_eligibility(source, candidate, settings)
    # Select topology itself even when other families also participate in training.
    topology = candidate["coherence"]["families"]["topology"]
    score = float(topology.get("selection_score", topology["total"]))
    report.update(metric=score, mse=float(candidate["mse_normalized"]), metrics=dict(candidate))
    report["eligible"] &= math.isfinite(score)
    if settings.get("max_relative_native_topology_increase") is not None:
        budget = float(settings["max_relative_native_topology_increase"])
        before, after = source["native_topology"], candidate["native_topology"]
        failed = [
            name
            for name, value in after["components"].items()
            if not math.isfinite(float(value))
            or float(value) > float(before["components"][name]) * (1 + budget)
        ]
        report["failed_native_topology_components"] = failed
        report["eligible"] &= not failed
    return report
