#!/usr/bin/env python3
"""Compare a completed cached-child readout with an existing support readout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np

from scripts.analysis.car_round8_cached_summary import ref
from scripts.analysis.car_round8_child_relations import SCALE_KEYS


def compare(candidate, reference, *, reference_label):
    if (candidate.get("status") != "completed" or reference.get("status") != "completed"
            or candidate.get("sample_ids") != reference.get("sample_ids")
            or candidate.get("field_order") != reference.get("field_order")
            or candidate.get("members") != reference.get("members")
            or candidate.get("split") != reference.get("split")
            or candidate.get("inputs", {}).get("source_readout", {}).get("sha256") is None):
        raise ValueError("completed matched panel identities required")
    ids = candidate["sample_ids"]
    if ([r["sample_id"] for r in candidate["records"]] != ids
            or [r["sample_id"] for r in reference["records"]] != ids):
        raise ValueError("matched condition order required")
    keys = (*SCALE_KEYS, "reference_power_over_floor", "reference_low_energy", "field_pair_indices")
    errors = {}
    for actual_name, expected_name in (("SOURCE", "SOURCE"), ("candidate", reference_label)):
        fields = ("state",) + tuple("scale." + key for key in keys)
        for field in fields:
            def value(row, model):
                item = row["models"][model]
                for key in field.split("."):
                    item = item[key]
                return item
            a = np.asarray([value(row, actual_name) for row in candidate["records"]], dtype=float)
            b = np.asarray([value(row, expected_name) for row in reference["records"]], dtype=float)
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError(f"invalid matched {actual_name} {field} values")
            maximum = float(np.max(np.abs(a-b))) if a.size else 0.
            if not np.allclose(a, b, rtol=2e-8, atol=1e-10):
                raise ValueError(f"readout parity failed for {actual_name} {field}: max abs {maximum}")
            errors[actual_name + "." + field] = maximum
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate, reference = json.loads(args.candidate.read_text()), json.loads(args.reference.read_text())
    if candidate["inputs"]["source_readout"]["sha256"] != ref(args.reference)["sha256"]:
        raise ValueError("readout does not bind the declared reference")
    errors = compare(candidate, reference, reference_label=args.reference_label)
    result = {"schema": "car_round8_relation_parity_v1", "status": "passed", "split": candidate["split"],
              "conditions": len(candidate["sample_ids"]), "members": candidate["members"],
              "endpoint_label": candidate["endpoint_label"], "reference_model": args.reference_label,
              "maximum_absolute_differences": errors, "tolerance": {"rtol": 2e-8, "atol": 1e-10},
              "candidate": ref(args.candidate), "reference": ref(args.reference), "script": ref(Path(__file__)),
              "cached_child_cpu_timing": candidate["timing"], "additional_inference": False, "test_locked": True}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": "passed", "conditions": result["conditions"], "max_abs": max(errors.values())}))


if __name__ == "__main__":
    main()
