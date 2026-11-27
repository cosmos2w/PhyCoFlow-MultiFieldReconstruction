#!/usr/bin/env python3
"""Describe SOURCE risk and reference-energy strata on cached matched panels.

All reference moments use the already saved native fields and physical masses;
no model, training data loader, TEST sample, or causal explanation is used.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
import torch

from scripts.analysis.car_round8_cached_summary import checked_ref, ref
from scripts.posttraining.assess_car_round7 import read_row


def reference_moments(truth, masses, scale, offset):
    x = truth.double()[0] * scale.double() + offset.double()
    mass = masses.double()
    if mass.ndim != 1 or x.shape[0] != len(mass) or (mass < 0).any() or mass.sum() <= 0:
        raise ValueError("valid matched native physical masses required")
    weight = mass / mass.sum()
    mean = (x * weight[:, None]).sum(0)
    energy = (x.square() * weight[:, None]).sum(0)
    variance = ((x-mean).square() * weight[:, None]).sum(0)
    return mean.tolist(), energy.tolist(), variance.tolist()


def summarize_strata(risk, energy, variance, ids, fields):
    risk, energy, variance = (np.asarray(x, dtype=np.float64) for x in (risk, energy, variance))
    if any(x.shape != (len(ids), len(fields)) or not np.isfinite(x).all() for x in (risk, energy, variance)):
        raise ValueError("finite condition-by-field data required")
    rows = []
    for f, field in enumerate(fields):
        median = float(np.median(variance[:, f]))
        lower = variance[:, f] <= median
        upper = ~lower
        strata = []
        for name, select in (("lower_or_equal_median", lower), ("above_median", upper)):
            strata.append({"name": name, "conditions": int(select.sum()),
                           "mean_reference_variance": float(variance[select, f].mean()) if select.any() else None,
                           "mean_member_risk": float(risk[select, f].mean()) if select.any() else None})
        rows.append({"field": field, "source_member_mse_mean": float(risk[:, f].mean()),
                     "source_member_mse_quantiles_10_50_90": np.quantile(risk[:, f], [.1,.5,.9]).tolist(),
                     "reference_uncentered_energy_mean": float(energy[:, f].mean()),
                     "reference_spatial_variance_mean": float(variance[:, f].mean()),
                     "reference_spatial_variance_quantiles_10_50_90": np.quantile(variance[:, f], [.1,.5,.9]).tolist(),
                     "source_member_mse_over_reference_spatial_variance": (float(risk[:, f].mean()/variance[:, f].mean()) if variance[:, f].mean() > 0 else None),
                     "reference_variance_median": median, "within_split_strata": strata})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-assessment", type=Path, required=True)
    p.add_argument("--validation-assessment", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, help="smoke only")
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args(); start = time.perf_counter(); cpu = time.process_time(); torch.set_num_threads(args.threads)
    assessments = {"train": json.loads(args.train_assessment.read_text()),
                   "validation": json.loads(args.validation_assessment.read_text())}
    out = {}
    for split, assessment in assessments.items():
        ids = assessment["sample_ids"]; expected = 800 if split == "train" else 100
        if assessment.get("split") != split or assessment.get("status") != "completed" or len(ids) != expected or assessment.get("members") != 4:
            raise ValueError("declared TRAIN800 and VALIDATION100 J4 cached panels required")
        path = checked_ref(assessment["source_native"]); receipt = json.loads(path.read_text())
        if receipt["sample_ids"] != ids:
            raise ValueError("SOURCE panel identity differs")
        count = len(ids) if args.limit is None else min(len(ids), args.limit)
        means, energies, variances = [], [], []
        for i in range(count):
            row = read_row(path, receipt, i)
            mean, energy, variance = reference_moments(row["truth_model"], row["native_masses"], row["normalizer_scale"], row["normalizer_offset"])
            means.append(mean); energies.append(energy); variances.append(variance)
        risks = [row["member_mse"] for row in assessment["per_condition_fields"]["SOURCE"][:count]]
        out[split] = {"sample_ids": ids[:count], "records": [{"sample_id": sid, "mean": mean, "uncentered_energy": en,
                     "spatial_variance": var, "member_mse": risk} for sid, mean, en, var, risk in zip(ids[:count], means, energies, variances, risks, strict=True)],
                     "fields": summarize_strata(risks, energies, variances, ids[:count], assessment["field_order"]),
                     "source_native_receipt": ref(path)}
    if assessments["train"]["field_order"] != assessments["validation"]["field_order"]:
        raise ValueError("split field order differs")
    result = {"schema": "car_round8_reference_strata_v1", "status": "completed" if args.limit is None else "smoke_only",
              "fields": assessments["train"]["field_order"], "split": out,
              "definition": "positive native physical area weights; dataset-unit field means, uncentered mean square and centered spatial variance; per-field within-split median reference-variance strata are descriptive",
              "scope": "cached SOURCE J4 and reference only; one trajectory; no inference or causal attribution", "test_locked": True,
              "inputs": {"train_assessment": ref(args.train_assessment), "validation_assessment": ref(args.validation_assessment), "script": ref(Path(__file__))},
              "timing": {"wall_seconds": time.perf_counter()-start, "cpu_seconds": time.process_time()-cpu, "threads": args.threads}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")


if __name__ == "__main__":
    main()
