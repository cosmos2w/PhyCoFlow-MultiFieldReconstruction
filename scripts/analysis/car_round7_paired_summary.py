#!/usr/bin/env python3
"""Summarize matched cached Scale decomposition and native field records.

This is a CPU-only readout, not an inference or scoring driver. Inputs must
already have been produced under the declared Round 7 bank contracts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BLOCKS = ("signed", "joint", "log_power_self", "rho", "third_order")
TERMS = ("attraction", "separation", "score")


def paired_summary(source, child, indices, unit=1.0):
    source = np.asarray(source, dtype=float)
    child = np.asarray(child, dtype=float)
    if source.shape != child.shape or source.ndim == 0 or not np.isfinite(source).all() or not np.isfinite(child).all():
        raise ValueError("matched finite condition records required")
    delta = (child - source) / unit
    # Bound peak memory for TRAIN800 localization arrays while preserving the
    # exact same paired bootstrap draws and condition-level resampling.
    means = np.concatenate([np.mean(delta[chunk], axis=1)
                            for chunk in np.array_split(indices, max(1, (len(indices) + 63) // 64))])
    return {"mean": np.mean(delta, axis=0).tolist(),
            "interval_95": np.quantile(means, [0.025, 0.975], axis=0).tolist()}


def summarize_scale(doc, child, unit, bootstrap=2000, seed=6042026):
    if not np.isfinite(unit) or unit <= 0 or bootstrap < 1:
        raise ValueError("positive finite SOURCE unit and bootstrap count required")
    rows = doc["records"]
    ids = [row["sample_id"] for row in rows]
    if not ids or len(set(ids)) != len(ids) or ids != doc["sample_ids"] or doc["members"] != 4:
        raise ValueError("unique ordered J4 condition IDs required")
    weights = doc["prefit_coefficients"]
    factors = {key: float(weights["scale." + key]) * (0.25 if key == "joint" else 1.0) for key in BLOCKS}
    if not all(np.isfinite(v) and v > 0 for v in factors.values()):
        raise ValueError("positive finite frozen Scale weights required")
    for row in rows:
        for name in ("SOURCE", child):
            model = row["models"][name]
            for key in BLOCKS:
                block = model["blocks"][key]
                if not np.isclose(block["attraction"] - block["separation"], block["score"], rtol=1e-8, atol=1e-10):
                    raise ValueError(f"{name} {key}: A-R does not reconstruct score")
            reconstructed = sum(factors[k] * model["blocks"][k]["score"] for k in BLOCKS)
            if not np.isclose(reconstructed, model["definitions"]["Complete_Expanded_Prefit"], rtol=1e-8, atol=1e-10):
                raise ValueError(f"{name}: expanded Scale does not reconstruct")
    indices = np.random.default_rng(seed).integers(0, len(rows), size=(bootstrap, len(rows)))
    def values(model, section, key, term=None):
        result = [r["models"][model][section][key] for r in rows]
        return [x[term] for x in result] if term else result
    blocks = {key: {term: paired_summary(values("SOURCE", "blocks", key, term),
                                         values(child, "blocks", key, term), indices, unit / factors[key])
                    for term in TERMS} for key in BLOCKS}
    definitions = {key: paired_summary(values("SOURCE", "definitions", key), values(child, "definitions", key), indices, unit)
                   for key in ("Published_Core_Default", "Complete_Core_Prefit", "Complete_Expanded_Prefit")}
    # The raw full-joint score is distinct from its weighted composite contribution.
    joint = {term: paired_summary(values("SOURCE", "blocks", "joint", term), values(child, "blocks", "joint", term), indices, unit)
             for term in TERMS}
    localization = {key: {term: paired_summary(values("SOURCE", "localization", key, term), values(child, "localization", key, term), indices, unit)
                          for term in TERMS} for key in rows[0]["models"]["SOURCE"]["localization"]}
    return {"sample_ids": ids, "split": doc["split"], "child": child, "source_unit": unit,
            "bootstrap": {"seed": seed, "replicates": bootstrap, "unit": "paired condition"},
            "definitions": definitions, "weighted_blocks": blocks, "raw_joint": joint,
            "localization_raw": localization,
            "interpretation": "Lower score is better; score = attraction - positive separation. Localization is descriptive, not a proper field score."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decomposition", type=Path, required=True)
    parser.add_argument("--child", required=True)
    parser.add_argument("--source-unit", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.decomposition.read_bytes()
    result = summarize_scale(json.loads(data), args.child, args.source_unit)
    result.update(schema="car_round7_paired_scale_summary_v1", input_sha256=hashlib.sha256(data).hexdigest())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
