"""Bounded, reproducible TRAIN-only candidate generation and bank freezing.

Candidate ranking inputs are measured by the caller on the declared TRAIN/SOURCE
panel; this module does not load data or infer evidence from candidate generation.
"""

from __future__ import annotations

import hashlib
import json
import math
import random

VERSION = "adaptive_train_bank_v1"


def _id(kind, payload):
    canonical = json.dumps([VERSION, kind, payload], sort_keys=True, separators=(",", ":"))
    return kind + ":" + hashlib.sha256(canonical.encode()).hexdigest()[:24]


def candidates(field_ids, *, bands, regions, seed=0):
    """Generate at most min(8*C,128) non-exhaustive signed groups and triples.

    Field order defines a balanced tree, and seeded exploration fills the rest.
    No pair/triple combination catalogue is materialized.
    """
    fields = tuple(field_ids)
    if (
        not fields
        or len(set(fields)) != len(fields)
        or any(not isinstance(x, str) or not x for x in fields)
    ):
        raise ValueError("unique field IDs required")
    if bands < 1 or regions < 1:
        raise ValueError("positive band and region counts required")
    n, rng = len(fields), random.Random(seed)
    cap = min(8 * n, 128)
    result, seen = [], set()

    def add(kind, payload):
        ident = _id(kind, payload)
        if ident not in seen and len(result) < cap:
            seen.add(ident)
            result.append({"id": ident, "kind": kind, **payload})

    def group(indices, signs):
        # Fix global sign by making first nonzero positive.
        pairs = sorted(zip(indices, signs))
        if pairs[0][1] < 0:
            pairs = [(i, -s) for i, s in pairs]
        add(
            "state.group_projection",
            {"fields": [fields[i] for i, _ in pairs], "signs": [s for _, s in pairs]},
        )

    def tree(lo, hi):
        if hi - lo < 2:
            return
        mid = (lo + hi) // 2
        if len(result) < cap // 2:
            group(list(range(lo, hi)), [1] * (hi - lo))
        if len(result) < cap // 2:
            group(list(range(lo, hi)), [1 if i < mid else -1 for i in range(lo, hi)])
        tree(lo, mid)
        tree(mid, hi)

    tree(0, n)
    # Bounded attempts also terminate for tiny populations or duplicate draws.
    for _ in range(cap * 16):
        if len(result) >= cap:
            break
        if n >= 3 and rng.random() < 0.65:
            indices = sorted(rng.sample(range(n), 3))
            add(
                "scale.third_order",
                {
                    "region": rng.randrange(regions),
                    "terms": [[rng.randrange(bands), fields[i]] for i in indices],
                },
            )
        elif n >= 2:
            k = rng.randint(2, min(n, 8))
            indices = sorted(rng.sample(range(n), k))
            group(indices, [rng.choice((-1, 1)) for _ in indices])
    return {
        "version": VERSION,
        "field_order": fields,
        "seed": int(seed),
        "bands": int(bands),
        "regions": int(regions),
        "candidates": result,
    }


def freeze(
    candidate_bank,
    measurements,
    *,
    split,
    population_ids,
    source_identity,
    projection_limit=4,
    triple_limit=8,
):
    """Select measured candidates and freeze a content-addressed TRAIN bank.

    measurements maps IDs to nonnegative discrepancy, repeatability,
    sensitivity and positive incremental cost. Sensitivity/repeatability zero
    excludes a candidate. Ranking is a heuristic, not a statistical guarantee.
    """
    if split != "TRAIN" or not population_ids or not source_identity:
        raise ValueError("explicit nonempty TRAIN panel and source identity required")
    if (
        candidate_bank.get("version") != VERSION
        or not 0 <= projection_limit <= 4
        or not 0 <= triple_limit <= 8
    ):
        raise ValueError("invalid bank version or limits")
    valid_ids = {row["id"] for row in candidate_bank["candidates"]}
    if set(measurements) - valid_ids:
        raise ValueError("measurements contain unknown candidate")
    ranked = {"state.group_projection": [], "scale.third_order": []}
    for item in candidate_bank["candidates"]:
        m = measurements.get(item["id"])
        if m is None:
            continue
        values = [float(m[k]) for k in ("discrepancy", "repeatability", "sensitivity", "cost")]
        if any(not math.isfinite(v) for v in values) or min(values[:3]) < 0 or values[3] <= 0:
            raise ValueError("invalid discovery measurements")
        if values[1] and values[2]:
            priority = values[0] * values[1] * values[2] / math.sqrt(values[3])
            ranked[item["kind"]].append((priority, item["id"], item))
    chosen = []
    for kind, limit in (
        ("state.group_projection", projection_limit),
        ("scale.third_order", triple_limit),
    ):
        chosen.extend(row[2] for row in sorted(ranked[kind], key=lambda r: (-r[0], r[1]))[:limit])
    bank = {
        "version": VERSION,
        "split": split,
        "population_ids": list(population_ids),
        "source_identity": source_identity,
        "field_order": list(candidate_bank["field_order"]),
        "seed": candidate_bank["seed"],
        "selected": chosen,
        "measurements": {q["id"]: measurements[q["id"]] for q in chosen},
    }
    bank["sha256"] = hashlib.sha256(
        json.dumps(bank, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return bank
