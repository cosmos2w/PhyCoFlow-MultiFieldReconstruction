"""Stable, bounded field-relation populations for the coherence views."""

from __future__ import annotations

import itertools
import random


class QueryBank:
    """A field-ID keyed bank with at most two unordered relations per field."""

    def __init__(self, field_ids, seed=0):
        self.field_ids = tuple(field_ids)
        if not self.field_ids or any(not isinstance(x, str) or not x for x in self.field_ids):
            raise ValueError("field IDs must be nonempty strings")
        if len(set(self.field_ids)) != len(self.field_ids):
            raise ValueError("field IDs must be unique")
        fields = sorted(self.field_ids)
        n = len(fields)
        count = min(n * (n - 1) // 2, 2 * n)
        if n * (n - 1) // 2 <= 2 * n:
            pairs = set(itertools.combinations(fields, 2))
        else:
            rng = random.Random(seed)
            pairs = set()
            for _ in range(2):
                cycle = fields.copy()
                rng.shuffle(cycle)
                pairs.update(tuple(sorted((cycle[i], cycle[(i + 1) % n]))) for i in range(n))
            while len(pairs) < count:
                pairs.add(tuple(sorted(rng.sample(fields, 2))))
        self.pairs = tuple(sorted(pairs))
        self.seed = int(seed)

    def manifest(self):
        return {
            "version": 1,
            "field_ids": list(self.field_ids),
            "pairs": [list(p) for p in self.pairs],
            "seed": self.seed,
            "unrepresented_pair_count": len(self.field_ids) * (len(self.field_ids) - 1) // 2
            - len(self.pairs),
        }


def resolve_coverage_config(config, dataset_field_names):
    """Resolve the common bounded Scale/finite-primary Shape query population.

    The input mapping is never mutated. Legacy noncoverage configuration is
    untouched; already expanded banks keep their existing configuration identity.
    """
    from copy import deepcopy

    resolved = deepcopy(dict(config))
    coverage = resolved.get("query_coverage", {})
    if not coverage.get("enabled", False):
        return resolved
    families = resolved.get("families", {})
    scale = families.get("multiscale_spectrum")
    fields = tuple((scale.get("fields") if isinstance(scale, dict) else None)
                   or dataset_field_names or ())
    if not fields:
        return resolved
    bank = QueryBank(fields, seed=int(coverage.get("relation_seed", 0)))
    if isinstance(scale, dict) and scale.get("enabled", True) and len(bank.pairs) < len(fields) * (len(fields) - 1) // 2:
        components = scale.setdefault("components", {})
        cospectrum = components.setdefault("signed_band_cospectrum", {})
        if cospectrum.get("enabled", True) and "pairs" not in cospectrum:
            cospectrum["pairs"] = [list(pair) for pair in bank.pairs]
    topology = families.get("topology")
    if (isinstance(topology, dict) and topology.get("enabled", True)
            and topology.get("strategy") == "cubical_persistence"
            and topology.get("persistence", {}).get("definition") == "finite_primary_v1"):
        previous_fields = tuple(topology.get("fields") or dataset_field_names or ())
        changed_fields = previous_fields != fields
        if changed_fields:
            topology["fields"] = list(fields)
        components = topology.setdefault("components", {})
        self_component = components.setdefault("self", {})
        if changed_fields or ("fields" in self_component and tuple(self_component["fields"]) != fields):
            self_component["fields"] = list(fields)
        mutual = components.setdefault("mutual", {})
        pairs = [list(pair) for pair in bank.pairs]
        if not pairs:
            mutual["enabled"] = False
            mutual["groups"] = []
        else:
            mutual["enabled"] = True
            if mutual.get("groups") != pairs:
                mutual["groups"] = pairs
    return resolved
