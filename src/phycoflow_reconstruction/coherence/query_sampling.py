"""Versioned implicit all-field queries and unbiased, bounded work estimates.

Sampling is independent of model graphs. Call ``draw`` once and pass the same
batch to both families; call ``finish`` only after observing detached difficulties.
"""

from __future__ import annotations

import math
import random
from collections import Counter, OrderedDict
from dataclasses import dataclass, replace

VERSION = "round8_implicit_queries_v2"
SHAPE_SAMPLING_VERSION = "round8_pooled_shape_inner_sample_v2"
PAIR_SIGNS = ((1, 1), (-1, -1), (1, -1), (-1, 1))
_line_unnormalized = (1 / math.sqrt(2), 1 / math.sqrt(5), 1 / math.sqrt(5))
LINE_WEIGHTS = tuple(x / sum(_line_unnormalized) for x in _line_unnormalized)


def pair_count(n):
    if n < 1:
        raise ValueError("at least one field is required")
    return n * (n - 1) // 2


def pair_index(n, a, b):
    if not 0 <= a < b < n:
        raise ValueError("pair must satisfy 0 <= a < b < n")
    return a * (2 * n - a - 1) // 2 + b - a - 1


def index_pair(n, index):
    if not 0 <= index < pair_count(n):
        raise ValueError("pair index out of range")
    lo, hi = 0, n - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if mid * (2 * n - mid - 1) // 2 <= index:
            lo = mid
        else:
            hi = mid
    return lo, lo + 1 + index - lo * (2 * n - lo - 1) // 2


@dataclass(frozen=True)
class Draw:
    index: int
    count: int
    probability: float
    population: int
    draws: int
    inner_sign: tuple[int, ...] | None = None
    inner_line: int | None = None
    inner_probability: float | None = None

    @property
    def weight(self):
        return self.count / (self.draws * self.population * self.probability)


@dataclass(frozen=True)
class QueryBatch:
    self_queries: tuple[Draw, ...]
    pair_queries: tuple[Draw, ...]
    number: int


class SharedQuerySampler:
    """Sparse EMA priority with a frozen pre-draw mixture and exact resume.

    ``observe`` accepts one normalized value from each family for every observed
    ID. Zero priorities contribute no priority mass. Cache eviction is LRU;
    refreshed distributions are snapshots, bounded by the same capacity.
    """

    def __init__(
        self,
        field_ids,
        *,
        seed=0,
        mutual_draws=3,
        epsilon=0.5,
        decay=0.9,
        capacity=None,
        refresh_epochs=10,
    ):
        self.fields = tuple(field_ids)
        if (
            not self.fields
            or any(not isinstance(x, str) or not x for x in self.fields)
            or len(set(self.fields)) != len(self.fields)
        ):
            raise ValueError("unique nonempty field IDs required")
        if mutual_draws < 1 or not 0 < epsilon <= 1 or not 0 <= decay < 1 or refresh_epochs < 1:
            raise ValueError("invalid sampler settings")
        self.mutual_draws, self.epsilon, self.decay = (
            int(mutual_draws),
            float(epsilon),
            float(decay),
        )
        self.capacity = max(64, 8 * len(self.fields)) if capacity is None else int(capacity)
        if self.capacity < 1:
            raise ValueError("capacity must be positive")
        self.refresh_epochs = int(refresh_epochs)
        self.rng = random.Random(seed)
        self.inner_seed = int(seed) ^ 0x524F554E4438
        self.inner_rng = random.Random(self.inner_seed)
        self.cache = {"self": OrderedDict(), "pair": OrderedDict()}
        self.distribution = {"self": {}, "pair": {}}
        self.last_refresh = 0
        self.counter = 0
        self.pending = None

    def _draw(self, kind, population, count):
        if not population:
            return ()
        dist = self.distribution[kind]
        ids = tuple(dist)
        probs = tuple(dist.values())
        picks = []
        for _ in range(count):
            if ids and self.rng.random() >= self.epsilon:
                index = self.rng.choices(ids, weights=probs, k=1)[0]
            else:
                index = self.rng.randrange(population)
            picks.append(index)
        counts = Counter(picks)
        return tuple(
            Draw(
                i,
                k,
                self.epsilon / population + (1 - self.epsilon) * dist.get(i, 0.0)
                if ids
                else 1 / population,
                population,
                count,
            )
            for i, k in counts.items()
        )

    def draw(self):
        if self.pending is not None:
            raise RuntimeError("finish pending batch before drawing")
        self.counter += 1
        self_draws = self._draw("self", len(self.fields), 1)
        pair_draws = self._draw("pair", pair_count(len(self.fields)), self.mutual_draws)
        # Inner draws use a separate checkpointed RNG and do not alter the
        # historical outer-ID stream. Duplicate outer IDs share one inner draw.
        selected_self = tuple(
            replace(d, inner_sign=(self.inner_rng.choice((1, -1)),), inner_probability=0.5)
            for d in self_draws
        )
        selected_pairs = []
        for d in pair_draws:
            sign = self.inner_rng.choice(PAIR_SIGNS)
            line = self.inner_rng.choices(range(3), weights=LINE_WEIGHTS, k=1)[0]
            selected_pairs.append(
                replace(
                    d, inner_sign=sign, inner_line=line, inner_probability=LINE_WEIGHTS[line] / 4
                )
            )
        self.pending = QueryBatch(selected_self, tuple(selected_pairs), self.counter)
        return self.pending

    def finish(self, *, self_difficulties=None, pair_difficulties=None, exposure_sink=None):
        """Difficulties map sampled index to (Scale, Shape), already TRAIN-normalized."""
        if self.pending is None:
            raise RuntimeError("no pending batch")
        for kind, rows, observations in (
            ("self", self.pending.self_queries, self_difficulties),
            ("pair", self.pending.pair_queries, pair_difficulties),
        ):
            observations = observations or {}
            if set(observations) - {d.index for d in rows}:
                raise ValueError("difficulty IDs must belong to pending batch")
            for d in rows:
                if exposure_sink is not None:
                    exposure_sink(
                        {
                            "batch": self.pending.number,
                            "kind": kind,
                            "index": d.index,
                            "count": d.count,
                            "probability": d.probability,
                            "inner_sign": list(d.inner_sign),
                            "inner_line": d.inner_line,
                            "inner_probability": d.inner_probability,
                            "query_protocol": VERSION,
                            "shape_sampling": SHAPE_SAMPLING_VERSION,
                            "inner_seed": self.inner_seed,
                        }
                    )
                if d.index in observations:
                    scale, shape = (float(x) for x in observations[d.index])
                    if not math.isfinite(scale) or not math.isfinite(shape):
                        raise ValueError("difficulty must be finite and detached")
                    value = (max(0.0, scale) + max(0.0, shape)) / 2
                    cache = self.cache[kind]
                    if d.index in cache:
                        value = self.decay * cache.pop(d.index) + (1 - self.decay) * value
                    cache[d.index] = value
                    while len(cache) > self.capacity:
                        cache.popitem(last=False)
        self.pending = None

    def refresh(self, completed_epoch):
        if self.pending is not None:
            raise RuntimeError("cannot refresh during a pending batch")
        if completed_epoch < self.last_refresh:
            raise ValueError("epochs cannot decrease")
        if completed_epoch - self.last_refresh < self.refresh_epochs:
            return False
        for kind, cache in self.cache.items():
            total = sum(cache.values())
            self.distribution[kind] = (
                {i: v / total for i, v in cache.items() if v > 0} if total else {}
            )
        self.last_refresh = int(completed_epoch)
        return True

    def state_dict(self):
        def encode(rows):
            return [vars(x).copy() for x in rows]

        return {
            "version": VERSION,
            "fields": self.fields,
            "mutual_draws": self.mutual_draws,
            "epsilon": self.epsilon,
            "decay": self.decay,
            "capacity": self.capacity,
            "refresh_epochs": self.refresh_epochs,
            "rng": self.rng.getstate(),
            "inner_seed": self.inner_seed,
            "inner_rng": self.inner_rng.getstate(),
            "shape_sampling": SHAPE_SAMPLING_VERSION,
            "cache": {k: list(v.items()) for k, v in self.cache.items()},
            "distribution": {k: list(v.items()) for k, v in self.distribution.items()},
            "last_refresh": self.last_refresh,
            "counter": self.counter,
            "pending": None
            if self.pending is None
            else {
                "self_queries": encode(self.pending.self_queries),
                "pair_queries": encode(self.pending.pair_queries),
                "number": self.pending.number,
            },
        }

    @classmethod
    def from_state_dict(cls, state):
        if state["version"] != VERSION:
            raise ValueError("incompatible sampler version")
        obj = cls(
            state["fields"],
            mutual_draws=state["mutual_draws"],
            epsilon=state["epsilon"],
            decay=state["decay"],
            capacity=state["capacity"],
            refresh_epochs=state["refresh_epochs"],
        )

        def tuples(x):
            return tuple(tuples(y) for y in x) if isinstance(x, (list, tuple)) else x

        if state.get("shape_sampling") != SHAPE_SAMPLING_VERSION:
            raise ValueError("incompatible Shape inner sampling protocol")
        obj.rng.setstate(tuples(state["rng"]))
        obj.inner_seed = int(state["inner_seed"])
        obj.inner_rng.setstate(tuples(state["inner_rng"]))
        obj.cache = {
            k: OrderedDict((int(i), float(v)) for i, v in rows)
            for k, rows in state["cache"].items()
        }
        obj.distribution = {
            k: {int(i): float(v) for i, v in rows} for k, rows in state["distribution"].items()
        }
        obj.last_refresh, obj.counter = state["last_refresh"], state["counter"]
        p = state["pending"]
        if p is not None:

            def restore_draw(d):
                data = dict(d)
                if data.get("inner_sign") is not None:
                    data["inner_sign"] = tuple(data["inner_sign"])
                return Draw(**data)

            obj.pending = QueryBatch(
                tuple(restore_draw(d) for d in p["self_queries"]),
                tuple(restore_draw(d) for d in p["pair_queries"]),
                p["number"],
            )
            for draw in obj.pending.self_queries:
                if (
                    draw.inner_sign not in ((1,), (-1,))
                    or draw.inner_line is not None
                    or draw.inner_probability != 0.5
                ):
                    raise ValueError("invalid pending self inner draw")
            for draw in obj.pending.pair_queries:
                line = draw.inner_line
                if (
                    draw.inner_sign not in PAIR_SIGNS
                    or isinstance(line, bool)
                    or not isinstance(line, int)
                    or not 0 <= line < len(LINE_WEIGHTS)
                    or draw.inner_probability is None
                    or not math.isclose(
                        draw.inner_probability, LINE_WEIGHTS[line] / 4, rel_tol=1e-12
                    )
                ):
                    raise ValueError("invalid pending pair inner draw")
        return obj


ADAPTIVE_VERSION = "adaptive_hierarchy_v1"


class HierarchicalQuerySampler(SharedQuerySampler):
    """Opt-in balanced-tree pair mixture with transactional accepted updates.

    Tree nodes are contiguous intervals in the declared field order. Each
    internal node owns precisely the pairs crossing its two children. The
    hierarchy is uniform over internal nodes and over each node's cross pairs.
    The sparse cache receives detached gradient-second-moment/cost priorities.
    """

    def __init__(
        self,
        field_ids,
        *,
        seed=0,
        mutual_draws=3,
        epsilon=0.25,
        hierarchy_weight=0.5,
        decay=0.9,
        capacity=None,
        refresh_epochs=10,
        fixed_probabilities=None,
        profile_sha256=None,
    ):
        super().__init__(
            field_ids,
            seed=seed,
            mutual_draws=mutual_draws,
            epsilon=epsilon,
            decay=decay,
            capacity=capacity,
            refresh_epochs=refresh_epochs,
        )
        if not 0 <= hierarchy_weight <= 1 - epsilon:
            raise ValueError("invalid hierarchy mixture weight")
        self.hierarchy_weight = float(hierarchy_weight)
        self.fixed_probabilities = None if fixed_probabilities is None else tuple(float(x) for x in fixed_probabilities)
        self.profile_sha256 = profile_sha256
        if self.fixed_probabilities is not None:
            count = pair_count(len(self.fields))
            if (len(self.fields) > 5 or len(self.fixed_probabilities) != count or not count
                or any(not math.isfinite(p) or p < 0.25 / count for p in self.fixed_probabilities)
                or not math.isclose(sum(self.fixed_probabilities), 1., rel_tol=1e-10, abs_tol=1e-10)
                or not isinstance(profile_sha256, str) or len(profile_sha256) != 64):
                raise ValueError("invalid fixed SOURCE profile probabilities or identity")
        self.nodes = []

        def build(lo, hi):
            if hi - lo > 1:
                mid = (lo + hi) // 2
                self.nodes.append((lo, mid, hi))
                build(lo, mid)
                build(mid, hi)

        build(0, len(self.fields))
        self._transaction = None

    def pair_probability(self, index):
        if self.fixed_probabilities is not None:
            return self.fixed_probabilities[index]
        n = len(self.fields)
        a, b = index_pair(n, index)
        lo, hi = 0, n
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if a < mid <= b:
                group = 1 / (len(self.nodes) * (mid - lo) * (hi - mid))
                break
            if b < mid:
                hi = mid
            else:
                lo = mid
        cache = self.distribution["pair"]
        cw = 1 - self.epsilon - self.hierarchy_weight if cache else 0.0
        # Empty cache mass goes to uniform support.
        return (
            (1 - self.hierarchy_weight - cw) / pair_count(n)
            + self.hierarchy_weight * group
            + cw * cache.get(index, 0.0)
        )

    def _draw(self, kind, population, count):
        if kind != "pair" or not population:
            return super()._draw(kind, population, count)
        if self.fixed_probabilities is not None:
            picks = self.rng.choices(range(population), weights=self.fixed_probabilities, k=count)
            return tuple(Draw(i, k, self.fixed_probabilities[i], population, count)
                         for i, k in Counter(picks).items())
        cache = self.distribution["pair"]
        cw = 1 - self.epsilon - self.hierarchy_weight if cache else 0.0
        picks = []
        for _ in range(count):
            u = self.rng.random()
            if u < self.hierarchy_weight:
                lo, mid, hi = self.rng.choice(self.nodes)
                a, b = self.rng.randrange(lo, mid), self.rng.randrange(mid, hi)
                picks.append(pair_index(len(self.fields), a, b))
            elif u < self.hierarchy_weight + cw:
                picks.append(self.rng.choices(tuple(cache), weights=tuple(cache.values()), k=1)[0])
            else:
                picks.append(self.rng.randrange(population))
        return tuple(
            Draw(i, k, self.pair_probability(i), population, count)
            for i, k in Counter(picks).items()
        )

    def draw(self):
        if self._transaction is not None:
            raise RuntimeError("pending transaction")
        self._transaction = (self.rng.getstate(), self.inner_rng.getstate(), self.counter)
        try:
            return super().draw()
        except Exception:
            self._transaction = None
            raise

    def reject(self):
        """Roll back random streams and cursor for a rejected update."""
        if self._transaction is None or self.pending is None:
            raise RuntimeError("no pending transaction")
        self.rng.setstate(self._transaction[0])
        self.inner_rng.setstate(self._transaction[1])
        self.counter = self._transaction[2]
        self.pending = None
        self._transaction = None

    def finish(self, **kwargs):
        super().finish(**kwargs)
        self._transaction = None

    def finish_measurements(
        self, *, self_measurements=None, pair_measurements=None, exposure_sink=None
    ):
        """Commit measured detached gradient second moment and incremental cost.

        Each value is (gradient_second_moment, incremental_cost). Measurements
        should be gathered from lagged shared-endpoint gradient sketches, not
        scalar loss values. The result is a heuristic; full gradients still train.
        """

        def priorities(rows):
            output = {}
            for index, (second_moment, cost) in (rows or {}).items():
                second_moment, cost = float(second_moment), float(cost)
                if (
                    not math.isfinite(second_moment)
                    or second_moment < 0
                    or not math.isfinite(cost)
                    or cost <= 0
                ):
                    raise ValueError("invalid gradient/cost measurement")
                priority = math.sqrt(second_moment / cost)
                output[index] = (priority, priority)
            return output

        self.finish(
            self_difficulties=priorities(self_measurements),
            pair_difficulties=priorities(pair_measurements),
            exposure_sink=exposure_sink,
        )

    def state_dict(self):
        state = super().state_dict()
        state["adaptive_version"] = ADAPTIVE_VERSION
        state["hierarchy_weight"] = self.hierarchy_weight
        state["transaction"] = self._transaction
        state["fixed_probabilities"] = self.fixed_probabilities
        state["profile_sha256"] = self.profile_sha256
        return state

    @classmethod
    def from_state_dict(cls, state):
        if state.get("adaptive_version") != ADAPTIVE_VERSION:
            raise ValueError("incompatible adaptive sampler")
        base = SharedQuerySampler.from_state_dict(state)
        obj = cls(
            base.fields,
            mutual_draws=base.mutual_draws,
            epsilon=base.epsilon,
            hierarchy_weight=state["hierarchy_weight"],
            decay=base.decay,
            capacity=base.capacity,
            refresh_epochs=base.refresh_epochs,
            fixed_probabilities=state.get("fixed_probabilities"),
            profile_sha256=state.get("profile_sha256"),
        )
        obj.__dict__.update({k: v for k, v in base.__dict__.items() if k != "_transaction"})

        def tuples(x):
            return tuple(tuples(y) for y in x) if isinstance(x, (list, tuple)) else x

        obj._transaction = tuples(state.get("transaction"))
        if (obj.pending is None) != (obj._transaction is None):
            raise ValueError("inconsistent pending transaction")
        return obj


def allocate_counts(weights, deviations, costs, budget):
    """Heuristic for a draw-count cap; this is not a fixed-cost optimum."""
    if not (len(weights) == len(deviations) == len(costs)):
        raise ValueError("stratum arrays must have matching lengths")
    if any(not math.isfinite(x) or x < 0 for x in (*weights, *deviations)) or any(
        not math.isfinite(x) or x <= 0 for x in costs
    ):
        raise ValueError("invalid stratum statistics")
    active = [i for i, w in enumerate(weights) if w > 0]
    if isinstance(budget, bool) or int(budget) != budget or budget < len(active):
        raise ValueError("budget must cover every active stratum")
    counts = [int(i in active) for i in range(len(weights))]
    if not active:
        if budget:
            raise ValueError("no active strata")
        return counts
    scores = [weights[i] * deviations[i] / math.sqrt(costs[i]) for i in active]
    if sum(scores) == 0:
        scores = [1.0] * len(active)
    remaining = int(budget) - len(active)
    quotas = [remaining * s / sum(scores) for s in scores]
    for i, q in zip(active, quotas):
        counts[i] += int(q)
    for j in sorted(range(len(active)), key=lambda j: (-(quotas[j] % 1), active[j]))[
        : int(budget) - sum(counts)
    ]:
        counts[active[j]] += 1
    return counts


def allocate_cost_counts(weights, deviations, costs, cost_budget):
    """Round the fixed-cost variance allocation with one draw per active stratum.

    The continuous solution is n_h=max(1,lambda*W_h*sigma_h/sqrt(c_h)).
    Rounding uses affordable marginal variance decrease per unit cost. Integer
    optimality is not asserted. All counts are chosen before any current draw.
    """
    if not (len(weights) == len(deviations) == len(costs)):
        raise ValueError("stratum arrays must have matching lengths")
    if any(not math.isfinite(x) or x < 0 for x in (*weights, *deviations)) or any(
        not math.isfinite(x) or x <= 0 for x in costs
    ) or not math.isfinite(cost_budget) or cost_budget < 0:
        raise ValueError("finite nonnegative statistics and positive costs required")
    active = [i for i, value in enumerate(weights) if value > 0]
    minimum = sum(costs[i] for i in active)
    if cost_budget < minimum:
        raise ValueError("cost budget must cover every active stratum")
    counts = [int(i in active) for i in range(len(weights))]
    variable = [i for i in active if deviations[i] > 0]
    if not variable:
        return counts
    scores = {i: weights[i] * deviations[i] / math.sqrt(costs[i]) for i in variable}
    def charge(multiplier):
        return sum(costs[i] * max(1.0, multiplier * scores.get(i, 0.0)) for i in active)
    lower, upper = 0.0, 1.0
    while charge(upper) < cost_budget:
        upper *= 2
    for _ in range(80):
        middle = (lower + upper) / 2
        if charge(middle) <= cost_budget:
            lower = middle
        else:
            upper = middle
    for i in variable:
        counts[i] = max(1, math.floor(lower * scores[i]))
    remaining = cost_budget - sum(costs[i] * counts[i] for i in active)
    while True:
        affordable = [i for i in variable if costs[i] <= remaining]
        if not affordable:
            return counts
        selected = max(affordable, key=lambda i: (
            (weights[i] * deviations[i]) ** 2 / (counts[i] * (counts[i] + 1) * costs[i]),
            -i,
        ))
        counts[selected] += 1
        remaining -= costs[selected]


def stable_query_id(kind, field_ids):
    """Versioned ID for unary and unordered pair query populations."""
    import hashlib
    import json

    fields = tuple(field_ids)
    if kind == "self" and len(fields) == 1:
        pass
    elif kind == "pair" and len(fields) == 2 and fields[0] != fields[1]:
        fields = tuple(sorted(fields))
    else:
        raise ValueError("invalid unary or pair query")
    if any(not isinstance(x, str) or not x for x in fields):
        raise ValueError("nonempty field IDs required")
    encoded = json.dumps([ADAPTIVE_VERSION, kind, fields], separators=(",", ":"))
    return kind + ":" + hashlib.sha256(encoded.encode()).hexdigest()[:24]
