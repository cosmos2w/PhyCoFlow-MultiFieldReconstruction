"""Versioned implicit all-field queries and unbiased, bounded work estimates.

Sampling is independent of model graphs. Call ``draw`` once and pass the same
batch to both families; call ``finish`` only after observing detached difficulties.
"""
from __future__ import annotations

from collections import OrderedDict, Counter
from dataclasses import dataclass, replace
import math
import random

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
    def __init__(self, field_ids, *, seed=0, mutual_draws=3, epsilon=0.5,
                 decay=0.9, capacity=None, refresh_epochs=10):
        self.fields = tuple(field_ids)
        if not self.fields or any(not isinstance(x, str) or not x for x in self.fields) or len(set(self.fields)) != len(self.fields):
            raise ValueError("unique nonempty field IDs required")
        if mutual_draws < 1 or not 0 < epsilon <= 1 or not 0 <= decay < 1 or refresh_epochs < 1:
            raise ValueError("invalid sampler settings")
        self.mutual_draws, self.epsilon, self.decay = int(mutual_draws), float(epsilon), float(decay)
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
        return tuple(Draw(i, k, self.epsilon / population + (1 - self.epsilon) * dist.get(i, 0.0) if ids else 1 / population, population, count) for i, k in counts.items())

    def draw(self):
        if self.pending is not None:
            raise RuntimeError("finish pending batch before drawing")
        self.counter += 1
        self_draws = self._draw("self", len(self.fields), 1)
        pair_draws = self._draw("pair", pair_count(len(self.fields)), self.mutual_draws)
        # Inner draws use a separate checkpointed RNG and do not alter the
        # historical outer-ID stream. Duplicate outer IDs share one inner draw.
        selected_self = tuple(replace(d, inner_sign=(self.inner_rng.choice((1, -1)),),
                                      inner_probability=0.5) for d in self_draws)
        selected_pairs = []
        for d in pair_draws:
            sign = self.inner_rng.choice(PAIR_SIGNS)
            line = self.inner_rng.choices(range(3), weights=LINE_WEIGHTS, k=1)[0]
            selected_pairs.append(replace(d, inner_sign=sign, inner_line=line,
                                          inner_probability=LINE_WEIGHTS[line] / 4))
        self.pending = QueryBatch(selected_self, tuple(selected_pairs), self.counter)
        return self.pending

    def finish(self, *, self_difficulties=None, pair_difficulties=None, exposure_sink=None):
        """Difficulties map sampled index to (Scale, Shape), already TRAIN-normalized."""
        if self.pending is None:
            raise RuntimeError("no pending batch")
        for kind, rows, observations in (("self", self.pending.self_queries, self_difficulties),
                                          ("pair", self.pending.pair_queries, pair_difficulties)):
            observations = observations or {}
            if set(observations) - {d.index for d in rows}:
                raise ValueError("difficulty IDs must belong to pending batch")
            for d in rows:
                if exposure_sink is not None:
                    exposure_sink({"batch": self.pending.number, "kind": kind, "index": d.index, "count": d.count,
                                   "probability": d.probability, "inner_sign": list(d.inner_sign),
                                   "inner_line": d.inner_line, "inner_probability": d.inner_probability,
                                   "query_protocol": VERSION, "shape_sampling": SHAPE_SAMPLING_VERSION,
                                   "inner_seed": self.inner_seed})
                if d.index in observations:
                    scale, shape = (float(x) for x in observations[d.index])
                    if not math.isfinite(scale) or not math.isfinite(shape):
                        raise ValueError("difficulty must be finite and detached")
                    value = (max(0., scale) + max(0., shape)) / 2
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
            self.distribution[kind] = {i: v / total for i, v in cache.items() if v > 0} if total else {}
        self.last_refresh = int(completed_epoch)
        return True

    def state_dict(self):
        def encode(rows): return [vars(x).copy() for x in rows]
        return {"version": VERSION, "fields": self.fields, "mutual_draws": self.mutual_draws,
                "epsilon": self.epsilon, "decay": self.decay, "capacity": self.capacity,
                "refresh_epochs": self.refresh_epochs, "rng": self.rng.getstate(),
                "inner_seed": self.inner_seed, "inner_rng": self.inner_rng.getstate(),
                "shape_sampling": SHAPE_SAMPLING_VERSION,
                "cache": {k: list(v.items()) for k, v in self.cache.items()},
                "distribution": {k: list(v.items()) for k, v in self.distribution.items()},
                "last_refresh": self.last_refresh, "counter": self.counter,
                "pending": None if self.pending is None else {"self_queries": encode(self.pending.self_queries), "pair_queries": encode(self.pending.pair_queries), "number": self.pending.number}}

    @classmethod
    def from_state_dict(cls, state):
        if state["version"] != VERSION:
            raise ValueError("incompatible sampler version")
        obj = cls(state["fields"], mutual_draws=state["mutual_draws"], epsilon=state["epsilon"], decay=state["decay"], capacity=state["capacity"], refresh_epochs=state["refresh_epochs"])
        def tuples(x): return tuple(tuples(y) for y in x) if isinstance(x, (list, tuple)) else x
        if state.get("shape_sampling") != SHAPE_SAMPLING_VERSION:
            raise ValueError("incompatible Shape inner sampling protocol")
        obj.rng.setstate(tuples(state["rng"]))
        obj.inner_seed = int(state["inner_seed"])
        obj.inner_rng.setstate(tuples(state["inner_rng"]))
        obj.cache = {k: OrderedDict((int(i), float(v)) for i, v in rows) for k, rows in state["cache"].items()}
        obj.distribution = {k: {int(i): float(v) for i, v in rows} for k, rows in state["distribution"].items()}
        obj.last_refresh, obj.counter = state["last_refresh"], state["counter"]
        p = state["pending"]
        if p is not None:
            def restore_draw(d):
                data = dict(d)
                if data.get("inner_sign") is not None: data["inner_sign"] = tuple(data["inner_sign"])
                return Draw(**data)
            obj.pending = QueryBatch(tuple(restore_draw(d) for d in p["self_queries"]), tuple(restore_draw(d) for d in p["pair_queries"]), p["number"])
            for draw in obj.pending.self_queries:
                if draw.inner_sign not in ((1,), (-1,)) or draw.inner_line is not None or draw.inner_probability != .5:
                    raise ValueError("invalid pending self inner draw")
            for draw in obj.pending.pair_queries:
                line = draw.inner_line
                if (draw.inner_sign not in PAIR_SIGNS or isinstance(line, bool) or not isinstance(line, int)
                        or not 0 <= line < len(LINE_WEIGHTS) or draw.inner_probability is None
                        or not math.isclose(draw.inner_probability, LINE_WEIGHTS[line] / 4, rel_tol=1e-12)):
                    raise ValueError("invalid pending pair inner draw")
        return obj
