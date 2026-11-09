"""Checkpointable, independent query selection using detached Scale losses."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QueryDraw:
    query_id: tuple
    kind: str
    fields: tuple
    sign: tuple
    line: int | None
    probability: float
    target_weight: float
    multiplicity: int
    estimator_weight: float


@dataclass(frozen=True)
class QuerySelection:
    queries: tuple[QueryDraw, ...]
    self_draws: int
    mutual_draws: int
    bank_manifest: dict | None = None


class AdaptiveCoverageSampler:
    def __init__(self, bank, policy="uniform", seed=0, window_epochs=10, decay=0.9):
        if policy not in ("uniform", "adaptive") or window_epochs <= 0 or not 0 <= decay < 1:
            raise ValueError("invalid query coverage policy")
        self.bank, self.policy = bank, policy
        self.window_epochs, self.decay = int(window_epochs), float(decay)
        self.rng = random.Random(seed)
        self.epoch = 0
        self.scores = {"self": [0.0] * len(bank.field_ids), "mutual": [0.0] * len(bank.pairs)}
        self.sums = {k: [0.0] * len(v) for k, v in self.scores.items()}
        self.observations = 0
        self.probabilities = {k: self._prob(v) for k, v in self.scores.items()}
        self.exposure = {}

    @staticmethod
    def _prob(scores):
        if not scores:
            return []
        total = sum(scores)
        n = len(scores)
        return [0.5 / n + 0.5 * (x / total if total else 1 / n) for x in scores]

    def select(self, self_draws=1, mutual_draws=3):
        if self_draws < 0 or mutual_draws < 0 or (self_draws == 0 and mutual_draws == 0):
            raise ValueError("invalid query draw counts")
        if not self.bank.pairs:
            mutual_draws = 0
        line_weights = [1 / math.sqrt(2), 1 / math.sqrt(5), 1 / math.sqrt(5)]
        line_weights = [x / sum(line_weights) for x in line_weights]
        signs = ((1, 1), (-1, -1), (1, -1), (-1, 1))
        draws = {}
        for kind, count in (("self", self_draws), ("mutual", mutual_draws)):
            population = self.bank.field_ids if kind == "self" else self.bank.pairs
            for _ in range(count):
                idx = self.rng.choices(range(len(population)), weights=self.probabilities[kind])[0]
                if kind == "self":
                    fields, sign, line = (population[idx],), (self.rng.choice((1, -1)),), None
                    p = self.probabilities[kind][idx] / 2
                    target = 1 / (2 * len(population))
                else:
                    fields, sign = population[idx], self.rng.choice(signs)
                    line = self.rng.choices(range(3), weights=line_weights)[0]
                    p = self.probabilities[kind][idx] * line_weights[line] / 4
                    target = line_weights[line] / (4 * len(population))
                key = (kind, fields, sign, line)
                if key not in draws:
                    draws[key] = [p, target, 0, count]
                draws[key][2] += 1
                self.exposure[key] = self.exposure.get(key, 0) + 1
        queries = tuple(
            QueryDraw(key, *key, p, target, multiplicity, multiplicity * target / (p * count))
            for key, (p, target, multiplicity, count) in draws.items()
        )
        return QuerySelection(queries, self_draws, mutual_draws, self.bank.manifest())

    def observe(self, self_losses, pair_losses):
        for kind, values in (("self", self_losses), ("mutual", pair_losses)):
            values = torch.as_tensor(values).detach().cpu().reshape(-1).tolist()
            if len(values) != len(self.sums[kind]) or any(
                not math.isfinite(x) or x < 0 for x in values
            ):
                raise ValueError(
                    "Scale priority losses must be finite nonnegative bank-aligned values"
                )
            self.sums[kind] = [a + b for a, b in zip(self.sums[kind], values)]
        self.observations += 1

    def finalize_epoch(self):
        self.epoch += 1
        if self.epoch % self.window_epochs == 0:
            if self.observations:
                for kind in self.scores:
                    means = [x / self.observations for x in self.sums[kind]]
                    self.scores[kind] = [
                        self.decay * x + (1 - self.decay) * y
                        for x, y in zip(self.scores[kind], means)
                    ]
                    if self.policy == "adaptive":
                        self.probabilities[kind] = self._prob(self.scores[kind])
            self.sums = {k: [0.0] * len(v) for k, v in self.sums.items()}
            self.observations = 0

    def state_dict(self):
        from copy import deepcopy

        return deepcopy(
            {
                "manifest": self.bank.manifest(),
                "policy": self.policy,
                "window_epochs": self.window_epochs,
                "decay": self.decay,
                "rng": self.rng.getstate(),
                "epoch": self.epoch,
                "scores": self.scores,
                "sums": self.sums,
                "observations": self.observations,
                "probabilities": self.probabilities,
                "exposure": self.exposure,
            }
        )

    def load_state_dict(self, state):
        from copy import deepcopy

        if (
            state["manifest"] != self.bank.manifest()
            or state["policy"] != self.policy
            or state["window_epochs"] != self.window_epochs
            or state["decay"] != self.decay
        ):
            raise ValueError("query coverage checkpoint identity mismatch")
        state = deepcopy(state)
        self.rng.setstate(state["rng"])
        for key in ("epoch", "scores", "sums", "observations", "probabilities", "exposure"):
            setattr(self, key, state[key])
