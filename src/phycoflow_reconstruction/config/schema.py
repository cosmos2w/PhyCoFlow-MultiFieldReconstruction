"""Stage schemas kept in Python for strict runtime validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageSchema:
    required: frozenset[str]
    forbidden: frozenset[str]
    requires_one_of: frozenset[str] = frozenset()


STAGE_SCHEMAS = {
    "base_training": StageSchema(
        required=frozenset(
            {"case", "dataset", "model", "observations", "optimization", "runtime", "output"}
        ),
        forbidden=frozenset({"source_run", "source_checkpoint", "coherence", "physics"}),
    ),
    "post_training": StageSchema(
        required=frozenset(
            {
                "case",
                "dataset",
                "model",
                "observations",
                "source_run",
                "source_checkpoint",
                "optimization",
                "runtime",
                "output",
            }
        ),
        forbidden=frozenset(),
        requires_one_of=frozenset({"coherence", "physics"}),
    ),
    "direct_physics": StageSchema(
        required=frozenset(
            {
                "case",
                "dataset",
                "model",
                "observations",
                "physics",
                "optimization",
                "runtime",
                "output",
            }
        ),
        forbidden=frozenset({"source_run", "source_checkpoint", "coherence"}),
    ),
}
