"""The broad panel must retain the original nonlinear ensemble estimator."""

from dataclasses import dataclass

import pytest
import torch

from phycoflow_reconstruction.evaluation.fixed_groups import evaluate_fixed_groups


@dataclass
class Batch:
    values: torch.Tensor
    sample_ids: tuple
    logical_shapes: tuple
    metadata: dict
    query_valid_mask: torch.Tensor | None = None

    def validate(self):
        assert len(self.sample_ids) == self.values.shape[0]
        if self.query_valid_mask is not None:
            assert self.query_valid_mask.shape[0] == self.values.shape[0]


def test_panel_averages_exact_group_scores_and_keeps_raw_identities():
    values = torch.arange(128, dtype=torch.float32)
    batch = Batch(
        values,
        tuple(map(str, range(128))),
        (),
        {
            "query_indices": torch.arange(128).reshape(128, 1),
            "sample_context": {"value": values},
        },
        torch.ones(128, 1, dtype=torch.bool),
    )
    seen = []

    def evaluate(model, complete, comparison, family, bank, names, config, **kwargs):
        assert "group_size" not in config["evaluation"]
        seen.append((complete.sample_ids, config["evaluation"]["seed"]))
        assert torch.equal(complete.values, complete.metadata["sample_context"]["value"])
        return {
            "coherence": {"total": float(complete.values.var(unbiased=False))},
            "sample_ids": list(complete.sample_ids),
            "query_ids": None,
            "inference": {"seconds": 1.0, "samples": len(complete.sample_ids)},
        }

    result = evaluate_fixed_groups(
        evaluate,
        None,
        batch,
        batch,
        None,
        None,
        (),
        {"evaluation": {"group_size": 32, "seed": 2027}},
    )
    assert [len(ids) for ids, seed in seen] == [32] * 4
    assert [seed for ids, seed in seen] == [2027 + index * 100_003 for index in range(4)]
    assert result["coherence"]["total"] == float(values[:32].var(unbiased=False))
    assert result["coherence"]["total"] != float(values.var(unbiased=False))
    assert len(result["fixed_estimator_groups"]) == 4
    assert result["sample_ids"] == list(batch.sample_ids)
    assert result["inference"]["seconds"] == 4.0


def test_partial_covariance_group_is_rejected():
    batch = Batch(
        torch.arange(33), tuple(map(str, range(33))), (), {}, torch.ones(33, 1, dtype=torch.bool)
    )
    with pytest.raises(ValueError, match="exact group size"):
        evaluate_fixed_groups(
            None, None, batch, batch, None, None, (), {"evaluation": {"group_size": 32}}
        )


def test_metric_pooling_weights_valid_queries_and_recomputes_worst_field():
    values = torch.zeros(4, 3)
    valid = torch.tensor(
        [[True, True, False], [True, True, False], [True, False, False], [True, False, False]]
    )
    batch = Batch(
        values,
        ("0", "1", "2", "3"),
        (),
        {"query_indices": torch.arange(12).reshape(4, 3)},
        valid,
    )

    def evaluate(_model, complete, _comparison, *_args, **_kwargs):
        high_group = complete.sample_ids[0] == "2"
        return {
            "mse_normalized": 100.0 if high_group else 10.0,
            "per_field_mse_normalized": {
                "u": 10.0 if high_group else 1.0,
                "v": 100.0 if high_group else 3.0,
            },
            "mse_physical": 200.0 if high_group else 20.0,
            "per_field_mse_physical": {
                "u": 20.0 if high_group else 2.0,
                "v": 200.0 if high_group else 6.0,
            },
            "endpoint_mse_model_units": {
                "u": 5.0 if high_group else 1.0,
                "v": 7.0 if high_group else 3.0,
            },
            "per_field_relative_l2": {
                "u": 0.9 if high_group else 0.1,
                "v": 0.1 if high_group else 0.9,
            },
            "worst_field_relative_l2": 0.9,
            "observed_entry_mse_normalized": 20.0 if high_group else 2.0,
            "unobserved_entry_mse_normalized": 40.0 if high_group else 4.0,
            "inference": {"seconds": 1.0, "samples": 2},
            "sample_ids": list(complete.sample_ids),
        }

    result = evaluate_fixed_groups(
        evaluate,
        None,
        batch,
        batch,
        None,
        None,
        (),
        {"evaluation": {"group_size": 2}},
    )

    assert result["mse_normalized"] == pytest.approx(40.0)
    assert result["per_field_mse_normalized"] == pytest.approx(
        {"u": 4.0, "v": 35.0 + 1.0 / 3.0}
    )
    assert result["mse_physical"] == pytest.approx(80.0)
    assert result["per_field_mse_physical"] == pytest.approx(
        {"u": 8.0, "v": 70.0 + 2.0 / 3.0}
    )
    assert result["endpoint_mse_model_units"] == pytest.approx(
        {"u": 7.0 / 3.0, "v": 13.0 / 3.0}
    )
    assert result["per_field_relative_l2"] == pytest.approx({"u": 0.5, "v": 0.5})
    assert result["worst_field_relative_l2"] == pytest.approx(0.5)
    assert "observed_entry_mse_normalized" not in result
    assert "unobserved_entry_mse_normalized" not in result
    assert result["fixed_estimator_groups"][0]["observed_entry_mse_normalized"] == 2.0
    assert result["fixed_estimator_groups"][1]["unobserved_entry_mse_normalized"] == 40.0
