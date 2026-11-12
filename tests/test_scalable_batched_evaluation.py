"""Independent sample scores preserve the full panel under memory batching."""

import pytest
import torch

from phycoflow_reconstruction.contracts import ObservationBatch
from phycoflow_reconstruction.evaluation.metrics import reconstruction_metrics
from phycoflow_reconstruction.evaluation.scalable_batches import evaluate_scalable_batches


def test_uneven_batches_preserve_sample_and_observed_entry_weighting():
    target = torch.arange(1, 16, dtype=torch.float32).reshape(5, 3, 1)
    batch = ObservationBatch(
        obs_coords=torch.zeros(5, 3, 1),
        obs_values=torch.zeros(5, 3, 1),
        obs_field_ids=torch.zeros(5, 3, dtype=torch.long),
        obs_valid_mask=torch.tensor([[1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1]]).bool(),
        query_coords=torch.arange(3).float()[None, :, None].expand(5, -1, -1),
        query_valid_mask=torch.ones(5, 3, dtype=torch.bool),
        target_fields=target,
        obs_indices=torch.arange(3)[None].expand(5, -1),
        sample_ids=tuple(map(str, range(5))),
        logical_shapes=((3,),) * 5,
        metadata={"query_indices": torch.arange(3)[None].expand(5, -1)},
    )
    seen = []

    def evaluate(_model, complete, comparison, _family, _bank, names, config, **kwargs):
        seen.append((config["evaluation"]["seed"], kwargs["_generation_seed"]))
        assert "batch_size" not in config["evaluation"]
        prediction = 2 * comparison.target_fields
        result = reconstruction_metrics(
            prediction, comparison.target_fields, comparison, names, include_counts=True
        )
        total = float(comparison.target_fields.square().mean())
        result.update(
            {
                "physical_member_mse_model_units": {"u": total},
                "coherence": {
                    "total": total,
                    "reference_ids": list(complete.sample_ids),
                    "families": {
                        "state": {
                            "total": total,
                            "weighted_total": 2 * total,
                            "reference_ids": list(complete.sample_ids),
                            "diagnostics": {"observer_sha256": str(len(seen))},
                        }
                    },
                },
                "inference": {
                    "seconds": 1.0,
                    "conditional_extra_seconds": 2.0,
                    "all_member_seconds": 3.0,
                    "samples": len(complete.sample_ids),
                },
                "sample_ids": list(complete.sample_ids),
                "query_ids": complete.metadata["query_indices"].tolist(),
            }
        )
        return result

    result = evaluate_scalable_batches(
        evaluate,
        None,
        batch,
        batch,
        None,
        None,
        ("u",),
        {"evaluation": {"batch_size": 2, "seed": 17}},
    )
    expected = reconstruction_metrics(2 * target, target, batch, ("u",), include_counts=True)
    for key, value in expected.items():
        assert result[key] == pytest.approx(value)
    assert result["coherence"]["total"] == pytest.approx(float(target.square().mean()))
    assert result["physical_member_mse_model_units"] == pytest.approx(
        {"u": float(target.square().mean())}
    )
    assert result["sample_ids"] == list(batch.sample_ids)
    assert result["query_ids"] == batch.metadata["query_indices"].tolist()
    assert result["inference"]["all_member_seconds"] == 9.0
    assert result["evaluation_batching"]["batch_sizes"] == [2, 2, 1]
    assert seen == [(17, 17), (17, 100020), (17, 200023)]
    assert result["coherence"]["families"]["state"]["diagnostics"]["observer_sha256s"] == [
        "1",
        "2",
        "3",
    ]
