"""Per-update telemetry remains descriptive and recovery follows durable state."""

import json
from copy import deepcopy

import pytest
import torch

from phycoflow_reconstruction.training.gradient_balance import coherence_primal_dual_update
from phycoflow_reconstruction.training.run_store import RunStore


@pytest.mark.parametrize("clip,coefficient,applied", [
    (1., 1. / (5. + 1.e-6), True), (10., 1., False), (None, 1., False),
])
def test_clip_telemetry_matches_applied_sgd_displacement(clip, coefficient, applied):
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    loss = (model.weight * torch.tensor([[3., 4.]], dtype=torch.float64)).sum()
    row = coherence_primal_dual_update(
        model, optimizer, {"A": loss}, loss * 0.,
        method="weighted_sum", grad_clip=clip, diagnostics=True,
    )
    assert row["gradient/preclip_norm"] == pytest.approx(5.)
    assert row["gradient/clip_coefficient"] == pytest.approx(coefficient)
    assert row["gradient/clipping_applied"] is applied
    assert row["gradient/clip_limit"] == clip
    expected = -.1 * coefficient * torch.tensor([[3., 4.]], dtype=torch.float64)
    assert torch.equal(model.weight, expected)
    assert row["update/actual_dot/A"] == pytest.approx(-2.5 * coefficient)


def test_update_stream_preserves_risk_order_and_trims_replayed_or_torn_rows(tmp_path):
    store = RunStore.create(tmp_path, "telemetry", {"stage": "post_training"})
    caller = {"step": 1, "epoch": 1, "fidelity/p/multiplier": .3,
              "gradient/clipping_applied": True,
              "gradient/A/fidelity/cosine": None,
              "gradient/A/fidelity/cosine/undefined_reason": "zero_gradient",
              "topology/line_sampling": {"selected_indices": [[2, 3]]},
              "reference_ids": ["discard_bulk_nontelemetry"],
              "update/actual_dot/A": -.1}
    original = deepcopy(caller)
    store.append_coherence_update(caller)
    store.append_coherence_update({**caller, "step": 2, "fidelity/p/multiplier": .4})
    store.append_coherence_update({**caller, "step": 3, "fidelity/p/multiplier": .5})
    path = store.run_dir / "metrics/coherence_updates.jsonl"
    with path.open("a") as handle:
        handle.write('{"step": 4,')
    store.recover_metric_histories(2)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2]
    assert [row["fidelity/p/multiplier"] for row in rows] == [.3, .4]
    assert rows[0]["gradient/A/fidelity/cosine"] is None
    assert rows[0]["topology/line_sampling"] == {"selected_indices": [[2, 3]]}
    assert "reference_ids" not in rows[0]
    assert caller == original
    store.append_coherence_update({**caller, "step": 3, "fidelity/p/multiplier": .6})
    assert [json.loads(line)["step"] for line in path.read_text().splitlines()] == [1, 2, 3]
    before = path.read_bytes()
    with pytest.raises(ValueError):
        store.append_coherence_update({"step": 4, "fidelity/p/live_risk": float("nan")})
    assert path.read_bytes() == before


def test_mid_epoch_resume_preserves_every_dual_and_line_sample(tmp_path):
    from helpers.coherence import _base_config, _write_fixture
    from test_upgrade_1002_lifecycle import upgraded_config

    from phycoflow_reconstruction.training.base_training import run_base_training
    from phycoflow_reconstruction.training.post_training import run_post_training

    path = tmp_path / "fixture.h5"
    _write_fixture(path)
    source = run_base_training(_base_config(path), case_dir=tmp_path / "case")
    assert not (source / "metrics/coherence_updates.jsonl").exists()
    post = upgraded_config(path, source)
    post["optimization"]["train_fraction"] = 1.
    post["optimization"]["batch_size"] = 1
    post["coherence"]["compute_budget"]["batch_size"] = 1
    child = run_post_training(post, case_dir=tmp_path / "case", max_steps=1)
    run_post_training(post, case_dir=tmp_path / "case", max_steps=3, resume=child)
    whole = run_post_training(post, case_dir=tmp_path / "whole", max_steps=4)

    def records(run):
        return [json.loads(line) for line in (run / "metrics/coherence_updates.jsonl").read_text().splitlines()]

    resumed, uninterrupted = records(child), records(whole)
    assert [row["step"] for row in resumed] == [1, 2, 3, 4]
    assert [row["step"] for row in uninterrupted] == [1, 2, 3, 4]
    assert resumed[0]["epoch"] == resumed[1]["epoch"]
    for left, right in zip(resumed, uninterrupted):
        keys = [key for key in left if key.startswith(("fidelity/", "loss/", "gradient/", "update/"))]
        assert {key: left[key] for key in keys} == {key: right[key] for key in keys}
        sampling = left["topology/line_sampling"]
        assert sampling == right["topology/line_sampling"]
        assert sampling["global_step"] == left["step"] - 1
        assert len(sampling["selected_indices"][0]) == 2
        assert isinstance(left["gradient/clipping_applied"], bool)
