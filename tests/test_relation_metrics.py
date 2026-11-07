import pytest
import torch

from phycoflow_reconstruction.evaluation.relation_metrics import (
    assess_candidate,
    bridge_comparison,
    confirm_checkpoint,
    relation_coordinates,
    relation_errors,
    select_checkpoint,
    train_floors,
)


def gram(a, b, cross):
    return torch.tensor([[[[a, cross], [cross, b]]]], dtype=torch.float64)


def test_native_relation_metrics_use_all_members_signed_pairs_and_train_floors():
    scales = torch.tensor([[2.0, 4.0]], dtype=torch.float64)
    floor = torch.tensor([[1.0, 4.0]], dtype=torch.float64)
    truth = gram(4, 16, 4)
    members = torch.stack([truth, gram(8, 16, -4)])
    coordinates = relation_coordinates(truth, scales, floor)
    assert coordinates["self"].flatten().tolist() == [1, 1]
    assert coordinates["mutual"].item() == pytest.approx(0.5)
    assert coordinates["rho"].item() == pytest.approx(4 / 10)
    errors = relation_errors(members, truth, scales, floor)
    assert errors["self"]["squared"].shape == (2, 1, 1, 2)
    assert errors["self"]["mean"] == pytest.approx(0.25)
    assert errors["mutual"]["mean"] == pytest.approx(0.5)
    assert errors["rho"]["mean"] == pytest.approx(((-4 / (9 * 20) ** 0.5) - 0.4) ** 2 / 2)
    floors = train_floors(truth, scales, floor)
    assert floors == pytest.approx({"self": 0.001, "mutual": 0.00025, "rho": 0.00016})


def test_single_field_is_available_for_self_but_not_mutual_adoption():
    x = torch.ones(1, 1, 1, 1, dtype=torch.float64)
    errors = relation_errors(x[None], x, x[0, :, :, 0], x[0, :, :, 0])
    assert errors["mutual"]["mean"] is None
    assert errors["rho"]["squared"].numel() == 0


def item(checkpoint="c", epoch=200, **values):
    base = {"checkpoint": checkpoint, "epoch": epoch, "self": 0.5, "mutual": 0.4,
            "rho": 0.4, "field_mse": [1.0, 1.0], "signed": 1.0, "state": 1.0,
            "shape": 1.0}
    base.update(values)
    return base


def test_selection_maturity_guards_ranking_and_confirmation_cannot_rerank():
    source = item(self=0.5, mutual=0.5, rho=0.5)
    floors = dict.fromkeys(("self", "mutual", "rho"), 0.001)
    early = item("early", 100, mutual=0.1)
    field_fail = item("fail", 300, mutual=0.1, field_mse=[1, 1.10001])
    best = item("best", 450, mutual=0.3)
    other = item("other", 500, mutual=0.4)
    selection = select_checkpoint([early, field_fail, other, best], source, floors)
    assert selection["checkpoint"] == "best"
    assert "early" not in selection["assessments"]
    assert not selection["assessments"]["fail"]["guards"]["field"]
    assert confirm_checkpoint(selection, best, source, floors, [1, 1], [1, 1])["deployment"] == "best"
    assert confirm_checkpoint(selection, best, source, floors, [1, 1.11], [1, 1])["deployment"] == "SOURCE"
    with pytest.raises(ValueError, match="frozen"):
        confirm_checkpoint(selection, other, source, floors, [1, 1], [1, 1])


def test_no_headroom_preservation_requires_other_mutual_gain():
    floors = {"self": 0.001, "mutual": 0.1, "rho": 0.1}
    source = item(mutual=0.05, rho=0.5)
    assert assess_candidate(item(mutual=0.052, rho=0.49), source, floors)["eligible"]
    assert not assess_candidate(item(mutual=0.05201, rho=0.49), source, floors)["eligible"]
    assert not assess_candidate(item(mutual=0.05, rho=0.49), item(mutual=0.05, rho=0.05), floors)["eligible"]


def test_bridge_compares_change_on_matching_absolute_arrays():
    s = torch.tensor([[1.0]])
    def result(v):
        return {h: {"squared": s * v} for h in ("self", "mutual", "rho")}
    bridge = bridge_comparison(result(1), result(0.8), result(2), result(1.6),
                               dict.fromkeys(("self", "mutual", "rho"), 0.01))
    assert bridge["max_paired_change_error"] == pytest.approx(0.1)
    assert bridge["queries"]["rho"]["native_child_squared"].item() == pytest.approx(1.6)
