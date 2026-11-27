import numpy as np
import torch

from scripts.analysis.car_round8_cached_summary import summarize
from scripts.analysis.car_round8_support_readout import native_support_risks


def test_native_risk_uses_total_physical_mass_and_affine_scale():
    # A deliberately nonuniform measure catches self-normalization of the
    # fixed support; offsets cancel in physical errors.
    row = {"members_model": torch.tensor([[[[1.], [2.], [3.], [4.]]]] * 4),
           "truth_model": torch.zeros(1, 4, 1), "normalizer_scale": torch.tensor([2.]),
           "normalizer_offset": torch.tensor([100.]), "native_masses": torch.tensor([1., 2., 3., 4.])}
    result = native_support_risks(row, torch.tensor([0, 1]), seed=5)
    expected_native = 4 * (1 + 8 + 27 + 64) / 10
    assert np.allclose(result["native"], [expected_native])
    assert np.allclose(result["fixed_contribution_to_native"], [4 * 9 / 10])
    assert np.allclose(result["complement_contribution_to_native"], [4 * 91 / 10])
    assert np.allclose(result["fixed_self_normalized"], [4 * 9 / 3])
    # All complement points fit in the 512-point design here.
    assert np.allclose(result["fixed_plus_uniform_complement512_estimate"], [expected_native])


def test_cached_summary_pools_conditions_before_dividing_and_retains_old_verdict():
    ids = ["frame-0", "frame-1"]
    def fields(member):
        return {"member_mse": [member], "ensemble_mean_mse": [member - .75],
                "unbiased_member_variance": [1.], "signed_mean_risk": [member - 1.], "sensor_mse": [0.]}
    def score():
        row = {"shape": 2.0}
        for kind in ("self", "mutual"):
            for category in ("finite", "essential"):
                row[f"shape.{kind}.{category}.attraction"] = 1.0
                row[f"shape.{kind}.{category}.separation"] = .5
        return row
    assessment = {"status": "completed", "split": "validation", "sample_ids": ids, "members": 4,
        "field_order": ["field"], "field_units": ["unknown"], "selection_use": "development",
        "per_condition_fields": {"SOURCE": [fields(1), fields(9)], "candidate": [fields(2), fields(9)]},
        "per_condition_scores": {"SOURCE": [score(), score()], "candidate": [score(), score()]},
        "native_allowance_physical": [.25], "native_safeguards": {"field_difference": [.5]}}
    blocks = ("signed", "joint", "log_power_self", "rho", "third_order")
    def model():
        return {"blocks": {k: {"attraction": 2., "separation": 1., "score": 1.} for k in blocks},
                "definitions": {"Published_Core_Default": 3.25, "Complete_Expanded_Prefit": 4.25}}
    decomposition = {"status": "completed", "split": "validation", "sample_ids": ids, "members": 4,
        "field_order": ["field"], "prefit_coefficients": {"scale." + k: 1. for k in blocks},
        "records": [{"sample_id": sid, "models": {"SOURCE": model(), "candidate": model()}} for sid in ids]}
    result = summarize(assessment, decomposition, bootstrap=100, seed=2)
    field = result["fields"][0]
    assert np.isclose(field["member_ratio"], 11/10)
    assert field["member_point_pass"]
    assert not field["old_signed_point_pass"]
    assert result["example_selection"]["representative_median"]["sample_id"] == "frame-0"
    assert result["example_selection"]["difficult_maximum"]["sample_id"] == "frame-1"
