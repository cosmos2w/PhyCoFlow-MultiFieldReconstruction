import torch

from scripts.qualification.car_round7_support_fixture import compare, risk


def test_support_comparison_distinguishes_reproducibility_from_query_effect():
    original = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[2.0, 3.0], [4.0, 5.0]]]])
    repeated = original + 1e-7
    assert compare(original, repeated, repeated, repeated)['all_within_repeatability_tolerance']
    changed = original.clone()
    changed[0, 0, 1, 0] += 0.1
    result = compare(original, changed, repeated, changed)
    assert not result['all_within_repeatability_tolerance']
    assert result['field_max_abs_difference'][0] > 0.09
    assert result['field_max_abs_difference'][1] == 0


def test_signed_risk_and_individual_mse_use_declared_inclusion_weights():
    members = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    truth = torch.zeros((1, 2, 1))
    result = risk(members, truth, torch.tensor([1.0, 2.0]), torch.tensor(5.0),
                  torch.tensor([1.0, 0.5]))
    torch.testing.assert_close(torch.tensor(result['member_mse_each']), torch.tensor([[3.4], [14.6]]))
    torch.testing.assert_close(torch.tensor(result['member_mse_mean']), torch.tensor([9.0]))
    torch.testing.assert_close(torch.tensor(result['signed_mean_risk']), torch.tensor([7.0]))
