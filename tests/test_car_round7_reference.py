from types import SimpleNamespace

import torch

from phycoflow_reconstruction.coherence.physical_risk import physical_ensemble_risks
from scripts.qualification.car_round7_training_reference import reference_from_moments, stratified_risks


def test_reference_uses_new_member_scale_and_floor_does_not_enlarge_allowance():
    reference = reference_from_moments(["ordinary", "small"], [-0.03, -1e-10], [0.2, 1e-10], 1.8591270517644938)
    assert reference.risk_scales == (0.2, 1e-8)
    assert reference.absolute_allowance == (0.020000000000000004, 1.0000000000000001e-11)
    torch.testing.assert_close(reference.violations(torch.tensor(reference.reference_risks)),
        torch.tensor([-0.1, -0.001], dtype=torch.float64), rtol=1e-6, atol=1e-9)


def test_reference_member_and_signed_risk_match_native_integral_in_expectation():
    members = torch.tensor([[[[1.0], [3.0], [7.0]],
                              [[2.0], [1.0], [4.0]]]], dtype=torch.float64)
    truth = torch.tensor([[[0.5], [2.0], [5.0]]], dtype=torch.float64)
    dataset = SimpleNamespace(physical_weights=torch.tensor([1.0, 3.0, 2.0], dtype=torch.float64))
    native = physical_ensemble_risks(members, truth, dataset.physical_weights)
    draws = []
    for complement in (1, 2):
        endpoint = {
            "prediction": members[:, 0, :1],
            "members": tuple(members[:, member, :1] for member in range(2)),
            "target": truth[:, :1],
            "coupling_selected_batch": SimpleNamespace(metadata={"query_indices": torch.tensor([[0]])}),
            "complement_indices": torch.tensor([[complement]]),
            "complement_members": tuple(members[:, member, complement:complement+1] for member in range(2)),
            "complement_target": truth[:, complement:complement+1],
        }
        draws.append(stratified_risks(endpoint, dataset))
    for index in (0, 1):
        torch.testing.assert_close(torch.stack([draw[index] for draw in draws]).mean(0), native[index])
