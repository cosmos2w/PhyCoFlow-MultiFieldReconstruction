from copy import deepcopy
import torch
from phycoflow_reconstruction.training.direct_supervision import direct_scalar, finite_field_action


def test_transaction_reject_then_accept_and_half_action():
    model = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad(): model.weight.fill_(1.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1, weight_decay=.2)
    old = deepcopy(model.state_dict())
    # The proposal improves the scalar but the paired field constraint rejects both scales.
    result = finite_field_action(model, optimizer, model.weight.square().sum(), torch.tensor([0.]), [1.], [0.], lambda: torch.tensor([1.]))
    assert not result['update_accepted'] and result['action_dot'] < 0
    assert torch.equal(model.weight, old['weight']) and not optimizer.state
    # A full step fails, but a half step is accepted; Adam moments advance once.
    def risk():
        return torch.tensor([.2 if model.weight.item() < .94 else 0.])
    result = finite_field_action(model, optimizer, model.weight.square().sum(), torch.tensor([0.]), [1.], [0.], risk)
    assert result['update_accepted'] and result['action_alpha'] == .5
    assert torch.allclose(model.weight, torch.tensor([[.94]], dtype=torch.double))
    assert next(iter(optimizer.state.values()))['step'].item() == 1


def test_native_state_scale_shape_accumulation_order():
    class Family: family_weight = 1.
    families = {k: Family() for k in ('global_distribution', 'multiscale_spectrum', 'topology')}
    native = torch.tensor(8.)
    losses = {k: torch.tensor(float(i)) for i, k in enumerate(families, 1)}
    result = direct_scalar(native, losses, {'native': 2, 'state': 1, 'scale': 1, 'shape': 1}, {k: 1 for k in families}, families)
    assert result.item() == 6.


def test_exception_rolls_back_proposal():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
    old = model.weight.detach().clone()
    def fail():
        raise RuntimeError('replay failed')
    import pytest
    with pytest.raises(RuntimeError, match='replay failed'):
        finite_field_action(model, optimizer, model.weight.square().sum(), torch.tensor([0.]), [1.], [0.], fail)
    assert torch.equal(model.weight, old) and not optimizer.state
