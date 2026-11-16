from copy import deepcopy

import torch
from test_gl_rbf_cq_coverage import _config, _inputs

from phycoflow_reconstruction.contracts import ObservationBatch
from phycoflow_reconstruction.models.flows.pointcloud.adapters.gl_rbf_cq_adapter import GLRbfCQ
from phycoflow_reconstruction.training.rollout import differentiable_reconstruction


def test_native_rf_checkpointed_every_member_parameter_and_update_parity():
    inputs = _inputs()
    batch = ObservationBatch(
        obs_coords=inputs["obs_coords"], obs_values=inputs["obs_values"],
        obs_field_ids=inputs["obs_field_ids"], obs_valid_mask=inputs["obs_mask"],
        query_coords=inputs["coords"], query_valid_mask=torch.ones(2, 17, dtype=torch.bool),
        target_fields=None, sample_ids=("a", "b"), obs_indices=inputs["obs_indices"],
        logical_shapes=((17,), (17,)), metadata={"query_indices": torch.arange(17).expand(2, -1)},
    )
    torch.manual_seed(378)
    config = _config()
    config.pop("model_name")
    config.pop("coord_dim")
    direct = GLRbfCQ(coordinate_dim=2, num_fields=3, logical_shape=(17,), **config).eval()
    replay = deepcopy(direct)
    replay.rollout_checkpointing = True
    records = []
    for model in (direct, replay):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        members = torch.stack([differentiable_reconstruction(
            model, batch, steps=4, solver="euler",
            generator=torch.Generator().manual_seed(131 + j),
            observation_config={"mode": "endpoint_smooth", "final_clamp": True},
        ) for j in range(2)])
        loss = members.square().mean() + (members[0] * members[1]).mean()
        parameters = tuple(model.parameters())
        gradients = torch.autograd.grad(loss, (*parameters, members), allow_unused=True)
        for parameter, grad in zip(parameters, gradients[:-1]):
            parameter.grad = grad
        optimizer.step()
        records.append((members.detach(), gradients, tuple(p.detach().clone() for p in parameters)))
    for a, b in zip(records[0], records[1]):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
        else:
            for left, right in zip(a, b):
                assert (left is None) == (right is None)
                if left is not None:
                    torch.testing.assert_close(left, right, rtol=3e-5, atol=3e-6)
