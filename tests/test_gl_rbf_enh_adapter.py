"""Full ENH identity, portable numerical parity and resumable trainer lifecycle."""

import copy

import pytest
import torch

from phycoflow_reconstruction.contracts import DataSpec, ObservationBatch
from phycoflow_reconstruction.models import build_model
from phycoflow_reconstruction.models.flows.pointcloud.core.portable_core import (
    ConditionalPointHybridLocalGlobalRBF,
)
from phycoflow_reconstruction.models.flows.pointcloud.runtime import build_pointcloud_model
from phycoflow_reconstruction.training.model_lifecycle import load_training_aux_state


def config():
    return {
        "name": "gl_rbf_enh",
        "backbone": "GL_rbf_ENH",
        "hidden_dim": 16,
        "cond_dim": 16,
        "field_embed_dim": 8,
        "latent_dim": 16,
        "num_latents": 4,
        "num_heads": 4,
        "num_latent_blocks": 1,
        "fourier_pe_num_bands": 2,
        "rff_features": 8,
        "gather_topk": 3,
        "sensor_local_topk": 3,
        "neighbor_backend": "torch",
        "train_query_microbatch_size": 7,
        "reconstruction_query_chunk_size": 7,
    }


def spec():
    return DataSpec(("vorticity",), ("dimensionless",), 2, (21,))


def batch():
    torch.manual_seed(71)
    coords = torch.rand(2, 21, 2)
    target = torch.randn(2, 21, 1)
    slots = torch.tensor([[0, 3, 7, 12, 18], [1, 4, 8, 13, 19]])
    return ObservationBatch(
        obs_coords=torch.stack([coords[i, slots[i]] for i in range(2)]),
        obs_values=torch.stack([target[i, slots[i]] for i in range(2)]),
        obs_field_ids=torch.zeros(2, 5, dtype=torch.long),
        obs_valid_mask=torch.ones(2, 5, dtype=torch.bool),
        query_coords=coords,
        query_valid_mask=torch.ones(2, 21, dtype=torch.bool),
        target_fields=target,
        sample_ids=("a", "b"),
        obs_indices=slots,
        logical_shapes=((21,), (21,)),
        metadata={"query_indices": torch.arange(21).repeat(2, 1)},
    )


def test_full_enh_uses_portable_backbone_and_matches_direct_velocity():
    torch.manual_seed(42)
    model = build_model(config(), spec())
    assert type(model.model) is ConditionalPointHybridLocalGlobalRBF
    assert model.model.enhanced_backbone
    assert not hasattr(model.model, "cq_measurement_support_norm")
    assert model.backbone_name == "GL_rbf_ENH"
    assert not any(key.startswith("cq_") for key in model.portable_config)
    torch.manual_seed(42)
    direct = build_pointcloud_model(model.portable_config, n_fields=1)
    assert model.state_dict().keys() == direct.state_dict().keys()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, direct.state_dict()[key], rtol=0, atol=0)
    b = batch()
    state, time = torch.randn_like(b.target_fields), torch.tensor([0.2, 0.7])
    torch.testing.assert_close(
        model.velocity(b, state, time),
        direct.model(
            time,
            state,
            b.query_coords,
            b.obs_coords,
            b.obs_values,
            b.obs_valid_mask,
            b.obs_field_ids,
        ),
        rtol=0,
        atol=0,
    )


def test_enh_microbatch_gradients_and_ema_checkpoint_round_trip():
    model = build_model(config(), spec())
    full = copy.deepcopy(model)
    b = batch()
    torch.manual_seed(81)
    chunk_loss = model.training_backward(b, loss_scale=1.0)
    torch.manual_seed(81)
    loss, _ = full.training_loss_microbatched(
        x1=b.target_fields,
        coords=b.query_coords,
        obs_coords=b.obs_coords,
        obs_values=b.obs_values,
        obs_mask=b.obs_valid_mask,
        obs_field_ids=b.obs_field_ids,
        query_microbatch_size=21,
        backward=True,
    )
    torch.testing.assert_close(chunk_loss.total, loss, rtol=2e-6, atol=2e-7)
    for (name, p), (other_name, q) in zip(model.named_parameters(), full.named_parameters()):
        assert name == other_name
        if p.grad is None or q.grad is None:
            assert p.grad is None and q.grad is None, name
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=5e-4, atol=2e-6)
    torch.optim.AdamW(model.parameters(), lr=1e-4).step()
    model.after_optimizer_step()
    restored = build_model(config(), spec())
    restored.load_state_dict(model.state_dict(), strict=True)
    load_training_aux_state(restored, {"training_aux_state": model.training_aux_state_dict()})
    assert restored._ema.num_updates == 1
    predictions = []
    for m in (model, restored):
        m.eval()
        with torch.no_grad(), m.evaluation_weight_context():
            p = m.reconstruct(
                b, steps=2, generator=torch.Generator().manual_seed(91), obs_consistency_mode="none"
            ).prediction
            assert torch.isfinite(p).all()
            predictions.append(p)
    torch.testing.assert_close(*predictions, rtol=0, atol=0)


@pytest.mark.parametrize(
    "override",
    [
        {"backbone": "GL_rbf_ENH_CQ"},
        {"cq_measurement_support_normalize": False},
    ],
)
def test_full_enh_rejects_mislabeled_cq_configuration(override):
    cfg = config()
    cfg.update(override)
    with pytest.raises(ValueError):
        build_model(cfg, spec())
