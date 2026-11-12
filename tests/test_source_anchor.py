import pytest
import torch

from phycoflow_reconstruction.evaluation.source_anchor import blend_matched_outputs


def test_matched_train_qualified_blend_and_rejection():
    source = torch.zeros(2, 3, 1)
    child = torch.ones_like(source, requires_grad=True)
    coupling = {
        "observations": "obs",
        "field_order": ("u",),
        "latent_prior": "latent",
        "solver": "solver",
        "normalizer": "norm",
        "clamp": "clamp",
    }
    qualification = {
        "coupling": {k: v for k, v in coupling.items() if k != "observations"},
        "split": "train",
        "source_checkpoint_sha256": "a" * 64,
        "child_checkpoint_sha256": "b" * 64,
        "support_id": "grid",
        "sample_ids": ["train-1"],
        "alpha": 0.3,
        "cost": {"source_seconds": 1, "child_seconds": 2},
    }
    kwargs = {
        "alpha": 0.3,
        "source_sample_ids": ["a", "b"],
        "child_sample_ids": ["a", "b"],
        "source_support_id": "grid",
        "child_support_id": "grid",
        "source_checkpoint_sha256": "a" * 64,
        "child_checkpoint_sha256": "b" * 64,
        "source_coupling": coupling,
        "child_coupling": coupling,
        "qualification": qualification,
    }
    result = blend_matched_outputs(source, child, **kwargs)
    assert torch.allclose(result, torch.full_like(source, 0.3))
    result.sum().backward()
    assert torch.allclose(child.grad, torch.full_like(child, 0.3))
    with pytest.raises(ValueError, match="sample IDs"):
        blend_matched_outputs(source, child, **{**kwargs, "child_sample_ids": ["b", "a"]})
    with pytest.raises(ValueError, match="qualification"):
        blend_matched_outputs(
            source, child, **{**kwargs, "qualification": {**qualification, "split": "test"}}
        )

    for key in ("latent_prior", "normalizer"):
        with pytest.raises(ValueError, match="coupling"):
            blend_matched_outputs(
                source, child, **{**kwargs, "child_coupling": {**coupling, key: "other"}}
            )
    with pytest.raises(TypeError, match="bool"):
        blend_matched_outputs(source, child, **{**kwargs, "alpha": True})
    for ids in ("train-1", ["duplicate", "duplicate"]):
        with pytest.raises(ValueError, match="sample IDs"):
            blend_matched_outputs(
                source, child, **{**kwargs, "qualification": {**qualification, "sample_ids": ids}}
            )
    hard_source = source.clone()
    hard_child = child.detach().clone()
    hard_source[:, 0] = hard_child[:, 0] = 2
    assert torch.equal(
        blend_matched_outputs(hard_source, hard_child, **kwargs)[:, 0], hard_source[:, 0]
    )
