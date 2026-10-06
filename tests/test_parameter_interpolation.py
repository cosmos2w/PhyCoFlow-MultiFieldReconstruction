import copy

import pytest
import torch

from phycoflow_reconstruction.training.parameter_interpolation import (
    inference_only_checkpoint,
    interpolate_live_parameters,
    require_training_checkpoint,
)


def states():
    source = {"weight": torch.tensor([1., 2.]), "spectral": torch.tensor([1+2j]),
              "prior": torch.tensor([4.]), "coords": torch.tensor([0, 1]),
              "_metadata": {"version": 1}}
    child = copy.deepcopy(source)
    child["weight"] += 2
    child["spectral"] += 2-4j
    return source, child


def blend(source, child, alpha=.5, **kw):
    return interpolate_live_parameters(source, child, trainable_names=["weight", "spectral"],
        alpha=alpha, source_semantics={"fields": ["CH4", "CO", "T", "U_1", "p"]},
        child_semantics=kw.pop("child_semantics", {"fields": ["CH4", "CO", "T", "U_1", "p"]}), **kw)


@pytest.mark.parametrize("alpha", [0., 1.])
def test_exact_endpoints_and_detached_inputs(alpha):
    source, child = states()
    source["weight"].requires_grad_()
    before = copy.deepcopy(source)
    result = blend(source, child, alpha)
    for name in ("weight", "spectral", "prior", "coords"):
        assert torch.equal(result[name], (source if alpha == 0 else child)[name])
        assert torch.equal(source[name], before[name])
        assert result[name].data_ptr() != source[name].data_ptr()
        assert not result[name].requires_grad


def test_complex_modulus_native_interpolation_and_frozen_prior():
    source, child = states()
    result = blend(source, child)
    assert torch.equal(result["weight"], torch.tensor([2., 3.]))
    assert torch.equal(result["spectral"], torch.tensor([2+0j]))
    assert torch.equal(result["prior"], source["prior"])


@pytest.mark.parametrize("name", ["prior", "coords"])
def test_changed_frozen_parameter_or_buffer_rejected(name):
    source, child = states()
    child[name][0] += 1
    with pytest.raises(ValueError, match="frozen"):
        blend(source, child)


def test_fields_and_tensor_layout_rejected():
    source, child = states()
    with pytest.raises(ValueError, match="semantic"):
        blend(source, child, child_semantics={"fields": ["T", "CO"]})
    child["spectral"] = torch.ones(2, dtype=torch.complex64)
    with pytest.raises(ValueError, match="layout"):
        blend(source, child)


@pytest.mark.parametrize("alpha", [-1., 2., float("nan")])
def test_invalid_alpha_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        blend(*states(), alpha)


def test_export_has_no_optimizer_controller_or_ema_and_rejects_resume():
    source, child = states()
    provenance = {"alpha": .5, "source_checkpoint_sha256": "source", "child_checkpoint_sha256": "child",
                  "source_parameter_sha256": "s", "child_parameter_sha256": "c", "semantics": {"fields": ["T"]}}
    payload = inference_only_checkpoint(blend(source, child), normalization={"mean": [0.]}, provenance=provenance)
    assert not {"optimizer", "fidelity_controller", "training_aux_state", "global_step"} & payload.keys()
    with pytest.raises(ValueError, match="inference-only"):
        require_training_checkpoint(payload)
    require_training_checkpoint({"model": source, "optimizer": {}})
