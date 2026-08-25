"""Focused contracts for optional model-owned trainer lifecycle hooks."""

from contextlib import contextmanager

import pytest
import torch

from phycoflow_reconstruction.contracts import LossBundle
from phycoflow_reconstruction.training.model_lifecycle import (
    add_training_aux_state,
    after_optimizer_step,
    backward_and_clip_model_loss,
    evaluation_weight_context,
    load_training_aux_state,
)


class _PlainModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def training_loss(self, batch):
        loss = (self.weight * batch).square()
        return LossBundle(loss, {"data": loss})


class _LifecycleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.ema_weight = torch.tensor(7.0)
        self.update_count = 0
        self.backward_calls = 0

    def training_loss(self, batch):  # pragma: no cover - a failure guard
        raise AssertionError("trainer must not compute a second monolithic loss")

    def training_backward(
        self,
        batch,
        *,
        loss_scale,
        start_phase=None,
        end_phase=None,
    ):
        self.backward_calls += 1
        if start_phase is not None:
            start_phase("forward_native_loss")
        loss = (self.weight * batch).square()
        if end_phase is not None:
            end_phase("forward_native_loss")
            start_phase("backward")
        (loss * loss_scale).backward()
        if end_phase is not None:
            end_phase("backward")
        return LossBundle(loss.detach(), {"data": loss.detach()})

    def after_optimizer_step(self):
        self.update_count += 1
        self.ema_weight = self.weight.detach().clone()

    @contextmanager
    def evaluation_weight_context(self):
        live = self.weight.detach().clone()
        with torch.no_grad():
            self.weight.copy_(self.ema_weight)
        try:
            yield
        finally:
            with torch.no_grad():
                self.weight.copy_(live)

    def training_aux_state_dict(self):
        return {
            "ema_weight": self.ema_weight.clone(),
            "update_count": self.update_count,
        }

    def load_training_aux_state_dict(self, state):
        self.ema_weight = state["ema_weight"].clone()
        self.update_count = int(state["update_count"])


def test_hookless_backward_uses_unchanged_loss_closure_path():
    model = _PlainModel()

    losses, norm, scale, retries = backward_and_clip_model_loss(
        model, torch.tensor(3.0), model.parameters(), 1.0
    )

    assert float(losses.total) == 36.0
    assert float(norm) == 36.0
    assert scale == 1.0
    assert retries == 0
    assert float(model.weight.grad) == pytest.approx(1.0)


def test_model_owned_backward_runs_once_and_keeps_phase_boundaries():
    model = _LifecycleModel()
    phases = []

    losses, norm, scale, retries = backward_and_clip_model_loss(
        model,
        torch.tensor(3.0),
        model.parameters(),
        1.0,
        start_phase=lambda name: phases.append(("start", name)),
        end_phase=lambda name: phases.append(("end", name)),
    )

    assert model.backward_calls == 1
    assert float(losses.total) == 36.0
    assert float(norm) == 36.0
    assert scale == 1.0
    assert retries == 0
    assert phases == [
        ("start", "forward_native_loss"),
        ("end", "forward_native_loss"),
        ("start", "backward"),
        ("end", "backward"),
    ]


@pytest.mark.parametrize(
    ("initial_scale", "adaptive", "message"),
    [
        (0.5, False, "requires loss_scale=1"),
        (1.0, True, "does not support adaptive scaling"),
    ],
)
def test_model_owned_backward_rejects_unsupported_scaling(
    initial_scale, adaptive, message
):
    model = _LifecycleModel()
    with pytest.raises(ValueError, match=message):
        backward_and_clip_model_loss(
            model,
            torch.tensor(1.0),
            model.parameters(),
            1.0,
            initial_scale=initial_scale,
            adaptive=adaptive,
        )


def test_lifecycle_update_evaluation_and_aux_state_round_trip():
    model = _LifecycleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.weight.grad = torch.tensor(1.0)
    optimizer.step()
    after_optimizer_step(model)

    assert model.update_count == 1
    live = model.weight.detach().clone()
    model.ema_weight = torch.tensor(11.0)
    with evaluation_weight_context(model):
        assert float(model.weight) == 11.0
    assert torch.equal(model.weight, live)

    payload = add_training_aux_state({"model": {}}, model)
    restored = _LifecycleModel()
    load_training_aux_state(restored, payload)
    assert restored.update_count == 1
    assert float(restored.ema_weight) == 11.0


def test_hookless_models_keep_payload_schema_and_noop_context():
    model = _PlainModel()
    payload = {"model": {}}

    assert add_training_aux_state(payload, model) is payload
    assert "training_aux_state" not in payload
    load_training_aux_state(model, payload)
    with evaluation_weight_context(model):
        assert float(model.weight) == 2.0
    after_optimizer_step(model)


def test_aux_state_restore_is_strict_in_both_directions():
    with pytest.raises(KeyError, match="missing required"):
        load_training_aux_state(_LifecycleModel(), {"model": {}})
    with pytest.raises(TypeError, match="cannot restore"):
        load_training_aux_state(_PlainModel(), {"training_aux_state": {}})
