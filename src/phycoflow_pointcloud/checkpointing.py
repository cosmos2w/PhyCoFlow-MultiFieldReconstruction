"""Checkpoint state selection with explicit live/EMA compatibility semantics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class ResolvedCheckpointState:
    state_dict: Mapping[str, object]
    selection: str


def resolve_checkpoint_state(
    checkpoint,
    prefer_ema: bool | None = None,
    model: nn.Module | None = None,
) -> ResolvedCheckpointState:
    """Resolve weights and report whether live or repaired EMA state was chosen."""
    state_dict = checkpoint
    selection = "raw_state_dict"
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        if prefer_ema is None:
            prefer_ema = bool(
                checkpoint.get(
                    "model_ema_eval", checkpoint.get("model_ema_enabled", False)
                )
            )
        use_ema = bool(prefer_ema and "model_ema" in checkpoint)
        if use_ema:
            ema_state = checkpoint["model_ema"]
            state_dict = (
                ema_state.get("shadow", ema_state)
                if isinstance(ema_state, dict)
                else ema_state
            )
            averaged_names = (
                ema_state.get("averaged_parameter_names")
                if isinstance(ema_state, dict)
                else None
            )
            live_state = checkpoint["model"]
            if averaged_names is not None:
                ema_shadow = ema_state.get("shadow", ema_state)
                state_dict = dict(live_state)
                state_dict.update({name: ema_shadow[name] for name in averaged_names})
                selection = "ema_trainable_plus_live_frozen"
            elif model is not None:
                state_dict = dict(state_dict)
                frozen_names = {name for name, _ in model.named_buffers()} | {
                    name
                    for name, parameter in model.named_parameters()
                    if not parameter.requires_grad
                }
                for name in frozen_names:
                    if name in live_state:
                        state_dict[name] = live_state[name]
                selection = "ema_legacy_repaired_with_live_frozen"
            else:
                selection = "ema_legacy_unrepaired"
        else:
            state_dict = checkpoint["model"]
            selection = "live"
    if isinstance(state_dict, dict) and "_metadata" in state_dict:
        state_dict = dict(state_dict)
        state_dict.pop("_metadata", None)
    return ResolvedCheckpointState(state_dict=state_dict, selection=selection)


def checkpoint_model_state(
    checkpoint,
    prefer_ema: bool | None = None,
    model: nn.Module | None = None,
):
    """Historical state-only API retained for old imports and scripts."""
    return resolve_checkpoint_state(
        checkpoint, prefer_ema=prefer_ema, model=model
    ).state_dict
