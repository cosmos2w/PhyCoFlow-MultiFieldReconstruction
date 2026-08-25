"""Downstream adapter for the dataset-independent portable GL-RBF/CQ core.

The portable package owns the model implementation and tensor-level RF helpers.
This module only translates the downstream :class:`ObservationBatch` contract,
keeps query/sensor index semantics explicit, and exposes the model lifecycle
hooks consumed by the generic trainer.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F

from phycoflow_pointcloud import (
    ModelEMA,
    ReconstructionConfig,
    build_pointcloud_model,
    reconstruct_from_tensors,
    rectified_flow_loss_microbatched,
)
from phycoflow_pointcloud import PointCloudFFM as PortablePointCloudFFM
from phycoflow_pointcloud.priors import IIDGaussianPrior as PortableIIDGaussianPrior
from phycoflow_pointcloud.priors import RFFGaussianPrior as PortableRFFGaussianPrior

from ...contracts import LossBundle, ModelCapabilities, ObservationBatch, ReconstructionBatch

_ALIAS_KEYS = {
    "heads": "num_heads",
    "latent_blocks": "num_latent_blocks",
    "field_embedding_dim": "field_embed_dim",
    "fourier_bands": "fourier_pe_num_bands",
    "fourier_max_frequency": "fourier_pe_max_freq",
    "gather_topk_rbf_glres": "gather_topk",
    "query_microbatch_size": "train_query_microbatch_size",
    "reuse_condition_context": "reuse_condition_context_across_query_microbatches",
}


def _portable_config(
    config: Mapping[str, Any], *, coordinate_dim: int
) -> dict[str, Any]:
    """Map downstream spelling to the frozen portable public configuration."""

    resolved = {
        key: value
        for key, value in config.items()
        if key not in {"name", "model_name", "backbone"}
    }
    for source, target in _ALIAS_KEYS.items():
        if source in resolved:
            if target in resolved and resolved[target] != resolved[source]:
                raise ValueError(
                    f"conflicting GL_rbf_CQ settings {source!r} and {target!r}"
                )
            resolved.setdefault(target, resolved[source])
            del resolved[source]
    defaults = {
        "sigma_min": 1.0e-4,
        "prior": "rff",
        "rff_features": 256,
        "rff_lengthscale": 0.15,
        "hidden_dim": 256,
        "cond_dim": 128,
        "field_embed_dim": 128,
        "rbf_sigma": 0.05,
        "USE_FOURIER_PE": True,
        "fourier_pe_num_bands": 32,
        "fourier_pe_max_freq": 64.0,
        "latent_dim": 256,
        "num_latents": 128,
        "num_heads": 8,
        "num_latent_blocks": 4,
        "ff_mult": 4,
        "attn_dropout": 0.0,
        "mlp_dropout": 0.0,
        "summary_type": "mean",
        "sensor_coord_encoding": "fourier",
        "latent_sensor_reinject": True,
        "latent_reinject_every": 1,
        "condition_attention_execution": "cached_kv",
        "sensor_attention_padding_mode": "full",
        "gather_mode": "topk_rbf_glres",
        "gather_topk": 32,
        "gather_query_chunk_size": 2048,
        "learnable_rbf_sigma": True,
        "neighbor_backend": "keops",
        "sensor_local_topk": 16,
        "sensor_local_dropout": 0.0,
        "query_latent_readout": True,
        "query_readout_type": "coord",
        "query_readout_scale_init": 1.0e-2,
        "enhanced_head_norm": True,
        "glres_scale_init": 1.0e-2,
        "cq_query_dim": 128,
        "cq_readout_mode": "lowrank",
        "cq_readout_rank": 64,
        "cq_readout_heads": 4,
        "cq_fusion_mode": "additive",
        "cq_global_scale_init": 1.0,
        "cq_local_scale_init": 1.0,
        "cq_readout_scale_init": 1.0e-2,
        "cq_time_conditioning": "sinusoidal_film",
        "cq_time_embed_dim": 128,
        "cq_time_max_period": 10000.0,
        "cq_time_film_zero_init": True,
        "cq_measurement_support_mode": "rbf_value_support",
        "cq_measurement_support_normalize": True,
    }
    for key, value in defaults.items():
        resolved.setdefault(key, value)
    resolved["model_name"] = "GL_rbf_CQ"
    resolved["coord_dim"] = int(coordinate_dim)
    return resolved


def _generator_matches_device(generator: torch.Generator, device: torch.device) -> bool:
    generator_device = torch.device(generator.device)
    if generator_device.type != device.type:
        return False
    if device.type == "cuda":
        return generator_device.index in {None, device.index}
    return True


@contextmanager
def _generator_rng_context(
    generator: torch.Generator | None, device: torch.device
) -> Iterator[None]:
    """Run the portable global-RNG sampler from a downstream generator state."""

    if generator is None:
        yield
        return
    if not _generator_matches_device(generator, device):
        raise ValueError(
            "GL_rbf_CQ reconstruction generator must use the query tensor device; "
            f"got generator={generator.device}, query={device}"
        )
    if device.type == "cuda":
        live_state = torch.cuda.get_rng_state(device)
        torch.cuda.set_rng_state(generator.get_state(), device)
    else:
        live_state = torch.get_rng_state()
        torch.set_rng_state(generator.get_state())
    try:
        yield
    finally:
        if device.type == "cuda":
            generator.set_state(torch.cuda.get_rng_state(device))
            torch.cuda.set_rng_state(live_state, device)
        else:
            generator.set_state(torch.get_rng_state())
            torch.set_rng_state(live_state)


class GLRbfCQ(PortablePointCloudFFM):
    """Point-cloud RF adapter backed by the frozen portable GL-RBF/CQ model."""

    capabilities = ModelCapabilities(
        "point", True, True, False, True, ("base_training", "post_training")
    )

    def __init__(
        self,
        coordinate_dim: int,
        num_fields: int,
        logical_shape: tuple[int, ...],
        **config: Any,
    ) -> None:
        del logical_shape  # The portable point model is query-coordinate based.
        resolved = _portable_config(config, coordinate_dim=coordinate_dim)
        portable = build_pointcloud_model(
            resolved,
            n_fields=int(num_fields),
            device="cpu",
        )
        # Deliberately register the portable children directly.  This preserves
        # the release schema (model.* / prior.*) without a downstream prefix.
        super().__init__(portable.model, portable.prior, sigma_min=portable.sigma_min)
        self.num_fields = int(num_fields)
        self.coordinate_dim = int(coordinate_dim)
        self.backbone_name = "GL_rbf_ENH_CQ"
        self.portable_config = dict(resolved)

        self.train_query_microbatch_size = max(
            1,
            int(
                config.get(
                    "train_query_microbatch_size",
                    config.get("query_microbatch_size", 2048),
                )
            ),
        )
        self.reuse_condition_context = bool(
            config.get(
                "reuse_condition_context_across_query_microbatches",
                config.get("reuse_condition_context", True),
            )
        )
        self.reconstruction_execution_mode = str(
            config.get("reconstruction_execution_mode", "cached_streamed")
        )
        self.reconstruction_query_chunk_size = max(
            1, int(config.get("reconstruction_query_chunk_size", 8192))
        )
        self.reconstruction_cache_level = str(
            config.get("reconstruction_cache_level", "static_features")
        )
        self.ode_solver = str(config.get("ode_solver", "euler"))
        self.obs_consistency_mode = str(
            config.get("obs_consistency_mode", "endpoint_smooth")
        )
        self.obs_consistency_strength = float(
            config.get("obs_consistency_strength", 1.0)
        )
        self.obs_consistency_sigma = float(config.get("obs_consistency_sigma", 0.05))
        self.obs_consistency_schedule_power = float(
            config.get("obs_consistency_schedule_power", 2.0)
        )
        self.obs_consistency_final_clamp = bool(
            config.get("obs_consistency_final_clamp", True)
        )
        self.obs_consistency_chunk_size = max(
            1, int(config.get("obs_consistency_chunk_size", 8192))
        )

        self._ema_eval = bool(config.get("model_ema_eval", True))
        self._ema: ModelEMA | None = None
        if bool(config.get("model_ema_enabled", True)):
            self._ema = ModelEMA(
                self,
                decay=float(config.get("model_ema_decay", 0.999)),
            )

    def _validate_batch(
        self, batch: ObservationBatch, *, require_target: bool
    ) -> None:
        batch.validate()
        if require_target and batch.target_fields is None:
            raise ValueError("GL_rbf_CQ rectified-flow training requires target_fields")
        if batch.target_fields is not None and batch.target_fields.shape[-1] != self.num_fields:
            raise ValueError(
                "GL_rbf_CQ target field count does not match the constructed model: "
                f"{batch.target_fields.shape[-1]} != {self.num_fields}"
            )
        valid_fields = batch.obs_field_ids[batch.obs_valid_mask]
        if torch.any(valid_fields >= self.num_fields):
            raise ValueError("GL_rbf_CQ observation field ID exceeds model field count")
        if not bool(torch.all(batch.query_valid_mask)):
            raise ValueError(
                "GL_rbf_CQ requires dense query slots; padded query masks must be "
                "compacted by the downstream batch builder before training or reconstruction"
            )

    def _portable_training_loss_microbatched(
        self,
        *,
        x1: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        obs_indices: torch.Tensor | None = None,
        query_microbatch_size: int,
        backward: bool = False,
        reuse_condition_context: bool = True,
        synchronize_timing: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Call the portable helper without colliding with the batch API below."""

        if int(query_microbatch_size) >= int(coords.shape[1]):
            loss, metrics = PortablePointCloudFFM.training_loss(
                self,
                x1=x1,
                coords=coords,
                obs_coords=obs_coords,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_field_ids=obs_field_ids,
                obs_indices=obs_indices,
            )
            if backward:
                loss.backward()
            metrics.update(
                {
                    "rf_bridge_ms": 0.0,
                    "condition_context_ms": 0.0,
                    "query_chunk_forward_ms": 0.0,
                    "query_chunk_backward_ms": 0.0,
                    "query_microbatches": 1.0,
                }
            )
            return (loss.detach() if backward else loss), metrics
        return PortablePointCloudFFM.training_loss_microbatched(
            self,
            x1=x1,
            coords=coords,
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
            obs_indices=obs_indices,
            query_microbatch_size=int(query_microbatch_size),
            backward=backward,
            reuse_condition_context=reuse_condition_context,
            synchronize_timing=synchronize_timing,
        )

    def training_loss_microbatched(
        self,
        *,
        x1: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        obs_indices: torch.Tensor | None = None,
        query_microbatch_size: int,
        backward: bool = False,
        reuse_condition_context: bool = True,
        synchronize_timing: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Expose the portable microbatched RF entry without changing its API."""

        return self._portable_training_loss_microbatched(
            x1=x1,
            coords=coords,
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
            obs_indices=obs_indices,
            query_microbatch_size=query_microbatch_size,
            backward=backward,
            reuse_condition_context=reuse_condition_context,
            synchronize_timing=synchronize_timing,
        )

    def _loss_bundle_from_tensors(
        self, loss: torch.Tensor, metrics: Mapping[str, float]
    ) -> LossBundle:
        return LossBundle(
            loss,
            {"rectified_flow_mse": loss},
            diagnostics=dict(metrics),
        )

    def training_loss(self, batch: ObservationBatch) -> LossBundle:
        self._validate_batch(batch, require_target=True)
        loss, metrics = rectified_flow_loss_microbatched(
            self,
            x1=batch.target_fields,
            coords=batch.query_coords,
            obs_coords=batch.obs_coords,
            obs_values=batch.obs_values,
            obs_mask=batch.obs_valid_mask,
            obs_field_ids=batch.obs_field_ids,
            obs_indices=batch.obs_indices,
            query_microbatch_size=self.train_query_microbatch_size,
            backward=False,
            reuse_condition_context=self.reuse_condition_context,
        )
        return self._loss_bundle_from_tensors(loss, metrics)

    def _training_backward_microbatched(
        self,
        *,
        x1: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        obs_indices: torch.Tensor | None,
        query_microbatch_size: int,
        reuse_condition_context: bool,
        start_phase: Any = None,
        end_phase: Any = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Backward variant of the portable helper with phase callbacks."""

        del obs_indices
        chunk_size = max(1, int(query_microbatch_size))
        n_query = int(coords.shape[1])

        def phase_start(name: str) -> None:
            if start_phase is not None:
                start_phase(name)

        def phase_end(name: str) -> None:
            if end_phase is not None:
                end_phase(name)

        if chunk_size >= n_query:
            phase_start("forward_native_loss")
            try:
                loss, metrics = PortablePointCloudFFM.training_loss(
                    self,
                    x1=x1,
                    coords=coords,
                    obs_coords=obs_coords,
                    obs_values=obs_values,
                    obs_mask=obs_mask,
                    obs_field_ids=obs_field_ids,
                )
            finally:
                phase_end("forward_native_loss")
            phase_start("backward")
            try:
                loss.backward()
            finally:
                phase_end("backward")
            metrics.update(
                {
                    "rf_bridge_ms": 0.0,
                    "condition_context_ms": 0.0,
                    "query_chunk_forward_ms": 0.0,
                    "query_chunk_backward_ms": 0.0,
                    "query_microbatches": 1.0,
                }
            )
            return loss.detach(), metrics

        phase_start("forward_native_loss")
        try:
            bridge_start = perf_counter()
            bridge = PortablePointCloudFFM.prepare_training_bridge(self, x1, coords)
            bridge_ms = (perf_counter() - bridge_start) * 1000.0

            condition_context = None
            condition_start = perf_counter()
            if reuse_condition_context:
                if not hasattr(self.model, "prepare_condition_context"):
                    raise ValueError(
                        "Condition-context reuse is unavailable for this backbone."
                    )
                condition_context = self.model.prepare_condition_context(
                    obs_coords,
                    obs_values,
                    obs_mask,
                    obs_field_ids,
                )
            condition_ms = (perf_counter() - condition_start) * 1000.0
        finally:
            phase_end("forward_native_loss")

        total_elements = int(bridge["target"].numel())
        total_loss = x1.new_zeros(())
        forward_ms = 0.0
        backward_ms = 0.0
        chunks = 0
        for start_index in range(0, n_query, chunk_size):
            end_index = min(start_index + chunk_size, n_query)
            query_slice = slice(start_index, end_index)
            phase_start("forward_native_loss")
            forward_start = perf_counter()
            try:
                if condition_context is not None:
                    predicted = self.model.forward_query_chunk(
                        t=bridge["t"],
                        x_t_chunk=bridge["x_t"][:, query_slice],
                        coords_chunk=coords[:, query_slice],
                        condition_context=condition_context,
                    )
                else:
                    predicted = self.model(
                        bridge["t"],
                        bridge["x_t"][:, query_slice],
                        coords[:, query_slice],
                        obs_coords,
                        obs_values,
                        obs_mask,
                        obs_field_ids,
                    )
                chunk_loss = F.mse_loss(
                    predicted,
                    bridge["target"][:, query_slice],
                    reduction="sum",
                ) / total_elements
            finally:
                forward_ms += (perf_counter() - forward_start) * 1000.0
                phase_end("forward_native_loss")

            phase_start("backward")
            backward_start = perf_counter()
            try:
                chunk_loss.backward(
                    retain_graph=condition_context is not None and end_index < n_query
                )
            finally:
                backward_ms += (perf_counter() - backward_start) * 1000.0
                phase_end("backward")
            total_loss = total_loss + chunk_loss.detach()
            chunks += 1
            del predicted, chunk_loss

        target = bridge["target"]
        metrics = {
            "loss": float(total_loss.detach().cpu()),
            "target_rms": float(target.pow(2).mean().sqrt().detach().cpu()),
            "rf_bridge_ms": bridge_ms,
            "condition_context_ms": condition_ms,
            "query_chunk_forward_ms": forward_ms,
            "query_chunk_backward_ms": backward_ms,
            "query_microbatches": float(chunks),
        }
        return total_loss, metrics

    def training_backward(
        self,
        batch: ObservationBatch,
        *,
        loss_scale: float,
        start_phase: Any = None,
        end_phase: Any = None,
    ) -> LossBundle:
        """Run exactly one portable microbatched backward pass.

        The generic trainer owns gradient clearing, stable clipping, and finite
        checks.  The portable helper owns chunked forward/backward accumulation;
        calling ``training_loss(...).backward()`` here would retain the full
        query graph and would double-backward under the trainer hook.
        """

        self._validate_batch(batch, require_target=True)
        if float(loss_scale) != 1.0:
            raise ValueError("GL_rbf_CQ training_backward currently requires loss_scale=1")
        if (start_phase is None) != (end_phase is None):
            raise ValueError("start_phase and end_phase must be provided together")
        loss, metrics = self._training_backward_microbatched(
            x1=batch.target_fields,
            coords=batch.query_coords,
            obs_coords=batch.obs_coords,
            obs_values=batch.obs_values,
            obs_mask=batch.obs_valid_mask,
            obs_field_ids=batch.obs_field_ids,
            obs_indices=batch.obs_indices,
            query_microbatch_size=self.train_query_microbatch_size,
            reuse_condition_context=self.reuse_condition_context,
            start_phase=start_phase,
            end_phase=end_phase,
        )
        return self._loss_bundle_from_tensors(loss, metrics)

    def _sample_prior(
        self, coords: torch.Tensor, generator: torch.Generator | None
    ) -> torch.Tensor:
        if generator is None:
            return PortablePointCloudFFM.sample_source(self, coords)
        if isinstance(self.prior, PortableIIDGaussianPrior):
            return torch.randn(
                coords.shape[0],
                coords.shape[1],
                self.num_fields,
                device=coords.device,
                dtype=coords.dtype,
                generator=generator,
            )
        if isinstance(self.prior, PortableRFFGaussianPrior):
            features = self.prior._features(coords)
            weights = torch.randn(
                coords.shape[0],
                self.num_fields,
                self.prior.n_features,
                device=coords.device,
                dtype=coords.dtype,
                generator=generator,
            )
            return torch.einsum("bnf,bcf->bnc", features, weights)
        raise TypeError(f"unsupported portable prior type {type(self.prior)!r}")

    def sample_source(
        self,
        batch_or_coords: ObservationBatch | torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        coords = (
            batch_or_coords.query_coords
            if isinstance(batch_or_coords, ObservationBatch)
            else batch_or_coords
        )
        if generator is None:
            return self._sample_prior(coords, None)
        return self._sample_prior(coords, generator)

    def velocity(
        self,
        batch: ObservationBatch,
        state: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_batch(batch, require_target=False)
        return self.model(
            time,
            state,
            batch.query_coords,
            batch.obs_coords,
            batch.obs_values,
            batch.obs_valid_mask,
            batch.obs_field_ids,
        )

    def _local_clamp_indices(
        self, batch: ObservationBatch
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Translate global dataset indices to the compact query index space."""

        if batch.obs_indices is None:
            return None, None
        query_indices = batch.metadata.get("query_indices")
        if not isinstance(query_indices, torch.Tensor):
            if int(batch.obs_indices.max().item()) < int(batch.query_coords.shape[1]):
                return batch.obs_indices.to(batch.query_coords.device), batch.obs_valid_mask
            return None, None
        query_indices = query_indices.to(device=batch.query_coords.device, dtype=torch.long)
        obs_indices = batch.obs_indices.to(device=batch.query_coords.device, dtype=torch.long)
        local = torch.full_like(obs_indices, -1)
        for batch_index in range(obs_indices.shape[0]):
            matches = obs_indices[batch_index, :, None] == query_indices[batch_index, None, :]
            found = matches.any(dim=1)
            local[batch_index] = torch.where(
                found,
                matches.to(dtype=torch.long).argmax(dim=1),
                torch.full_like(found, -1, dtype=torch.long),
            )
        clamp_mask = batch.obs_valid_mask.to(device=local.device) & (local >= 0)
        return local, clamp_mask

    @torch.no_grad()
    def reconstruct(
        self,
        batch: ObservationBatch,
        *,
        steps: int = 4,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> ReconstructionBatch:
        self._validate_batch(batch, require_target=False)
        if steps < 1:
            raise ValueError("reconstruction steps must be at least one")

        observation_config = kwargs.get("observation_config", {})
        if not isinstance(observation_config, Mapping):
            raise TypeError("observation_config must be a mapping")
        get = lambda key, default: observation_config.get(key, kwargs.get(key, default))
        mode = str(get("obs_consistency_mode", self.obs_consistency_mode))
        local_indices, clamp_mask = self._local_clamp_indices(batch)
        all_observations_in_query = (
            clamp_mask is not None and bool(torch.equal(clamp_mask, batch.obs_valid_mask))
        )
        if (
            local_indices is None
            and mode in {"default_hard", "hard", "endpoint"}
            and batch.obs_indices is not None
        ):
            raise ValueError(
                f"obs_consistency_mode={mode!r} requires query-local observation indices"
            )
        portable_clamp = local_indices if all_observations_in_query else None
        config = ReconstructionConfig(
            n_steps=int(steps),
            ode_solver=str(get("ode_solver", self.ode_solver)),
            obs_consistency_mode=mode,
            obs_consistency_strength=float(
                get("obs_consistency_strength", self.obs_consistency_strength)
            ),
            obs_consistency_sigma=float(
                get("obs_consistency_sigma", self.obs_consistency_sigma)
            ),
            obs_consistency_schedule_power=float(
                get("obs_consistency_schedule_power", self.obs_consistency_schedule_power)
            ),
            obs_consistency_final_clamp=bool(
                get("obs_consistency_final_clamp", self.obs_consistency_final_clamp)
            ),
            obs_consistency_chunk_size=int(
                get("obs_consistency_chunk_size", self.obs_consistency_chunk_size)
            ),
            execution_mode=str(
                get("reconstruction_execution_mode", self.reconstruction_execution_mode)
            ),
            query_chunk_size=int(
                get("reconstruction_query_chunk_size", self.reconstruction_query_chunk_size)
            ),
            cache_level=str(
                get("reconstruction_cache_level", self.reconstruction_cache_level)
            ),
        )
        with _generator_rng_context(generator, batch.query_coords.device):
            prediction = reconstruct_from_tensors(
                self,
                coords=batch.query_coords,
                obs_coords=batch.obs_coords,
                obs_values=batch.obs_values,
                obs_mask=batch.obs_valid_mask,
                obs_field_ids=batch.obs_field_ids,
                obs_indices=portable_clamp,
                config=config,
            )
        if (
            not all_observations_in_query
            and clamp_mask is not None
            and bool(config.obs_consistency_final_clamp)
            and mode not in {"none"}
        ):
            prediction = prediction.clone()
            for batch_index in range(prediction.shape[0]):
                valid = clamp_mask[batch_index]
                if not bool(valid.any()):
                    continue
                prediction[batch_index, local_indices[batch_index, valid], batch.obs_field_ids[batch_index, valid]] = (
                    batch.obs_values[batch_index, valid, 0].to(prediction.dtype)
                )
        return ReconstructionBatch(
            prediction,
            diagnostics={
                "sampling_steps": int(steps),
                "backbone": self.backbone_name,
                "reconstruction_execution_mode": config.execution_mode,
                "query_microbatch_size": self.train_query_microbatch_size,
            },
        )

    def after_optimizer_step(self) -> None:
        if self._ema is not None:
            self._ema.update(self)

    @contextmanager
    def evaluation_weight_context(self) -> Iterator[None]:
        if self._ema is None or not self._ema_eval:
            with nullcontext():
                yield
            return
        with self._ema.average_parameters(self):
            yield

    def training_aux_state_dict(self) -> dict[str, Any]:
        if self._ema is None:
            return {}
        return {"model_ema": self._ema.state_dict()}

    def load_training_aux_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        if self._ema is None:
            raise RuntimeError(
                "checkpoint contains GL_rbf_CQ EMA state but model_ema_enabled is false"
            )
        payload = state.get("model_ema", state)
        if not isinstance(payload, Mapping):
            raise TypeError("model_ema auxiliary checkpoint must be a mapping")
        self._ema.load_state_dict(payload)


GL_rbf_CQ = GLRbfCQ

__all__ = ["GLRbfCQ", "GL_rbf_CQ"]
