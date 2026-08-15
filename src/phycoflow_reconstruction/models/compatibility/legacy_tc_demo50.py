"""Strict, isolated compatibility adapter for turbulent-combustion DemoN50.

DemoN50 predates the narrowed PointCloudFFM API and used the historical
``GL_rbf_ENH/topk_rbf_glres`` velocity network. This module preserves only the
architecture and Euler/RFF behavior needed by that one run. It is deliberately
not registered as a model or gather mode for new training, and it never imports
the old demo package at runtime.

The implementation is a focused extraction from the repository-local DemoN50
source. Module names and tensor operations are retained so the checkpoint can
be loaded strictly and compared numerically before Phase-5 post-training.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import torch
import yaml
from torch import nn

from ...contracts import LossBundle, ModelCapabilities, ObservationBatch, ReconstructionBatch
from ..base import masked_mse

try:  # KeOps is used by the historical run but torch remains a validation fallback.
    from pykeops.torch import LazyTensor
except ImportError:  # pragma: no cover - exercised only in minimal CPU environments
    LazyTensor = None


DEMO50_DATASET_FIELDS = ("CO", "T", "U_0", "U_1", "p")
DEMO50_STALE_CHECKPOINT_FIELDS = ("CH4", "CO", "T", "U_1", "p")


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = in_dim
    for _ in range(depth - 1):
        layers.extend((nn.Linear(width, hidden_dim), nn.GELU()))
        width = hidden_dim
    layers.append(nn.Linear(width, out_dim))
    return nn.Sequential(*layers)


class _FourierPositionalEncoding(nn.Module):
    def __init__(self, coordinate_dim: int, num_bands: int, max_frequency: float) -> None:
        super().__init__()
        self.coordinate_dim = coordinate_dim
        self.num_bands = num_bands
        self.out_dim = coordinate_dim * num_bands * 2
        self.register_buffer("freqs", torch.linspace(1.0, max_frequency / 2.0, num_bands))

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        scaled = coordinates[..., : self.coordinate_dim] * 2.0 - 1.0
        angles = scaled.unsqueeze(-1) * self.freqs * math.pi
        encoded = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return encoded.reshape(*scaled.shape[:-1], self.out_dim)


class _FeedForward(nn.Module):
    def __init__(self, width: int, multiplier: int, dropout: float) -> None:
        super().__init__()
        inner = width * multiplier
        self.net = nn.Sequential(
            nn.Linear(width, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, width),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class _CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        multiplier: int,
        attention_dropout: float,
        mlp_dropout: float,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(width)
        self.norm_kv = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, attention_dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(width)
        self.ff = _FeedForward(width, multiplier, mlp_dropout)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention, _ = self.attn(
            self.norm_q(q),
            self.norm_kv(kv),
            self.norm_kv(kv),
            key_padding_mask=kv_padding_mask,
            need_weights=False,
        )
        value = q + attention
        return value + self.ff(self.norm_ff(value))


class _SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        multiplier: int,
        attention_dropout: float,
        mlp_dropout: float,
    ) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, attention_dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(width)
        self.ff = _FeedForward(width, multiplier, mlp_dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_attn(value)
        attention, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        value = value + attention
        return value + self.ff(self.norm_ff(value))


def _batched_gather_2d(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(values.shape[0], device=values.device).view(-1, 1, 1).expand_as(indices)
    return values[batch, indices]


def _batched_gather_3d(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(values.shape[0], device=values.device).view(-1, 1, 1).expand_as(indices)
    return values[batch, indices]


class _LegacyDemo50Velocity(nn.Module):
    """Exact parameter layout and GL-residual forward path used by DemoN50."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        n_fields = 5
        coordinate_dim = 3
        hidden_dim = int(config["hidden_dim"])
        cond_dim = int(config["cond_dim"])
        latent_dim = int(config["latent_dim"])
        heads = int(config["num_heads"])
        ff_mult = int(config["ff_mult"])
        mlp_dropout = float(config["mlp_dropout"])

        self.n_fields = n_fields
        self.coord_dim = coordinate_dim
        self.rbf_sigma = float(config["rbf_sigma"])
        self.latent_dim = latent_dim
        self.num_latents = int(config["num_latents"])
        self.summary_type = str(config["summary_type"])
        self.use_fourier_pe = bool(config["USE_FOURIER_PE"])
        self.pos_enc = _FourierPositionalEncoding(
            coordinate_dim,
            int(config["fourier_pe_num_bands"]),
            float(config["fourier_pe_max_freq"]),
        )
        self.coord_feat_dim = self.pos_enc.out_dim
        self.enhanced_backbone = True
        self.sensor_coord_encoding = str(config["sensor_coord_encoding"])
        self.latent_sensor_reinject = bool(config["latent_sensor_reinject"])
        self.latent_reinject_every = int(config["latent_reinject_every"])
        self.query_latent_readout_enabled = bool(config["query_latent_readout"])
        self.query_readout_type = str(config["query_readout_type"])
        self.enhanced_head_norm = bool(config["enhanced_head_norm"])
        self.gather_mode = "topk_rbf_glres"
        self.gather_topk = int(config["gather_topk"])
        self.gather_query_chunk_size = config["gather_query_chunk_size"]
        self.learnable_rbf_sigma = bool(config["learnable_rbf_sigma"])
        self.neighbor_backend = str(config["neighbor_backend"])
        self.log_rbf_sigma = nn.Parameter(torch.log(torch.tensor(self.rbf_sigma)))

        # Query and sparse-sensor encoders retain their historical module names.
        self.point_encoder = _make_mlp(
            self.coord_feat_dim + n_fields + 1, hidden_dim, hidden_dim, depth=3
        )
        self.field_embed = nn.Embedding(n_fields, int(config["field_embed_dim"]))
        self.sensor_in_proj = _make_mlp(
            self.coord_feat_dim + 1 + int(config["field_embed_dim"]),
            latent_dim,
            latent_dim,
            depth=3,
        )
        self.sensor_out_proj = _make_mlp(latent_dim, cond_dim, cond_dim, depth=2)

        # Enhanced query-to-latent readout and GL residual scaffold.
        self.use_query_latent_readout = True
        self.query_decoder_token = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.query_readout_in = nn.Linear(self.coord_feat_dim + hidden_dim, latent_dim, bias=False)
        self.query_latent_readout = _CrossAttentionBlock(
            latent_dim,
            max(1, min(heads, 4)),
            max(1, ff_mult // 2),
            float(config["attn_dropout"]),
            mlp_dropout,
        )
        self.query_readout_out = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.query_readout_scale = nn.Parameter(
            torch.tensor(float(config["query_readout_scale_init"]))
        )
        self.coarse_film = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.coarse_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, n_fields),
        )
        self.coarse_scale = nn.Parameter(torch.tensor(float(config["glres_scale_init"])))
        self.sensor_importance = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, 1),
        )
        self.sensor_importance_scale = nn.Parameter(torch.tensor(float(config["glres_scale_init"])))

        # Perceiver-style global sensor processor and final local/global head.
        self.latents = nn.Parameter(
            torch.randn(self.num_latents, latent_dim) / math.sqrt(latent_dim)
        )
        cross_args = (
            latent_dim,
            heads,
            ff_mult,
            float(config["attn_dropout"]),
            mlp_dropout,
        )
        self.input_cross_attn = _CrossAttentionBlock(*cross_args)
        self.latent_blocks = nn.ModuleList(
            [_SelfAttentionBlock(*cross_args) for _ in range(int(config["num_latent_blocks"]))]
        )
        self.sensor_back_attn = _CrossAttentionBlock(*cross_args)
        self.summary_proj = _make_mlp(latent_dim, hidden_dim, hidden_dim, depth=2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, n_fields),
        )
        self.head_in_norm = nn.LayerNorm(hidden_dim + hidden_dim + cond_dim)

    def _sensor_tokens(
        self,
        coordinates: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        field_ids: torch.Tensor,
    ) -> torch.Tensor:
        field_features = self.field_embed(field_ids.clamp_min(0)) * mask.unsqueeze(-1)
        coordinate_features = self.pos_enc(coordinates)
        tokens = self.sensor_in_proj(
            torch.cat((coordinate_features, values, field_features), dim=-1)
        )
        return tokens * mask.unsqueeze(-1)

    def _latents(self, sensor_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        latents = self.latents.unsqueeze(0).expand(sensor_tokens.shape[0], -1, -1)
        padding_mask = ~mask.bool()
        latents = self.input_cross_attn(latents, sensor_tokens, padding_mask)
        for index, block in enumerate(self.latent_blocks):
            if (
                self.latent_sensor_reinject
                and index > 0
                and index % self.latent_reinject_every == 0
            ):
                latents = self.input_cross_attn(latents, sensor_tokens, padding_mask)
            latents = block(latents)
        return latents

    def _query_readout(
        self,
        point_features: torch.Tensor,
        coordinates: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        chunk_size = self.gather_query_chunk_size
        if chunk_size is None and coordinates.shape[1] > 4096:
            chunk_size = 4096
        chunk_size = coordinates.shape[1] if chunk_size is None else int(chunk_size)
        for start in range(0, coordinates.shape[1], chunk_size):
            stop = min(start + chunk_size, coordinates.shape[1])
            coord_features = self.pos_enc(coordinates[:, start:stop])
            decoder = self.query_decoder_token.view(1, 1, -1).expand(
                coordinates.shape[0], stop - start, -1
            )
            query = self.query_readout_in(torch.cat((coord_features, decoder), dim=-1))
            outputs.append(self.query_readout_out(self.query_latent_readout(query, latents)))
        return torch.cat(outputs, dim=1)

    def _topk(
        self,
        query_coordinates: torch.Tensor,
        sensor_coordinates: torch.Tensor,
        sensor_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        k = min(self.gather_topk, sensor_coordinates.shape[1])
        if self.neighbor_backend == "keops":
            if LazyTensor is None:
                raise ImportError("DemoN50 requires pykeops for neighbor_backend='keops'")
            query = LazyTensor(query_coordinates.contiguous()[:, :, None, :])
            sensors = LazyTensor(sensor_coordinates.contiguous()[:, None, :, :])
            distances = ((query - sensors) ** 2).sum(-1)
            valid = LazyTensor(mask[:, None, :, None].to(query_coordinates.dtype).contiguous())
            distances = distances + (1.0 - valid) * 1e6
            topk_distances, indices = distances.Kmin_argKmin(K=k, dim=2)
            indices = indices.long()
        else:
            distances = torch.cdist(query_coordinates, sensor_coordinates).square()
            distances = torch.where(mask.unsqueeze(1), distances, torch.full_like(distances, 1e6))
            topk_distances, indices = torch.topk(distances, k=k, dim=-1, largest=False)
        return (
            topk_distances,
            indices,
            _batched_gather_3d(sensor_features, indices),
            _batched_gather_2d(mask, indices).bool(),
        )

    def _local_gather(
        self,
        query_coordinates: torch.Tensor,
        sensor_coordinates: torch.Tensor,
        sensor_features: torch.Tensor,
        mask: torch.Tensor,
        sensor_importance: torch.Tensor,
    ) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        chunk_size = self.gather_query_chunk_size or query_coordinates.shape[1]
        sigma = torch.exp(self.log_rbf_sigma).clamp_min(1e-6)
        for start in range(0, query_coordinates.shape[1], chunk_size):
            stop = min(start + chunk_size, query_coordinates.shape[1])
            distances, indices, neighbors, valid = self._topk(
                query_coordinates[:, start:stop], sensor_coordinates, sensor_features, mask
            )
            logits = -distances / (2 * sigma.square() + 1e-12)
            logits = logits + self.sensor_importance_scale * _batched_gather_2d(
                sensor_importance, indices
            )
            weights = torch.softmax(logits.masked_fill(~valid, -1e9), dim=-1)
            chunks.append(torch.sum(weights.unsqueeze(-1) * neighbors, dim=2))
        return torch.cat(chunks, dim=1)

    def forward(
        self,
        time: torch.Tensor,
        state: torch.Tensor,
        coordinates: torch.Tensor,
        obs_coordinates: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, point_count, _ = state.shape
        time_features = time.view(batch_size, 1, 1).expand(batch_size, point_count, 1)
        point_features = self.point_encoder(
            torch.cat((self.pos_enc(coordinates), state, time_features), dim=-1)
        )
        sensor_tokens = self._sensor_tokens(obs_coordinates, obs_values, obs_mask, obs_field_ids)
        latents = self._latents(sensor_tokens, obs_mask)
        global_features = self.summary_proj(latents.mean(dim=1))
        query_global = self._query_readout(point_features, coordinates, latents)
        global_for_head = global_features.unsqueeze(1) + self.query_readout_scale * query_global

        refined = self.sensor_back_attn(sensor_tokens, latents) * obs_mask.unsqueeze(-1)
        sensor_features = self.sensor_out_proj(refined) * obs_mask.unsqueeze(-1)
        importance = self.sensor_importance(sensor_features).squeeze(-1) * obs_mask
        local = self._local_gather(
            coordinates, obs_coordinates, sensor_features, obs_mask, importance
        )

        gamma, beta = self.coarse_film(global_features).chunk(2, dim=-1)
        coarse_features = point_features * (1.0 + torch.tanh(gamma).unsqueeze(1)) + beta.unsqueeze(
            1
        )
        coarse = self.coarse_scale * self.coarse_head(coarse_features)
        head_input = torch.cat((point_features, global_for_head, local), dim=-1)
        return coarse + self.head(self.head_in_norm(head_input))


class _LegacyRFFGaussianPrior(nn.Module):
    """Historical smooth Gaussian prior, including checkpoint buffer names."""

    def __init__(self, coordinate_dim: int, features: int, lengthscale: float) -> None:
        super().__init__()
        self.coord_dim = coordinate_dim
        self.n_features = features
        self.lengthscale = lengthscale
        self.register_buffer(
            "omega", torch.randn(coordinate_dim, features) / max(lengthscale, 1e-6)
        )
        self.register_buffer("phase", 2 * math.pi * torch.rand(features))

    def forward(self, coordinates: torch.Tensor, channels: int) -> torch.Tensor:
        features = math.sqrt(2.0 / self.n_features) * torch.cos(
            coordinates @ self.omega + self.phase
        )
        weights = torch.randn(
            coordinates.shape[0],
            channels,
            self.n_features,
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        return torch.einsum("bnf,bcf->bnc", features, weights)


class LegacyDemo50Model(nn.Module):
    """Version-locked RF wrapper whose state dictionary matches DemoN50."""

    capabilities = ModelCapabilities("point", True, True, False, True, ("post_training",))

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.model = _LegacyDemo50Velocity(config)
        self.prior = _LegacyRFFGaussianPrior(
            3, int(config["rff_features"]), float(config["rff_lengthscale"])
        )
        self.sigma_min = float(config.get("sigma_min", 1e-4))

    def sample_source(
        self,
        batch_or_coordinates: ObservationBatch | torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample the historical prior while containing its global RNG use."""
        coordinates = (
            batch_or_coordinates.query_coords
            if isinstance(batch_or_coordinates, ObservationBatch)
            else batch_or_coordinates
        )
        if generator is None:
            return self.prior(coordinates, self.model.n_fields)
        device_indices = [coordinates.device.index or 0] if coordinates.is_cuda else []
        with torch.random.fork_rng(devices=device_indices):
            torch.manual_seed(generator.initial_seed())
            return self.prior(coordinates, self.model.n_fields)

    def velocity(
        self,
        batch: ObservationBatch,
        state: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Expose the version-locked velocity through the common RF hook."""
        return self.model(
            time,
            state,
            batch.query_coords,
            batch.obs_coords,
            batch.obs_values,
            batch.obs_valid_mask,
            batch.obs_field_ids,
        )

    def training_loss(self, batch: ObservationBatch) -> LossBundle:
        if batch.target_fields is None:
            raise ValueError("DemoN50 RF data retention requires target_fields")
        source = self.sample_source(batch)
        time = torch.rand(batch.target_fields.shape[0], device=batch.target_fields.device)
        state = (1 - time[:, None, None]) * source + time[:, None, None] * batch.target_fields
        velocity = self.model(
            time,
            state,
            batch.query_coords,
            batch.obs_coords,
            batch.obs_values,
            batch.obs_valid_mask,
            batch.obs_field_ids,
        )
        loss = masked_mse(velocity, batch.target_fields - source, batch.query_valid_mask)
        return LossBundle(loss, {"rectified_flow_mse": loss})

    def integrate(
        self,
        coordinates: torch.Tensor,
        obs_coordinates: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        *,
        steps: int = 32,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate the historical 1-RF ODE with its default Euler convention."""
        if steps < 1:
            raise ValueError("steps must be at least one")
        state = self.sample_source(coordinates) if initial_state is None else initial_state.clone()
        times = torch.linspace(0.0, 1.0, steps + 1, device=state.device, dtype=state.dtype)
        for index in range(steps):
            time = times[index].expand(state.shape[0])
            state = state + (times[index + 1] - times[index]) * self.model(
                time, state, coordinates, obs_coordinates, obs_values, obs_mask, obs_field_ids
            )
        return state

    def reconstruct(
        self,
        batch: ObservationBatch,
        *,
        steps: int = 32,
        generator: torch.Generator | None = None,
        **_: Any,
    ) -> ReconstructionBatch:
        # Sampling uses the global RNG in the historical RFF convention. Fork
        # it when an explicit generator seed is supplied so callers can obtain
        # repeatable samples without altering checkpoint state.
        if generator is None:
            prediction = self.integrate(
                batch.query_coords,
                batch.obs_coords,
                batch.obs_values,
                batch.obs_valid_mask,
                batch.obs_field_ids,
                steps=steps,
            )
        else:
            device_indices = (
                [batch.query_coords.device.index or 0] if batch.query_coords.is_cuda else []
            )
            with torch.random.fork_rng(devices=device_indices):
                torch.manual_seed(generator.initial_seed())
                prediction = self.integrate(
                    batch.query_coords,
                    batch.obs_coords,
                    batch.obs_values,
                    batch.obs_valid_mask,
                    batch.obs_field_ids,
                    steps=steps,
                )
        return ReconstructionBatch(
            prediction,
            diagnostics={"compatibility_version": "demo50-v1", "sampling_steps": steps},
        )


@dataclass(frozen=True)
class Demo50CompatibilityManifest:
    compatibility_version: str
    source_run_directory: str
    source_checkpoint: str
    source_dataset_path: str
    source_hashes: dict[str, str]
    checkpoint_field_labels: tuple[str, ...]
    dataset_field_names: tuple[str, ...]
    channel_mapping: tuple[dict[str, Any], ...]
    normalization_mean: tuple[float, ...]
    normalization_std: tuple[float, ...]
    rectified_flow: dict[str, Any]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_fingerprint(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    return hashlib.sha256(payload).hexdigest()


def _read_h5_fields(path: Path) -> tuple[str, ...]:
    with h5py.File(path, "r") as handle:
        if "meta/field_names" in handle:
            raw = handle["meta/field_names"][...]
            return tuple(
                value.decode() if isinstance(value, bytes) else str(value) for value in raw
            )
        # The legacy combustion files predate the shared schema and record their
        # exact channel order on the dense field dataset.
        selected = handle["fields"].attrs.get("selected_fields")
        if isinstance(selected, bytes):
            selected = selected.decode()
        if isinstance(selected, str):
            return tuple(part.strip() for part in selected.split(","))
    raise ValueError(f"cannot resolve field names from {path}")


def _validate_channel_mapping(
    mapping: Sequence[Mapping[str, Any]],
    checkpoint_labels: Sequence[str],
    dataset_fields: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    normalized = tuple(
        {
            "channel": int(item["channel"]),
            "checkpoint_label": str(item["checkpoint_label"]),
            "dataset_field": str(item["dataset_field"]),
        }
        for item in mapping
    )
    expected = tuple(
        {"channel": index, "checkpoint_label": stale, "dataset_field": actual}
        for index, (stale, actual) in enumerate(zip(checkpoint_labels, dataset_fields))
    )
    if normalized != expected:
        raise ValueError(
            "DemoN50 has stale checkpoint field labels; provide the exact verified positional "
            f"mapping {expected}, received {normalized}"
        )
    return normalized


def _strict_load(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = sorted(
        f"{key}: checkpoint={tuple(state[key].shape)} model={tuple(expected[key].shape)}"
        for key in set(expected).intersection(state)
        if state[key].shape != expected[key].shape
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "DemoN50 strict state validation failed\n"
            f"missing={missing}\nunexpected={unexpected}\nshape_mismatches={mismatched}"
        )
    model.load_state_dict(state, strict=True)


def load_legacy_demo50(
    run_directory: str | Path,
    dataset_path: str | Path,
    channel_mapping: Sequence[Mapping[str, Any]],
    *,
    checkpoint: str = "best.pt",
    map_location: str | torch.device = "cpu",
) -> tuple[LegacyDemo50Model, Demo50CompatibilityManifest]:
    """Load DemoN50 only after validating provenance, fields, and every state key."""
    if checkpoint not in {"best.pt", "last.pt"}:
        raise ValueError("DemoN50 checkpoint must be best.pt or last.pt")
    run_directory = Path(run_directory).resolve()
    dataset_path = Path(dataset_path).resolve()
    required = {
        "args": run_directory / "args.json",
        "config": run_directory / "run_config.yaml",
        "checkpoint": run_directory / checkpoint,
        "normalization": run_directory / "dataset_stats.pt",
    }
    missing_files = [str(path) for path in required.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"DemoN50 compatibility inputs are missing: {missing_files}")

    args = json.loads(required["args"].read_text())
    run_config = yaml.safe_load(required["config"].read_text())
    payload = torch.load(required["checkpoint"], map_location=map_location, weights_only=False)
    normalization = torch.load(required["normalization"], map_location="cpu", weights_only=False)
    if args.get("Demo_Num") != 50 or run_config.get("Demo_Num") != 50:
        raise ValueError("compatibility adapter accepts only Demo_Num=50")
    if args.get("backbone") != "GL_rbf_ENH" or args.get("gather_mode") != "topk_rbf_glres":
        raise ValueError("DemoN50 source architecture does not match the version-locked adapter")

    actual_fields = _read_h5_fields(dataset_path)
    if actual_fields != DEMO50_DATASET_FIELDS:
        raise ValueError(f"expected source fields {DEMO50_DATASET_FIELDS}, found {actual_fields}")
    checkpoint_labels = tuple(payload["field_names"])
    if checkpoint_labels != DEMO50_STALE_CHECKPOINT_FIELDS:
        raise ValueError(f"unexpected checkpoint labels {checkpoint_labels}")
    verified_mapping = _validate_channel_mapping(channel_mapping, checkpoint_labels, actual_fields)

    model = LegacyDemo50Model(args)
    _strict_load(model, payload["model"])
    checkpoint_mean = tuple(float(value) for value in payload["mean"].cpu())
    checkpoint_std = tuple(float(value) for value in payload["std"].cpu())
    stats_mean = tuple(float(value) for value in normalization["mean"].cpu())
    stats_std = tuple(float(value) for value in normalization["std"].cpu())
    if checkpoint_mean != stats_mean or checkpoint_std != stats_std:
        raise ValueError("checkpoint normalization and dataset_stats.pt disagree")

    manifest = Demo50CompatibilityManifest(
        compatibility_version="demo50-v1",
        source_run_directory=str(run_directory),
        source_checkpoint=checkpoint,
        source_dataset_path=str(dataset_path),
        source_hashes={
            **{name: _sha256(path) for name, path in required.items()},
            # The multi-gigabyte legacy payload has no historical content hash;
            # record an explicit stat fingerprint rather than pretending it is
            # a cryptographic content checksum.
            "dataset_stat_fingerprint": _stat_fingerprint(dataset_path),
        },
        checkpoint_field_labels=checkpoint_labels,
        dataset_field_names=actual_fields,
        channel_mapping=verified_mapping,
        normalization_mean=checkpoint_mean,
        normalization_std=checkpoint_std,
        rectified_flow={
            "method": payload.get("method"),
            "prior": args["prior"],
            "rff_features": int(args["rff_features"]),
            "rff_lengthscale": float(args["rff_lengthscale"]),
            "ode_solver": payload["ode_solver"],
            "generation_steps": int(args["n_steps_generation"]),
            "coordinate_normalization": "legacy_minmax_unit_box",
        },
    )
    return model, manifest
