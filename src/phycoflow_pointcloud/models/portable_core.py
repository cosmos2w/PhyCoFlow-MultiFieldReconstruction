"""Self-contained frozen GL-RBF/CQ model and rectified-flow implementation.

This module is extracted verbatim from the Stage-8 numerical oracle except for
package-local imports. Public modules re-export its classes without renaming
parameters or state-dict keys.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pykeops.torch import LazyTensor
except ImportError:  # Torch remains a supported portable fallback.
    LazyTensor = None

from ..cache.geometry import (
    build_persistent_topk_geometry_cache,
    cache_tensors,
    validate_persistent_topk_geometry_cache,
)
from ..observation import (
    apply_endpoint_observation_consistency,
    build_pointwise_observation_maps,
    build_smooth_observation_maps,
    normalize_obs_consistency_mode,
    scatter_observed_values,
)

def make_mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int = 3, act=nn.GELU) -> nn.Sequential:
    layers = []
    dim = in_dim
    for _ in range(depth - 1):
        layers += [nn.Linear(dim, hidden_dim), act()]
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class FourierPositionalEncoding(nn.Module):
    """Sine-cosine frequency encoding for spatial coordinates."""

    def __init__(self, coord_dim: int, num_bands: int = 32, max_freq: float = 64.0):
        super().__init__()
        self.coord_dim = coord_dim
        self.num_bands = num_bands
        self.out_dim = coord_dim * num_bands * 2
        freqs = torch.linspace(1.0, max_freq / 2.0, num_bands)
        self.register_buffer("freqs", freqs)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords[..., : self.coord_dim] * 2.0 - 1.0
        x = coords.unsqueeze(-1) * self.freqs * math.pi
        enc = torch.cat([x.sin(), x.cos()], dim=-1)
        return enc.reshape(*coords.shape[:-1], self.out_dim)

# ------------------------------
# for gathering in GL_rbf
# ------------------------------
def batched_gather_2d(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather from x with shape [B, M] using idx with shape [B, N, K].
    Returns shape [B, N, K].
    """
    bsz = x.shape[0]
    batch_idx = torch.arange(bsz, device=x.device).view(bsz, 1, 1).expand_as(idx)
    return x[batch_idx, idx]


def batched_gather_3d(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather from x with shape [B, M, C] using idx with shape [B, N, K].
    Returns shape [B, N, K, C].
    """
    bsz = x.shape[0]
    batch_idx = torch.arange(bsz, device=x.device).view(bsz, 1, 1).expand_as(idx)
    return x[batch_idx, idx]

class FeedForward(nn.Module):
    """
    Standard Transformer feed-forward block used after attention.
    """
    def __init__(self, dim: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim * ff_mult
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block with residual connection and FFN.

    q  : [B, Tq, D]
    kv : [B, Tk, D]
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim=dim, ff_mult=ff_mult, dropout=mlp_dropout)
        # Runtime-only instrumentation. These counters are deliberately not
        # buffers so Stage-8 execution choices add no state-dict entries.
        self.kv_projection_calls = 0

    def reset_execution_counters(self) -> None:
        self.kv_projection_calls = 0

    def prepare_kv(
        self,
        kv: torch.Tensor,
        kv_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Normalize/project condition-static K/V once without detaching it."""
        kv_in = self.norm_kv(kv)
        dim = self.attn.embed_dim
        kv_proj = F.linear(
            kv_in,
            self.attn.in_proj_weight[dim:],
            None if self.attn.in_proj_bias is None else self.attn.in_proj_bias[dim:],
        )
        key, value = kv_proj.chunk(2, dim=-1)
        batch_size, source_length, _ = key.shape
        head_dim = dim // self.attn.num_heads
        key = key.view(batch_size, source_length, self.attn.num_heads, head_dim)
        value = value.view(batch_size, source_length, self.attn.num_heads, head_dim)
        attn_mask = None
        if kv_padding_mask is not None:
            attn_mask = torch.zeros(
                (batch_size, 1, 1, source_length),
                dtype=key.dtype,
                device=key.device,
            ).masked_fill(kv_padding_mask[:, None, None, :].bool(), float("-inf"))
            # Match MultiheadAttention's canonicalized key-padding-mask layout
            # so SDPA sees the same explicit per-head strides as the oracle.
            attn_mask = attn_mask.expand(
                batch_size, self.attn.num_heads, 1, source_length,
            ).contiguous()
        self.kv_projection_calls += 1
        return {
            "key": key.transpose(1, 2),
            "value": value.transpose(1, 2),
            "attn_mask": attn_mask,
        }

    def forward_prepared(
        self,
        q: torch.Tensor,
        prepared_kv: Mapping[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        """Run the original residual/FFN block using preprojected sensor K/V."""
        q_in = self.norm_q(q)
        dim = self.attn.embed_dim
        q_proj = F.linear(
            q_in,
            self.attn.in_proj_weight[:dim],
            None if self.attn.in_proj_bias is None else self.attn.in_proj_bias[:dim],
        )
        batch_size, target_length, _ = q_proj.shape
        head_dim = dim // self.attn.num_heads
        q_proj = q_proj.view(
            batch_size, target_length, self.attn.num_heads, head_dim,
        ).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            q_proj,
            prepared_kv["key"],
            prepared_kv["value"],
            attn_mask=prepared_kv["attn_mask"],
            dropout_p=self.attn.dropout if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch_size, target_length, dim,
        )
        attn_out = self.attn.out_proj(attn_out)
        x = q + attn_out
        return x + self.ff(self.norm_ff(x))

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Normalize queries and keys/values independently.
        q_in = self.norm_q(q)
        kv_in = self.norm_kv(kv)
        self.kv_projection_calls += 1

        # key_padding_mask: True means "ignore this token".
        attn_out, _ = self.attn(
            q_in,
            kv_in,
            kv_in,
            key_padding_mask=kv_padding_mask,
            need_weights=False,
        )

        x = q + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


class CompactLatentReadout(nn.Module):
    """Lightweight query-specific latent readout with cacheable latent K/V."""

    def __init__(
        self,
        query_in_dim: int,
        latent_dim: int,
        query_dim: int,
        rank: int = 64,
        num_heads: int = 4,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank < 1 or rank % num_heads != 0:
            raise ValueError(
                f"cq_readout_rank must be positive and divisible by cq_readout_heads; "
                f"got rank={rank}, heads={num_heads}."
            )
        if query_dim < 1 or query_dim % num_heads != 0:
            raise ValueError(
                f"cq_query_dim must be positive and divisible by cq_readout_heads; "
                f"got query_dim={query_dim}, heads={num_heads}."
            )
        self.rank = int(rank)
        self.num_heads = int(num_heads)
        self.head_rank = self.rank // self.num_heads
        self.query_dim = int(query_dim)
        self.head_value_dim = self.query_dim // self.num_heads
        self.attn_dropout = float(attn_dropout)

        self.query_norm = nn.LayerNorm(query_in_dim)
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.q_proj = nn.Linear(query_in_dim, rank, bias=False)
        self.k_proj = nn.Linear(latent_dim, rank, bias=False)
        self.v_proj = nn.Linear(latent_dim, query_dim, bias=False)
        self.out_norm = nn.LayerNorm(query_dim)

    def project_latents(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project condition-static latent memory once for reuse across chunks/NFEs."""
        bsz, n_latents, _ = latents.shape
        normalized = self.latent_norm(latents)
        keys = self.k_proj(normalized).view(
            bsz, n_latents, self.num_heads, self.head_rank,
        ).transpose(1, 2)
        values = self.v_proj(normalized).view(
            bsz, n_latents, self.num_heads, self.head_value_dim,
        ).transpose(1, 2)
        return keys, values

    def forward(
        self,
        query_features: torch.Tensor,
        *,
        latents: Optional[torch.Tensor] = None,
        projected_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if projected_kv is None:
            if latents is None:
                raise ValueError("CompactLatentReadout requires latents or projected_kv.")
            projected_kv = self.project_latents(latents)
        keys, values = projected_kv
        bsz, n_query, _ = query_features.shape
        queries = self.q_proj(self.query_norm(query_features)).view(
            bsz, n_query, self.num_heads, self.head_rank,
        ).transpose(1, 2)
        logits = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_rank)
        weights = torch.softmax(logits, dim=-1)
        weights = F.dropout(weights, p=self.attn_dropout, training=self.training)
        readout = torch.matmul(weights, values).transpose(1, 2).reshape(
            bsz, n_query, self.query_dim,
        )
        return self.out_norm(readout)


class SelfAttentionBlock(nn.Module):
    """
    Standard latent self-attention block with residual connection and FFN.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim=dim, ff_mult=ff_mult, dropout=mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.norm_attn(x)
        attn_out, _ = self.attn(x_in, x_in, x_in, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x

class ConditionalPointHybridLocalGlobalRBF(nn.Module):
    """
    Hybrid local-global backbone for conditional point-cloud FFM.

    Core Pipeline:
      1) Tokenization: Build sparse sensor tokens from (obs_coords, obs_values, obs_field_ids).
      2) Global Latent Encoding: A learned latent array cross-attends to the sparse sensor tokens,
         processing the field globally.
      3) Double-Dip Refinement: The sparse sensor tokens cross-attend back to the processed latents,
         yielding globally enriched local sensor tokens.
      4) Query Point Aggregation: Gather these enriched sensor tokens to arbitrary query points.
         Supported gather modes:
           - "rbf": Full dense RBF distance-based aggregation.
           - "topk_rbf": Sparse K-Nearest Neighbor RBF aggregation.
           - "topk_rbf_gate": Top-K RBF aggregation modulated by a learned query-sensor content gate.
           - "topk_rbf_ptlocal": Top-K RBF with a lightweight sensor-side local graph refinement.
           - "topk_rbf_glres": Top-K RBF plus cheap global residual readout/scaffold terms.
      5) Global Summary: Extract a global summary from the latents (via 'cls' or 'mean') and 
         concatenate it separately to every query point.
         The latent summary / CLS-like token acts strictly as a concatenated global feature 

    Hardware & Optimization Context:
      - neighbor_backend: Supports "torch" (standard pairwise matrices) and "keops" (LazyTensors).
      - KeOps Integration: The "keops" backend fundamentally eliminates the O(B * N * M) memory 
        bottleneck during pairwise distance computations, reducing it to O(N + M). This allows 
        for massive point clouds and largely removes the need for 'gather_query_chunk_size' loops.
      - Memory Layout: Inputs to KeOps routines are strictly enforced as `.contiguous()` to 
        prevent silent C++ reallocation bottlenecks.
    """

    _printed_gather_notices: set[tuple[str, int]] = set()

    @classmethod
    def _print_gather_notice_once(cls, gather_mode: str, gather_topk: int) -> None:
        key = (str(gather_mode), int(gather_topk))
        if key in cls._printed_gather_notices:
            return
        cls._printed_gather_notices.add(key)
        if key[0] == "rbf":
            print(f"\nThe gather mode is {gather_mode} as default choice.\n")
        else:
            print(f"\nNOTICE: The gather mode is {gather_mode} with top-k {gather_topk} !!!\n")

    def __init__(
        self,
        n_fields: int,
        coord_dim: int = 3,
        hidden_dim: int = 256,
        cond_dim: int = 128,
        field_embed_dim: int = 32,
        latent_dim: int = 256,
        num_latents: int = 64,
        num_heads: int = 8,
        num_latent_blocks: int = 3,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        rbf_sigma: float = 0.05,
        summary_type: str = "cls",   # ["cls", "mean"]

        gather_mode: str = "rbf",    # ["rbf", "topk_rbf", "topk_rbf_gate", "topk_rbf_ptlocal", "topk_rbf_glres"]
        gather_topk: int = 32,
        gather_query_chunk_size: Optional[int] = None,
        learnable_rbf_sigma: bool = False,
        neighbor_backend: str = "torch",      # ["auto", "torch", "keops"]

        sensor_local_topk: int = 8,
        sensor_local_dropout: float = 0.0,
        use_fourier_pe: bool = False,
        fourier_pe_num_bands: int = 32,
        fourier_pe_max_freq: float = 64.0,
        enhanced_backbone: bool = False,
        sensor_coord_encoding: str = "raw",
        latent_sensor_reinject: bool = False,
        latent_reinject_every: int = 1,
        condition_attention_execution: str = "legacy_mha",
        sensor_attention_padding_mode: str = "full",
        sensor_attention_buckets: Sequence[int] = (256, 320, 384),
        query_latent_readout: bool = False,
        query_readout_type: str = "point",
        query_readout_scale_init: float = 0.0,
        enhanced_head_norm: bool = False,
        glres_scale_init: float = 0.0,
    ) -> None:
        super().__init__()

        if summary_type not in ["cls", "mean"]:
            raise ValueError(f"summary_type must be 'cls' or 'mean', got {summary_type}")
        if sensor_coord_encoding not in ["raw", "fourier"]:
            raise ValueError(
                f"sensor_coord_encoding must be one of ['raw', 'fourier'], got {sensor_coord_encoding}"
            )
        if query_readout_type not in ["point", "coord"]:
            raise ValueError(
                f"query_readout_type must be one of ['point', 'coord'], got {query_readout_type}"
            )
        if latent_reinject_every < 1:
            raise ValueError(f"latent_reinject_every must be >= 1, got {latent_reinject_every}")
        if condition_attention_execution not in ("legacy_mha", "cached_kv"):
            raise ValueError(
                "condition_attention_execution must be 'legacy_mha' or 'cached_kv'."
            )
        if sensor_attention_padding_mode not in ("full", "static_buckets"):
            raise ValueError(
                "sensor_attention_padding_mode must be 'full' or 'static_buckets'."
            )
        normalized_buckets = tuple(sorted({int(value) for value in sensor_attention_buckets}))
        if not normalized_buckets or normalized_buckets[0] < 1:
            raise ValueError("sensor_attention_buckets must contain positive lengths.")

        self.n_fields = n_fields
        self.coord_dim = coord_dim
        self.rbf_sigma = rbf_sigma
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.summary_type = summary_type
        self.use_fourier_pe = use_fourier_pe
        self.pos_enc = FourierPositionalEncoding(
            coord_dim, num_bands=fourier_pe_num_bands, max_freq=fourier_pe_max_freq
        ) if use_fourier_pe else None
        self.coord_feat_dim = self.pos_enc.out_dim if self.pos_enc is not None else coord_dim
        self.enhanced_backbone = bool(enhanced_backbone)
        self.sensor_coord_encoding = sensor_coord_encoding
        self.latent_sensor_reinject = bool(latent_sensor_reinject)
        self.latent_reinject_every = int(latent_reinject_every)
        self.condition_attention_execution = str(condition_attention_execution)
        self.sensor_attention_padding_mode = str(sensor_attention_padding_mode)
        self.sensor_attention_buckets = normalized_buckets
        self.query_latent_readout_enabled = bool(query_latent_readout)
        self.query_readout_type = query_readout_type
        self.enhanced_head_norm = bool(enhanced_head_norm)

        gather_modes = ["rbf", "topk_rbf", "topk_rbf_gate", "topk_rbf_ptlocal", "topk_rbf_glres"]
        if gather_mode not in gather_modes:
            raise ValueError(
                f"gather_mode must be one of {gather_modes}, got {gather_mode}"
            )
        if neighbor_backend not in ["auto", "torch", "keops"]:
            raise ValueError(
                f"neighbor_backend must be one of ['auto', 'torch', 'keops'], got {neighbor_backend}"
            )
        self.gather_mode = gather_mode
        self.gather_topk = int(gather_topk)
        self.gather_query_chunk_size = gather_query_chunk_size
        self.learnable_rbf_sigma = learnable_rbf_sigma
        self.neighbor_backend = neighbor_backend

        self._print_gather_notice_once(gather_mode, gather_topk)

        # Only build the heavy query-side gate when the gate mode is actually selected.
        if self.gather_mode == "topk_rbf_gate":
            self.query_to_cond = nn.Linear(hidden_dim, cond_dim, bias=False)

            # Scalar query-neighbor reweighting.
            gate_in_dim = cond_dim + cond_dim + coord_dim + 1
            self.gather_gate = nn.Sequential(
                nn.Linear(gate_in_dim, cond_dim),
                nn.GELU(),
                nn.Linear(cond_dim, 1),
            )

        if self.gather_topk < 1:
            raise ValueError(f"gather_topk must be >= 1, got {self.gather_topk}")
        # Optional learnable locality scale
        if learnable_rbf_sigma:
            self.log_rbf_sigma = nn.Parameter(torch.log(torch.tensor(float(rbf_sigma))))
        # else:
        #     self.register_buffer("_fixed_rbf_sigma", torch.tensor(float(rbf_sigma)))
        #     self.log_rbf_sigma = None

        self.sensor_local_topk = int(sensor_local_topk)
        self.sensor_local_dropout_p = float(sensor_local_dropout)

        if self.sensor_local_topk < 1:
            raise ValueError(f"sensor_local_topk must be >= 1, got {self.sensor_local_topk}")

        # -------------------------
        # Point/query branch
        # -------------------------
        # Query point token from [coords, x_t, t]
        self.point_encoder = make_mlp(
            in_dim=self.coord_feat_dim + n_fields + 1,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            depth=3,
        )

        # -------------------------
        # Sparse sensor branch
        # -------------------------
        self.field_embed = nn.Embedding(n_fields, field_embed_dim)

        # Initial sparse sensor token from [obs_coords, obs_value, field_embed]
        sensor_coord_dim = (
            self.coord_feat_dim
            if sensor_coord_encoding == "fourier" and self.pos_enc is not None
            else coord_dim
        )
        self.sensor_in_proj = make_mlp(
            in_dim=sensor_coord_dim + 1 + field_embed_dim,
            hidden_dim=latent_dim,
            out_dim=latent_dim,
            depth=3,
        )

        # Project the refined sensor tokens to the local conditioning width
        # used by the RBF gather.
        self.sensor_out_proj = make_mlp(
            in_dim=latent_dim,
            hidden_dim=cond_dim,
            out_dim=cond_dim,
            depth=2,
        )

        # --------------------------------------------------
        # Optional sensor-side local refinement block Used only in gather_mode == "topk_rbf_ptlocal"
        # This is intentionally placed AFTER sensor_out_proj so it works on cond_dim features, 
        # which keeps memory and compute lower than refining in latent_dim.
        # --------------------------------------------------
        if self.gather_mode == "topk_rbf_ptlocal":
            self.sensor_local_q = nn.Linear(cond_dim, cond_dim, bias=False)
            self.sensor_local_k = nn.Linear(cond_dim, cond_dim, bias=False)
            self.sensor_local_v = nn.Linear(cond_dim, cond_dim, bias=False)
            # Relative position encoding: [dx, dy, dz, ||d||]
            self.sensor_local_pos = make_mlp(
                in_dim=coord_dim + 1,
                hidden_dim=cond_dim,
                out_dim=cond_dim,
                depth=2,
            )
            # Lightweight Point-Transformer-style scalar attention over local neighbors.
            self.sensor_local_attn = nn.Sequential(
                nn.Linear(cond_dim, cond_dim),
                nn.GELU(),
                nn.Linear(cond_dim, 1),
            )
            self.sensor_local_out = nn.Linear(cond_dim, cond_dim, bias=False)
            self.sensor_local_dropout = nn.Dropout(sensor_local_dropout)
            self.sensor_local_norm = nn.LayerNorm(cond_dim)

        # Optional query-to-latent readout can be used by enhanced GL_rbf for any gather mode.
        # The legacy topk_rbf_glres path reuses these same modules to preserve old behavior.
        self.use_query_latent_readout = self.query_latent_readout_enabled or self.gather_mode == "topk_rbf_glres"
        if self.use_query_latent_readout:
            if self.query_readout_type == "coord":
                self.query_decoder_token = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
                self.query_readout_in = nn.Linear(self.coord_feat_dim + hidden_dim, latent_dim, bias=False)
            else:
                self.query_decoder_token = None
                self.query_readout_in = nn.Linear(hidden_dim, latent_dim, bias=False)
            self.query_latent_readout = CrossAttentionBlock(
                dim=latent_dim,
                num_heads=max(1, min(num_heads, 4)),
                ff_mult=max(1, ff_mult // 2),
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
            )
            self.query_readout_out = nn.Linear(latent_dim, hidden_dim, bias=False)
            self.query_readout_scale = nn.Parameter(torch.tensor(float(query_readout_scale_init)))

        if self.gather_mode == "topk_rbf_glres":
            # Coarse scaffold is summary-driven and pointwise, so it avoids [B, N, K, C] tensors.
            self.coarse_film = nn.Linear(hidden_dim, 2 * hidden_dim)
            self.coarse_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(hidden_dim, n_fields),
            )
            self.coarse_scale = nn.Parameter(torch.tensor(float(glres_scale_init)))

            # Sensor importance is computed once per refined sensor token, then gathered as scalars.
            self.sensor_importance = nn.Sequential(
                nn.LayerNorm(cond_dim),
                nn.Linear(cond_dim, cond_dim),
                nn.GELU(),
                nn.Linear(cond_dim, 1),
            )
            self.sensor_importance_scale = nn.Parameter(torch.tensor(float(glres_scale_init)))

        # -------------------------
        # Latent global processor
        # -------------------------
        self.latents = nn.Parameter(
            torch.randn(num_latents, latent_dim) / math.sqrt(latent_dim)
        )

        # Latents attend to sparse sensor tokens
        self.input_cross_attn = CrossAttentionBlock(
            dim=latent_dim,
            num_heads=num_heads,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
        )

        # Process latents in latent space
        self.latent_blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=latent_dim,
                num_heads=num_heads,
                ff_mult=ff_mult,
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
            )
            for _ in range(num_latent_blocks)
        ])

        # Double-dip: refined local sensor tokens query the processed latents
        self.sensor_back_attn = CrossAttentionBlock(
            dim=latent_dim,
            num_heads=num_heads,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
        )

        # Separate projection for the latent summary used as a global feature
        self.summary_proj = make_mlp(
            in_dim=latent_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            depth=2,
        )

        # -------------------------
        # Final velocity head
        # -------------------------
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, n_fields),
        )
        head_in_dim = hidden_dim + hidden_dim + cond_dim
        self.head_in_norm = nn.LayerNorm(head_in_dim) if enhanced_head_norm else nn.Identity()

    def _build_sensor_tokens(
        self,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build sparse sensor tokens from:
          - sensor coordinates
          - observed scalar value
          - field identity embedding
        """
        safe_field_ids = obs_field_ids.clamp_min(0)
        field_feat = self.field_embed(safe_field_ids)                 # [B, M, E]
        field_feat = field_feat * obs_mask.unsqueeze(-1)             # zero padded rows

        # Enhanced mode uses the same Fourier coordinate representation for sensors and queries.
        # This mirrors Senseiver-style spatial tokenization while preserving GL_rbf's local gather.
        if self.sensor_coord_encoding == "fourier" and self.pos_enc is not None:
            sensor_coord_feat = self.pos_enc(obs_coords)
        else:
            sensor_coord_feat = obs_coords

        sensor_in = torch.cat([sensor_coord_feat, obs_values, field_feat], dim=-1)
        sensor_tokens = self.sensor_in_proj(sensor_in)               # [B, M, D]
        sensor_tokens = sensor_tokens * obs_mask.unsqueeze(-1)
        return sensor_tokens

    def _encode_latents(
        self,
        sensor_tokens: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Let the learned latent array absorb and process the sparse sensor set.
        """
        bsz = sensor_tokens.shape[0]

        # Expand learned latents across the batch
        latents = self.latents.unsqueeze(0).expand(bsz, -1, -1)      # [B, L, D]

        # key_padding_mask: True means "ignore this token"
        sensor_padding_mask = ~obs_mask.bool()

        bucket_groups = self._sensor_attention_bucket_groups(obs_mask)
        if bucket_groups is not None:
            prepared_groups = []
            for batch_indices, bucket_length in bucket_groups:
                tokens = sensor_tokens.index_select(0, batch_indices)[:, :bucket_length]
                padding = sensor_padding_mask.index_select(0, batch_indices)[:, :bucket_length]
                prepared = (
                    self.input_cross_attn.prepare_kv(tokens, padding)
                    if self.condition_attention_execution == "cached_kv"
                    else None
                )
                prepared_groups.append((batch_indices, tokens, padding, prepared))

            def attend(current_latents: torch.Tensor) -> torch.Tensor:
                output = torch.zeros_like(current_latents)
                for batch_indices, tokens, padding, prepared in prepared_groups:
                    group_latents = current_latents.index_select(0, batch_indices)
                    if prepared is None:
                        group_output = self.input_cross_attn(
                            q=group_latents, kv=tokens, kv_padding_mask=padding,
                        )
                    else:
                        group_output = self.input_cross_attn.forward_prepared(
                            group_latents, prepared,
                        )
                    output = output.index_copy(0, batch_indices, group_output)
                return output
        elif self.condition_attention_execution == "cached_kv":
            prepared = self.input_cross_attn.prepare_kv(
                sensor_tokens, sensor_padding_mask,
            )

            def attend(current_latents: torch.Tensor) -> torch.Tensor:
                return self.input_cross_attn.forward_prepared(current_latents, prepared)
        else:
            def attend(current_latents: torch.Tensor) -> torch.Tensor:
                return self.input_cross_attn(
                    q=current_latents,
                    kv=sensor_tokens,
                    kv_padding_mask=sensor_padding_mask,
                )

        # Latents attend to sparse sensor tokens
        latents = attend(latents)

        # Process in latent space, optionally re-reading sparse sensors between blocks.
        for i, block in enumerate(self.latent_blocks):
            if (
                self.latent_sensor_reinject
                and i > 0
                and i % self.latent_reinject_every == 0
            ):
                # Senseiver-style re-injection: latents re-read the sparse measurements.
                # Cost scales with L*M, not N*M, so this preserves query-side efficiency.
                latents = attend(latents)
            latents = block(latents)

        return latents

    def _sensor_attention_bucket_groups(
        self,
        obs_mask: torch.Tensor,
    ) -> Optional[list[tuple[torch.Tensor, int]]]:
        """Return stable batch groups, or None for the exact full-padding path."""
        if self.sensor_attention_padding_mode != "static_buckets":
            return None
        valid = obs_mask.bool()
        counts = valid.sum(dim=1)
        positions = torch.arange(valid.shape[1], device=valid.device).unsqueeze(0)
        if not torch.equal(valid, positions < counts.unsqueeze(1)):
            return None
        assigned = []
        max_length = int(valid.shape[1])
        for count in counts.tolist():
            bucket = next(
                (size for size in self.sensor_attention_buckets if size >= int(count)),
                max_length,
            )
            assigned.append(min(bucket, max_length))
        bucket_tensor = torch.tensor(assigned, device=valid.device)
        return [
            (torch.nonzero(bucket_tensor == bucket, as_tuple=False).flatten(), int(bucket))
            for bucket in sorted(set(assigned))
        ]

    def _refine_sensor_tokens(
        self,
        sensor_tokens: torch.Tensor,
        latents: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run sensor-back-attention without evaluating known padded tail slots."""
        bucket_groups = self._sensor_attention_bucket_groups(obs_mask)
        if bucket_groups is None:
            return self.sensor_back_attn(
                q=sensor_tokens, kv=latents, kv_padding_mask=None,
            )
        refined = torch.zeros_like(sensor_tokens)
        max_length = sensor_tokens.shape[1]
        for batch_indices, bucket_length in bucket_groups:
            group_refined = self.sensor_back_attn(
                q=sensor_tokens.index_select(0, batch_indices)[:, :bucket_length],
                kv=latents.index_select(0, batch_indices),
                kv_padding_mask=None,
            )
            if bucket_length < max_length:
                group_refined = F.pad(group_refined, (0, 0, 0, max_length - bucket_length))
            refined = refined.index_copy(0, batch_indices, group_refined)
        return refined

    def _extract_global_summary(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Convert the latent array into one global summary vector.

        If summary_type == 'cls', the last latent slot is treated as the summary token.
        If summary_type == 'mean', use the mean of all latent slots.
        """
        if self.summary_type == "cls":
            summary = latents[:, -1]         # [B, D]
        else:
            summary = latents.mean(dim=1)    # [B, D]

        return self.summary_proj(summary)    # [B, H]

    def _build_query_readout_tokens(
        self,
        point_feat: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build query tokens for latent readout from point features or coordinate decoder tokens.
        """
        if self.query_readout_type == "coord":
            bsz, n_query, _ = coords.shape
            coord_feat = self.pos_enc(coords) if self.pos_enc is not None else coords
            dq = self.query_decoder_token.view(1, 1, -1).expand(bsz, n_query, -1)
            return self.query_readout_in(torch.cat([coord_feat, dq], dim=-1))
        return self.query_readout_in(point_feat)

    def _readout_query_global_chunked(
        self,
        point_feat: torch.Tensor,
        coords: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Query-to-latent readout in chunks. This is O(B * N * L), avoiding query-sensor
        [B, N, K, C] feature materialization.
        """
        n_query = point_feat.shape[1]
        chunk_size = self.gather_query_chunk_size
        if chunk_size is None and n_query > 4096:
            chunk_size = 4096

        if chunk_size is None or n_query <= chunk_size:
            q = self._build_query_readout_tokens(point_feat, coords)
            readout = self.query_latent_readout(q=q, kv=latents, kv_padding_mask=None)
            return self.query_readout_out(readout)

        outputs = []
        for start in range(0, n_query, chunk_size):
            end = min(start + chunk_size, n_query)
            q = self._build_query_readout_tokens(point_feat[:, start:end], coords[:, start:end])
            readout = self.query_latent_readout(q=q, kv=latents, kv_padding_mask=None)
            outputs.append(self.query_readout_out(readout))
        return torch.cat(outputs, dim=1)

    def _predict_global_coarse(
        self,
        point_feat: torch.Tensor,
        global_feat: torch.Tensor,
    ) -> torch.Tensor:
        gamma, beta = self.coarse_film(global_feat).chunk(2, dim=-1)
        coarse_feat = (
            point_feat * (1.0 + torch.tanh(gamma).unsqueeze(1))
            + beta.unsqueeze(1)
        )
        return self.coarse_head(coarse_feat)

    def _compute_sensor_importance_bias(
        self,
        refined_sensor_feat: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        bias = self.sensor_importance(refined_sensor_feat).squeeze(-1)
        return bias * obs_mask.to(dtype=bias.dtype)

    def _use_keops(self) -> bool:
        """
        Decide whether to use KeOps.

        - rbf mode can benefit a lot from KeOps soft reductions
        - topk modes can use KeOps KNN search
        """
        if self.neighbor_backend == "torch":
            return False

        if self.neighbor_backend == "keops":
            if LazyTensor is None:
                raise ImportError(
                    "neighbor_backend='keops' was requested, but pykeops is not installed."
                )
            return True

        # auto
        return LazyTensor is not None

    def _aggregate_rbf_keops(
        self,
        query_coords: torch.Tensor,         # [B, N, D]
        obs_coords: torch.Tensor,           # [B, M, D]
        refined_sensor_feat: torch.Tensor,  # [B, M, Cc]
        obs_mask: torch.Tensor,             # [B, M]
    ) -> torch.Tensor:
        """
        Full RBF gather using KeOps sumsoftmaxweight, without building the dense [B, N, M] matrix.
        """
        sigma = torch.exp(self.log_rbf_sigma).clamp_min(1e-6) if self.learnable_rbf_sigma else self.rbf_sigma
        gamma = 1.0 / (2 * sigma ** 2 + 1e-12)

        # --- Force contiguous memory for KeOps ---
        query_coords = query_coords.contiguous()
        obs_coords = obs_coords.contiguous()
        refined_sensor_feat = refined_sensor_feat.contiguous()
        # -----------------------------------------

        # KeOps symbolic tensors
        x_i = LazyTensor(query_coords[:, :, None, :])                 # [B, N, 1, D]
        y_j = LazyTensor(obs_coords[:, None, :, :])                   # [B, 1, M, D]
        v_j = LazyTensor(refined_sensor_feat[:, None, :, :])          # [B, 1, M, Cc]

        # Scalar logits: -gamma * ||x_i - y_j||^2
        sqdist_ij = ((x_i - y_j) ** 2).sum(-1)                        # [B, N, M, 1]
        logits_ij = -gamma * sqdist_ij

        # Mask invalid sensor slots by adding a large negative number
        mask_j = LazyTensor(obs_mask[:, None, :, None].to(query_coords.dtype).contiguous())   # [B, 1, M, 1]
        logits_ij = logits_ij + (mask_j - 1.0) * 1e6

        # Softmax-weighted sum over the sensor axis.
        # With one batch dimension, the j-axis is dim=2.
        local_cond = logits_ij.sumsoftmaxweight(v_j, dim=2)           # [B, N, Cc]
        return local_cond

    def _knn_search_keops(
        self,
        query_coords: torch.Tensor,         # [B, N, D]
        obs_coords: torch.Tensor,           # [B, M, D]
        refined_sensor_feat: torch.Tensor,  # [B, M, Cc]
        obs_mask: torch.Tensor,             # [B, M]
        k: int,
        return_features: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        Top-k neighbor search using KeOps Kmin_argKmin.
        """

        # --- Force contiguous memory for KeOps ---
        query_coords = query_coords.contiguous()
        obs_coords = obs_coords.contiguous()
        # -----------------------------------------

        x_i = LazyTensor(query_coords[:, :, None, :])                 # [B, N, 1, D]
        y_j = LazyTensor(obs_coords[:, None, :, :])                   # [B, 1, M, D]

        sqdist_ij = ((x_i - y_j) ** 2).sum(-1)                        # [B, N, M, 1]

        # Mask invalid sensor slots
        mask_j = LazyTensor(obs_mask[:, None, :, None].to(query_coords.dtype).contiguous())
        sqdist_ij = sqdist_ij + (1.0 - mask_j) * 1e6

        # With one batch dimension, the j-axis is dim=2.
        topk_d2, topk_idx = sqdist_ij.Kmin_argKmin(K=k, dim=2)

        # KeOps can return indices in a non-long dtype; convert explicitly.
        topk_idx = topk_idx.long()

        topk_valid = batched_gather_2d(obs_mask, topk_idx).bool()
        if not return_features:
            return topk_d2, topk_idx, None, None, topk_valid

        topk_sensor_feat = batched_gather_3d(refined_sensor_feat, topk_idx)
        topk_sensor_coords = batched_gather_3d(obs_coords, topk_idx)
        return topk_d2, topk_idx, topk_sensor_feat, topk_sensor_coords, topk_valid

    def _knn_search_torch(
        self,
        query_coords: torch.Tensor,
        obs_coords: torch.Tensor,
        refined_sensor_feat: torch.Tensor,
        obs_mask: torch.Tensor,
        k: int,
        return_features: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        Fallback KNN search using torch.cdist + torch.topk.
        """
        d2 = torch.cdist(query_coords, obs_coords, p=2.0) ** 2
        large = torch.full_like(d2, 1e6)
        d2 = torch.where(obs_mask.unsqueeze(1) > 0, d2, large)

        topk_d2, topk_idx = torch.topk(d2, k=k, dim=-1, largest=False)

        topk_valid = batched_gather_2d(obs_mask, topk_idx).bool()
        if not return_features:
            return topk_d2, topk_idx, None, None, topk_valid

        topk_sensor_feat = batched_gather_3d(refined_sensor_feat, topk_idx)
        topk_sensor_coords = batched_gather_3d(obs_coords, topk_idx)
        return topk_d2, topk_idx, topk_sensor_feat, topk_sensor_coords, topk_valid

    def _get_topk_neighbors(
        self,
        query_coords: torch.Tensor,
        obs_coords: torch.Tensor,
        refined_sensor_feat: torch.Tensor,
        obs_mask: torch.Tensor,
        k: int,
        return_features: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        Unified top-k neighbor retrieval.
        """
        if self._use_keops():
            return self._knn_search_keops(
                query_coords=query_coords,
                obs_coords=obs_coords,
                refined_sensor_feat=refined_sensor_feat,
                obs_mask=obs_mask,
                k=k,
                return_features=return_features,
            )

        return self._knn_search_torch(
            query_coords=query_coords,
            obs_coords=obs_coords,
            refined_sensor_feat=refined_sensor_feat,
            obs_mask=obs_mask,
            k=k,
            return_features=return_features,
        )

    def _sensor_local_refine(
        self,
        sensor_coords: torch.Tensor,      # [B, M, D]
        sensor_feat: torch.Tensor,        # [B, M, Cc]
        obs_mask: torch.Tensor,           # [B, M]
    ) -> torch.Tensor:
        """
        Point-Transformer-style local refinement on the sensor graph.

        - This operates on M sensors, not N query points, so its memory cost is much
          smaller than query-side gating.
        - It gives each refined sensor token awareness of its local sensor neighborhood
          before the final query-side top-k RBF gather.

        Implementation notes:
        - Uses the existing neighbor backend (torch / keops) through _get_topk_neighbors.
        - Uses K+1 neighbors and drops the first one, which is usually the sensor itself.
        """
        # Search one extra neighbor so we can discard self-neighbor.
        k_search = min(self.sensor_local_topk + 1, sensor_coords.shape[1])

        nbr_d2, _, nbr_feat, nbr_coords, nbr_valid = self._get_topk_neighbors(
            query_coords=sensor_coords,
            obs_coords=sensor_coords,
            refined_sensor_feat=sensor_feat,
            obs_mask=obs_mask,
            k=k_search,
        )

        # Drop the first neighbor slot, which is typically the point itself.
        if k_search > 1:
            nbr_d2 = nbr_d2[:, :, 1:]
            nbr_feat = nbr_feat[:, :, 1:]
            nbr_coords = nbr_coords[:, :, 1:]
            nbr_valid = nbr_valid[:, :, 1:]

        # If there was only one valid sensor total, keep the feature unchanged.
        if nbr_feat.shape[2] == 0:
            return sensor_feat

        q = self.sensor_local_q(sensor_feat).unsqueeze(2)   # [B, M, 1, Cc]
        k = self.sensor_local_k(nbr_feat)                   # [B, M, Ks, Cc]
        v = self.sensor_local_v(nbr_feat)                   # [B, M, Ks, Cc]

        rel = sensor_coords.unsqueeze(2) - nbr_coords       # [B, M, Ks, D]
        rel_dist = torch.sqrt(nbr_d2.clamp_min(0.0)).unsqueeze(-1)  # [B, M, Ks, 1]
        pos = self.sensor_local_pos(torch.cat([rel, rel_dist], dim=-1))  # [B, M, Ks, Cc]

        # Lightweight Point-Transformer-style attention:
        # attention is driven by query-key difference plus relative position.
        attn_logits = self.sensor_local_attn(torch.tanh(q - k + pos)).squeeze(-1)  # [B, M, Ks]
        attn_logits = attn_logits.masked_fill(~nbr_valid, -1e9)
        attn = torch.softmax(attn_logits, dim=-1)

        update = torch.sum(attn.unsqueeze(-1) * (v + pos), dim=2)       # [B, M, Cc]
        out = self.sensor_local_norm(sensor_feat + self.sensor_local_dropout(self.sensor_local_out(update)))

        # Keep padded sensor rows zeroed out.
        out = out * obs_mask.unsqueeze(-1)
        return out

    def _aggregate_chunk(
        self,
        query_coords: torch.Tensor,         # [B, Nc, D]
        query_feat: torch.Tensor,           # [B, Nc, H]
        obs_coords: torch.Tensor,           # [B, M, D]
        refined_sensor_feat: torch.Tensor,  # [B, M, Cc]
        obs_mask: torch.Tensor,             # [B, M]
        sensor_importance_bias: Optional[torch.Tensor] = None,  # [B, M]
    ) -> torch.Tensor:
        """
        Aggregate one query chunk.
        """
        # sigma = self._get_rbf_sigma()
        sigma = torch.exp(self.log_rbf_sigma).clamp_min(1e-6) if self.learnable_rbf_sigma else self.rbf_sigma

        # --------------------------------------------------
        # Default: full RBF gather
        # --------------------------------------------------
        if self.gather_mode == "rbf":
            if self._use_keops():
                return self._aggregate_rbf_keops(
                    query_coords=query_coords,
                    obs_coords=obs_coords,
                    refined_sensor_feat=refined_sensor_feat,
                    obs_mask=obs_mask,
                )

            d2 = torch.cdist(query_coords, obs_coords, p=2.0) ** 2
            large = torch.full_like(d2, 1e6)
            d2 = torch.where(obs_mask.unsqueeze(1) > 0, d2, large)

            logits = -d2 / (2 * sigma ** 2 + 1e-12)
            weights = torch.softmax(logits, dim=-1)
            return torch.einsum("bnm,bmd->bnd", weights, refined_sensor_feat)

        # --------------------------------------------------
        # top-k modes
        # --------------------------------------------------
        k = min(self.gather_topk, obs_coords.shape[1])

        topk_d2, topk_idx, topk_sensor_feat, topk_sensor_coords, topk_valid = self._get_topk_neighbors(
            query_coords=query_coords,
            obs_coords=obs_coords,
            refined_sensor_feat=refined_sensor_feat,
            obs_mask=obs_mask,
            k=k,
        )

        logits = -topk_d2 / (2 * sigma ** 2 + 1e-12)

        if self.gather_mode == "topk_rbf_gate":
            query_cond = self.query_to_cond(query_feat)                    # [B, Nc, Cc]
            query_cond = query_cond.unsqueeze(2).expand(-1, -1, k, -1)    # [B, Nc, k, Cc]

            rel = query_coords.unsqueeze(2) - topk_sensor_coords           # [B, Nc, k, D]
            rel_dist = torch.sqrt(topk_d2.clamp_min(0.0)).unsqueeze(-1)    # [B, Nc, k, 1]

            gate_in = torch.cat([query_cond, topk_sensor_feat, rel, rel_dist], dim=-1)
            gate_logits = self.gather_gate(gate_in).squeeze(-1)            # [B, Nc, k]

            logits = logits + gate_logits

        if sensor_importance_bias is not None:
            topk_sensor_bias = batched_gather_2d(sensor_importance_bias, topk_idx)
            logits = logits + self.sensor_importance_scale * topk_sensor_bias

        logits = logits.masked_fill(~topk_valid, -1e9)
        weights = torch.softmax(logits, dim=-1)
        local_cond = torch.sum(weights.unsqueeze(-1) * topk_sensor_feat, dim=2)
        return local_cond

    def aggregate_sparse_obs(
        self,
        query_coords: torch.Tensor,
        query_feat: torch.Tensor,
        obs_coords: torch.Tensor,
        refined_sensor_feat: torch.Tensor,
        obs_mask: torch.Tensor,
        sensor_importance_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Gather the globally enriched local sensor features back to query points.

        Policy:
          - rbf: with KeOps, chunking can usually be disabled
          - topk_rbf: with KeOps, chunking can usually be disabled
          - topk_rbf_gate: still keep optional chunking because gate tensors are [B, N, K, U]
        """
        n_query = query_coords.shape[1]

        if self.gather_mode == "topk_rbf_gate":
            # Gate mode still benefits from chunking because it builds [B, N, K, ...] tensors.
            chunk_size = self.gather_query_chunk_size if self.gather_query_chunk_size is not None else 2048
        else:
            # rbf / topk_rbf / topk_rbf_ptlocal all keep the cheaper gather path.
            chunk_size = self.gather_query_chunk_size

        if chunk_size is None or n_query <= chunk_size:
            return self._aggregate_chunk(
                query_coords=query_coords,
                query_feat=query_feat,
                obs_coords=obs_coords,
                refined_sensor_feat=refined_sensor_feat,
                obs_mask=obs_mask,
                sensor_importance_bias=sensor_importance_bias,
            )

        outputs = []
        for start in range(0, n_query, chunk_size):
            end = min(start + chunk_size, n_query)

            local_chunk = self._aggregate_chunk(
                query_coords=query_coords[:, start:end],
                query_feat=query_feat[:, start:end],
                obs_coords=obs_coords,
                refined_sensor_feat=refined_sensor_feat,
                obs_mask=obs_mask,
                sensor_importance_bias=sensor_importance_bias,
            )
            outputs.append(local_chunk)

        return torch.cat(outputs, dim=1)

    def prepare_condition_context(
        self,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Build differentiable observation-only state once per condition."""
        sensor_tokens = self._build_sensor_tokens(
            obs_coords, obs_values, obs_mask, obs_field_ids,
        )
        latents = self._encode_latents(sensor_tokens, obs_mask)
        global_feat = self._extract_global_summary(latents)
        refined = self._refine_sensor_tokens(sensor_tokens, latents, obs_mask)
        refined = refined * obs_mask.unsqueeze(-1)
        refined = self.sensor_out_proj(refined) * obs_mask.unsqueeze(-1)
        if self.gather_mode == "topk_rbf_ptlocal":
            refined = self._sensor_local_refine(obs_coords, refined, obs_mask)
        context = {
            "obs_coords": obs_coords,
            "obs_mask": obs_mask,
            "latents": latents,
            "global_feat": global_feat,
            "refined_sensor_feat": refined,
        }
        if self.gather_mode == "topk_rbf_glres":
            context["sensor_importance_bias"] = self._compute_sensor_importance_bias(
                refined, obs_mask,
            )
        return context

    def _aggregate_topk_from_geometry(
        self,
        topk_d2: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_valid: torch.Tensor,
        condition_context: Mapping[str, torch.Tensor],
        topk_sensor_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.gather_mode not in ("topk_rbf", "topk_rbf_glres"):
            raise ValueError("Geometry caching supports topk_rbf and topk_rbf_glres.")
        sensor_feat = topk_sensor_feat
        if sensor_feat is None:
            sensor_feat = batched_gather_3d(
                condition_context["refined_sensor_feat"], topk_idx,
            )
        sigma = (
            torch.exp(self.log_rbf_sigma).clamp_min(1e-6)
            if self.learnable_rbf_sigma else self.rbf_sigma
        )
        logits = -topk_d2 / (2 * sigma ** 2 + 1e-12)
        importance = condition_context.get("sensor_importance_bias")
        if importance is not None:
            logits = logits + self.sensor_importance_scale * batched_gather_2d(
                importance, topk_idx,
            )
        weights = torch.softmax(logits.masked_fill(~topk_valid, -1e9), dim=-1)
        return torch.sum(weights.unsqueeze(-1) * sensor_feat, dim=2)

    def prepare_query_context(
        self,
        coords: torch.Tensor,
        condition_context: Mapping[str, torch.Tensor],
        cache_level: str = "none",
        chunk_size: Optional[int] = None,
        precomputed_geometry: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Cache geometry or inference-only static query features in FP32."""
        if cache_level not in ("none", "geometry", "static_features"):
            raise ValueError("Unknown reconstruction cache level.")
        chunk_size = max(1, int(chunk_size or self.gather_query_chunk_size or 8192))
        context: Dict[str, Any] = {
            "cache_level": cache_level,
            "n_query": int(coords.shape[1]),
            "chunk_size": chunk_size,
        }
        if cache_level == "none":
            if precomputed_geometry is not None:
                raise ValueError("Persistent geometry requires geometry or static_features cache level.")
            return context
        if self.gather_mode not in ("topk_rbf", "topk_rbf_glres"):
            raise ValueError("Query caching supports topk_rbf and topk_rbf_glres.")
        persistent_topk = None
        if precomputed_geometry is not None:
            validate_persistent_topk_geometry_cache(
                precomputed_geometry, self, coords=coords,
                obs_coords=condition_context["obs_coords"],
                obs_mask=condition_context["obs_mask"],
            )
            persistent_topk = cache_tensors(precomputed_geometry)
        if self.pos_enc is None:
            context["coord_feat"] = coords
        else:
            coord_feat = coords.new_empty(coords.shape[0], coords.shape[1], self.coord_feat_dim)
            for start in range(0, coords.shape[1], chunk_size):
                end = min(start + chunk_size, coords.shape[1])
                coord_feat[:, start:end] = self.pos_enc(coords[:, start:end])
            context["coord_feat"] = coord_feat
        if cache_level == "geometry":
            if persistent_topk is not None:
                topk_d2, topk_idx, topk_valid = persistent_topk
                context.update(topk_d2=topk_d2, topk_idx=topk_idx, topk_valid=topk_valid)
                return context
            k = min(self.gather_topk, condition_context["obs_coords"].shape[1])
            topk_d2 = coords.new_empty(coords.shape[0], coords.shape[1], k)
            topk_idx = torch.empty(
                coords.shape[0], coords.shape[1], k,
                dtype=torch.long, device=coords.device,
            )
            topk_valid = torch.empty(
                coords.shape[0], coords.shape[1], k,
                dtype=torch.bool, device=coords.device,
            )
            for start in range(0, coords.shape[1], chunk_size):
                end = min(start + chunk_size, coords.shape[1])
                d2, idx, _, _, valid = self._get_topk_neighbors(
                    coords[:, start:end],
                    condition_context["obs_coords"],
                    condition_context["refined_sensor_feat"],
                    condition_context["obs_mask"],
                    k,
                )
                topk_d2[:, start:end] = d2
                topk_idx[:, start:end] = idx
                topk_valid[:, start:end] = valid
            context.update(
                topk_d2=topk_d2,
                topk_idx=topk_idx,
                topk_valid=topk_valid,
            )
            return context
        if self.training and torch.is_grad_enabled():
            raise ValueError("static_features caching is inference-only.")
        if self.use_query_latent_readout and self.query_readout_type != "coord":
            raise ValueError("static_features requires coordinate query readout.")
        local_cache = coords.new_empty(
            coords.shape[0], coords.shape[1],
            condition_context["refined_sensor_feat"].shape[-1],
        )
        query_global_cache = (
            coords.new_empty(coords.shape[0], coords.shape[1], condition_context["global_feat"].shape[-1])
            if self.use_query_latent_readout else None
        )
        for start in range(0, coords.shape[1], chunk_size):
            end = min(start + chunk_size, coords.shape[1])
            coords_c = coords[:, start:end]
            empty_feat = coords_c.new_empty(coords_c.shape[0], coords_c.shape[1], 0)
            if persistent_topk is None:
                local_cache[:, start:end] = self._aggregate_chunk(
                    coords_c,
                    empty_feat,
                    condition_context["obs_coords"],
                    condition_context["refined_sensor_feat"],
                    condition_context["obs_mask"],
                    condition_context.get("sensor_importance_bias"),
                )
            else:
                topk_d2, topk_idx, topk_valid = persistent_topk
                local_cache[:, start:end] = self._aggregate_topk_from_geometry(
                    topk_d2[:, start:end],
                    topk_idx[:, start:end],
                    topk_valid[:, start:end],
                    condition_context,
                )
            if self.use_query_latent_readout:
                q = self._build_query_readout_tokens(empty_feat, coords_c)
                readout = self.query_latent_readout(
                    q=q, kv=condition_context["latents"], kv_padding_mask=None,
                )
                query_global_cache[:, start:end] = self.query_readout_out(readout)
        context["local_cond"] = local_cache
        if query_global_cache is not None:
            context["query_global"] = query_global_cache
        return context

    def forward_query_chunk(
        self,
        t: torch.Tensor,
        x_t_chunk: torch.Tensor,
        coords_chunk: torch.Tensor,
        condition_context: Mapping[str, torch.Tensor],
        query_context: Optional[Mapping[str, Any]] = None,
        query_slice: Optional[slice] = None,
    ) -> torch.Tensor:
        """Run the complete dynamic velocity head for one query chunk."""
        bsz, n_pts, _ = x_t_chunk.shape
        query_slice = query_slice or slice(0, n_pts)
        cached_coord = None if query_context is None else query_context.get("coord_feat")
        coord_feat = (
            cached_coord[:, query_slice]
            if cached_coord is not None
            else (self.pos_enc(coords_chunk) if self.pos_enc is not None else coords_chunk)
        )
        t_feat = t.view(bsz, 1, 1).expand(bsz, n_pts, 1)
        point_feat = self.point_encoder(torch.cat([coord_feat, x_t_chunk, t_feat], dim=-1))
        global_feat = condition_context["global_feat"]
        if self.use_query_latent_readout:
            cached_global = None if query_context is None else query_context.get("query_global")
            if cached_global is None:
                q = self._build_query_readout_tokens(point_feat, coords_chunk)
                readout = self.query_latent_readout(
                    q=q, kv=condition_context["latents"], kv_padding_mask=None,
                )
                query_global = self.query_readout_out(readout)
            else:
                query_global = cached_global[:, query_slice]
            global_for_head = global_feat.unsqueeze(1) + self.query_readout_scale * query_global
        else:
            global_for_head = global_feat.unsqueeze(1).expand(bsz, n_pts, -1)
        cached_local = None if query_context is None else query_context.get("local_cond")
        if cached_local is not None:
            local_cond = cached_local[:, query_slice]
        elif query_context is not None and "topk_idx" in query_context:
            local_cond = self._aggregate_topk_from_geometry(
                query_context["topk_d2"][:, query_slice],
                query_context["topk_idx"][:, query_slice],
                query_context["topk_valid"][:, query_slice],
                condition_context,
            )
        else:
            local_cond = self.aggregate_sparse_obs(
                coords_chunk,
                point_feat,
                condition_context["obs_coords"],
                condition_context["refined_sensor_feat"],
                condition_context["obs_mask"],
                condition_context.get("sensor_importance_bias"),
            )
        head_in = torch.cat([point_feat, global_for_head, local_cond], dim=-1)
        residual = self.head(self.head_in_norm(head_in))
        if self.gather_mode == "topk_rbf_glres":
            return self.coarse_scale * self._predict_global_coarse(point_feat, global_feat) + residual
        return residual

    @staticmethod
    def context_nbytes(context: Mapping[str, Any]) -> int:
        seen: set[tuple[int, int]] = set()
        total = 0

        def visit(value: Any) -> None:
            nonlocal total
            if torch.is_tensor(value):
                storage = value.untyped_storage()
                key = (storage.data_ptr(), storage.nbytes())
                if key not in seen:
                    seen.add(key)
                    total += storage.nbytes()
            elif isinstance(value, Mapping):
                for child in value.values():
                    visit(child)

        visit(context)
        return total

    def forward(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Output:
            velocity field of shape [B, N, C]
        """
        bsz, n_pts, _ = x_t.shape

        # -------------------------
        # Query-point features
        # -------------------------
        t_feat = t.view(bsz, 1, 1).expand(bsz, n_pts, 1)
        coord_feat = self.pos_enc(coords) if self.pos_enc is not None else coords
        point_feat = self.point_encoder(torch.cat([coord_feat, x_t, t_feat], dim=-1))  # [B, N, H]

        # -------------------------
        # Local sensor tokens
        # -------------------------
        sensor_tokens = self._build_sensor_tokens(
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
        )  # [B, M, D]

        # -------------------------
        # Global latent processing
        # -------------------------
        latents = self._encode_latents(sensor_tokens=sensor_tokens, obs_mask=obs_mask)  # [B, L, D]

        # The global summary remains the cheap broadcast path; enhanced mode can add
        # a per-query latent readout before the final local-global fusion head.
        global_feat = self._extract_global_summary(latents)                 # [B, H]
        if self.use_query_latent_readout:
            query_global = self._readout_query_global_chunked(point_feat, coords, latents)  # [B, N, H]
            global_for_head = global_feat.unsqueeze(1) + self.query_readout_scale * query_global
        else:
            global_for_head = global_feat.unsqueeze(1).expand(bsz, n_pts, -1)

        # -------------------------
        # Double-dip refinement:
        # sensor tokens query back into the latent memory
        # -------------------------
        refined_sensor_tokens = self._refine_sensor_tokens(
            sensor_tokens, latents, obs_mask,
        )  # [B, M, D]

        # Zero out padded sensor rows again after attention
        refined_sensor_tokens = refined_sensor_tokens * obs_mask.unsqueeze(-1)

        # Project refined sensor tokens to the local conditioning width
        refined_sensor_feat = self.sensor_out_proj(refined_sensor_tokens)   # [B, M, cond_dim]
        refined_sensor_feat = refined_sensor_feat * obs_mask.unsqueeze(-1)

        if self.gather_mode == "topk_rbf_glres":
            sensor_importance_bias = self._compute_sensor_importance_bias(
                refined_sensor_feat=refined_sensor_feat,
                obs_mask=obs_mask,
            )

            local_cond = self.aggregate_sparse_obs(
                query_coords=coords,
                query_feat=point_feat,
                obs_coords=obs_coords,
                refined_sensor_feat=refined_sensor_feat,
                obs_mask=obs_mask,
                sensor_importance_bias=sensor_importance_bias,
            )  # [B, N, cond_dim]

            coarse_pred = self.coarse_scale * self._predict_global_coarse(point_feat, global_feat)

            head_in = torch.cat([point_feat, global_for_head, local_cond], dim=-1)
            residual = self.head(self.head_in_norm(head_in))
            return coarse_pred + residual

        # Optional sensor-side local graph refinement.
        if self.gather_mode == "topk_rbf_ptlocal":
            refined_sensor_feat = self._sensor_local_refine(
                sensor_coords=obs_coords,
                sensor_feat=refined_sensor_feat,
                obs_mask=obs_mask,)

        # -------------------------
        # Gather back to queries
        # -------------------------
        local_cond = self.aggregate_sparse_obs(
            query_coords=coords,
            query_feat=point_feat,
            obs_coords=obs_coords,
            refined_sensor_feat=refined_sensor_feat,
            obs_mask=obs_mask,
        )  # [B, N, cond_dim]

        # -------------------------
        # Final velocity prediction
        # -------------------------
        head_in = torch.cat([point_feat, global_for_head, local_cond], dim=-1)
        out = self.head(self.head_in_norm(head_in))
        return out


# ------------------------------
class ConditionalPointHybridLocalGlobalRBFCQ(ConditionalPointHybridLocalGlobalRBF):
    """Compact-query sibling of GL_rbf_ENH with an unchanged condition/local core."""

    def __init__(
        self,
        n_fields: int,
        coord_dim: int = 3,
        hidden_dim: int = 256,
        cond_dim: int = 128,
        field_embed_dim: int = 32,
        latent_dim: int = 256,
        num_latents: int = 64,
        num_heads: int = 8,
        num_latent_blocks: int = 3,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        rbf_sigma: float = 0.05,
        summary_type: str = "cls",
        gather_mode: str = "topk_rbf_glres",
        gather_topk: int = 32,
        gather_query_chunk_size: Optional[int] = None,
        learnable_rbf_sigma: bool = False,
        neighbor_backend: str = "torch",
        sensor_local_topk: int = 8,
        sensor_local_dropout: float = 0.0,
        use_fourier_pe: bool = False,
        fourier_pe_num_bands: int = 32,
        fourier_pe_max_freq: float = 64.0,
        sensor_coord_encoding: str = "fourier",
        latent_sensor_reinject: bool = True,
        latent_reinject_every: int = 1,
        condition_attention_execution: str = "legacy_mha",
        sensor_attention_padding_mode: str = "full",
        sensor_attention_buckets: Sequence[int] = (256, 320, 384),
        glres_scale_init: float = 1.0e-2,
        cq_query_dim: int = 128,
        cq_readout_mode: str = "lowrank",
        cq_readout_rank: int = 64,
        cq_readout_heads: int = 4,
        cq_global_scale_init: float = 1.0,
        cq_local_scale_init: float = 1.0,
        cq_readout_scale_init: float = 1.0e-2,
        cq_fusion_mode: str = "additive",
        cq_time_conditioning: str = "scalar_concat",
        cq_time_embed_dim: int = 128,
        cq_time_max_period: float = 10000.0,
        cq_time_film_zero_init: bool = True,
        cq_measurement_support_mode: str = "none",
        cq_measurement_support_normalize: bool = True,
    ) -> None:
        if cq_readout_mode not in ("full", "lowrank"):
            raise ValueError(
                f"cq_readout_mode must be one of ['full', 'lowrank'], got {cq_readout_mode!r}."
            )
        if cq_fusion_mode not in ("additive", "structured_concat"):
            raise ValueError(
                "cq_fusion_mode must be one of ['additive', 'structured_concat'], "
                f"got {cq_fusion_mode!r}."
            )
        if cq_readout_heads < 1:
            raise ValueError(f"cq_readout_heads must be positive, got {cq_readout_heads}.")
        if cq_query_dim < 1 or cq_query_dim % cq_readout_heads != 0:
            raise ValueError(
                "cq_query_dim must be positive and divisible by cq_readout_heads; "
                f"got query_dim={cq_query_dim}, heads={cq_readout_heads}."
            )
        if cq_readout_rank < 1 or cq_readout_rank % cq_readout_heads != 0:
            raise ValueError(
                "cq_readout_rank must be positive and divisible by cq_readout_heads; "
                f"got rank={cq_readout_rank}, heads={cq_readout_heads}."
            )
        if cq_time_conditioning not in ("scalar_concat", "sinusoidal_film"):
            raise ValueError(
                "cq_time_conditioning must be 'scalar_concat' or 'sinusoidal_film'."
            )
        if cq_time_embed_dim < 2:
            raise ValueError("cq_time_embed_dim must be at least 2.")
        if cq_time_max_period <= 0:
            raise ValueError("cq_time_max_period must be positive.")
        if cq_measurement_support_mode not in ("none", "rbf_value_support"):
            raise ValueError(
                "cq_measurement_support_mode must be 'none' or 'rbf_value_support'."
            )

        # Build the unchanged F0 condition/global/local core first. The inherited
        # query modules are then removed and replaced, leaving GL_rbf_ENH untouched.
        super().__init__(
            n_fields=n_fields,
            coord_dim=coord_dim,
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
            field_embed_dim=field_embed_dim,
            latent_dim=latent_dim,
            num_latents=num_latents,
            num_heads=num_heads,
            num_latent_blocks=num_latent_blocks,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
            rbf_sigma=rbf_sigma,
            summary_type=summary_type,
            gather_mode=gather_mode,
            gather_topk=gather_topk,
            gather_query_chunk_size=gather_query_chunk_size,
            learnable_rbf_sigma=learnable_rbf_sigma,
            neighbor_backend=neighbor_backend,
            sensor_local_topk=sensor_local_topk,
            sensor_local_dropout=sensor_local_dropout,
            use_fourier_pe=use_fourier_pe,
            fourier_pe_num_bands=fourier_pe_num_bands,
            fourier_pe_max_freq=fourier_pe_max_freq,
            enhanced_backbone=True,
            sensor_coord_encoding=sensor_coord_encoding,
            latent_sensor_reinject=latent_sensor_reinject,
            latent_reinject_every=latent_reinject_every,
            condition_attention_execution=condition_attention_execution,
            sensor_attention_padding_mode=sensor_attention_padding_mode,
            sensor_attention_buckets=sensor_attention_buckets,
            query_latent_readout=True,
            query_readout_type="coord",
            query_readout_scale_init=cq_readout_scale_init,
            enhanced_head_norm=True,
            glres_scale_init=glres_scale_init,
        )

        self.cq_query_dim = int(cq_query_dim)
        self.cq_readout_mode = str(cq_readout_mode)
        self.cq_fusion_mode = str(cq_fusion_mode)
        self.cq_readout_rank = int(cq_readout_rank)
        self.cq_readout_heads = int(max(1, min(num_heads, 4)) if cq_readout_mode == "full" else cq_readout_heads)
        self.cq_time_conditioning = str(cq_time_conditioning)
        self.cq_time_embed_dim = int(cq_time_embed_dim)
        self.cq_time_max_period = float(cq_time_max_period)
        self.cq_time_film_zero_init = bool(cq_time_film_zero_init)
        self.cq_measurement_support_mode = str(cq_measurement_support_mode)
        self.cq_measurement_support_normalize = bool(cq_measurement_support_normalize)
        self.cq_timestep_film_enabled = self.cq_time_conditioning == "sinusoidal_film"
        self.cq_measurement_support_enabled = self.cq_measurement_support_mode == "rbf_value_support"
        self.hidden_dim = int(hidden_dim)
        self.cond_dim = int(cond_dim)

        for name in (
            "point_encoder", "query_decoder_token", "query_readout_in",
            "query_latent_readout", "query_readout_out", "query_readout_scale",
            "head", "head_in_norm", "coarse_film", "coarse_head", "coarse_scale",
        ):
            if hasattr(self, name):
                delattr(self, name)

        if self.gather_mode == "topk_rbf_gate":
            self.query_to_cond = nn.Linear(self.cq_query_dim, cond_dim, bias=False)

        self.cq_point_encoder = make_mlp(
            in_dim=self.coord_feat_dim + n_fields + 1,
            hidden_dim=self.cq_query_dim,
            out_dim=self.cq_query_dim,
            depth=3,
        )
        self.cq_global_proj = nn.Linear(hidden_dim, self.cq_query_dim)
        if self.cq_timestep_film_enabled:
            self.cq_timestep_mlp = nn.Sequential(
                nn.Linear(self.cq_time_embed_dim, self.cq_time_embed_dim),
                nn.SiLU(),
                nn.Linear(self.cq_time_embed_dim, self.cq_time_embed_dim),
                nn.SiLU(),
            )
            self.cq_timestep_film = nn.Linear(
                self.cq_time_embed_dim, 2 * self.cq_query_dim,
            )
            if self.cq_time_film_zero_init:
                nn.init.zeros_(self.cq_timestep_film.weight)
                nn.init.zeros_(self.cq_timestep_film.bias)
        raw_feature_dim = 2 * self.n_fields if self.cq_measurement_support_enabled else 0
        if self.cq_measurement_support_enabled and self.cq_measurement_support_normalize:
            self.cq_measurement_support_norm = nn.LayerNorm(raw_feature_dim)
        if self.cq_fusion_mode == "additive":
            # Keep this module set, ordering, and all shapes unchanged so CQ
            # checkpoints created before cq_fusion_mode continue to strict-load.
            self.cq_local_proj = (
                nn.Identity()
                if cond_dim == self.cq_query_dim
                else nn.Linear(cond_dim, self.cq_query_dim)
            )
            self.cq_global_scale = nn.Parameter(torch.tensor(float(cq_global_scale_init)))
            self.cq_local_scale = nn.Parameter(torch.tensor(float(cq_local_scale_init)))
            self.cq_readout_scale = nn.Parameter(torch.tensor(float(cq_readout_scale_init)))
            fusion_dim = self.cq_query_dim + raw_feature_dim
            self.cq_fusion_norm = nn.LayerNorm(fusion_dim)
            self.cq_head = nn.Sequential(
                nn.Linear(fusion_dim, self.cq_query_dim),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(self.cq_query_dim, self.cq_query_dim),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(self.cq_query_dim, n_fields),
            )
        else:
            fusion_dim = 2 * self.cq_query_dim + self.cond_dim + raw_feature_dim
            self.cq_readout_scale = nn.Parameter(torch.tensor(float(cq_readout_scale_init)))
            self.cq_fusion_norm = nn.LayerNorm(fusion_dim)
            self.cq_head = nn.Sequential(
                nn.Linear(fusion_dim, self.cq_query_dim),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(self.cq_query_dim, self.cq_query_dim),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(self.cq_query_dim, n_fields),
            )
        if self.gather_mode == "topk_rbf_glres":
            self.cq_coarse_film = nn.Linear(self.cq_query_dim, 2 * self.cq_query_dim)
            self.cq_coarse_head = nn.Sequential(
                nn.LayerNorm(self.cq_query_dim),
                nn.Linear(self.cq_query_dim, self.cq_query_dim),
                nn.GELU(),
                nn.Linear(self.cq_query_dim, n_fields),
            )
            self.cq_coarse_scale = nn.Parameter(torch.tensor(float(glres_scale_init)))

        # Construct the only variant-specific module last so CQ-Full and CQ-LR
        # receive identical seed-controlled initialization for every shared CQ module.
        if self.cq_readout_mode == "full":
            self.cq_query_decoder_token = nn.Parameter(
                torch.randn(1, self.cq_query_dim) * 0.02
            )
            self.cq_readout_in = nn.Linear(
                self.coord_feat_dim + self.cq_query_dim, latent_dim, bias=False,
            )
            self.cq_latent_readout = CrossAttentionBlock(
                dim=latent_dim,
                num_heads=max(1, min(num_heads, 4)),
                ff_mult=max(1, ff_mult // 2),
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
            )
            self.cq_readout_out = nn.Linear(latent_dim, self.cq_query_dim, bias=False)
        else:
            self.cq_latent_readout = CompactLatentReadout(
                query_in_dim=self.coord_feat_dim,
                latent_dim=latent_dim,
                query_dim=self.cq_query_dim,
                rank=self.cq_readout_rank,
                num_heads=self.cq_readout_heads,
                attn_dropout=attn_dropout,
            )

    def _cq_timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Return a deterministic sinusoidal embedding without cacheable time state."""
        t = t.reshape(-1).to(dtype=self.cq_global_proj.weight.dtype)
        half_dim = self.cq_time_embed_dim // 2
        exponent = torch.arange(half_dim, device=t.device, dtype=t.dtype)
        frequencies = torch.exp(
            -math.log(self.cq_time_max_period) * exponent / max(half_dim - 1, 1)
        )
        angles = t.unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if embedding.shape[-1] < self.cq_time_embed_dim:
            embedding = F.pad(embedding, (0, self.cq_time_embed_dim - embedding.shape[-1]))
        return embedding

    def _cq_apply_timestep_film(
        self,
        point_q: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if not self.cq_timestep_film_enabled:
            return point_q
        embedding = self.cq_timestep_mlp(self._cq_timestep_embedding(t))
        scale, shift = self.cq_timestep_film(embedding).chunk(2, dim=-1)
        # Standard residual FiLM. The zero-initialized projection makes this an
        # exact identity at initialization without normalizing every query token.
        return point_q * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def _cq_measurement_support_from_geometry(
        self,
        topk_d2: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_valid: torch.Tensor,
        condition_context: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build [normalized raw measurement, soft support] from existing Top-K geometry."""
        if not self.cq_measurement_support_enabled:
            raise RuntimeError("CQ measurement/support shortcut is disabled.")
        values = batched_gather_3d(condition_context["raw_obs_values"], topk_idx).squeeze(-1)
        field_ids = batched_gather_2d(condition_context["raw_obs_field_ids"], topk_idx)
        sigma = (
            torch.exp(self.log_rbf_sigma).clamp_min(1e-6)
            if self.learnable_rbf_sigma else self.rbf_sigma
        )
        logits = -topk_d2 / (2 * sigma ** 2 + 1e-12)
        weights = torch.softmax(logits.masked_fill(~topk_valid, -1e9), dim=-1)
        weights = weights * topk_valid.to(dtype=weights.dtype)
        # Accumulate directly into the five field slots. This is equivalent to
        # the one-hot [B,Q,K,F] formulation but avoids materializing that large
        # tensor and remains differentiable with respect to the RBF weights.
        output_shape = (*weights.shape[:2], self.n_fields)
        support = weights.new_zeros(output_shape).scatter_add(2, field_ids, weights)
        numerator = weights.new_zeros(output_shape).scatter_add(
            2, field_ids, weights * values,
        )
        measurement = numerator / support.clamp_min(1e-6)
        measurement = torch.where(support > 0, measurement, torch.zeros_like(measurement))
        return torch.cat([measurement, support], dim=-1)

    def _cq_uncached_local_and_raw(
        self,
        query_coords: torch.Tensor,
        condition_context: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run exactly one Top-K search and reuse it for learned and explicit features."""
        k = min(self.gather_topk, condition_context["obs_coords"].shape[1])
        topk_d2, topk_idx, _, _, topk_valid = self._get_topk_neighbors(
            query_coords,
            condition_context["obs_coords"],
            condition_context["refined_sensor_feat"],
            condition_context["obs_mask"],
            k,
            return_features=False,
        )
        raw_features = self._cq_measurement_support_from_geometry(
            topk_d2, topk_idx, topk_valid, condition_context,
        )
        topk_sensor_feat = batched_gather_3d(
            condition_context["refined_sensor_feat"], topk_idx,
        )
        local_cond = self._aggregate_topk_from_geometry(
            topk_d2, topk_idx, topk_valid, condition_context, topk_sensor_feat,
        )
        return local_cond, raw_features

    def _cq_readout(
        self,
        coord_feat: torch.Tensor,
        condition_context: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.cq_readout_mode == "full":
            bsz, n_query, _ = coord_feat.shape
            token = self.cq_query_decoder_token.view(1, 1, -1).expand(
                bsz, n_query, -1,
            )
            query = self.cq_readout_in(torch.cat([coord_feat, token], dim=-1))
            readout = self.cq_latent_readout(
                q=query, kv=condition_context["latents"], kv_padding_mask=None,
            )
            return self.cq_readout_out(readout)
        return self.cq_latent_readout(
            coord_feat,
            projected_kv=(
                condition_context["cq_latent_k"],
                condition_context["cq_latent_v"],
            ),
        )

    def _cq_readout_chunked(
        self,
        coord_feat: torch.Tensor,
        condition_context: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        n_query = int(coord_feat.shape[1])
        chunk_size = self.gather_query_chunk_size
        if chunk_size is None and n_query > 4096:
            chunk_size = 4096
        if chunk_size is None or n_query <= chunk_size:
            return self._cq_readout(coord_feat, condition_context)
        return torch.cat([
            self._cq_readout(coord_feat[:, start:start + chunk_size], condition_context)
            for start in range(0, n_query, chunk_size)
        ], dim=1)

    def _predict_cq_coarse(
        self,
        point_q: torch.Tensor,
        global_q: torch.Tensor,
    ) -> torch.Tensor:
        gamma, beta = self.cq_coarse_film(global_q).chunk(2, dim=-1)
        coarse_feat = (
            point_q * (1.0 + torch.tanh(gamma).unsqueeze(1))
            + beta.unsqueeze(1)
        )
        return self.cq_coarse_head(coarse_feat)

    def prepare_condition_context(
        self,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        context = super().prepare_condition_context(
            obs_coords, obs_values, obs_mask, obs_field_ids,
        )
        context["global_q"] = self.cq_global_proj(context["global_feat"])
        if self.cq_measurement_support_enabled:
            context["raw_obs_values"] = obs_values
            context["raw_obs_field_ids"] = obs_field_ids
        if self.cq_readout_mode == "lowrank":
            keys, values = self.cq_latent_readout.project_latents(context["latents"])
            context["cq_latent_k"] = keys
            context["cq_latent_v"] = values
        return context

    def prepare_query_context(
        self,
        coords: torch.Tensor,
        condition_context: Mapping[str, torch.Tensor],
        cache_level: str = "none",
        chunk_size: Optional[int] = None,
        precomputed_geometry: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if cache_level not in ("none", "geometry", "static_features"):
            raise ValueError("Unknown reconstruction cache level.")
        chunk_size = max(1, int(chunk_size or self.gather_query_chunk_size or 8192))
        context: Dict[str, Any] = {
            "cache_level": cache_level,
            "n_query": int(coords.shape[1]),
            "chunk_size": chunk_size,
        }
        if cache_level == "none":
            if precomputed_geometry is not None:
                raise ValueError("Persistent geometry requires geometry or static_features cache level.")
            return context
        if self.gather_mode not in ("topk_rbf", "topk_rbf_glres"):
            raise ValueError("Query caching supports topk_rbf and topk_rbf_glres.")
        persistent_topk = None
        if precomputed_geometry is not None:
            validate_persistent_topk_geometry_cache(
                precomputed_geometry, self, coords=coords,
                obs_coords=condition_context["obs_coords"],
                obs_mask=condition_context["obs_mask"],
            )
            persistent_topk = cache_tensors(precomputed_geometry)

        coord_feat = coords.new_empty(
            coords.shape[0], coords.shape[1], self.coord_feat_dim,
        )
        for start in range(0, coords.shape[1], chunk_size):
            end = min(start + chunk_size, coords.shape[1])
            coord_feat[:, start:end] = (
                self.pos_enc(coords[:, start:end])
                if self.pos_enc is not None else coords[:, start:end]
            )
        context["coord_feat"] = coord_feat

        if cache_level == "geometry":
            if persistent_topk is not None:
                topk_d2, topk_idx, topk_valid = persistent_topk
                context.update(topk_d2=topk_d2, topk_idx=topk_idx, topk_valid=topk_valid)
                return context
            k = min(self.gather_topk, condition_context["obs_coords"].shape[1])
            topk_d2 = coords.new_empty(coords.shape[0], coords.shape[1], k)
            topk_idx = torch.empty(
                coords.shape[0], coords.shape[1], k,
                dtype=torch.long, device=coords.device,
            )
            topk_valid = torch.empty(
                coords.shape[0], coords.shape[1], k,
                dtype=torch.bool, device=coords.device,
            )
            for start in range(0, coords.shape[1], chunk_size):
                end = min(start + chunk_size, coords.shape[1])
                d2, idx, _, _, valid = self._get_topk_neighbors(
                    coords[:, start:end],
                    condition_context["obs_coords"],
                    condition_context["refined_sensor_feat"],
                    condition_context["obs_mask"],
                    k,
                )
                topk_d2[:, start:end] = d2
                topk_idx[:, start:end] = idx
                topk_valid[:, start:end] = valid
            context.update(topk_d2=topk_d2, topk_idx=topk_idx, topk_valid=topk_valid)
            return context

        if self.training and torch.is_grad_enabled():
            raise ValueError("static_features caching is inference-only.")
        local_cache = coords.new_empty(
            coords.shape[0], coords.shape[1],
            condition_context["refined_sensor_feat"].shape[-1],
        )
        readout_cache = coords.new_empty(
            coords.shape[0], coords.shape[1], self.cq_query_dim,
        )
        raw_cache = (
            coords.new_empty(coords.shape[0], coords.shape[1], 2 * self.n_fields)
            if self.cq_measurement_support_enabled else None
        )
        for start in range(0, coords.shape[1], chunk_size):
            end = min(start + chunk_size, coords.shape[1])
            coords_c = coords[:, start:end]
            empty_feat = coords_c.new_empty(coords_c.shape[0], coords_c.shape[1], 0)
            if persistent_topk is None:
                if self.cq_measurement_support_enabled:
                    local_chunk, raw_chunk = self._cq_uncached_local_and_raw(
                        coords_c, condition_context,
                    )
                    local_cache[:, start:end] = local_chunk
                    raw_cache[:, start:end] = raw_chunk
                else:
                    local_cache[:, start:end] = self._aggregate_chunk(
                        coords_c,
                        empty_feat,
                        condition_context["obs_coords"],
                        condition_context["refined_sensor_feat"],
                        condition_context["obs_mask"],
                        condition_context.get("sensor_importance_bias"),
                    )
            else:
                topk_d2, topk_idx, topk_valid = persistent_topk
                local_cache[:, start:end] = self._aggregate_topk_from_geometry(
                    topk_d2[:, start:end],
                    topk_idx[:, start:end],
                    topk_valid[:, start:end],
                    condition_context,
                )
                if raw_cache is not None:
                    raw_cache[:, start:end] = self._cq_measurement_support_from_geometry(
                        topk_d2[:, start:end],
                        topk_idx[:, start:end],
                        topk_valid[:, start:end],
                        condition_context,
                    )
            readout_cache[:, start:end] = self._cq_readout(
                coord_feat[:, start:end], condition_context,
            )
        context["local_cond"] = local_cache
        context["query_global"] = readout_cache
        if raw_cache is not None:
            context["raw_measurement_support"] = raw_cache
        return context

    def forward_query_chunk(
        self,
        t: torch.Tensor,
        x_t_chunk: torch.Tensor,
        coords_chunk: torch.Tensor,
        condition_context: Mapping[str, torch.Tensor],
        query_context: Optional[Mapping[str, Any]] = None,
        query_slice: Optional[slice] = None,
    ) -> torch.Tensor:
        bsz, n_pts, _ = x_t_chunk.shape
        query_slice = query_slice or slice(0, n_pts)
        cached_coord = None if query_context is None else query_context.get("coord_feat")
        coord_feat = (
            cached_coord[:, query_slice]
            if cached_coord is not None
            else (self.pos_enc(coords_chunk) if self.pos_enc is not None else coords_chunk)
        )
        t_feat = t.view(bsz, 1, 1).expand(bsz, n_pts, 1)
        point_q = self.cq_point_encoder(
            torch.cat([coord_feat, x_t_chunk, t_feat], dim=-1),
        )
        point_q = self._cq_apply_timestep_film(point_q, t)

        cached_readout = None if query_context is None else query_context.get("query_global")
        query_global_q = (
            cached_readout[:, query_slice]
            if cached_readout is not None
            else self._cq_readout_chunked(coord_feat, condition_context)
        )
        cached_local = None if query_context is None else query_context.get("local_cond")
        raw_features = None
        cached_raw = None if query_context is None else query_context.get("raw_measurement_support")
        if cached_local is not None:
            local_cond = cached_local[:, query_slice]
            if cached_raw is not None:
                raw_features = cached_raw[:, query_slice]
        elif query_context is not None and "topk_idx" in query_context:
            local_cond = self._aggregate_topk_from_geometry(
                query_context["topk_d2"][:, query_slice],
                query_context["topk_idx"][:, query_slice],
                query_context["topk_valid"][:, query_slice],
                condition_context,
            )
            if self.cq_measurement_support_enabled:
                raw_features = self._cq_measurement_support_from_geometry(
                    query_context["topk_d2"][:, query_slice],
                    query_context["topk_idx"][:, query_slice],
                    query_context["topk_valid"][:, query_slice],
                    condition_context,
                )
        elif self.cq_measurement_support_enabled:
            local_cond, raw_features = self._cq_uncached_local_and_raw(
                coords_chunk, condition_context,
            )
        else:
            local_cond = self.aggregate_sparse_obs(
                coords_chunk,
                point_q,
                condition_context["obs_coords"],
                condition_context["refined_sensor_feat"],
                condition_context["obs_mask"],
                condition_context.get("sensor_importance_bias"),
            )
        global_q = condition_context["global_q"]
        if self.cq_fusion_mode == "structured_concat":
            global_for_head = (
                global_q.unsqueeze(1)
                + self.cq_readout_scale * query_global_q
            )
            head_input = torch.cat([point_q, global_for_head, local_cond], dim=-1)
        else:
            head_input = (
                point_q
                + self.cq_global_scale * global_q.unsqueeze(1)
                + self.cq_local_scale * self.cq_local_proj(local_cond)
                + self.cq_readout_scale * query_global_q
            )
        if self.cq_measurement_support_enabled:
            if raw_features is None:
                raise RuntimeError("CQ measurement/support features were not constructed.")
            if self.cq_measurement_support_normalize:
                raw_features = self.cq_measurement_support_norm(raw_features)
            head_input = torch.cat([head_input, raw_features], dim=-1)
        residual = self.cq_head(self.cq_fusion_norm(head_input))
        if self.gather_mode == "topk_rbf_glres":
            return self.cq_coarse_scale * self._predict_cq_coarse(point_q, global_q) + residual
        return residual

    def forward(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        condition_context = self.prepare_condition_context(
            obs_coords, obs_values, obs_mask, obs_field_ids,
        )
        return self.forward_query_chunk(t, x_t, coords, condition_context)

    def model_summary(self) -> Dict[str, Any]:
        query_prefixes = ("cq_", "query_to_cond", "gather_gate")
        query_parameters = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith(query_prefixes)
        )
        total_parameters = sum(parameter.numel() for parameter in self.parameters())
        point_input_dim = self.coord_feat_dim + self.n_fields + 1
        raw_feature_dim = 2 * self.n_fields if self.cq_measurement_support_enabled else 0
        fusion_dim = (
            self.cq_query_dim + raw_feature_dim
            if self.cq_fusion_mode == "additive"
            else 2 * self.cq_query_dim + self.cond_dim + raw_feature_dim
        )
        if self.cq_fusion_mode == "additive":
            local_projection_macs = (
                0 if self.cond_dim == self.cq_query_dim
                else self.cond_dim * self.cq_query_dim
            )
            head_and_coarse_macs = (
                3 * self.cq_query_dim ** 2
                + 2 * self.cq_query_dim * self.n_fields
                + raw_feature_dim * self.cq_query_dim
            )
        else:
            local_projection_macs = 0
            head_and_coarse_macs = (
                fusion_dim * self.cq_query_dim
                + 2 * self.cq_query_dim ** 2
                + 2 * self.cq_query_dim * self.n_fields
            )
        common_linear_macs = (
            point_input_dim * self.cq_query_dim
            + 2 * self.cq_query_dim ** 2
            + local_projection_macs
            + head_and_coarse_macs
        )
        if self.cq_readout_mode == "full":
            readout_linear_macs = (
                (self.coord_feat_dim + self.cq_query_dim) * self.latent_dim
                + 2 * self.latent_dim ** 2
                + 2 * self.latent_dim * (
                    self.cq_latent_readout.ff.net[0].out_features
                )
                + self.latent_dim * self.cq_query_dim
            )
            attention_macs = 2 * self.num_latents * self.latent_dim
        else:
            readout_linear_macs = self.coord_feat_dim * self.cq_readout_rank
            attention_macs = self.num_latents * (
                self.cq_readout_rank + self.cq_query_dim
            )
        return {
            "backbone": "GL_rbf_ENH_CQ",
            "total_parameters": total_parameters,
            "condition_core_parameters": total_parameters - query_parameters,
            "query_decoder_parameters": query_parameters,
            "query_dim": self.cq_query_dim,
            "latent_dim": self.latent_dim,
            "cond_dim": self.cond_dim,
            "readout_mode": self.cq_readout_mode,
            "fusion_mode": self.cq_fusion_mode,
            "time_conditioning": self.cq_time_conditioning,
            "time_embed_dim": (
                self.cq_time_embed_dim if self.cq_timestep_film_enabled else None
            ),
            "measurement_support_mode": self.cq_measurement_support_mode,
            "measurement_support_normalize": self.cq_measurement_support_normalize,
            "measurement_support_width": raw_feature_dim,
            "condition_attention_execution": self.condition_attention_execution,
            "sensor_attention_padding_mode": self.sensor_attention_padding_mode,
            "sensor_attention_buckets": list(self.sensor_attention_buckets),
            "readout_rank": (self.cq_readout_rank if self.cq_readout_mode == "lowrank" else None),
            "readout_heads": self.cq_readout_heads,
            "point_state_width": self.cq_query_dim,
            "global_width": self.cq_query_dim,
            "local_width": (
                self.cq_query_dim
                if self.cq_fusion_mode == "additive" else self.cond_dim
            ),
            "legacy_concat_width": 2 * self.hidden_dim + self.cond_dim,
            "cq_fused_width": fusion_dim,
            "theoretical_query_linear_macs_per_query_excluding_attention": (
                common_linear_macs + readout_linear_macs
            ),
            "theoretical_query_attention_macs_per_query": attention_macs,
            "mac_estimate_note": (
                "Linear/attention multiply-accumulates only; excludes activations, "
                "normalization, RBF gather, and condition-static projections."
            ),
        }


# FNO backbone

# Gaussian splatting condition: FNOs truncate the Fourier series to a specific number of low-frequency modes (n_modes_x, n_modes_y), 
# they are inherently low-pass filters. They struggle to resolve sharp, single-pixel spikes. 
# Feeding a grid of sharp spikes into an FNO often causes ringing artifacts (the Gibbs phenomenon) 
# and makes it difficult for the network to understand the spatial influence of that sensor.

# ------------------------------

class PointCloudFFM(nn.Module):
    """
    This block implements 1-Rectified Flow instead of the previous noisy
    Functional Flow Matching bridge.

    Core 1-RF idea: (https://github.com/gnobitab/RectifiedFlow)
        1) Draw a source sample x0 ~ prior
        2) Draw a target sample x1 from data
        3) Interpolate linearly: x_t = (1 - t) * x0 + t * x1
        4) Train the velocity model to predict the constant displacement x1 - x0
    """
    def __init__(self, model: nn.Module, prior: nn.Module, sigma_min: float = 1e-4):
        super().__init__()
        self.model = model
        self.prior = prior

        # Kept only so old checkpoints / YAML files do not break / It is not used in 1-RF.
        self.sigma_min = sigma_min

    def sample_source(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Draw a source sample x0 from the chosen prior on the query coordinates.
        This is the pi_0 endpoint in rectified flow.
        """
        return self.prior(coords, self.model.n_fields)

    @torch.no_grad()
    def prepare_reconstruction_geometry_cache(
        self,
        *,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_mask: torch.Tensor,
        chunk_size: int = 8192,
    ) -> Any:
        """Build reusable Top-K geometry without condition-dependent features."""
        return build_persistent_topk_geometry_cache(
            self.model, coords=coords, obs_coords=obs_coords,
            obs_mask=obs_mask, chunk_size=chunk_size,
        )

    def simulate(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        Straight-line interpolation between source x0 and target x1.

        x_t = (1 - t) * x0 + t * x1
        """
        alpha = t.view(-1, 1, 1)
        # print(f'alpha.shape: {alpha.shape}')
        # print(f'x0.shape: {x0.shape}')
        # print(f'x1.shape: {x1.shape}')
        return (1.0 - alpha) * x0 + alpha * x1

    def target_vector_field(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        1-RF target velocity is the constant straight-line displacement.

        v*(x_t, t) = x1 - x0
        """
        return x1 - x0

    def training_loss(
        self,
        x1: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        obs_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Sample x0 from the source prior for the current query coordinates.
        x0 = self.sample_source(coords)

        # Uniform time for standard 1-RF training.
        bsz = x1.shape[0]
        t = torch.rand(bsz, device=x1.device, dtype=x1.dtype)

        # Straight interpolation and constant target velocity.
        x_t = self.simulate(t, x0, x1)
        target = self.target_vector_field(x0, x1)

        # Predict the velocity under sparse conditioning.
        pred = self.model(t, x_t, coords, obs_coords, obs_values, obs_mask, obs_field_ids)

        # Standard supervised regression loss used in 1-RF.
        loss = F.mse_loss(pred, target)

        return loss, {
            "loss": float(loss.detach().cpu()),
            "target_rms": float(target.pow(2).mean().sqrt().detach().cpu()),
        }

    def prepare_training_bridge(
        self,
        x1: torch.Tensor,
        coords: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Sample one coherent RF stochastic bridge for all effective queries."""
        x0 = self.sample_source(coords)
        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        return {
            "x0": x0,
            "t": t,
            "x_t": self.simulate(t, x0, x1),
            "target": self.target_vector_field(x0, x1),
        }

    def training_loss_microbatched(
        self,
        *,
        x1: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        obs_indices: Optional[torch.Tensor] = None,
        query_microbatch_size: int,
        backward: bool = False,
        reuse_condition_context: bool = True,
        synchronize_timing: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Evaluate one unchanged RF objective with bounded query activations."""
        del obs_indices  # Point-cloud GL-RBF uses coordinates/field IDs directly.
        n_query = int(coords.shape[1])
        chunk_size = max(1, int(query_microbatch_size))
        if chunk_size >= n_query:
            loss, metrics = self.training_loss(
                x1=x1, coords=coords, obs_coords=obs_coords,
                obs_values=obs_values, obs_mask=obs_mask,
                obs_field_ids=obs_field_ids,
            )
            if backward:
                loss.backward()
            metrics.update({
                "rf_bridge_ms": 0.0,
                "condition_context_ms": 0.0,
                "query_chunk_forward_ms": 0.0,
                "query_chunk_backward_ms": 0.0,
                "query_microbatches": 1.0,
            })
            return loss.detach() if backward else loss, metrics

        def sync() -> None:
            if synchronize_timing and x1.device.type == "cuda":
                torch.cuda.synchronize(x1.device)

        sync()
        start = time.perf_counter()
        bridge = self.prepare_training_bridge(x1, coords)
        sync()
        bridge_ms = (time.perf_counter() - start) * 1000.0

        condition_context = None
        sync()
        start = time.perf_counter()
        if reuse_condition_context:
            if not hasattr(self.model, "prepare_condition_context"):
                raise ValueError("Condition-context reuse is unavailable for this backbone.")
            condition_context = self.model.prepare_condition_context(
                obs_coords, obs_values, obs_mask, obs_field_ids,
            )
        sync()
        condition_ms = (time.perf_counter() - start) * 1000.0

        total_elements = int(bridge["target"].numel())
        total_loss = x1.new_zeros(())
        forward_ms = 0.0
        backward_ms = 0.0
        chunks = 0
        for start_index in range(0, n_query, chunk_size):
            end_index = min(start_index + chunk_size, n_query)
            query_slice = slice(start_index, end_index)
            sync()
            start = time.perf_counter()
            if condition_context is not None:
                pred = self.model.forward_query_chunk(
                    t=bridge["t"],
                    x_t_chunk=bridge["x_t"][:, query_slice],
                    coords_chunk=coords[:, query_slice],
                    condition_context=condition_context,
                )
            else:
                pred = self.model(
                    bridge["t"],
                    bridge["x_t"][:, query_slice],
                    coords[:, query_slice],
                    obs_coords,
                    obs_values,
                    obs_mask,
                    obs_field_ids,
                )
            chunk_loss = F.mse_loss(
                pred, bridge["target"][:, query_slice], reduction="sum",
            ) / total_elements
            sync()
            forward_ms += (time.perf_counter() - start) * 1000.0
            if backward:
                sync()
                start = time.perf_counter()
                chunk_loss.backward(
                    retain_graph=condition_context is not None and end_index < n_query,
                )
                sync()
                backward_ms += (time.perf_counter() - start) * 1000.0
                total_loss = total_loss + chunk_loss.detach()
            else:
                total_loss = total_loss + chunk_loss
            chunks += 1
            del pred, chunk_loss

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

    def _sample_cached_streamed(
        self,
        *,
        x: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        clamp_indices: Optional[torch.Tensor],
        ts: torch.Tensor,
        ode_solver: str,
        obs_consistency_mode: str,
        obs_consistency_strength: float,
        obs_consistency_schedule_power: float,
        obs_consistency_final_clamp: bool,
        value_map: Optional[torch.Tensor],
        mask_map: Optional[torch.Tensor],
        reconstruction_query_chunk_size: int,
        reconstruction_cache_level: str,
        reconstruction_geometry_cache: Optional[Any],
    ) -> torch.Tensor:
        if not hasattr(self.model, "prepare_condition_context"):
            raise ValueError(
                "cached_streamed reconstruction requires a backbone with the context API."
            )
        chunk_size = max(1, int(reconstruction_query_chunk_size))
        profile = bool(getattr(self, "_reconstruction_profile_enabled", False))

        def profile_sync() -> None:
            if profile and coords.device.type == "cuda":
                torch.cuda.synchronize(coords.device)

        profile_sync()
        condition_start = time.perf_counter()
        condition_context = self.model.prepare_condition_context(
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
        )
        profile_sync()
        condition_seconds = time.perf_counter() - condition_start
        query_start = time.perf_counter()
        query_context = self.model.prepare_query_context(
            coords=coords,
            condition_context=condition_context,
            cache_level=reconstruction_cache_level,
            chunk_size=chunk_size,
            precomputed_geometry=reconstruction_geometry_cache,
        )
        self._last_reconstruction_condition_bytes = self.model.context_nbytes(condition_context)
        self._last_reconstruction_cache_bytes = self.model.context_nbytes(query_context)
        profile_sync()
        self._last_reconstruction_condition_seconds = condition_seconds
        self._last_reconstruction_query_seconds = time.perf_counter() - query_start
        ode_start = time.perf_counter()
        bsz = coords.shape[0]
        for i in range(len(ts) - 1):
            t0 = ts[i].expand(bsz)
            dt = ts[i + 1] - ts[i]
            for start in range(0, coords.shape[1], chunk_size):
                end = min(start + chunk_size, coords.shape[1])
                query_slice = slice(start, end)
                x_chunk = x[:, query_slice]
                coords_chunk = coords[:, query_slice]
                v0 = self.model.forward_query_chunk(
                    t=t0,
                    x_t_chunk=x_chunk,
                    coords_chunk=coords_chunk,
                    condition_context=condition_context,
                    query_context=query_context,
                    query_slice=query_slice,
                )
                if obs_consistency_mode in ("endpoint", "endpoint_smooth"):
                    v0 = apply_endpoint_observation_consistency(
                        x_t=x_chunk,
                        v=v0,
                        t=t0,
                        value_map=value_map[:, query_slice],
                        mask_map=mask_map[:, query_slice],
                        strength=obs_consistency_strength,
                        schedule_power=obs_consistency_schedule_power,
                    )
                if ode_solver == "heun":
                    x_euler = x_chunk + dt * v0
                    t1 = ts[i + 1].expand(bsz)
                    v1 = self.model.forward_query_chunk(
                        t=t1,
                        x_t_chunk=x_euler,
                        coords_chunk=coords_chunk,
                        condition_context=condition_context,
                        query_context=query_context,
                        query_slice=query_slice,
                    )
                    if (
                        obs_consistency_mode in ("endpoint", "endpoint_smooth")
                        and float(ts[i + 1].item()) < 1.0
                    ):
                        v1 = apply_endpoint_observation_consistency(
                            x_t=x_euler,
                            v=v1,
                            t=t1,
                            value_map=value_map[:, query_slice],
                            mask_map=mask_map[:, query_slice],
                            strength=obs_consistency_strength,
                            schedule_power=obs_consistency_schedule_power,
                        )
                    x[:, query_slice] = x_chunk + 0.5 * dt * (v0 + v1)
                else:
                    x[:, query_slice] = x_chunk + dt * v0

            if obs_consistency_mode == "default_hard" and clamp_indices is not None:
                x = scatter_observed_values(
                    x=x,
                    obs_values=obs_values,
                    obs_mask=obs_mask,
                    obs_indices=clamp_indices,
                    obs_field_ids=obs_field_ids,
                    strength=1.0,
                )

        if (
            obs_consistency_final_clamp
            and obs_consistency_mode != "none"
            and clamp_indices is not None
        ):
            x = scatter_observed_values(
                x=x,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_indices=clamp_indices,
                obs_field_ids=obs_field_ids,
                strength=1.0,
            )
        profile_sync()
        self._last_reconstruction_ode_seconds = time.perf_counter() - ode_start
        return x

    @torch.no_grad()
    def sample(
        self,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        n_steps: int = 8,
        clamp_indices: Optional[torch.Tensor] = None,
        ode_solver: str = "euler",
        obs_consistency_mode: str = "default_hard",
        obs_consistency_strength: float = 1.0,
        obs_consistency_sigma: float = 0.05,
        obs_consistency_schedule_power: float = 2.0,
        obs_consistency_final_clamp: bool = True,
        obs_consistency_chunk_size: int = 8192,
        reconstruction_execution_mode: str = "legacy_full",
        reconstruction_query_chunk_size: int = 8192,
        reconstruction_cache_level: str = "static_features",
        reconstruction_geometry_cache: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Integrate the learned rectified-flow ODE from x0 ~ prior to x1.

        Euler is the default solver because low-step Euler is the main use case
        for 1-RF. Heun is kept as an optional baseline / sanity check.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if reconstruction_execution_mode not in ("legacy_full", "cached_streamed"):
            raise ValueError(
                "reconstruction_execution_mode must be 'legacy_full' or 'cached_streamed'."
            )
        if reconstruction_geometry_cache is not None and reconstruction_execution_mode != "cached_streamed":
            raise ValueError(
                "reconstruction_geometry_cache requires cached_streamed execution."
            )

        bsz = coords.shape[0]
        x = self.sample_source(coords)
        obs_consistency_mode = normalize_obs_consistency_mode(obs_consistency_mode)
        if obs_consistency_mode != "none" and clamp_indices is None:
            if obs_consistency_mode in ("default_hard", "endpoint"):
                raise ValueError(
                    f"obs_consistency_mode={obs_consistency_mode!r} requires clamp_indices."
                )

        value_map = None
        mask_map = None
        if obs_consistency_mode == "endpoint":
            value_map, mask_map = build_pointwise_observation_maps(
                coords=coords,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_indices=clamp_indices,
                obs_field_ids=obs_field_ids,
                n_fields=self.model.n_fields,
            )
        elif obs_consistency_mode == "endpoint_smooth":
            value_map, mask_map = build_smooth_observation_maps(
                coords=coords,
                obs_coords=obs_coords,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_field_ids=obs_field_ids,
                n_fields=self.model.n_fields,
                sigma=obs_consistency_sigma,
                chunk_size=obs_consistency_chunk_size,
            )

        ts = torch.linspace(
            0.0, 1.0, n_steps + 1, device=coords.device, dtype=coords.dtype
        )

        if reconstruction_execution_mode == "cached_streamed":
            return self._sample_cached_streamed(
                x=x,
                coords=coords,
                obs_coords=obs_coords,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_field_ids=obs_field_ids,
                clamp_indices=clamp_indices,
                ts=ts,
                ode_solver=ode_solver,
                obs_consistency_mode=obs_consistency_mode,
                obs_consistency_strength=obs_consistency_strength,
                obs_consistency_schedule_power=obs_consistency_schedule_power,
                obs_consistency_final_clamp=obs_consistency_final_clamp,
                value_map=value_map,
                mask_map=mask_map,
                reconstruction_query_chunk_size=reconstruction_query_chunk_size,
                reconstruction_cache_level=reconstruction_cache_level,
                reconstruction_geometry_cache=reconstruction_geometry_cache,
            )
        self._last_reconstruction_condition_bytes = 0
        self._last_reconstruction_cache_bytes = 0

        for i in range(n_steps):
            t0 = ts[i].expand(bsz)
            dt = ts[i + 1] - ts[i]

            # Velocity at the current state.
            v0 = self.model(t0, x, coords, obs_coords, obs_values, obs_mask, obs_field_ids)
            if obs_consistency_mode in ("endpoint", "endpoint_smooth"):
                # RF clean-endpoint observation masking: guide x1_hat, then
                # convert the consistent endpoint back to a velocity.
                v0 = apply_endpoint_observation_consistency(
                    x_t=x,
                    v=v0,
                    t=t0,
                    value_map=value_map,
                    mask_map=mask_map,
                    strength=obs_consistency_strength,
                    schedule_power=obs_consistency_schedule_power,
                )

            if ode_solver == "heun":
                # Optional predictor-corrector step.
                x_euler = x + dt * v0
                t1 = ts[i + 1].expand(bsz)
                v1 = self.model(t1, x_euler, coords, obs_coords, obs_values, obs_mask, obs_field_ids)
                if obs_consistency_mode in ("endpoint", "endpoint_smooth") and float(ts[i + 1].item()) < 1.0:
                    v1 = apply_endpoint_observation_consistency(
                        x_t=x_euler,
                        v=v1,
                        t=t1,
                        value_map=value_map,
                        mask_map=mask_map,
                        strength=obs_consistency_strength,
                        schedule_power=obs_consistency_schedule_power,
                    )
                x = x + 0.5 * dt * (v0 + v1)
            else:
                # Default 1-RF benchmark solver.
                x = x + dt * v0

            # default_hard preserves the previous per-step pointwise sensor
            # replacement behavior for SenConsis.
            if obs_consistency_mode == "default_hard" and clamp_indices is not None:
                x = scatter_observed_values(
                    x=x,
                    obs_values=obs_values,
                    obs_mask=obs_mask,
                    obs_indices=clamp_indices,
                    obs_field_ids=obs_field_ids,
                    strength=1.0,
                )

        if obs_consistency_final_clamp and obs_consistency_mode != "none" and clamp_indices is not None:
            x = scatter_observed_values(
                x=x,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_indices=clamp_indices,
                obs_field_ids=obs_field_ids,
                strength=1.0,
            )

        return x
