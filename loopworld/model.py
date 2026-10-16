"""Looped DiT-style RGB-D generator (1 pre + 2 shared core + 1 post).

Next loop input is hk (not post(hk)). Shared post/head across exits.
Generator API takes only context, cameras, noise — no scene/renderer handles.

Attention: explicit QKV + per-head RMSNorm on q/k (LightningDiT/SD3 style)
+ fp32 scaled_dot_product_attention. in_proj/out_proj Parameter names match
nn.MultiheadAttention so older checkpoints load with only q_norm/k_norm missing.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbed(nn.Module):
    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t in [0,1]
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0) * 2 * math.pi
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.freq_dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb.to(dtype=next(self.mlp.parameters()).dtype))


class RMSNorm(nn.Module):
    """RMSNorm with learnable gain (init ones). Applied per-head on head_dim."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight.float() * x32).to(dtype)


class Attention(nn.Module):
    """Self-attention with QK-RMSNorm and fp32 SDPA.

    Parameter names for the projection path match nn.MultiheadAttention
    (in_proj_weight/bias, out_proj.*) so pre-QK-norm checkpoints load cleanly.
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        # Match MultiheadAttention layout: packed QKV
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * dim))
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self._reset_parameters()
        # Updated when record_logit=True (detached float for logging)
        self.last_max_logit: float = 0.0

    def _reset_parameters(self):
        # Same init as nn.MultiheadAttention
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
        nn.init.ones_(self.q_norm.weight)
        nn.init.ones_(self.k_norm.weight)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        record_logit: bool = False,
    ) -> torch.Tensor:
        """x: (B, N, C). key_padding_mask: (B, N) True = ignore."""
        B, N, C = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, self.heads, self.head_dim).transpose(1, 2)  # B,H,N,D
        k = k.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # fp32 attention (logits + softmax inside SDPA)
        q_f, k_f, v_f = q.float(), k.float(), v.float()
        # Float additive mask (-inf). Bool SDPA masks disagreed with
        # nn.MultiheadAttention key_padding_mask on this torch build and produced NaNs.
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = torch.zeros(
                key_padding_mask.shape[0], 1, 1, key_padding_mask.shape[1],
                device=x.device, dtype=torch.float32,
            )
            attn_mask = attn_mask.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        if record_logit:
            with torch.no_grad():
                # Chunked max |logit| (matches diagnosis probe); avoid full BxHxNxN
                m = 0.0
                chunk = 256
                for i in range(0, N, chunk):
                    logits = torch.matmul(q_f[:, :, i : i + chunk], k_f.transpose(-2, -1)) * self.scale
                    if attn_mask is not None:
                        logits = logits + attn_mask
                        logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
                    val = logits.abs().amax()
                    if torch.isfinite(val):
                        m = max(m, float(val))
                self.last_max_logit = m

        out = F.scaled_dot_product_attention(
            q_f, k_f, v_f, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        out = out.to(dtype=x.dtype)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return self.out_proj(out)


class AdaLNBlock(nn.Module):
    """Transformer block with adaLN-zero style conditioning."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        record_attn_logit: bool = False,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(t_emb).chunk(6, dim=-1)
        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(h, key_padding_mask=key_padding_mask, record_logit=record_attn_logit)
        x = x + gate_msa.unsqueeze(1) * attn_out
        h2 = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h2)
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim: int, patch_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.proj = nn.Linear(dim, patch_dim)
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(t_emb).chunk(2, dim=-1)
        x = _modulate(self.norm(x), shift, scale)
        return self.proj(x)


class LoopWorldModel(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 4,
        width: int = 256,
        heads: int = 4,
        context_slots: int = 2,
        in_channels: int = 4,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        self.heads = heads
        self.context_slots = context_slots
        self.in_channels = in_channels
        self.grid = image_size // patch_size
        self.n_tokens_per_view = self.grid * self.grid
        self.n_views = context_slots + 1  # context + target
        self.n_tokens = self.n_views * self.n_tokens_per_view
        patch_dim = patch_size * patch_size * in_channels
        # When True, each Attention records last_max_logit during forward
        self.record_attn_logit = False

        self.t_embed = TimestepEmbed(width)
        # patchify: RGB-D (4) + Plücker (6) = 10 channels per pixel -> patch
        self.view_channels = in_channels + 6  # rgbd + plucker
        view_patch_dim = patch_size * patch_size * self.view_channels
        self.patch_embed = nn.Linear(view_patch_dim, width)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, width))
        nn.init.normal_(self.pos_embed, std=0.02)
        # slot embedding to distinguish context vs target
        self.slot_embed = nn.Parameter(torch.zeros(1, self.n_views, 1, width))
        nn.init.normal_(self.slot_embed, std=0.02)

        self.pre = AdaLNBlock(width, heads)
        self.core = nn.ModuleList([AdaLNBlock(width, heads), AdaLNBlock(width, heads)])
        self.post = AdaLNBlock(width, heads)
        self.head = FinalLayer(width, patch_dim)  # predict velocity for target RGB-D only

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B,H,W,C) -> (B, G*G, P*P*C)."""
        B, H, W, C = x.shape
        p = self.patch_size
        g = H // p
        x = x.reshape(B, g, p, g, p, C).permute(0, 1, 3, 2, 4, 5).reshape(B, g * g, p * p * C)
        return x

    def unpatchify(self, tokens: torch.Tensor, C: int) -> torch.Tensor:
        """(B, G*G, P*P*C) -> (B,H,W,C)."""
        B = tokens.shape[0]
        p = self.patch_size
        g = self.grid
        x = tokens.reshape(B, g, g, p, p, C).permute(0, 1, 3, 2, 4, 5).reshape(B, g * p, g * p, C)
        return x

    def _embed_views(
        self,
        target_xt: torch.Tensor,
        target_plucker: torch.Tensor,
        context_rgbd: torch.Tensor,
        context_plucker: torch.Tensor,
        context_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build token sequence and key padding mask.

        target_xt: (B,H,W,4)
        context_rgbd: (B,S,H,W,4)
        context_plucker: (B,S,H,W,6)
        context_valid: (B,S) bool — False means slot missing (mask keys)
        """
        B = target_xt.shape[0]
        S = self.context_slots
        # context tokens
        ctx_feats = []
        for s in range(S):
            feat = torch.cat([context_rgbd[:, s], context_plucker[:, s]], dim=-1)  # B,H,W,10
            ctx_feats.append(self.patchify(feat))
        tgt_feat = torch.cat([target_xt, target_plucker], dim=-1)
        tgt_tok = self.patchify(tgt_feat)
        toks = torch.stack(ctx_feats + [tgt_tok], dim=1)  # B, V, N, PpC
        toks = self.patch_embed(toks)  # B, V, N, D
        toks = toks + self.slot_embed
        toks = toks.reshape(B, self.n_tokens, self.width)
        toks = toks + self.pos_embed

        # padding mask: True = ignore. Missing context slots mask all their tokens.
        # Target always valid. Context tokens still attend among themselves if valid.
        pad = torch.zeros(B, self.n_tokens, device=target_xt.device, dtype=torch.bool)
        n = self.n_tokens_per_view
        for s in range(S):
            invalid = ~context_valid[:, s]  # B
            if invalid.any():
                pad[:, s * n : (s + 1) * n] = invalid.unsqueeze(-1).expand(B, n)
        return toks, pad

    def max_attn_logits(self) -> Dict[str, float]:
        """Return last recorded max logits for pre / core.0 / core.1 / post."""
        return {
            "pre": float(self.pre.attn.last_max_logit),
            "core0": float(self.core[0].attn.last_max_logit),
            "core1": float(self.core[1].attn.last_max_logit),
            "post": float(self.post.attn.last_max_logit),
        }

    def forward_exits(
        self,
        target_xt: torch.Tensor,
        t: torch.Tensor,
        target_plucker: torch.Tensor,
        context_rgbd: torch.Tensor,
        context_plucker: torch.Tensor,
        context_valid: torch.Tensor,
        num_loops: int = 4,
        return_all_exits: bool = True,
    ) -> List[torch.Tensor]:
        """Run loops; return list of velocity predictions (B,H,W,4) for each exit (or last only)."""
        tokens, pad = self._embed_views(target_xt, target_plucker, context_rgbd, context_plucker, context_valid)
        t_emb = self.t_embed(t)
        rec = bool(self.record_attn_logit)
        h = self.pre(tokens, t_emb, key_padding_mask=pad, record_attn_logit=rec)
        exits: List[torch.Tensor] = []
        for k in range(num_loops):
            for block in self.core:
                h = block(h, t_emb, key_padding_mask=pad, record_attn_logit=rec)
            if return_all_exits or k == num_loops - 1:
                z = self.post(h, t_emb, key_padding_mask=pad, record_attn_logit=rec)
                # only target tokens
                n = self.n_tokens_per_view
                tgt = z[:, -n :, :]
                vel_patches = self.head(tgt, t_emb)
                vel = self.unpatchify(vel_patches, self.in_channels)
                exits.append(vel)
            # next loop input is hk (state after core), NOT post(hk)
        return exits

    def forward_velocity(
        self,
        target_xt: torch.Tensor,
        t: torch.Tensor,
        target_plucker: torch.Tensor,
        context_rgbd: torch.Tensor,
        context_plucker: torch.Tensor,
        context_valid: torch.Tensor,
        num_loops: int = 4,
    ) -> torch.Tensor:
        """Single final-exit velocity (for sampling)."""
        return self.forward_exits(
            target_xt, t, target_plucker, context_rgbd, context_plucker, context_valid,
            num_loops=num_loops, return_all_exits=False,
        )[-1]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model_from_config(cfg: dict) -> LoopWorldModel:
    return LoopWorldModel(
        image_size=int(cfg.get("image_size", 64)),
        patch_size=int(cfg.get("patch_size", 4)),
        width=int(cfg.get("width", 256)),
        heads=int(cfg.get("heads", 4)),
        context_slots=int(cfg.get("context_slots", 2)),
    )
