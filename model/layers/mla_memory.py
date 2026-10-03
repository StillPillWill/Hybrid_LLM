"""Compact MLA-style channel memory used by the router and CSA indexer."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLALatentMemory(nn.Module):
    """Project token-local context to a small latent channel.

    Temporal compression is intentionally separate from this channel
    compression.  The module only changes feature rank; callers decide which
    positions are causally visible.
    """

    def __init__(self, d_model: int, latent_dim: int = 256) -> None:
        super().__init__()
        self.down = nn.Linear(d_model, latent_dim, bias=False)
        self.up = nn.Linear(latent_dim, d_model, bias=False)
        self.norm = nn.LayerNorm(latent_dim)

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.norm(self.down(hidden))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.up(latent)

    def forward(self, hidden: torch.Tensor, *, local_context: int = 128) -> torch.Tensor:
        latent = self.encode(hidden)
        W = int(local_context)
        if W <= 1:
            return latent
        # Depthwise causal box filter: one fused convolution replaces the
        # cumulative-sum, gather, subtraction, and division sequence.
        x = latent.transpose(1, 2)
        channels = x.shape[1]
        padded = F.pad(x, (W - 1, 0))
        kernel = torch.ones(
            (channels, 1, W), device=x.device, dtype=x.dtype
        ) / float(W)
        context = F.conv1d(padded, kernel, groups=channels)
        positions = torch.arange(latent.shape[1], device=x.device)
        actual = torch.minimum(
            positions + 1,
            torch.full_like(positions, W),
        ).to(x.dtype)
        context = context * (float(W) / actual).view(1, 1, -1)
        return context.transpose(1, 2)


class LatentAttention(nn.Module):
    """Small multi-head attention over already causal latent memories."""

    def __init__(self, d_model: int, latent_dim: int, heads: int = 4, head_dim: int = 64) -> None:
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        width = self.heads * self.head_dim
        self.q = nn.Linear(d_model, width, bias=False)
        self.k = nn.Linear(latent_dim, width, bias=False)
        self.v = nn.Linear(latent_dim, width, bias=False)
        self.o = nn.Linear(width, d_model, bias=False)

    def forward(self, query: torch.Tensor, latents: torch.Tensor,
                *, causal: bool = True,
                key_valid: torch.Tensor | None = None) -> torch.Tensor:
        if latents.ndim != 3 or latents.shape[1] == 0:
            return torch.zeros_like(query)
        batch, length, _ = query.shape
        q = self.q(query).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(latents).view(batch, latents.shape[1], self.heads, self.head_dim).transpose(1, 2)
        v = self.v(latents).view(batch, latents.shape[1], self.heads, self.head_dim).transpose(1, 2)
        if key_valid is not None and key_valid.shape != (batch, latents.shape[1]):
            raise ValueError("key_valid must have shape [batch, latent_length]")
        if key_valid is None:
            mask = None
        else:
            mask = (~key_valid[:, None, None, :]).to(q.dtype) * torch.finfo(q.dtype).min
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            is_causal=(causal and length == latents.shape[1]),
            dropout_p=0.0,
        )
        out = out.transpose(1, 2).reshape(batch, length, -1)
        return self.o(out)
