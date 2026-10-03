"""Compressed attention residuals between variable-depth recurrent loops."""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers.mla_memory import LatentAttention, MLALatentMemory
from .recurrent_state import RecurrentState


class LoopMemoryHook(nn.Module):
    def __init__(
        self,
        d_model: int,
        latent_dim: int,
        *,
        enabled: bool = False,
        heads: int = 4,
        head_dim: int = 32,
        residual_gate_init: float = 0.05,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.compressor = MLALatentMemory(d_model, latent_dim)
        self.query_norm = nn.LayerNorm(d_model)
        self.attention = LatentAttention(
            d_model, latent_dim, heads=heads, head_dim=head_dim
        )
        gate = torch.tensor(float(residual_gate_init)).clamp(1e-5, 1 - 1e-5)
        self.gating_logit = nn.Parameter(torch.logit(gate))

    @property
    def gating_factor(self) -> torch.Tensor:
        return torch.sigmoid(self.gating_logit)

    def encode(self, streams: torch.Tensor) -> torch.Tensor:
        """Compress one loop endpoint, preserving gradients for loop AttnRes."""
        return self.compressor.encode(streams.mean(dim=-2))

    def inject(self, streams: torch.Tensor, endpoints: list[torch.Tensor]) -> torch.Tensor:
        """Attend over prior loop endpoints at each token position."""
        if not self.enabled or not endpoints:
            return streams
        batch, length, stream_count, d_model = streams.shape
        memory = torch.stack(endpoints, dim=2)  # [B, T, prior_loops, latent]
        query = self.query_norm(streams.mean(dim=-2)).reshape(batch * length, 1, d_model)
        memory = memory.reshape(batch * length, len(endpoints), memory.shape[-1])
        attended = self.attention(query, memory, causal=False)
        attended = attended.reshape(batch, length, 1, d_model)
        return streams + self.gating_factor.to(dtype=streams.dtype) * attended.expand(
            -1, -1, stream_count, -1
        )

    def record(
        self,
        streams: torch.Tensor,
        state: RecurrentState,
        *,
        force: bool = False,
    ) -> torch.Tensor | None:
        if self.enabled or force:
            endpoint = self.encode(streams)
            # Keep the serialized/telemetry state detached, while returning the
            # live tensor so the next loop retains a gradient path.
            state.loop_endpoints.append(endpoint.detach())
            return endpoint
        return None
