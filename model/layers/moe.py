"""Slow, transparent V0 MoE dispatch with 128 routed and 2 shared experts."""
from __future__ import annotations

import os

from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .context_router import ContextAwareRouter
from .dense_ffn import DenseSwiGLU
from .expert import SwiGLUExpert

try:
    from runpod.kernels.triton_swiglu_moe import (
        triton_moe_dispatch as _triton_moe_dispatch,
        _TRITON_AVAILABLE as _triton_available,
    )
    _triton_moe_step = None
except Exception:
    _triton_moe_dispatch = None
    _triton_moe_step = None
    _triton_available = False



def _moe_dispatch_fused(
    flat: torch.Tensor,
    indices: torch.Tensor,
    probabilities: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Device-side capacity-batched grouped expert dispatch."""
    tokens, hidden = flat.shape
    routes = indices.shape[1]
    experts = w_gate.shape[0]
    device = flat.device
    flat_experts = indices.reshape(-1)
    flat_probabilities = probabilities.reshape(-1)
    token_ids = torch.arange(tokens, device=device).repeat_interleave(routes)

    sorted_experts, sort_order = torch.sort(flat_experts)
    sorted_tokens = token_ids[sort_order]
    sorted_probs = flat_probabilities[sort_order]
    counts = torch.bincount(sorted_experts, minlength=experts)

    total_routes = tokens * routes
    mean_capacity = max(1, (total_routes + experts - 1) // experts)
    capacity = 2 * mean_capacity + 4
    if capacity * experts < total_routes:
        capacity = mean_capacity + 4
    # token_dropping is disabled in the V0 configuration. Preserve every
    # route rather than silently changing the model when one expert is hot.
    capacity = max(capacity, int(counts.max().item()) if counts.numel() else 1)

    offsets = torch.cumsum(counts, 0) - counts
    positions = torch.arange(total_routes, device=device, dtype=torch.long)
    slots = positions - offsets[sorted_experts]
    keep = slots < capacity
    safe_experts = torch.where(keep, sorted_experts, torch.zeros_like(sorted_experts))
    safe_slots = torch.where(keep, slots, torch.zeros_like(slots))
    flat_destinations = safe_experts * capacity + safe_slots

    padded = flat.new_zeros((experts, capacity, hidden))
    routed_input = flat[sorted_tokens] * keep.to(flat.dtype).unsqueeze(-1)
    padded.view(-1, hidden).index_add_(0, flat_destinations, routed_input)

    gate_up_weights = torch.cat([w_gate, w_up], dim=1)
    gate_up = torch.bmm(padded, gate_up_weights.transpose(1, 2))
    expert_hidden = w_gate.shape[1]
    activated = torch.nn.functional.silu(gate_up[..., :expert_hidden]) * gate_up[..., expert_hidden:]
    expert_output = torch.bmm(activated, w_down.transpose(1, 2))

    gathered = expert_output.view(-1, hidden).index_select(0, flat_destinations)
    gathered = gathered * sorted_probs.to(gathered.dtype).unsqueeze(-1)
    output = flat.new_zeros((tokens, hidden))
    output.index_add_(0, sorted_tokens, gathered)
    return output

class SparseMoE(nn.Module):
    """Top-k MoE without capacity dropping.

    The implementation is intentionally correctness-first.  The selected
    token/expert operations are grouped by expert, which is compatible with a
    later fused grouped-GEMM replacement without changing checkpoint names.
    """

    def __init__(self, d_model: int, *, routed_experts: int = 128, shared_experts: int = 2,
                 top_k: int = 8, expert_hidden: int = 640, latent_dim: int = 256,
                 context_aware: bool = True, query_heads: int = 4, query_head_dim: int = 64,
                 local_context: int = 128, aux_balance_coef: float = 0.001,
                 use_hca_global_latents: bool = True) -> None:
        super().__init__()
        self.routed_experts = int(routed_experts)
        self.shared_experts = int(shared_experts)
        self.top_k = int(top_k)
        self.aux_balance_coef = float(aux_balance_coef)
        self.latent_dim = int(latent_dim)
        self.use_hca_global_latents = bool(use_hca_global_latents)
        self.router = ContextAwareRouter(
            d_model, routed_experts, top_k, latent_dim=latent_dim,
            query_heads=query_heads, query_head_dim=query_head_dim,
            local_context=local_context, context_aware=context_aware,
        )
        self.experts = nn.ModuleList([SwiGLUExpert(d_model, expert_hidden) for _ in range(routed_experts)])
        self.shared = nn.ModuleList([SwiGLUExpert(d_model, expert_hidden) for _ in range(shared_experts)])
        self.global_latent_projection = nn.Linear(d_model, latent_dim, bias=False)
        # Routed and shared experts must have separate inference cache entries.
        # A device/dtype-only key aliases the two ModuleLists, causing shared
        # experts to reuse the routed expert weights during sampling.
        self._inference_weight_cache: dict[tuple[int, str, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._inference_gate_up_cache: dict[tuple[int, str, str], torch.Tensor] = {}
        self._last_router_health: dict[str, Any] = {}

    @staticmethod
    def _stack_expert_weights(experts: nn.ModuleList, *, device: torch.device,
                               dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(
            torch.stack([getattr(expert, name).weight for expert in experts]).to(device=device, dtype=dtype)
            for name in ("gate_proj", "up_proj", "down_proj")
        )  # type: ignore[return-value]

    def _weights(self, experts: nn.ModuleList, *, device: torch.device,
                 dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training:
            return self._stack_expert_weights(experts, device=device, dtype=dtype)
        key = (id(experts), str(device), str(dtype))
        cached = self._inference_weight_cache.get(key)
        if cached is None:
            cached = self._stack_expert_weights(experts, device=device, dtype=dtype)
            self._inference_weight_cache[key] = cached
        return cached

    def _gate_up_weights(self, experts: nn.ModuleList, *, device: torch.device,
                         dtype: torch.dtype) -> torch.Tensor:
        """Return cached [experts, 2*hidden, d_model] gate/up weights."""
        if self.training:
            gate, up, _ = self._weights(experts, device=device, dtype=dtype)
            return torch.cat((gate, up), dim=1)
        key = (id(experts), str(device), str(dtype))
        cached = self._inference_gate_up_cache.get(key)
        if cached is None:
            gate, up, _ = self._weights(experts, device=device, dtype=dtype)
            cached = torch.cat((gate, up), dim=1)
            self._inference_gate_up_cache[key] = cached
        return cached

    def train(self, mode: bool = True) -> "SparseMoE":
        super().train(mode)
        if mode:
            self._inference_weight_cache.clear()
            self._inference_gate_up_cache.clear()
        return self

    def _dispatch(self, flat: torch.Tensor, indices: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
        w_gate, w_up, w_down = self._weights(
            self.experts, device=flat.device, dtype=flat.dtype
        )
        if flat.is_cuda and _triton_available and _triton_moe_dispatch is not None:
            return _triton_moe_dispatch(
                flat, indices, probabilities, self.routed_experts,
                w_gate, w_up, w_down,
            )
        return _moe_dispatch_fused(
            flat, indices, probabilities, w_gate, w_up, w_down
        )

    def forward(self, hidden: torch.Tensor, *, hca_global_latents: torch.Tensor | None = None,
                return_stats: bool = True, return_health: bool = False) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if not self.use_hca_global_latents:
            hca_global_latents = None
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        if hca_global_latents is not None and hca_global_latents.shape[-1] != self.latent_dim:
            hca_global_latents = self.global_latent_projection(hca_global_latents)
        indices, probabilities, stats = self.router(
            hidden, hca_global_latents=hca_global_latents,
            return_stats=return_stats, return_health=return_health,
        )
        routed = self._dispatch(flat, indices.reshape(-1, self.top_k), probabilities.reshape(-1, self.top_k))
        if flat.shape[0] <= 128 and self.shared_experts > 0:
            w_gate, w_up, w_down = self._weights(self.shared, device=flat.device, dtype=flat.dtype)
            shared_gate_up = self._gate_up_weights(
                self.shared, device=flat.device, dtype=flat.dtype
            )
            shared_gate_up = torch.einsum("nd,edh->neh", flat, shared_gate_up.transpose(1, 2))
            shared_hidden_size = w_gate.shape[1]
            shared_gate = shared_gate_up[..., :shared_hidden_size]
            shared_up = shared_gate_up[..., shared_hidden_size:]
            shared_hidden = torch.nn.functional.silu(shared_gate) * shared_up
            shared = torch.einsum("neh,edh->ned", shared_hidden, w_down).sum(1)
            shared = shared / self.shared_experts ** 0.5
        else:
            shared = sum(expert(flat) for expert in self.shared) / max(self.shared_experts, 1) ** 0.5
        output = (routed + shared).reshape(shape)
        balance = self.router.balance_loss(indices, probabilities)
        aux = self.aux_balance_coef * balance
        if return_stats or return_health:
            stats.update({"router_balance_loss": float(balance.detach()), "moe_aux_loss": float(aux.detach())})
        return output, aux, stats if (return_stats or return_health) else {}

    @torch.inference_mode()
    def forward_step(self, hidden: torch.Tensor, *, local_history: torch.Tensor | None = None,
                     local_history_valid: torch.Tensor | None = None,
                     hca_global_latents: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], torch.Tensor]:
        if not self.use_hca_global_latents:
            hca_global_latents = None
        """Decode one token with retained router context and fused experts."""
        indices, probabilities, _, latents = self.router.forward_step(
            hidden, local_history=local_history, local_history_valid=local_history_valid,
            hca_global_latents=hca_global_latents,
            return_stats=False,
        )
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        routed = self._dispatch(flat, indices.reshape(-1, self.top_k), probabilities.reshape(-1, self.top_k))
        if self.shared_experts > 0:
            w_gate, w_up, w_down = self._weights(self.shared, device=flat.device, dtype=flat.dtype)
            shared_gate_up = self._gate_up_weights(
                self.shared, device=flat.device, dtype=flat.dtype
            )
            shared_gate_up = torch.einsum("nd,edh->neh", flat, shared_gate_up.transpose(1, 2))
            shared_hidden_size = w_gate.shape[1]
            shared_gate = shared_gate_up[..., :shared_hidden_size]
            shared_up = shared_gate_up[..., shared_hidden_size:]
            shared_hidden = torch.nn.functional.silu(shared_gate) * shared_up
            shared = torch.einsum("neh,edh->ned", shared_hidden, w_down).sum(1)
            shared = shared / self.shared_experts ** 0.5
        else:
            shared = torch.zeros_like(flat)
        return (routed + shared).reshape(shape), hidden.new_zeros((), dtype=torch.float32), {}, latents
