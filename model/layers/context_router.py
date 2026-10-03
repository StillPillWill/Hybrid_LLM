"""Context-aware top-k expert router with a hidden-only fallback."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .mla_memory import LatentAttention, MLALatentMemory
from .norms import RMSNorm


class ContextAwareRouter(nn.Module):
    def __init__(self, d_model: int, experts: int, top_k: int, *, latent_dim: int = 256,
                 query_heads: int = 4, query_head_dim: int = 64, local_context: int = 128,
                 context_aware: bool = True) -> None:
        super().__init__()
        self.experts = int(experts)
        self.top_k = int(top_k)
        self.context_aware = bool(context_aware)
        self.local_context = int(local_context)
        self.norm = RMSNorm(d_model)
        self.latent_memory = MLALatentMemory(d_model, latent_dim)
        self.latent_attention = LatentAttention(d_model, latent_dim, query_heads, query_head_dim)
        self.local_context_out = nn.Linear(latent_dim, d_model, bias=False)
        self.route_gate = nn.Parameter(torch.tensor(-4.59512))
        self.context_out = nn.Linear(d_model, d_model, bias=False)
        self.logits = nn.Linear(d_model, experts, bias=False)

    @staticmethod
    def _health_stats(top_indices: torch.Tensor, probabilities: torch.Tensor,
                      expert_count: int) -> dict[str, Any]:
        """Return scalar routing health metrics without retaining route tensors."""
        with torch.no_grad():
            flat_indices = top_indices.reshape(-1).long()
            flat_probabilities = probabilities.reshape(-1).float()
            counts = torch.bincount(flat_indices, minlength=int(expert_count)).float()
            distribution = counts / max(flat_indices.numel(), 1)
            load_entropy = -(distribution * distribution.clamp_min(1e-9).log()).sum()
            route_entropy = -(
                probabilities.float() * probabilities.float().clamp_min(1e-9).log()
            ).sum(-1).mean()
            active = (counts > 0).sum().to(torch.float32)
            dead = float(expert_count) - active
            max_frac = distribution.max()
            mean_prob = flat_probabilities.mean()
            values = torch.stack([
                route_entropy, active, dead, load_entropy, max_frac, mean_prob
            ]).tolist()
            return {
                "route_entropy": float(values[0]),
                "active_experts": int(values[1]),
                "dead_experts": int(values[2]),
                "expert_load_entropy": float(values[3]),
                "max_expert_fraction": float(values[4]),
                "mean_selected_probability": float(values[5]),
            }

    def forward(self, hidden: torch.Tensor, *, hca_global_latents: torch.Tensor | None = None,
                return_context: bool = False, return_stats: bool = True,
                return_health: bool = False) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        local = self.norm(hidden)
        if self.context_aware:
            local_latents = self.latent_memory(local, local_context=self.local_context)
            context = self.local_context_out(local_latents)
            if hca_global_latents is not None and hca_global_latents.shape[1] > 0:
                global_context = self.latent_attention(local, hca_global_latents, causal=False)
                context = context + global_context
            route_input = local + torch.sigmoid(self.route_gate) * self.context_out(context)
        else:
            route_input = local
        route_logits = self.logits(route_input)
        top_values, top_indices = torch.topk(route_logits.float(), self.top_k, dim=-1)
        probabilities = top_values.softmax(dim=-1)
        stats: dict[str, Any] = {}
        if return_stats or return_health:
            stats.update(self._health_stats(top_indices, probabilities, self.experts))
        if return_stats:
            stats.update({
                "route_logits": route_logits,
                "topk_indices": top_indices,
                "topk_probabilities": probabilities,
            })
        if return_context and self.context_aware:
            stats["context"] = context
        return top_indices, probabilities, stats

    @torch.inference_mode()
    def forward_step(self, hidden: torch.Tensor, *, local_history: torch.Tensor | None = None,
                     local_history_valid: torch.Tensor | None = None,
                     hca_global_latents: torch.Tensor | None = None,
                     return_stats: bool = False) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], torch.Tensor]:
        """Route one token using the same rolling-context rule as ``forward``.

        For short prefixes the training path uses causal latent attention. Once
        the history exceeds ``local_context`` it uses the rolling mean followed
        by ``local_context_out``.  The previous decode path always used latent
        attention, so prefill and generation were routing through different
        functions.
        """
        local = self.norm(hidden)
        current_latent = self.latent_memory.encode(local)
        if local_history is not None and local_history.shape[1] > 0:
            full_latents = torch.cat((local_history, current_latent), dim=1)
            if local_history_valid is not None:
                full_valid = torch.cat((local_history_valid, torch.ones(
                    (hidden.shape[0], 1), device=hidden.device, dtype=torch.bool
                )), dim=1)
            else:
                full_valid = torch.ones(
                    (hidden.shape[0], full_latents.shape[1]), device=hidden.device, dtype=torch.bool
                )
            latents = full_latents[:, -self.local_context:]
            valid = full_valid[:, -self.local_context:]
        else:
            latents = current_latent
            valid = torch.ones(
                (hidden.shape[0], 1), device=hidden.device, dtype=torch.bool
            )
        if self.context_aware:
            denominator = valid.sum(dim=1, keepdim=True).clamp_min(1).to(latents.dtype)
            context_latent = (
                latents * valid.unsqueeze(-1).to(latents.dtype)
            ).sum(1, keepdim=True) / denominator.unsqueeze(-1)
            context = self.local_context_out(context_latent)
            if hca_global_latents is not None and hca_global_latents.shape[1] > 0:
                context = context + self.latent_attention(local, hca_global_latents, causal=False)
            route_input = local + torch.sigmoid(self.route_gate) * self.context_out(context)
        else:
            route_input = local
        route_logits = self.logits(route_input)
        top_values, top_indices = torch.topk(route_logits.float(), self.top_k, dim=-1)
        probabilities = top_values.softmax(dim=-1)
        stats: dict[str, Any] = {}
        if return_stats:
            stats = {
                "route_entropy": float((-(probabilities * probabilities.clamp_min(1e-9).log()).sum(-1).mean()).detach()),
                "route_logits": route_logits,
                "topk_indices": top_indices,
                "topk_probabilities": probabilities,
            }
        return top_indices, probabilities, stats, latents

    def balance_loss(self, topk_indices: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
        # Switch-style importance/load balancing without token dropping.
        flat_indices = topk_indices.reshape(-1)
        counts = torch.bincount(flat_indices, minlength=self.experts).float()
        load = counts / max(flat_indices.numel(), 1)
        importance = torch.zeros(self.experts, device=probabilities.device, dtype=probabilities.dtype)
        importance.scatter_add_(0, flat_indices, probabilities.reshape(-1).to(importance.dtype))
        importance = importance / max(flat_indices.numel(), 1)
        return self.experts * (load * importance).sum()
