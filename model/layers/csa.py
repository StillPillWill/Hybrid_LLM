"""Causal Selective Attention: local raw branch plus top-k remote summaries."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .compressed_attention import (
    CausalTemporalCompressor, CompressionResult, add_position_offset, update_static_cache,
)


class CompressedSelectiveAttention(nn.Module):
    def __init__(self, d_model: int, *, compression: int = 4, local_window: int = 128,
                 index_heads: int = 16, index_head_dim: int = 128, top_k: int = 32) -> None:
        super().__init__()
        self.local_window = int(local_window)
        self.top_k = int(top_k)
        width = int(index_heads * index_head_dim)
        self.query = nn.Linear(d_model, width, bias=False)
        self.key = nn.Linear(d_model, width, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.local_out = nn.Linear(d_model, d_model, bias=False)
        self.remote_out = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.compressor = CausalTemporalCompressor(d_model, compression, latent_dim=width)
        self.index_heads = int(index_heads)
        self.index_head_dim = int(index_head_dim)
        self.d_model = int(d_model)
        self.use_triton_banded = False

    def forward(self, hidden: torch.Tensor, *, segment_ids: torch.Tensor | None = None,
                memory=None, return_stats: bool = True,
                position: torch.Tensor | int | None = None) -> tuple[torch.Tensor, Any, dict[str, Any]]:
        batch, length, _ = hidden.shape
        q = self.query(hidden).view(batch, length, self.index_heads, self.index_head_dim)
        local_k = self.key(hidden).view(batch, length, self.index_heads, self.index_head_dim)
        local_v = self.value(hidden)
        window = max(1, min(self.local_window, length))
        positions = torch.arange(length, device=hidden.device)
        if self.use_triton_banded and segment_ids is None:
            # The custom kernel handles matching q/k/v head widths.  For the
            # current multi-head-index/shared-value CSA shape, its SDPA
            # fallback broadcasts the shared value stream across heads and
            # preserves the original mean-over-heads math.
            from runpod.kernels.triton_banded_attention import triton_banded_local_attention
            local_heads = triton_banded_local_attention(
                q, local_k, local_v.unsqueeze(2), window_size=window,
            )
            local_result = local_heads.mean(2)
        else:
            key_sequence = local_k.permute(0, 2, 3, 1)
            key_windows = F.pad(key_sequence, (window - 1, 0)).unfold(-1, window, 1)
            key_windows = key_windows.permute(0, 3, 1, 4, 2)
            local_scores = torch.einsum("bthd,bthwd->bthw", q, key_windows) / self.index_head_dim ** 0.5
            positions = torch.arange(length, device=hidden.device)
            valid_window = torch.arange(window, device=hidden.device).view(1, -1) >= (
                window - 1 - positions.view(-1, 1)
            ).clamp_min(0)
            local_scores = local_scores.masked_fill(
                ~valid_window.view(1, length, 1, window), torch.finfo(local_scores.dtype).min
            )
            value_windows = F.pad(local_v.permute(0, 2, 1), (window - 1, 0)).unfold(-1, window, 1)
            value_windows = value_windows.permute(0, 2, 3, 1)
            if segment_ids is not None:
                segment_windows = F.pad(segment_ids, (window - 1, 0), value=-1).unfold(1, window, 1)
                local_scores = local_scores.masked_fill(
                    segment_windows[:, :, None, :] != segment_ids[:, :, None, None],
                    torch.finfo(local_scores.dtype).min,
                )
            local_weights = local_scores.softmax(-1).to(local_v.dtype)
            local_result = torch.einsum("btw,btwd->btd", local_weights.mean(2), value_windows)

        current = add_position_offset(self.compressor.compress(hidden, segment_ids), position)
        if memory is not None and memory.keys is not None and memory.capacity > 0 and memory.static_cache:
            compressed = update_static_cache(memory, current)
        elif memory is not None and memory.keys is not None:
            compressed = CompressionResult(
                keys=torch.cat([memory.keys, current.keys], dim=1),
                values=torch.cat([memory.values, current.values], dim=1),
                positions=torch.cat([memory.positions, current.positions], dim=1),
                segment_ids=torch.cat([memory.segment_ids, current.segment_ids], dim=1),
            )
        else:
            compressed = current
        remote_scores = torch.einsum(
            "bthd,bshd->bhts",
            q,
            compressed.keys.view(batch, -1, self.index_heads, self.index_head_dim),
        ) / self.index_head_dim ** 0.5
        visible = compressed.positions[:, None, :] <= positions[None, :, None]
        if segment_ids is not None:
            visible &= compressed.segment_ids[:, None, :] == segment_ids[:, :, None]
        remote_visible = visible[:, None, :, :]
        remote_scores = remote_scores.masked_fill(~remote_visible, torch.finfo(remote_scores.dtype).min)
        k = min(self.top_k, compressed.keys.shape[1])
        top_scores, top_indices = torch.topk(remote_scores, k=k, dim=-1)
        selected_visible = torch.gather(
            remote_visible.expand_as(remote_scores), -1, top_indices
        )
        remote_weights = top_scores.softmax(-1) * selected_visible.to(top_scores.dtype)
        remote_weights = remote_weights / remote_weights.sum(-1, keepdim=True).clamp_min(1e-9)
        def _weighted_remote(values, indices_chunk, weights_chunk):
            source_length, value_dim = values.shape[1], values.shape[2]
            offsets = (torch.arange(batch, device=values.device) * source_length).view(batch, 1, 1, 1)
            flat_indices = (indices_chunk + offsets).reshape(-1)
            selected = values.reshape(batch * source_length, value_dim).index_select(0, flat_indices)
            selected = selected.view(batch, self.index_heads, length, indices_chunk.shape[-1], value_dim)
            return (weights_chunk.unsqueeze(-1) * selected).sum(3)

        remote_accum = torch.zeros(
            batch, length, compressed.values.shape[-1],
            device=compressed.values.device, dtype=local_result.dtype,
        )
        # Keep the selected-value workspace below the 24-GiB card's remaining
        # activation headroom.  This changes only the accumulation tiling; the
        # selected experts/keys and the resulting attention are unchanged.
        for start in range(0, k, 2):
            end = min(start + 2, k)
            remote_chunk = checkpoint(
                _weighted_remote,
                compressed.values,
                top_indices[..., start:end],
                remote_weights[..., start:end],
                use_reentrant=False,
            )
            remote_accum = remote_accum + remote_chunk.sum(1).to(remote_accum.dtype)
        remote_result = (remote_accum / self.index_heads).to(dtype=local_result.dtype)
        output = self.out(self.local_out(local_result) + self.remote_out(remote_result))
        if return_stats:
            selected_positions = torch.gather(
                compressed.positions[:, None, None, :].expand(batch, self.index_heads, length, -1),
                3, top_indices,
            )
            local_boundary = positions.view(1, 1, length, 1) - self.local_window + 1
            remote_mass = (
                remote_weights * (selected_positions < local_boundary).to(remote_weights.dtype)
            ).sum(-1)
            rm = remote_mass.mean().detach()
            ent = (
                -(remote_weights * remote_weights.clamp_min(1e-9).log())
                .sum(-1).mean()
            ).detach()
            stats = {
                "csa_remote_fraction": float(rm),
                "csa_remote_mass": float(rm),
                "csa_selection_entropy": float(ent),
                "csa_topk": k,
            }
        else:
            stats = {}
        return output, compressed, stats

    @torch.inference_mode()
    def forward_step(self, hidden: torch.Tensor, *, segment_ids: torch.Tensor | None = None,
                     memory=None, local_history: torch.Tensor | None = None,
                     local_history_valid: torch.Tensor | None = None,
                     position: torch.Tensor | int = 0) -> tuple[torch.Tensor, Any, dict[str, Any]]:
        """Run the causal attention for one new token.

        ``memory`` contains compressed history and ``local_history`` contains
        the raw local window. Keeping those separately preserves the causal
        result while reducing the work from the full prompt length to one
        token plus the configured local window.
        """
        batch, length, _ = hidden.shape
        if length != 1:
            raise ValueError("forward_step expects exactly one token")
        q = self.query(hidden).view(batch, 1, self.index_heads, self.index_head_dim)
        local_k = self.key(hidden).view(batch, 1, self.index_heads, self.index_head_dim)
        local_v = self.value(hidden)
        local_valid = None
        if local_history is not None and local_history.shape[1] > 0:
            history_k = self.key(local_history).view(batch, -1, self.index_heads, self.index_head_dim)
            history_v = self.value(local_history)
            local_k = torch.cat((history_k, local_k), dim=1)
            local_v = torch.cat((history_v, local_v), dim=1)
            if local_history_valid is not None:
                local_valid = torch.cat((local_history_valid, torch.ones(
                    (batch, 1), device=hidden.device, dtype=torch.bool
                )), dim=1)
        window = min(self.local_window, local_k.shape[1])
        local_k = local_k[:, -window:]
        local_v = local_v[:, -window:]
        if local_valid is not None:
            local_valid = local_valid[:, -window:]
        local_scores = torch.einsum("bthd,bshd->bhts", q.float(), local_k.float()) / self.index_head_dim ** 0.5
        if local_valid is not None:
            local_scores = local_scores.masked_fill(
                ~local_valid[:, None, None, :], torch.finfo(local_scores.dtype).min
            )
        if segment_ids is not None:
            current_segment = segment_ids[:, -1:]
            history_segments = torch.zeros((batch, local_v.shape[1] - 1), device=hidden.device, dtype=segment_ids.dtype)
            segment_mask = torch.cat((history_segments, current_segment), dim=1)
            local_scores = local_scores.masked_fill(
                segment_mask[:, None, None, :] != current_segment[:, None, :, None],
                torch.finfo(local_scores.dtype).min,
            )
        local_weights = local_scores.softmax(-1).to(local_v.dtype)
        local_result = (
            local_weights.mean(1).squeeze(1).unsqueeze(-1) * local_v
        ).sum(1, keepdim=True)

        current = self.compressor.compress_step(
            hidden, position=position, segment_ids=segment_ids, memory=memory,
        )
        offset = torch.as_tensor(position, device=hidden.device, dtype=torch.long).view(batch, 1)
        # ``compress_step`` already marks incomplete temporal blocks as
        # invisible and complete blocks at their endpoint.
        if memory is not None and memory.keys is not None:
            prior_length = memory.keys.shape[1] if memory.length is None else int(memory.length)
            new_length = prior_length + current.keys.shape[1]
            if memory.capacity >= new_length:
                memory.keys[:, prior_length:new_length] = current.keys
                memory.values[:, prior_length:new_length] = current.values
                memory.positions[:, prior_length:new_length] = current.positions
                memory.segment_ids[:, prior_length:new_length] = current.segment_ids
                memory.length = new_length
                compressed_keys = memory.keys[:, :new_length]
                compressed_values = memory.values[:, :new_length]
                compressed_positions = memory.positions[:, :new_length]
                compressed_segments = memory.segment_ids[:, :new_length]
            else:
                compressed_keys = torch.cat((memory.keys[:, :prior_length], current.keys), dim=1)
                compressed_values = torch.cat((memory.values[:, :prior_length], current.values), dim=1)
                compressed_positions = torch.cat((memory.positions[:, :prior_length], current.positions), dim=1)
                compressed_segments = torch.cat((memory.segment_ids[:, :prior_length], current.segment_ids), dim=1)
        else:
            compressed_keys, compressed_values = current.keys, current.values
            compressed_positions, compressed_segments = current.positions, current.segment_ids
        compressed = CompressionResult(
            compressed_keys, compressed_values, compressed_positions, compressed_segments,
            current.pending_sum, current.pending_count, current.pending_segment_ids,
        )
        remote_scores = torch.einsum(
            "bthd,bshd->bhts", q.float(), compressed.keys.float().view(batch, -1, self.index_heads, self.index_head_dim)
        ) / self.index_head_dim ** 0.5
        visible = (compressed.positions[:, None, :] >= 0) & (compressed.positions[:, None, :] <= offset[:, :, None])
        if segment_ids is not None:
            visible &= compressed.segment_ids[:, None, :] == segment_ids[:, :, None]
        remote_visible = visible[:, None, :, :]
        remote_scores = remote_scores.masked_fill(~remote_visible, torch.finfo(remote_scores.dtype).min)
        k = min(self.top_k, compressed.keys.shape[1])
        top_scores, top_indices = torch.topk(remote_scores, k=k, dim=-1)
        selected_visible = torch.gather(
            remote_visible.expand_as(remote_scores), -1, top_indices
        )
        remote_weights = top_scores.softmax(-1) * selected_visible.to(top_scores.dtype)
        remote_weights = remote_weights / remote_weights.sum(-1, keepdim=True).clamp_min(1e-9)
        expanded_values = compressed.values[:, None, None, :, :].expand(batch, self.index_heads, 1, -1, -1)
        gather_index = top_indices.unsqueeze(-1).expand(*top_indices.shape, compressed.values.shape[-1])
        selected_values = torch.gather(expanded_values, 3, gather_index)
        remote_result = (remote_weights.unsqueeze(-1) * selected_values).sum(3).mean(1).to(local_result.dtype)
        output = self.out(self.local_out(local_result) + self.remote_out(remote_result))
        return output, compressed, {}
