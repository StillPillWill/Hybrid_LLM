"""Hierarchical Compressed Attention over dense coarse global summaries."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .compressed_attention import (
    CausalTemporalCompressor, CompressionResult, add_position_offset, update_static_cache,
)


class HierarchicalCompressedAttention(nn.Module):
    def __init__(self, d_model: int, *, compression: int = 32, local_window: int = 128,
                 dense_global_attention: bool = True, global_top_k: int = 32) -> None:
        super().__init__()
        self.local_window = int(local_window)
        self.dense_global_attention = bool(dense_global_attention)
        self.global_top_k = int(global_top_k)
        self.d_model = int(d_model)
        self.use_triton_banded = False
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.local_out = nn.Linear(d_model, d_model, bias=False)
        self.global_out = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.compressor = CausalTemporalCompressor(d_model, compression)

    def forward(self, hidden: torch.Tensor, *, segment_ids: torch.Tensor | None = None,
                memory=None, return_stats: bool = True,
                position: torch.Tensor | int | None = None) -> tuple[torch.Tensor, Any, dict[str, Any]]:
        batch, length, _ = hidden.shape
        q = self.query(hidden)
        local_k = self.key(hidden)
        local_v = self.value(hidden)
        window = max(1, min(self.local_window, length))
        if self.use_triton_banded and segment_ids is None:
            from runpod.kernels.triton_banded_attention import triton_banded_local_attention
            local = triton_banded_local_attention(
                q.unsqueeze(2), local_k.unsqueeze(2), local_v.unsqueeze(2), window_size=window,
            ).squeeze(2)
        else:
            key_windows = F.pad(local_k.transpose(1, 2), (window - 1, 0)).unfold(-1, window, 1)
            key_windows = key_windows.permute(0, 2, 3, 1)
            scores = (q.float().unsqueeze(2) * key_windows.float()).sum(-1) / hidden.shape[-1] ** 0.5
            positions = torch.arange(length, device=hidden.device)
            valid_window = torch.arange(window, device=hidden.device).view(1, -1) >= (
                window - 1 - positions.view(-1, 1)
            ).clamp_min(0)
            scores = scores.masked_fill(
                ~valid_window.view(1, length, window), torch.finfo(scores.dtype).min
            )
            value_windows = F.pad(local_v.transpose(1, 2), (window - 1, 0)).unfold(-1, window, 1)
            value_windows = value_windows.permute(0, 2, 3, 1)
            if segment_ids is not None:
                segment_windows = F.pad(segment_ids, (window - 1, 0), value=-1).unfold(1, window, 1)
                scores = scores.masked_fill(
                    segment_windows != segment_ids[:, :, None], torch.finfo(scores.dtype).min
                )
            local = (scores.softmax(-1).to(local_v.dtype).unsqueeze(-1) * value_windows).sum(2)
        positions = torch.arange(length, device=hidden.device)

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
        global_scores = torch.matmul(q.float(), compressed.keys.float().transpose(-1, -2)) / hidden.shape[-1] ** 0.5
        visible = compressed.positions[:, None, :] <= positions[None, :, None]
        if segment_ids is not None:
            visible &= compressed.segment_ids[:, None, :] == segment_ids[:, :, None]
        global_scores = global_scores.masked_fill(~visible, torch.finfo(global_scores.dtype).min)
        global_values = compressed.values
        if self.dense_global_attention:
            global_weights = global_scores.softmax(-1) * visible.to(global_scores.dtype)
            global_weights = global_weights / global_weights.sum(-1, keepdim=True).clamp_min(1e-9)
            global_context = torch.matmul(global_weights.to(global_values.dtype), global_values)
        else:
            k = min(self.global_top_k, compressed.keys.shape[1])
            top_scores, top_indices = torch.topk(global_scores, k=k, dim=-1)
            selected_visible = torch.gather(visible, -1, top_indices)
            global_weights = top_scores.softmax(-1) * selected_visible.to(top_scores.dtype)
            global_weights = global_weights / global_weights.sum(-1, keepdim=True).clamp_min(1e-9)
            selected_values = torch.gather(
                global_values[:, None, :, :].expand(batch, length, -1, -1),
                2, top_indices.unsqueeze(-1).expand(batch, length, k, global_values.shape[-1]),
            )
            global_context = (
                global_weights.to(global_values.dtype).unsqueeze(-1) * selected_values
            ).sum(2)
        output = self.out(self.local_out(local) + self.global_out(global_context))
        stats = {"hca_global_mass": float(visible.float().mean().detach()),
                 "hca_global_entries": int(compressed.keys.shape[1])} if return_stats else {}
        return output, compressed, stats

    @torch.inference_mode()
    def forward_step(self, hidden: torch.Tensor, *, segment_ids: torch.Tensor | None = None,
                     memory=None, local_history: torch.Tensor | None = None,
                     local_history_valid: torch.Tensor | None = None,
                     position: torch.Tensor | int = 0) -> tuple[torch.Tensor, Any, dict[str, Any]]:
        """Incremental causal attention over one token and retained history."""
        batch, length, _ = hidden.shape
        if length != 1:
            raise ValueError("forward_step expects exactly one token")
        q = self.query(hidden)
        local_k = self.key(hidden)
        local_v = self.value(hidden)
        local_valid = None
        if local_history is not None and local_history.shape[1] > 0:
            local_k = torch.cat((self.key(local_history), local_k), dim=1)
            local_v = torch.cat((self.value(local_history), local_v), dim=1)
            if local_history_valid is not None:
                local_valid = torch.cat((local_history_valid, torch.ones(
                    (batch, 1), device=hidden.device, dtype=torch.bool
                )), dim=1)
        window = min(self.local_window, local_k.shape[1])
        local_k = local_k[:, -window:]
        local_v = local_v[:, -window:]
        if local_valid is not None:
            local_valid = local_valid[:, -window:]
        local_scores = torch.einsum("btd,bsd->bts", q.float(), local_k.float()) / hidden.shape[-1] ** 0.5
        if local_valid is not None:
            local_scores = local_scores.masked_fill(
                ~local_valid[:, None, :], torch.finfo(local_scores.dtype).min
            )
        local = (
            local_scores.softmax(-1).to(local_v.dtype).squeeze(1).unsqueeze(-1) * local_v
        ).sum(1, keepdim=True)

        current = self.compressor.compress_step(
            hidden, position=position, segment_ids=segment_ids, memory=memory,
        )
        offset = torch.as_tensor(position, device=hidden.device, dtype=torch.long).view(batch, 1)
        # ``compress_step`` marks only complete temporal blocks as visible.
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
        global_scores = torch.matmul(q.float(), compressed.keys.float().transpose(-1, -2)) / hidden.shape[-1] ** 0.5
        visible = (compressed.positions[:, None, :] >= 0) & (compressed.positions[:, None, :] <= offset[:, :, None])
        if segment_ids is not None:
            visible &= compressed.segment_ids[:, None, :] == segment_ids[:, :, None]
        global_scores = global_scores.masked_fill(~visible, torch.finfo(global_scores.dtype).min)
        if self.dense_global_attention:
            global_weights = global_scores.softmax(-1) * visible.to(global_scores.dtype)
            global_weights = global_weights / global_weights.sum(-1, keepdim=True).clamp_min(1e-9)
            global_context = torch.matmul(global_weights.to(compressed.values.dtype), compressed.values)
        else:
            k = min(self.global_top_k, compressed.keys.shape[1])
            top_scores, top_indices = torch.topk(global_scores, k=k, dim=-1)
            selected_visible = torch.gather(visible, -1, top_indices)
            global_weights = top_scores.softmax(-1) * selected_visible.to(top_scores.dtype)
            global_weights = global_weights / global_weights.sum(-1, keepdim=True).clamp_min(1e-9)
            selected_values = torch.gather(
                compressed.values[:, None, :, :].expand(batch, 1, -1, -1),
                2, top_indices.unsqueeze(-1).expand(batch, 1, k, compressed.values.shape[-1]),
            )
            global_context = (
                global_weights.to(compressed.values.dtype).unsqueeze(-1) * selected_values
            ).sum(2)
        output = self.out(self.local_out(local) + self.global_out(global_context))
        return output, compressed, {}
