"""Shared causal temporal-compression primitives for CSA and HCA."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..recurrent_state import MemoryState


def _contiguous_segments(segment_ids: torch.Tensor | None, length: int) -> list[tuple[int, int, int]]:
    if segment_ids is None:
        return [(0, length, 0)]
    values = segment_ids.detach().cpu().tolist()
    result: list[tuple[int, int, int]] = []
    start = 0
    current = int(values[0]) if values else 0
    for index, value in enumerate(values[1:], 1):
        if int(value) != current:
            result.append((start, index, current))
            start, current = index, int(value)
    if length:
        result.append((start, length, current))
    return result


@dataclass
class CompressionResult:
    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor
    segment_ids: torch.Tensor
    pending_sum: torch.Tensor | None = None
    pending_count: torch.Tensor | None = None
    pending_segment_ids: torch.Tensor | None = None


def update_static_cache(memory: MemoryState, current: CompressionResult) -> CompressionResult:
    """Write valid compressed entries into a preallocated ring cache.

    Static caches are used by decode/graph paths.  The old implementation
    tried to write one entry using an undefined ``position`` variable and then
    forgot to construct the ``CompressionResult`` consumed by attention.
    """
    if memory.keys is None or memory.capacity <= 0:
        return current
    with torch.no_grad():
        capacity = int(memory.capacity)
        for batch_index in range(current.keys.shape[0]):
            valid = current.positions[batch_index] >= 0
            if not bool(valid.any()):
                continue
            slots = current.positions[batch_index, valid].remainder(capacity)
            memory.keys[batch_index].index_copy_(0, slots, current.keys[batch_index, valid])
            memory.values[batch_index].index_copy_(0, slots, current.values[batch_index, valid])
            memory.positions[batch_index].index_copy_(0, slots, current.positions[batch_index, valid])
            memory.segment_ids[batch_index].index_copy_(0, slots, current.segment_ids[batch_index, valid])
        memory.length = capacity
    return CompressionResult(memory.keys, memory.values, memory.positions, memory.segment_ids)


def add_position_offset(current: CompressionResult, position: torch.Tensor | int | None) -> CompressionResult:
    """Apply a scalar/per-row absolute position offset when supplied."""
    if position is None:
        return current
    offset = torch.as_tensor(position, device=current.positions.device, dtype=torch.long)
    if offset.ndim == 0:
        positions = current.positions + offset
    elif offset.ndim == 1 and offset.shape[0] == current.positions.shape[0]:
        positions = current.positions + offset[:, None]
    else:
        # A full [time] position vector already describes the sequence and is
        # not an offset to apply to compressor-produced absolute positions.
        positions = current.positions
    return CompressionResult(
        current.keys, current.values, positions, current.segment_ids,
        current.pending_sum, current.pending_count, current.pending_segment_ids,
    )


class CausalTemporalCompressor(nn.Module):
    def __init__(self, d_model: int, ratio: int, latent_dim: int | None = None) -> None:
        super().__init__()
        self.ratio = int(ratio)
        width = int(latent_dim or d_model)
        self.key = nn.Linear(d_model, width, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)

    def compress(self, hidden: torch.Tensor, segment_ids: torch.Tensor | None = None) -> CompressionResult:
        batch, length, _ = hidden.shape
        row_keys: list[torch.Tensor] = []
        row_values: list[torch.Tensor] = []
        row_positions: list[torch.Tensor] = []
        row_segments: list[torch.Tensor] = []
        pending_sums: list[torch.Tensor] = []
        pending_counts: list[int] = []
        pending_segments: list[int] = []
        for batch_index in range(batch):
            row_keys_i: list[torch.Tensor] = []
            row_values_i: list[torch.Tensor] = []
            row_positions_i: list[int] = []
            row_segments_i: list[int] = []
            row_segments_input = segment_ids[batch_index] if segment_ids is not None else None
            for start, end, segment in _contiguous_segments(row_segments_input, length):
                pending_sum_i = hidden.new_zeros(hidden.shape[-1])
                pending_count_i = 0
                pending_segment_i = int(segment)
                for block_start in range(start, end, self.ratio):
                    block_end = min(block_start + self.ratio, end)
                    token_slice = hidden[batch_index:batch_index + 1, block_start:block_end]
                    block_count = block_end - block_start
                    if block_count < self.ratio:
                        pending_sum_i = token_slice.sum(dim=1).squeeze(0)
                        pending_count_i = block_count
                        pending_segment_i = int(segment)
                        continue
                    # The endpoint is the first position at which this summary
                    # is visible, so no summary contains a future token.
                    row_keys_i.append(self.key(token_slice.mean(dim=1)).squeeze(0))
                    row_values_i.append(self.value(token_slice.mean(dim=1)).squeeze(0))
                    row_positions_i.append(block_end - 1)
                    row_segments_i.append(segment)
                # A partial block from an earlier segment cannot continue over
                # a document boundary. Only the final segment's pending block
                # is carried into the next decode call.
            if row_keys_i:
                row_keys.append(torch.stack(row_keys_i, dim=0))
                row_values.append(torch.stack(row_values_i, dim=0))
            else:
                row_keys.append(hidden.new_zeros((0, self.key.out_features)))
                row_values.append(hidden.new_zeros((0, self.value.out_features)))
            row_positions.append(torch.tensor(row_positions_i, device=hidden.device, dtype=torch.long))
            row_segments.append(torch.tensor(row_segments_i, device=hidden.device, dtype=torch.long))
            pending_sums.append(pending_sum_i)
            pending_counts.append(pending_count_i)
            pending_segments.append(pending_segment_i)
        max_entries = max(1, max(row.shape[0] for row in row_keys))
        width = row_keys[0].shape[-1]
        model_width = row_values[0].shape[-1]
        keys = hidden.new_zeros((batch, max_entries, width))
        values = hidden.new_zeros((batch, max_entries, model_width))
        positions = torch.full((batch, max_entries), -1, device=hidden.device, dtype=torch.long)
        segments = torch.full((batch, max_entries), -1, device=hidden.device, dtype=torch.long)
        for batch_index, (keys_i, values_i) in enumerate(zip(row_keys, row_values)):
            count = keys_i.shape[0]
            keys[batch_index, :count] = keys_i
            values[batch_index, :count] = values_i
            positions[batch_index, :count] = row_positions[batch_index]
            segments[batch_index, :count] = row_segments[batch_index]
        return CompressionResult(
            keys=keys, values=values, positions=positions, segment_ids=segments,
            pending_sum=torch.stack(pending_sums, dim=0),
            pending_count=torch.tensor(pending_counts, device=hidden.device, dtype=torch.long),
            pending_segment_ids=torch.tensor(pending_segments, device=hidden.device, dtype=torch.long),
        )

    def compress_step(self, hidden: torch.Tensor, *, position: torch.Tensor | int,
                      segment_ids: torch.Tensor | None = None,
                      memory: MemoryState | None = None) -> CompressionResult:
        """Incrementally compress one token with the training block ratio.

        The sequence path emits one summary for every complete ``ratio``-token
        block.  Decode must keep the incomplete block pending; emitting a
        one-token summary changes both the memory density and the attention
        distribution seen by the model.
        """
        if hidden.ndim != 3 or hidden.shape[1] != 1:
            raise ValueError("compress_step expects hidden with shape [batch, 1, hidden]")
        batch, _, width = hidden.shape
        device = hidden.device
        token_segment = (
            segment_ids[:, 0].to(dtype=torch.long)
            if segment_ids is not None
            else torch.zeros(batch, device=device, dtype=torch.long)
        )
        pending_sum = None if memory is None else memory.pending_sum
        pending_count = None if memory is None else memory.pending_count
        pending_segments = None if memory is None else memory.pending_segment_ids
        if pending_sum is None or pending_sum.shape != (batch, width):
            pending_sum = hidden.new_zeros((batch, width))
            pending_count = torch.zeros(batch, device=device, dtype=torch.long)
            pending_segments = token_segment.clone()
        else:
            pending_count = pending_count.to(device=device, dtype=torch.long)
            pending_segments = pending_segments.to(device=device, dtype=torch.long)
            changed = pending_segments != token_segment
            pending_sum = torch.where(changed[:, None], torch.zeros_like(pending_sum), pending_sum)
            pending_count = torch.where(changed, torch.zeros_like(pending_count), pending_count)
            pending_segments = torch.where(changed, token_segment, pending_segments)

        pending_sum = pending_sum + hidden[:, 0]
        pending_count = pending_count + 1
        emitted = pending_count >= self.ratio
        mean = pending_sum / pending_count.clamp_min(1).to(hidden.dtype)[:, None]
        keys = self.key(mean).unsqueeze(1)
        values = self.value(mean).unsqueeze(1)
        offset = torch.as_tensor(position, device=device, dtype=torch.long).view(batch)
        positions = torch.where(emitted, offset, torch.full_like(offset, -1)).unsqueeze(1)
        emitted_segments = torch.where(emitted, token_segment, torch.full_like(token_segment, -1)).unsqueeze(1)

        pending_sum = torch.where(emitted[:, None], torch.zeros_like(pending_sum), pending_sum)
        pending_count = torch.where(emitted, torch.zeros_like(pending_count), pending_count)
        if memory is not None:
            memory.pending_sum = pending_sum
            memory.pending_count = pending_count
            memory.pending_segment_ids = token_segment
        return CompressionResult(
            keys, values, positions, emitted_segments,
            pending_sum, pending_count, token_segment,
        )

    def append(self, prior: MemoryState | None, current: CompressionResult) -> MemoryState:
        if prior is None or prior.keys is None:
            return MemoryState(
                current.keys, current.values, current.positions, current.segment_ids,
                pending_sum=current.pending_sum,
                pending_count=current.pending_count,
                pending_segment_ids=current.pending_segment_ids,
            )
        return MemoryState(
            keys=torch.cat([prior.keys, current.keys], dim=1),
            values=torch.cat([prior.values, current.values], dim=1),
            positions=torch.cat([prior.positions, current.positions], dim=1),
            segment_ids=torch.cat([prior.segment_ids, current.segment_ids], dim=1),
            pending_sum=current.pending_sum,
            pending_count=current.pending_count,
            pending_segment_ids=current.pending_segment_ids,
        )
