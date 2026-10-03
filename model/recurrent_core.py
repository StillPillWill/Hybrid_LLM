"""Reusable recurrent-core execution wrapper."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .recurrent_state import RecurrentState


class RecurrentCore(nn.Module):
    """Execute a physical block list repeatedly without cloning its weights."""

    def __init__(self, blocks: nn.ModuleList) -> None:
        super().__init__()
        self.blocks = blocks

    def forward(self, streams: torch.Tensor, state: RecurrentState, *, num_loops: int,
                segment_ids: torch.Tensor | None = None, reset_mask: torch.Tensor | None = None,
                diagnostics: dict[str, Any] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        diagnostics = diagnostics if diagnostics is not None else {}
        auxiliary = streams.new_zeros((), dtype=torch.float32)
        for _ in range(int(num_loops)):
            for block in self.blocks:
                streams, block_aux = block(
                    streams, state, segment_ids=segment_ids,
                    reset_mask=reset_mask, diagnostics=diagnostics,
                )
                auxiliary = auxiliary + block_aux.float()
        return streams, auxiliary
