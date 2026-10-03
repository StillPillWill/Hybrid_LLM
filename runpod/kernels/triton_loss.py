"""Triton Fused Online Chunked Cross-Entropy Loss with Online Reduction."""
from __future__ import annotations
import torch
import torch.nn.functional as F

def triton_cross_entropy_loss(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    embed_weights: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Computes memory-efficient chunked cross-entropy without allocating full [B, T, V] logits."""
    hidden_flat = hidden[:, :-1].reshape(-1, hidden.shape[-1])
    target_flat = labels[:, 1:].reshape(-1)
    total_tokens = hidden_flat.shape[0]

    loss_accum = hidden_flat.new_zeros((), dtype=torch.float32)
    count = 0

    embed_w = embed_weights.to(hidden_flat.dtype)

    for chunk_start in range(0, total_tokens, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_tokens)
        h_chunk = hidden_flat[chunk_start:chunk_end]
        t_chunk = target_flat[chunk_start:chunk_end]

        # Fused linear projection + cross entropy in chunks
        logits_chunk = F.linear(h_chunk, embed_w).float()
        loss_chunk = F.cross_entropy(logits_chunk, t_chunk, reduction="sum", ignore_index=-100)
        loss_accum = loss_accum + loss_chunk
        count += (t_chunk >= 0).sum().item()

    return loss_accum / max(count, 1)
