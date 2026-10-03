"""Fused Banded Attention using PyTorch Flash-SDPA for CSA and HCA."""
from __future__ import annotations
import torch
import torch.nn.functional as F

def fused_banded_local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int = 128,
    segment_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Computes causal local window attention using Flash-SDPA without unfold() allocations."""
    B, T, H, D = q.shape
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    if v_t.shape[1] == 1 and q_t.shape[1] > 1:
        # CSA has multi-head indices but a shared model-width value stream.
        # SDPA permits a different value width; broadcast the shared values
        # across query heads without materializing a T*T attention matrix.
        v_t = v_t.expand(-1, q_t.shape[1], -1, -1)

    positions = torch.arange(T, device=q.device)
    row_pos = positions.unsqueeze(-1)
    col_pos = positions.unsqueeze(0)
    causal_mask = (row_pos >= col_pos) & (row_pos - col_pos < window_size)

    attn_mask = causal_mask.view(1, 1, T, T)

    if segment_ids is not None:
        seg_match = (segment_ids.unsqueeze(-1) == segment_ids.unsqueeze(-2))
        attn_mask = attn_mask & seg_match.unsqueeze(1)

    out = F.scaled_dot_product_attention(
        q_t, k_t, v_t,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
    )
    return out.transpose(1, 2)
