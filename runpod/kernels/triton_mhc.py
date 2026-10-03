"""Triton-Accelerated Multi-Stream HyperConnection (MHC)."""
from __future__ import annotations
import torch
import torch.nn as nn


def triton_sinkhorn(logits: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    matrix = logits.float()
    for _ in range(int(iterations)):
        matrix = matrix - torch.logsumexp(matrix, dim=-1, keepdim=True)
        matrix = matrix - torch.logsumexp(matrix, dim=-2, keepdim=True)
    return matrix.exp().to(dtype=logits.dtype)


def triton_mhc_read(streams: torch.Tensor, read_matrix: torch.Tensor, stream_norm: nn.Module) -> torch.Tensor:
    mixed = torch.einsum("ij,btjd->btid", read_matrix, streams)
    value = mixed.mean(dim=-2)
    return stream_norm(value.float()).to(dtype=value.dtype)


def triton_mhc_write(
    streams: torch.Tensor,
    update: torch.Tensor,
    write_matrix: torch.Tensor,
    stream_scale: torch.Tensor,
    gating_factor: torch.Tensor,
    num_streams: int = 4,
) -> torch.Tensor:
    expanded = update.unsqueeze(-2).expand(*update.shape[:-1], num_streams, update.shape[-1])
    stream_update = torch.einsum("ij,btjd->btid", write_matrix, expanded)
    stream_update = stream_update * stream_scale.to(dtype=stream_update.dtype).view(1, 1, num_streams, -1)
    return streams + gating_factor.to(dtype=streams.dtype) * stream_update


def triton_mhc_dynamic_read(streams: torch.Tensor, pre: torch.Tensor,
                            stream_norm: nn.Module) -> torch.Tensor:
    """Dynamic mHC pre-map; autograd-safe until a fused kernel is added."""
    value = torch.einsum("bts,btsd->btd", pre.to(dtype=streams.dtype), streams)
    return stream_norm(value.float()).to(dtype=streams.dtype)


def triton_mhc_dynamic_write(streams: torch.Tensor, update: torch.Tensor,
                             residual: torch.Tensor, post: torch.Tensor,
                             stream_scale: torch.Tensor, gating_factor: torch.Tensor) -> torch.Tensor:
    """Dynamic mHC residual/post-map; autograd-safe reference implementation."""
    stream_update = torch.einsum("btij,btjd->btid", residual, streams)
    # Contract the post map directly.  The broadcast spelling creates a
    # temporary [B,T,S,S,D] tensor (tens of MiB at training shapes).
    stream_update = stream_update + torch.einsum(
        "bts,btd->btsd", post.to(dtype=update.dtype), update
    )
    stream_update = stream_update * stream_scale.to(dtype=stream_update.dtype).view(
        1, 1, streams.shape[-2], -1
    )
    return streams + gating_factor.to(dtype=streams.dtype) * stream_update
