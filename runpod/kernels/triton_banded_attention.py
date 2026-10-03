"""Triton Fused Banded Causal Flash-Attention Kernel.

Computes local causal window attention in on-chip SRAM:
- Skips blocks outside [i - W, i] completely (zero HBM loads, zero unnecessary FLOPs)
- Implements online softmax (FlashAttention-2 algorithm) in Triton
- Eliminates PyTorch full-matrix SDPA mask fallback overhead
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit
    def _banded_attn_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, Out_ptr, Lse_ptr,
        sm_scale,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        Z, H, N_CTX,
        WINDOW_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        q_offset = off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        k_offset = off_z * stride_kz + off_h * stride_kh + offs_d[:, None] * stride_kk
        v_offset = off_z * stride_vz + off_h * stride_vh + offs_d[None, :] * stride_vk

        q = tl.load(Q_ptr + q_offset, mask=offs_m[:, None] < N_CTX, other=0.0)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        min_n_idx = tl.maximum(0, (start_m * BLOCK_M - WINDOW_SIZE) // BLOCK_N)
        max_n_idx = tl.minimum(tl.cdiv(N_CTX, BLOCK_N), ((start_m + 1) * BLOCK_M + BLOCK_N - 1) // BLOCK_N)

        for n_block in range(min_n_idx, max_n_idx):
            cur_offs_n = n_block * BLOCK_N + offs_n

            k = tl.load(K_ptr + k_offset + cur_offs_n[None, :] * stride_kn, mask=cur_offs_n[None, :] < N_CTX, other=0.0)
            v = tl.load(V_ptr + v_offset + cur_offs_n[:, None] * stride_vn, mask=cur_offs_n[:, None] < N_CTX, other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k) * sm_scale

            causal_mask = (offs_m[:, None] >= cur_offs_n[None, :]) & (offs_m[:, None] - cur_offs_n[None, :] < WINDOW_SIZE)
            valid_mask = causal_mask & (cur_offs_n[None, :] < N_CTX)
            qk = tl.where(valid_mask, qk, -float("inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)

            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]

            p = p.to(v.dtype)
            acc += tl.dot(p, v)

            m_i = m_ij

        acc = acc / l_i[:, None]

        out_offset = off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
        tl.store(Out_ptr + out_offset, acc.to(Q_ptr.dtype.element_ty), mask=offs_m[:, None] < N_CTX)

        lse_offset = off_hz * N_CTX + offs_m
        tl.store(Lse_ptr + lse_offset, m_i + tl.log(l_i), mask=offs_m < N_CTX)


class _TritonBandedAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, window_size: int, sm_scale: float | None = None):
        B, H, T, D = q.shape
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(D)

        out = torch.empty_like(q)
        lse = torch.empty(B * H, T, device=q.device, dtype=torch.float32)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_DMODEL = D

        grid = (triton.cdiv(T, BLOCK_M), B * H)

        _banded_attn_fwd_kernel[grid](
            q, k, v, out, lse,
            sm_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            B, H, T,
            WINDOW_SIZE=window_size,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(q, k, v)
        ctx.sm_scale = sm_scale
        ctx.window_size = window_size
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """Reference backward for correctness until a fused bw kernel exists.

        Recomputing through the PyTorch banded implementation keeps gradients
        correct for training.  The Triton forward path is still used, while
        backward is deliberately explicit and easy to validate.
        """
        q, k, v = ctx.saved_tensors
        with torch.enable_grad():
            q_ref = q.detach().requires_grad_(True)
            k_ref = k.detach().requires_grad_(True)
            v_ref = v.detach().requires_grad_(True)
            from .fused_attention import fused_banded_local_attention
            ref = fused_banded_local_attention(
                q_ref.transpose(1, 2), k_ref.transpose(1, 2), v_ref.transpose(1, 2),
                window_size=ctx.window_size,
            ).transpose(1, 2)
            grads = torch.autograd.grad(
                ref, (q_ref, k_ref, v_ref), grad_out,
                allow_unused=True,
            )
        return (
            grads[0] if q.requires_grad else None,
            grads[1] if k.requires_grad else None,
            grads[2] if v.requires_grad else None,
            None,
            None,
        )


def triton_banded_local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int = 128,
) -> torch.Tensor:
    """Triton fused Flash banded local attention for CSA and HCA."""
    if (
        _TRITON_AVAILABLE and q.is_cuda and q.shape[-1] in (32, 64, 128)
        and k.shape[-1] == q.shape[-1]
        and v.shape[-1] == q.shape[-1]
        and q.shape[2] == k.shape[2] == v.shape[2]
    ):
        q_h = q.transpose(1, 2).contiguous()
        k_h = k.transpose(1, 2).contiguous()
        v_h = v.transpose(1, 2).contiguous()
        out_h = _TritonBandedAttentionFunction.apply(q_h, k_h, v_h, window_size)
        return out_h.transpose(1, 2).contiguous()

    from .fused_attention import fused_banded_local_attention
    return fused_banded_local_attention(q, k, v, window_size=window_size)
