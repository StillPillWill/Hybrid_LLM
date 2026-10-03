"""Self-contained sampler for the current looped hybrid-MoE checkpoint format.

This file intentionally contains the model definitions and sampler in one module.
It does not import model/, train/, data/, or runpod/ modules from this repository.
Run with:
  python sample_standalone.py -c CHECKPOINT -p "prompt" -n 32
  python sample_standalone.py --gui -c CHECKPOINT --port 7860
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import subprocess
import sys
import time
import types
import warnings
import zipfile
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from itertools import count
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint, checkpoint as activation_checkpoint

_REPO_ROOT = Path(__file__).resolve().parent
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Triton is an optional Linux/CUDA acceleration path.  In particular, the
# Windows Python environment cannot execute Triton kernels even when a
# ``triton`` package happens to be importable.  Keep the standalone sampler
# usable locally by selecting the reference PyTorch implementations there.
# RunPod/Linux still gets Triton automatically unless the caller explicitly
# disables it with LLMY_DISABLE_TRITON=1.
_TRITON_DISABLED = (
    os.name == "nt"
    or os.environ.get("LLMY_DISABLE_TRITON", "0") == "1"
)

# Match the distributed sampler's warning policy: suppress only Numba's
# advisory occupancy message, never runtime warnings or kernel failures.
try:
    from numba.core.errors import NumbaPerformanceWarning
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
except Exception:
    pass

def load_config(path: str) -> dict[str, Any]:
    """Load a legacy external YAML config without importing train.pretrain."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required only for legacy checkpoints without embedded config") from exc
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return value


# ===== inlined optional acceleration kernels =====
def fused_banded_local_attention(q, k, v, window_size=128, segment_ids=None):
    """Self-contained Flash-SDPA fallback used by the banded kernel wrapper."""
    q_t, k_t, v_t = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if v_t.shape[1] == 1 and q_t.shape[1] > 1:
        v_t = v_t.expand(-1, q_t.shape[1], -1, -1)
    positions = torch.arange(q.shape[1], device=q.device)
    causal = (positions[:, None] >= positions[None, :]) & (
        positions[:, None] - positions[None, :] < int(window_size)
    )
    mask = causal.view(1, 1, q.shape[1], q.shape[1])
    if segment_ids is not None:
        mask = mask & (segment_ids.unsqueeze(-1) == segment_ids.unsqueeze(-2)).unsqueeze(1)
    out = torch.nn.functional.scaled_dot_product_attention(
        q_t, k_t, v_t, attn_mask=mask, dropout_p=0.0, is_causal=False
    )
    return out.transpose(1, 2)


# ===== inlined optional acceleration: rmsnorm =====
"""Triton Fused RMSNorm (Forward + Backward) with Zero-HBM-Allocation Backward."""import torch
import torch.nn as nn

if _TRITON_DISABLED:
    _RMSNORM_TRITON_AVAILABLE = False
else:
    try:
        import triton
        import triton.language as tl
        _RMSNORM_TRITON_AVAILABLE = True
    except Exception:
        _RMSNORM_TRITON_AVAILABLE = False


if _RMSNORM_TRITON_AVAILABLE:
    @triton.jit
    def _rmsnorm_fwd_kernel(
        X_ptr, W_ptr, Out_ptr, Rrms_ptr,
        stride_x_row, stride_out_row,
        N: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        row_x_ptr = X_ptr + row_idx * stride_x_row
        row_out_ptr = Out_ptr + row_idx * stride_out_row

        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x = tl.load(row_x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        var = tl.sum(x * x, axis=0) / N
        rrms = 1.0 / tl.sqrt(var + eps)

        tl.store(Rrms_ptr + row_idx, rrms)

        out = (x * rrms) * w
        tl.store(row_out_ptr + cols, out.to(X_ptr.dtype.element_ty), mask=mask)


    @triton.jit
    def _rmsnorm_bwd_kernel(
        GradOut_ptr, X_ptr, W_ptr, Rrms_ptr,
        GradX_ptr, GradW_acc_ptr,
        stride_go_row, stride_x_row, stride_gx_row,
        M, N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        ROWS_PER_BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # Local register accumulator for grad_w across rows
        dw_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        row_start = pid * ROWS_PER_BLOCK
        row_end = tl.minimum(row_start + ROWS_PER_BLOCK, M)

        for r in range(row_start, row_end):
            dout = tl.load(GradOut_ptr + r * stride_go_row + cols, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(X_ptr + r * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
            rrms = tl.load(Rrms_ptr + r).to(tl.float32)

            dw_acc += dout * x * rrms

            dout_w = dout * w
            sum_dout_w_x = tl.sum(dout_w * x, axis=0)
            dx = rrms * (dout_w - (rrms * rrms / N) * x * sum_dout_w_x)
            tl.store(GradX_ptr + r * stride_gx_row + cols, dx.to(X_ptr.dtype.element_ty), mask=mask)

        # Atomic reduction once per block: reduces memory traffic by ROWS_PER_BLOCK (16x)
        tl.atomic_add(GradW_acc_ptr + cols, dw_acc, mask=mask)


class _TritonRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
        M, N = x_2d.shape
        out_2d = torch.empty_like(x_2d)
        rrms = torch.empty(M, device=x.device, dtype=torch.float32)

        BLOCK_SIZE = triton.next_power_of_2(N)
        _rmsnorm_fwd_kernel[(M,)](
            x_2d, weight, out_2d, rrms,
            x_2d.stride(0), out_2d.stride(0),
            N=N, eps=eps, BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

        ctx.save_for_backward(x_2d, weight, rrms)
        ctx.N = N
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.orig_shape = orig_shape
        return out_2d.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_out):
        x_2d, weight, rrms = ctx.saved_tensors
        M, N = x_2d.shape
        grad_out_2d = grad_out.reshape(-1, N).contiguous()
        grad_x_2d = torch.empty_like(x_2d)
        grad_w_acc = torch.zeros_like(weight, dtype=torch.float32)

        ROWS_PER_BLOCK = 16
        grid = (triton.cdiv(M, ROWS_PER_BLOCK),)

        _rmsnorm_bwd_kernel[grid](
            grad_out_2d, x_2d, weight, rrms,
            grad_x_2d, grad_w_acc,
            grad_out_2d.stride(0), x_2d.stride(0), grad_x_2d.stride(0),
            M=M, N=N, BLOCK_SIZE=ctx.BLOCK_SIZE,
            ROWS_PER_BLOCK=ROWS_PER_BLOCK,
            num_warps=4,
        )

        return grad_x_2d.reshape(ctx.orig_shape), grad_w_acc.to(weight.dtype), None


class TritonRMSNorm(nn.Module):
    """Drop-in ultra-fast replacement for RMSNorm with zero-copy HBM backward."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _RMSNORM_TRITON_AVAILABLE and x.is_cuda:
            w = self.weight if self.weight.device == x.device else self.weight.to(device=x.device)
            return _TritonRMSNormFunction.apply(x, w, self.eps)
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        return normalized * self.weight.to(dtype=x.dtype)


# ===== inlined optional acceleration: banded =====
"""Triton Fused Banded Causal Flash-Attention Kernel.

Computes local causal window attention in on-chip SRAM:
- Skips blocks outside [i - W, i] completely (zero HBM loads, zero unnecessary FLOPs)
- Implements online softmax (FlashAttention-2 algorithm) in Triton
- Eliminates PyTorch full-matrix SDPA mask fallback overhead
"""
import math
import torch
import torch.nn as nn

if _TRITON_DISABLED:
    _BANDED_TRITON_AVAILABLE = False
else:
    try:
        import triton
        import triton.language as tl
        _BANDED_TRITON_AVAILABLE = True
    except Exception:
        _BANDED_TRITON_AVAILABLE = False


if _BANDED_TRITON_AVAILABLE:
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
        _BANDED_TRITON_AVAILABLE and q.is_cuda and q.shape[-1] in (32, 64, 128)
        and k.shape[-1] == q.shape[-1]
        and v.shape[-1] == q.shape[-1]
        and q.shape[2] == k.shape[2] == v.shape[2]
    ):
        q_h = q.transpose(1, 2).contiguous()
        k_h = k.transpose(1, 2).contiguous()
        v_h = v.transpose(1, 2).contiguous()
        out_h = _TritonBandedAttentionFunction.apply(q_h, k_h, v_h, window_size)
        return out_h.transpose(1, 2).contiguous()

    return fused_banded_local_attention(q, k, v, window_size=window_size)

# ===== inlined optional acceleration: gdn =====
"""Triton Fused Forward and Backward Recurrent Scan for GatedDeltaNet.

Optimized for Ampere (sm_86 / A5000):
- Forward: Evaluates the delta recurrence on-chip in SRAM blocks (64x64)
- Backward: Recomputes adjoint gradients via backward pass on-chip
- Zero Python looping; full Tensor Core and vector load utilization.
"""
import torch
import torch.nn as nn

if _TRITON_DISABLED:
    _GDN_TRITON_AVAILABLE = False
else:
    try:
        import triton
        import triton.language as tl
        _GDN_TRITON_AVAILABLE = True
    except Exception:
        _GDN_TRITON_AVAILABLE = False


if _GDN_TRITON_AVAILABLE:
    @triton.jit
    def _gdn_step_kernel(
        Q_ptr, K_ptr, V_ptr, Beta_ptr, Gate_ptr, Reset_ptr, Decay_ptr,
        State_ptr, Out_ptr,
        sqb, sqt, sqh, sqk, skb, skt, skh, skk, svb, svt, svh, svv,
        sbb, sbt, sbh, sgb, sgt, sgh, sgv, srb, srt, sdh,
        ssb, ssh, ssk, ssv, sob, sot, soh, sov,
        H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    ):
        b_idx = tl.program_id(0)
        h_idx = tl.program_id(1)
        k_offs = tl.arange(0, K)
        v_offs = tl.arange(0, V)
        state = tl.load(
            State_ptr + b_idx * ssb + h_idx * ssh + k_offs[:, None] * ssk + v_offs[None, :] * ssv
        ).to(tl.float32)
        reset = tl.load(Reset_ptr + b_idx * srb + 0 * srt)
        state = tl.where(reset, tl.zeros((K, V), dtype=tl.float32), state)
        k_val = tl.load(K_ptr + b_idx * skb + 0 * skt + h_idx * skh + k_offs * skk).to(tl.float32)
        q_val = tl.load(Q_ptr + b_idx * sqb + 0 * sqt + h_idx * sqh + k_offs * sqk).to(tl.float32)
        v_val = tl.load(V_ptr + b_idx * svb + 0 * svt + h_idx * svh + v_offs * svv).to(tl.float32)
        beta_val = tl.load(Beta_ptr + b_idx * sbb + 0 * sbt + h_idx * sbh).to(tl.float32)
        gate_val = tl.load(Gate_ptr + b_idx * sgb + 0 * sgt + h_idx * sgh + v_offs * sgv).to(tl.float32)
        decay = tl.load(Decay_ptr + h_idx * sdh).to(tl.float32)
        prediction = tl.sum(k_val[:, None] * state, axis=0)
        delta = v_val - prediction
        state = decay * state + beta_val * (k_val[:, None] * delta[None, :])
        output = tl.sum(q_val[:, None] * state, axis=0) * gate_val
        tl.store(Out_ptr + b_idx * sob + 0 * sot + h_idx * soh + v_offs * sov, output)
        tl.store(
            State_ptr + b_idx * ssb + h_idx * ssh + k_offs[:, None] * ssk + v_offs[None, :] * ssv,
            state,
        )


    @triton.jit
    def _gdn_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, Beta_ptr, Gate_ptr, Reset_ptr, Decay_ptr, StateIn_ptr,
        Out_ptr, StateOut_ptr, AllStates_ptr,
        stride_qb, stride_qt, stride_qh, stride_qk,
        stride_kb, stride_kt, stride_kh, stride_kk,
        stride_vb, stride_vt, stride_vh, stride_vv,
        stride_betab, stride_betat, stride_betah,
        stride_gateb, stride_gatet, stride_gateh, stride_gatev,
        stride_resetb, stride_resett,
        stride_decayh,
        stride_sinb, stride_sinh, stride_sink, stride_sinv,
        stride_outb, stride_outt, stride_outh, stride_outv,
        stride_soutb, stride_south, stride_soutk, stride_soutv,
        stride_asb, stride_ast, stride_ash, stride_ask, stride_asv,
        B, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    ):
        # 1 program instance per (batch, head)
        b_idx = tl.program_id(0)
        h_idx = tl.program_id(1)

        k_offs = tl.arange(0, K)
        v_offs = tl.arange(0, V)
        state_offs = k_offs[:, None] * stride_sink + v_offs[None, :] * stride_sinv

        # Load initial state from StateIn [B, H, K, V]
        cur_state_ptr = StateIn_ptr + b_idx * stride_sinb + h_idx * stride_sinh + state_offs
        state = tl.load(cur_state_ptr).to(tl.float32)

        decay = tl.load(Decay_ptr + h_idx * stride_decayh).to(tl.float32)

        # Store initial state into AllStates at t=0
        as_offs = k_offs[:, None] * stride_ask + v_offs[None, :] * stride_asv
        tl.store(AllStates_ptr + b_idx * stride_asb + 0 * stride_ast + h_idx * stride_ash + as_offs, state)

        for t in range(T):
            reset = tl.load(Reset_ptr + b_idx * stride_resetb + t * stride_resett)
            if reset:
                state = tl.zeros((K, V), dtype=tl.float32)

            k_val = tl.load(K_ptr + b_idx * stride_kb + t * stride_kt + h_idx * stride_kh + k_offs * stride_kk).to(tl.float32)
            v_val = tl.load(V_ptr + b_idx * stride_vb + t * stride_vt + h_idx * stride_vh + v_offs * stride_vv).to(tl.float32)
            q_val = tl.load(Q_ptr + b_idx * stride_qb + t * stride_qt + h_idx * stride_qh + k_offs * stride_qk).to(tl.float32)
            beta_val = tl.load(Beta_ptr + b_idx * stride_betab + t * stride_betat + h_idx * stride_betah).to(tl.float32)
            gate_val = tl.load(Gate_ptr + b_idx * stride_gateb + t * stride_gatet + h_idx * stride_gateh + v_offs * stride_gatev).to(tl.float32)

            # pred[v] = sum_k (k[k] * state[k, v])
            pred = tl.sum(k_val[:, None] * state, axis=0)  # [V]
            delta = v_val - pred                           # [V]

            # state[k, v] = decay * state[k, v] + beta * k[k] * delta[v]
            state = decay * state + beta_val * (k_val[:, None] * delta[None, :])

            # Save state at t+1
            tl.store(AllStates_ptr + b_idx * stride_asb + (t + 1) * stride_ast + h_idx * stride_ash + as_offs, state)

            # out[v] = (sum_k q[k] * state[k, v]) * gate[v]
            out_val = tl.sum(q_val[:, None] * state, axis=0) * gate_val
            tl.store(Out_ptr + b_idx * stride_outb + t * stride_outt + h_idx * stride_outh + v_offs * stride_outv, out_val)

        # Store final state into StateOut
        sout_offs = k_offs[:, None] * stride_soutk + v_offs[None, :] * stride_soutv
        tl.store(StateOut_ptr + b_idx * stride_soutb + h_idx * stride_south + sout_offs, state)


    @triton.jit
    def _gdn_bwd_kernel(
        GradOut_ptr, Q_ptr, K_ptr, V_ptr, Beta_ptr, Gate_ptr, Reset_ptr, Decay_ptr, AllStates_ptr,
        GradQ_ptr, GradK_ptr, GradV_ptr, GradBeta_ptr, GradGate_ptr, GradDecay_ptr, GradStateIn_ptr,
        stride_gob, stride_got, stride_goh, stride_gov,
        stride_qb, stride_qt, stride_qh, stride_qk,
        stride_kb, stride_kt, stride_kh, stride_kk,
        stride_vb, stride_vt, stride_vh, stride_vv,
        stride_betab, stride_betat, stride_betah,
        stride_gateb, stride_gatet, stride_gateh, stride_gatev,
        stride_resetb, stride_resett,
        stride_decayh,
        stride_asb, stride_ast, stride_ash, stride_ask, stride_asv,
        stride_gqb, stride_gqt, stride_gqh, stride_gqk,
        stride_gkb, stride_gkt, stride_gkh, stride_gkk,
        stride_gvb, stride_gvt, stride_gvh, stride_gvv,
        stride_gbetab, stride_gbetat, stride_gbetah,
        stride_ggateb, stride_ggatet, stride_ggateh, stride_ggatev,
        stride_gdecayb, stride_gdecayh,
        stride_gsinb, stride_gsinh, stride_gsink, stride_gsinv,
        B, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    ):
        b_idx = tl.program_id(0)
        h_idx = tl.program_id(1)

        k_offs = tl.arange(0, K)
        v_offs = tl.arange(0, V)
        decay = tl.load(Decay_ptr + h_idx * stride_decayh).to(tl.float32)

        dstate = tl.zeros((K, V), dtype=tl.float32)
        total_ddecay = 0.0

        for t in range(T - 1, -1, -1):
            q_val = tl.load(Q_ptr + b_idx * stride_qb + t * stride_qt + h_idx * stride_qh + k_offs * stride_qk).to(tl.float32)
            k_val = tl.load(K_ptr + b_idx * stride_kb + t * stride_kt + h_idx * stride_kh + k_offs * stride_kk).to(tl.float32)
            v_val = tl.load(V_ptr + b_idx * stride_vb + t * stride_vt + h_idx * stride_vh + v_offs * stride_vv).to(tl.float32)
            beta_val = tl.load(Beta_ptr + b_idx * stride_betab + t * stride_betat + h_idx * stride_betah).to(tl.float32)
            gate_val = tl.load(Gate_ptr + b_idx * stride_gateb + t * stride_gatet + h_idx * stride_gateh + v_offs * stride_gatev).to(tl.float32)
            gout = tl.load(GradOut_ptr + b_idx * stride_gob + t * stride_got + h_idx * stride_goh + v_offs * stride_gov).to(tl.float32)

            # Load S_{t+1} and S_t
            as_offs = k_offs[:, None] * stride_ask + v_offs[None, :] * stride_asv
            s_tp1 = tl.load(AllStates_ptr + b_idx * stride_asb + (t + 1) * stride_ast + h_idx * stride_ash + as_offs).to(tl.float32)
            s_t   = tl.load(AllStates_ptr + b_idx * stride_asb + t * stride_ast + h_idx * stride_ash + as_offs).to(tl.float32)

            reset = tl.load(Reset_ptr + b_idx * stride_resetb + t * stride_resett)
            if reset:
                s_t = tl.zeros((K, V), dtype=tl.float32)

            # Grad gate
            acc_out = tl.sum(q_val[:, None] * s_tp1, axis=0)
            ggate = gout * acc_out
            tl.store(GradGate_ptr + b_idx * stride_ggateb + t * stride_ggatet + h_idx * stride_ggateh + v_offs * stride_ggatev, ggate)

            # dstate accumulation from output
            dstate = dstate + (q_val[:, None] * (gout * gate_val)[None, :])

            # Grad q
            gq = tl.sum(s_tp1 * (gout * gate_val)[None, :], axis=1)
            tl.store(GradQ_ptr + b_idx * stride_gqb + t * stride_gqt + h_idx * stride_gqh + k_offs * stride_gqk, gq)

            # Grad through S_{t+1} = decay * S_t + beta * (k (x) delta)
            pred = tl.sum(k_val[:, None] * s_t, axis=0)
            delta = v_val - pred

            gbeta = tl.sum(dstate * (k_val[:, None] * delta[None, :]))
            tl.store(GradBeta_ptr + b_idx * stride_gbetab + t * stride_gbetat + h_idx * stride_gbetah, gbeta)

            g_delta = tl.sum(dstate * (beta_val * k_val[:, None]), axis=0)  # [V]
            gv = g_delta
            tl.store(GradV_ptr + b_idx * stride_gvb + t * stride_gvt + h_idx * stride_gvh + v_offs * stride_gvv, gv)

            gk = tl.sum(dstate * (beta_val * delta[None, :]), axis=1) - tl.sum(s_t * g_delta[None, :], axis=1)
            tl.store(GradK_ptr + b_idx * stride_gkb + t * stride_gkt + h_idx * stride_gkh + k_offs * stride_gkk, gk)

            total_ddecay += tl.sum(dstate * s_t)
            dstate = decay * dstate - (k_val[:, None] * g_delta[None, :])

            if reset:
                dstate = tl.zeros((K, V), dtype=tl.float32)

        tl.store(GradDecay_ptr + b_idx * stride_gdecayb + h_idx * stride_gdecayh, total_ddecay)
        gsin_offs = k_offs[:, None] * stride_gsink + v_offs[None, :] * stride_gsinv
        tl.store(GradStateIn_ptr + b_idx * stride_gsinb + h_idx * stride_gsinh + gsin_offs, dstate)


class _TritonGDNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, beta, gate, reset_mask, decay_logit, state_in):
        B, T, H, K = q.shape
        V = v.shape[-1]
        if K not in (64, 128) or V not in (64, 128):
            raise ValueError("GDN Triton kernel requires K,V in {64,128}")

        out = torch.empty(B, T, H, V, device=q.device, dtype=q.dtype)
        state_out = torch.empty(B, H, K, V, device=q.device, dtype=torch.float32)
        all_states = torch.empty(B, T + 1, H, K, V, device=q.device, dtype=torch.float32)

        if reset_mask is None:
            reset_mask = torch.zeros(B, T, device=q.device, dtype=torch.bool)
        else:
            reset_mask = reset_mask.bool()

        sig_decay = torch.sigmoid(decay_logit.float()).contiguous()

        grid = (B, H)
        _gdn_fwd_kernel[grid](
            q, k, v, beta.squeeze(-1) if beta.ndim == 4 else beta, gate, reset_mask, sig_decay.squeeze(), state_in,
            out, state_out, all_states,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            gate.stride(0), gate.stride(1), gate.stride(2), gate.stride(3),
            reset_mask.stride(0), reset_mask.stride(1),
            sig_decay.squeeze().stride(0),
            state_in.stride(0), state_in.stride(1), state_in.stride(2), state_in.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            state_out.stride(0), state_out.stride(1), state_out.stride(2), state_out.stride(3),
            all_states.stride(0), all_states.stride(1), all_states.stride(2), all_states.stride(3), all_states.stride(4),
            B=B, T=T, H=H, K=K, V=V,
            num_warps=8 if K == 128 else 4,
        )

        ctx.save_for_backward(q, k, v, beta, gate, reset_mask, sig_decay, decay_logit, all_states)
        return out, state_out

    @staticmethod
    def backward(ctx, grad_out, grad_state_out):
        q, k, v, beta, gate, reset_mask, sig_decay, decay_logit, all_states = ctx.saved_tensors
        B, T, H, K = q.shape
        V = v.shape[-1]

        grad_q = torch.empty_like(q)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        grad_beta = torch.empty(B, T, H, device=q.device, dtype=beta.dtype)
        grad_gate = torch.empty_like(gate)
        grad_decay_b = torch.empty(B, H, device=q.device, dtype=torch.float32)
        grad_state_in = torch.empty(B, H, K, V, device=q.device, dtype=torch.float32)

        grid = (B, H)
        _gdn_bwd_kernel[grid](
            grad_out, q, k, v, beta.squeeze(-1) if beta.ndim == 4 else beta, gate, reset_mask, sig_decay.squeeze(), all_states,
            grad_q, grad_k, grad_v, grad_beta, grad_gate, grad_decay_b, grad_state_in,
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3),
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            gate.stride(0), gate.stride(1), gate.stride(2), gate.stride(3),
            reset_mask.stride(0), reset_mask.stride(1),
            sig_decay.squeeze().stride(0),
            all_states.stride(0), all_states.stride(1), all_states.stride(2), all_states.stride(3), all_states.stride(4),
            grad_q.stride(0), grad_q.stride(1), grad_q.stride(2), grad_q.stride(3),
            grad_k.stride(0), grad_k.stride(1), grad_k.stride(2), grad_k.stride(3),
            grad_v.stride(0), grad_v.stride(1), grad_v.stride(2), grad_v.stride(3),
            grad_beta.stride(0), grad_beta.stride(1), grad_beta.stride(2),
            grad_gate.stride(0), grad_gate.stride(1), grad_gate.stride(2), grad_gate.stride(3),
            grad_decay_b.stride(0), grad_decay_b.stride(1),
            grad_state_in.stride(0), grad_state_in.stride(1), grad_state_in.stride(2), grad_state_in.stride(3),
            B=B, T=T, H=H, K=K, V=V,
            num_warps=8 if K == 128 else 4,
        )

        grad_beta = grad_beta.unsqueeze(-1) if beta.ndim == 4 else grad_beta
        dsig = (sig_decay * (1.0 - sig_decay)).view(1, H)
        grad_decay = (grad_decay_b * dsig).sum(dim=0).view(decay_logit.shape)
        return grad_q, grad_k, grad_v, grad_beta, grad_gate, None, grad_decay, grad_state_in


def triton_gdn_scan(q, k, v, beta, gate, reset_mask, decay, state_in):
    """Entry point for Triton-accelerated GDN Scan."""
    if not _GDN_TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available. Please install triton on Linux/Ampere.")
    return _TritonGDNFunction.apply(q, k, v, beta, gate, reset_mask, decay, state_in)


def triton_gdn_step(q, k, v, beta, gate, reset_mask, decay, state):
    """One-token Triton recurrence that updates the FP32 state in place."""
    if not _GDN_TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available. Please install triton on Linux/Ampere.")
    batch, _, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if q.shape[1] != 1 or key_dim not in (64, 128) or value_dim not in (64, 128):
        raise ValueError("triton_gdn_step expects one token with 64- or 128-wide heads")
    out = torch.empty((batch, 1, heads, value_dim), device=q.device, dtype=q.dtype)
    reset = torch.zeros((batch, 1), device=q.device, dtype=torch.bool) if reset_mask is None else reset_mask.bool()
    sig_decay = torch.sigmoid(decay.float()).reshape(-1).contiguous()
    _gdn_step_kernel[(batch, heads)](
        q, k, v, beta.squeeze(-1) if beta.ndim == 4 else beta, gate, reset, sig_decay, state, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        gate.stride(0), gate.stride(1), gate.stride(2), gate.stride(3),
        reset.stride(0), reset.stride(1), sig_decay.stride(0),
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        H=heads, K=key_dim, V=value_dim,
        num_warps=8 if key_dim == 128 else 4,
    )
    return out, state

# ===== inlined optional acceleration: moe =====
"""Triton Fused SwiGLU and Batched-GEMM MoE Dispatcher with Fused Gate-Up."""import torch
import torch.nn as nn
import torch.nn.functional as F

if _TRITON_DISABLED:
    _MOE_TRITON_AVAILABLE = False
else:
    try:
        import triton
        import triton.language as tl
        _MOE_TRITON_AVAILABLE = True
    except Exception:
        _MOE_TRITON_AVAILABLE = False


if _MOE_TRITON_AVAILABLE:
    @triton.jit
    def _moe_token_gate_up_kernel(
        X_ptr, Indices_ptr, Gate_ptr, Up_ptr, GateUp_ptr,
        D: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
        BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr,
        sx0, sx1, si0, si1, sg0, sg1, sg2, su0, su1, su2,
        so0, so1, so2,
    ):
        pid = tl.program_id(0)
        blocks_h = tl.cdiv(H, BLOCK_H)
        token = pid // (K * blocks_h)
        route = (pid // blocks_h) % K
        h_block = pid % blocks_h
        h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        expert = tl.load(Indices_ptr + token * si0 + route * si1)
        gate_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for d_start in tl.range(0, D, BLOCK_D):
            d = d_start + tl.arange(0, BLOCK_D)
            mask_h = h < H
            mask_d = d < D
            x = tl.load(X_ptr + token * sx0 + d * sx1, mask=mask_d, other=0.0).to(tl.float32)
            gate_w = tl.load(
                Gate_ptr + expert * sg0 + h[:, None] * sg1 + d[None, :] * sg2,
                mask=mask_h[:, None] & mask_d[None, :], other=0.0,
            ).to(tl.float32)
            up_w = tl.load(
                Up_ptr + expert * su0 + h[:, None] * su1 + d[None, :] * su2,
                mask=mask_h[:, None] & mask_d[None, :], other=0.0,
            ).to(tl.float32)
            gate_acc += tl.sum(gate_w * x[None, :], axis=1)
            up_acc += tl.sum(up_w * x[None, :], axis=1)
        tl.store(GateUp_ptr + token * so0 + route * so1 + h * so2, gate_acc, mask=h < H)
        tl.store(GateUp_ptr + token * so0 + route * so1 + (H + h) * so2, up_acc, mask=h < H)


    @triton.jit
    def _moe_token_down_kernel(
        GateUp_ptr, Indices_ptr, Prob_ptr, Down_ptr, Out_ptr,
        D: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
        BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr,
        sg0, sg1, sg2, si0, si1, sp0, sp1, sd0, sd1, sd2, so0, so1,
    ):
        pid = tl.program_id(0)
        token = pid // tl.cdiv(D, BLOCK_D)
        d_block = pid % tl.cdiv(D, BLOCK_D)
        d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        output = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for route in tl.static_range(K):
            expert = tl.load(Indices_ptr + token * si0 + route * si1)
            probability = tl.load(Prob_ptr + token * sp0 + route * sp1).to(tl.float32)
            route_output = tl.zeros((BLOCK_D,), dtype=tl.float32)
            for h_start in tl.range(0, H, BLOCK_H):
                h = h_start + tl.arange(0, BLOCK_H)
                gate = tl.load(
                    GateUp_ptr + token * sg0 + route * sg1 + h * sg2,
                    mask=h < H, other=0.0,
                ).to(tl.float32)
                up = tl.load(
                    GateUp_ptr + token * sg0 + route * sg1 + (H + h) * sg2,
                    mask=h < H, other=0.0,
                ).to(tl.float32)
                down_w = tl.load(
                    Down_ptr + expert * sd0 + d[:, None] * sd1 + h[None, :] * sd2,
                    mask=(d < D)[:, None] & (h < H)[None, :], other=0.0,
                ).to(tl.float32)
                route_output += tl.sum((gate * tl.sigmoid(gate) * up)[None, :] * down_w, axis=1)
            output += probability * route_output
        tl.store(Out_ptr + token * so0 + d * so1, output, mask=d < D)


    @triton.jit
    def _fused_swiglu_fwd_kernel(
        Gate_ptr, Up_ptr, Out_ptr,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        gate = tl.load(Gate_ptr + offs, mask=mask).to(tl.float32)
        up = tl.load(Up_ptr + offs, mask=mask).to(tl.float32)

        silu_gate = gate * tl.sigmoid(gate)
        out = silu_gate * up

        tl.store(Out_ptr + offs, out.to(Gate_ptr.dtype.element_ty), mask=mask)


    @triton.jit
    def _fused_swiglu_bwd_kernel(
        GradOut_ptr, Gate_ptr, Up_ptr, GradGate_ptr, GradUp_ptr,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        gout = tl.load(GradOut_ptr + offs, mask=mask).to(tl.float32)
        gate = tl.load(Gate_ptr + offs, mask=mask).to(tl.float32)
        up = tl.load(Up_ptr + offs, mask=mask).to(tl.float32)

        sig = tl.sigmoid(gate)
        silu = gate * sig
        dsilu_dgate = sig * (1.0 + gate * (1.0 - sig))

        ggate = gout * up * dsilu_dgate
        gup = gout * silu

        tl.store(GradGate_ptr + offs, ggate.to(Gate_ptr.dtype.element_ty), mask=mask)
        tl.store(GradUp_ptr + offs, gup.to(Up_ptr.dtype.element_ty), mask=mask)


class _TritonSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        assert gate.shape == up.shape
        out = torch.empty_like(gate)
        N = gate.numel()
        grid = (triton.cdiv(N, 2048),)
        _fused_swiglu_fwd_kernel[grid](gate, up, out, N, BLOCK_SIZE=2048, num_warps=8)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        gate, up = ctx.saved_tensors
        grad_gate = torch.empty_like(gate)
        grad_up = torch.empty_like(up)
        N = gate.numel()
        grid = (triton.cdiv(N, 2048),)
        _fused_swiglu_bwd_kernel[grid](grad_out, gate, up, grad_gate, grad_up, N, BLOCK_SIZE=2048, num_warps=8)
        return grad_gate, grad_up


def triton_fused_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if _MOE_TRITON_AVAILABLE and gate.is_cuda:
        return _TritonSwiGLUFunction.apply(gate, up)
    return F.silu(gate) * up


def triton_moe_dispatch(
    flat: torch.Tensor,
    indices: torch.Tensor,
    probabilities: torch.Tensor,
    routed_experts: int,
    w_gate: torch.Tensor,  # [E, H, D]
    w_up: torch.Tensor,    # [E, H, D]
    w_down: torch.Tensor,  # [E, D, H]
) -> torch.Tensor:
    """Grouped-GEMM MoE forward pass with single-bmm fused Gate-Up projections."""
    flat_indices = indices.reshape(-1)
    flat_probs = probabilities.reshape(-1)
    token_ids = torch.arange(flat.shape[0], device=flat.device).view(-1, 1).expand_as(indices).reshape(-1)

    sorted_indices, sort_order = torch.sort(flat_indices)
    sorted_tokens = token_ids[sort_order]
    sorted_probs = flat_probs[sort_order]

    counts = torch.bincount(sorted_indices, minlength=routed_experts)
    max_c = int(counts.max().item())
    if max_c == 0:
        return torch.zeros_like(flat)

    x_sorted = flat.index_select(0, sorted_tokens)

    cum_counts = torch.cumsum(counts, dim=0)
    offsets = torch.cat([torch.tensor([0], device=flat.device), cum_counts[:-1]])
    total_m = flat_indices.shape[0]
    token_seq = torch.arange(total_m, device=flat.device)
    exp_offsets = offsets[sorted_indices]
    slot_indices = token_seq - exp_offsets

    padded = torch.zeros(routed_experts, max_c, flat.shape[-1], device=flat.device, dtype=flat.dtype)
    padded[sorted_indices, slot_indices] = x_sorted

    # Fused Gate-Up GEMM: Concatenate w_gate and w_up along output dim -> single bmm launch!
    # [E, 2*H, D] -> transpose(1, 2) -> [E, D, 2*H]
    w_gate_up = torch.cat([w_gate, w_up], dim=1)
    h_gate_up = torch.bmm(padded, w_gate_up.transpose(1, 2))  # [E, max_c, 2*H]

    H = w_gate.shape[1]
    h_gate = h_gate_up[..., :H]
    h_up = h_gate_up[..., H:]

    act = triton_fused_swiglu(h_gate, h_up)
    out_padded = torch.bmm(act, w_down.transpose(1, 2)) # [E, max_c, D]

    out_sorted = out_padded[sorted_indices, slot_indices]
    output = torch.zeros_like(flat)
    output.index_add_(
        0,
        sorted_tokens,
        (out_sorted * sorted_probs.to(out_sorted.dtype).unsqueeze(-1)).to(output.dtype),
    )
    return output


# Keep the dispatcher aliases explicit.  The standalone copy used these names
# in SparseMoE._dispatch, but they were only present in the distributed module
# and therefore crashed the first local MoE forward when Triton was disabled.
_triton_available = _MOE_TRITON_AVAILABLE
_triton_moe_dispatch = triton_moe_dispatch if _MOE_TRITON_AVAILABLE else None


def triton_moe_step(
    flat: torch.Tensor,
    indices: torch.Tensor,
    probabilities: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Fused fixed-shape dispatcher for a one-token decode batch."""
    if not _MOE_TRITON_AVAILABLE or not flat.is_cuda:
        raise RuntimeError("Triton is required for triton_moe_step")
    batch, d_model = flat.shape
    routes = indices.shape[1]
    hidden = w_gate.shape[1]
    gate_up = torch.empty((batch, routes, hidden * 2), device=flat.device, dtype=torch.float32)
    output = torch.empty((batch, d_model), device=flat.device, dtype=torch.float32)
    block_h = 64
    block_d = 128
    _moe_token_gate_up_kernel[(batch * routes * triton.cdiv(hidden, block_h),)](
        flat, indices, w_gate, w_up, gate_up,
        D=d_model, H=hidden, K=routes, BLOCK_H=block_h, BLOCK_D=block_d,
        sx0=flat.stride(0), sx1=flat.stride(1),
        si0=indices.stride(0), si1=indices.stride(1),
        sg0=w_gate.stride(0), sg1=w_gate.stride(1), sg2=w_gate.stride(2),
        su0=w_up.stride(0), su1=w_up.stride(1), su2=w_up.stride(2),
        so0=gate_up.stride(0), so1=gate_up.stride(1), so2=gate_up.stride(2),
        num_warps=4,
    )
    _moe_token_down_kernel[(batch * triton.cdiv(d_model, block_d),)](
        gate_up, indices, probabilities, w_down, output,
        D=d_model, H=hidden, K=routes, BLOCK_H=block_h, BLOCK_D=block_d,
        sg0=gate_up.stride(0), sg1=gate_up.stride(1), sg2=gate_up.stride(2),
        si0=indices.stride(0), si1=indices.stride(1),
        sp0=probabilities.stride(0), sp1=probabilities.stride(1),
        sd0=w_down.stride(0), sd1=w_down.stride(1), sd2=w_down.stride(2),
        so0=output.stride(0), so1=output.stride(1),
        num_warps=4,
    )
    return output.to(dtype=flat.dtype)


# ===== inlined optional acceleration: mhc =====
"""Triton-Accelerated Multi-Stream HyperConnection (MHC)."""import torch
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
    stream_update = stream_update + post.to(dtype=update.dtype).unsqueeze(-1) * update.unsqueeze(-2)
    stream_update = stream_update * stream_scale.to(dtype=stream_update.dtype).view(
        1, 1, streams.shape[-2], -1
    )
    return streams + gating_factor.to(dtype=streams.dtype) * stream_update

# ===== inlined model component: config.py =====
"""Typed configuration and parameter accounting for the V0 architecture."""
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass
class MHCConfig:
    enabled: bool = True
    streams: int = 4
    sinkhorn_iterations: int = 20
    init_gating_factor: float = 0.01
    shared_across_loops: bool = True
    preserve_streams_across_loop_boundary: bool = True


@dataclass
class GDNConfig:
    qk_heads: int = 16
    value_heads: int = 16
    qk_head_dim: int = 128
    value_head_dim: int = 128
    conv_kernel: int = 4
    sensitive_state_dtype: str = "float32"


@dataclass
class MoEConfig:
    routed_experts: int = 128
    shared_experts: int = 4
    routed_top_k: int = 20
    expert_hidden: int = 640
    activation: str = "swiglu"
    normalize_topk_prob: bool = True
    aux_balance_coef: float = 0.001
    token_dropping: bool = False
    expert_weights_shared_across_loops: bool = True


@dataclass
class MLAConfig:
    latent_dim: int = 256


@dataclass
class RouterConfig:
    context_aware: bool = True
    query_heads: int = 4
    query_head_dim: int = 64
    local_context: int = 128
    use_hca_global_latents: bool = True
    scoring_dtype: str = "float32"


@dataclass
class CSAConfig:
    temporal_compression: int = 4
    local_window: int = 128
    index_heads: int = 16
    index_head_dim: int = 128
    index_top_k: int = 32


@dataclass
class HCAConfig:
    temporal_compression: int = 32
    local_window: int = 128
    dense_global_attention: bool = True
    global_top_k: int = 32


@dataclass
class DenseFFNConfig:
    hidden: int = 3584
    activation: str = "swiglu"


@dataclass
class LoopAttnResConfig:
    implemented_hook: bool = True
    enabled: bool = False
    latent_dim: int = 128
    heads: int = 4
    head_dim: int = 32
    residual_gate_init: float = 0.05


@dataclass
class AdaptiveLoopConfig:
    """Token-wise recurrent depth and truncated-backpropagation controls."""

    enabled: bool = False
    learned_halting: bool = True
    min_loops: int = 1
    max_loops: int = 1
    halt_penalty: float = 0.01
    halt_threshold: float = 0.5
    halt_temperature: float = 1.0
    initial_continue_prob: float = 0.6
    # ``0`` means backpropagate through every executed loop.  A positive value
    # keeps the forward depth unchanged but truncates gradients to the final N
    # recurrent passes.
    backprop_loops: int = 0


@dataclass
class V6Config:
    vocab_size: int = 248_320
    d_model: int = 1024
    tie_word_embeddings: bool = True
    physical_blocks: int = 10
    stem_blocks: int = 2
    recurrent_blocks: int = 6
    tail_blocks: int = 2
    num_loops: int = 2
    recurrent_layout: tuple[str, ...] = (
        "gdn_moe", "gdn_moe", "csa_moe", "gdn_moe", "gdn_moe", "gdn_moe"
    )
    max_position_embeddings: int = 4096
    pad_token_id: int | None = None
    mhc: MHCConfig = field(default_factory=MHCConfig)
    gdn: GDNConfig = field(default_factory=GDNConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    mla: MLAConfig = field(default_factory=MLAConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    csa: CSAConfig = field(default_factory=CSAConfig)
    hca: HCAConfig = field(default_factory=HCAConfig)
    dense_ffn: DenseFFNConfig = field(default_factory=DenseFFNConfig)
    loop_attnres: LoopAttnResConfig = field(default_factory=LoopAttnResConfig)
    adaptive_loops: AdaptiveLoopConfig = field(default_factory=AdaptiveLoopConfig)

    def __post_init__(self) -> None:
        if self.physical_blocks != self.stem_blocks + self.recurrent_blocks + self.tail_blocks:
            raise ValueError("physical_blocks must equal stem + recurrent + tail blocks")
        if len(self.recurrent_layout) != self.recurrent_blocks:
            raise ValueError("recurrent_layout must contain one entry per recurrent block")
        if self.num_loops < 1:
            raise ValueError("num_loops must be positive")
        if self.adaptive_loops.min_loops < 1:
            raise ValueError("adaptive loop minimum must be positive")
        if self.adaptive_loops.max_loops < self.adaptive_loops.min_loops:
            raise ValueError("adaptive loop maximum must be >= minimum")
        if self.adaptive_loops.max_loops > self.num_loops:
            raise ValueError("adaptive loop maximum cannot exceed num_loops")
        if not 0.0 < self.adaptive_loops.halt_threshold < 1.0:
            raise ValueError("halt threshold must be in (0, 1)")
        if self.adaptive_loops.halt_temperature <= 0.0:
            raise ValueError("halt temperature must be positive")
        if not 0.0 < self.adaptive_loops.initial_continue_prob < 1.0:
            raise ValueError("initial continue probability must be in (0, 1)")
        if self.adaptive_loops.halt_penalty < 0.0:
            raise ValueError("halt penalty cannot be negative")
        if self.adaptive_loops.backprop_loops < 0:
            raise ValueError("backprop loop count cannot be negative")
        if self.adaptive_loops.backprop_loops > self.num_loops:
            raise ValueError("backprop loop count cannot exceed num_loops")
        if self.mhc.streams < 1:
            raise ValueError("mHC streams must be positive")
        if self.moe.routed_top_k > self.moe.routed_experts:
            raise ValueError("routed_top_k cannot exceed routed_experts")
        if self.hca.global_top_k < 1:
            raise ValueError("hca.global_top_k must be positive")
        if self.gdn.qk_heads != self.gdn.value_heads:
            raise ValueError("V0 requires matching GDN query/key and value head counts")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "V6Config":
        data = dict(value.get("model", value))
        nested = {
            "mhc": MHCConfig,
            "gdn": GDNConfig,
            "moe": MoEConfig,
            "mla": MLAConfig,
            "router": RouterConfig,
            "csa": CSAConfig,
            "hca": HCAConfig,
            "dense_ffn": DenseFFNConfig,
            "loop_attnres": LoopAttnResConfig,
            "adaptive_loops": AdaptiveLoopConfig,
        }
        for name, typ in nested.items():
            if isinstance(data.get(name), dict):
                data[name] = typ(**data[name])
        if isinstance(data.get("recurrent_layout"), list):
            data["recurrent_layout"] = tuple(data["recurrent_layout"])
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_config() -> V6Config:
    """Return the exact architecture defaults from section 24 of the spec."""
    return V6Config()


def count_unique_parameters(module: Any) -> dict[str, Any]:
    """Count physical tensors, avoiding double-counts from loop reuse/tied weights."""
    named_parameters = list(module.named_parameters())
    unique: dict[int, tuple[str, Any]] = {}
    for name, parameter in named_parameters:
        unique.setdefault(id(parameter), (name, parameter))
    categories: dict[str, int] = {}
    for parameter_name, parameter in unique.values():
        parameter_category = "other"
        for prefix, category in (
            ("embed", "embedding_head"), ("lm_head", "embedding_head"),
            ("experts", "routed_experts"), ("shared", "shared_experts"),
            ("gdn", "gdn"), ("csa", "csa"), ("hca", "hca"),
            ("router", "mla_router_indexer"), ("mla", "mla_router_indexer"),
            ("mhc", "mhc"), ("norm", "norms_other"),
        ):
            if prefix in parameter_name:
                parameter_category = category
                break
        categories[parameter_category] = categories.get(parameter_category, 0) + int(parameter.numel())
    unique_total = sum(int(parameter.numel()) for _, parameter in unique.values() if parameter.requires_grad)
    frozen_total = sum(int(parameter.numel()) for _, parameter in unique.values() if not parameter.requires_grad)
    active_applications = sum(int(parameter.numel()) for _, parameter in named_parameters if parameter.requires_grad)
    report = {
        "unique_parameter_ids": len(unique),
        "unique_trainable_parameters": unique_total,
        "unique_frozen_parameters": frozen_total,
        "active_parameter_applications": active_applications,
        "by_category": categories,
    }
    config = getattr(module, "config", None)
    if config is not None and hasattr(config, "num_loops"):
        stem = sum(int(p.numel()) for name, p in named_parameters if name.startswith("stem."))
        core = sum(int(p.numel()) for name, p in named_parameters if name.startswith("recurrent_core."))
        tail = sum(int(p.numel()) for name, p in named_parameters if name.startswith("tail."))
        routed = sum(int(p.numel()) for name, p in named_parameters if ".moe.experts." in name)
        shared = sum(int(p.numel()) for name, p in named_parameters if ".moe.shared." in name)
        core_non_routed = max(0, core - routed - shared)
        routed_active = routed * float(config.moe.routed_top_k) / max(1, int(config.moe.routed_experts))
        report["effective_block_applications"] = int(config.stem_blocks + config.recurrent_blocks * config.num_loops + config.tail_blocks)
        report["active_parameter_applications_per_token_estimate"] = int(
            self_numel(module, "embed_tokens") + stem + tail
            + int(config.num_loops) * (core_non_routed + shared + routed_active)
        )
    return report


def self_numel(module: Any, prefix: str) -> int:
    return sum(int(parameter.numel()) for name, parameter in module.named_parameters() if name.startswith(prefix))



# ===== inlined model component: recurrent_state.py =====
"""Explicit recurrent state carried through blocks and loop boundaries."""
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class MemoryState:
    """Causal compressed-attention memory with document identity metadata."""

    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    segment_ids: torch.Tensor | None = None
    latents: torch.Tensor | None = None
    # Incremental temporal-compression bucket.  A compressed entry is emitted
    # only when a complete ratio-sized block has arrived, matching the
    # sequence training path instead of treating every decode token as a
    # complete block.
    pending_sum: torch.Tensor | None = None
    pending_count: torch.Tensor | None = None
    pending_segment_ids: torch.Tensor | None = None
    # ``length`` is the active prefix in a decode cache. ``capacity`` is zero
    # for ordinary sequence states and positive for preallocated decode state.
    length: int | None = None
    capacity: int = 0
    static_cache: bool = False

    def detach(self) -> "MemoryState":
        return MemoryState(
            keys=None if self.keys is None else self.keys.detach(),
            values=None if self.values is None else self.values.detach(),
            positions=None if self.positions is None else self.positions.detach(),
            segment_ids=None if self.segment_ids is None else self.segment_ids.detach(),
            latents=None if self.latents is None else self.latents.detach(),
            pending_sum=None if self.pending_sum is None else self.pending_sum.detach(),
            pending_count=None if self.pending_count is None else self.pending_count.detach(),
            pending_segment_ids=(
                None if self.pending_segment_ids is None else self.pending_segment_ids.detach()
            ),
            length=self.length,
            capacity=self.capacity,
            static_cache=self.static_cache,
        )

    def reserve(self, capacity: int) -> "MemoryState":
        """Return a detached, writable decode cache with fixed capacity."""
        if self.keys is None or self.values is None or self.positions is None or self.segment_ids is None:
            return self
        active = self.keys.shape[1] if self.length is None else int(self.length)
        capacity = max(int(capacity), active)
        if self.capacity >= capacity:
            self.length = active
            return self
        keys = self.keys.new_zeros((self.keys.shape[0], capacity, self.keys.shape[2]))
        values = self.values.new_zeros((self.values.shape[0], capacity, self.values.shape[2]))
        positions = self.positions.new_full((self.positions.shape[0], capacity), -1)
        segment_ids = self.segment_ids.new_full((self.segment_ids.shape[0], capacity), -1)
        keys[:, :active] = self.keys[:, :active]
        values[:, :active] = self.values[:, :active]
        positions[:, :active] = self.positions[:, :active]
        segment_ids[:, :active] = self.segment_ids[:, :active]
        latents = None
        if self.latents is not None:
            latents = self.latents.new_zeros((self.latents.shape[0], capacity, self.latents.shape[2]))
            latents[:, :active] = self.latents[:, :active]
        self.keys, self.values = keys, values
        self.positions, self.segment_ids, self.latents = positions, segment_ids, latents
        self.length, self.capacity = active, capacity
        return self


@dataclass
class RecurrentState:
    """All state that may survive one physical loop execution.

    ``streams`` is never collapsed between passes.  Per-block dictionaries make
    state reuse explicit and make checkpoint serialization deterministic.
    """

    streams: torch.Tensor | None = None
    gdn_states: dict[str, torch.Tensor] = field(default_factory=dict)
    csa_memories: dict[str, MemoryState] = field(default_factory=dict)
    hca_memories: dict[str, MemoryState] = field(default_factory=dict)
    # Short causal histories used by the one-token decode path. The regular
    # sequence path does not need these, but retaining them avoids recomputing
    # the local convolution/attention window during generation.
    gdn_conv_histories: dict[str, torch.Tensor] = field(default_factory=dict)
    local_histories: dict[str, torch.Tensor] = field(default_factory=dict)
    local_history_valid: dict[str, torch.Tensor] = field(default_factory=dict)
    router_latent_histories: dict[str, torch.Tensor] = field(default_factory=dict)
    router_history_valid: dict[str, torch.Tensor] = field(default_factory=dict)
    loop_endpoints: list[torch.Tensor] = field(default_factory=list)
    token_positions: torch.Tensor | None = None
    graph_mode: bool = False

    def detach(self) -> "RecurrentState":
        return RecurrentState(
            streams=None if self.streams is None else self.streams.detach(),
            gdn_states={k: v.detach() for k, v in self.gdn_states.items()},
            csa_memories={k: v.detach() for k, v in self.csa_memories.items()},
            hca_memories={k: v.detach() for k, v in self.hca_memories.items()},
            gdn_conv_histories={k: v.detach() for k, v in self.gdn_conv_histories.items()},
            local_histories={k: v.detach() for k, v in self.local_histories.items()},
            local_history_valid={k: v.detach() for k, v in self.local_history_valid.items()},
            router_latent_histories={k: v.detach() for k, v in self.router_latent_histories.items()},
            router_history_valid={k: v.detach() for k, v in self.router_history_valid.items()},
            loop_endpoints=[v.detach() for v in self.loop_endpoints],
            token_positions=None if self.token_positions is None else self.token_positions.detach(),
            graph_mode=self.graph_mode,
        )

    @staticmethod
    def _select_tensor(value: torch.Tensor | None, rows: torch.Tensor) -> torch.Tensor | None:
        return None if value is None else value.index_select(0, rows)

    @classmethod
    def _select_memory(cls, memory: MemoryState, rows: torch.Tensor) -> MemoryState:
        return MemoryState(
            keys=cls._select_tensor(memory.keys, rows),
            values=cls._select_tensor(memory.values, rows),
            positions=cls._select_tensor(memory.positions, rows),
            segment_ids=cls._select_tensor(memory.segment_ids, rows),
            latents=cls._select_tensor(memory.latents, rows),
            pending_sum=cls._select_tensor(memory.pending_sum, rows),
            pending_count=cls._select_tensor(memory.pending_count, rows),
            pending_segment_ids=cls._select_tensor(memory.pending_segment_ids, rows),
            length=memory.length,
            capacity=memory.capacity,
            static_cache=memory.static_cache,
        )

    def select_rows(self, rows: torch.Tensor) -> "RecurrentState":
        """Select independent batch rows for token-wise decode early exit."""
        rows = rows.to(device=(self.streams.device if self.streams is not None else rows.device), dtype=torch.long)
        return RecurrentState(
            streams=self._select_tensor(self.streams, rows),
            gdn_states={k: self._select_tensor(v, rows) for k, v in self.gdn_states.items()},
            csa_memories={k: self._select_memory(v, rows) for k, v in self.csa_memories.items()},
            hca_memories={k: self._select_memory(v, rows) for k, v in self.hca_memories.items()},
            gdn_conv_histories={k: self._select_tensor(v, rows) for k, v in self.gdn_conv_histories.items()},
            local_histories={k: self._select_tensor(v, rows) for k, v in self.local_histories.items()},
            local_history_valid={k: self._select_tensor(v, rows) for k, v in self.local_history_valid.items()},
            router_latent_histories={k: self._select_tensor(v, rows) for k, v in self.router_latent_histories.items()},
            router_history_valid={k: self._select_tensor(v, rows) for k, v in self.router_history_valid.items()},
            loop_endpoints=[self._select_tensor(v, rows) for v in self.loop_endpoints],
            token_positions=self._select_tensor(self.token_positions, rows),
            graph_mode=self.graph_mode,
        )

    @staticmethod
    def _scatter_tensor(
        target: torch.Tensor | None, source: torch.Tensor | None, rows: torch.Tensor,
        batch_size: int, *, fill_value: float = 0.0,
    ) -> torch.Tensor | None:
        if source is None:
            return target
        if target is None or tuple(target.shape[1:]) != tuple(source.shape[1:]):
            shape = (batch_size, *source.shape[1:])
            resized = source.new_full(shape, fill_value)
            if target is not None and target.ndim == source.ndim:
                common = tuple(min(old, new) for old, new in zip(target.shape[1:], source.shape[1:]))
                if common:
                    target_slices = (slice(None),) + tuple(slice(0, size) for size in common)
                    resized[target_slices] = target[target_slices]
            target = resized
        # Some state tensors (notably token_positions after prefill) can be
        # expanded views with overlapping storage. Clone before row updates.
        target = target.clone()
        target.index_copy_(0, rows, source)
        return target

    @classmethod
    def _scatter_memory(
        cls, target: MemoryState | None, source: MemoryState, rows: torch.Tensor, batch_size: int,
    ) -> MemoryState:
        if target is None:
            target = MemoryState(length=source.length, capacity=source.capacity, static_cache=source.static_cache)
        for name in (
            "keys", "values", "positions", "segment_ids", "latents",
            "pending_sum", "pending_count", "pending_segment_ids",
        ):
            source_value = getattr(source, name)
            fill_value = -1 if name in {"positions", "segment_ids", "pending_segment_ids"} else 0.0
            setattr(target, name, cls._scatter_tensor(
                getattr(target, name), source_value, rows, batch_size, fill_value=fill_value,
            ))
        target.length = source.length
        target.capacity = source.capacity
        target.static_cache = source.static_cache
        return target

    def scatter_rows(self, rows: torch.Tensor, source: "RecurrentState") -> None:
        """Merge selected rows from a compact decode state back into this state."""
        rows = rows.to(device=(self.streams.device if self.streams is not None else rows.device), dtype=torch.long)
        batch_size = int(self.streams.shape[0]) if self.streams is not None else int(self.token_positions.shape[0])
        self.streams = self._scatter_tensor(self.streams, source.streams, rows, batch_size)
        for name in ("gdn_states", "gdn_conv_histories", "local_histories", "local_history_valid",
                     "router_latent_histories", "router_history_valid"):
            target = getattr(self, name)
            for key, value in getattr(source, name).items():
                target[key] = self._scatter_tensor(target.get(key), value, rows, batch_size)
        for name in ("csa_memories", "hca_memories"):
            target = getattr(self, name)
            for key, value in getattr(source, name).items():
                target[key] = self._scatter_memory(target.get(key), value, rows, batch_size)
        if source.token_positions is not None:
            self.token_positions = self._scatter_tensor(self.token_positions, source.token_positions, rows, batch_size)

    def reset_rows(self, reset_rows: torch.Tensor) -> None:
        """Reset packed-document rows before processing a new packed sample."""
        reset_rows = reset_rows.to(dtype=torch.bool)
        if self.streams is not None:
            self.streams[reset_rows] = 0
        for state in self.gdn_states.values():
            state[reset_rows] = 0
        for histories in (self.gdn_conv_histories, self.local_histories, self.router_latent_histories):
            for value in histories.values():
                value[reset_rows] = 0
        for valid_histories in (self.local_history_valid, self.router_history_valid):
            for value in valid_histories.values():
                value[reset_rows] = False
        for memory in (*self.csa_memories.values(), *self.hca_memories.values()):
            if memory.pending_sum is not None:
                memory.pending_sum[reset_rows] = 0
            if memory.pending_count is not None:
                memory.pending_count[reset_rows] = 0
            if memory.pending_segment_ids is not None:
                memory.pending_segment_ids[reset_rows] = -1
        for memory in (*self.csa_memories.values(), *self.hca_memories.values()):
            if memory.segment_ids is None:
                continue
            if bool(reset_rows.all()):
                memory.keys = memory.values = memory.positions = memory.segment_ids = memory.latents = None
                memory.pending_sum = memory.pending_count = memory.pending_segment_ids = None
                memory.length = memory.capacity = 0
                continue
            for name in ("keys", "values", "positions", "segment_ids", "latents"):
                value = getattr(memory, name)
                if value is not None and value.shape[0] == reset_rows.shape[0]:
                    value[reset_rows] = -1 if name in {"positions", "segment_ids"} else 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "streams": self.streams,
            "gdn_states": self.gdn_states,
            "csa_memories": {k: vars(v) for k, v in self.csa_memories.items()},
            "hca_memories": {k: vars(v) for k, v in self.hca_memories.items()},
            "gdn_conv_histories": self.gdn_conv_histories,
            "local_histories": self.local_histories,
            "local_history_valid": self.local_history_valid,
            "router_latent_histories": self.router_latent_histories,
            "router_history_valid": self.router_history_valid,
            "loop_endpoints": self.loop_endpoints,
            "token_positions": self.token_positions,
            "graph_mode": self.graph_mode,
        }

    @classmethod
    def from_state_dict(cls, value: dict[str, Any]) -> "RecurrentState":
        return cls(
            streams=value.get("streams"),
            gdn_states=dict(value.get("gdn_states", {})),
            csa_memories={k: MemoryState(**v) for k, v in value.get("csa_memories", {}).items()},
            hca_memories={k: MemoryState(**v) for k, v in value.get("hca_memories", {}).items()},
            gdn_conv_histories=dict(value.get("gdn_conv_histories", {})),
            local_histories=dict(value.get("local_histories", {})),
            local_history_valid=dict(value.get("local_history_valid", {})),
            router_latent_histories=dict(value.get("router_latent_histories", {})),
            router_history_valid=dict(value.get("router_history_valid", {})),
            loop_endpoints=list(value.get("loop_endpoints", [])),
            token_positions=value.get("token_positions"),
            graph_mode=bool(value.get("graph_mode", False)),
        )



# ===== inlined model component: norms.py =====
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        variance = value.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = value * torch.rsqrt(variance + self.eps).to(dtype=value.dtype)
        return normalized * self.weight.to(dtype=value.dtype)



# ===== inlined model component: mla_memory.py =====
"""Compact MLA-style channel memory used by the router and CSA indexer."""
import torch
import torch.nn as nn


class MLALatentMemory(nn.Module):
    """Project token-local context to a small latent channel.

    Temporal compression is intentionally separate from this channel
    compression.  The module only changes feature rank; callers decide which
    positions are causally visible.
    """

    def __init__(self, d_model: int, latent_dim: int = 256) -> None:
        super().__init__()
        self.down = nn.Linear(d_model, latent_dim, bias=False)
        self.up = nn.Linear(latent_dim, d_model, bias=False)
        self.norm = nn.LayerNorm(latent_dim)

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.norm(self.down(hidden))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.up(latent)

    def forward(self, hidden: torch.Tensor, *, local_context: int = 128) -> torch.Tensor:
        latent = self.encode(hidden)
        W = int(local_context)
        if W <= 1:
            return latent
        # Depthwise causal box filter: one fused convolution replaces the
        # cumulative-sum, gather, subtraction, and division sequence.
        x = latent.transpose(1, 2)
        channels = x.shape[1]
        padded = F.pad(x, (W - 1, 0))
        kernel = torch.ones(
            (channels, 1, W), device=x.device, dtype=x.dtype
        ) / float(W)
        context = F.conv1d(padded, kernel, groups=channels)
        positions = torch.arange(latent.shape[1], device=x.device)
        actual = torch.minimum(
            positions + 1,
            torch.full_like(positions, W),
        ).to(x.dtype)
        context = context * (float(W) / actual).view(1, 1, -1)
        return context.transpose(1, 2)


class LatentAttention(nn.Module):
    """Small multi-head attention over already causal latent memories."""

    def __init__(self, d_model: int, latent_dim: int, heads: int = 4, head_dim: int = 64) -> None:
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        width = self.heads * self.head_dim
        self.q = nn.Linear(d_model, width, bias=False)
        self.k = nn.Linear(latent_dim, width, bias=False)
        self.v = nn.Linear(latent_dim, width, bias=False)
        self.o = nn.Linear(width, d_model, bias=False)

    def forward(self, query: torch.Tensor, latents: torch.Tensor,
                *, causal: bool = True,
                key_valid: torch.Tensor | None = None) -> torch.Tensor:
        if latents.ndim != 3 or latents.shape[1] == 0:
            return torch.zeros_like(query)
        batch, length, _ = query.shape
        q = self.q(query).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(latents).view(batch, latents.shape[1], self.heads, self.head_dim).transpose(1, 2)
        v = self.v(latents).view(batch, latents.shape[1], self.heads, self.head_dim).transpose(1, 2)
        if key_valid is not None and key_valid.shape != (batch, latents.shape[1]):
            raise ValueError("key_valid must have shape [batch, latent_length]")
        if key_valid is None:
            mask = None
        else:
            mask = (~key_valid[:, None, None, :]).to(q.dtype) * torch.finfo(q.dtype).min
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            is_causal=(causal and length == latents.shape[1]),
            dropout_p=0.0,
        )
        out = out.transpose(1, 2).reshape(batch, length, -1)
        return self.o(out)



# ===== inlined model component: compressed_attention.py =====
"""Shared causal temporal-compression primitives for CSA and HCA."""
from dataclasses import dataclass

import torch
import torch.nn as nn



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


def update_graph_decode_cache(
    memory: MemoryState, current: CompressionResult, *, compression_ratio: int = 1,
) -> CompressionResult:
    """Update a fixed-size cache without depending on Python state changes.

    CUDA Graph replay repeats the captured tensor operations, but it does not
    rerun assignments to ``MemoryState.length``. Convert absolute token
    positions to compressor-block slots and keep invalid (incomplete bucket)
    writes from disturbing the existing cache contents.
    """
    capacity = int(memory.capacity)
    if capacity <= 0 or memory.keys is None:
        return current
    positions = current.positions[:, 0]
    valid = positions >= 0
    slots = torch.where(
        valid, (positions // max(1, int(compression_ratio))).remainder(capacity),
        torch.zeros_like(positions),
    )
    for batch_index in range(current.keys.shape[0]):
        slot = slots[batch_index:batch_index + 1]
        is_valid = valid[batch_index]
        for name in ("keys", "values", "positions", "segment_ids"):
            cache = getattr(memory, name)
            new_value = getattr(current, name)[batch_index]
            old_value = cache[batch_index].index_select(0, slot)
            mask = is_valid.view(1, *([1] * (new_value.ndim - 1)))
            cache[batch_index].index_copy_(
                0, slot, torch.where(mask, new_value, old_value)
            )
    memory.length = capacity
    return CompressionResult(
        memory.keys, memory.values, memory.positions, memory.segment_ids,
        memory.pending_sum, memory.pending_count, memory.pending_segment_ids,
    )


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
        result_pending_segments = token_segment
        if memory is not None:
            if memory.static_cache:
                # Graph replay cannot observe Python-side rebinding. Keep the
                # recurrent compression bucket at stable addresses so each
                # replay consumes the state written by the previous replay.
                memory.pending_sum.copy_(pending_sum)
                memory.pending_count.copy_(pending_count)
                memory.pending_segment_ids.copy_(token_segment)
                pending_sum = memory.pending_sum
                pending_count = memory.pending_count
                result_pending_segments = memory.pending_segment_ids
            else:
                memory.pending_sum = pending_sum
                memory.pending_count = pending_count
                memory.pending_segment_ids = token_segment
        return CompressionResult(
            keys, values, positions, emitted_segments,
            pending_sum, pending_count, result_pending_segments,
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



# ===== inlined model component: csa.py =====
"""Causal Selective Attention: local raw branch plus top-k remote summaries."""
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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
        if memory is not None and memory.keys is not None and memory.static_cache:
            compressed = update_graph_decode_cache(
                memory, current, compression_ratio=self.compressor.ratio,
            )
            compressed_keys = compressed.keys
            compressed_values = compressed.values
            compressed_positions = compressed.positions
            compressed_segments = compressed.segment_ids
        elif memory is not None and memory.keys is not None:
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

# ===== inlined model component: hca.py =====
"""Hierarchical Compressed Attention over dense coarse global summaries."""
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F



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
        if memory is not None and memory.keys is not None and memory.static_cache:
            compressed = update_graph_decode_cache(
                memory, current, compression_ratio=self.compressor.ratio,
            )
            compressed_keys = compressed.keys
            compressed_values = compressed.values
            compressed_positions = compressed.positions
            compressed_segments = compressed.segment_ids
        elif memory is not None and memory.keys is not None:
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



# ===== inlined model component: mhc.py =====
"""Input-dependent multi-stream hyper-connection (mHC)."""
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


_NATIVE_CUDA_LIBRARIES: dict[Path, Any] = {}
_NATIVE_CUDA_FUNCTIONS: dict[str, Any | None] = {}
_NATIVE_CUDA_DLL_DIRECTORY_HANDLES: list[Any] = []
_NATIVE_SINKHORN_CALL_COUNT = 0
_NATIVE_MOE_CALL_COUNT = 0


def _prepare_native_cuda_dll_search_path():
    """Keep the CUDA toolkit DLL directory available to Windows' loader."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    if _NATIVE_CUDA_DLL_DIRECTORY_HANDLES:
        return

    candidates = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")
    versioned_paths = sorted(
        (Path(value) / "bin" for key, value in os.environ.items()
         if key.startswith("CUDA_PATH_V") and value),
        key=lambda path: path.as_posix(),
        reverse=True,
    )
    candidates.extend(versioned_paths)
    cuda_root = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    cuda_root = cuda_root / "NVIDIA GPU Computing Toolkit" / "CUDA"
    if cuda_root.is_dir():
        versioned_directories = sorted(
            cuda_root.glob("v*/bin"),
            key=lambda path: path.as_posix(),
            reverse=True,
        )
        candidates.extend(versioned_directories)

    for directory in candidates:
        if directory.is_dir():
            _NATIVE_CUDA_DLL_DIRECTORY_HANDLES.append(
                os.add_dll_directory(str(directory))
            )
            return


def _native_cuda_library_candidates() -> list[Path]:
    explicit_path = os.environ.get("LLMY_MHC_CUDA_LIBRARY")
    candidates = [Path(explicit_path)] if explicit_path else []
    native_root = _REPO_ROOT / "llmy_cuda"
    discovered = []
    for pattern in ("*.dll", "*.so", "*.dylib"):
        discovered.extend(native_root.glob(f"build*/{pattern}"))
        discovered.extend(native_root.glob(f"build/*/{pattern}"))
    candidates.extend(sorted(discovered, key=lambda path: path.stat().st_mtime, reverse=True))
    return candidates


def _load_native_cuda_function(symbol: str) -> Any | None:
    """Load one exported function from the optional native CUDA library."""
    if symbol in _NATIVE_CUDA_FUNCTIONS:
        return _NATIVE_CUDA_FUNCTIONS[symbol]

    import ctypes

    signatures = {
        "llmy_mhc_sinkhorn4": [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
            ctypes.c_int32, ctypes.c_void_p,
        ],
        "llmy_moe_selected_fp32": [
            *([ctypes.c_void_p] * 8),
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_void_p,
        ],
    }
    try:
        argtypes = signatures[symbol]
    except KeyError as error:
        raise ValueError(f"Unsupported native CUDA function: {symbol}") from error

    for path in _native_cuda_library_candidates():
        if not path.is_file():
            continue
        try:
            _prepare_native_cuda_dll_search_path()
            library = _NATIVE_CUDA_LIBRARIES.get(path)
            if library is None:
                library = ctypes.CDLL(str(path))
                _NATIVE_CUDA_LIBRARIES[path] = library
            function = getattr(library, symbol)
        except (OSError, AttributeError):
            continue
        function.argtypes = argtypes
        function.restype = ctypes.c_int32
        _NATIVE_CUDA_FUNCTIONS[symbol] = function
        return function

    _NATIVE_CUDA_FUNCTIONS[symbol] = None
    return None


def _load_native_fused_sinkhorn():
    return _load_native_cuda_function("llmy_mhc_sinkhorn4")


def _load_native_fused_moe():
    return _load_native_cuda_function("llmy_moe_selected_fp32")


def sinkhorn(logits: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    """Sinkhorn normalization, with an optional one-launch CUDA inference path."""
    global _NATIVE_SINKHORN_CALL_COUNT
    use_native = (
        os.environ.get("LLMY_COMPILE_SINKHORN", "0") == "1"
        and logits.is_cuda
        and logits.dtype == torch.float32
        and not torch.is_grad_enabled()
        and not logits.requires_grad
        and logits.ndim >= 2
        and logits.shape[-2:] == (4, 4)
        and int(iterations) >= 0
    )
    if use_native:
        function = _load_native_fused_sinkhorn()
        if function is not None:
            import ctypes

            source = logits.contiguous()
            matrix = torch.empty_like(source)
            matrices = source.numel() // 16
            if matrices == 0:
                return matrix
            stream = torch.cuda.current_stream(source.device).cuda_stream
            status = function(
                ctypes.c_void_p(source.data_ptr()),
                ctypes.c_void_p(matrix.data_ptr()),
                matrices,
                int(iterations),
                ctypes.c_void_p(stream),
            )
            if status != 0:
                raise RuntimeError(
                    f"Fused CUDA Sinkhorn launch failed with CUDA status {status}"
                )
            _NATIVE_SINKHORN_CALL_COUNT += 1
            return matrix

    matrix = logits.float()
    for _ in range(int(iterations)):
        matrix = matrix - torch.logsumexp(matrix, dim=-1, keepdim=True)
        matrix = matrix - torch.logsumexp(matrix, dim=-2, keepdim=True)
    return matrix.exp().to(dtype=logits.dtype)


@dataclass
class MHCMaps:
    pre: torch.Tensor
    post: torch.Tensor
    residual: torch.Tensor


class MultiStreamHyperConnection(nn.Module):
    """Maintain live residual streams around a shared sublayer.

    The static logits remain as checkpoint-compatible initialization biases.
    The actual pre, post, and residual maps are input-dependent for every
    token, with bounded sigmoid/Sinkhorn parameterizations.
    """

    def __init__(self, d_model: int, streams: int = 4, sinkhorn_iterations: int = 10,
                 init_gating_factor: float = 0.01) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.streams = int(streams)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        # Keep the old names so old checkpoints load their learned static
        # initialization. New dynamic parameters are warm-started by the
        # trainer's shape-aware loader.
        self.read_logits = nn.Parameter(torch.eye(streams) * 4.0)
        self.write_logits = nn.Parameter(torch.eye(streams) * 4.0)
        self.stream_scale = nn.Parameter(torch.ones(streams, d_model))
        self.stream_norm = nn.LayerNorm(d_model)
        self.mapping_norm = nn.RMSNorm(streams * d_model)
        # One fused projection computes the three input-dependent mappings.
        theta_out = streams + streams + streams * streams
        self.theta = nn.Linear(streams * d_model, theta_out, bias=False)
        self.pre_bias = nn.Parameter(torch.full(
            (streams,), float(torch.logit(torch.tensor(1.0 / max(streams, 1))))
        ))
        self.post_bias = nn.Parameter(torch.zeros(streams))
        self.residual_bias = nn.Parameter(torch.eye(streams) * 4.0)
        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_residual = nn.Parameter(torch.tensor(0.01))
        init = torch.logit(torch.tensor(float(init_gating_factor)).clamp(1e-5, 1 - 1e-5))
        self.gating_logit = nn.Parameter(init)
        self.use_triton_kernels = False
        self._inference_matrices: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def gating_factor(self) -> torch.Tensor:
        return torch.sigmoid(self.gating_logit)

    def matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return static compatibility matrices for diagnostics/tools."""
        if not self.training and self._inference_matrices is not None:
            read, write = self._inference_matrices
            if read.device == self.read_logits.device and read.dtype == self.read_logits.dtype:
                return read, write
        matrices = (
            sinkhorn(self.read_logits, self.sinkhorn_iterations),
            sinkhorn(self.write_logits, self.sinkhorn_iterations),
        )
        if not self.training:
            self._inference_matrices = matrices
        return matrices

    def eval(self) -> "MultiStreamHyperConnection":
        super().eval()
        self._inference_matrices = (
            sinkhorn(self.read_logits, self.sinkhorn_iterations),
            sinkhorn(self.write_logits, self.sinkhorn_iterations),
        )
        return self

    def train(self, mode: bool = True) -> "MultiStreamHyperConnection":
        super().train(mode)
        self._inference_matrices = None
        if not mode:
            self._inference_matrices = (
                sinkhorn(self.read_logits, self.sinkhorn_iterations),
                sinkhorn(self.write_logits, self.sinkhorn_iterations),
            )
        return self

    def compute_mappings(self, streams: torch.Tensor) -> MHCMaps:
        if streams.ndim != 4 or streams.shape[-2] != self.streams:
            raise ValueError(f"expected [batch, time, {self.streams}, hidden] streams")
        flat = self.mapping_norm(streams.reshape(streams.shape[0], streams.shape[1], -1))
        all_out = self.theta(flat)
        s = self.streams
        pre_raw, post_raw, residual_raw = all_out.split([s, s, s * s], dim=-1)
        pre = torch.sigmoid(self.alpha_pre * pre_raw + self.pre_bias)
        post = 2.0 * torch.sigmoid(self.alpha_post * post_raw + self.post_bias)
        residual_logits = residual_raw.view(
            streams.shape[0], streams.shape[1], s, s
        )
        residual = sinkhorn(
            self.alpha_residual * residual_logits + self.residual_bias,
            self.sinkhorn_iterations,
        )
        return MHCMaps(pre=pre, post=post, residual=residual)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Migrate checkpoints containing the three legacy theta projections."""
        pre_k = prefix + "pre_theta.weight"
        post_k = prefix + "post_theta.weight"
        res_k = prefix + "residual_theta.weight"
        theta_k = prefix + "theta.weight"
        if theta_k not in state_dict and all(
            key in state_dict for key in (pre_k, post_k, res_k)
        ):
            s = self.streams
            state_dict[theta_k] = torch.cat([
                state_dict[pre_k],
                state_dict[post_k],
                state_dict[res_k].reshape(s * s, -1),
            ], dim=0)
        for key in (pre_k, post_k, res_k):
            state_dict.pop(key, None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def read(self, streams: torch.Tensor, mappings: MHCMaps | None = None) -> torch.Tensor:
        if mappings is None:
            mappings = self.compute_mappings(streams)
        if self.use_triton_kernels and streams.is_cuda:
            return triton_mhc_dynamic_read(streams, mappings.pre, self.stream_norm)
        value = torch.einsum("bts,btsd->btd", mappings.pre, streams)
        norm_dtype = self.stream_norm.weight.dtype if self.stream_norm.weight is not None else value.dtype
        return self.stream_norm(value.to(dtype=norm_dtype)).to(dtype=value.dtype)

    def write(self, streams: torch.Tensor, update: torch.Tensor,
              mappings: MHCMaps | None = None) -> torch.Tensor:
        if mappings is None:
            mappings = self.compute_mappings(streams)
        if self.use_triton_kernels and streams.is_cuda:
            return triton_mhc_dynamic_write(
                streams, update, mappings.residual, mappings.post,
                self.stream_scale, self.gating_factor,
            )
        residual = torch.einsum("btij,btjd->btid", mappings.residual, streams)
        stream_update = residual + torch.einsum(
            "bts,btd->btsd", mappings.post.to(dtype=update.dtype), update
        )
        stream_update = stream_update * self.stream_scale.to(dtype=stream_update.dtype).view(
            1, 1, self.streams, -1
        )
        return streams + self.gating_factor.to(dtype=streams.dtype) * stream_update

    def forward(self, streams: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        mappings = self.compute_mappings(streams)
        return self.write(streams, sublayer(self.read(streams, mappings)), mappings)

    def constraint_errors(self) -> dict[str, float]:
        read, write = self.matrices()
        ones = torch.ones(self.streams, device=read.device, dtype=read.dtype)
        return {
            "mhc_read_row_error": float((read @ ones - ones).abs().max().detach()),
            "mhc_read_col_error": float((read.transpose(0, 1) @ ones - ones).abs().max().detach()),
            "mhc_write_row_error": float((write @ ones - ones).abs().max().detach()),
            "mhc_write_col_error": float((write.transpose(0, 1) @ ones - ones).abs().max().detach()),
            "mhc_gate": float(self.gating_factor.detach()),
            "mhc_dynamic": 1.0,
        }

# ===== inlined model component: dense_ffn.py =====
import torch
import torch.nn as nn


class DenseSwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden))



# ===== inlined model component: expert.py =====
import torch
import torch.nn as nn


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden))



# ===== inlined model component: context_router.py =====
"""Context-aware top-k expert router with a hidden-only fallback."""
from typing import Any

import torch
import torch.nn as nn



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



# ===== inlined model component: moe.py =====
"""Slow, transparent V0 MoE dispatch with 128 routed and 2 shared experts."""
import os

from typing import Any

import torch
import torch.nn as nn

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


def _moe_dispatch_single_token(flat: torch.Tensor, indices: torch.Tensor,
                               probabilities: torch.Tensor,
                               gate_up_weights: torch.Tensor,
                               w_down: torch.Tensor) -> torch.Tensor:
    """Run only selected experts for one inference token, without capacity padding."""
    global _NATIVE_MOE_CALL_COUNT
    use_native = (
        os.environ.get("LLMY_COMPILE_MOE", "0") == "1"
        and flat.is_cuda
        and flat.dtype == torch.float32
        and flat.is_contiguous()
        and indices.dtype == torch.int64
        and indices.is_contiguous()
        and probabilities.dtype == torch.float32
        and probabilities.is_contiguous()
        and gate_up_weights.dtype == torch.float32
        and gate_up_weights.is_contiguous()
        and w_down.dtype == torch.float32
        and w_down.is_contiguous()
        and not torch.is_grad_enabled()
    )
    if use_native:
        function = _load_native_fused_moe()
        if function is not None:
            import ctypes

            routes = int(indices.shape[1])
            experts, packed_hidden, hidden = gate_up_weights.shape
            expert_hidden = packed_hidden // 2
            if (
                packed_hidden == 2 * expert_hidden
                and flat.shape == (1, hidden)
                and w_down.shape == (experts, hidden, expert_hidden)
                and probabilities.shape == indices.shape
            ):
                activation = torch.empty(
                    (routes, expert_hidden), device=flat.device, dtype=flat.dtype
                )
                route_output = torch.empty(
                    (routes, hidden), device=flat.device, dtype=flat.dtype
                )
                output = torch.empty_like(flat)
                stream = torch.cuda.current_stream(flat.device).cuda_stream
                status = function(
                    ctypes.c_void_p(flat.data_ptr()),
                    ctypes.c_void_p(indices.data_ptr()),
                    ctypes.c_void_p(probabilities.data_ptr()),
                    ctypes.c_void_p(gate_up_weights.data_ptr()),
                    ctypes.c_void_p(w_down.data_ptr()),
                    ctypes.c_void_p(activation.data_ptr()),
                    ctypes.c_void_p(route_output.data_ptr()),
                    ctypes.c_void_p(output.data_ptr()),
                    hidden,
                    expert_hidden,
                    experts,
                    routes,
                    ctypes.c_void_p(stream),
                )
                if status != 0:
                    raise RuntimeError(
                        f"Fused CUDA selected-expert MoE launch failed with CUDA status {status}"
                    )
                _NATIVE_MOE_CALL_COUNT += 1
                return output

    route_ids = indices.reshape(-1).long()
    route_probs = probabilities.reshape(-1).to(dtype=flat.dtype)
    selected_gate_up = gate_up_weights.index_select(0, route_ids)
    selected_down = w_down.index_select(0, route_ids)
    token_rows = flat.expand(route_ids.numel(), -1).contiguous()

    gate_up = torch.bmm(selected_gate_up, token_rows.unsqueeze(-1)).squeeze(-1)
    expert_hidden = gate_up.shape[-1] // 2
    activated = torch.nn.functional.silu(gate_up[:, :expert_hidden]) * gate_up[:, expert_hidden:]
    expert_output = torch.bmm(selected_down, activated.unsqueeze(-1)).squeeze(-1)
    return (expert_output * route_probs.unsqueeze(-1)).sum(dim=0, keepdim=True)

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
        if (
            flat.is_cuda
            and flat.shape[0] == 1
            and not self.training
            and not torch.is_grad_enabled()
        ):
            w_down = self._weights(
                self.experts, device=flat.device, dtype=flat.dtype
            )[2]
            gate_up_weights = self._gate_up_weights(
                self.experts, device=flat.device, dtype=flat.dtype
            )
            return _moe_dispatch_single_token(
                flat, indices, probabilities, gate_up_weights, w_down
            )

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

# ===== inlined model component: gated_deltanet.py =====
"""Reference Gated DeltaNet mixer with a chunkable recurrent scan."""
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

_NUMBA_AVAILABLE = False
_fused_gdn_scan = None
_fused_gdn_step = None
_triton_gdn_step = triton_gdn_step
_triton_gdn_scan = triton_gdn_scan
_triton_gdn_available = _GDN_TRITON_AVAILABLE


def _scan_reference(step, initial, xs):
    """Small sequential oracle used by correctness tests and fallback runtimes."""
    carry = initial
    outputs = []
    for index in range(xs[0].shape[0]):
        carry, output = step(carry, tuple(x[index] for x in xs))
        outputs.append(output)
    return carry, torch.stack(outputs, dim=0)


def _scan(step, initial, xs, *, reference: bool = False):
    return _scan_reference(step, initial, xs)


def _causal_conv(hidden: torch.Tensor, conv: nn.Conv1d,
                 reset_mask: torch.Tensor | None) -> torch.Tensor:
    """Apply the causal depthwise gate without crossing reset boundaries."""
    kernel = int(conv.kernel_size[0])
    needed = kernel - 1
    if needed <= 0:
        return conv(hidden.transpose(1, 2))
    if reset_mask is None or not bool(reset_mask[:, 1:].any()):
        return conv(F.pad(hidden.transpose(1, 2), (needed, 0)))

    x = hidden.transpose(1, 2)
    padded = F.pad(x, (needed, 0))
    segment_numbers = reset_mask.to(dtype=torch.long).cumsum(dim=1)
    padded_segments = F.pad(segment_numbers, (needed, 0), value=-1)
    out = x.new_zeros((x.shape[0], conv.out_channels, x.shape[-1]))
    weight = conv.weight[:, 0, :]
    for j in range(kernel):
        seg_tap = padded_segments[:, j:j + x.shape[-1]]
        valid = (seg_tap == segment_numbers).to(dtype=x.dtype).unsqueeze(1)
        out = out + padded[:, :, j:j + x.shape[-1]] * weight[:, j].view(1, -1, 1) * valid
    return out


class GatedDeltaNet(nn.Module):
    """Causal gated delta-rule mixer.

    The production path uses the framework scan primitive when available, so
    training does not create a Python token loop.  ``reference=True`` remains
    available for short-sequence parity tests.
    """

    def __init__(self, d_model: int, qk_heads: int = 16, value_heads: int = 16,
                 qk_head_dim: int = 128, value_head_dim: int = 128,
                 conv_kernel: int = 4, sensitive_state_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        if qk_heads != value_heads:
            raise ValueError("this implementation requires equal q/k and value head counts")
        self.d_model = int(d_model)
        self.heads = int(qk_heads)
        self.key_dim = int(qk_head_dim)
        self.value_dim = int(value_head_dim)
        self.state_dtype = sensitive_state_dtype
        # Full-scan activation checkpointing is the safe default on 24 GiB
        # cards; selected blocks can opt out when there is measured headroom.
        self.checkpoint_scan = True
        width_qk = self.heads * self.key_dim
        width_v = self.heads * self.value_dim
        self.q_proj = nn.Linear(d_model, width_qk, bias=False)
        self.k_proj = nn.Linear(d_model, width_qk, bias=False)
        self.v_proj = nn.Linear(d_model, width_v, bias=False)
        self.beta_proj = nn.Linear(d_model, self.heads, bias=True)
        self.gate_proj = nn.Linear(d_model, width_v, bias=False)
        self.out_proj = nn.Linear(width_v, d_model, bias=False)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel, groups=d_model,
                              padding=0, bias=False)
        self.decay_logit = nn.Parameter(torch.full((self.heads,), 2.0))

    def initial_state(self, batch: int, *, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
        return torch.zeros(batch, self.heads, self.key_dim, self.value_dim,
                           device=device, dtype=dtype or self.state_dtype)

    def forward(self, hidden: torch.Tensor, state: torch.Tensor | None = None,
                reset_mask: torch.Tensor | None = None, *, reference: bool = False,
                conv_history: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.ndim != 3:
            raise ValueError("GatedDeltaNet expects [batch, time, hidden]")
        batch, length, _ = hidden.shape
        q = self.q_proj(hidden).view(batch, length, self.heads, self.key_dim)
        k = self.k_proj(hidden).view(batch, length, self.heads, self.key_dim)
        v = self.v_proj(hidden).view(batch, length, self.heads, self.value_dim)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True).clamp_min(1e-12))
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True).clamp_min(1e-12))
        beta = torch.sigmoid(self.beta_proj(hidden)).unsqueeze(-1)
        gate = torch.sigmoid(self.gate_proj(hidden)).view(batch, length, self.heads, self.value_dim)
        decay = torch.sigmoid(self.decay_logit).view(1, self.heads, 1, 1).to(self.state_dtype)
        if state is None:
            state = self.initial_state(batch, device=hidden.device)
        state = state.to(device=hidden.device, dtype=self.state_dtype)
        if tuple(state.shape) != (batch, self.heads, self.key_dim, self.value_dim):
            raise ValueError("invalid GatedDeltaNet recurrent state shape")
        reset = None if reset_mask is None else reset_mask.to(device=hidden.device, dtype=torch.bool)

        def step(carry: torch.Tensor, values: tuple[torch.Tensor, ...]):
            q_t, k_t, v_t, beta_t, gate_t, reset_t = values
            carry = torch.where(reset_t[:, None, None, None], torch.zeros_like(carry), carry)
            k_state = k_t.to(carry.dtype)
            v_state = v_t.to(carry.dtype)
            prediction = torch.einsum("bhk,bhkv->bhv", k_state, carry)
            delta = v_state - prediction
            updated = decay * carry + beta_t.unsqueeze(-1).to(carry.dtype) * torch.einsum("bhk,bhv->bhkv", k_state, delta)
            out = torch.einsum("bhk,bhkv->bhv", q_t.to(updated.dtype), updated)
            return updated, out.to(dtype=hidden.dtype) * gate_t

        if reset is None:
            reset = torch.zeros(batch, length, device=hidden.device, dtype=torch.bool)

        # CUDA Graph capture cannot reliably retain the external Numba launch
        # used by the training scan. Keep a graph-safe one-token fallback that
        # updates the recurrent buffer in place. The normal sampler still uses
        # the faster custom kernel when it is not being captured.
        if (
            hidden.is_cuda and length == 1 and not reference
            and _fused_gdn_step is None and not _triton_gdn_available
        ):
            reset_one = reset[:, 0]
            if reset_mask is not None and bool(reset_one.any()):
                state[reset_one] = 0
            k_state = k[:, 0].to(state.dtype)
            v_state = v[:, 0].to(state.dtype)
            prediction = torch.einsum("bhk,bhkv->bhv", k_state, state)
            delta = v_state - prediction
            beta_one = beta[:, 0].squeeze(-1).to(state.dtype).unsqueeze(-1).unsqueeze(-1)
            updated = decay * state + beta_one * torch.einsum(
                "bhk,bhv->bhkv", k_state, delta
            )
            state.copy_(updated)
            final_state = state
            output = torch.einsum("bhk,bhkv->bhv", q[:, 0].to(state.dtype), state)
            output = output.to(dtype=hidden.dtype) * gate[:, 0]
            output = output.reshape(batch, length, self.heads * self.value_dim)
        # --- Fast path: Fused On-Chip CUDA Shared Memory Scan ---
        elif (
            _triton_gdn_available and hidden.is_cuda and length == 1
            and self.key_dim in (64, 128) and self.value_dim in (64, 128)
            and _triton_gdn_step is not None
        ):
            output, final_state = _triton_gdn_step(
                q, k, v, beta, gate, reset, self.decay_logit, state
            )
            output = output.reshape(batch, length, self.heads * self.value_dim)
        elif (
            _NUMBA_AVAILABLE and not reference and hidden.is_cuda
            and length == 1 and self.key_dim in (64, 128) and self.value_dim in (64, 128)
            and _fused_gdn_step is not None
        ):
            output, final_state = _fused_gdn_step(q, k, v, beta, gate, reset, self.decay_logit, state)
            output = output.reshape(batch, length, self.heads * self.value_dim)
        elif _NUMBA_AVAILABLE and not reference and hidden.is_cuda and _fused_gdn_scan is not None:
            # Checkpoint the fused scan so its large per-token recurrent history
            # is recomputed during backward rather than retained per GDN block.
            if self.checkpoint_scan and self.training and torch.is_grad_enabled():
                def _checkpointed_scan(q_, k_, v_, beta_, gate_, reset_, decay_, state_):
                    return _fused_gdn_scan(
                        q_, k_, v_, beta_, gate_, reset_, decay_, state_
                    )
                output, final_state = activation_checkpoint(
                    _checkpointed_scan,
                    q, k, v, beta, gate, reset, self.decay_logit, state,
                    use_reentrant=False,
                )
            else:
                output, final_state = _fused_gdn_scan(
                    q, k, v, beta, gate, reset, self.decay_logit, state
                )
            output = output.reshape(batch, length, self.heads * self.value_dim)
        elif (
            _triton_gdn_available and not reference and hidden.is_cuda
            and self.key_dim in (64, 128) and self.value_dim in (64, 128)
            and _triton_gdn_scan is not None
        ):
            if self.checkpoint_scan and self.training and torch.is_grad_enabled():
                def _checkpointed_triton_scan(q_, k_, v_, beta_, gate_, reset_, decay_, state_):
                    return _triton_gdn_scan(
                        q_, k_, v_, beta_, gate_, reset_, decay_, state_
                    )
                output, final_state = activation_checkpoint(
                    _checkpointed_triton_scan,
                    q, k, v, beta, gate, reset, self.decay_logit, state,
                    use_reentrant=False,
                )
            else:
                output, final_state = _triton_gdn_scan(
                    q, k, v, beta, gate, reset, self.decay_logit, state
                )
            output = output.reshape(batch, length, self.heads * self.value_dim)
        else:
            # --- Fallback: PyTorch reference scan ---
            def step(carry: torch.Tensor, values: tuple[torch.Tensor, ...]):
                q_t, k_t, v_t, beta_t, gate_t, reset_t = values
                carry = torch.where(reset_t[:, None, None, None], torch.zeros_like(carry), carry)
                k_state = k_t.to(carry.dtype)
                v_state = v_t.to(carry.dtype)
                prediction = torch.einsum("bhk,bhkv->bhv", k_state, carry)
                delta = v_state - prediction
                updated = decay * carry + beta_t.unsqueeze(-1).to(carry.dtype) * torch.einsum("bhk,bhv->bhkv", k_state, delta)
                out = torch.einsum("bhk,bhkv->bhv", q_t.to(updated.dtype), updated)
                return updated, out.to(dtype=hidden.dtype) * gate_t

            scan_inputs = tuple(value.transpose(0, 1) for value in (q, k, v, beta, gate, reset))
            final_state, output = _scan(step, state, scan_inputs, reference=reference)
            output = output.transpose(0, 1)
            output = output.reshape(batch, length, self.heads * self.value_dim)

        # Causal depthwise conv gate (local temporal mixing)
        if conv_history is None:
            conv_gate = _causal_conv(hidden, self.conv, reset)
        else:
            history = conv_history.transpose(1, 2)
            needed = self.conv.kernel_size[0] - 1
            if history.shape[-1] < needed:
                history = torch.nn.functional.pad(history, (needed - history.shape[-1], 0))
            if reset is not None:
                # Keep reset handling tensorized so CUDA Graph capture never
                # asks the host to inspect a device boolean.
                reset_rows = reset[:, 0].view(batch, 1, 1)
                history = torch.where(reset_rows, torch.zeros_like(history), history)
            conv_input = torch.cat((history[..., -needed:], hidden.transpose(1, 2)), dim=-1)
            conv_gate = self.conv(conv_input)
        conv_gate = conv_gate.transpose(1, 2)
        # The recurrent scan emits heads * value_dim, while the depthwise
        # convolution gate is model-width. Project before applying the gate;
        # multiplying the two widths directly fails for normal configs where
        # heads * value_dim != d_model.
        output = self.out_proj(output)
        output = output * torch.sigmoid(conv_gate.to(output.dtype))
        return output, final_state



# ===== inlined model component: loop_memory.py =====
"""Compressed attention residuals between variable-depth recurrent loops."""
import torch
import torch.nn as nn



class LoopMemoryHook(nn.Module):
    def __init__(
        self,
        d_model: int,
        latent_dim: int,
        *,
        enabled: bool = False,
        heads: int = 4,
        head_dim: int = 32,
        residual_gate_init: float = 0.05,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.compressor = MLALatentMemory(d_model, latent_dim)
        self.query_norm = nn.LayerNorm(d_model)
        self.attention = LatentAttention(
            d_model, latent_dim, heads=heads, head_dim=head_dim
        )
        gate = torch.tensor(float(residual_gate_init)).clamp(1e-5, 1 - 1e-5)
        self.gating_logit = nn.Parameter(torch.logit(gate))

    @property
    def gating_factor(self) -> torch.Tensor:
        return torch.sigmoid(self.gating_logit)

    def encode(self, streams: torch.Tensor) -> torch.Tensor:
        """Compress one loop endpoint, preserving gradients for loop AttnRes."""
        return self.compressor.encode(streams.mean(dim=-2))

    def inject(self, streams: torch.Tensor, endpoints: list[torch.Tensor]) -> torch.Tensor:
        """Attend over prior loop endpoints at each token position."""
        if not self.enabled or not endpoints:
            return streams
        batch, length, stream_count, d_model = streams.shape
        memory = torch.stack(endpoints, dim=2)  # [B, T, prior_loops, latent]
        query = self.query_norm(streams.mean(dim=-2)).reshape(batch * length, 1, d_model)
        memory = memory.reshape(batch * length, len(endpoints), memory.shape[-1])
        attended = self.attention(query, memory, causal=False)
        attended = attended.reshape(batch, length, 1, d_model)
        return streams + self.gating_factor.to(dtype=streams.dtype) * attended.expand(
            -1, -1, stream_count, -1
        )

    def record(
        self,
        streams: torch.Tensor,
        state: RecurrentState,
        *,
        force: bool = False,
    ) -> torch.Tensor | None:
        if self.enabled or force:
            endpoint = self.encode(streams)
            # Keep the serialized/telemetry state detached, while returning the
            # live tensor so the next loop retains a gradient path.
            state.loop_endpoints.append(endpoint.detach())
            return endpoint
        return None



# ===== inlined model component: model.py =====
"""The V0 2B looped hybrid-MoE language model."""
import copy
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint



_ROUTER_HEALTH_KEYS = (
    "route_entropy", "router_balance_loss", "moe_aux_loss",
    "active_experts", "dead_experts", "expert_load_entropy",
    "max_expert_fraction", "mean_selected_probability",
)

def _checkpointed_vocab_ce(
    hidden_chunk: torch.Tensor,
    target_chunk: torch.Tensor,
    embed_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute one vocabulary CE chunk without retaining its logits graph."""
    projected = F.linear(hidden_chunk, embed_weight).float()
    return F.cross_entropy(projected, target_chunk, reduction="sum")


def _chunk_ce_eager(
    hidden_chunk: torch.Tensor,
    target_chunk: torch.Tensor,
    embed_weight: torch.Tensor,
) -> torch.Tensor:
    logits = F.linear(hidden_chunk, embed_weight).float()
    return F.cross_entropy(logits, target_chunk, reduction="sum")


_chunk_ce_compiled = None


def _chunk_ce(hidden_chunk: torch.Tensor, target_chunk: torch.Tensor,
              embed_weight: torch.Tensor) -> torch.Tensor:
    global _chunk_ce_compiled
    if hidden_chunk.is_cuda and os.getenv("LLMY_ENABLE_COMPILED_CE", "0") == "1":
        if _chunk_ce_compiled is None and hasattr(torch, "compile"):
            _chunk_ce_compiled = torch.compile(_chunk_ce_eager, dynamic=True)
        if _chunk_ce_compiled is not None:
            return _chunk_ce_compiled(hidden_chunk, target_chunk, embed_weight)
    return _chunk_ce_eager(hidden_chunk, target_chunk, embed_weight)


@dataclass
class LoopedHybridOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    state: RecurrentState
    auxiliary_loss: torch.Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


class PhysicalBlock(nn.Module):
    """One physical block; the same instance is called on every loop pass."""

    def __init__(self, cfg: V6Config, kind: str, *, block_name: str, with_moe: bool) -> None:
        super().__init__()
        self.kind = kind
        self.block_name = block_name
        self.with_moe = bool(with_moe)
        self.mhc = MultiStreamHyperConnection(
            cfg.d_model, cfg.mhc.streams, cfg.mhc.sinkhorn_iterations, cfg.mhc.init_gating_factor
        ) if cfg.mhc.enabled else None
        self.mixer_norm = RMSNorm(cfg.d_model)
        self.ffn_norm = RMSNorm(cfg.d_model)
        if kind == "gdn":
            self.mixer = GatedDeltaNet(
                cfg.d_model, cfg.gdn.qk_heads, cfg.gdn.value_heads,
                cfg.gdn.qk_head_dim, cfg.gdn.value_head_dim, cfg.gdn.conv_kernel,
                torch.float32 if cfg.gdn.sensitive_state_dtype.lower() in {"float32", "fp32"} else torch.float16,
            )
        elif kind == "csa":
            self.mixer = CompressedSelectiveAttention(
                cfg.d_model, compression=cfg.csa.temporal_compression,
                local_window=cfg.csa.local_window, index_heads=cfg.csa.index_heads,
                index_head_dim=cfg.csa.index_head_dim, top_k=cfg.csa.index_top_k,
            )
        elif kind == "hca":
            self.mixer = HierarchicalCompressedAttention(
                cfg.d_model, compression=cfg.hca.temporal_compression,
                local_window=cfg.hca.local_window, dense_global_attention=cfg.hca.dense_global_attention,
            )
        else:
            raise ValueError(f"unknown V0 mixer kind: {kind}")
        self.moe = SparseMoE(
            cfg.d_model, routed_experts=cfg.moe.routed_experts,
            shared_experts=cfg.moe.shared_experts, top_k=cfg.moe.routed_top_k,
            expert_hidden=cfg.moe.expert_hidden, latent_dim=cfg.mla.latent_dim,
            context_aware=cfg.router.context_aware, query_heads=cfg.router.query_heads,
            query_head_dim=cfg.router.query_head_dim, local_context=cfg.router.local_context,
            aux_balance_coef=cfg.moe.aux_balance_coef,
        ) if with_moe else None
        self.ffn = DenseSwiGLU(cfg.d_model, cfg.dense_ffn.hidden) if not with_moe else None
        self.hca_latent = MLALatentMemory(cfg.d_model, cfg.mla.latent_dim) if kind == "hca" else None

    def _read(self, streams: torch.Tensor):
        if self.mhc is None:
            return streams.mean(dim=-2), None
        mappings = self.mhc.compute_mappings(streams)
        return self.mhc.read(streams, mappings), mappings

    def _write(self, streams: torch.Tensor, update: torch.Tensor, mappings=None) -> torch.Tensor:
        return self.mhc.write(streams, update, mappings) if self.mhc is not None else streams + update.unsqueeze(-2)

    @staticmethod
    def _latest_hca_latents(
        state: RecurrentState,
        state_namespace: str | None,
    ) -> torch.Tensor | None:
        prefix = None if state_namespace is None else f"{state_namespace}."
        latest = None
        for key, memory in state.hca_memories.items():
            if prefix is None:
                if key.startswith("loop"):
                    continue
            elif not key.startswith(prefix):
                continue
            if memory.latents is not None:
                latest = memory.latents
        return latest

    def _moe_checkpointed(self, hidden: torch.Tensor, hca_latents: torch.Tensor | None,
                          collect_router_diagnostics: bool = False):
        """Recompute the pure MoE path in backward instead of retaining dispatch buffers.

        Training-time MoE dispatch stacks every expert's weights and creates large
        grouped-routing intermediates.  Those tensors are safe to recompute because
        the MoE path has no recurrent-state side effects; retaining them across both
        recurrent loops was the real activation peak, not the small mHC parameter set.
        """
        output, aux, metrics = self.moe(
            hidden, hca_global_latents=hca_latents, return_stats=False,
            return_health=collect_router_diagnostics,
        )
        if collect_router_diagnostics:
            self.moe._last_router_health = metrics
        return output, aux

    def forward(self, streams: torch.Tensor, state: RecurrentState, *, segment_ids: torch.Tensor | None,
                reset_mask: torch.Tensor | None, diagnostics: dict[str, Any],
                collect_diagnostics: bool = True,
                collect_router_diagnostics: bool = False,
                position: torch.Tensor | int | None = None,
                state_namespace: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        state_key = (
            f"{state_namespace}.{self.block_name}"
            if state_namespace else self.block_name
        )
        hidden, mixer_mappings = self._read(streams)
        normed = self.mixer_norm(hidden)
        if self.kind == "gdn":
            prior = state.gdn_states.get(state_key)
            # The fused GDN scan performs its own activation checkpointing.
            # Wrapping the entire mixer again makes stateful recomputation
            # invalid when truncated BPTT changes the grad boundary.
            mixed, next_state = self.mixer(normed, prior, reset_mask)
            state.gdn_states[state_key] = next_state.detach()
            history_size = self.mixer.conv.kernel_size[0] - 1
            state.gdn_conv_histories[state_key] = normed[:, -history_size:].detach()
        else:
            prior = state.csa_memories.get(state_key) if self.kind == "csa" else state.hca_memories.get(state_key)
            # CSA's remote-value tiles are checkpointed internally. Avoid an
            # outer stateful checkpoint here; it can replay against mutated
            # compressed-memory metadata after a loop is detached.
            mixed, compressed, metrics = self.mixer(
                normed, segment_ids=segment_ids, memory=prior,
                return_stats=collect_diagnostics,
            )
            state_value = MemoryState(
                compressed.keys.detach(), compressed.values.detach(), compressed.positions,
                compressed.segment_ids, pending_sum=compressed.pending_sum,
                pending_count=compressed.pending_count,
                pending_segment_ids=compressed.pending_segment_ids,
            )
            if self.kind == "csa":
                state.csa_memories[state_key] = state_value
            else:
                state.hca_memories[state_key] = state_value
                state.hca_memories[state_key].latents = self.hca_latent.encode(
                    compressed.values
                ).detach()
            for key, value in metrics.items():
                diagnostics[key] = value
            history_size = self.mixer.local_window - 1
            state.local_histories[state_key] = normed[:, -history_size:].detach()
            state.local_history_valid[state_key] = torch.ones(
                state.local_histories[state_key].shape[:2], device=normed.device, dtype=torch.bool
            )
        streams = self._write(streams, mixed, mixer_mappings)

        hidden, ffn_mappings = self._read(streams)
        if self.moe is not None:
            hca_latents = None
            # The HCA block's current global summaries condition its own MoE;
            # later blocks can reuse the explicit memory from the state object.
            hca_latents = self._latest_hca_latents(state, state_namespace)
            moe_input = self.ffn_norm(hidden)
            if self.training and torch.is_grad_enabled() and not collect_diagnostics:
                # Avoid retaining the per-expert stacked weights and grouped dispatch
                # activations for every recurrent pass.  The backward recomputation is
                # substantially cheaper than reducing the whole training microbatch.
                ffn, aux = activation_checkpoint(
                    self._moe_checkpointed, moe_input, hca_latents,
                    collect_router_diagnostics, use_reentrant=False,
                )
                metrics = self.moe._last_router_health if collect_router_diagnostics else {}
            else:
                ffn, aux, metrics = self.moe(
                    moe_input, hca_global_latents=hca_latents,
                    return_stats=collect_diagnostics,
                    return_health=collect_router_diagnostics,
                )
            if not self.training and not collect_diagnostics:
                router_input = self.ffn_norm(hidden)
                router = self.moe.router
                router_local = router.norm(router_input)
                router_latents = router.latent_memory.encode(router_local)
                state.router_latent_histories[state_key] = router_latents[
                    :, -router.local_context:
                ].detach()
                state.router_history_valid[state_key] = torch.ones(
                    state.router_latent_histories[state_key].shape[:2],
                    device=hidden.device, dtype=torch.bool,
                )
            if collect_router_diagnostics:
                health = {key: metrics[key] for key in _ROUTER_HEALTH_KEYS if key in metrics}
                if health:
                    diagnostics.setdefault("_router_health_samples", []).append(health)
            if collect_diagnostics:
                diagnostics.update(metrics)
        else:
            ffn = self.ffn(self.ffn_norm(hidden))
            aux = hidden.new_zeros(())
        streams = self._write(streams, ffn, ffn_mappings)
        return streams, aux

    @torch.inference_mode()
    def forward_step(self, streams: torch.Tensor, state: RecurrentState, *, position: torch.Tensor | int,
                     segment_ids: torch.Tensor | None, reset_mask: torch.Tensor | None,
                     state_namespace: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply this physical block to one token using retained state."""
        state_key = (
            f"{state_namespace}.{self.block_name}"
            if state_namespace else self.block_name
        )
        hidden, mixer_mappings = self._read(streams)
        normed = self.mixer_norm(hidden)
        if self.kind == "gdn":
            prior = state.gdn_states.get(state_key)
            history = state.gdn_conv_histories.get(state_key)
            mixed, next_state = self.mixer(normed, prior, reset_mask, conv_history=history)
            state.gdn_states[state_key] = next_state.detach()
            history_size = self.mixer.conv.kernel_size[0] - 1
            if history is not None and history.shape[1] == history_size:
                history[:, :-1].copy_(history[:, 1:].clone())
                history[:, -1:].copy_(normed)
                state.gdn_conv_histories[state_key] = history
            else:
                combined = normed if history is None else torch.cat((history, normed), dim=1)
                state.gdn_conv_histories[state_key] = combined[:, -history_size:].detach()
        else:
            prior = state.csa_memories.get(state_key) if self.kind == "csa" else state.hca_memories.get(state_key)
            if prior is not None and prior.keys is not None and not prior.static_cache:
                # Generation is bounded by the training context window. Keep
                # the decode cache writable so appending a token does not
                # copy the complete history on every step.
                active_length = prior.keys.shape[1] if prior.length is None else int(prior.length)
                prior.reserve(max(256, active_length + 1))
            history = state.local_histories.get(state_key)
            history_valid = state.local_history_valid.get(state_key)
            mixed, compressed, _ = self.mixer.forward_step(
                normed, segment_ids=segment_ids, memory=prior,
                local_history=history, local_history_valid=history_valid, position=position,
            )
            if prior is not None and prior.capacity > 0:
                state_value = prior
            else:
                state_value = MemoryState(
                    compressed.keys.detach(), compressed.values.detach(), compressed.positions,
                    compressed.segment_ids, pending_sum=compressed.pending_sum,
                    pending_count=compressed.pending_count,
                    pending_segment_ids=compressed.pending_segment_ids,
                    length=compressed.keys.shape[1],
                ).reserve(256)
            if self.kind == "csa":
                state.csa_memories[state_key] = state_value
            else:
                state.hca_memories[state_key] = state_value
                if state_value.capacity > 0:
                    index = torch.as_tensor(position, device=normed.device, dtype=torch.long).view(normed.shape[0], 1)
                    index = index.remainder(state_value.capacity)
                    current_value = state_value.values.gather(
                        1, index.unsqueeze(-1).expand(-1, -1, state_value.values.shape[-1])
                    )
                    current_latent = self.hca_latent.encode(current_value).detach()
                    if state_value.latents is None:
                        state_value.latents = current_latent.new_zeros(
                            state_value.keys.shape[0], state_value.capacity, current_latent.shape[-1]
                        )
                    state_value.latents.scatter_(
                        1, index.unsqueeze(-1).expand(-1, -1, current_latent.shape[-1]), current_latent
                    )
                else:
                    current_latent = self.hca_latent.encode(compressed.values[:, -1:]).detach()
                    state_value.latents = current_latent
            history_size = self.mixer.local_window - 1
            if state.graph_mode and history is not None and history.shape[1] == history_size:
                history[:, :-1].copy_(history[:, 1:].clone())
                history[:, -1:].copy_(normed)
                valid = state.local_history_valid[state_key]
                valid[:, :-1].copy_(valid[:, 1:].clone())
                valid[:, -1] = True
            else:
                combined = normed if history is None else torch.cat((history, normed), dim=1)
                state.local_histories[state_key] = combined[:, -history_size:].detach()
                state.local_history_valid[state_key] = torch.ones(
                    state.local_histories[state_key].shape[:2], device=normed.device, dtype=torch.bool
                )

        streams = self._write(streams, mixed, mixer_mappings)
        hidden, ffn_mappings = self._read(streams)
        if self.moe is not None:
            hca_latents = None
            hca_latents = self._latest_hca_latents(state, state_namespace)
            ffn, aux, _, router_latents = self.moe.forward_step(
                self.ffn_norm(hidden),
                local_history=state.router_latent_histories.get(state_key),
                local_history_valid=state.router_history_valid.get(state_key),
                hca_global_latents=hca_latents,
            )
            router_history = state.router_latent_histories.get(state_key)
            if state.graph_mode and router_history is not None and router_history.shape[1] == self.moe.router.local_context:
                router_history[:, :-1].copy_(router_history[:, 1:].clone())
                router_history[:, -1:].copy_(router_latents[:, -1:])
                valid = state.router_history_valid[state_key]
                valid[:, :-1].copy_(valid[:, 1:].clone())
                valid[:, -1] = True
            else:
                state.router_latent_histories[state_key] = router_latents[:, -self.moe.router.local_context:].detach()
                state.router_history_valid[state_key] = torch.ones(
                    state.router_latent_histories[state_key].shape[:2], device=hidden.device, dtype=torch.bool
                )
        else:
            ffn = self.ffn(self.ffn_norm(hidden))
            aux = hidden.new_zeros(())
        streams = self._write(streams, ffn, ffn_mappings)
        return streams, aux


class LoopedHybridMoEForCausalLM(nn.Module):
    """Ten physical blocks with a shared six-block recurrent middle stack."""

    def __init__(self, config: V6Config | None = None) -> None:
        super().__init__()
        self.config = config or V6Config()
        cfg = self.config
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        # GPT-style init: std = 1/sqrt(d_model) prevents NaN logits at large vocab
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=cfg.d_model ** -0.5)
        self.stream_seed = nn.Parameter(torch.zeros(cfg.mhc.streams, cfg.d_model))
        nn.init.normal_(self.stream_seed, mean=0.0, std=1e-3)
        self.stem = nn.ModuleList([
            PhysicalBlock(cfg, "gdn", block_name=f"stem.{i}", with_moe=False)
            for i in range(cfg.stem_blocks)
        ])
        self.recurrent_core = nn.ModuleList([
            PhysicalBlock(cfg, layout.split("_")[0], block_name=f"core.{i}", with_moe=True)
            for i, layout in enumerate(cfg.recurrent_layout)
        ])
        self.recurrent_core_extra = nn.ModuleList()
        if not cfg.mhc.shared_across_loops:
            for loop_index in range(1, max(1, cfg.num_loops)):
                loop_blocks = nn.ModuleList()
                for block_index, layout in enumerate(cfg.recurrent_layout):
                    block = copy.deepcopy(self.recurrent_core[block_index])
                    block.block_name = f"core.loop{loop_index}.{block_index}"
                    if cfg.moe.expert_weights_shared_across_loops and block.moe is not None:
                        block.moe.experts = self.recurrent_core[block_index].moe.experts
                        block.moe.shared = self.recurrent_core[block_index].moe.shared
                    loop_blocks.append(block)
                self.recurrent_core_extra.append(loop_blocks)
        self.tail = nn.ModuleList([
            PhysicalBlock(cfg, "gdn", block_name=f"tail.{i}", with_moe=False)
            for i in range(cfg.tail_blocks)
        ])
        self.mhc_collapse = nn.Linear(cfg.mhc.streams * cfg.d_model, cfg.d_model, bias=False)
        with torch.no_grad():
            self.mhc_collapse.weight.zero_()
            for stream in range(cfg.mhc.streams):
                start = stream * cfg.d_model
                self.mhc_collapse.weight[:, start:start + cfg.d_model].copy_(
                    torch.eye(cfg.d_model) / cfg.mhc.streams
                )
        self.final_norm = RMSNorm(cfg.d_model)
        self.loop_memory = LoopMemoryHook(
            cfg.d_model,
            cfg.loop_attnres.latent_dim,
            enabled=cfg.loop_attnres.enabled,
            heads=cfg.loop_attnres.heads,
            head_dim=cfg.loop_attnres.head_dim,
            residual_gate_init=cfg.loop_attnres.residual_gate_init,
        )
        # Reads the representation emitted by the final recurrent block and
        # predicts whether each token should enter another recursive pass.
        self.recurrent_halt_head = nn.Linear(cfg.d_model, 1)
        nn.init.zeros_(self.recurrent_halt_head.weight)
        initial_prob = cfg.adaptive_loops.initial_continue_prob
        with torch.no_grad():
            self.recurrent_halt_head.bias.fill_(
                torch.logit(torch.tensor(initial_prob)).item()
            )

    def _streams_from_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        return hidden.unsqueeze(-2) + self.stream_seed.to(hidden.dtype).view(1, 1, -1, self.config.d_model)

    def _run_stack(self, blocks: nn.ModuleList, streams: torch.Tensor, state: RecurrentState,
                   *, segment_ids: torch.Tensor | None, reset_mask: torch.Tensor | None,
                   diagnostics: dict[str, Any], collect_diagnostics: bool = True,
                   collect_router_diagnostics: bool = False,
                   position: torch.Tensor | int | None = None,
                   state_namespace: str | None = None,
                   active_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        auxiliary = streams.new_zeros((), dtype=torch.float32)
        for block in blocks:
            previous = streams
            streams, block_aux = block(
                streams, state, segment_ids=segment_ids, reset_mask=reset_mask,
                diagnostics=diagnostics, collect_diagnostics=collect_diagnostics,
                collect_router_diagnostics=collect_router_diagnostics, position=position,
                state_namespace=state_namespace,
            )
            # Halted tokens are frozen at every physical block boundary.  This
            # is the sequence-path equivalent of decode_step's token-wise
            # early-exit behavior; masking only after the complete recurrent
            # stack would allow later blocks to update halted tokens.
            if active_mask is not None:
                active_view = active_mask.to(dtype=streams.dtype).unsqueeze(-1).unsqueeze(-1)
                streams = streams * active_view + previous * (1.0 - active_view)
            auxiliary = auxiliary + block_aux.float()
        return streams, auxiliary

    def _recurrent_stack(self, loop_index: int) -> nn.ModuleList:
        if loop_index == 0 or not self.recurrent_core_extra:
            return self.recurrent_core
        return self.recurrent_core_extra[min(loop_index - 1, len(self.recurrent_core_extra) - 1)]

    def _sample_loop_budgets(self, batch: int, length: int, loops: int,
                             device: torch.device) -> torch.Tensor:
        """Sample a loop budget per token for training-time adaptive compute.

        The recurrent stack is still evaluated as a dense tensor for kernel and
        state compatibility; inactive tokens are restored after each pass. This
        preserves token-wise behavior and the loop attention residual path while
        avoiding unsafe per-token state mutation.
        """
        cfg = self.config.adaptive_loops
        if not self.training or not cfg.enabled:
            return torch.full((batch, length), loops, device=device, dtype=torch.long)
        minimum = max(1, min(int(cfg.min_loops), loops))
        maximum = max(minimum, min(int(cfg.max_loops), loops))
        if minimum == maximum:
            return torch.full((batch, length), minimum, device=device, dtype=torch.long)
        return torch.randint(minimum, maximum + 1, (batch, length), device=device)

    def _halt_decision(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a hard forward decision with a soft backward gradient."""
        cfg = self.config.adaptive_loops
        hidden = self.mhc_collapse(streams.reshape(streams.shape[0], streams.shape[1], -1))
        logits = self.recurrent_halt_head(hidden).squeeze(-1)
        probabilities = torch.clamp(
            torch.sigmoid(logits.float() / float(cfg.halt_temperature)),
            0.0, 1.0
        ).nan_to_num(0.5)
        if self.training:
            hard = torch.bernoulli(probabilities).to(streams.dtype)
        else:
            hard = (probabilities >= float(cfg.halt_threshold)).to(streams.dtype)
        straight_through = hard.detach() - probabilities.detach() + probabilities
        return straight_through, probabilities

    def _run_step_stack(self, blocks: nn.ModuleList, streams: torch.Tensor, state: RecurrentState,
                        *, position: torch.Tensor | int, segment_ids: torch.Tensor | None,
                        reset_mask: torch.Tensor | None,
                        state_namespace: str | None = None,
                        active_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        auxiliary = streams.new_zeros((), dtype=torch.float32)
        for block in blocks:
            previous = streams
            streams, block_aux = block.forward_step(
                streams, state, position=position, segment_ids=segment_ids,
                reset_mask=reset_mask,
                state_namespace=state_namespace,
            )
            if active_mask is not None:
                active_view = active_mask.to(dtype=streams.dtype).view(-1, 1, 1, 1)
                streams = streams * active_view + previous * (1.0 - active_view)
            auxiliary = auxiliary + block_aux.float()
        return streams, auxiliary

    def forward(self, input_ids: torch.Tensor, *, labels: torch.Tensor | None = None,
                state: RecurrentState | None = None, num_loops: int | None = None,
                segment_ids: torch.Tensor | None = None, reset_mask: torch.Tensor | None = None,
                record_loop_memory: bool | None = None,
                collect_diagnostics: bool = True,
                collect_router_diagnostics: bool = False,
                last_logits_only: bool = False) -> LoopedHybridOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        loops = int(self.config.num_loops if num_loops is None else num_loops)
        if loops < 1:
            raise ValueError("num_loops must be positive")
        batch, length = input_ids.shape
        state = state or RecurrentState()
        reset_rows = None
        if reset_mask is not None:
            if reset_mask.shape != input_ids.shape:
                raise ValueError("reset_mask must have the same shape as input_ids")
            reset_rows = reset_mask[:, 0]
            if state.streams is not None and bool(reset_rows.any()):
                state.reset_rows(reset_rows)
        if segment_ids is not None and segment_ids.shape != input_ids.shape:
            raise ValueError("segment_ids must have the same shape as input_ids")
        if state.streams is None:
            streams = self._streams_from_tokens(input_ids)
        else:
            streams = state.streams.clone()
            if tuple(streams.shape) != (batch, length, self.config.mhc.streams, self.config.d_model):
                raise ValueError("recurrent state streams do not match input shape")
            if reset_rows is not None and bool(reset_rows.any()):
                streams[reset_rows] = self._streams_from_tokens(input_ids[reset_rows])
        adaptive_cfg = self.config.adaptive_loops
        adaptive_on = bool(adaptive_cfg.enabled)
        learned_halting = bool(adaptive_on and adaptive_cfg.learned_halting)
        uncapped_recursion = bool(
            learned_halting
            and not self.training
            and getattr(self, "_uncapped_recursion", False)
        )
        if learned_halting and not uncapped_recursion:
            # The sampler's maximum is a runtime cap and may exceed the
            # checkpoint's default depth; recurrent weights are reused beyond
            # the explicitly materialized loop stacks.
            loops = max(1, int(adaptive_cfg.max_loops))
        if not adaptive_on:
            loop_budgets = None
            active = None
            effective_loop_count = None
            halt_penalty_loss = streams.new_zeros(())
            halt_probability_sum = streams.new_zeros(())
        else:
            loop_budgets = (
                None if learned_halting
                else self._sample_loop_budgets(batch, length, loops, input_ids.device)
            )
            active = streams.new_ones((batch, length), dtype=torch.float32)
            effective_loop_count = streams.new_zeros((batch, length), dtype=torch.float32)
            halt_penalty_loss = streams.new_zeros(())
            halt_probability_sum = streams.new_zeros(())
        diagnostics: dict[str, Any] = {
            "num_loops": loops,
            "loop_auxiliary_loss": [],
        }
        if loop_budgets is not None:
            diagnostics["adaptive_loop_budget_mean"] = float(loop_budgets.float().mean().detach())
        if learned_halting:
            diagnostics["learned_halt_penalty"] = float(adaptive_cfg.halt_penalty)
        if collect_diagnostics and self.training and adaptive_cfg.enabled and not learned_halting:
            for budget in range(1, loops + 1):
                diagnostics[f"adaptive_loop_budget_{budget}_fraction"] = float(
                    (loop_budgets == budget).float().mean().detach()
                )
        streams, auxiliary = self._run_stack(
            self.stem, streams, state, segment_ids=segment_ids,
            reset_mask=reset_mask, diagnostics=diagnostics,
            collect_diagnostics=collect_diagnostics,
            collect_router_diagnostics=collect_router_diagnostics,
            position=torch.arange(length, device=input_ids.device, dtype=torch.long),
        )
        position = torch.arange(length, device=input_ids.device, dtype=torch.long)
        loop_endpoints: list[torch.Tensor] = []
        loop_boundary_streams = streams
        halt_decision_count = 0
        configured_backprop_loops = int(adaptive_cfg.backprop_loops)
        backprop_loops = loops if configured_backprop_loops <= 0 else min(
            configured_backprop_loops, loops
        )
        backprop_start = loops - backprop_loops
        diagnostics["backprop_loops"] = backprop_loops
        diagnostics["backprop_start_loop"] = backprop_start
        for loop_index in (count() if uncapped_recursion else range(loops)):
            # Once every token has halted, no later recurrent pass contributes
            # to the output.  This is a real early exit, not post-hoc masking.
            if active is not None and not bool(active.any()):
                diagnostics["executed_loop_count"] = loop_index
                break
            loop_requires_grad = loop_index >= backprop_start
            if loop_index == backprop_start and backprop_start > 0:
                # Truncated BPTT: preserve the forward value, but sever the
                # graph accumulated through earlier recurrent passes.
                streams = streams.detach()
                loop_endpoints = [endpoint.detach() for endpoint in loop_endpoints]
            # Re-run every MoE router on every recurrent pass.  Routing indices
            # are intentionally not carried in RecurrentState, so changed
            # hidden states may select different routed experts per loop.
            with torch.set_grad_enabled(loop_requires_grad):
                if loop_index > 0 and not self.config.mhc.preserve_streams_across_loop_boundary:
                    streams = loop_boundary_streams.clone()
                streams = self.loop_memory.inject(streams, loop_endpoints)
                if loop_budgets is not None:
                    active = (loop_budgets >= (loop_index + 1)).to(torch.float32)
                candidate, loop_aux = self._run_stack(
                    self._recurrent_stack(loop_index), streams, state, segment_ids=segment_ids,
                    reset_mask=reset_mask, diagnostics=diagnostics,
                    collect_diagnostics=collect_diagnostics,
                    collect_router_diagnostics=collect_router_diagnostics,
                    position=position,
                    state_namespace=f"loop{loop_index}",
                    active_mask=active,
                )
                # ``_run_stack`` freezes inactive tokens at every physical
                # block.  Keep this boundary mask as a final guard for the
                # loop-attention residual path.
                if active is not None:
                    active_view = active.to(dtype=candidate.dtype).unsqueeze(-1).unsqueeze(-1)
                    streams = candidate * active_view + streams * (1.0 - active_view)
                else:
                    streams = candidate
                auxiliary = auxiliary + loop_aux
                if effective_loop_count is not None:
                    effective_loop_count = effective_loop_count + active
                if collect_diagnostics:
                    diagnostics["loop_auxiliary_loss"].append(float(loop_aux.detach()))
                should_record = record_loop_memory if record_loop_memory is not None else self.config.loop_attnres.enabled
                if should_record:
                    endpoint = self.loop_memory.record(
                        streams, state, force=record_loop_memory is True
                    )
                    if endpoint is not None:
                        loop_endpoints.append(endpoint)
                if uncapped_recursion or loop_index < loops - 1:
                    if not adaptive_on:
                        continue
                    if loop_index + 1 < adaptive_cfg.min_loops:
                        continue_decision = active
                        continue_probability = active
                    elif learned_halting:
                        continue_decision, continue_probability = self._halt_decision(streams)
                    else:
                        continue_decision = (loop_budgets >= (loop_index + 2)).to(torch.float32)
                        continue_probability = continue_decision
                    if learned_halting or loop_budgets is not None:
                        halt_probability_sum = halt_probability_sum + (
                            active * continue_probability
                        ).mean()
                        halt_penalty_loss = halt_penalty_loss + (
                            float(adaptive_cfg.halt_penalty) * (active * continue_probability).mean()
                        )
                        halt_decision_count += 1
                    active = active * continue_decision
                    if uncapped_recursion:
                        print(
                            f"\n[uncapped prefill] pass={loop_index + 1} "
                            f"continuing_tokens={int(active.sum().item())}/{active.numel()} "
                            f"mean_continue_probability={float(continue_probability.float().mean().item()):.4f}",
                            flush=True,
                        )
        diagnostics.setdefault("executed_loop_count", loops)
        if uncapped_recursion:
            diagnostics["num_loops"] = int(diagnostics["executed_loop_count"])
        streams, tail_aux = self._run_stack(
            self.tail, streams, state, segment_ids=segment_ids,
            reset_mask=reset_mask, diagnostics=diagnostics,
            collect_diagnostics=collect_diagnostics,
            collect_router_diagnostics=collect_router_diagnostics,
            position=position,
        )
        auxiliary = auxiliary + tail_aux
        state.streams = streams.detach()
        state.token_positions = torch.arange(length, device=input_ids.device).view(1, -1).expand(batch, -1)
        collapsed = self.mhc_collapse(streams.reshape(batch, length, -1))
        hidden = self.final_norm(collapsed)
        # Only materialise full [B, T, V] logits for inference (labels=None).
        # During training we use chunked CE directly on hidden to save ~4 GB peak VRAM.
        if labels is None:
            projection_hidden = hidden[:, -1:, :] if last_logits_only else hidden
            logits = F.linear(projection_hidden, self.embed_tokens.weight)
        else:
            logits = hidden.new_zeros(*hidden.shape[:-1], 1)
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            # Chunked fused cross-entropy.  MoE activation checkpointing keeps enough
            # headroom for the original large GEMMs without thousands of tiny matmuls.
            hidden_flat = hidden[:, :-1].reshape(-1, hidden.shape[-1])           # [(B*(T-1)), D]
            target_flat = labels[:, 1:].reshape(-1)                              # [(B*(T-1))]
            # Keep the fused full-vocabulary GEMM while leaving peak headroom for
            # the checkpointed MoE backward graph on the 24GB card.
            # Keep the full-vocabulary projection below the A5000's peak
            # allocation when a recurrent/MoE activation spike coincides with
            # the loss projection.  This is deliberately smaller than the
            # 256 rows bounds the vocabulary projection to 242.5 MiB,
            # leaving plenty of safety buffer for the backward graph on the 24GB RTX 3090.
            chunk_size = 512
            total_tokens = hidden_flat.shape[0]
            loss_accum = hidden_flat.new_zeros((), dtype=torch.float32)
            count_t = torch.zeros((), device=hidden_flat.device, dtype=torch.long)
            embed_w = self.embed_tokens.weight
            for chunk_start in range(0, total_tokens, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_tokens)
                h_chunk = hidden_flat[chunk_start:chunk_end]          # [C, D]
                t_chunk = target_flat[chunk_start:chunk_end]          # [C]
                h_proj = h_chunk if h_chunk.dtype == embed_w.dtype else h_chunk.to(embed_w.dtype)
                loss_chunk = _chunk_ce(h_proj, t_chunk, embed_w)
                loss_accum = loss_accum + loss_chunk
                count_t = count_t + (t_chunk >= 0).sum()
            loss = loss_accum / count_t.clamp_min(1).float() + auxiliary
            if adaptive_on:
                loss = loss + halt_penalty_loss
        if collect_diagnostics:
            mhc_metrics: dict[str, float] = {}
            for module_name, module in self.named_modules():
                if isinstance(module, MultiStreamHyperConnection):
                    for metric_name, value in module.constraint_errors().items():
                        mhc_metrics[f"{module_name}/{metric_name}"] = value
            diagnostics["mhc"] = mhc_metrics
            diagnostics["auxiliary_loss"] = float(auxiliary.detach())
        # These are intentionally emitted even when the trainer disables the
        # expensive per-block diagnostics.  Otherwise the learned halting
        # controller would be invisible in the normal low-overhead telemetry.
        if learned_halting:
            diagnostics["effective_loop_count"] = float(effective_loop_count.mean().detach())
            diagnostics["halt_penalty_loss"] = float(halt_penalty_loss.detach())
            if halt_decision_count:
                diagnostics["halt_continue_probability"] = float(
                    (halt_probability_sum / halt_decision_count).detach()
                )
        if collect_router_diagnostics:
            samples = diagnostics.pop("_router_health_samples", [])
            if samples:
                diagnostics["router_health"] = {
                    key: sum(float(sample[key]) for sample in samples if key in sample)
                    / max(1, sum(1 for sample in samples if key in sample))
                    for key in _ROUTER_HEALTH_KEYS
                    if any(key in sample for sample in samples)
                }
        return LoopedHybridOutput(logits, loss, state, auxiliary, diagnostics)

    @torch.inference_mode()
    def decode_step(self, input_ids: torch.Tensor, *, state: RecurrentState,
                    segment_ids: torch.Tensor | None = None,
                    reset_mask: torch.Tensor | None = None,
                    num_loops: int | None = None,
                    position: torch.Tensor | None = None) -> LoopedHybridOutput:
        """Generate logits for one token using recurrent and local state.

        The ordinary ``forward`` method intentionally remains a sequence
        implementation for training and evaluation. This path carries GDN
        state, compressed attention memory, and bounded local histories so a
        new token is not reprocessed with the entire prompt.
        """
        if input_ids.ndim == 1:
            input_ids = input_ids[:, None]
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("decode_step expects input_ids with shape [batch, 1]")
        batch = input_ids.shape[0]
        if state.token_positions is None:
            raise ValueError("decode_step requires state initialized by a prefill forward pass")
        if state.token_positions.ndim != 2 or state.token_positions.shape[0] != batch:
            raise ValueError("state token positions do not match input batch")
        if segment_ids is not None and segment_ids.shape != input_ids.shape:
            raise ValueError("segment_ids must have shape [batch, 1]")
        if reset_mask is not None and reset_mask.shape != input_ids.shape:
            raise ValueError("reset_mask must have shape [batch, 1]")

        reset_rows = reset_mask[:, 0].to(torch.bool) if reset_mask is not None else None
        if reset_rows is not None and bool(reset_rows.any()):
            state.reset_rows(reset_rows)
        if position is None:
            previous = state.token_positions[:, -1]
            position = previous + 1
            if reset_rows is not None:
                position = torch.where(reset_rows, torch.zeros_like(position), position)
        else:
            position = torch.as_tensor(position, device=input_ids.device, dtype=torch.long).view(batch)

        streams = self._streams_from_tokens(input_ids)
        loops = int(self.config.num_loops if num_loops is None else num_loops)
        if loops < 1:
            raise ValueError("num_loops must be positive")
        adaptive_cfg = self.config.adaptive_loops
        learned_halting = bool(adaptive_cfg.enabled and adaptive_cfg.learned_halting)
        uncapped_recursion = bool(
            learned_halting
            and not self.training
            and getattr(self, "_uncapped_recursion", False)
        )
        if learned_halting and not uncapped_recursion:
            loops = max(1, int(adaptive_cfg.max_loops))
        minimum_loops = min(loops, max(1, int(adaptive_cfg.min_loops))) if learned_halting else loops
        # When no row can halt before the final loop, skip active-row tensors
        # and device-to-host loop-count reads in this token-by-token path.
        fixed_loop_decode = (
            not learned_halting
            or (not uncapped_recursion and minimum_loops >= loops)
        )
        auxiliary = streams.new_zeros((), dtype=torch.float32)
        streams, block_aux = self._run_step_stack(
            self.stem, streams, state, position=position,
            segment_ids=segment_ids, reset_mask=reset_mask,
        )
        auxiliary = auxiliary + block_aux
        loop_endpoints: list[torch.Tensor] = []
        loop_boundary_streams = streams
        active = None if fixed_loop_decode else streams.new_ones((batch, 1), dtype=torch.float32)
        executed_loop_counts = (
            None if fixed_loop_decode
            else torch.zeros((batch,), device=streams.device, dtype=torch.float32)
        )
        continue_probability_sum = None if fixed_loop_decode else streams.new_zeros(())
        continue_probability_count = 0
        for loop_index in (count() if uncapped_recursion else range(loops)):
            if fixed_loop_decode:
                active_rows = None
                active_row_count = batch
                all_rows_active = True
            else:
                active_rows = torch.nonzero(active[:, 0] > 0, as_tuple=False).flatten()
                if active_rows.numel() == 0:
                    break
                active_row_count = active_rows.numel()
                all_rows_active = active_row_count == batch
                executed_loop_counts.index_add_(
                    0,
                    active_rows,
                    torch.ones((active_row_count,), device=streams.device, dtype=executed_loop_counts.dtype),
                )
            # Decode rows are independent at this point.  The compact path is
            # needed when only some batch rows continue, but the common batch-1
            # case has every row active.  Avoiding select/scatter copies of all
            # recurrent memories on every loop removes a large decode-only
            # overhead while preserving the exact same state transitions.
            if all_rows_active:
                sub_state = state
                sub_streams = streams
                sub_position = position
                sub_segments = segment_ids
                sub_reset = reset_mask
                sub_endpoints = loop_endpoints
            else:
                sub_state = state.select_rows(active_rows)
                sub_streams = streams.index_select(0, active_rows)
                sub_position = position.index_select(0, active_rows)
                sub_segments = None if segment_ids is None else segment_ids.index_select(0, active_rows)
                sub_reset = None if reset_mask is None else reset_mask.index_select(0, active_rows)
                sub_endpoints = [endpoint.index_select(0, active_rows) for endpoint in loop_endpoints]
            if loop_index > 0 and not self.config.mhc.preserve_streams_across_loop_boundary:
                sub_streams = loop_boundary_streams.clone() if all_rows_active else loop_boundary_streams.index_select(0, active_rows).clone()
            sub_streams = self.loop_memory.inject(sub_streams, sub_endpoints)
            sub_streams, loop_aux = self._run_step_stack(
                self._recurrent_stack(loop_index), sub_streams, sub_state, position=sub_position,
                segment_ids=sub_segments, reset_mask=sub_reset,
                state_namespace=f"loop{loop_index}",
            )
            auxiliary = auxiliary + loop_aux * (active_row_count / max(batch, 1))
            endpoint = self.loop_memory.record(sub_streams, sub_state)
            if endpoint is not None:
                if all_rows_active:
                    loop_endpoints.append(endpoint)
                else:
                    full_endpoint = endpoint.new_zeros((batch, *endpoint.shape[1:]))
                    full_endpoint.index_copy_(0, active_rows, endpoint)
                    loop_endpoints.append(full_endpoint)
            if all_rows_active:
                # The batch-1 fast path must carry this pass's activations into
                # the next recurrent pass, just as the compact-row path does.
                streams = sub_streams
            else:
                streams.index_copy_(0, active_rows, sub_streams)
                state.scatter_rows(active_rows, sub_state)
            if (not fixed_loop_decode
                    and (uncapped_recursion or loop_index < loops - 1)
                    and learned_halting):
                if loop_index + 1 < minimum_loops:
                    continue_decision = torch.ones_like(active_rows, dtype=torch.float32).view(-1, 1)
                    continue_probability = continue_decision
                else:
                    continue_decision, continue_probability = self._halt_decision(sub_streams)
                continue_probability_sum = continue_probability_sum + continue_probability.float().sum()
                continue_probability_count += int(continue_probability.numel())
                next_active = torch.zeros_like(active)
                next_active.index_copy_(0, active_rows, continue_decision.to(next_active.dtype))
                active = next_active
                if uncapped_recursion:
                    print(
                        f"\n[uncapped decode] pass={loop_index + 1} "
                        f"continuing_rows={int(active.sum().item())}/{batch} "
                        f"mean_continue_probability={float(continue_probability.float().mean().item()):.4f}",
                        flush=True,
                    )
        streams, tail_aux = self._run_step_stack(
            self.tail, streams, state, position=position,
            segment_ids=segment_ids, reset_mask=reset_mask,
        )
        auxiliary = auxiliary + tail_aux
        state.streams = streams.detach()
        state.token_positions = position[:, None].detach()
        collapsed = self.mhc_collapse(streams.reshape(batch, 1, -1))
        hidden = self.final_norm(collapsed)
        logits = F.linear(hidden, self.embed_tokens.weight)
        if fixed_loop_decode:
            executed_loop_count = loops
            effective_loop_count = float(loops)
            executed_counts_summary: Any = [loops] * batch
        else:
            executed_loop_count = int(executed_loop_counts.max().item())
            effective_loop_count = float(executed_loop_counts.mean().item())
            executed_counts_summary = executed_loop_counts.detach()
        diagnostics: dict[str, Any] = {
            "num_loops": int(executed_loop_counts.max().item()) if uncapped_recursion else loops,
            "min_loops": minimum_loops,
            "max_loops": None if uncapped_recursion else loops,
            "uncapped_recursion": uncapped_recursion,
            "executed_loop_count": executed_loop_count,
            "effective_loop_count": effective_loop_count,
            "executed_loop_counts": executed_counts_summary,
            "loop_attention_residual_count": len(loop_endpoints),
        }
        if continue_probability_count:
            diagnostics["halt_continue_probability"] = float(
                (continue_probability_sum / continue_probability_count).item()
            )
        return LoopedHybridOutput(logits, None, state, auxiliary, diagnostics)

    def parameter_report(self) -> dict[str, Any]:

        return count_unique_parameters(self)


# ===== inlined sampler =====
def apply_all_triton_optimizations(model: nn.Module) -> nn.Module:
    """Apply the same guarded Triton patching policy as the distributed runner."""
    patched_norms = 0
    for parent in model.modules():
        for child_name, child in list(parent.named_children()):
            if child.__class__.__name__ in ("RMSNorm", "LlamaRMSNorm"):
                eps = getattr(child, "eps", None) or 1e-6
                replacement = TritonRMSNorm(
                    child.weight.shape[0], eps=eps
                ).to(device=child.weight.device, dtype=child.weight.dtype)
                replacement.weight.data.copy_(child.weight.data)
                setattr(parent, child_name, replacement)
                patched_norms += 1

    patched_mhc = patched_hca = patched_csa = 0
    for module in model.modules():
        if isinstance(module, MultiStreamHyperConnection):
            module.use_triton_kernels = bool(_BANDED_TRITON_AVAILABLE)
            patched_mhc += int(module.use_triton_kernels)
        elif isinstance(module, HierarchicalCompressedAttention):
            module.use_triton_banded = bool(_BANDED_TRITON_AVAILABLE)
            patched_hca += int(module.use_triton_banded)
        elif isinstance(module, CompressedSelectiveAttention):
            compatible = module.index_head_dim in (32, 64, 128)
            module.use_triton_banded = bool(_BANDED_TRITON_AVAILABLE and compatible)
            patched_csa += int(module.use_triton_banded)
    print(
        f"[Triton Optimization Patcher] RMSNorm={patched_norms}, "
        f"mHC={patched_mhc}, HCA-banded={patched_hca}, CSA-banded={patched_csa}; "
        "GDN/MoE use their native guarded paths"
    )
    return model


def _autocast_context(device: torch.device, dtype: torch.dtype):
    """Use the same AMP policy as the live trainer without casting weights.

    The RunPod trainer keeps parameters in FP32 and runs CUDA operations under
    BF16 autocast on the A5000 (FP16 only on GPUs without BF16 support).  The
    old sampler permanently converted all weights to FP16, which changed the
    numerics before generation even when it loaded the right checkpoint.
    """
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _prepare_cuda_graph_state(state, *, capacity: int = 4096, model=None) -> None:
    """Make state fixed-shape for graph replay; ``capacity`` is token positions."""
    state.graph_mode = True
    for memory_kind, memories in (
        ("csa", state.csa_memories), ("hca", state.hca_memories),
    ):
        cfg = getattr(getattr(model, "config", None), memory_kind, None)
        ratio = max(1, int(getattr(cfg, "temporal_compression", 1)))
        cache_capacity = max(1, (int(capacity) + ratio - 1) // ratio + 1)
        for memory in memories.values():
            if memory.keys is None:
                continue
            active = memory.keys.shape[1] if memory.length is None else int(memory.length)
            memory.reserve(max(cache_capacity, active))
            batch_size = memory.keys.shape[0]
            model_width = memory.values.shape[-1]
            if memory.pending_sum is None:
                memory.pending_sum = memory.values.new_zeros((batch_size, model_width))
            if memory.pending_count is None:
                memory.pending_count = torch.zeros(
                    batch_size, device=memory.keys.device, dtype=torch.long
                )
            if memory.pending_segment_ids is None:
                memory.pending_segment_ids = torch.full(
                    (batch_size,), -1, device=memory.keys.device, dtype=torch.long
                )
            # Prefill stores compressed entries contiguously. Graph decode
            # indexes the ring by absolute token position, so relocate the
            # existing entries once before capture.
            active = min(active, memory.keys.shape[1])
            old_keys = memory.keys[:, :active].clone()
            old_values = memory.values[:, :active].clone()
            old_positions = memory.positions[:, :active].clone()
            old_segments = memory.segment_ids[:, :active].clone()
            old_latents = (
                None if memory.latents is None else memory.latents[:, :active].clone()
            )
            memory.keys.zero_()
            memory.values.zero_()
            memory.positions.fill_(-1)
            memory.segment_ids.fill_(-1)
            if memory.latents is not None:
                memory.latents.zero_()
            for batch_index in range(batch_size):
                valid = old_positions[batch_index] >= 0
                if not bool(valid.any()):
                    continue
                slots = (
                    old_positions[batch_index, valid] // ratio
                ).remainder(memory.capacity)
                memory.keys[batch_index].index_copy_(0, slots, old_keys[batch_index, valid])
                memory.values[batch_index].index_copy_(0, slots, old_values[batch_index, valid])
                memory.positions[batch_index].index_copy_(0, slots, old_positions[batch_index, valid])
                memory.segment_ids[batch_index].index_copy_(0, slots, old_segments[batch_index, valid])
                if old_latents is not None:
                    memory.latents[batch_index].index_copy_(0, slots, old_latents[batch_index, valid])
            memory.static_cache = True
            memory.length = memory.capacity

    def pad_history(values: dict[str, torch.Tensor], valid_values: dict[str, torch.Tensor], width: int) -> None:
        for name, history in list(values.items()):
            count = min(history.shape[1], width)
            padded = history.new_zeros((history.shape[0], width, history.shape[2]))
            if count:
                padded[:, -count:] = history[:, -count:]
            values[name] = padded
            valid = torch.zeros((history.shape[0], width), device=history.device, dtype=torch.bool)
            if count:
                valid[:, -count:] = True
            valid_values[name] = valid

    pad_history(state.gdn_conv_histories, {}, 3)
    pad_history(state.local_histories, state.local_history_valid, 127)
    pad_history(state.router_latent_histories, state.router_history_valid, 128)


def _restore_eager_state(state) -> None:
    """Undo graph-only storage changes when capture is unavailable."""
    state.graph_mode = False
    for memory in (*state.csa_memories.values(), *state.hca_memories.values()):
        if memory.keys is None:
            continue
        active = memory.keys.shape[1] if memory.length is None else int(memory.length)
        for name in ("keys", "values", "positions", "segment_ids", "latents"):
            value = getattr(memory, name)
            if value is not None:
                setattr(memory, name, value[:, :active].contiguous())
        memory.length = None
        memory.capacity = 0
        memory.static_cache = False
    for values, valid_values in (
        (state.local_histories, state.local_history_valid),
        (state.router_latent_histories, state.router_history_valid),
    ):
        for name, history in list(values.items()):
            valid = valid_values.get(name)
            if valid is None:
                continue
            count = int(valid.sum(dim=1).min().item())
            values[name] = history[:, -count:].contiguous() if count else history[:, :0].contiguous()


def _snapshot_graph_capture_state(state):
    """Clone mutable decode buffers before CUDA Graph capture mutates them."""
    tensor_attrs = {}
    for name in ("streams", "token_positions"):
        value = getattr(state, name)
        if value is not None:
            tensor_attrs[name] = (value, value.clone())
    maps = {}
    for name in (
        "gdn_states", "gdn_conv_histories", "local_histories",
        "local_history_valid", "router_latent_histories", "router_history_valid",
    ):
        values = getattr(state, name)
        maps[name] = {
            key: (value, value.clone())
            for key, value in values.items() if value is not None
        }
    memories = {}
    for name in ("csa_memories", "hca_memories"):
        memories[name] = {}
        for key, memory in getattr(state, name).items():
            fields = {}
            for field_name in (
                "keys", "values", "positions", "segment_ids", "latents",
                "pending_sum", "pending_count", "pending_segment_ids",
            ):
                value = getattr(memory, field_name)
                if value is not None:
                    fields[field_name] = (value, value.clone())
            memories[name][key] = (
                memory, fields, memory.length, memory.capacity, memory.static_cache,
            )
    endpoints = [(value, value.clone()) for value in state.loop_endpoints]
    return tensor_attrs, maps, memories, endpoints, list(state.loop_endpoints), state.graph_mode


def _restore_graph_capture_state(state, snapshot) -> None:
    """Undo the one execution performed while a CUDA Graph is captured."""
    tensor_attrs, maps, memories, endpoints, endpoint_list, graph_mode = snapshot
    for name, (target, saved) in tensor_attrs.items():
        setattr(state, name, target)
        target.copy_(saved)
    for name, saved_values in maps.items():
        values = getattr(state, name)
        for key in list(values):
            if key not in saved_values:
                del values[key]
        for key, (target, saved) in saved_values.items():
            values[key] = target
            target.copy_(saved)
    for name, saved_memories in memories.items():
        values = getattr(state, name)
        for key, (memory, fields, length, capacity, static_cache) in saved_memories.items():
            values[key] = memory
            for field_name in (
                "keys", "values", "positions", "segment_ids", "latents",
                "pending_sum", "pending_count", "pending_segment_ids",
            ):
                saved_field = fields.get(field_name)
                if saved_field is None:
                    setattr(memory, field_name, None)
                else:
                    target, saved = saved_field
                    setattr(memory, field_name, target)
                    target.copy_(saved)
            memory.length = length
            memory.capacity = capacity
            memory.static_cache = static_cache
    for target, saved in endpoints:
        target.copy_(saved)
    state.loop_endpoints = endpoint_list
    state.graph_mode = graph_mode


class _CudaGraphDecoder:
    """Replay a fixed-shape one-token decode without Python launch gaps."""

    def __init__(self, model, state, token: torch.Tensor, position: int, amp_dtype: torch.dtype) -> None:
        global _fused_gdn_step
        self.model = model
        self.state = state
        self.static_input = token.clone()
        self.static_position = torch.full(
            (token.shape[0],), position, device=token.device, dtype=torch.long
        )
        self.graph = torch.cuda.CUDAGraph()

        # Numba's external launch is not graph-safe on this CUDA runtime.
        # Disable the module-level fused path during capture, then restore it
        # for ordinary eager inference.
        gdn_step = _fused_gdn_step
        capture_state = _snapshot_graph_capture_state(state)
        _fused_gdn_step = None
        capture_succeeded = False
        try:
            # Initialize lazy CUDA/cuBLAS paths before capture. Capturing the
            # very first decode kernel sequence produced different numerics on
            # Pascal even though later warmed replays matched eager.
            warmup_stream = torch.cuda.Stream(device=token.device)
            current_stream = torch.cuda.current_stream(token.device)
            warmup_stream.wait_stream(current_stream)
            with torch.cuda.stream(warmup_stream):
                for _ in range(2):
                    with _autocast_context(token.device, amp_dtype):
                        model.decode_step(
                            self.static_input,
                            state=state,
                            position=self.static_position,
                        )
            current_stream.wait_stream(warmup_stream)
            torch.cuda.synchronize(token.device)
            _restore_graph_capture_state(state, capture_state)

            with torch.cuda.graph(self.graph):
                with _autocast_context(token.device, amp_dtype):
                    self.output = model.decode_step(
                        self.static_input,
                        state=state,
                        position=self.static_position,
                    )
            capture_succeeded = True
        finally:
            _fused_gdn_step = gdn_step
            if capture_succeeded:
                _restore_graph_capture_state(state, capture_state)

    def step(self, token: torch.Tensor, position: int) -> torch.Tensor:
        self.static_input.copy_(token)
        self.static_position.fill_(position)
        self.graph.replay()
        return self.output.logits[:, -1, :]


def get_tokenizer():
    cache_dir = _REPO_ROOT / ".hf_cache"
    snapshot = list(cache_dir.glob("models--Qwen--Qwen3.5-0.8B-Base/snapshots/*"))
    if snapshot and (snapshot[0] / "tokenizer.json").is_file():
        try:
            from tokenizers import Tokenizer
            raw = Tokenizer.from_file(str(snapshot[0] / "tokenizer.json"))
            class FastTok:
                def __init__(self, tok):
                    self.tok = tok
                    self._chat_tokenizer = None
                    eos = tok.token_to_id("<|endoftext|>")
                    self.eos_token_id = 248044 if eos is None else int(eos)
                def encode(self, text):
                    # Match training exactly: raw text, no injected special tokens.
                    return list(self.tok.encode(text, add_special_tokens=False).ids)
                def decode(self, ids):
                    return self.tok.decode(ids)
                def apply_chat_template(self, messages, **kwargs):
                    if self._chat_tokenizer is None:
                        from transformers import AutoTokenizer
                        self._chat_tokenizer = AutoTokenizer.from_pretrained(
                            "Qwen/Qwen3.5-0.8B-Base",
                            cache_dir=str(cache_dir),
                            use_fast=True,
                            local_files_only=True,
                        )
                    return self._chat_tokenizer.apply_chat_template(messages, **kwargs)
            return FastTok(raw)
        except Exception:
            pass

    # Keep this standalone script independent of the repository data package
    # while preserving the same tokenizer loader and vocabulary guard.
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.5-0.8B-Base",
            cache_dir=str(cache_dir),
            use_fast=True,
            local_files_only=True,
        )
        base_vocab_size = int(getattr(tokenizer, "vocab_size", 0))
        effective_vocab_size = len(tokenizer)
        if base_vocab_size > 248_320 or effective_vocab_size > 248_320:
            raise ValueError(
                f"loaded tokenizer vocabulary exceeds canonical V0 vocabulary 248320: "
                f"base={base_vocab_size}, effective={effective_vocab_size}"
            )
        return tokenizer
    except Exception as exc:
        raise RuntimeError(
            f"Tokenizer cache is unavailable at {cache_dir}; "
            "standalone sampling could not load Qwen3.5 locally."
        ) from exc


def _chat_template_ids(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    """Render a complete conversation with the tokenizer's chat template."""
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        raise RuntimeError("This checkpoint requires a tokenizer chat template for chat sampling.")
    try:
        rendered = apply_template(messages, tokenize=True, add_generation_prompt=True)
    except Exception as exc:
        raise RuntimeError(
            "The tokenizer could not render the system/user/assistant conversation."
        ) from exc
    if hasattr(rendered, "__contains__") and "input_ids" in rendered:
        rendered = rendered["input_ids"]
    if torch.is_tensor(rendered):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], list):
        if len(rendered) != 1:
            raise RuntimeError("The tokenizer returned multiple chat sequences for one conversation.")
        rendered = rendered[0]
    ids = [int(token) for token in rendered]
    if not ids:
        raise RuntimeError("The tokenizer produced an empty chat prompt.")
    return ids


def _encode_messages(model, tokenizer, messages: list[dict[str, str]]) -> list[int]:
    """Encode system/user/assistant turns using the trained conversation boundary."""
    if not messages:
        raise ValueError("At least one conversation message is required.")
    try:
        return _chat_template_ids(tokenizer, messages)
    except Exception:
        if getattr(model, "_distillation_chat_format", False):
            raise
        plain = "\n\n".join(
            f"{message['role'].capitalize()}: {message.get('content', '')}"
            for message in messages
        )
        try:
            return [int(token) for token in tokenizer.encode(plain, add_special_tokens=False)]
        except TypeError:
            return [int(token) for token in tokenizer.encode(plain)]


def _encode_prompt(model, tokenizer, text: str) -> list[int]:
    """Encode a single user prompt using the same conversation path as the GUI."""
    if getattr(model, "_distillation_chat_format", False):
        try:
            return _encode_messages(model, tokenizer, [{"role": "user", "content": text}])
        except Exception as exc:
            raise RuntimeError(
                "This distillation checkpoint requires the Qwen chat template "
                "for sampling, but it could not be rendered locally."
            ) from exc
    try:
        return [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]
    except TypeError:
        return [int(token) for token in tokenizer.encode(text)]


def _stop_token_ids(tokenizer) -> frozenset[int]:
    """Return all valid generation terminators for this tokenizer.

    Distillation examples use Qwen's chat template, whose assistant-turn
    terminator is ``<|im_end|>``.  It is distinct from the base tokenizer's
    ``<|endoftext|>`` EOS token, so checking only ``eos_token_id`` can make
    generation continue past the end of the assistant response.
    """
    stop_ids: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        stop_ids.add(int(eos))

    raw = getattr(tokenizer, "tok", None)
    raw_token_to_id = getattr(raw, "token_to_id", None)
    converter = getattr(tokenizer, "convert_tokens_to_ids", None)
    for token in ("<|im_end|>", "<|eot_id|>", "<|endoftext|>"):
        token_id = None
        if callable(raw_token_to_id):
            token_id = raw_token_to_id(token)
        elif callable(converter):
            token_id = converter(token)
        if token_id is not None:
            stop_ids.add(int(token_id))
    return frozenset(stop_ids or {248044})


def _checkpoint_is_readable(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if not names:
                return False
            root = names[0].split("/", 1)[0]
            archive.read(f"{root}/byteorder")
            archive.read(f"{root}/version")
            archive.read(f"{root}/data.pkl")
        return True
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return False


def find_checkpoint():
    # Ignore partial transfers and compare completed backups with latest.pt.
    candidates: set[Path] = set()
    candidates.add(_REPO_ROOT / "runs" / "a5000_moe24" / "latest.pt")
    backup_dirs = [
        _REPO_ROOT / "runpod_backups" / "a5000_moe24",
        _REPO_ROOT / "runpod_archive" / "a5000_moe24",
    ]
    archive_dir = os.environ.get("LLMY_CHECKPOINT_ARCHIVE")
    if archive_dir:
        backup_dirs.append(Path(archive_dir).expanduser())
    for backup_dir in backup_dirs:
        candidates.update(backup_dir.glob("checkpoint_*.pt"))
    valid = [
        path for path in candidates
        if path.is_file() and not path.name.endswith(".partial") and _checkpoint_is_readable(path)
    ]
    return max(valid, key=lambda path: path.stat().st_mtime) if valid else None


def _resolve_config(config_path: str | None) -> Path:
    if config_path:
        path = Path(config_path)
        if not path.is_absolute():
            path = _REPO_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Config not found: {path}")
        return path
    for path in (
        _REPO_ROOT / "runpod" / "config_a5000_moe24.yaml",
        _REPO_ROOT / "runpod" / "config_a5000.yaml",
    ):
        if path.is_file():
            return path
    raise FileNotFoundError("No a5000_moe24 or a5000 sampler config is available")


def load_model(ckpt_path: Path, device: torch.device, dtype: torch.dtype,
               config_path: str | None = None):
    """Load weights and prefer the architecture config embedded in the checkpoint."""
    print(f"Loading checkpoint: {ckpt_path.name} ({ckpt_path.stat().st_size / 1e9:.2f} GB)...")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    embedded = ckpt.get("config") if isinstance(ckpt, dict) else None
    if embedded is not None:
        cfg = V6Config.from_mapping(embedded)
        cfg_source = "checkpoint"
        if config_path:
            external_path = _resolve_config(config_path)
            external_cfg = V6Config.from_mapping(load_config(str(external_path)))
            if external_cfg != cfg:
                raise RuntimeError(
                    "Explicit sampler config does not match checkpoint config. "
                    "Refusing to sample with different architecture semantics."
                )
    else:
        cfg_path = _resolve_config(config_path)
        cfg = V6Config.from_mapping(load_config(str(cfg_path)))
        cfg_source = str(cfg_path)
        print(
            "WARNING: checkpoint has no embedded config; using external YAML. "
            "Architecture parity is not guaranteed."
        )

    # Keep parameters FP32 like training. Autocast controls execution precision.
    model = LoopedHybridMoEForCausalLM(cfg).to(device=device)
    data_cfg = embedded.get("data") if isinstance(embedded, dict) else None
    after_tokens = data_cfg.get("after_tokens") if isinstance(data_cfg, dict) else None
    model._distillation_chat_format = bool(
        isinstance(after_tokens, dict)
        and str(after_tokens.get("loader", "")).lower() == "distillation"
    )
    state = ckpt.get("model", ckpt)
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    try:
        model.load_state_dict(clean_state, strict=True)
    except RuntimeError as exc:
        # Checkpoints written before adaptive halting was added legitimately do
        # not contain this optional head. Keep every other parameter strict so
        # an unrelated architecture mismatch still aborts sampling.
        missing, unexpected = model.load_state_dict(clean_state, strict=False)
        allowed_missing = {"recurrent_halt_head.weight", "recurrent_halt_head.bias"}
        if set(missing) != allowed_missing or unexpected:
            raise RuntimeError(
                f"Checkpoint/model mismatch. Sampling blocked; config source={cfg_source}: {exc}"
            ) from exc
        model._legacy_no_halt_head = True
        print("Legacy checkpoint: adaptive halting head absent; using the trained fixed loop count.")

    model.eval()
    step = ckpt.get("step", "N/A") if isinstance(ckpt, dict) else "N/A"
    tokens = ckpt.get("token_count", "N/A") if isinstance(ckpt, dict) else "N/A"
    print(
        f"Model loaded (Step: {step}, Tokens: {tokens}, Device: {device}, "
        f"AMP: {dtype}, Weights: FP32)"
    )
    print(f"Architecture config: {cfg_source}")
    if isinstance(step, int) and step == 0:
        print("WARNING: this is a step-0 checkpoint; weights are untrained.")
    return model, step


def set_fast_mhc(model: torch.nn.Module, enabled: bool = True) -> None:
    """Select the optional constant-map mHC inference path."""
    if enabled and not any(hasattr(module, "set_fast_inference") for module in model.modules()):
        raise RuntimeError(
            "--fast-mhc is not supported by the live runtime; use dynamic mHC "
            "so sampling matches training"
        )
    for module in model.modules():
        if isinstance(module, MultiStreamHyperConnection):
            setter = getattr(module, "set_fast_inference", None)
            if setter is not None:
                setter(enabled)





def _adaptive_cfg(model):
    return getattr(getattr(model, "config", None), "adaptive_loops", None)


def configure_adaptive_compute(model, args) -> bool:
    """Use the model's learned adaptive-halting configuration by default."""
    uncapped_recursion = bool(getattr(args, "uncapped_recursion", False))
    cfg = _adaptive_cfg(model)
    if cfg is None:
        if uncapped_recursion:
            raise RuntimeError("Uncapped recursion requires a learned halt head.")
        print("Adaptive compute: unavailable in this model build")
        return False

    if getattr(model, "_legacy_no_halt_head", False):
        if any(value is not None for value in (
            args.min_loops, args.max_loops, args.halt_threshold,
            args.halt_temperature, args.fixed_loops,
        )) or getattr(args, "adaptive", False) or uncapped_recursion:
            raise RuntimeError(
                "This legacy checkpoint has no learned halting head; loop overrides are unavailable."
            )
        cfg.enabled = False
        cfg.learned_halting = False
        cfg.min_loops = cfg.max_loops = int(getattr(model.config, "num_loops", 1))
        print(f"Adaptive compute: disabled for legacy checkpoint; fixed {cfg.max_loops} loops")
        return False

    if uncapped_recursion and (args.max_loops is not None or args.fixed_loops is not None):
        raise ValueError("--uncapped-recursion cannot be combined with --max-loops or --fixed-loops")

    if args.min_loops is not None:
        cfg.min_loops = int(args.min_loops)
    if args.max_loops is not None:
        cfg.max_loops = int(args.max_loops)
    if args.halt_threshold is not None:
        cfg.halt_threshold = float(args.halt_threshold)
    if args.halt_temperature is not None:
        cfg.halt_temperature = float(args.halt_temperature)

    if args.fixed_loops is not None:
        fixed = int(args.fixed_loops)
        if fixed < 1:
            raise ValueError("--fixed-loops must be >= 1")
        cfg.min_loops = fixed
        cfg.max_loops = fixed

    if not 1 <= int(cfg.min_loops) <= int(cfg.max_loops):
        raise ValueError(
            f"invalid adaptive loop range: {cfg.min_loops}..{cfg.max_loops}"
        )
    cfg.enabled = True
    if args.fixed_loops is None:
        cfg.learned_halting = True

    active = bool(cfg.enabled and getattr(cfg, "learned_halting", False) and args.fixed_loops is None)
    if uncapped_recursion and not active:
        raise RuntimeError("--uncapped-recursion requires learned adaptive halting.")
    model._uncapped_recursion = uncapped_recursion
    mode = "learned halting" if active else f"fixed {cfg.min_loops} loop(s)"
    loop_range = f"{cfg.min_loops}-unbounded" if uncapped_recursion else f"{cfg.min_loops}-{cfg.max_loops}"
    print(
        "Adaptive compute: "
        f"{mode} | range={loop_range} | "
        f"threshold={float(getattr(cfg, 'halt_threshold', 0.5)):.3f} | "
        f"temperature={float(getattr(cfg, 'halt_temperature', 1.0)):.3f}"
    )
    return active


def _small_detached(value):
    """Retain only tiny halting telemetry tensors without synchronizing decode."""
    if torch.is_tensor(value):
        if value.numel() <= 64:
            return value.detach()
        return None
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, (list, tuple)) and len(value) <= 16:
        vals = [_small_detached(v) for v in value]
        return vals if all(v is not None for v in vals) else None
    return None


class AdaptiveTrace:
    """Collect learned-halting and loop-attention telemetry.

    Different model revisions have used slightly different diagnostic names,
    so this intentionally discovers loop-count / halt-probability fields by
    semantics instead of depending on one exact output dataclass revision.
    The decode path now reports the actual number of recurrent passes per
    token, after early stopping, rather than assuming the configured maximum.
    """

    def __init__(self, model):
        self.cfg = _adaptive_cfg(model)
        self.uncapped_recursion = bool(getattr(model, "_uncapped_recursion", False))
        self.records: list[dict[str, object]] = []

    @property
    def threshold(self) -> float:
        return float(getattr(self.cfg, "halt_threshold", 0.5)) if self.cfg is not None else 0.5

    def observe(self, output, *, phase: str) -> None:
        if self.cfg is None or output is None:
            return

        candidates: dict[str, object] = {}
        diagnostics = getattr(output, "diagnostics", None)
        if isinstance(diagnostics, dict):
            candidates.update(diagnostics)

        # Direct dataclass/object fields take precedence over generic diagnostics.
        try:
            attrs = vars(output)
        except TypeError:
            attrs = {}
        for key, value in attrs.items():
            lk = key.lower()
            if "halt" in lk or "loop" in lk or "ponder" in lk or "continue" in lk:
                candidates[key] = value

        state = getattr(output, "state", None)
        if state is not None:
            try:
                state_attrs = vars(state)
            except TypeError:
                state_attrs = {}
            for key, value in state_attrs.items():
                lk = key.lower()
                if "halt" in lk or ("loop" in lk and ("count" in lk or "execut" in lk or "used" in lk)):
                    candidates[f"state.{key}"] = value

        record: dict[str, object] = {"phase": phase}
        for key, value in candidates.items():
            lk = key.lower()
            keep = False
            if "halt" in lk or "ponder" in lk or "continue" in lk:
                keep = True
            if "loop" in lk and any(word in lk for word in ("count", "used", "execut", "active", "expected")):
                keep = True
            if keep:
                detached = _small_detached(value)
                if detached is not None:
                    record[key] = detached
        if len(record) > 1:
            self.records.append(record)

    @staticmethod
    def _to_python(value):
        if torch.is_tensor(value):
            value = value.float().cpu()
            if value.numel() == 1:
                return float(value.item())
            return value.reshape(-1).tolist()
        if isinstance(value, list):
            out = []
            for v in value:
                pv = AdaptiveTrace._to_python(v)
                if isinstance(pv, list):
                    out.extend(pv)
                else:
                    out.append(pv)
            return out
        return value

    def _extract_loop_count(self, record: dict[str, object]) -> int | None:
        # Prefer an explicit executed-loop count from the model.
        for key, value in record.items():
            lk = key.lower()
            if "loop" in lk and any(word in lk for word in ("execut", "used", "count")):
                py = self._to_python(value)
                if isinstance(py, (int, float)):
                    v = int(round(float(py)))
                    if v >= 1:
                        return v

        # Fallback: if the model exposes per-loop halting probabilities, infer
        # the first loop crossing the configured threshold after min_loops.
        for key, value in record.items():
            lk = key.lower()
            if "halt" in lk and "prob" in lk:
                py = self._to_python(value)
                if isinstance(py, list) and py:
                    min_loops = int(getattr(self.cfg, "min_loops", 1))
                    max_loops = int(getattr(self.cfg, "max_loops", len(py)))
                    for i, prob in enumerate(py[:max_loops], start=1):
                        if i >= min_loops and float(prob) >= self.threshold:
                            return i
                    return min(max_loops, len(py))
        return None

    def summary(self, *, decode_only: bool = True) -> str | None:
        records = [r for r in self.records if (r.get("phase") == "decode" or not decode_only)]
        if not records:
            return None
        counts = [self._extract_loop_count(r) for r in records]
        counts = [c for c in counts if c is not None]
        if not counts:
            # We found adaptive diagnostics, just not a canonical count field.
            keys = sorted({k for r in records for k in r if k != "phase"})
            return "halting telemetry present: " + ", ".join(keys[:8])

        hist = Counter(counts)
        avg = sum(counts) / len(counts)
        max_loops = int(getattr(self.cfg, "max_loops", max(counts)))
        compute_frac = avg / max(max_loops, 1)
        dist = ", ".join(f"L{k}:{hist[k]}" for k in sorted(hist))
        residual_counts = []
        for record in records:
            value = record.get("loop_attention_residual_count")
            py = self._to_python(value) if value is not None else None
            if isinstance(py, (int, float)):
                residual_counts.append(float(py))
        residual_suffix = (
            f" | loop-attention residuals/step={sum(residual_counts) / len(residual_counts):.2f}"
            if residual_counts else ""
        )
        depth_label = "unbounded" if self.uncapped_recursion else str(max_loops)
        depth_fraction = "" if self.uncapped_recursion else f" ({compute_frac*100:.1f}% recurrent depth)"
        return (
            f"avg loops/token={avg:.2f}/{depth_label}{depth_fraction} | "
            f"distribution [{dist}]" + residual_suffix
        )

def set_moe_decode_mode(model: torch.nn.Module, mode: str) -> None:
    """Patch only batch-1 routed-expert dispatch for decode experiments.

    sparse:
        Use the model's original selected-expert dispatch.
    packed-hybrid:
        One packed gate/up projection over every routed expert, then execute only
        the selected experts' down projections. This trades some FLOPs for a much
        larger, regular GEMM while preserving exact top-k routing semantics.
    packed-dense:
        Evaluate every routed expert end-to-end, then multiply by a sparse route
        vector. No selected-weight gathers are needed; useful as a throughput
        baseline when there are only ~48 experts.

    Sequence/prefill dispatch (N>1) always uses the model's original path.
    """
    if mode not in {"sparse", "packed-hybrid", "packed-dense"}:
        raise ValueError(f"unknown MoE mode: {mode}")

    patched = 0
    for module in model.modules():
        # Duck-type SparseMoE so this script stays compatible with standalone/runtime variants.
        required = ("_dispatch", "_weights", "_gate_up_weights", "experts", "routed_experts")
        if not all(hasattr(module, name) for name in required):
            continue
        if hasattr(module, "_sample_original_dispatch"):
            original = module._sample_original_dispatch
        else:
            original = module._dispatch
            module._sample_original_dispatch = original

        if mode == "sparse":
            module._dispatch = original
            continue

        def _packed_dispatch(self, flat, indices, probabilities, *, _original=original, _mode=mode):
            # Keep normal training/prefill implementation. The optimization is specifically
            # for autoregressive batch-1 decode, where sparse dispatch overhead dominates.
            if flat.ndim != 2 or flat.shape[0] != 1:
                return _original(flat, indices, probabilities)

            route_ids = indices.reshape(-1).long()
            route_probs = probabilities.reshape(-1).to(flat.dtype)
            expert_count = int(self.routed_experts)

            w_gate, _, w_down = self._weights(
                self.experts, device=flat.device, dtype=flat.dtype
            )
            gate_up = self._gate_up_weights(
                self.experts, device=flat.device, dtype=flat.dtype
            )  # [E, 2H, D]

            # One large projection instead of gathering top-k gate/up matrices and
            # launching tiny BMMs. F.linear treats all expert rows as one packed matrix.
            e, two_h, d = gate_up.shape
            if e != expert_count:
                raise RuntimeError(f"packed MoE expert count mismatch: {e} != {expert_count}")
            packed_gate_up = gate_up.reshape(e * two_h, d)
            gu = F.linear(flat, packed_gate_up).view(e, two_h)
            h = two_h // 2
            act_all = F.silu(gu[:, :h]) * gu[:, h:]

            if _mode == "packed-hybrid":
                # Only the top-k down projections. The gather is now limited to W_down;
                # the much larger gate/up side is a single contiguous projection.
                act = act_all.index_select(0, route_ids)
                selected_down = w_down.index_select(0, route_ids)  # [K, D, H]
                out = torch.bmm(selected_down, act.unsqueeze(-1)).squeeze(-1)
                return (out * route_probs.unsqueeze(-1)).sum(0, keepdim=True)

            # Fully dense expert execution. Build a sparse route vector, but use it only
            # for the final reduction; inactive experts incur no Python/gather overhead.
            out_all = torch.bmm(w_down, act_all.unsqueeze(-1)).squeeze(-1)  # [E, D]
            route = flat.new_zeros((expert_count,))
            route.scatter_(0, route_ids, route_probs)
            return (out_all * route.unsqueeze(-1)).sum(0, keepdim=True)

        module._dispatch = types.MethodType(_packed_dispatch, module)
        patched += 1

    print(f"MoE decode mode: {mode}" + (f" ({patched} routed MoE modules patched)" if mode != "sparse" else ""))


def _sample_next_token(logits: torch.Tensor, generated: list[int], *, temp: float,
                       top_k: int, top_p: float, rep: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one token while keeping raw model log-probabilities for diagnostics."""
    raw_log_probs = F.log_softmax(logits.float(), dim=-1)
    work = logits.float().clone()

    if rep != 1.0 and generated:
        seen = torch.tensor(tuple(set(generated)), device=work.device, dtype=torch.long)
        seen_logits = work.index_select(1, seen)
        work.scatter_(
            1, seen[None, :],
            torch.where(seen_logits > 0, seen_logits / rep, seen_logits * rep),
        )

    if temp <= 1e-4:
        next_tok = torch.argmax(work, dim=-1, keepdim=True)
        model_logprob = raw_log_probs.gather(1, next_tok)
        return next_tok, model_logprob

    work = work / temp
    if top_k > 0:
        values, indices = torch.topk(work, min(top_k, work.size(-1)), dim=-1)
    else:
        values = work
        indices = torch.arange(work.size(-1), device=work.device).expand_as(work)

    if 0.0 < top_p < 1.0:
        values, order = torch.sort(values, descending=True, dim=-1)
        indices = torch.gather(indices, 1, order)
        probs = F.softmax(values, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        values = values.masked_fill(remove, -float("inf"))

    probs = F.softmax(values, dim=-1)
    choice = torch.multinomial(probs, 1)
    next_tok = indices.gather(1, choice)
    model_logprob = raw_log_probs.gather(1, next_tok)
    return next_tok, model_logprob


@torch.no_grad()
def full_recompute_sample(model, tokenizer, prompt: str, device, dtype, max_tokens=64,
                          temp=0.7, top_k=50, top_p=0.9, rep=1.1):
    """Gold-standard slow generation: rebuild a clean state from the whole prefix each token."""
    sequence = torch.tensor([_encode_prompt(model, tokenizer, prompt)], dtype=torch.long, device=device)
    if sequence.numel() == 0:
        return

    generated: list[int] = []
    logprobs: list[float] = []
    stop_token_ids = _stop_token_ids(tokenizer)
    adaptive_trace = AdaptiveTrace(model)
    print("Prediction: ", end="", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    for _ in range(max_tokens):
        with _autocast_context(device, dtype):
            out = model(sequence, collect_diagnostics=False, last_logits_only=True)
        adaptive_trace.observe(out, phase="decode")
        logits = out.logits[:, -1, :].float()
        next_tok, lp = _sample_next_token(
            logits, generated, temp=temp, top_k=top_k, top_p=top_p, rep=rep
        )
        tok_id = int(next_tok.item())
        generated.append(tok_id)
        logprobs.append(float(lp.item()))
        print(tokenizer.decode([tok_id]), end="", flush=True)
        sequence = torch.cat((sequence, next_tok), dim=1)
        if tok_id in stop_token_ids:
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    n = len(generated)
    nll = -sum(logprobs) / max(n, 1)
    print("\n" + "-" * 70)
    print(f"Full recompute: {n} tokens in {dt:.2f}s | {n / max(dt, 1e-9):.2f} tok/s")
    print(f"Sample NLL/PPL: {nll:.4f}/{math.exp(min(nll, 50.0)):.2f}")
    adaptive_summary = adaptive_trace.summary()
    if adaptive_summary:
        print(f"Adaptive: {adaptive_summary}")
    print("-" * 70 + "\n")


@torch.no_grad()
def parity_check(model, tokenizer, text: str, device, dtype, max_positions: int = 32):
    """Compare full-sequence, clean-prefix, and incremental logits on fixed tokens.

    A(full) vs B(prefix) detects non-causal/full-forward discrepancies.
    B(prefix) vs C(decode) detects incremental cache/state discrepancies.
    """
    ids = torch.tensor([_encode_prompt(model, tokenizer, text)], dtype=torch.long, device=device)
    length = min(ids.shape[1], int(max_positions))
    if length < 2:
        print("Parity check needs at least two tokenizer tokens.")
        return
    ids = ids[:, :length]

    with _autocast_context(device, dtype):
        full = model(ids, collect_diagnostics=False, last_logits_only=False)
        inc = model(ids[:, :1], collect_diagnostics=False, last_logits_only=True)
    full_logits = full.logits.float()
    state = inc.state
    decode_logits = inc.logits[:, -1, :].float()

    print(f"\nTeacher-forced parity check: {length} tokens")
    print("pos | cos(full,prefix) | cos(prefix,decode) | maxd full/prefix | maxd prefix/decode | top1 match")
    print("-" * 108)
    worst_fp = 1.0
    worst_pd = 1.0

    for target_pos in range(1, length):
        prefix = ids[:, :target_pos]
        with _autocast_context(device, dtype):
            prefix_out = model(prefix, collect_diagnostics=False, last_logits_only=True)
        prefix_logits = prefix_out.logits[:, -1, :].float()
        sequence_logits = full_logits[:, target_pos - 1, :]

        cos_fp = float(F.cosine_similarity(sequence_logits, prefix_logits, dim=-1).item())
        cos_pd = float(F.cosine_similarity(prefix_logits, decode_logits, dim=-1).item())
        d_fp = float((sequence_logits - prefix_logits).abs().max().item())
        d_pd = float((prefix_logits - decode_logits).abs().max().item())
        top_full = int(sequence_logits.argmax(-1).item())
        top_prefix = int(prefix_logits.argmax(-1).item())
        top_decode = int(decode_logits.argmax(-1).item())
        match = "Y" if top_full == top_prefix == top_decode else f"N {top_full}/{top_prefix}/{top_decode}"
        print(f"{target_pos:3d} | {cos_fp:16.8f} | {cos_pd:18.8f} | {d_fp:16.6g} | {d_pd:18.6g} | {match}")
        worst_fp = min(worst_fp, cos_fp)
        worst_pd = min(worst_pd, cos_pd)

        if target_pos + 1 < length:
            token = ids[:, target_pos:target_pos + 1]
            position = torch.full((ids.shape[0],), target_pos, device=device, dtype=torch.long)
            with _autocast_context(device, dtype):
                inc = model.decode_step(token, state=state, position=position)
            decode_logits = inc.logits[:, -1, :].float()

    print("-" * 108)
    print(f"Worst cosine: full/prefix={worst_fp:.8f}, prefix/decode={worst_pd:.8f}\n")


@torch.no_grad()
def sample(model, tokenizer, prompt: str, device, dtype, max_tokens=64, temp=0.7, top_k=50, top_p=0.9, rep=1.1,
           show_prompt_ppl=False, use_cuda_graph=False, *, messages=None, emit_output=True,
           ignore_eos=False, collect_token_stats=False):
    encoded = _encode_messages(model, tokenizer, messages) if messages is not None else _encode_prompt(model, tokenizer, prompt)
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    if input_ids.numel() == 0:
        return None

    # Prompt PPL is part of the default report; pass --no-prompt-ppl to skip it.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Generation loop
    if emit_output:
        sys.stdout.write("Prediction: ")
        sys.stdout.flush()

    gen_tokens = []
    gen_logprobs: list[float] = []
    token_stats: list[dict[str, object]] = []
    adaptive_trace = AdaptiveTrace(model)
    stop_token_ids = _stop_token_ids(tokenizer)
    blocked_stop_ids = sorted(stop_token_ids) if ignore_eos else []
    t0 = time.perf_counter()
    ttft = None
    graph_decoder = None
    prompt_length = input_ids.shape[1]

    # Prefill once. Subsequent tokens use the model's recurrent decode path
    # and do not rerun the prompt or a growing 256-token window.
    # The sampler only needs the final prompt position.  Avoid allocating the
    # full [batch, prompt, vocab] projection unless perplexity was requested.
    prefill_started = time.perf_counter()
    with _autocast_context(device, dtype):
        prefill = model(
            input_ids,
            collect_diagnostics=False,
            last_logits_only=not show_prompt_ppl,
        )
    if collect_token_stats and device.type == "cuda":
        torch.cuda.synchronize(device)
    pending_forward_ms = (time.perf_counter() - prefill_started) * 1000.0
    pending_diagnostics = getattr(prefill, "diagnostics", {}) or {}
    adaptive_trace.observe(prefill, phase="prefill")
    state = prefill.state
    logits = prefill.logits[:, -1, :].float()
    prompt_ppl: float | None = None
    if show_prompt_ppl and input_ids.size(1) > 1:
        try:
            shift_logits = prefill.logits[:, :-1, :].float()
            shift_labels = input_ids[:, 1:]
            loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            prompt_ppl = math.exp(min(float(loss.item()), 50.0))
        except Exception:
            pass
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gen_t0 = time.perf_counter()

    printed = []
    for i in range(max_tokens):
        token_started = time.perf_counter()
        # Keep the unmodified model distribution for honest confidence/PPL
        # reporting. Sampling controls below may apply repetition penalties,
        # temperature, top-k, and nucleus truncation to a separate tensor.
        raw_log_probs = F.log_softmax(logits.float(), dim=-1)
        if blocked_stop_ids:
            logits = logits.clone()
            logits[:, blocked_stop_ids] = -float("inf")
        # Repetition penalty
        if rep != 1.0 and gen_tokens:
            seen = torch.tensor(tuple(set(gen_tokens)), device=device, dtype=torch.long)
            seen_logits = logits.index_select(1, seen)
            logits.scatter_(1, seen[None, :], torch.where(
                seen_logits > 0, seen_logits / rep, seen_logits * rep
            ))

        # Sampling
        if temp <= 1e-4:
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temp
            if top_k > 0:
                top_values, top_indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            else:
                top_values, top_indices = logits, torch.arange(logits.size(-1), device=device).expand_as(logits)
            # Nucleus filtering only needs to sort the already selected top-k.
            if 0.0 < top_p < 1.0:
                sorted_values, order = torch.sort(top_values, descending=True, dim=-1)
                sorted_indices = torch.gather(top_indices, 1, order)
                sorted_probs = F.softmax(sorted_values, dim=-1)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                remove = cumulative > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_values = sorted_values.masked_fill(remove, -float("inf"))
                probs = F.softmax(sorted_values, dim=-1)
                choice = torch.multinomial(probs, 1)
                next_tok = sorted_indices.gather(1, choice)
                prob_tensor = probs.gather(1, choice)
            else:
                probs = F.softmax(top_values, dim=-1)
                choice = torch.multinomial(probs, 1)
                next_tok = top_indices.gather(1, choice)
                prob_tensor = probs.gather(1, choice)
            model_logprob_tensor = raw_log_probs.gather(1, next_tok)

        # Move both scalar values across the device boundary together.  Each
        # individual .item() can synchronize a CUDA stream, which is costly
        # when the model emits one token at a time.
        if collect_token_stats:
            selected_logprob = raw_log_probs.gather(1, next_tok).reshape(-1)
            top_logprob, top_token_id = raw_log_probs.max(dim=-1)
            raw_probabilities = raw_log_probs.exp()
            entropy = -torch.where(
                raw_probabilities > 0,
                raw_probabilities * raw_log_probs,
                torch.zeros_like(raw_probabilities),
            ).sum(dim=-1)
            packed_stats = torch.stack((
                next_tok.reshape(-1).float(),
                selected_logprob,
                top_token_id.reshape(-1).float(),
                top_logprob.reshape(-1),
                entropy.reshape(-1),
            ), dim=1)[0].detach().float().cpu().tolist()
            tok_id = int(packed_stats[0])
            token_logprob = float(packed_stats[1])
            raw_top1_token_id = int(packed_stats[2])
            raw_top1_logprob = float(packed_stats[3])
            token_entropy = float(packed_stats[4])
        elif temp <= 1e-4:
            tok_id = int(next_tok.item())
            token_logprob = float(raw_log_probs.gather(1, next_tok).item())
        else:
            tok_id, token_logprob = torch.cat(
                (next_tok.reshape(1), model_logprob_tensor.reshape(1))
            ).float().cpu().tolist()
            tok_id = int(tok_id)
            token_logprob = float(token_logprob)
        if ttft is None:
            ttft = time.perf_counter() - t0

        gen_tokens.append(tok_id)
        gen_logprobs.append(token_logprob)

        token_piece = tokenizer.decode([tok_id])
        printed.append(token_piece)
        if collect_token_stats:
            loop_count = pending_diagnostics.get("executed_loop_count") if isinstance(pending_diagnostics, dict) else None
            if torch.is_tensor(loop_count):
                loop_count = loop_count.detach().float().max().item()
            elif isinstance(loop_count, (list, tuple)) and loop_count:
                loop_count = loop_count[0]
            if loop_count is None:
                loop_count = getattr(_adaptive_cfg(model), "max_loops", None)
            residual_count = pending_diagnostics.get("loop_attention_residual_count") if isinstance(pending_diagnostics, dict) else None
            if torch.is_tensor(residual_count):
                residual_count = residual_count.detach().float().max().item()
            token_stats.append({
                "token_index": len(gen_tokens),
                "token_id": tok_id,
                "token_piece": token_piece,
                "raw_logprob": token_logprob,
                "nll_nats": -token_logprob,
                "token_perplexity": math.exp(min(-token_logprob, 50.0)),
                "selected_token_probability": math.exp(min(token_logprob, 0.0)),
                "raw_top1_token_id": raw_top1_token_id,
                "raw_top1_probability": math.exp(min(raw_top1_logprob, 0.0)),
                "raw_entropy_nats": token_entropy,
                "selected_is_raw_top1": tok_id == raw_top1_token_id,
                "forward_latency_ms": pending_forward_ms,
                "sampling_latency_ms": (time.perf_counter() - token_started) * 1000.0,
                "loops_executed": int(loop_count) if loop_count is not None else None,
                "loop_attention_residual_count": residual_count,
                "halt_continue_probability": _small_detached(
                    pending_diagnostics.get("halt_continue_probability")
                    if isinstance(pending_diagnostics, dict) else None
                ),
            })
        if len(printed) >= 8:
            if emit_output:
                sys.stdout.write("".join(printed))
                sys.stdout.flush()
            printed.clear()

        if tok_id in stop_token_ids:
            break

        if i + 1 < max_tokens:
            decode_position = prompt_length + i
            if (
                graph_decoder is None and use_cuda_graph and device.type == "cuda"
                and input_ids.shape[0] == 1
            ):
                try:
                    _prepare_cuda_graph_state(
                        state,
                        capacity=max(
                            prompt_length,
                            min(
                                int(getattr(model.config, "max_position_embeddings", 4096)),
                                prompt_length + int(max_tokens),
                            ),
                        ),
                        model=model,
                    )
                    graph_decoder = _CudaGraphDecoder(
                        model, state, next_tok, decode_position, dtype
                    )
                    logits = graph_decoder.step(next_tok, decode_position).float()
                except Exception as exc:
                    # Restore the ordinary active-prefix path if graph capture
                    # is unavailable on a particular driver/runtime.
                    _restore_eager_state(state)
                    graph_decoder = False
                    if emit_output:
                        print(f"CUDA Graph decode unavailable: {exc}")
                    with _autocast_context(device, dtype):
                        prefill = model.decode_step(
                            next_tok, state=state,
                            position=torch.full((input_ids.shape[0],), decode_position, device=device, dtype=torch.long),
                        )
                    adaptive_trace.observe(prefill, phase="decode")
                    logits = prefill.logits[:, -1, :].float()
            elif graph_decoder:
                logits = graph_decoder.step(next_tok, decode_position).float()
            else:
                decode_started = time.perf_counter()
                with _autocast_context(device, dtype):
                    prefill = model.decode_step(
                        next_tok, state=state,
                        position=torch.full((input_ids.shape[0],), decode_position, device=device, dtype=torch.long),
                    )
                if collect_token_stats and device.type == "cuda":
                    torch.cuda.synchronize(device)
                pending_forward_ms = (time.perf_counter() - decode_started) * 1000.0
                pending_diagnostics = getattr(prefill, "diagnostics", {}) or {}
                adaptive_trace.observe(prefill, phase="decode")
                logits = prefill.logits[:, -1, :].float()

    if printed and emit_output:
        sys.stdout.write("".join(printed))
        sys.stdout.flush()

    # Report steady-state decode speed separately from prompt prefill latency.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_total = time.perf_counter() - gen_t0
    n_gen = len(gen_tokens)
    tps = n_gen / t_total if t_total > 0 else 0.0
    lat = (t_total / n_gen) * 1000.0 if n_gen > 0 else 0.0
    sample_nll = (-sum(gen_logprobs) / n_gen) if gen_logprobs else 0.0
    sample_ppl = math.exp(min(sample_nll, 50.0)) if gen_logprobs else 0.0
    avg_conf = (sum(math.exp(min(lp, 0.0)) for lp in gen_logprobs) / n_gen) * 100.0 if gen_logprobs else 0.0

    if emit_output:
        print("\n" + "-" * 70)
        print(f"Stats: {n_gen} tokens in {t_total:.2f}s | Speed: {tps:.2f} tok/s ({lat:.1f} ms/tok) | TTFT: {(ttft or 0)*1000:.1f}ms")
    p_str = f"{prompt_ppl:.2f}" if prompt_ppl is not None else "disabled"
    if emit_output:
        print(f"Theory: Prompt PPL: {p_str} | Sample NLL/PPL: {sample_nll:.4f}/{sample_ppl:.2f} | Raw Confidence: {avg_conf:.1f}%")
    adaptive_summary = adaptive_trace.summary()
    if adaptive_summary and emit_output:
        print(f"Adaptive: {adaptive_summary}")
    if device.type == "cuda":
        vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        total_vram = torch.cuda.get_device_properties(device).total_memory / (1024**2)
        if emit_output:
            print(f"Hardware: Peak VRAM: {vram_mb:.0f} MiB / {total_vram:.0f} MiB ({vram_mb/total_vram*100:.1f}%)")
    if emit_output:
        print("-" * 70 + "\n")

    content_tokens = gen_tokens[:-1] if gen_tokens and gen_tokens[-1] in stop_token_ids else gen_tokens
    try:
        generated_text = tokenizer.decode(content_tokens, skip_special_tokens=True)
    except TypeError:
        generated_text = tokenizer.decode(content_tokens)
    for marker in ("<|im_end|>", "<|eot_id|>", "<|endoftext|>"):
        generated_text = generated_text.replace(marker, "")
    return {
        "text": generated_text,
        "tokens": n_gen,
        "tps": tps,
        "ttft_ms": (ttft or 0.0) * 1000.0,
        "prompt_ppl": prompt_ppl,
        "sample_ppl": sample_ppl,
        "confidence": avg_conf,
        "sample_nll": sample_nll,
        "adaptive": adaptive_summary,
        "token_stats": token_stats if collect_token_stats else None,
        "decode_seconds": t_total,
        "average_token_latency_ms": lat,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str)
    parser.add_argument("--config", type=str, help="fallback config for old checkpoints; new checkpoints use their embedded config")
    parser.add_argument("-p", "--prompt", type=str)
    parser.add_argument("-n", "--max-tokens", type=int, default=64)
    parser.add_argument(
        "--ignore-eos", action="store_true",
        help="suppress stop tokens and continue until --max-tokens is reached",
    )
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--rep", type=float, default=1.1)
    parser.add_argument("--no-prompt-ppl", action="store_true", help="skip prompt perplexity calculation")
    parser.add_argument(
        "--prompt-ppl", action="store_true",
        help="explicitly request prompt perplexity (the default; retained for CLI compatibility)",
    )
    parser.add_argument("--cuda-graph", action="store_true", help="EXPERIMENTAL: fixed-control-flow decode only; incompatible with learned adaptive halting")
    parser.add_argument("--full-recompute", action="store_true", help="recompute the complete prefix every token; slow correctness oracle")
    parser.add_argument("--parity-check", action="store_true", help="compare full-sequence, clean-prefix and decode_step logits")
    parser.add_argument("--parity-tokens", type=int, default=32)
    parser.add_argument("--min-loops", type=int, default=None, help="override adaptive minimum recurrent loops")
    parser.add_argument("--max-loops", type=int, default=None, help="override adaptive maximum recurrent loops")
    parser.add_argument(
        "--uncapped-recursion", action="store_true",
        help="repeat recurrent inference until the learned halt head stops every active token",
    )
    parser.add_argument("--halt-threshold", type=float, default=None, help="override learned halting threshold")
    parser.add_argument("--halt-temperature", type=float, default=None, help="override learned halting temperature")
    parser.add_argument("--fixed-loops", type=int, default=None, help="debug only: disable adaptive depth by forcing exactly N recurrent loops")
    parser.add_argument(
        "--adaptive", action="store_true",
        help="explicitly enable learned adaptive halting, even if the checkpoint was trained at fixed depth",
    )
    parser.add_argument(
        "--moe-mode", choices=("sparse", "packed-hybrid", "packed-dense"),
        default="sparse",
        help="batch-1 routed MoE implementation to benchmark",
    )
    parser.add_argument("--fast-mhc", action="store_true", help="use constant mHC maps (changes inference behavior)")
    parser.add_argument("--greedy", action="store_true", help="use argmax decoding for deterministic checkpoint evaluation")
    parser.add_argument("--compile-sinkhorn", action="store_true", help="fuse exact dynamic mHC Sinkhorn iterations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1337, help="fixed sampling/model initialization seed")
    # Match the trainer's CUDA autocast policy. The user can request FP32 for
    # numerical comparison, but inference should use the trained CUDA dtype.
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--gui", action="store_true",
                        help="launch the browser chat interface instead of the CLI sampler")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="GUI bind address (used with --gui)")
    parser.add_argument("--port", type=int, default=7860,
                        help="GUI port (used with --gui)")
    args = parser.parse_args()

    if args.gui:
        if args.prompt:
            parser.error("--gui cannot be combined with --prompt")
        gui_script = _REPO_ROOT / "chat_gui.py"
        gui_cmd = [
            sys.executable, str(gui_script),
            "--backend", "sample_standalone",
            "--device", args.device,
            "--host", args.host,
            "--port", str(args.port),
            "--max-tokens", str(args.max_tokens),
            "--temp", str(args.temp),
            "--top-k", str(args.top_k),
            "--top-p", str(args.top_p),
            "--repetition", str(args.rep),
        ]
        if args.checkpoint:
            gui_cmd.extend(["--checkpoint", args.checkpoint])
        if args.config:
            gui_cmd.extend(["--config", args.config])
        if args.cuda_graph:
            gui_cmd.append("--cuda-graph")
        raise SystemExit(subprocess.call(gui_cmd, cwd=str(_REPO_ROOT)))

    if args.cuda_graph:
        print("WARNING: CUDA Graph decode is experimental; use eager decode for parity/correctness tests.")

    # Seed before model construction and keep it fixed for the whole process.
    # This makes multinomial sampling reproducible while preserving the same
    # random sequence across prompts entered in one interactive session.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ckpt = Path(args.checkpoint) if args.checkpoint else find_checkpoint()
    if not ckpt or not ckpt.is_file():
        print("No valid completed a5000_moe24 checkpoint is available yet; the newest transfer may still be partial or corrupt.")
        sys.exit(1)

    device = torch.device(args.device)
    if args.dtype:
        dtype_name = args.dtype
    elif device.type == "cuda":
        # is_bf16_supported() can report software support on older cards, but
        # Pascal lacks native BF16 Tensor Cores and runs it dramatically more
        # slowly than FP16.  Use BF16 only on Ampere or newer.
        major, _ = torch.cuda.get_device_capability(device)
        dtype_name = "bfloat16" if major >= 8 and torch.cuda.is_bf16_supported() else "float16"
    else:
        dtype_name = "float32"
    dtype = getattr(torch, dtype_name) if hasattr(torch, dtype_name) else torch.float32

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        # Pascal has no Tensor Cores, but its FP16 cuBLAS kernels can use the
        # reduced precision reduction path.  This keeps every projection and
        # recurrent pass intact while avoiding an unnecessary FP32 reduction
        # fallback on supported Torch builds.
        if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    tokenizer = get_tokenizer()
    if args.compile_sinkhorn:
        os.environ["LLMY_COMPILE_SINKHORN"] = "1"
        if _load_native_fused_sinkhorn() is None:
            print(
                "Fused CUDA Sinkhorn unavailable; using the PyTorch implementation. "
                "Build llmy_cuda to enable it."
            )
        else:
            print("Fused CUDA Sinkhorn enabled (one kernel launch per 4x4 matrix batch).")
    model, _ = load_model(ckpt, device, dtype, args.config)
    # Match the trainer's input-dependent mHC path by default.
    set_fast_mhc(model, args.fast_mhc)
    adaptive_active = configure_adaptive_compute(model, args)
    if adaptive_active and args.cuda_graph:
        raise RuntimeError(
            "--cuda-graph cannot be used with learned adaptive halting: graph replay captures "
            "one fixed loop/control-flow path. Use eager decode, or --fixed-loops N for debugging."
        )

    model = apply_all_triton_optimizations(model)

    # Apply the decode MoE experiment after optional runtime patching so the
    # selected implementation is the one actually used during generation.
    set_moe_decode_mode(model, args.moe_mode)

    def run_sample(prompt: str) -> None:
        temp, top_k, top_p, rep = args.temp, args.top_k, args.top_p, args.rep
        if args.greedy:
            temp, top_k, top_p, rep = 0.0, 0, 1.0, 1.0
        # Preserve the original sampler behavior: prompt perplexity is shown
        # unless the caller explicitly opts out. --prompt-ppl remains as a
        # harmless compatibility alias for scripts that pass it explicitly.
        show_prompt_ppl = not args.no_prompt_ppl

        if args.parity_check:
            parity_check(model, tokenizer, prompt, device, dtype, args.parity_tokens)
            return

        if args.full_recompute:
            full_recompute_sample(
                model, tokenizer, prompt, device, dtype, args.max_tokens,
                temp, top_k, top_p, rep,
            )
            return

        sample(
            model, tokenizer, prompt, device, dtype, args.max_tokens,
            temp, top_k, top_p, rep,
            show_prompt_ppl,
            args.cuda_graph,
            ignore_eos=args.ignore_eos,
        )

    if args.prompt:
        run_sample(args.prompt)
        return

    print("\n--- Interactive Sampling REPL (Type your prompt, or /exit to quit) ---")
    while True:
        try:
            p = input("Prompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not p or p.lower() in ("/exit", "/quit"):
            break
        run_sample(p)


if __name__ == "__main__":
    main()
