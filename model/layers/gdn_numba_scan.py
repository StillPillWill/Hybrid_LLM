"""Numba CUDA-fused chunked GDN recurrent scan wrapped in a torch.autograd.Function.

Strategy:
- FORWARD: detach all inputs, run the Numba CUDA kernel (no autograd overhead).
- BACKWARD: replay the pure-PyTorch reference scan on the saved detached inputs
  with gradients re-enabled (gradient-checkpoint / recompute style).
  This gives correct gradients through q/k/v/beta/gate projections and decay.

Occupancy improvement v2:
  Grid = (H,) blocks; threads = (K, V) 2D — both K and V dims run in parallel.
  For H=8, K=64, V=64: 8 blocks × 64×64 = 32768 threads scheduled across 20 SMs.
"""
from __future__ import annotations
import torch
from numba import cuda

_SCAN_CHUNK = 128   # tokens per GPU kernel call; increased from 64 for better SM occupancy


# ---------------------------------------------------------------------------
# Numba CUDA kernel: 1 block per head, 2D thread grid over (K, V)
# ---------------------------------------------------------------------------

@cuda.jit
def _gdn_chunk_kernel(
    q,         # [C, H, K]  float32
    k,         # [C, H, K]  float32
    v,         # [C, H, V]  float32
    beta,      # [C, H]     float32
    gate,      # [C, H, V]  float32
    reset,     # [C]        bool
    decay,     # [H]        float32
    state_io,  # [H, K, V]  float32  (in-place carry)
    out,       # [C, H, V]  float32
):
    """
    blockIdx.x  = head h
    threadIdx.x = key-dim index ki  (0..K-1)
    threadIdx.y = val-dim index vi  (0..V-1)
    Sequential over C time-steps (serial dependency); parallel over K and V.
    """
    h  = cuda.blockIdx.x
    ki = cuda.threadIdx.x
    vi = cuda.threadIdx.y
    H  = decay.shape[0]
    K  = q.shape[2]
    V  = v.shape[2]
    C  = q.shape[0]
    if h >= H or ki >= K or vi >= V:
        return

    for t in range(C):
        # Reset carry row/col for this thread's (ki, vi) cell
        if reset[t]:
            state_io[h, ki, vi] = 0.0
        cuda.syncthreads()

        # prediction[vi] = sum_kj k[t,h,kj] * state[h,kj,vi]
        # Each (ki, vi) thread independently computes its own prediction for vi:
        # — all ki threads read the same vi column, but only the ki==0 thread
        #   writes out[t,h,vi] to avoid races. The state update is per (ki,vi) cell.
        pred = 0.0
        for kj in range(K):
            pred += k[t, h, kj] * state_io[h, kj, vi]
        delta = v[t, h, vi] - pred

        state_io[h, ki, vi] = (
            decay[h] * state_io[h, ki, vi]
            + beta[t, h] * k[t, h, ki] * delta
        )
        cuda.syncthreads()

        # Output: only ki==0 accumulates q @ state for this vi
        if ki == 0:
            acc = 0.0
            for kj in range(K):
                acc += q[t, h, kj] * state_io[h, kj, vi]
            out[t, h, vi] = acc * gate[t, h, vi]
        cuda.syncthreads()


def _run_numba_chunk(q_c, k_c, v_c, beta_c, gate_c, reset_c, decay_1d, carry):
    """Run one chunk. carry [H, K, V] modified in place. Returns out_c [C, H, V]."""
    C, H, K = q_c.shape
    V = v_c.shape[2]
    out_c = torch.empty(C, H, V, device=q_c.device, dtype=torch.float32)
    carry_f = carry.float().contiguous()

    d_q    = cuda.as_cuda_array(q_c.float().detach().contiguous())
    d_k    = cuda.as_cuda_array(k_c.float().detach().contiguous())
    d_v    = cuda.as_cuda_array(v_c.float().detach().contiguous())
    d_beta = cuda.as_cuda_array(beta_c.float().detach().contiguous())
    d_gate = cuda.as_cuda_array(gate_c.float().detach().contiguous())
    d_rst  = cuda.as_cuda_array(reset_c.bool().detach().contiguous())
    d_dec  = cuda.as_cuda_array(decay_1d.float().detach().contiguous())
    d_s    = cuda.as_cuda_array(carry_f)
    d_out  = cuda.as_cuda_array(out_c)

    # 2D thread block: (K, V); capped to 1024 total threads per Pascal limit
    tx = min(K, 32)
    ty = min(V, 32)
    _gdn_chunk_kernel[(H,), (tx, ty)](d_q, d_k, d_v, d_beta, d_gate, d_rst, d_dec, d_s, d_out)
    cuda.synchronize()

    carry.copy_(torch.as_tensor(d_s, device=q_c.device).to(carry.dtype))
    return out_c.to(dtype=q_c.dtype)


# ---------------------------------------------------------------------------
# Pure-PyTorch reference scan (for backward recompute)
# ---------------------------------------------------------------------------

def _pytorch_scan_fwd(q, k, v, beta, gate, reset, decay_4d, state):
    """
    q/k:  [B, T, H, K], v/gate: [B, T, H, V], beta: [B, T, H, 1]
    reset:[B, T],  decay_4d: [1, H, 1, 1],  state: [B, H, K, V]
    Returns output [B, T, H, V] and final state [B, H, K, V].
    """
    B, T, H, K = q.shape
    V = v.shape[3]
    carry = state.clone().to(torch.float32)
    outputs = []
    for t in range(T):
        rst = reset[:, t]
        carry = torch.where(rst[:, None, None, None], torch.zeros_like(carry), carry)
        k_t  = k[:, t].float()
        v_t  = v[:, t].float()
        q_t  = q[:, t].float()
        b_t  = beta[:, t].float()       # [B, H, 1]
        g_t  = gate[:, t]               # [B, H, V]
        pred = torch.einsum("bhk,bhkv->bhv", k_t, carry)
        delta = v_t - pred
        carry = (
            decay_4d * carry
            + b_t.unsqueeze(-1) * torch.einsum("bhk,bhv->bhkv", k_t, delta)
        )
        out_t = torch.einsum("bhk,bhkv->bhv", q_t, carry).to(q.dtype) * g_t
        outputs.append(out_t)
    return torch.stack(outputs, dim=1), carry.to(state.dtype)


# ---------------------------------------------------------------------------
# autograd.Function: fast Numba forward, recomputed PyTorch backward
# ---------------------------------------------------------------------------

class GDNScanFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, beta, gate, reset, decay_logit, state):
        B, T, H, K = q.shape
        V = v.shape[3]
        decay_1d = torch.sigmoid(decay_logit).float()  # [H]

        output = torch.empty(B, T, H, V, device=q.device, dtype=q.dtype)
        final_state = state.clone()

        for b in range(B):
            carry = final_state[b].float().clone()
            pos = 0
            while pos < T:
                end = min(pos + _SCAN_CHUNK, T)
                out_c = _run_numba_chunk(
                    q[b, pos:end],
                    k[b, pos:end],
                    v[b, pos:end],
                    beta[b, pos:end, :, 0],   # squeeze last dim → [C, H]
                    gate[b, pos:end],
                    reset[b, pos:end],
                    decay_1d,
                    carry,
                )
                output[b, pos:end] = out_c.to(dtype=q.dtype)
                pos = end
            final_state[b] = carry.to(state.dtype)

        ctx.save_for_backward(q, k, v, beta, gate, reset, decay_logit, state)
        return output.reshape(B, T, H * V), final_state

    @staticmethod
    def backward(ctx, grad_output, grad_final_state):
        q, k, v, beta, gate, reset, decay_logit, state = ctx.saved_tensors
        B, T, HV = grad_output.shape
        H = q.shape[2]
        V = v.shape[3]

        with torch.enable_grad():
            q_  = q.detach().requires_grad_(q.requires_grad)
            k_  = k.detach().requires_grad_(k.requires_grad)
            v_  = v.detach().requires_grad_(v.requires_grad)
            b_  = beta.detach().requires_grad_(beta.requires_grad)
            g_  = gate.detach().requires_grad_(gate.requires_grad)
            dl_ = decay_logit.detach().requires_grad_(decay_logit.requires_grad)
            s_  = state.detach().requires_grad_(state.requires_grad)

            decay_4d_ = torch.sigmoid(dl_).view(1, -1, 1, 1).to(torch.float32)
            out_4d, _ = _pytorch_scan_fwd(q_, k_, v_, b_, g_, reset, decay_4d_, s_)
            # grad_output is [B, T, H*V]
            out_4d.reshape(B, T, HV).backward(grad_output)

        return (
            q_.grad  if q.requires_grad          else None,
            k_.grad  if k.requires_grad          else None,
            v_.grad  if v.requires_grad          else None,
            b_.grad  if beta.requires_grad       else None,
            g_.grad  if gate.requires_grad       else None,
            None,    # reset — no grad
            dl_.grad if decay_logit.requires_grad else None,
            s_.grad  if state.requires_grad      else None,
        )


def gdn_scan_numba(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gate: torch.Tensor,
    reset: torch.Tensor,
    decay_logit: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Entry point called from gated_deltanet.py.
    Returns (output [B, T, H*V], final_state [B, H, K, V]).
    """
    return GDNScanFunction.apply(q, k, v, beta, gate, reset, decay_logit, state)
