"""Fully Fused Forward and Backward CUDA Kernels for GatedDeltaNet.

Architecture:
- Forward Kernel: On-chip Shared Memory recurrence (T=0 .. T-1) -> ~8 ms per call
- Backward Kernel: On-chip Shared Memory adjoint recurrence (T=T-1 .. 0) -> ~12 ms per call
- Saves all Python loop overhead in both forward and backward passes!
"""
from __future__ import annotations
import torch
from numba import cuda, float32

_GDN_BWD_CHUNK = 128


@cuda.jit
def _gdn_fwd_kernel(
    q, k, v, beta, gate, reset, decay, state_in, out, state_out, all_states
):
    b = cuda.blockIdx.x
    h = cuda.blockIdx.y
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tid = ty * 32 + tx

    s_mem = cuda.shared.array(shape=(64, 64), dtype=float32)
    k_shared = cuda.shared.array(shape=(64,), dtype=float32)
    q_shared = cuda.shared.array(shape=(64,), dtype=float32)
    v_shared = cuda.shared.array(shape=(64,), dtype=float32)
    pred_shared = cuda.shared.array(shape=(64,), dtype=float32)

    for idx in range(tid, 4096, 256):
        ki = idx // 64
        vi = idx % 64
        val = state_in[b, h, ki, vi]
        s_mem[ki, vi] = val
        all_states[b, 0, h, ki, vi] = val
    cuda.syncthreads()

    dec_val = decay[h]
    T = q.shape[1]

    for t in range(T):
        if reset[b, t]:
            for idx in range(tid, 4096, 256):
                ki = idx // 64
                vi = idx % 64
                s_mem[ki, vi] = 0.0
        cuda.syncthreads()

        if tid < 64:
            k_shared[tid] = k[b, t, h, tid]
            q_shared[tid] = q[b, t, h, tid]
            v_shared[tid] = v[b, t, h, tid]
        cuda.syncthreads()

        b_val = beta[b, t, h]

        if tid < 64:
            acc = 0.0
            for kj in range(64):
                acc += k_shared[kj] * s_mem[kj, tid]
            pred_shared[tid] = acc
        cuda.syncthreads()

        for idx in range(tid, 4096, 256):
            ki = idx // 64
            vi = idx % 64
            delta = v_shared[vi] - pred_shared[vi]
            s_val = dec_val * s_mem[ki, vi] + b_val * k_shared[ki] * delta
            s_mem[ki, vi] = s_val
            all_states[b, t + 1, h, ki, vi] = s_val
        cuda.syncthreads()

        if tid < 64:
            acc_out = 0.0
            for kj in range(64):
                acc_out += q_shared[kj] * s_mem[kj, tid]
            out[b, t, h, tid] = acc_out * gate[b, t, h, tid]
        cuda.syncthreads()

    for idx in range(tid, 4096, 256):
        ki = idx // 64
        vi = idx % 64
        state_out[b, h, ki, vi] = s_mem[ki, vi]


@cuda.jit
def _gdn_fwd_compact_kernel(
    q, k, v, beta, gate, reset, decay, state_in, out, state_out,
    boundary_states,
):
    """Forward scan without retaining a per-token state history.

    The training backward pass can regenerate the history when it actually
    needs it.  Keeping this separate from ``_gdn_fwd_kernel`` avoids making
    the full ``[B, T + 1, H, K, V]`` history live during the forward peak.
    """
    b = cuda.blockIdx.x
    h = cuda.blockIdx.y
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tid = ty * 32 + tx

    s_mem = cuda.shared.array(shape=(64, 64), dtype=float32)
    k_shared = cuda.shared.array(shape=(64,), dtype=float32)
    q_shared = cuda.shared.array(shape=(64,), dtype=float32)
    v_shared = cuda.shared.array(shape=(64,), dtype=float32)
    pred_shared = cuda.shared.array(shape=(64,), dtype=float32)

    for idx in range(tid, 4096, 256):
        ki = idx // 64
        vi = idx % 64
        s_mem[ki, vi] = state_in[b, h, ki, vi]
    cuda.syncthreads()

    for idx in range(tid, 4096, 256):
        ki = idx // 64
        vi = idx % 64
        boundary_states[b, 0, h, ki, vi] = s_mem[ki, vi]

    dec_val = decay[h]
    T = q.shape[1]

    for t in range(T):
        if reset[b, t]:
            for idx in range(tid, 4096, 256):
                ki = idx // 64
                vi = idx % 64
                s_mem[ki, vi] = 0.0
        cuda.syncthreads()

        if tid < 64:
            k_shared[tid] = k[b, t, h, tid]
            q_shared[tid] = q[b, t, h, tid]
            v_shared[tid] = v[b, t, h, tid]
        cuda.syncthreads()

        b_val = beta[b, t, h]

        if tid < 64:
            acc = 0.0
            for kj in range(64):
                acc += k_shared[kj] * s_mem[kj, tid]
            pred_shared[tid] = acc
        cuda.syncthreads()

        for idx in range(tid, 4096, 256):
            ki = idx // 64
            vi = idx % 64
            delta = v_shared[vi] - pred_shared[vi]
            s_mem[ki, vi] = dec_val * s_mem[ki, vi] + b_val * k_shared[ki] * delta
        cuda.syncthreads()

        if (t + 1) % _GDN_BWD_CHUNK == 0 or t + 1 == T:
            boundary_idx = (t + 1) // _GDN_BWD_CHUNK
            if (t + 1) % _GDN_BWD_CHUNK != 0:
                boundary_idx += 1
            for idx in range(tid, 4096, 256):
                ki = idx // 64
                vi = idx % 64
                boundary_states[b, boundary_idx, h, ki, vi] = s_mem[ki, vi]
        cuda.syncthreads()

        if tid < 64:
            acc_out = 0.0
            for kj in range(64):
                acc_out += q_shared[kj] * s_mem[kj, tid]
            out[b, t, h, tid] = acc_out * gate[b, t, h, tid]
        cuda.syncthreads()

    for idx in range(tid, 4096, 256):
        ki = idx // 64
        vi = idx % 64
        state_out[b, h, ki, vi] = s_mem[ki, vi]


@cuda.jit
def _gdn_bwd_kernel(
    grad_out, q, k, v, beta, gate, reset, decay, all_states,
    grad_q, grad_k, grad_v, grad_beta, grad_gate, grad_decay, grad_state_in
):
    b = cuda.blockIdx.x
    h = cuda.blockIdx.y
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tid = ty * 32 + tx

    ds_mem = cuda.shared.array(shape=(64, 64), dtype=float32)
    q_shared = cuda.shared.array(shape=(64,), dtype=float32)
    k_shared = cuda.shared.array(shape=(64,), dtype=float32)
    gate_shared = cuda.shared.array(shape=(64,), dtype=float32)
    gout_shared = cuda.shared.array(shape=(64,), dtype=float32)
    delta_shared = cuda.shared.array(shape=(64,), dtype=float32)
    k_ds_shared = cuda.shared.array(shape=(64,), dtype=float32)
    b_val_shared = cuda.shared.array(shape=(1,), dtype=float32)
    red_buf = cuda.shared.array(shape=(256,), dtype=float32)

    for idx in range(tid, 4096, 256):
        ki = idx // 64
        vi = idx % 64
        ds_mem[ki, vi] = 0.0
    cuda.syncthreads()

    dec_val = decay[h]
    decay_accum = 0.0
    T = q.shape[1]

    for t in range(T - 1, -1, -1):
        if tid < 64:
            q_shared[tid] = q[b, t, h, tid]
            k_shared[tid] = k[b, t, h, tid]
            gate_shared[tid] = gate[b, t, h, tid]
            gout_shared[tid] = grad_out[b, t, h, tid]
        if tid == 0:
            b_val_shared[0] = beta[b, t, h]
        cuda.syncthreads()

        if tid < 64:
            acc_s = 0.0
            for kj in range(64):
                acc_s += q_shared[kj] * all_states[b, t + 1, h, kj, tid]
            grad_gate[b, t, h, tid] = acc_s * gout_shared[tid]

            acc_q = 0.0
            for vi in range(64):
                acc_q += all_states[b, t + 1, h, tid, vi] * (gout_shared[vi] * gate_shared[vi])
            grad_q[b, t, h, tid] = acc_q

            pred = 0.0
            for kj in range(64):
                pred += k_shared[kj] * all_states[b, t, h, kj, tid]
            delta_shared[tid] = v[b, t, h, tid] - pred
        cuda.syncthreads()

        for idx in range(tid, 4096, 256):
            ki = idx // 64
            vi = idx % 64
            ds_mem[ki, vi] += q_shared[ki] * (gout_shared[vi] * gate_shared[vi])
        cuda.syncthreads()

        if tid < 64:
            acc_kds = 0.0
            for rk in range(64):
                acc_kds += k_shared[rk] * ds_mem[rk, tid]
            k_ds_shared[tid] = acc_kds
        cuda.syncthreads()

        b_val = b_val_shared[0]

        if tid < 64:
            grad_v[b, t, h, tid] = b_val * k_ds_shared[tid]
            acc_k = 0.0
            for vi in range(64):
                acc_k += b_val * delta_shared[vi] * ds_mem[tid, vi]
                acc_k -= b_val * k_ds_shared[vi] * all_states[b, t, h, tid, vi]
            grad_k[b, t, h, tid] = acc_k

        local_b_acc = 0.0
        for idx in range(tid, 4096, 256):
            ki = idx // 64
            vi = idx % 64
            local_b_acc += k_shared[ki] * delta_shared[vi] * ds_mem[ki, vi]
            st_val = all_states[b, t, h, ki, vi]
            decay_accum += ds_mem[ki, vi] * st_val
            ds_mem[ki, vi] = dec_val * ds_mem[ki, vi] - b_val * k_shared[ki] * k_ds_shared[vi]

        red_buf[tid] = local_b_acc
        cuda.syncthreads()

        s = 128
        while s > 0:
            if tid < s:
                red_buf[tid] += red_buf[tid + s]
            cuda.syncthreads()
            s //= 2

        if tid == 0:
            grad_beta[b, t, h] = red_buf[0]
        cuda.syncthreads()

        if reset[b, t]:
            for idx in range(tid, 4096, 256):
                ki = idx // 64
                vi = idx % 64
                ds_mem[ki, vi] = 0.0
        cuda.syncthreads()

    for idx in range(tid, 4096, 256):
        ki = idx // 64
        vi = idx % 64
        grad_state_in[b, h, ki, vi] = ds_mem[ki, vi]

    red_buf[tid] = decay_accum
    cuda.syncthreads()
    s = 128
    while s > 0:
        if tid < s:
            red_buf[tid] += red_buf[tid + s]
        cuda.syncthreads()
        s //= 2
    if tid == 0:
        grad_decay[b, h, 0] = red_buf[0]


class FullyFusedGDNScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, beta, gate, reset, decay_logit, state):
        B, T, H, K = q.shape
        V = v.shape[3]

        decay_1d = torch.sigmoid(decay_logit).float().contiguous()
        out = torch.empty((B, T, H, V), device=q.device, dtype=torch.float32)
        state_out = torch.empty((B, H, K, V), device=q.device, dtype=torch.float32)

        d_q = cuda.as_cuda_array(q.float().detach().contiguous())
        d_k = cuda.as_cuda_array(k.float().detach().contiguous())
        d_v = cuda.as_cuda_array(v.float().detach().contiguous())
        beta_2d = beta.squeeze(-1) if beta.ndim == 4 else beta
        d_beta = cuda.as_cuda_array(beta_2d.float().detach().contiguous())
        d_gate = cuda.as_cuda_array(gate.float().detach().contiguous())
        d_reset = cuda.as_cuda_array(reset.int().detach().contiguous())
        d_decay = cuda.as_cuda_array(decay_1d)
        d_sin = cuda.as_cuda_array(state.float().detach().contiguous())
        d_out = cuda.as_cuda_array(out)
        d_sout = cuda.as_cuda_array(state_out)
        num_boundary = (T + _GDN_BWD_CHUNK - 1) // _GDN_BWD_CHUNK + 2
        boundary_states = torch.empty((B, num_boundary, H, K, V), device=q.device, dtype=torch.float32)
        d_boundary = cuda.as_cuda_array(boundary_states)

        grid = (B, H)
        block = (32, 8)  # 256 threads per block
        _gdn_fwd_compact_kernel[grid, block](
            d_q, d_k, d_v, d_beta, d_gate, d_reset, d_decay,
            d_sin, d_out, d_sout, d_boundary,
        )

        # Regenerate the full state history in backward so it is not retained
        # during the forward activation peak (~448.5 MiB at B=7,T=1024).
        ctx.save_for_backward(q, k, v, beta, gate, reset, decay_logit, state)
        return out.to(dtype=q.dtype), state_out.to(dtype=state.dtype)

    @staticmethod
    def backward(ctx, grad_output, grad_final_state):
        q, k, v, beta, gate, reset, decay_logit, state = ctx.saved_tensors
        B, T, H, K = q.shape
        V = v.shape[3]

        decay_1d = torch.sigmoid(decay_logit).float().contiguous()
        all_states = torch.empty((B, T + 1, H, K, V), device=q.device, dtype=torch.float16)
        replay_out = torch.empty((B, T, H, V), device=q.device, dtype=torch.float32)
        replay_state = torch.empty((B, H, K, V), device=q.device, dtype=torch.float32)
        grad_q = torch.empty((B, T, H, K), device=q.device, dtype=torch.float32)
        grad_k = torch.empty((B, T, H, K), device=q.device, dtype=torch.float32)
        grad_v = torch.empty((B, T, H, V), device=q.device, dtype=torch.float32)
        grad_beta = torch.empty((B, T, H), device=q.device, dtype=torch.float32)
        grad_gate = torch.empty((B, T, H, V), device=q.device, dtype=torch.float32)
        grad_decay_raw = torch.zeros((B, H, K), device=q.device, dtype=torch.float32)
        grad_sin = torch.empty((B, H, K, V), device=q.device, dtype=torch.float32)

        beta_2d = beta.squeeze(-1) if beta.ndim == 4 else beta
        d_go = cuda.as_cuda_array(grad_output.float().detach().contiguous())
        d_q = cuda.as_cuda_array(q.float().detach().contiguous())
        d_k = cuda.as_cuda_array(k.float().detach().contiguous())
        d_v = cuda.as_cuda_array(v.float().detach().contiguous())
        d_beta = cuda.as_cuda_array(beta_2d.float().detach().contiguous())
        d_gate = cuda.as_cuda_array(gate.float().detach().contiguous())
        d_reset = cuda.as_cuda_array(reset.int().detach().contiguous())
        d_decay = cuda.as_cuda_array(decay_1d.detach())
        d_sin = cuda.as_cuda_array(state.float().detach().contiguous())
        d_replay_out = cuda.as_cuda_array(replay_out)
        d_replay_state = cuda.as_cuda_array(replay_state)
        d_ast = cuda.as_cuda_array(all_states)

        grid = (B, H)
        block = (32, 8)  # 256 threads per block
        _gdn_fwd_kernel[grid, block](
            d_q, d_k, d_v, d_beta, d_gate, d_reset, d_decay,
            d_sin, d_replay_out, d_replay_state, d_ast,
        )

        d_gq = cuda.as_cuda_array(grad_q)
        d_gk = cuda.as_cuda_array(grad_k)
        d_gv = cuda.as_cuda_array(grad_v)
        d_gb = cuda.as_cuda_array(grad_beta)
        d_gg = cuda.as_cuda_array(grad_gate)
        d_gdec = cuda.as_cuda_array(grad_decay_raw)
        d_gsin = cuda.as_cuda_array(grad_sin)

        _gdn_bwd_kernel[grid, block](
            d_go, d_q, d_k, d_v, d_beta, d_gate, d_reset, d_decay, d_ast,
            d_gq, d_gk, d_gv, d_gb, d_gg, d_gdec, d_gsin,
        )

        sig_decay = torch.sigmoid(decay_logit).float()
        dsig = sig_decay * (1.0 - sig_decay)
        grad_decay_logit = (grad_decay_raw.sum(dim=(0, 2)) * dsig).to(dtype=decay_logit.dtype)
        grad_beta_out = grad_beta.unsqueeze(-1).to(dtype=beta.dtype) if beta.ndim == 4 else grad_beta.to(dtype=beta.dtype)

        return (
            grad_q.to(dtype=q.dtype),
            grad_k.to(dtype=k.dtype),
            grad_v.to(dtype=v.dtype),
            grad_beta_out,
            grad_gate.to(dtype=gate.dtype),
            None,
            grad_decay_logit,
            grad_sin.to(dtype=q.dtype),
        )


def fused_gdn_scan(q, k, v, beta, gate, reset, decay_logit, state):
    return FullyFusedGDNScan.apply(q, k, v, beta, gate, reset, decay_logit, state)
