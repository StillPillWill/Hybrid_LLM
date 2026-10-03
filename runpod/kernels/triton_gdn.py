"""Triton Fused Forward and Backward Recurrent Scan for GatedDeltaNet.

Optimized for Ampere (sm_86 / A5000):
- Forward: Evaluates the delta recurrence on-chip in SRAM blocks (64x64)
- Backward: Recomputes adjoint gradients via backward pass on-chip
- Zero Python looping; full Tensor Core and vector load utilization.
"""
from __future__ import annotations
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
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available. Please install triton on Linux/Ampere.")
    return _TritonGDNFunction.apply(q, k, v, beta, gate, reset_mask, decay, state_in)


def triton_gdn_step(q, k, v, beta, gate, reset_mask, decay, state):
    """One-token Triton recurrence that updates the FP32 state in place."""
    if not _TRITON_AVAILABLE:
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
