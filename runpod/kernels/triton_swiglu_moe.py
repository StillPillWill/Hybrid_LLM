"""Triton Fused SwiGLU and Batched-GEMM MoE Dispatcher with Fused Gate-Up."""
from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
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
    if _TRITON_AVAILABLE and gate.is_cuda:
        return _TritonSwiGLUFunction.apply(gate, up)
    return F.silu(gate) * up


def _triton_moe_dispatch_chunk(
    flat: torch.Tensor,
    indices: torch.Tensor,
    probabilities: torch.Tensor,
    routed_experts: int,
    w_gate: torch.Tensor,  # [E, H, D]
    w_up: torch.Tensor,    # [E, H, D]
    w_down: torch.Tensor,  # [E, D, H]
) -> torch.Tensor:
    """Run one bounded MoE dispatch chunk.

    The routing math is unchanged; the public wrapper below splits long
    sequences before this workspace-heavy implementation is entered.
    """
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


def triton_moe_dispatch(
    flat: torch.Tensor,
    indices: torch.Tensor,
    probabilities: torch.Tensor,
    routed_experts: int,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Grouped-GEMM MoE dispatch with bounded peak workspace.

    The old implementation built ``padded``, ``h_gate_up`` and
    ``out_padded`` for the entire sequence at once.  With long contexts and
    skewed routing, the final weighted-gather temporary could require another
    ~232 MiB after the allocator was already near the card limit.  Dispatch
    is token-independent, so processing fixed token chunks is mathematically
    equivalent while bounding those intermediates.  ``LLMY_MOE_TOKEN_CHUNK``
    is intentionally runtime-configurable for safe tuning without changing
    the checkpoint or model.
    """
    configured = os.environ.get("LLMY_MOE_TOKEN_CHUNK", "4096")
    try:
        chunk_tokens = max(1, int(configured))
    except ValueError:
        chunk_tokens = 4096

    if flat.shape[0] <= chunk_tokens:
        return _triton_moe_dispatch_chunk(
            flat, indices, probabilities, routed_experts, w_gate, w_up, w_down
        )

    outputs = []
    for start in range(0, flat.shape[0], chunk_tokens):
        end = min(start + chunk_tokens, flat.shape[0])
        outputs.append(
            _triton_moe_dispatch_chunk(
                flat[start:end],
                indices[start:end],
                probabilities[start:end],
                routed_experts,
                w_gate,
                w_up,
                w_down,
            )
        )
    return torch.cat(outputs, dim=0)


def triton_moe_step(
    flat: torch.Tensor,
    indices: torch.Tensor,
    probabilities: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Fused fixed-shape dispatcher for a one-token decode batch."""
    if not _TRITON_AVAILABLE or not flat.is_cuda:
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
