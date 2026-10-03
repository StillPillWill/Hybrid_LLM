"""Reference Gated DeltaNet mixer with a chunkable recurrent scan."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

try:
    import numba.cuda as _numba_cuda
    _NUMBA_AVAILABLE = _numba_cuda.is_available()
except Exception:
    _NUMBA_AVAILABLE = False

if _NUMBA_AVAILABLE:
    try:
        from .fused_gdn_cuda import fused_gdn_scan as _fused_gdn_scan
    except Exception:
        _fused_gdn_scan = None
    try:
        from .fused_gdn_cuda import fused_gdn_step as _fused_gdn_step
    except Exception:
        _fused_gdn_step = None
else:
    _fused_gdn_scan = None
    _fused_gdn_step = None

try:
    from runpod.kernels.triton_gdn import (
        triton_gdn_step as _triton_gdn_step,
        triton_gdn_scan as _triton_gdn_scan,
        _TRITON_AVAILABLE as _triton_gdn_available,
    )
except Exception:
    _triton_gdn_step = None
    _triton_gdn_scan = None
    _triton_gdn_available = False


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
    # A regular Conv1d sees the preceding tokens even when the recurrent scan
    # resets at a document transition. Mask those preceding convolution taps
    # using the same transition sequence as the recurrent state.
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
                    # Match the historical fast path: the scan itself is a
                    # tensor-only recomputation boundary; the outer model
                    # detaches the returned recurrent state after the block.
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
