"""Input-dependent multi-stream hyper-connection (mHC)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import os

import torch
import torch.nn as nn


def _sinkhorn_impl(logits: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    matrix = logits.float()
    for _ in range(int(iterations)):
        matrix = matrix - torch.logsumexp(matrix, dim=-1, keepdim=True)
        matrix = matrix - torch.logsumexp(matrix, dim=-2, keepdim=True)
    return matrix.exp().to(dtype=logits.dtype)


_compiled_sinkhorn = None


def sinkhorn(logits: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    """Sinkhorn normalization with an optional fused CUDA inference path."""
    global _compiled_sinkhorn
    if (
        os.environ.get("LLMY_COMPILE_SINKHORN", "0") == "1"
        and logits.is_cuda and not logits.requires_grad
    ):
        if _compiled_sinkhorn is None:
            try:
                _compiled_sinkhorn = torch.compile(
                    _sinkhorn_impl, mode="reduce-overhead", fullgraph=True
                )
            except Exception:
                os.environ["LLMY_COMPILE_SINKHORN"] = "0"
                return _sinkhorn_impl(logits, iterations)
        try:
            return _compiled_sinkhorn(logits, iterations)
        except Exception:
            _compiled_sinkhorn = None
            os.environ["LLMY_COMPILE_SINKHORN"] = "0"
            return _sinkhorn_impl(logits, iterations)
    return _sinkhorn_impl(logits, iterations)


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

    def __init__(self, d_model: int, streams: int = 4, sinkhorn_iterations: int = 20,
                 init_gating_factor: float = 0.01) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.streams = int(streams)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        # One fused projection computes the three input-dependent mappings.
        # This is the layout used by the historical 40--42k fast run.
        self.read_logits = nn.Parameter(torch.eye(streams) * 4.0)
        self.write_logits = nn.Parameter(torch.eye(streams) * 4.0)
        self.stream_scale = nn.Parameter(torch.ones(streams, d_model))
        self.stream_norm = nn.LayerNorm(d_model)
        self.mapping_norm = nn.RMSNorm(streams * d_model)
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
        self._static_inference = False
        self._static_maps: MHCMaps | None = None

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
        self._refresh_inference_cache()
        return self

    def train(self, mode: bool = True) -> "MultiStreamHyperConnection":
        super().train(mode)
        self._inference_matrices = None
        self._static_maps = None
        self._static_inference = False
        if not mode:
            self._refresh_inference_cache()
        return self

    def _refresh_inference_cache(self) -> None:
        self._inference_matrices = (
            sinkhorn(self.read_logits, self.sinkhorn_iterations),
            sinkhorn(self.write_logits, self.sinkhorn_iterations),
        )
        # A zero dynamic projection makes the maps input-independent. This is
        # true for the current checkpoint and is also safe for trained models
        # whose dynamic weights remain zero; learned biases are still kept.
        self._static_inference = int(torch.count_nonzero(self.theta.weight.detach()).item()) == 0
        self._static_maps = None

    def set_fast_inference(self, enabled: bool = True) -> "MultiStreamHyperConnection":
        """Use constant inference maps when a caller accepts fast mHC mode."""
        if self.training and enabled:
            raise RuntimeError("fast mHC inference requires eval mode")
        self._static_inference = bool(enabled)
        self._static_maps = None
        return self

    def _static_maps_for(self, streams: torch.Tensor) -> MHCMaps:
        maps = self._static_maps
        if maps is None or maps.pre.device != streams.device or maps.pre.dtype != streams.dtype:
            maps = MHCMaps(
                pre=torch.sigmoid(self.pre_bias).to(device=streams.device, dtype=streams.dtype),
                post=(2.0 * torch.sigmoid(self.post_bias)).to(device=streams.device, dtype=streams.dtype),
                residual=sinkhorn(self.residual_bias, self.sinkhorn_iterations).to(
                    device=streams.device, dtype=streams.dtype
                ),
            )
            self._static_maps = maps
        return maps

    def compute_mappings(self, streams: torch.Tensor) -> MHCMaps:
        if streams.ndim != 4 or streams.shape[-2] != self.streams:
            raise ValueError(f"expected [batch, time, {self.streams}, hidden] streams")
        if not self.training and self._static_inference:
            return self._static_maps_for(streams)
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
        """Migrate checkpoints containing the former three theta projections."""
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
            from runpod.kernels.triton_mhc import triton_mhc_dynamic_read
            return triton_mhc_dynamic_read(streams, mappings.pre, self.stream_norm)
        if not self.training and self._static_inference:
            value = (streams * mappings.pre.view(1, 1, self.streams, 1)).sum(dim=-2)
        else:
            value = torch.einsum("bts,btsd->btd", mappings.pre, streams)
        norm_dtype = self.stream_norm.weight.dtype if self.stream_norm.weight is not None else value.dtype
        return self.stream_norm(value.to(dtype=norm_dtype)).to(dtype=value.dtype)

    def write(self, streams: torch.Tensor, update: torch.Tensor,
              mappings: MHCMaps | None = None) -> torch.Tensor:
        if mappings is None:
            mappings = self.compute_mappings(streams)
        if self.use_triton_kernels and streams.is_cuda:
            from runpod.kernels.triton_mhc import triton_mhc_dynamic_write
            return triton_mhc_dynamic_write(
                streams, update, mappings.residual, mappings.post,
                self.stream_scale, self.gating_factor,
            )
        if not self.training and self._static_inference:
            residual = torch.einsum("ij,btjd->btid", mappings.residual, streams)
            stream_update = residual + mappings.post.view(1, 1, self.streams, 1) * update.unsqueeze(-2)
        else:
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

