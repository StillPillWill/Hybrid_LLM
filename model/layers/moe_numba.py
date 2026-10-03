"""Numba CUDA kernels for the batch-1 routed MoE decode path.

The framework implementation launches a separate batched GEMM for the selected
experts.  Decode always has one input row, so two fixed-shape kernels are a
better fit: one computes all selected gate/up projections and the other applies
SiLU, down projection, and route probability in one pass.
"""
from __future__ import annotations

import torch
from math import exp

try:
    from numba import cuda

    _NUMBA_AVAILABLE = bool(cuda.is_available())
except Exception:
    cuda = None  # type: ignore[assignment]
    _NUMBA_AVAILABLE = False


if cuda is not None:
    @cuda.jit(fastmath=True)
    def _fused_route_kernel(x, expert_ids, probabilities, gate, up, down, output):
        route = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        if route >= expert_ids.shape[0]:
            return
        expert = expert_ids[route]
        hidden = gate.shape[1]
        d_model = gate.shape[2]
        gate_up = cuda.shared.array(shape=1280, dtype='float32')
        x_shared = cuda.shared.array(shape=512, dtype='float32')
        if tid < d_model:
            x_shared[tid] = x[tid]
        if tid < hidden:
            gate_acc = 0.0
            up_acc = 0.0
            for d in range(d_model):
                value = x_shared[d]
                gate_acc += value * gate[expert, tid, d]
                up_acc += value * up[expert, tid, d]
            gate_up[tid] = gate_acc
            gate_up[hidden + tid] = up_acc
        cuda.syncthreads()
        if tid < d_model:
            acc = 0.0
            for h in range(hidden):
                gate_value = gate_up[h]
                up_value = gate_up[hidden + h]
                acc += (gate_value / (1.0 + exp(-gate_value)) * up_value) * down[expert, tid, h]
            output[route, tid] = probabilities[route] * acc


    @cuda.jit(fastmath=True)
    def _gate_up_kernel(x, expert_ids, gate, up, gate_up):
        hidden_tiles = (gate.shape[1] + cuda.blockDim.x - 1) // cuda.blockDim.x
        block = cuda.blockIdx.x
        route = block // hidden_tiles
        tile = block - route * hidden_tiles
        tid = cuda.threadIdx.x
        routes = expert_ids.shape[0]
        hidden = gate.shape[1]
        d_model = gate.shape[2]
        if route >= routes:
            return
        expert = expert_ids[route]
        x_shared = cuda.shared.array(shape=512, dtype='float32')
        if tid < d_model:
            x_shared[tid] = x[tid]
        cuda.syncthreads()
        h = tile * cuda.blockDim.x + tid
        if h < hidden:
            gate_acc = 0.0
            up_acc = 0.0
            for d in range(d_model):
                value = x_shared[d]
                gate_acc += value * gate[expert, h, d]
                up_acc += value * up[expert, h, d]
            gate_up[route, h] = gate_acc
            gate_up[route, hidden + h] = up_acc


    @cuda.jit(fastmath=True)
    def _down_kernel(gate_up, expert_ids, probabilities, down, output):
        d_tiles = (down.shape[1] + cuda.blockDim.x - 1) // cuda.blockDim.x
        block = cuda.blockIdx.x
        route = block // d_tiles
        tile = block - route * d_tiles
        tid = cuda.threadIdx.x
        routes = expert_ids.shape[0]
        hidden = down.shape[2]
        d_model = down.shape[1]
        if route >= routes:
            return
        expert = expert_ids[route]
        probability = probabilities[route]
        d = tile * cuda.blockDim.x + tid
        if d < d_model:
            acc = 0.0
            for h in range(hidden):
                gate_value = gate_up[route, h]
                up_value = gate_up[route, hidden + h]
                # Stable enough for inference and matches torch.silu closely.
                silu = gate_value / (1.0 + exp(-gate_value))
                acc += (silu * up_value) * down[expert, d, h]
            output[route, d] = probability * acc


def numba_moe_step(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    probabilities: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    """Compute one token's routed experts with two Numba CUDA launches."""
    if not _NUMBA_AVAILABLE or not x.is_cuda:
        raise RuntimeError("Numba CUDA MoE is unavailable")
    if x.ndim != 2 or x.shape[0] != 1:
        raise ValueError("numba_moe_step expects x shaped [1, d_model]")
    if x.shape[-1] > 512:
        raise ValueError("numba_moe_step supports d_model <= 512")
    output_dtype = x.dtype
    x = x[0].float().contiguous()
    expert_ids = expert_ids.reshape(-1).to(device=x.device, dtype=torch.int32).contiguous()
    probabilities = probabilities.reshape(-1).float().contiguous()
    gate = gate.float().contiguous()
    up = up.float().contiguous()
    down = down.float().contiguous()
    routes = int(expert_ids.numel())
    hidden = int(gate.shape[1])
    d_model = int(gate.shape[2])
    route_output = torch.empty((routes, d_model), device=x.device, dtype=torch.float32)

    # Numba's CUDA array interface uses the current CUDA stream.  Keep the
    # launches asynchronous so they compose with the surrounding PyTorch step.
    stream = cuda.external_stream(torch.cuda.current_stream(x.device).cuda_stream)
    # The A5000 config is hidden=320, d_model=512. A 512-thread block covers
    # the full output width while keeping two resident blocks possible on
    # Pascal; the old 1024-thread launch needlessly restricted occupancy.
    _fused_route_kernel[routes, 512, stream](
        cuda.as_cuda_array(x), cuda.as_cuda_array(expert_ids),
        cuda.as_cuda_array(probabilities), cuda.as_cuda_array(gate),
        cuda.as_cuda_array(up), cuda.as_cuda_array(down), cuda.as_cuda_array(route_output),
    )
    return route_output.sum(dim=0, keepdim=True).to(dtype=output_dtype)
