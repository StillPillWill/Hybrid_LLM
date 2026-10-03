#include "kernels.h"
#include <cuda_runtime.h>
#include <cfloat>

namespace llmy {

// ---------------------------------------------------------------------------
// moe_expert_fwd
//
// Single expert. x [D], gate_w [H, D], up_w [H, D], down_w [D, H].
// Computes: gate = W_gate @ x, up = W_up @ x, act = silu(gate) * up,
// out = W_down @ act.
//
// One block. Shared memory holds h_gate [H] and act [H].
// ---------------------------------------------------------------------------

__global__ void moe_expert_fwd_kernel(const float* __restrict__ x,
                                      const float* __restrict__ gate_w,
                                      const float* __restrict__ up_w,
                                      const float* __restrict__ down_w,
                                      float* __restrict__ out,
                                      int D, int H)
{
    extern __shared__ float smem[];
    float* act = smem;   // [H]

    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (nt + 31) >> 5;

    __shared__ float red[32];

    // Compute act[h] = silu(gate[h]) * up[h].
    // Parallelize over h; each warp handles a subset.
    for (int h = warp; h < H; h += nwarp) {
        const float* gw = gate_w + (size_t)h * D;
        const float* uw = up_w + (size_t)h * D;
        float sg = 0.0f, su = 0.0f;
        for (int d = lane; d < D; d += 32) {
            float xv = x[d];
            sg += gw[d] * xv;
            su += uw[d] * xv;
        }
        for (int off = 16; off > 0; off >>= 1) {
            sg += __shfl_down_sync(0xffffffffu, sg, off);
            su += __shfl_down_sync(0xffffffffu, su, off);
        }
        if (lane == 0) {
            float g = sg / (1.0f + __expf(-sg));
            act[h] = g * su;
        }
    }
    __syncthreads();

    // Compute out[d] = sum_h down_w[d, h] * act[h].
    // Parallelize over d.
    for (int d = tid; d < D; d += nt) {
        const float* dw = down_w + (size_t)d * H;
        float acc = 0.0f;
        for (int h = 0; h < H; ++h) acc += dw[h] * act[h];
        out[d] = acc;
    }
}

void moe_expert_fwd(const float* x,
                    const float* gate_w, const float* up_w, const float* down_w,
                    float* out, int D, int H, cudaStream_t s) {
    if (D <= 0 || H <= 0) return;
    int threads = 256;
    size_t shmem = sizeof(float) * H;
    moe_expert_fwd_kernel<<<1, threads, shmem, s>>>(
        x, gate_w, up_w, down_w, out, D, H);
}

// ---------------------------------------------------------------------------
// moe_route
//
// Full MoE for one token. Iterates over k selected experts, calls the
// expert computation inline (using the same math as moe_expert_fwd) but
// writes into a scratch buffer indexed by the route, then combines the
// weighted sum.
//
// Grid: 1 block. The kernel is sequential over experts but parallel over
// the D and H dimensions inside each expert's computation.
// ---------------------------------------------------------------------------

__global__ void moe_route_kernel(const float* __restrict__ x,
                                 const float* __restrict__ experts_gate,
                                 const float* __restrict__ experts_up,
                                 const float* __restrict__ experts_down,
                                 const int32_t* __restrict__ idx,
                                 const float* __restrict__ prob,
                                 float* __restrict__ out,
                                 float* __restrict__ expert_scratch,
                                 int D, int H, int E, int k)
{
    extern __shared__ float smem[];
    float* act = smem;                 // [H]

    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (nt + 31) >> 5;

    // Zero output.
    for (int d = tid; d < D; d += nt) out[d] = 0.0f;
    __syncthreads();

    // Iterate experts.
    for (int t = 0; t < k; ++t) {
        int e = idx[t];
        if (e < 0 || e >= E) continue;
        float p = prob[t];

        const float* gw = experts_gate + (size_t)e * H * D;
        const float* uw = experts_up   + (size_t)e * H * D;
        const float* dw = experts_down + (size_t)e * D * H;

        // Compute act[h] = silu(gate[h]) * up[h].
        for (int h = warp; h < H; h += nwarp) {
            const float* grow = gw + (size_t)h * D;
            const float* urow = uw + (size_t)h * D;
            float sg = 0.0f, su = 0.0f;
            for (int d = lane; d < D; d += 32) {
                float xv = x[d];
                sg += grow[d] * xv;
                su += urow[d] * xv;
            }
            for (int off = 16; off > 0; off >>= 1) {
                sg += __shfl_down_sync(0xffffffffu, sg, off);
                su += __shfl_down_sync(0xffffffffu, su, off);
            }
            if (lane == 0) {
                float g = sg / (1.0f + __expf(-sg));
                act[h] = g * su;
            }
        }
        __syncthreads();

        // Accumulate into out[d] with weight p.
        for (int d = tid; d < D; d += nt) {
            const float* drow = dw + (size_t)d * H;
            float acc = 0.0f;
            for (int h = 0; h < H; ++h) acc += drow[h] * act[h];
            out[d] += p * acc;
        }
        __syncthreads();
    }
    (void)expert_scratch;
}

void moe_route(const float* x,
               const float* experts_gate, const float* experts_up,
               const float* experts_down,
               const int32_t* idx, const float* prob,
               float* out, float* expert_scratch,
               int D, int H, int E, int k, cudaStream_t s) {
    if (D <= 0 || H <= 0 || k <= 0) return;
    int threads = 256;
    size_t shmem = sizeof(float) * H;
    moe_route_kernel<<<1, threads, shmem, s>>>(
        x, experts_gate, experts_up, experts_down,
        idx, prob, out, expert_scratch, D, H, E, k);
}

// ---------------------------------------------------------------------------
// Routing has picked the k experts by decode time. Give each route its own grid
// work so they run in parallel without copying expert weights.
// ---------------------------------------------------------------------------

__global__ void moe_selected_gate_up_kernel(
        const float* __restrict__ x,
        const int64_t* __restrict__ idx,
        const float* __restrict__ gate_up,
        float* __restrict__ activation,
        int D, int H, int E, int k) {
    const int route = blockIdx.y;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (route >= k) return;

    const int64_t expert = idx[route];
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    if (row < H && expert >= 0 && expert < E) {
        const size_t expert_offset = (size_t)expert * (2 * (size_t)H * D);
        const float* gate_row = gate_up + expert_offset + (size_t)row * D;
        const float* up_row = gate_up + expert_offset + (size_t)(H + row) * D;
        for (int col = lane; col < D; col += 32) {
            const float value = x[col];
            gate_acc = fmaf(gate_row[col], value, gate_acc);
            up_acc = fmaf(up_row[col], value, up_acc);
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_acc += __shfl_down_sync(0xffffffffu, gate_acc, offset);
        up_acc += __shfl_down_sync(0xffffffffu, up_acc, offset);
    }
    if (lane == 0 && row < H) {
        const float gate = gate_acc / (1.0f + expf(-gate_acc));
        activation[(size_t)route * H + row] = gate * up_acc;
    }
}

__global__ void moe_selected_down_kernel(
        const int64_t* __restrict__ idx,
        const float* __restrict__ prob,
        const float* __restrict__ down,
        const float* __restrict__ activation,
        float* __restrict__ route_output,
        int D, int H, int E, int k) {
    const int route = blockIdx.y;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (route >= k) return;

    float acc = 0.0f;
    const int64_t expert = idx[route];
    if (row < D && expert >= 0 && expert < E) {
        const float* down_row = down + (size_t)expert * D * H + (size_t)row * H;
        const float* act = activation + (size_t)route * H;
        for (int col = lane; col < H; col += 32) {
            acc = fmaf(down_row[col], act[col], acc);
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0 && row < D) {
        route_output[(size_t)route * D + row] = prob[route] * acc;
    }
}

__global__ void moe_selected_reduce_kernel(
        const float* __restrict__ route_output,
        float* __restrict__ out,
        int D, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= D) return;
    float sum = 0.0f;
    for (int route = 0; route < k; ++route) {
        sum += route_output[(size_t)route * D + row];
    }
    out[row] = sum;
}

void moe_selected_token_fp32(const float* x, const int64_t* idx,
                             const float* prob, const float* gate_up,
                             const float* down, float* activation,
                             float* route_output, float* out,
                             int D, int H, int E, int k,
                             cudaStream_t s) {
    if (D <= 0 || H <= 0 || E <= 0 || k <= 0) return;
    constexpr int kWarpsPerBlock = 8;
    constexpr int kThreads = kWarpsPerBlock * 32;
    const dim3 gate_grid((H + kWarpsPerBlock - 1) / kWarpsPerBlock, k);
    const dim3 down_grid((D + kWarpsPerBlock - 1) / kWarpsPerBlock, k);
    moe_selected_gate_up_kernel<<<gate_grid, kThreads, 0, s>>>(
        x, idx, gate_up, activation, D, H, E, k);
    moe_selected_down_kernel<<<down_grid, kThreads, 0, s>>>(
        idx, prob, down, activation, route_output, D, H, E, k);
    moe_selected_reduce_kernel<<<(D + 255) / 256, 256, 0, s>>>(
        route_output, out, D, k);
}

}  // namespace llmy
