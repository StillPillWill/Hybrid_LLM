#include "kernels.h"
#include <cuda_runtime.h>

namespace llmy {

// y[i] = silu(gate[i]) * up[i], where silu(x) = x * sigmoid(x).
// Uses float4 vectorized loads for the bulk, scalar tail.
__global__ void silu_mul_kernel(const float* __restrict__ gate,
                                const float* __restrict__ up,
                                float* __restrict__ y,
                                int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    // Vectorized loop: 4 elements per iteration when all valid.
    int64_t v4_end = n & ~3LL;
    for (; i * 4 + 3 < v4_end; i += stride) {
        int64_t base = i * 4;
        float4 g = *reinterpret_cast<const float4*>(gate + base);
        float4 u = *reinterpret_cast<const float4*>(up + base);
        float4 r;
        r.x = (g.x / (1.0f + __expf(-g.x))) * u.x;
        r.y = (g.y / (1.0f + __expf(-g.y))) * u.y;
        r.z = (g.z / (1.0f + __expf(-g.z))) * u.z;
        r.w = (g.w / (1.0f + __expf(-g.w))) * u.w;
        *reinterpret_cast<float4*>(y + base) = r;
    }
    // Scalar remainder.
    int64_t rem_start = (v4_end / 4) * 4 + ((v4_end % 4) ? 0 : (i - (v4_end / 4)) * 4);
    // Simpler: just do the tail with a separate scalarized loop.
    // (Handling mixed vectorized/scalar indexing is error-prone; the
    // vectorized loop above already covers all 4-aligned blocks, so the
    // remaining work is at most 3 elements.)
    for (int64_t j = v4_end; j < n; ++j) {
        float g = gate[j];
        y[j] = (g / (1.0f + __expf(-g))) * up[j];
    }
}

void silu_mul(const float* gate, const float* up, float* y,
              int64_t n, cudaStream_t s) {
    if (n == 0) return;
    const int threads = 256;
    int64_t blocks_needed = (n / 4 + threads - 1) / threads;
    // Cap grid at something reasonable to keep occupancy high.
    int64_t blocks = blocks_needed > 65535 ? 65535 : blocks_needed;
    silu_mul_kernel<<<(unsigned)blocks, threads, 0, s>>>(gate, up, y, n);
}

// ---------------------------------------------------------------------------
// x *= sigmoid(gate)
// ---------------------------------------------------------------------------

__global__ void sigmoid_mul_kernel(float* __restrict__ x,
                                   const float* __restrict__ gate,
                                   int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    x[i] *= 1.0f / (1.0f + __expf(-g));
}

void sigmoid_mul_(float* x, const float* gate, int64_t n, cudaStream_t s) {
    if (n == 0) return;
    const int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    sigmoid_mul_kernel<<<(unsigned)blocks, threads, 0, s>>>(x, gate, n);
}

}  // namespace llmy
