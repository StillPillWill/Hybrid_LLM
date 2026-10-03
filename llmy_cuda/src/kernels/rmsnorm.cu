#include "kernels.h"
#include <cuda_runtime.h>

namespace llmy {

// One block per row. Computes sum(x^2) via block reduction, then normalizes
// and scales by weight. Assumes cols <= 4096 and cols is a multiple of 4
// (float4 vectorized path); falls back to scalar for odd sizes.
//
// Numerically: variance = mean(x^2), not (mean(x))^2 removal. This matches
// the PyTorch reference which uses `x.float().pow(2).mean(-1)`.
__global__ void rmsnorm_kernel(const float* __restrict__ x,
                               const float* __restrict__ w,
                               float* __restrict__ out,
                               int cols, float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const float* xr = x + (size_t)row * cols;
    float* orow = out + (size_t)row * cols;

    // Sum of squares.
    float ss = 0.0f;
    // Each element must be visited exactly once. The old float4-style loop
    // overlapped the first four lanes and skipped the remainder for cols
    // larger than one block (512-wide model rows were affected).
    for (int i = tid; i < cols; i += stride) {
        float v = xr[i];
        ss += v * v;
    }

    // Warp reduce.
    for (int off = 16; off > 0; off >>= 1)
        ss += __shfl_down_sync(0xffffffffu, ss, off);

    __shared__ float smem[32];
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) smem[warp] = ss;
    __syncthreads();

    if (warp == 0) {
        int nw = (blockDim.x + 31) >> 5;
        ss = (tid < nw) ? smem[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            ss += __shfl_down_sync(0xffffffffu, ss, off);
        if (tid == 0) smem[0] = rsqrtf(ss / (float)cols + eps);
    }
    __syncthreads();

    const float inv_rms = smem[0];

    // Normalize.
    for (int j = tid; j < cols; j += stride) {
        orow[j] = xr[j] * inv_rms * w[j];
    }
}

void rmsnorm_fwd(const float* x, const float* w, float* y,
                 int rows, int cols, float eps, cudaStream_t s) {
    if (rows == 0 || cols == 0) return;
    int threads = 256;
    if (cols <= 128) threads = 128;
    else if (cols >= 2048) threads = 512;
    rmsnorm_kernel<<<rows, threads, 0, s>>>(x, w, y, cols, eps);
}

}  // namespace llmy
