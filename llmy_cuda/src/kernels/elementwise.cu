#include "kernels.h"
#include <cuda_runtime.h>

namespace llmy {

// ---------------------------------------------------------------------------
// residual_add_: y += x
// ---------------------------------------------------------------------------

__global__ void residual_add_kernel(float* __restrict__ y,
                                    const float* __restrict__ x,
                                    int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] += x[i];
}

void residual_add_(float* y, const float* x, int64_t n, cudaStream_t s) {
    if (n == 0) return;
    const int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    residual_add_kernel<<<(unsigned)blocks, threads, 0, s>>>(y, x, n);
}

// ---------------------------------------------------------------------------
// scale_: y = x * scale  (scale read from device memory)
// ---------------------------------------------------------------------------

__global__ void scale_kernel(float* __restrict__ y,
                             const float* __restrict__ x,
                             const float* __restrict__ scale,
                             int64_t n) {
    const float s = scale[0];
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = x[i] * s;
}

void scale_(float* y, const float* x, const float* scale,
            int64_t n, cudaStream_t s) {
    if (n == 0) return;
    const int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    scale_kernel<<<(unsigned)blocks, threads, 0, s>>>(y, x, scale, n);
}

// ---------------------------------------------------------------------------
// layernorm_fwd: LayerNorm over last dim with affine.
// ---------------------------------------------------------------------------

__global__ void layernorm_kernel(float* __restrict__ x,
                                 const float* __restrict__ w,
                                 const float* __restrict__ b,
                                 int rows, int cols, float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    float* xr = x + (size_t)row * cols;

    float sum = 0.0f;
    for (int i = tid; i < cols; i += stride) {
        float v = xr[i];
        sum += v;
    }

    // Warp reduce both.
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    }
    __shared__ float smem_sum[32];
    __shared__ float smem_sq[32];
    int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) smem_sum[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        int nw = (blockDim.x + 31) >> 5;
        sum = (tid < nw) ? smem_sum[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, off);
        }
        if (tid == 0) {
            float mean = sum / (float)cols;
            smem_sum[0] = mean;
        }
    }
    __syncthreads();

    const float mean = smem_sum[0];
    float sq = 0.0f;
    for (int i = tid; i < cols; i += stride) {
        float d = xr[i] - mean;
        sq += d * d;
    }
    for (int off = 16; off > 0; off >>= 1)
        sq += __shfl_down_sync(0xffffffffu, sq, off);
    if (lane == 0) smem_sq[warp] = sq;
    __syncthreads();
    if (warp == 0) {
        int nw = (blockDim.x + 31) >> 5;
        sq = (tid < nw) ? smem_sq[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            sq += __shfl_down_sync(0xffffffffu, sq, off);
        if (tid == 0) smem_sq[0] = rsqrtf(sq / (float)cols + eps);
    }
    __syncthreads();
    const float inv = smem_sq[0];

    for (int i = tid; i < cols; i += stride) {
        xr[i] = (xr[i] - mean) * inv * w[i] + b[i];
    }
}

void layernorm_fwd(float* x, const float* w, const float* b,
                   int rows, int cols, float eps, cudaStream_t s) {
    if (rows == 0 || cols == 0) return;
    int threads = 256;
    if (cols <= 128) threads = 128;
    else if (cols >= 2048) threads = 512;
    layernorm_kernel<<<rows, threads, 0, s>>>(x, w, b, rows, cols, eps);
}

// ---------------------------------------------------------------------------
// embedding_lookup
// ---------------------------------------------------------------------------

__global__ void embedding_lookup_kernel(const float* __restrict__ embed_w,
                                        int32_t token,
                                        float* __restrict__ out,
                                        int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    out[d] = embed_w[(size_t)token * D + d];
}

void embedding_lookup(const float* embed_w, int32_t token, float* out,
                      int D, cudaStream_t s) {
    const int threads = 256;
    int blocks = (D + threads - 1) / threads;
    embedding_lookup_kernel<<<blocks, threads, 0, s>>>(embed_w, token, out, D);
}

}  // namespace llmy
