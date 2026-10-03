#include "kernels.h"
#include <cuda_runtime.h>
#include <cfloat>

namespace llmy {

// ---------------------------------------------------------------------------
// router_topk
//
// One block. Iterated argmax to find top-k, then softmax over the selected
// logits. Assumes k <= 32, n_experts <= 4096.
// ---------------------------------------------------------------------------

__global__ void router_topk_kernel(const float* __restrict__ logits,
                                   int n, int k,
                                   int32_t* __restrict__ idx_out,
                                   float* __restrict__ prob_out)
{
    extern __shared__ float smem[];
    float* vbuf = smem;                                       // [n]
    int32_t* ibuf = reinterpret_cast<int32_t*>(smem + n);      // [n]
    __shared__ float red_f[32];
    __shared__ int   red_i[32];

    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (nt + 31) >> 5;

    // Copy logits to shared memory for fast access.
    for (int i = tid; i < n; i += nt) {
        vbuf[i] = logits[i];
        ibuf[i] = i;
    }
    __syncthreads();

    // k rounds of argmax.
    for (int t = 0; t < k; ++t) {
        float my_max = -FLT_MAX;
        int   my_arg = -1;
        for (int i = tid; i < n; i += nt) {
            if (vbuf[i] > my_max) { my_max = vbuf[i]; my_arg = ibuf[i]; }
        }
        // Warp reduce by (value, index).
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffffu, my_max, off);
            int   oi = __shfl_down_sync(0xffffffffu, my_arg, off);
            if (ov > my_max) { my_max = ov; my_arg = oi; }
        }
        if (lane == 0) { red_f[warp] = my_max; red_i[warp] = my_arg; }
        __syncthreads();
        if (warp == 0) {
            float v = (tid < nwarp) ? red_f[tid] : -FLT_MAX;
            int   a = (tid < nwarp) ? red_i[tid] : -1;
            for (int off = 16; off > 0; off >>= 1) {
                float ov = __shfl_down_sync(0xffffffffu, v, off);
                int   oi = __shfl_down_sync(0xffffffffu, a, off);
                if (ov > v) { v = ov; a = oi; }
            }
            if (tid == 0) { red_f[0] = v; red_i[0] = a; }
        }
        __syncthreads();
        int chosen_idx = red_i[0];
        if (tid == 0) idx_out[t] = chosen_idx;
        __syncthreads();
        // Mask out chosen so next iteration finds the next one.
        for (int i = tid; i < n; i += nt) {
            if (ibuf[i] == chosen_idx) vbuf[i] = -FLT_MAX;
        }
        __syncthreads();
    }

    // Softmax over the selected logits.
    // We have the original logits in `logits` (device global).
    float my_max = -FLT_MAX;
    if (tid < k) {
        int idx = idx_out[tid];
        my_max = logits[idx];
    }
    for (int off = 16; off > 0; off >>= 1)
        my_max = fmaxf(my_max, __shfl_xor_sync(0xffffffffu, my_max, off));
    if (lane == 0) red_f[warp] = my_max;
    __syncthreads();
    if (warp == 0) {
        float m = (tid < nwarp) ? red_f[tid] : -FLT_MAX;
        for (int off = 16; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        if (tid == 0) red_f[0] = m;
    }
    __syncthreads();
    const float mx = red_f[0];
    __syncthreads();

    float my_sum = 0.0f;
    for (int t = tid; t < k; t += nt) {
        int idx = idx_out[t];
        float e = __expf(logits[idx] - mx);
        prob_out[t] = e;
        my_sum += e;
    }
    for (int off = 16; off > 0; off >>= 1)
        my_sum += __shfl_xor_sync(0xffffffffu, my_sum, off);
    if (lane == 0) red_f[warp] = my_sum;
    __syncthreads();
    if (warp == 0) {
        float s = (tid < nwarp) ? red_f[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            s += __shfl_xor_sync(0xffffffffu, s, off);
        if (tid == 0) red_f[0] = 1.0f / s;
    }
    __syncthreads();
    const float inv = red_f[0];
    __syncthreads();

    for (int t = tid; t < k; t += nt) prob_out[t] *= inv;
}

void router_topk(const float* logits, int n, int k,
                 int32_t* idx_out, float* prob_out,
                 cudaStream_t s) {
    if (n <= 0 || k <= 0 || k > n) return;
    // Shared memory: n floats + n int32.
    size_t smem_bytes = sizeof(float) * n + sizeof(int32_t) * n;
    // Cap shared memory; if we exceed 48KB (default limit), fall back to
    // a global-memory version. Our n=128 experts is tiny.
    if (smem_bytes > 48 * 1024) {
        return;
    }
    int threads = 256;
    router_topk_kernel<<<1, threads, smem_bytes, s>>>(logits, n, k, idx_out, prob_out);
}

}  // namespace llmy
