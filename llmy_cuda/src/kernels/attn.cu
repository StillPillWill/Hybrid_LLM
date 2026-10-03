#include "kernels.h"
#include <cuda_runtime.h>
#include <math.h>
#include <cfloat>

namespace llmy {

// ---------------------------------------------------------------------------
// banded_attn — prefill path with T > 1
//
// One block per (head, query position). For each query position t, attend
// over keys in [max(0, t-W+1), t]. Optional segment mask.
//
// We use the standard two-pass flash attention (max then softmax then sum).
// ---------------------------------------------------------------------------

__global__ void banded_attn_kernel(const float* __restrict__ q,   // [H, T, D]
                                   const float* __restrict__ k,
                                   const float* __restrict__ v,
                                   const int32_t* __restrict__ seg, // [T] or null
                                   float* __restrict__ out,        // [H, T, D]
                                   int H, int T, int D, int W) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    if (t >= T) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = nthreads >> 5;

    const float* qh = q + ((size_t)h * T + t) * D;
    const int lo = (t - W + 1 > 0) ? (t - W + 1) : 0;
    const int n = t - lo + 1;
    const float scale = rsqrtf((float)D);

    // Shared scores buffer.
    extern __shared__ float smem[];
    float* scores = smem;                    // [W]
    __shared__ float reduce_buf[32];

    // Pass 1: compute raw scores.
    // Each warp handles a subset of keys [lo, t]; the warp does a dot
    // product over D for each key assigned to it.
    for (int s = warp; s < n; s += nwarp) {
        const int key_pos = lo + s;
        const float* kh = k + ((size_t)h * T + key_pos) * D;
        float dot = 0.0f;
        for (int d = lane; d < D; d += 32)
            dot += qh[d] * kh[d];
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, off);
        if (lane == 0) scores[s] = dot * scale;
    }
    __syncthreads();

    // Segment mask: mark incompatible scores as -inf.
    if (seg != nullptr) {
        int my_seg = seg[t];
        for (int s = tid; s < n; s += nthreads) {
            if (seg[lo + s] != my_seg) scores[s] = -FLT_MAX;
        }
        __syncthreads();
    }

    // Pass 2: max.
    float m_local = -FLT_MAX;
    for (int s = tid; s < n; s += nthreads)
        m_local = fmaxf(m_local, scores[s]);
    // Block reduce max.
    for (int off = 16; off > 0; off >>= 1)
        m_local = fmaxf(m_local, __shfl_xor_sync(0xffffffffu, m_local, off));
    if (lane == 0) reduce_buf[warp] = m_local;
    __syncthreads();
    if (warp == 0) {
        m_local = (tid < nwarp) ? reduce_buf[tid] : -FLT_MAX;
        for (int off = 16; off > 0; off >>= 1)
            m_local = fmaxf(m_local, __shfl_xor_sync(0xffffffffu, m_local, off));
        if (tid == 0) reduce_buf[0] = m_local;
    }
    __syncthreads();
    const float m = reduce_buf[0];

    // Pass 3: exp + sum.
    float s_local = 0.0f;
    for (int s = tid; s < n; s += nthreads) {
        float e = __expf(scores[s] - m);
        scores[s] = e;
        s_local += e;
    }
    for (int off = 16; off > 0; off >>= 1)
        s_local += __shfl_xor_sync(0xffffffffu, s_local, off);
    if (lane == 0) reduce_buf[warp] = s_local;
    __syncthreads();
    if (warp == 0) {
        s_local = (tid < nwarp) ? reduce_buf[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            s_local += __shfl_xor_sync(0xffffffffu, s_local, off);
        if (tid == 0) reduce_buf[0] = 1.0f / s_local;
    }
    __syncthreads();
    const float inv_sum = reduce_buf[0];

    // Pass 4: weighted sum over V.
    float* oh = out + ((size_t)h * T + t) * D;
    for (int d = tid; d < D; d += nthreads) {
        float acc = 0.0f;
        for (int s = 0; s < n; ++s) {
            const float* vh = v + ((size_t)h * T + (lo + s)) * D;
            acc += scores[s] * vh[d];
        }
        oh[d] = acc * inv_sum;
    }
}

void banded_attn(const float* q, const float* k, const float* v,
                 const int32_t* segment_ids,
                 float* out, int H, int T, int D, int W, cudaStream_t s) {
    if (H == 0 || T == 0 || D == 0) return;
    int threads = 128;
    size_t shmem = sizeof(float) * W;
    dim3 grid(H, T);
    banded_attn_kernel<<<grid, threads, shmem, s>>>(
        q, k, v, segment_ids, out, H, T, D, W);
}

// ---------------------------------------------------------------------------
// local_attn_decode — T=1 fast path
//
// One block per head. W_hist keys/values in the history buffer (typically
// <= 128). Uses similar two-pass softmax.
// ---------------------------------------------------------------------------

__global__ void local_attn_decode_kernel(const float* __restrict__ q,        // [H, D]
                                         const float* __restrict__ k_hist,   // [H, W_hist, D]
                                         const float* __restrict__ v_hist,   // [H, W_hist, D]
                                         const uint8_t* __restrict__ valid,  // [H, W_hist] or null
                                         float* __restrict__ out,            // [H, D]
                                         int H, int W_hist, int D) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = nthreads >> 5;

    extern __shared__ float smem[];
    float* scores = smem;             // [W_hist]
    __shared__ float reduce_buf[32];

    const float* qh = q + h * D;
    const float scale = rsqrtf((float)D);

    // Pass 1: scores per key.
    for (int w = warp; w < W_hist; w += nwarp) {
        if (valid != nullptr && !valid[h * W_hist + w]) {
            if (lane == 0) scores[w] = -FLT_MAX;
            continue;
        }
        const float* kh = k_hist + ((size_t)h * W_hist + w) * D;
        float dot = 0.0f;
        for (int d = lane; d < D; d += 32)
            dot += qh[d] * kh[d];
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, off);
        if (lane == 0) scores[w] = dot * scale;
    }
    __syncthreads();

    // Pass 2: max.
    float m_local = -FLT_MAX;
    for (int w = tid; w < W_hist; w += nthreads)
        m_local = fmaxf(m_local, scores[w]);
    for (int off = 16; off > 0; off >>= 1)
        m_local = fmaxf(m_local, __shfl_xor_sync(0xffffffffu, m_local, off));
    if (lane == 0) reduce_buf[warp] = m_local;
    __syncthreads();
    if (warp == 0) {
        m_local = (tid < nwarp) ? reduce_buf[tid] : -FLT_MAX;
        for (int off = 16; off > 0; off >>= 1)
            m_local = fmaxf(m_local, __shfl_xor_sync(0xffffffffu, m_local, off));
        if (tid == 0) reduce_buf[0] = m_local;
    }
    __syncthreads();
    const float m = reduce_buf[0];

    // Pass 3: exp + sum.
    float s_local = 0.0f;
    for (int w = tid; w < W_hist; w += nthreads) {
        float e = __expf(scores[w] - m);
        scores[w] = e;
        s_local += e;
    }
    for (int off = 16; off > 0; off >>= 1)
        s_local += __shfl_xor_sync(0xffffffffu, s_local, off);
    if (lane == 0) reduce_buf[warp] = s_local;
    __syncthreads();
    if (warp == 0) {
        s_local = (tid < nwarp) ? reduce_buf[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            s_local += __shfl_xor_sync(0xffffffffu, s_local, off);
        if (tid == 0) reduce_buf[0] = 1.0f / s_local;
    }
    __syncthreads();
    const float inv_sum = reduce_buf[0];

    // Pass 4: weighted sum.
    float* oh = out + h * D;
    for (int d = tid; d < D; d += nthreads) {
        float acc = 0.0f;
        for (int w = 0; w < W_hist; ++w) {
            const float* vh = v_hist + ((size_t)h * W_hist + w) * D;
            acc += scores[w] * vh[d];
        }
        oh[d] = acc * inv_sum;
    }
}

void local_attn_decode(const float* q, const float* k_hist, const float* v_hist,
                       const uint8_t* valid, float* out,
                       int H, int W_hist, int D, cudaStream_t s) {
    if (H == 0 || W_hist == 0 || D == 0) return;
    int threads = 128;
    if (W_hist >= 128) threads = 256;
    size_t shmem = sizeof(float) * W_hist;
    local_attn_decode_kernel<<<H, threads, shmem, s>>>(
        q, k_hist, v_hist, valid, out, H, W_hist, D);
}

// ---------------------------------------------------------------------------
// attn_topk_scores — q @ K^T with top-k selection
//
// One warp per head. Each lane maintains a small (index, value) pair stack.
// For M <= 4096 and k <= 32 this is fast enough.
// ---------------------------------------------------------------------------

__global__ void attn_topk_scores_kernel(const float* __restrict__ q,   // [H, D]
                                        const float* __restrict__ K,   // [M, D]
                                        int32_t* __restrict__ idx_out, // [H, k]
                                        int H, int M, int D, int k) {
    const int h = blockIdx.x;
    const int lane = threadIdx.x & 31;

    const float* qh = q + h * D;
    const float scale = rsqrtf((float)D);

    // We do this iteratively: k rounds of argmax, marking selected entries
    // as -inf. O(M * k) time. Fine for our M (up to a few thousand) and
    // small k (32).
    //
    // Uses shared memory to hold scores.
    extern __shared__ float smem[];
    float* scores = smem;    // [M]
    __shared__ float vmax[1];
    __shared__ int varg[1];

    // Compute all scores in parallel.
    for (int m = lane; m < M; m += 32) {
        const float* km = K + (size_t)m * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += qh[d] * km[d];
        scores[m] = dot * scale;
    }
    __syncwarp();

    // k rounds of argmax.
    for (int t = 0; t < k; ++t) {
        // Find argmax.
        float my_max = -FLT_MAX;
        int my_arg = -1;
        for (int m = lane; m < M; m += 32) {
            float v = scores[m];
            if (v > my_max) { my_max = v; my_arg = m; }
        }
        // Warp-reduce argmax by (value, index).
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffffu, my_max, off);
            int   oi = __shfl_down_sync(0xffffffffu, my_arg, off);
            if (ov > my_max) { my_max = ov; my_arg = oi; }
        }
        if (lane == 0) { vmax[0] = my_max; varg[0] = my_arg; }
        __syncwarp();
        int chosen = varg[0];
        if (lane == 0) idx_out[h * k + t] = chosen;
        __syncwarp();
        // Mark as -inf.
        for (int m = lane; m < M; m += 32) {
            if (m == chosen) scores[m] = -FLT_MAX;
        }
        __syncwarp();
    }
}

void attn_topk_scores(const float* q, const float* K,
                      int32_t* idx_out,
                      int H, int M, int D, int k, cudaStream_t s) {
    if (k > 32) {
        // Our k is <= 32 always. A larger k would need a full sort.
        return;
    }
    size_t shmem = sizeof(float) * M;
    attn_topk_scores_kernel<<<H, 32, shmem, s>>>(q, K, idx_out, H, M, D, k);
}

// ---------------------------------------------------------------------------
// remote_attn — attention over precomputed top-k indices
// ---------------------------------------------------------------------------

__global__ void remote_attn_kernel(const float* __restrict__ q,        // [H, D]
                                   const float* __restrict__ K,        // [M, D]
                                   const float* __restrict__ V,        // [M, D]
                                   const int32_t* __restrict__ idx,    // [H, k]
                                   float* __restrict__ out,            // [H, D]
                                   int H, int M, int D, int k) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = nthreads >> 5;

    const float* qh = q + h * D;
    const int32_t* ih = idx + h * k;
    const float scale = rsqrtf((float)D);

    extern __shared__ float smem[];
    float* scores = smem;             // [k]
    __shared__ float reduce_buf[32];

    // Pass 1: scores for the k selected keys.
    for (int i = warp; i < k; i += nwarp) {
        int m = ih[i];
        if (m < 0) {
            if (lane == 0) scores[i] = -FLT_MAX;
            continue;
        }
        const float* kh = K + (size_t)m * D;
        float dot = 0.0f;
        for (int d = lane; d < D; d += 32)
            dot += qh[d] * kh[d];
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, off);
        if (lane == 0) scores[i] = dot * scale;
    }
    __syncthreads();

    // Pass 2: max + softmax.
    float m_local = -FLT_MAX;
    for (int i = tid; i < k; i += nthreads)
        m_local = fmaxf(m_local, scores[i]);
    for (int off = 16; off > 0; off >>= 1)
        m_local = fmaxf(m_local, __shfl_xor_sync(0xffffffffu, m_local, off));
    if (lane == 0) reduce_buf[warp] = m_local;
    __syncthreads();
    if (warp == 0) {
        m_local = (tid < nwarp) ? reduce_buf[tid] : -FLT_MAX;
        for (int off = 16; off > 0; off >>= 1)
            m_local = fmaxf(m_local, __shfl_xor_sync(0xffffffffu, m_local, off));
        if (tid == 0) reduce_buf[0] = m_local;
    }
    __syncthreads();
    const float m = reduce_buf[0];

    float s_local = 0.0f;
    for (int i = tid; i < k; i += nthreads) {
        float e = __expf(scores[i] - m);
        scores[i] = e;
        s_local += e;
    }
    for (int off = 16; off > 0; off >>= 1)
        s_local += __shfl_xor_sync(0xffffffffu, s_local, off);
    if (lane == 0) reduce_buf[warp] = s_local;
    __syncthreads();
    if (warp == 0) {
        s_local = (tid < nwarp) ? reduce_buf[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            s_local += __shfl_xor_sync(0xffffffffu, s_local, off);
        if (tid == 0) reduce_buf[0] = 1.0f / s_local;
    }
    __syncthreads();
    const float inv_sum = reduce_buf[0];

    // Pass 3: weighted sum over V.
    float* oh = out + h * D;
    for (int d = tid; d < D; d += nthreads) {
        float acc = 0.0f;
        for (int i = 0; i < k; ++i) {
            int m = ih[i];
            if (m < 0) continue;
            const float* vh = V + (size_t)m * D;
            acc += scores[i] * vh[d];
        }
        oh[d] = acc * inv_sum;
    }
}

void remote_attn(const float* q, const float* K, const float* V,
                 const int32_t* idx, float* out,
                 int H, int M, int D, int k, cudaStream_t s) {
    if (H == 0 || D == 0 || k == 0) return;
    int threads = 128;
    if (k >= 64) threads = 256;
    size_t shmem = sizeof(float) * k;
    remote_attn_kernel<<<H, threads, shmem, s>>>(
        q, K, V, idx, out, H, M, D, k);
}

}  // namespace llmy
