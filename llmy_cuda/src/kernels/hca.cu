#include "kernels.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <algorithm>

namespace llmy {

// ---------------------------------------------------------------------------
// hca_local_attn_decode
//
// Single head (no head dimension). q [D], k_cache [W, D], v_cache [W, D].
// Softmax over W, weighted sum, out [D].
// ---------------------------------------------------------------------------

__global__ void hca_local_attn_decode_kernel(
    const float* __restrict__ q,        // [D]
    const float* __restrict__ k_cache,  // [W, D]
    const float* __restrict__ v_cache,  // [W, D]
    const uint8_t* __restrict__ valid,  // [W] or null
    float* __restrict__ out,            // [D]
    int W, int D)
{
    extern __shared__ float smem[];
    float* sc = smem;                 // [W]
    __shared__ float red[32];

    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (nt + 31) >> 5;

    const float scale = rsqrtf((float)D);

    // Scores.
    for (int w = tid; w < W; w += nt) {
        if (valid != nullptr && !valid[w]) { sc[w] = -FLT_MAX; continue; }
        const float* kw = k_cache + (size_t)w * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += q[d] * kw[d];
        sc[w] = dot * scale;
    }
    __syncthreads();

    // Max
    float m = -FLT_MAX;
    for (int w = tid; w < W; w += nt) m = fmaxf(m, sc[w]);
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    if (warp == 0) {
        m = (tid < nwarp) ? red[tid] : -FLT_MAX;
        for (int off = 16; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        if (tid == 0) red[0] = m;
    }
    __syncthreads();
    m = red[0];
    __syncthreads();

    // Exp + sum
    float s = 0.0f;
    for (int w = tid; w < W; w += nt) {
        float e = __expf(sc[w] - m);
        sc[w] = e;
        s += e;
    }
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_xor_sync(0xffffffffu, s, off);
    if (lane == 0) red[warp] = s;
    __syncthreads();
    if (warp == 0) {
        s = (tid < nwarp) ? red[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            s += __shfl_xor_sync(0xffffffffu, s, off);
        if (tid == 0) red[0] = 1.0f / s;
    }
    __syncthreads();
    const float inv = red[0];
    __syncthreads();

    // Normalize
    for (int w = tid; w < W; w += nt) sc[w] *= inv;
    __syncthreads();

    // Weighted sum.
    for (int d = tid; d < D; d += nt) {
        float acc = 0.0f;
        for (int w = 0; w < W; ++w) acc += sc[w] * v_cache[(size_t)w * D + d];
        out[d] = acc;
    }
}

void hca_local_attn_decode(const float* q,
                           const float* k_cache,
                           const float* v_cache,
                           const uint8_t* valid,
                           float* out,
                           int W, int D, cudaStream_t s) {
    if (W <= 0 || D <= 0) return;
    size_t shmem = sizeof(float) * W;
    int threads = 256;
    hca_local_attn_decode_kernel<<<1, threads, shmem, s>>>(
        q, k_cache, v_cache, valid, out, W, D);
}

// ---------------------------------------------------------------------------
// hca_compress_step
//
// Same structure as csa_compress_step but with Dk = Dv = D and no head
// dimension.
// ---------------------------------------------------------------------------

__global__ void hca_compress_step_kernel(
    const float* __restrict__ hidden,
    float* __restrict__ pending_sum,
    int32_t* __restrict__ pending_count,
    int32_t* __restrict__ pending_seg,
    const float* __restrict__ key_w,
    const float* __restrict__ value_w,
    float* __restrict__ mem_keys,
    float* __restrict__ mem_values,
    int32_t* __restrict__ mem_positions,
    int32_t* __restrict__ mem_segments,
    int32_t mem_length,
    int D, int capacity,
    int32_t position, int32_t segment, int ratio)
{
    extern __shared__ float smem[];
    float* mean = smem;  // [D]

    const int tid = threadIdx.x;
    const int nt = blockDim.x;

    __shared__ int s_seg_changed;
    __shared__ int s_emitted;
    __shared__ int s_slot;

    if (tid == 0) {
        s_seg_changed = (pending_seg[0] != segment) ? 1 : 0;
        if (s_seg_changed) pending_seg[0] = segment;
    }
    __syncthreads();

    for (int d = tid; d < D; d += nt) {
        float prev = s_seg_changed ? 0.0f : pending_sum[d];
        pending_sum[d] = prev + hidden[d];
    }
    __syncthreads();

    if (tid == 0) {
        int prev_count = s_seg_changed ? 0 : pending_count[0];
        pending_count[0] = prev_count + 1;
        s_emitted = (pending_count[0] >= ratio) ? 1 : 0;
        s_slot = mem_length % capacity;
    }
    __syncthreads();

    const float inv_count = 1.0f / fmaxf((float)pending_count[0], 1.0f);
    for (int d = tid; d < D; d += nt) {
        mean[d] = pending_sum[d] * inv_count;
    }
    __syncthreads();

    // Project to key.
    for (int i = tid; i < D; i += nt) {
        const float* wrow = key_w + (size_t)i * D;
        float acc = 0.0f;
        for (int d = 0; d < D; ++d) acc += wrow[d] * mean[d];
        mem_keys[(size_t)s_slot * D + i] = acc;
    }
    __syncthreads();

    // Project to value.
    for (int i = tid; i < D; i += nt) {
        const float* wrow = value_w + (size_t)i * D;
        float acc = 0.0f;
        for (int d = 0; d < D; ++d) acc += wrow[d] * mean[d];
        mem_values[(size_t)s_slot * D + i] = acc;
    }
    __syncthreads();

    if (tid == 0) {
        if (s_emitted) {
            mem_positions[s_slot] = position;
            mem_segments[s_slot] = segment;
            pending_count[0] = 0;
            for (int d = 0; d < D; ++d) pending_sum[d] = 0.0f;
        } else {
            mem_positions[s_slot] = -1;
            mem_segments[s_slot] = -1;
        }
    }
}

void hca_compress_step(const float* hidden,
                       float* pending_sum,
                       int32_t* pending_count,
                       int32_t* pending_seg,
                       const float* key_w,
                       const float* value_w,
                       float* mem_keys,
                       float* mem_values,
                       int32_t* mem_positions,
                       int32_t* mem_segments,
                       int32_t mem_length,
                       int D, int capacity,
                       int32_t position, int32_t segment, int ratio,
                       cudaStream_t s) {
    if (D <= 0 || capacity <= 0) return;
    int threads = D < 1024 ? D : 1024;
    if (threads < 128) threads = 128;
    size_t shmem = sizeof(float) * D;
    hca_compress_step_kernel<<<1, threads, shmem, s>>>(
        hidden, pending_sum, pending_count, pending_seg,
        key_w, value_w,
        mem_keys, mem_values, mem_positions, mem_segments, mem_length,
        D, capacity, position, segment, ratio);
}

// ---------------------------------------------------------------------------
// hca_global_attn_decode
//
// Dense attention over all visible compressed entries (no top-k; the
// reference implementation uses dense global attention by default).
// ---------------------------------------------------------------------------

__global__ void hca_global_attn_decode_kernel(
    const float* __restrict__ q,             // [D]
    const float* __restrict__ mem_keys,      // [capacity, D]
    const float* __restrict__ mem_values,    // [capacity, D]
    const int32_t* __restrict__ mem_positions,
    int32_t current_position,
    int32_t memory_length,
    float* __restrict__ out,                 // [D]
    int D, int capacity)
{
    extern __shared__ float smem[];
    float* sc = smem;  // [capacity]
    __shared__ float red[32];

    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (nt + 31) >> 5;

    const float scale = rsqrtf((float)D);

    // Scores.
    for (int m = tid; m < capacity; m += nt) {
        if (m >= memory_length) { sc[m] = -FLT_MAX; continue; }
        int pos = mem_positions[m];
        if (pos < 0 || pos > current_position) { sc[m] = -FLT_MAX; continue; }
        const float* km = mem_keys + (size_t)m * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += q[d] * km[d];
        sc[m] = dot * scale;
    }
    __syncthreads();

    // Max
    float m = -FLT_MAX;
    for (int i = tid; i < capacity; i += nt) m = fmaxf(m, sc[i]);
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    if (warp == 0) {
        m = (tid < nwarp) ? red[tid] : -FLT_MAX;
        for (int off = 16; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        if (tid == 0) red[0] = m;
    }
    __syncthreads();
    m = red[0];
    __syncthreads();

    float s = 0.0f;
    for (int i = tid; i < capacity; i += nt) {
        float e = __expf(sc[i] - m);
        sc[i] = e;
        s += e;
    }
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_xor_sync(0xffffffffu, s, off);
    if (lane == 0) red[warp] = s;
    __syncthreads();
    if (warp == 0) {
        s = (tid < nwarp) ? red[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            s += __shfl_xor_sync(0xffffffffu, s, off);
        if (tid == 0) red[0] = 1.0f / s;
    }
    __syncthreads();
    const float inv = red[0];
    __syncthreads();

    for (int i = tid; i < capacity; i += nt) sc[i] *= inv;
    __syncthreads();

    // Weighted sum.
    for (int d = tid; d < D; d += nt) {
        float acc = 0.0f;
        for (int i = 0; i < capacity; ++i) {
            if (i >= memory_length) break;
            acc += sc[i] * mem_values[(size_t)i * D + d];
        }
        out[d] = acc;
    }
}

void hca_global_attn_decode(const float* q,
                            const float* mem_keys,
                            const float* mem_values,
                            const int32_t* mem_positions,
                            int32_t current_position,
                            int32_t memory_length,
                            float* out,
                            int D, int capacity,
                            cudaStream_t s) {
    if (D <= 0 || capacity <= 0) return;
    size_t shmem = sizeof(float) * capacity;
    int threads = 256;
    hca_global_attn_decode_kernel<<<1, threads, shmem, s>>>(
        q, mem_keys, mem_values, mem_positions,
        current_position, memory_length, out, D, capacity);
}

// Masked top-k variant used when dense_global_attention is disabled.  The
// selection is intentionally simple: decode has one query, so one thread can
// select at most a few dozen entries while the block computes the D-wide
// weighted sum in parallel.
__global__ void hca_global_topk_attn_decode_kernel(
    const float* __restrict__ q,
    const float* __restrict__ mem_keys,
    const float* __restrict__ mem_values,
    const int32_t* __restrict__ mem_positions,
    int32_t current_position,
    int32_t memory_length,
    float* __restrict__ out,
    int D, int capacity, int top_k)
{
    extern __shared__ float smem[];
    float* scores = smem;
    int32_t* indices = reinterpret_cast<int32_t*>(scores + capacity);
    float* logits = reinterpret_cast<float*>(indices + top_k);
    float* weights = logits + top_k;
    const int tid = threadIdx.x;

    for (int i = tid; i < capacity; i += blockDim.x) {
        float score = -FLT_MAX;
        if (i < memory_length) {
            const int pos = mem_positions[i];
            if (pos >= 0 && pos <= current_position) {
                const float* key = mem_keys + (size_t)i * D;
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += q[d] * key[d];
                score = dot * rsqrtf((float)D);
            }
        }
        scores[i] = score;
    }
    __syncthreads();

    if (tid == 0) {
        for (int t = 0; t < top_k; ++t) {
            int best = -1;
            float best_score = -FLT_MAX;
            for (int i = 0; i < capacity; ++i) {
                if (scores[i] > best_score) {
                    best_score = scores[i];
                    best = i;
                }
            }
            if (best < 0 || best_score <= -FLT_MAX * 0.5f) {
                indices[t] = -1;
                logits[t] = -FLT_MAX;
            } else {
                indices[t] = best;
                logits[t] = best_score;
                scores[best] = -FLT_MAX;
            }
        }
        float max_score = -FLT_MAX;
        for (int t = 0; t < top_k; ++t)
            max_score = fmaxf(max_score, logits[t]);
        float sum = 0.0f;
        for (int t = 0; t < top_k; ++t) {
            weights[t] = indices[t] < 0 ? 0.0f : __expf(logits[t] - max_score);
            sum += weights[t];
        }
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int t = 0; t < top_k; ++t) weights[t] *= inv;
    }
    __syncthreads();

    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < top_k; ++t) {
            const int idx = indices[t];
            if (idx >= 0) acc += weights[t] * mem_values[(size_t)idx * D + d];
        }
        out[d] = acc;
    }
}

void hca_global_topk_attn_decode(const float* q,
                                 const float* mem_keys,
                                 const float* mem_values,
                                 const int32_t* mem_positions,
                                 int32_t current_position,
                                 int32_t memory_length,
                                 float* out,
                                 int D, int capacity, int top_k,
                                 cudaStream_t s) {
    if (D <= 0 || capacity <= 0 || top_k <= 0) return;
    top_k = std::min(top_k, capacity);
    size_t shmem = sizeof(float) * capacity
                 + sizeof(int32_t) * top_k
                 + sizeof(float) * top_k * 2;
    hca_global_topk_attn_decode_kernel<<<1, 256, shmem, s>>>(
        q, mem_keys, mem_values, mem_positions,
        current_position, memory_length, out, D, capacity, top_k);
}

}  // namespace llmy
