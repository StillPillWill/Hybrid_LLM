#include "kernels.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <algorithm>

namespace llmy {

// ---------------------------------------------------------------------------
// csa_local_attn_decode
//
// One block, all heads processed cooperatively. Shared memory holds the
// per-head scores and the head-averaged weight vector.
//
// Layout: scores[h * W + w] holds the logit for head h, key position w.
// The block is sized to W * H floats in shared memory.
// ---------------------------------------------------------------------------

__global__ void csa_local_attn_decode_kernel(
    const float* __restrict__ q,          // [H, Dh]
    const float* __restrict__ k_cache,    // [H, W, Dh]
    const float* __restrict__ v_cache,    // [W, D]
    const uint8_t* __restrict__ valid,    // [W] or nullptr
    float* __restrict__ out,              // [D]
    int H, int W, int Dh, int D)
{
    extern __shared__ float smem[];
    float* scores = smem;                 // [H * W]
    float* avg_w  = smem + H * W;         // [W]

    const int tid = threadIdx.x;
    const int nt  = blockDim.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (nt + 31) >> 5;

    __shared__ float red[32];

    const float scale = rsqrtf((float)Dh);

    // ---------------------------------------------------------------------
    // Pass 1: per-head raw scores.
    // ---------------------------------------------------------------------
    for (int h = 0; h < H; ++h) {
        const float* qh = q + h * Dh;
        float* sc = scores + h * W;
        for (int w = tid; w < W; w += nt) {
            if (valid != nullptr && !valid[w]) {
                sc[w] = -FLT_MAX;
                continue;
            }
            // The rolled decode cache is stored as [W, H*Dh] (time-major),
            // while the attention math is indexed as [H, W, Dh].  Do not
            // treat those layouts as interchangeable: that silently mixes
            // heads and time positions once H > 1.
            const float* kh = k_cache + ((size_t)w * H + h) * Dh;
            float dot = 0.0f;
            for (int d = 0; d < Dh; ++d) dot += qh[d] * kh[d];
            sc[w] = dot * scale;
        }
    }
    __syncthreads();

    // ---------------------------------------------------------------------
    // Pass 2: softmax per head.
    // ---------------------------------------------------------------------
    for (int h = 0; h < H; ++h) {
        float* sc = scores + h * W;

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
    }

    // ---------------------------------------------------------------------
    // Pass 3: average across heads.
    // ---------------------------------------------------------------------
    for (int w = tid; w < W; w += nt) {
        float acc = 0.0f;
        for (int h = 0; h < H; ++h) acc += scores[h * W + w];
        avg_w[w] = acc / (float)H;
    }
    __syncthreads();

    // ---------------------------------------------------------------------
    // Pass 4: weighted sum with V.
    // ---------------------------------------------------------------------
    for (int d = tid; d < D; d += nt) {
        float acc = 0.0f;
        for (int w = 0; w < W; ++w) {
            acc += avg_w[w] * v_cache[(size_t)w * D + d];
        }
        out[d] = acc;
    }
}

void csa_local_attn_decode(const float* q,
                           const float* k_cache,
                           const float* v_cache,
                           const uint8_t* valid,
                           float* out,
                           int H, int W, int Dh, int D,
                           cudaStream_t s) {
    if (H <= 0 || W <= 0 || Dh <= 0 || D <= 0) return;
    size_t shmem = sizeof(float) * (H * W + W);
    int threads = 256;
    csa_local_attn_decode_kernel<<<1, threads, shmem, s>>>(
        q, k_cache, v_cache, valid, out, H, W, Dh, D);
}

// ---------------------------------------------------------------------------
// csa_compress_step
//
// Updates the pending bucket, computes the mean when the block is complete,
// projects through key_w / value_w, and writes one row into the memory at
// slot (mem_length % capacity). Rows from incomplete blocks are written
// with position = -1 (invisible to attention).
//
// One block. Threads distribute across D and Dk + Dv.
// ---------------------------------------------------------------------------

__global__ void csa_compress_step_kernel(
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
    int D, int Dk, int Dv, int capacity,
    int32_t position, int32_t segment, int ratio)
{
    extern __shared__ float smem[];
    float* mean = smem;                 // [D]

    const int tid = threadIdx.x;
    const int nt = blockDim.x;

    __shared__ int s_seg_changed;
    __shared__ int s_emitted;
    __shared__ int s_slot;

    // Segment change detection.
    if (tid == 0) {
        s_seg_changed = (pending_seg[0] != segment) ? 1 : 0;
        if (s_seg_changed) pending_seg[0] = segment;
    }
    __syncthreads();

    // Accumulate into pending_sum; reset if the segment just changed.
    for (int d = tid; d < D; d += nt) {
        float prev = s_seg_changed ? 0.0f : pending_sum[d];
        pending_sum[d] = prev + hidden[d];
    }
    __syncthreads();

    // Update count, decide emit.
    if (tid == 0) {
        int prev_count = s_seg_changed ? 0 : pending_count[0];
        pending_count[0] = prev_count + 1;
        s_emitted = (pending_count[0] >= ratio) ? 1 : 0;
        s_slot = mem_length % capacity;
    }
    __syncthreads();

    // Compute mean (always; if we don't emit the projection is wasted, but
    // it's tiny compared to the rest of the block's work).
    const float inv_count = 1.0f / fmaxf((float)pending_count[0], 1.0f);
    for (int d = tid; d < D; d += nt) {
        mean[d] = pending_sum[d] * inv_count;
    }
    __syncthreads();

    // Project to key and value. One thread per output element.
    const int total_out = Dk + Dv;
    for (int i = tid; i < total_out; i += nt) {
        const float* W;
        int out_idx;
        if (i < Dk) { W = key_w; out_idx = i; }
        else        { W = value_w; out_idx = i - Dk; }
        const float* wrow = W + (size_t)out_idx * D;
        float acc = 0.0f;
        for (int d = 0; d < D; ++d) acc += wrow[d] * mean[d];
        if (i < Dk) {
            mem_keys[(size_t)s_slot * Dk + i] = acc;
        } else {
            mem_values[(size_t)s_slot * Dv + (i - Dk)] = acc;
        }
    }
    __syncthreads();

    // Write position / segment and advance length on emit.
    if (tid == 0) {
        if (s_emitted) {
            mem_positions[s_slot] = position;
            mem_segments[s_slot] = segment;
            pending_count[0] = 0;
            // Zero the pending sum so the next token starts fresh.
            for (int d = 0; d < D; ++d) pending_sum[d] = 0.0f;
        } else {
            mem_positions[s_slot] = -1;
            mem_segments[s_slot] = -1;
        }
    }
}

void csa_compress_step(const float* hidden,
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
                       int D, int Dk, int Dv, int capacity,
                       int32_t position, int32_t segment, int ratio,
                       cudaStream_t s) {
    if (D <= 0 || Dk <= 0 || Dv <= 0 || capacity <= 0) return;
    // Threads need to cover max(D, Dk + Dv); use the larger.
    int threads = std::max(D, Dk + Dv);
    if (threads > 1024) threads = 1024;
    if (threads < 128) threads = 128;
    size_t shmem = sizeof(float) * D;
    csa_compress_step_kernel<<<1, threads, shmem, s>>>(
        hidden, pending_sum, pending_count, pending_seg,
        key_w, value_w,
        mem_keys, mem_values, mem_positions, mem_segments, mem_length,
        D, Dk, Dv, capacity, position, segment, ratio);
}

// ---------------------------------------------------------------------------
// csa_remote_attn_decode
//
// One-token remote attention. Computes q @ K^T for all visible entries,
// top-k per head, softmax, weighted sum of V, average across heads.
// ---------------------------------------------------------------------------

__global__ void csa_remote_attn_decode_kernel(
    const float* __restrict__ q,             // [H, Dh]
    const float* __restrict__ mem_keys,      // [capacity, H*Dh]
    const float* __restrict__ mem_values,    // [capacity, D]
    const int32_t* __restrict__ mem_positions, // [capacity]
    int32_t current_position,
    int32_t memory_length,
    float* __restrict__ out,                 // [D]
    int H, int Dh, int D, int capacity, int top_k)
{
    extern __shared__ float smem[];
    // Layout: scores [H * capacity], topk_idx [H * top_k],
    // topk_logits [H * top_k], avg_w [top_k].
    float* scores = smem;
    int32_t* topk_idx = reinterpret_cast<int32_t*>(smem + H * capacity);
    size_t idx_floats = (sizeof(int32_t) * H * top_k + sizeof(float) - 1)
                        / sizeof(float);
    float* topk_logits = smem + H * capacity + idx_floats;
    float* avg_w = topk_logits + H * top_k;

    const int tid = threadIdx.x;
    const int nt = blockDim.x;

    const int n_valid = memory_length;
    const float scale = rsqrtf((float)Dh);

    // ---------------------------------------------------------------------
    // Pass 1: scores per head over all visible memory rows.
    // ---------------------------------------------------------------------
    for (int h = 0; h < H; ++h) {
        const float* qh = q + h * Dh;
        float* sc = scores + h * capacity;
        for (int m = tid; m < n_valid; m += nt) {
            int pos = mem_positions[m];
            if (pos < 0 || pos > current_position) {
                sc[m] = -FLT_MAX;
                continue;
            }
            // mem_keys is [capacity, H*Dh]; row m has all heads' keys.
            const float* km = mem_keys + (size_t)m * (H * Dh) + h * Dh;
            float dot = 0.0f;
            for (int d = 0; d < Dh; ++d) dot += qh[d] * km[d];
            sc[m] = dot * scale;
        }
        // Remaining rows are inactive.
        for (int m = n_valid + tid; m < capacity; m += nt) sc[m] = -FLT_MAX;
    }
    __syncthreads();

    // ---------------------------------------------------------------------
    // Pass 2: per-head top-k selection. Iterated argmax.
    // ---------------------------------------------------------------------
    for (int h = 0; h < H; ++h) {
        float* sc = scores + h * capacity;
        int32_t* ik = topk_idx + h * top_k;

        for (int t = 0; t < top_k; ++t) {
            // Find max over remaining scores.
            float my_max = -FLT_MAX;
            int   my_arg = -1;
            for (int m = tid; m < capacity; m += nt) {
                if (sc[m] > my_max) { my_max = sc[m]; my_arg = m; }
            }
            // Block-reduce max.
            for (int off = 16; off > 0; off >>= 1) {
                float ov = __shfl_down_sync(0xffffffffu, my_max, off);
                int   oi = __shfl_down_sync(0xffffffffu, my_arg, off);
                if (ov > my_max) { my_max = ov; my_arg = oi; }
            }
            __shared__ float smax[32];
            __shared__ int   sarg[32];
            int lane = tid & 31, warp = tid >> 5;
            if (lane == 0) { smax[warp] = my_max; sarg[warp] = my_arg; }
            __syncthreads();
            if (warp == 0) {
                int nw = (nt + 31) >> 5;
                float v = (tid < nw) ? smax[tid] : -FLT_MAX;
                int   a = (tid < nw) ? sarg[tid] : -1;
                for (int off = 16; off > 0; off >>= 1) {
                    float ov = __shfl_down_sync(0xffffffffu, v, off);
                    int   oi = __shfl_down_sync(0xffffffffu, a, off);
                    if (ov > v) { v = ov; a = oi; }
                }
                if (tid == 0) { smax[0] = v; sarg[0] = a; }
            }
            __syncthreads();
            int chosen = sarg[0];
            if (tid == 0) {
                ik[t] = chosen;
                topk_logits[h * top_k + t] = smax[0];
            }
            __syncthreads();
            // Mark chosen as -inf.
            for (int m = tid; m < capacity; m += nt) {
                if (m == chosen) sc[m] = -FLT_MAX;
            }
            __syncthreads();
        }
    }

    // ---------------------------------------------------------------------
    // Pass 4: weighted sum of V over the head-averaged weights.
    //
    // We need the selected indices. For head averaging to work cleanly,
    // each head's top-k is different; but the reference implementation
    // averages across heads AFTER the weighted sum. To match exactly we
    // need to redo this per head and then average the outputs.
    // ---------------------------------------------------------------------
    for (int d = tid; d < D; d += nt) out[d] = 0.0f;
    __syncthreads();

    for (int h = 0; h < H; ++h) {
        int32_t* ik = topk_idx + h * top_k;
        // Recompute softmax for this head only (cheap; k is small).
        // Actually we already have the weights in avg_w but they're averaged.
        // Redo a per-head pass here.
        __syncthreads();

        // Gather the preserved original logits.
        float* tk = avg_w;
        for (int t = tid; t < top_k; t += nt)
            tk[t] = topk_logits[h * top_k + t];
        __syncthreads();

        // Max
        float m = -FLT_MAX;
        for (int t = tid; t < top_k; t += nt) m = fmaxf(m, tk[t]);
        for (int off = 16; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        int lane = tid & 31, warp = tid >> 5;
        __shared__ float red[32];
        if (lane == 0) red[warp] = m;
        __syncthreads();
        if (warp == 0) {
            int nw = (nt + 31) >> 5;
            m = (tid < nw) ? red[tid] : -FLT_MAX;
            for (int off = 16; off > 0; off >>= 1)
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
            if (tid == 0) red[0] = m;
        }
        __syncthreads();
        m = red[0];
        __syncthreads();

        float s = 0.0f;
        for (int t = tid; t < top_k; t += nt) {
            float e = __expf(tk[t] - m);
            tk[t] = e;
            s += e;
        }
        for (int off = 16; off > 0; off >>= 1)
            s += __shfl_xor_sync(0xffffffffu, s, off);
        if (lane == 0) red[warp] = s;
        __syncthreads();
        if (warp == 0) {
            int nw = (nt + 31) >> 5;
            s = (tid < nw) ? red[tid] : 0.0f;
            for (int off = 16; off > 0; off >>= 1)
                s += __shfl_xor_sync(0xffffffffu, s, off);
            if (tid == 0) red[0] = 1.0f / s;
        }
        __syncthreads();
        const float inv = red[0];
        __syncthreads();

        // Accumulate head h's contribution to out[d], weighted by 1/H.
        for (int d = tid; d < D; d += nt) {
            float acc = 0.0f;
            for (int t = 0; t < top_k; ++t) {
                int m = ik[t];
                if (m < 0) continue;
                float w = tk[t] * inv / (float)H;
                acc += w * mem_values[(size_t)m * D + d];
            }
            out[d] += acc;
        }
        __syncthreads();
    }
}

void csa_remote_attn_decode(const float* q,
                            const float* mem_keys,
                            const float* mem_values,
                            const int32_t* mem_positions,
                            int32_t current_position,
                            int32_t memory_length,
                            float* out,
                            int H, int Dh, int D, int capacity, int top_k,
                            cudaStream_t s) {
    if (H <= 0 || Dh <= 0 || D <= 0 || capacity <= 0 || top_k <= 0) return;

    // Shared memory: scores [H * capacity] + topk_idx [H * top_k] + avg_w [top_k]
    // Round topk_idx storage to float boundaries.
    size_t scores_bytes = sizeof(float) * H * capacity;
    size_t idx_bytes = sizeof(int32_t) * H * top_k;
    size_t idx_padded = ((idx_bytes + sizeof(float) - 1) / sizeof(float)) * sizeof(float);
    size_t top_logits_bytes = sizeof(float) * H * top_k;
    size_t avg_bytes = sizeof(float) * top_k;
    size_t shmem = scores_bytes + idx_padded + top_logits_bytes + avg_bytes;

    int threads = 256;
    csa_remote_attn_decode_kernel<<<1, threads, shmem, s>>>(
        q, mem_keys, mem_values, mem_positions, current_position,
        memory_length, out, H, Dh, D, capacity, top_k);
}

}  // namespace llmy
