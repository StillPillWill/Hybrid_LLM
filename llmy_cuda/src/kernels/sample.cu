#include "kernels.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>
#include <vector>
#include <stdexcept>

namespace llmy {

// ---------------------------------------------------------------------------
// Top-k / top-p sampling.
//
// Strategy: do most of the work on device (rep penalty, top-k), then
// finish on the host (top-p truncation, CDF, sample). The transfer of the
// top-k values/indices is tiny (<= 64 entries).
// ---------------------------------------------------------------------------

__global__ void apply_rep_penalty_kernel(float* __restrict__ logits,
                                         const int32_t* __restrict__ recent,
                                         int n_recent,
                                         float penalty,
                                         int V) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;
    for (int r = 0; r < n_recent; ++r) {
        if (recent[r] == v) {
            float l = logits[v];
            logits[v] = (l > 0.0f) ? (l / penalty) : (l * penalty);
            return;
        }
    }
}

// Find the top-k logits via iterated argmax. Writes k indices and k values.
__global__ void find_topk_kernel(const float* __restrict__ logits, int V, int k,
                                 int32_t* __restrict__ idx_out,
                                 float*   __restrict__ val_out)
{
    // One block. Shared memory: V floats + V int32 for the working copy.
    // V can be 248320, so we can't fit it in shared memory. Instead we
    // read from global each pass and mask off chosen entries via a
    // shared bitmap. 248320 bits = 31 KB — that fits.
    extern __shared__ uint8_t mask[];  // ceil(V / 8) bytes
    const int mask_bytes = (V + 7) / 8;

    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (nt + 31) >> 5;

    for (int i = tid; i < mask_bytes; i += nt) mask[i] = 0;

    __shared__ float red_f[32];
    __shared__ int   red_i[32];

    for (int t = 0; t < k; ++t) {
        float my_max = -FLT_MAX;
        int   my_arg = -1;
        for (int v = tid; v < V; v += nt) {
            uint8_t bit = (mask[v >> 3] >> (v & 7)) & 1;
            if (bit) continue;
            float l = logits[v];
            if (l > my_max) { my_max = l; my_arg = v; }
        }
        // Reduce.
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
        int chosen = red_i[0];
        if (tid == 0) {
            idx_out[t] = chosen;
            val_out[t] = red_f[0];
        }
        __syncthreads();
        // Mark chosen.
        for (int i = tid; i < mask_bytes; i += nt) {
            int v_start = i * 8;
            if (chosen >= v_start && chosen < v_start + 8) {
                mask[i] |= (1 << (chosen - v_start));
            }
        }
        __syncthreads();
    }
}

// Host-side helper for CDF sampling.
static uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

// ---------------------------------------------------------------------------
// sample_topk_topp — public entry point
// ---------------------------------------------------------------------------

int32_t sample_topk_topp(const float* logits, int V,
                         float temperature, int top_k, float top_p,
                         float repetition_penalty,
                         const int32_t* recent_tokens, int n_recent,
                         uint64_t* rng_state, cudaStream_t s) {
    if (V <= 0) throw std::runtime_error("empty vocab");

    // Bounds-check top_k.
    if (top_k <= 0 || top_k > V) top_k = V;
    if (top_k > 64) top_k = 64;   // cap to keep our scratch tiny

    // --- Device buffers. We allocate scratch on device and copy the small
    // results back. These are cached on first call so subsequent tokens
    // don't re-malloc.
    static float*        d_logits = nullptr;
    static int32_t*      d_idx    = nullptr;
    static float*        d_vals   = nullptr;
    static int32_t*      d_recent = nullptr;
    static int           d_recent_cap = 0;
    static int           d_vocab_cap  = 0;

    if (d_vocab_cap < V) {
        if (d_logits) cudaFree(d_logits);
        if (d_idx)    cudaFree(d_idx);
        if (d_vals)   cudaFree(d_vals);
        cudaMalloc(&d_logits, V * sizeof(float));
        cudaMalloc(&d_idx, 64 * sizeof(int32_t));
        cudaMalloc(&d_vals, 64 * sizeof(float));
        d_vocab_cap = V;
    }
    if (d_recent_cap < n_recent) {
        if (d_recent) cudaFree(d_recent);
        cudaMalloc(&d_recent, (n_recent > 0 ? n_recent : 1) * sizeof(int32_t));
        d_recent_cap = n_recent;
    }

    // Copy logits to device.
    cudaMemcpyAsync(d_logits, logits, V * sizeof(float),
                    cudaMemcpyHostToDevice, s);

    // Copy recent tokens.
    if (n_recent > 0) {
        cudaMemcpyAsync(d_recent, recent_tokens, n_recent * sizeof(int32_t),
                        cudaMemcpyHostToDevice, s);
    }

    // 1. Repetition penalty (in place on device).
    if (repetition_penalty != 1.0f && n_recent > 0) {
        int threads = 256;
        int blocks = (V + threads - 1) / threads;
        apply_rep_penalty_kernel<<<blocks, threads, 0, s>>>(
            d_logits, d_recent, n_recent, repetition_penalty, V);
    }

    // 2. Temperature scaling (in place).
    if (temperature > 0.0f && temperature != 1.0f) {
        // Small vectorized kernel; for simplicity, do it via a memcpy
        // path or a separate kernel. We'll just do it inside the topk
        // kernel by scaling on the fly. Skipping here.
        // (The Python sampler divides logits by temperature; our topk
        //  uses raw logits so top-k selection is unaffected. We scale
        //  at the end when computing probabilities.)
    }

    // 3. Find top-k.
    {
        int threads = 256;
        size_t mask_bytes = (V + 7) / 8;
        find_topk_kernel<<<1, threads, mask_bytes, s>>>(
            d_logits, V, top_k, d_idx, d_vals);
    }

    // 4. Copy top-k values and indices to host.
    int32_t h_idx[64];
    float   h_vals[64];
    cudaMemcpyAsync(h_idx, d_idx, top_k * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(h_vals, d_vals, top_k * sizeof(float),
                    cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    // 5. Host-side top-p truncation + softmax + sample.
    //    Values are already sorted descending by construction of the
    //    iterated argmax.
    std::vector<float> probs(top_k);
    float pmax = h_vals[0];
    float sum = 0.0f;
    for (int i = 0; i < top_k; ++i) {
        probs[i] = std::exp((h_vals[i] - pmax) / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < top_k; ++i) probs[i] /= sum;

    // Top-p truncation.
    int cutoff = top_k;
    if (top_p > 0.0f && top_p < 1.0f) {
        float cdf = 0.0f;
        for (int i = 0; i < top_k; ++i) {
            cdf += probs[i];
            if (cdf >= top_p) { cutoff = i + 1; break; }
        }
    }

    // Renormalize.
    float renorm = 0.0f;
    for (int i = 0; i < cutoff; ++i) renorm += probs[i];
    for (int i = 0; i < cutoff; ++i) probs[i] /= renorm;

    // Sample via CDF.
    uint64_t r = xorshift64(rng_state);
    double u = (double)(r >> 11) / (double)(1ULL << 53);
    float acc = 0.0f;
    int32_t chosen = h_idx[cutoff - 1];
    for (int i = 0; i < cutoff; ++i) {
        acc += probs[i];
        if (u < acc) { chosen = h_idx[i]; break; }
    }
    return chosen;
}

}  // namespace llmy
