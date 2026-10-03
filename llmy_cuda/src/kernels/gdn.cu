#include "kernels.h"
#include <cuda_runtime.h>

namespace llmy {

// ---------------------------------------------------------------------------
// gdn_step — one token, updates state in place
//
// For each head h:
//   prediction[v] = sum_k k[k] * state[k, v]
//   delta[v]      = v[v] - prediction[v]
//   state[k, v]  += beta * k[k] * delta[v] - (1 - decay) * state[k, v]... 
//   Wait, actual formula is:
//     state_new[k, v] = decay * state[k, v] + beta * k[k] * delta[v]
//   out[v] = gate[v] * sum_k q[k] * state_new[k, v]
//
// The `state` array is row-major [K, V], so consecutive v's are contiguous.
// One block per head; threads split the V dimension. Each thread handles
// one v column and iterates over k.
// ---------------------------------------------------------------------------

__global__ void gdn_step_kernel(const float* __restrict__ q,   // [H, K]
                                const float* __restrict__ k,   // [H, K]
                                const float* __restrict__ v,   // [H, V]
                                const float* __restrict__ beta,// [H]
                                const float* __restrict__ gate,// [H, V]
                                const float* __restrict__ decay,// [H]
                                const uint8_t* __restrict__ reset_mask, // [H] or null
                                float* __restrict__ state,     // [H, K, V]
                                float* __restrict__ out,       // [H, V]
                                int H, int K, int V) {
    const int h = blockIdx.x;
    const int v_col = threadIdx.x;
    if (v_col >= V) return;

    if (reset_mask != nullptr && reset_mask[h]) {
        // Zero this head's state.
        for (int kk = 0; kk < K; ++kk) {
            state[(size_t)h * K * V + kk * V + v_col] = 0.0f;
        }
    }

    const float* qh = q + h * K;
    const float* kh = k + h * K;
    const float* vh = v + h * V;
    const float* gh = gate + h * V;
    float* Sh = state + (size_t)h * K * V;
    const float d = decay[h];
    const float b = beta[h];

    // prediction[v] = sum_k k[k] * state[k, v]
    float pred = 0.0f;
    for (int kk = 0; kk < K; ++kk) {
        pred += kh[kk] * Sh[(size_t)kk * V + v_col];
    }
    const float delta = vh[v_col] - pred;

    // state[k, v] = d * state[k, v] + b * k[k] * delta
    for (int kk = 0; kk < K; ++kk) {
        float* sp = Sh + (size_t)kk * V + v_col;
        *sp = d * (*sp) + b * kh[kk] * delta;
    }

    // out[v] = gate[v] * sum_k q[k] * state[k, v]
    float acc = 0.0f;
    for (int kk = 0; kk < K; ++kk) {
        acc += qh[kk] * Sh[(size_t)kk * V + v_col];
    }
    out[h * V + v_col] = gh[v_col] * acc;
}

void gdn_step(const float* q, const float* k, const float* v,
              const float* beta, const float* gate, const float* decay,
              float* state, float* out,
              int H, int K, int V, cudaStream_t s) {
    gdn_step_kernel<<<H, V, 0, s>>>(
        q, k, v, beta, gate, decay, nullptr, state, out, H, K, V);
}

void gdn_step_reset(const float* q, const float* k, const float* v,
                    const float* beta, const float* gate, const float* decay,
                    const uint8_t* reset_mask,
                    float* state, float* out,
                    int H, int K, int V, cudaStream_t s) {
    gdn_step_kernel<<<H, V, 0, s>>>(
        q, k, v, beta, gate, decay, reset_mask, state, out, H, K, V);
}

// ---------------------------------------------------------------------------
// gdn_conv_gate
//
// out[d] = sum_{kk=0}^{K-2} weight[d, kk] * history[kk, d]
//        + weight[d, K-1] * current[d]
// ---------------------------------------------------------------------------

__global__ void gdn_conv_gate_kernel(const float* __restrict__ history,
                                     const float* __restrict__ current,
                                     const float* __restrict__ weight,
                                     float* __restrict__ out,
                                     int D, int K) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    const float* w = weight + (size_t)d * K;
    float acc = 0.0f;
    for (int kk = 0; kk < K - 1; ++kk) {
        acc += w[kk] * history[(size_t)kk * D + d];
    }
    acc += w[K - 1] * current[d];
    out[d] = acc;
}

void gdn_conv_gate(const float* history, const float* current,
                   const float* weight, float* out,
                   int D, int K, cudaStream_t s) {
    const int threads = 256;
    const int blocks = (D + threads - 1) / threads;
    gdn_conv_gate_kernel<<<blocks, threads, 0, s>>>(
        history, current, weight, out, D, K);
}

// ---------------------------------------------------------------------------
// gdn_conv_roll — history = shift-left(history); history[-1] = current
// ---------------------------------------------------------------------------

__global__ void gdn_conv_roll_kernel(float* __restrict__ history,
                                     const float* __restrict__ current,
                                     int D, int K) {
    // history has (K-1) rows. Roll up by one, then write current into the
    // last row.
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    const int rows = K - 1;
    for (int r = 0; r < rows - 1; ++r) {
        history[(size_t)r * D + d] = history[(size_t)(r + 1) * D + d];
    }
    history[(size_t)(rows - 1) * D + d] = current[d];
}

void gdn_conv_roll(float* history, const float* current,
                   int D, int K, cudaStream_t s) {
    if (K <= 1) return;
    const int threads = 256;
    const int blocks = (D + threads - 1) / threads;
    gdn_conv_roll_kernel<<<blocks, threads, 0, s>>>(history, current, D, K);
}

}  // namespace llmy
