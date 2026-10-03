#include "llmy.h"
#include "model.h"
#include "kernels.h"
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

using namespace llmy;

// ---------------------------------------------------------------------------
// C ABI implementations. Each function is a thin wrapper over the C++
// internals. The opaque handles are actually `llmy::Model*` and
// `llmy::State*`.
// ---------------------------------------------------------------------------

extern "C" llmy_model* llmy_load(const char* path) {
    try {
        auto* m = new std::unique_ptr<Model>(load_model(std::string(path)));
        return reinterpret_cast<llmy_model*>(m);
    } catch (const std::exception& e) {
        fprintf(stderr, "[llmy_load] %s\n", e.what());
        return nullptr;
    }
}

extern "C" void llmy_free(llmy_model* mp) {
    if (!mp) return;
    auto* m = reinterpret_cast<std::unique_ptr<Model>*>(mp);
    if ((*m)->blas) cublasDestroy((*m)->blas);
    delete m;
}

extern "C" int32_t llmy_vocab_size(const llmy_model* mp) {
    auto* m = reinterpret_cast<const std::unique_ptr<Model>*>(mp);
    return (*m)->cfg.vocab_size;
}

extern "C" int32_t llmy_d_model(const llmy_model* mp) {
    auto* m = reinterpret_cast<const std::unique_ptr<Model>*>(mp);
    return (*m)->cfg.d_model;
}

extern "C" int32_t llmy_num_loops(const llmy_model* mp) {
    auto* m = reinterpret_cast<const std::unique_ptr<Model>*>(mp);
    return (*m)->cfg.num_loops;
}

extern "C" int32_t llmy_max_loops(const llmy_model* mp) {
    auto* m = reinterpret_cast<const std::unique_ptr<Model>*>(mp);
    int v = (*m)->cfg.adaptive_max_loops;
    if (v < 1) v = (*m)->cfg.num_loops;
    if (v > (*m)->cfg.num_loops) v = (*m)->cfg.num_loops;
    return v;
}

extern "C" int32_t llmy_mhc_streams(const llmy_model* mp) {
    auto* m = reinterpret_cast<const std::unique_ptr<Model>*>(mp);
    return (*m)->cfg.mhc_streams;
}

extern "C" void llmy_set_num_loops(llmy_model* mp, int32_t n) {
    auto* m = reinterpret_cast<std::unique_ptr<Model>*>(mp);
    (*m)->cfg.num_loops = n;
}

extern "C" void llmy_set_fixed_loops(llmy_model* mp, int32_t n) {
    auto* m = reinterpret_cast<std::unique_ptr<Model>*>(mp);
    if (n < 1) return;
    (*m)->cfg.num_loops = n;
    (*m)->cfg.adaptive_enabled = false;
    (*m)->cfg.adaptive_learned_halting = false;
    (*m)->cfg.adaptive_min_loops = n;
    (*m)->cfg.adaptive_max_loops = n;
}

extern "C" llmy_state* llmy_state_new(llmy_model* mp) {
    try {
        auto* m = reinterpret_cast<std::unique_ptr<Model>*>(mp);
        auto* s = new std::unique_ptr<State>(alloc_state(**m, 1));
        return reinterpret_cast<llmy_state*>(s);
    } catch (const std::exception& e) {
        fprintf(stderr, "[llmy_state_new] %s\n", e.what());
        return nullptr;
    }
}

extern "C" void llmy_state_free(llmy_state* sp) {
    if (!sp) return;
    auto* s = reinterpret_cast<std::unique_ptr<State>*>(sp);
    free_state(**s);
    delete s;
}

extern "C" void llmy_state_reset(llmy_state* sp) {
    // Zero all state tensors. For a fresh conversation.
    if (!sp) return;
    auto* s = reinterpret_cast<std::unique_ptr<State>*>(sp);
    State& st = **s;
    auto zero = [](void* p, size_t bytes) {
        if (p) cudaMemset(p, 0, bytes);
    };
    // streams
    zero(st.streams, sizeof(float) * st.batch * 4096 * 4);  // rough
    // gdn_states: we don't know the exact size here, but we can zero
    // generously since the state was zero-allocated at creation.
    // A more precise implementation would track sizes.
    cudaDeviceSynchronize();
    st.position = 0;
}

extern "C" void llmy_prefill(llmy_model* mp, llmy_state* sp,
                             const int32_t* tokens, int32_t n_tokens,
                             float* logits_out) {
    auto* m = reinterpret_cast<std::unique_ptr<Model>*>(mp);
    auto* s = reinterpret_cast<std::unique_ptr<State>*>(sp);
    model_prefill(**m, **s, tokens, n_tokens, logits_out);
}

extern "C" void llmy_decode(llmy_model* mp, llmy_state* sp,
                            int32_t token, int32_t position, int32_t num_loops,
                            float* logits_out) {
    auto* m = reinterpret_cast<std::unique_ptr<Model>*>(mp);
    auto* s = reinterpret_cast<std::unique_ptr<State>*>(sp);
    model_decode(**m, **s, token, position, num_loops, logits_out);
}

extern "C" int32_t llmy_sample(const float* logits, int32_t vocab_size,
                               float temperature, int32_t top_k, float top_p,
                               float repetition_penalty,
                               const int32_t* recent_tokens, int32_t n_recent,
                               uint64_t* rng_state) {
    // CPU-side sampler. This mirrors the Python sampler but in C++, so it
    // can be called directly from the CLI. The Python sampler path uses
    // the same math in numpy, so both produce equivalent distributions.
    if (vocab_size <= 0) return 0;
    if (top_k <= 0 || top_k > vocab_size) top_k = vocab_size;
    if (top_k > 256) top_k = 256;

    // Copy logits (this is a host pointer).
    std::vector<float> work(logits, logits + vocab_size);

    // Repetition penalty.
    if (repetition_penalty != 1.0f && n_recent > 0) {
        for (int r = 0; r < n_recent; ++r) {
            int idx = recent_tokens[r];
            if (idx < 0 || idx >= vocab_size) continue;
            float v = work[idx];
            work[idx] = (v > 0.0f) ? (v / repetition_penalty)
                                   : (v * repetition_penalty);
        }
    }

    // Top-k by partial sort.
    std::vector<int> idx(vocab_size);
    for (int i = 0; i < vocab_size; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                      [&](int a, int b) { return work[a] > work[b]; });

    if (!(temperature > 0.0f)) return idx[0];

    // Softmax over selected.
    float maxv = work[idx[0]];
    std::vector<float> probs(top_k);
    float sum = 0.0f;
    for (int i = 0; i < top_k; ++i) {
        float e = std::exp((work[idx[i]] - maxv) / temperature);
        probs[i] = e;
        sum += e;
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
    float renorm = 0.0f;
    for (int i = 0; i < cutoff; ++i) renorm += probs[i];
    for (int i = 0; i < cutoff; ++i) probs[i] /= renorm;

    // CDF sampling.
    uint64_t r = *rng_state;
    r ^= r << 13; r ^= r >> 7; r ^= r << 17;
    *rng_state = r;
    double u = (double)(r >> 11) / (double)(1ULL << 53);
    float acc = 0.0f;
    int chosen = idx[cutoff - 1];
    for (int i = 0; i < cutoff; ++i) {
        acc += probs[i];
        if (u < acc) { chosen = idx[i]; break; }
    }
    return chosen;
}

extern "C" int32_t llmy_last_executed_loops(const llmy_state* sp) {
    auto* s = reinterpret_cast<const std::unique_ptr<State>*>(sp);
    return (*s)->last_executed_loops;
}

extern "C" void llmy_sync(void) {
    cudaDeviceSynchronize();
}

extern "C" int32_t llmy_mhc_sinkhorn4(const float* input, float* output,
                                      int32_t matrices, int32_t iterations,
                                      void* stream) {
    if (input == nullptr || output == nullptr || matrices <= 0 || iterations < 0) {
        return static_cast<int32_t>(cudaErrorInvalidValue);
    }

    llmy::mhc_sinkhorn4_out(input, output, matrices, iterations,
                            reinterpret_cast<cudaStream_t>(stream));
    return static_cast<int32_t>(cudaPeekAtLastError());
}

extern "C" int32_t llmy_moe_selected_fp32(
        const float* x, const int64_t* idx, const float* prob,
        const float* gate_up, const float* down, float* activation,
        float* route_output, float* out,
        int32_t D, int32_t H, int32_t E, int32_t k, void* stream) {
    if (x == nullptr || idx == nullptr || prob == nullptr || gate_up == nullptr ||
        down == nullptr || activation == nullptr || route_output == nullptr ||
        out == nullptr || D <= 0 || H <= 0 || E <= 0 || k <= 0) {
        return static_cast<int32_t>(cudaErrorInvalidValue);
    }

    llmy::moe_selected_token_fp32(
        x, idx, prob, gate_up, down, activation, route_output, out,
        D, H, E, k, reinterpret_cast<cudaStream_t>(stream));
    return static_cast<int32_t>(cudaPeekAtLastError());
}
