#ifndef LLMY_H
#define LLMY_H

#include <stdint.h>
#include <stddef.h>

// ---------------------------------------------------------------------------
// C ABI. This is the only header the Python side sees. All functions are
// extern "C" and use only plain types so ctypes can bind them directly.
//
// Usage from Python:
//
//   lib = ctypes.CDLL("./libllmy.so")
//   lib.llmy_load.restype = ctypes.c_void_p
//   model = lib.llmy_load(b"model.llmy")
//   state = lib.llmy_state_new(model)
//   tokens = (ctypes.c_int32 * n)(*ids)
//   logits = (ctypes.c_float * vocab)()
//   lib.llmy_prefill(model, state, tokens, n, logits)
//   ...
// ---------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

typedef struct llmy_model llmy_model;
typedef struct llmy_state llmy_state;

// --- Model lifecycle -------------------------------------------------------
llmy_model* llmy_load(const char* path);
void        llmy_free(llmy_model* m);

// --- Model introspection ---------------------------------------------------
int32_t llmy_vocab_size(const llmy_model* m);
int32_t llmy_d_model(const llmy_model* m);
int32_t llmy_num_loops(const llmy_model* m);
int32_t llmy_max_loops(const llmy_model* m);
int32_t llmy_mhc_streams(const llmy_model* m);
void    llmy_set_num_loops(llmy_model* m, int32_t n);
void    llmy_set_fixed_loops(llmy_model* m, int32_t n);

// --- State lifecycle -------------------------------------------------------
llmy_state* llmy_state_new(llmy_model* m);
void        llmy_state_free(llmy_state* s);
void        llmy_state_reset(llmy_state* s);

// --- Forward passes --------------------------------------------------------
// logits_out must point to vocab_size floats. The pointer can live on either
// the host or the device; the engine copies as needed.
void llmy_prefill(llmy_model* m, llmy_state* s,
                  const int32_t* tokens, int32_t n_tokens,
                  float* logits_out);

// position is the absolute 0-based index of the new token.
// num_loops == 0 -> use the model default.
void llmy_decode(llmy_model* m, llmy_state* s,
                 int32_t token, int32_t position, int32_t num_loops,
                 float* logits_out);

// --- Sampling --------------------------------------------------------------
// All sampling math runs on the CPU. logits are a host-resident [vocab]
// buffer. Returns the sampled token id.
int32_t llmy_sample(const float* logits, int32_t vocab_size,
                    float temperature, int32_t top_k, float top_p,
                    float repetition_penalty,
                    const int32_t* recent_tokens, int32_t n_recent,
                    uint64_t* rng_state);

// --- Diagnostics -----------------------------------------------------------
int32_t llmy_last_executed_loops(const llmy_state* s);
void    llmy_sync(void);

// Run fused, out-of-place Sinkhorn on contiguous device FP32 [matrices, 4, 4]
// data. `stream` is the caller's CUDA stream handle (null means default stream).
int32_t llmy_mhc_sinkhorn4(const float* input, float* output,
                           int32_t matrices, int32_t iterations,
                           void* stream);

// Fused one-token sparse MoE inference, FP32 only. gate_up is [E, 2*H, D],
// down is [E, D, H], idx/prob are [k], and scratch buffers are [k, H] and
// [k, D]. All pointers reference contiguous CUDA allocations.
int32_t llmy_moe_selected_fp32(const float* x, const int64_t* idx,
                               const float* prob, const float* gate_up,
                               const float* down, float* activation,
                               float* route_output, float* out,
                               int32_t D, int32_t H, int32_t E, int32_t k,
                               void* stream);

#ifdef __cplusplus
}
#endif
#endif
