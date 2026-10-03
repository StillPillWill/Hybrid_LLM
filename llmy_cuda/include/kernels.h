#ifndef LLMY_KERNELS_H
#define LLMY_KERNELS_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>

// Every kernel the engine uses. All pointers are raw device pointers,
// all tensors are contiguous row-major, all work runs on the caller's
// stream. Kernels do not allocate or free; all scratch is provided by
// the caller.

namespace llmy {

// ---------------------------------------------------------------------------
// Elementwise + norm
// ---------------------------------------------------------------------------

// y = x * rsqrt(mean(x^2) + eps) * weight
// x, y shape [rows, cols]. One block per row.
void rmsnorm_fwd(const float* x, const float* w, float* y,
                 int rows, int cols, float eps, cudaStream_t s);

// y = silu(gate) * up
void silu_mul(const float* gate, const float* up, float* y,
              int64_t n, cudaStream_t s);

// y += x (elementwise add, in place)
void residual_add_(float* y, const float* x, int64_t n, cudaStream_t s);

// x *= sigmoid(gate)
void sigmoid_mul_(float* x, const float* gate, int64_t n, cudaStream_t s);

// y = x * scale (scalar scale read from device pointer)
void scale_(float* y, const float* x, const float* scale, int64_t n, cudaStream_t s);

// LayerNorm over last dim, with affine weight/bias.
void layernorm_fwd(float* x, const float* w, const float* b,
                   int rows, int cols, float eps, cudaStream_t s);

// ---------------------------------------------------------------------------
// mHC (multi-stream hyper-connection)
// ---------------------------------------------------------------------------
//
// compute_mappings takes a "flat" [B*T, S*D] tensor and produces per-token
// (pre, post, residual) maps.
//
//   flat[b, :] = RMSNorm(streams[b, :, :].reshape(S*D)) * mapping_norm_w
//   pre_raw    = alpha_pre * (pre_w @ flat[b, :]) + pre_bias
//   pre[b]     = sigmoid(pre_raw)
//   post_raw   = alpha_post * (post_w @ flat[b, :]) + post_bias
//   post[b]    = 2 * sigmoid(post_raw)
//   resid_raw  = alpha_residual * (residual_w @ flat[b, :]) + residual_bias
//   residual[b] = sinkhorn(resid_raw, iters)
//
// All weights are non-bias for the inner Linears (matching PyTorch's
// nn.Linear(..., bias=False)); pre_bias / post_bias / residual_bias are
// separate Parameters.
void mhc_compute_mappings(const float* streams,        // [B, S, D]
                          const float* mapping_norm_w, // [S*D]
                          const float* pre_w,          // [S, S*D]
                          const float* post_w,         // [S, S*D]
                          const float* residual_w,     // [S*S, S*D]
                          const float* pre_bias,       // [S]
                          const float* post_bias,      // [S]
                          const float* residual_bias,  // [S, S]
                          const float* alpha_pre,      // [1] device scalar
                          const float* alpha_post,     // [1]
                          const float* alpha_residual, // [1]
                          float* pre_out,              // [B, S]
                          float* post_out,             // [B, S]
                          float* residual_out,         // [B, S, S]
                          int B, int S, int D, int iters,
                          float* scratch_norm,         // [B, S*D] scratch
                          cudaStream_t s);

// In-place Sinkhorn on a [B, S, S] logits buffer.
void mhc_sinkhorn_(float* L, int B, int S, int iters, cudaStream_t s);

// Fused out-of-place Sinkhorn specialized for [matrices, 4, 4]. One warp
// keeps one matrix in registers and performs every normalization iteration.
void mhc_sinkhorn4_out(const float* input, float* output,
                      int matrices, int iters, cudaStream_t s);

// hidden[b, d] = sum_s pre[b, s] * streams[b, s, d]
// Optionally applies LayerNorm with (ln_w, ln_b) if non-null.
void mhc_read(const float* streams,       // [B, S, D]
              const float* pre,           // [B, S]
              const float* ln_w,          // [D] or nullptr
              const float* ln_b,          // [D] or nullptr
              float* hidden,              // [B, D]
              int B, int S, int D, cudaStream_t s);

// In-place update:
//   acc[b, s, d] = sum_ss residual[b, s, ss] * streams[b, ss, d]
//                + post[b, s] * update[b, d]
//   streams[b, s, d] += gating * stream_scale[s, d] * acc[b, s, d]
void mhc_write(float* streams,            // [B, S, D]
               const float* update,       // [B, D]
               const float* residual,     // [B, S, S]
               const float* post,         // [B, S]
               const float* stream_scale, // [S, D]
               const float* gating_factor,// [1] device scalar
               int B, int S, int D, cudaStream_t s);

// ---------------------------------------------------------------------------
// GDN (gated delta-rule recurrence)
// ---------------------------------------------------------------------------

// One token. Updates state in place, writes out.
//   state : [H, K, V] fp32, contiguous
//   q, k  : [H, K]
//   v     : [H, V]
//   beta  : [H]
//   gate  : [H, V]
//   decay : [H]
//   out   : [H, V]
void gdn_step(const float* q, const float* k, const float* v,
              const float* beta, const float* gate, const float* decay,
              float* state, float* out,
              int H, int K, int V, cudaStream_t s);

// If reset_mask[h] is nonzero, the state row for head h is zeroed before
// the recurrence step. Used at document boundaries in prefill.
void gdn_step_reset(const float* q, const float* k, const float* v,
                    const float* beta, const float* gate, const float* decay,
                    const uint8_t* reset_mask,  // [H] or nullptr
                    float* state, float* out,
                    int H, int K, int V, cudaStream_t s);

// Depthwise causal 1-D convolution gate.
//   history: [K-1, D] previous hidden states (last K-1 tokens)
//   current: [D]      current token's hidden state
//   weight:  [D, K]   conv weights (row-major, per output channel)
//   out:     [D]
void gdn_conv_gate(const float* history,   // [K-1, D]
                   const float* current,   // [D]
                   const float* weight,    // [D, K]
                   float* out,             // [D]
                   int D, int K, cudaStream_t s);

// Roll the conv history: history[:-1] = history[1:]; history[-1] = current.
void gdn_conv_roll(float* history,         // [K-1, D]
                   const float* current,   // [D]
                   int D, int K, cudaStream_t s);

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

// Banded causal local attention (prefill path).
//   q, k, v : [H, T, D]
//   out     : [H, T, D]
//   W       : window size (>= 1)
// One block per (head, query position). Handles reset_mask if non-null:
// only queries with segment_ids[q_pos] == segment_ids[key_pos] are visible.
void banded_attn(const float* q, const float* k, const float* v,
                 const int32_t* segment_ids,   // [T] or nullptr
                 float* out, int H, int T, int D, int W, cudaStream_t s);

// Local attention for a single decode token (T=1).
//   q       : [H, D]
//   k_hist  : [H, W_hist, D]
//   v_hist  : [H, W_hist, D]
//   valid   : [H, W_hist] (uint8 0/1) or nullptr
//   out     : [H, D]
void local_attn_decode(const float* q,
                       const float* k_hist, const float* v_hist,
                       const uint8_t* valid,
                       float* out,
                       int H, int W_hist, int D, cudaStream_t s);

// Top-k indices of q @ K^T / sqrt(D), per head.
//   q       : [H, D]
//   K       : [M, D]
//   idx_out : [H, k] int32
// Writes the k largest indices per head. Tie-breaking is by lower index.
// (Used by CSA remote attention to select which compressed memory entries
// to attend to.)
void attn_topk_scores(const float* q, const float* K,
                      int32_t* idx_out,
                      int H, int M, int D, int k, cudaStream_t s);

// Remote attention given precomputed top-k indices.
//   q   : [H, D]
//   K   : [M, D]
//   V   : [M, D]
//   idx : [H, k] int32 (indices into M)
//   out : [H, D]
// Applies softmax over the k selected scores, then weighted sum of V.
// Entries with idx < 0 are skipped.
void remote_attn(const float* q, const float* K, const float* V,
                 const int32_t* idx, float* out,
                 int H, int M, int D, int k, cudaStream_t s);

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

// Top-k selection + softmax on [n] logits. Writes k indices and probs.
void router_topk(const float* logits, int n, int k,
                 int32_t* idx_out, float* prob_out,
                 cudaStream_t s);

// ---------------------------------------------------------------------------
// MoE
// ---------------------------------------------------------------------------

// Single expert forward: out[d] = W_down @ (silu(W_gate @ x) * (W_up @ x))
//   x       : [D]
//   gate_w  : [H, D]
//   up_w    : [H, D]
//   down_w  : [D, H]
//   out     : [D]
void moe_expert_fwd(const float* x,
                    const float* gate_w, const float* up_w,
                    const float* down_w,
                    float* out, int D, int H, cudaStream_t s);

// Full MoE for one token: routes through k experts, weights outputs by prob.
//   x            : [D]
//   experts_gate : [E, H, D]
//   experts_up   : [E, H, D]
//   experts_down : [E, D, H]
//   idx          : [k] int32
//   prob         : [k]
//   out          : [D]
//   expert_scratch : [E_max, D] scratch for the per-expert outputs
void moe_route(const float* x,
               const float* experts_gate, const float* experts_up,
               const float* experts_down,
               const int32_t* idx, const float* prob,
               float* out, float* expert_scratch,
               int D, int H, int E, int k, cudaStream_t s);

// One-token selected-expert MoE. gate_up is packed as [E, 2*H, D],
// down is [E, D, H], and idx/prob contain k routes. Scratch is [k, H]
// activations and [k, D] weighted expert outputs.
void moe_selected_token_fp32(const float* x, const int64_t* idx,
                             const float* prob, const float* gate_up,
                             const float* down, float* activation,
                             float* route_output, float* out,
                             int D, int H, int E, int k,
                             cudaStream_t s);

// ---------------------------------------------------------------------------
// LM head
// ---------------------------------------------------------------------------

// logits[v] = embed_w[v] @ hidden for v in [0, V).
void lm_head(const float* hidden, const float* embed_w, float* logits,
             int V, int D, cublasHandle_t blas, cudaStream_t s);

// ---------------------------------------------------------------------------
// Embedding lookup
// ---------------------------------------------------------------------------

void embedding_lookup(const float* embed_w, int32_t token, float* out,
                      int D, cudaStream_t s);

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

int32_t sample_topk_topp(const float* logits, int V,
                         float temperature, int top_k, float top_p,
                         float repetition_penalty,
                         const int32_t* recent_tokens, int n_recent,
                         uint64_t* rng_state, cudaStream_t s);

}  // namespace llmy
#endif
