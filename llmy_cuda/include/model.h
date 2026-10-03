#ifndef LLMY_MODEL_H
#define LLMY_MODEL_H

#include "config.h"
#include "checkpoint.h"
#include "tensor.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmy {

// ---------------------------------------------------------------------------
// Weight bundles. All pointers are into the mmap'd checkpoint. They are
// const and never written.
// ---------------------------------------------------------------------------

struct MHCWeights {
    const float* read_logits     = nullptr;   // [S, S]
    const float* write_logits    = nullptr;   // [S, S]
    const float* stream_scale    = nullptr;   // [S, D]
    const float* stream_norm_w   = nullptr;   // [D]
    const float* stream_norm_b   = nullptr;   // [D]
    const float* mapping_norm_w  = nullptr;   // [S*D]
    const float* pre_w           = nullptr;   // [S, S*D]
    const float* post_w          = nullptr;   // [S, S*D]
    const float* residual_w      = nullptr;   // [S*S, S*D]
    const float* pre_bias        = nullptr;   // [S]
    const float* post_bias       = nullptr;   // [S]
    const float* residual_bias   = nullptr;   // [S, S]
    const float* alpha_pre       = nullptr;   // [1]
    const float* alpha_post      = nullptr;   // [1]
    const float* alpha_residual  = nullptr;   // [1]
    const float* gating_logit    = nullptr;   // [1]
};

struct GDNWeights {
    const float* q_w      = nullptr;   // [H*K, D]
    const float* k_w      = nullptr;   // [H*K, D]
    const float* v_w      = nullptr;   // [H*V, D]
    const float* beta_w   = nullptr;   // [H, D]
    const float* beta_b   = nullptr;   // [H]
    const float* gate_w   = nullptr;   // [H*V, D]
    const float* out_w    = nullptr;   // [D, H*V]
    const float* conv_w   = nullptr;   // [D, 1, K]
    const float* decay_logit = nullptr; // [H]
};

struct DenseFFNWeights {
    const float* gate_w = nullptr;   // [Hff, D]
    const float* up_w   = nullptr;   // [Hff, D]
    const float* down_w = nullptr;   // [D, Hff]
};

struct LatentAttentionWeights {
    const float* q_w = nullptr;   // [heads*hd, D]
    const float* k_w = nullptr;   // [heads*hd, latent]
    const float* v_w = nullptr;   // [heads*hd, latent]
    const float* o_w = nullptr;   // [D, heads*hd]
};

struct MLALatentWeights {
    const float* down_w = nullptr;   // [latent, D]
    const float* up_w   = nullptr;   // [D, latent]
    const float* norm_w = nullptr;   // [latent]
    const float* norm_b = nullptr;   // [latent]
};

struct RouterWeights {
    const float* norm_w        = nullptr;   // [D]
    MLALatentWeights latent_memory;
    LatentAttentionWeights latent_attention;
    const float* local_ctx_w   = nullptr;   // [D, latent]
    const float* route_gate    = nullptr;   // [1]
    const float* context_w     = nullptr;   // [D, D]
    const float* logits_w      = nullptr;   // [E, D]
};

struct MoEWeights {
    RouterWeights router;
    const float* experts_gate = nullptr;   // [E, H, D]
    const float* experts_up   = nullptr;   // [E, H, D]
    const float* experts_down = nullptr;   // [E, D, H]
    const float* shared_gate  = nullptr;   // [Ss, H, D]
    const float* shared_up    = nullptr;   // [Ss, H, D]
    const float* shared_down  = nullptr;   // [Ss, D, H]
};

struct CompressorWeights {
    const float* key_w   = nullptr;   // [C, D]
    const float* value_w = nullptr;   // [D, D]
};

struct CSAWeights {
    const float* query_w     = nullptr;   // [heads*hd, D]
    const float* key_w       = nullptr;   // [heads*hd, D]
    const float* value_w     = nullptr;   // [D, D]
    const float* local_out_w = nullptr;   // [D, D]
    const float* remote_out_w= nullptr;   // [D, D]
    const float* out_w       = nullptr;   // [D, D]
    CompressorWeights compressor;
};

struct HCAWeights {
    const float* query_w     = nullptr;   // [D, D]
    const float* key_w       = nullptr;   // [D, D]
    const float* value_w     = nullptr;   // [D, D]
    const float* local_out_w = nullptr;   // [D, D]
    const float* global_out_w= nullptr;   // [D, D]
    const float* out_w       = nullptr;   // [D, D]
    CompressorWeights compressor;
};

// One physical block.
struct Block {
    std::string kind;          // "gdn" | "csa" | "hca"
    bool with_moe = false;

    MHCWeights  mhc;
    const float* mixer_norm_w = nullptr;   // [D]
    const float* ffn_norm_w   = nullptr;   // [D]

    // Exactly one of these is populated.
    GDNWeights   gdn;
    CSAWeights   csa;
    HCAWeights   hca;

    // Exactly one of these is populated.
    DenseFFNWeights ffn;      // when with_moe == false
    MoEWeights      moe;      // when with_moe == true

    // Optional HCA latent memory projection (only for HCA blocks).
    MLALatentWeights hca_latent;
};

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

struct Model {
    ~Model();

    Config cfg;
    Checkpoint ck;
    std::vector<void*> device_weights;

    // Global weights
    const float* embed_w         = nullptr;   // [V, D]
    const float* stream_seed     = nullptr;   // [S, D]
    const float* final_norm_w    = nullptr;   // [D]
    const float* mhc_collapse_w  = nullptr;   // [D, S*D]
    const float* halt_w          = nullptr;   // [1, D]
    const float* halt_b          = nullptr;   // [1]

    // Stacked blocks
    std::vector<Block> stem;
    std::vector<Block> core;
    std::vector<Block> tail;

    // Loop memory hook (optional; usually disabled)
    bool loop_memory_enabled = false;
    const float* loop_mem_compressor_down = nullptr;
    const float* loop_mem_compressor_up   = nullptr;
    const float* loop_mem_norm_w          = nullptr;
    const float* loop_mem_norm_b          = nullptr;
    const float* loop_mem_query_norm_w    = nullptr;
    const float* loop_mem_query_norm_b    = nullptr;
    LatentAttentionWeights loop_mem_attention;
    const float* loop_mem_gating_logit    = nullptr;

    // cuBLAS handle shared across the whole model.
    cublasHandle_t blas = nullptr;
};

// ---------------------------------------------------------------------------
// Per-conversation state
// ---------------------------------------------------------------------------

// A compressed-memory buffer with fixed capacity, used by CSA/HCA decode
// caches. keys/values/positions/segment_ids each have shape [B, capacity, ...].
struct MemoryCache {
    float* keys = nullptr;         // [B, capacity, Dk]
    float* values = nullptr;       // [B, capacity, Dv]
    int32_t* positions = nullptr;  // [B, capacity]
    int32_t* segment_ids = nullptr;// [B, capacity]
    float* latents = nullptr;      // [B, capacity, latent] or nullptr
    int32_t length = 0;            // active prefix length
    int32_t capacity = 0;
    int32_t dk = 0;
    int32_t dv = 0;
    int32_t latent = 0;
    // Temporal compression pending bucket
    float* pending_sum = nullptr;   // [B, D]
    int32_t* pending_count = nullptr;// [B]
    int32_t* pending_seg = nullptr; // [B]
    int32_t pending_count_host = 0;
    int32_t pending_seg_host = 0;
    int32_t write_index = 0;       // next compressed row to overwrite
};

// One block's per-token history.
struct BlockHistory {
    // GDN conv history: last (K-1) hidden states, [B, K-1, D]
    float* conv_hist = nullptr;
    // Router latent history: [B, ctx, latent]
    float* router_latent_hist = nullptr;
    int32_t* router_hist_valid = nullptr;  // [B, ctx] bool
    // CSA/HCA local key and value caches: [B, W-1, ...]
    float* local_k_cache = nullptr;
    float* local_v_cache = nullptr;
    uint8_t* local_valid = nullptr;         // [B, W-1] bool
    int32_t local_valid_count = 0;           // valid rows currently cached
};

struct State {
    // mHC streams — the residual that flows through everything.
    float* streams = nullptr;   // [B, S, D]
    // Streams at the recurrent-loop boundary. The reference model restores
    // this value before every loop unless preserve_streams_across_loop_boundary
    // is enabled.
    float* loop_boundary_streams = nullptr; // [B, S, D]

    // GDN recurrent state per block, flattened [n_blocks][H, K, V]
    std::vector<float*> gdn_states;

    // Attention memory per block (only populated for CSA/HCA blocks)
    std::vector<MemoryCache> csa_memories;
    std::vector<MemoryCache> hca_memories;

    // Per-block history
    std::vector<BlockHistory> histories;

    // Token position (absolute, 0-based)
    int32_t position = 0;
    int32_t batch = 1;

    // Loop endpoint history for loop attention residual
    std::vector<float*> loop_endpoints;   // [B, T=1, latent]
    int loop_endpoint_count = 0;
    float* loop_endpoint_storage = nullptr;

    // Diagnostics
    int32_t last_executed_loops = 0;

    // Scratch buffers used across the forward pass (allocated once).
    float* scratch_hidden = nullptr;      // [B, D]
    float* scratch_hidden2 = nullptr;     // [B, D]
    float* scratch_update = nullptr;      // [B, D]
    float* scratch_mixed = nullptr;       // [B, max(D, H*K)]
    float* scratch_q = nullptr;           // [B, H*K]
    float* scratch_k = nullptr;           // [B, H*K]
    float* scratch_v = nullptr;           // [B, H*V]
    float* scratch_gate = nullptr;        // [B, H*V]
    float* scratch_beta = nullptr;        // [B, H]
    float* scratch_decay = nullptr;       // [H]
    float* scratch_mhc_pre = nullptr;     // [B, S]
    float* scratch_mhc_post = nullptr;    // [B, S]
    float* scratch_mhc_residual = nullptr;// [B, S, S]
    float* scratch_mhc_logits = nullptr;  // [B, S, S]
    float* scratch_router_logits = nullptr;// [B, E]
    int32_t* scratch_topk_idx = nullptr;   // [B, max(top_k)]
    float* scratch_topk_prob = nullptr;    // [B, max(top_k)]
    float* scratch_logits = nullptr;       // [V] — the final output buffer
};

// ---------------------------------------------------------------------------
// Public entry points (called from llmy.cpp, the C ABI layer)
// ---------------------------------------------------------------------------

std::unique_ptr<Model> load_model(const std::string& path);
std::unique_ptr<State> alloc_state(Model& m, int batch);
void free_state(State& s);

// Prefill the whole prompt at once. Writes logits for the last position.
void model_prefill(Model& m, State& s,
                   const int32_t* tokens, int n_tokens,
                   float* logits_out);

// Decode one token. Writes logits for that token.
void model_decode(Model& m, State& s,
                  int32_t token, int32_t position, int32_t num_loops,
                  float* logits_out);

}  // namespace llmy
#endif
