#ifndef LLMY_CONFIG_H
#define LLMY_CONFIG_H

#include <string>
#include <vector>
#include <cstdint>

namespace llmy {

// The runtime configuration, parsed from the checkpoint's embedded config
// text. The text format is a flat key=value listing, one entry per line,
// with list values comma-separated. See tools/convert_checkpoint.py for
// the emitter.
struct Config {
    // Global
    int32_t vocab_size = 248320;
    int32_t d_model = 1024;
    int32_t num_loops = 2;

    // Structure
    int32_t stem_blocks = 2;
    int32_t recurrent_blocks = 6;
    int32_t tail_blocks = 2;
    std::vector<std::string> recurrent_layout;  // "gdn" or "csa"

    // mHC
    bool    mhc_enabled = true;
    int32_t mhc_streams = 4;
    int32_t mhc_sinkhorn_iterations = 20;
    bool    mhc_shared_across_loops = true;
    bool    mhc_preserve_streams_across_loop_boundary = true;

    // GDN
    int32_t gdn_qk_heads = 16;
    int32_t gdn_value_heads = 16;
    int32_t gdn_qk_head_dim = 128;
    int32_t gdn_value_head_dim = 128;
    int32_t gdn_conv_kernel = 4;

    // CSA
    int32_t csa_temporal_compression = 4;
    int32_t csa_local_window = 128;
    int32_t csa_index_heads = 16;
    int32_t csa_index_head_dim = 128;
    int32_t csa_index_top_k = 32;

    // HCA
    int32_t hca_temporal_compression = 32;
    int32_t hca_local_window = 128;
    bool    hca_dense_global_attention = true;
    int32_t hca_global_top_k = 32;

    // MoE
    int32_t moe_routed_experts = 128;
    int32_t moe_shared_experts = 4;
    int32_t moe_routed_top_k = 20;
    int32_t moe_expert_hidden = 640;

    // MLA
    int32_t mla_latent_dim = 256;

    // Router
    bool    router_context_aware = true;
    int32_t router_query_heads = 4;
    int32_t router_query_head_dim = 64;
    int32_t router_local_context = 128;

    // Dense FFN
    int32_t dense_ffn_hidden = 3584;

    // Adaptive halting
    bool    adaptive_enabled = false;
    bool    adaptive_learned_halting = true;
    int32_t adaptive_min_loops = 1;
    int32_t adaptive_max_loops = 1;
    float   adaptive_halt_threshold = 0.5f;
    float   adaptive_halt_temperature = 1.0f;

    // Maximum context length for preallocation of decode caches.
    int32_t max_position_embeddings = 4096;

    // Optional loop-attention residual latent width.
    int32_t loop_attnres_latent_dim = 256;
    bool    loop_attnres_enabled = false;
    int32_t loop_attnres_heads = 4;
    int32_t loop_attnres_head_dim = 32;

    static Config parse(const std::string& text);
};

}  // namespace llmy
#endif
