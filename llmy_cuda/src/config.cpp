#include "config.h"
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <cstdlib>

namespace llmy {

namespace {

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r')) ++a;
    while (b > a && (s[b-1] == ' ' || s[b-1] == '\t' || s[b-1] == '\r')) --b;
    return s.substr(a, b - a);
}

bool parse_bool(const std::string& s) {
    if (s == "1" || s == "true" || s == "True" || s == "yes") return true;
    if (s == "0" || s == "false" || s == "False" || s == "no") return false;
    throw std::runtime_error("expected bool, got: " + s);
}

int32_t parse_i32(const std::string& s) {
    return (int32_t)std::stol(s);
}

float parse_f32(const std::string& s) {
    return std::stof(s);
}

std::vector<std::string> parse_list(const std::string& s) {
    std::vector<std::string> out;
    if (s.empty()) return out;
    std::string cur;
    for (char c : s) {
        if (c == ',') {
            out.push_back(trim(cur));
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(trim(cur));
    return out;
}

}  // namespace

Config Config::parse(const std::string& text) {
    Config c;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));

        // ---------------------------------------------------------------
        // Global
        // ---------------------------------------------------------------
        if      (key == "vocab_size")                       c.vocab_size = parse_i32(val);
        else if (key == "d_model")                          c.d_model = parse_i32(val);
        else if (key == "num_loops")                        c.num_loops = parse_i32(val);
        else if (key == "max_position_embeddings")          c.max_position_embeddings = parse_i32(val);
        else if (key == "loop_attnres.latent_dim")         c.loop_attnres_latent_dim = parse_i32(val);
        else if (key == "loop_attnres.enabled")           c.loop_attnres_enabled = parse_bool(val);
        else if (key == "loop_attnres.heads")             c.loop_attnres_heads = parse_i32(val);
        else if (key == "loop_attnres.head_dim")          c.loop_attnres_head_dim = parse_i32(val);

        // ---------------------------------------------------------------
        // Structure
        // ---------------------------------------------------------------
        else if (key == "stem_blocks")                      c.stem_blocks = parse_i32(val);
        else if (key == "recurrent_blocks")                 c.recurrent_blocks = parse_i32(val);
        else if (key == "tail_blocks")                      c.tail_blocks = parse_i32(val);
        else if (key == "recurrent_layout")                 c.recurrent_layout = parse_list(val);

        // ---------------------------------------------------------------
        // mHC
        // ---------------------------------------------------------------
        else if (key == "mhc.enabled")                      c.mhc_enabled = parse_bool(val);
        else if (key == "mhc.streams")                      c.mhc_streams = parse_i32(val);
        else if (key == "mhc.sinkhorn_iterations")          c.mhc_sinkhorn_iterations = parse_i32(val);
        else if (key == "mhc.shared_across_loops")          c.mhc_shared_across_loops = parse_bool(val);
        else if (key == "mhc.preserve_streams_across_loop_boundary")
                                                            c.mhc_preserve_streams_across_loop_boundary = parse_bool(val);

        // ---------------------------------------------------------------
        // GDN
        // ---------------------------------------------------------------
        else if (key == "gdn.qk_heads")                     c.gdn_qk_heads = parse_i32(val);
        else if (key == "gdn.value_heads")                  c.gdn_value_heads = parse_i32(val);
        else if (key == "gdn.qk_head_dim")                  c.gdn_qk_head_dim = parse_i32(val);
        else if (key == "gdn.value_head_dim")               c.gdn_value_head_dim = parse_i32(val);
        else if (key == "gdn.conv_kernel")                  c.gdn_conv_kernel = parse_i32(val);

        // ---------------------------------------------------------------
        // CSA
        // ---------------------------------------------------------------
        else if (key == "csa.temporal_compression")         c.csa_temporal_compression = parse_i32(val);
        else if (key == "csa.local_window")                 c.csa_local_window = parse_i32(val);
        else if (key == "csa.index_heads")                  c.csa_index_heads = parse_i32(val);
        else if (key == "csa.index_head_dim")               c.csa_index_head_dim = parse_i32(val);
        else if (key == "csa.index_top_k")                  c.csa_index_top_k = parse_i32(val);

        // ---------------------------------------------------------------
        // HCA
        // ---------------------------------------------------------------
        else if (key == "hca.temporal_compression")         c.hca_temporal_compression = parse_i32(val);
        else if (key == "hca.local_window")                 c.hca_local_window = parse_i32(val);
        else if (key == "hca.dense_global_attention")      c.hca_dense_global_attention = parse_bool(val);
        else if (key == "hca.global_top_k")                 c.hca_global_top_k = parse_i32(val);

        // ---------------------------------------------------------------
        // MoE
        // ---------------------------------------------------------------
        else if (key == "moe.routed_experts")               c.moe_routed_experts = parse_i32(val);
        else if (key == "moe.shared_experts")               c.moe_shared_experts = parse_i32(val);
        else if (key == "moe.routed_top_k")                 c.moe_routed_top_k = parse_i32(val);
        else if (key == "moe.expert_hidden")                c.moe_expert_hidden = parse_i32(val);

        // ---------------------------------------------------------------
        // MLA / Router
        // ---------------------------------------------------------------
        else if (key == "mla.latent_dim")                   c.mla_latent_dim = parse_i32(val);
        else if (key == "router.context_aware")             c.router_context_aware = parse_bool(val);
        else if (key == "router.query_heads")               c.router_query_heads = parse_i32(val);
        else if (key == "router.query_head_dim")            c.router_query_head_dim = parse_i32(val);
        else if (key == "router.local_context")             c.router_local_context = parse_i32(val);

        // ---------------------------------------------------------------
        // Dense FFN
        // ---------------------------------------------------------------
        else if (key == "dense_ffn.hidden")                 c.dense_ffn_hidden = parse_i32(val);

        // ---------------------------------------------------------------
        // Adaptive halting
        // ---------------------------------------------------------------
        else if (key == "adaptive_loops.enabled")           c.adaptive_enabled = parse_bool(val);
        else if (key == "adaptive_loops.learned_halting")   c.adaptive_learned_halting = parse_bool(val);
        else if (key == "adaptive_loops.min_loops")         c.adaptive_min_loops = parse_i32(val);
        else if (key == "adaptive_loops.max_loops")         c.adaptive_max_loops = parse_i32(val);
        else if (key == "adaptive_loops.halt_threshold")    c.adaptive_halt_threshold = parse_f32(val);
        else if (key == "adaptive_loops.halt_temperature")  c.adaptive_halt_temperature = parse_f32(val);

        // Silently ignore everything else (unknown sub-keys are fine).
    }

    // Sanity: clamp max_loops to num_loops.
    if (c.adaptive_max_loops > c.num_loops) c.adaptive_max_loops = c.num_loops;

    return c;
}

}  // namespace llmy
