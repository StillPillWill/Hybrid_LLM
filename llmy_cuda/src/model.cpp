#include "model.h"
#include "kernels.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <initializer_list>
#include <array>
#include <cstdio>
#include <cstdlib>

namespace llmy {

// ===========================================================================
// Loading
// ===========================================================================

namespace {

const float* W(const Checkpoint& ck, const std::string& name) {
    auto it = ck.tensors().find(name);
    if (it == ck.tensors().end()) {
        throw std::runtime_error("missing tensor: " + name);
    }
    if (it->second.dtype != DType::F32) {
        throw std::runtime_error("tensor is not fp32: " + name);
    }
    return it->second.as_f32();
}

MHCWeights load_mhc(const Checkpoint& ck, const std::string& p) {
    MHCWeights w;
    w.read_logits    = W(ck, p + ".mhc.read_logits");
    w.write_logits   = W(ck, p + ".mhc.write_logits");
    w.stream_scale   = W(ck, p + ".mhc.stream_scale");
    w.stream_norm_w  = W(ck, p + ".mhc.stream_norm.weight");
    w.stream_norm_b  = W(ck, p + ".mhc.stream_norm.bias");
    w.mapping_norm_w = W(ck, p + ".mhc.mapping_norm.weight");
    if (ck.has(p + ".mhc.theta.weight")) {
        // Current checkpoints pack the three dynamic mHC projections into
        // one Linear: [pre(S), post(S), residual(S*S), S*D].
        const auto& theta_ref = ck.get(p + ".mhc.theta.weight");
        const float* theta = theta_ref.as_f32();
        const int S = (int)ck.get(p + ".mhc.pre_bias").shape[0];
        const int D = (int)theta_ref.shape[1] / S;
        w.pre_w      = theta;
        w.post_w     = theta + (size_t)S * S * D;
        w.residual_w = theta + (size_t)2 * S * S * D;
    } else {
        w.pre_w      = W(ck, p + ".mhc.pre_theta.weight");
        w.post_w     = W(ck, p + ".mhc.post_theta.weight");
        w.residual_w = W(ck, p + ".mhc.residual_theta.weight");
    }
    w.pre_bias       = W(ck, p + ".mhc.pre_bias");
    w.post_bias      = W(ck, p + ".mhc.post_bias");
    w.residual_bias  = W(ck, p + ".mhc.residual_bias");
    w.alpha_pre      = W(ck, p + ".mhc.alpha_pre");
    w.alpha_post     = W(ck, p + ".mhc.alpha_post");
    w.alpha_residual = W(ck, p + ".mhc.alpha_residual");
    w.gating_logit   = W(ck, p + ".mhc.gating_logit");
    return w;
}

GDNWeights load_gdn(const Checkpoint& ck, const std::string& p) {
    GDNWeights w;
    w.q_w         = W(ck, p + ".q_proj.weight");
    w.k_w         = W(ck, p + ".k_proj.weight");
    w.v_w         = W(ck, p + ".v_proj.weight");
    w.beta_w      = W(ck, p + ".beta_proj.weight");
    w.beta_b      = W(ck, p + ".beta_proj.bias");
    w.gate_w      = W(ck, p + ".gate_proj.weight");
    w.out_w       = W(ck, p + ".out_proj.weight");
    w.conv_w      = W(ck, p + ".conv.weight");
    w.decay_logit = W(ck, p + ".decay_logit");
    return w;
}

DenseFFNWeights load_dffn(const Checkpoint& ck, const std::string& p) {
    DenseFFNWeights w;
    w.gate_w = W(ck, p + ".ffn.gate_proj.weight");
    w.up_w   = W(ck, p + ".ffn.up_proj.weight");
    w.down_w = W(ck, p + ".ffn.down_proj.weight");
    return w;
}

MLALatentWeights load_latent_memory(const Checkpoint& ck, const std::string& p) {
    MLALatentWeights w;
    w.down_w = W(ck, p + ".down.weight");
    w.up_w   = W(ck, p + ".up.weight");
    w.norm_w = W(ck, p + ".norm.weight");
    w.norm_b = W(ck, p + ".norm.bias");
    return w;
}

LatentAttentionWeights load_latent_attention(const Checkpoint& ck, const std::string& p) {
    LatentAttentionWeights w;
    w.q_w = W(ck, p + ".q.weight");
    w.k_w = W(ck, p + ".k.weight");
    w.v_w = W(ck, p + ".v.weight");
    w.o_w = W(ck, p + ".o.weight");
    return w;
}

RouterWeights load_router(const Checkpoint& ck, const std::string& p) {
    RouterWeights w;
    w.norm_w          = W(ck, p + ".norm.weight");
    w.latent_memory   = load_latent_memory(ck, p + ".latent_memory");
    w.latent_attention = load_latent_attention(ck, p + ".latent_attention");
    w.local_ctx_w     = W(ck, p + ".local_context_out.weight");
    w.route_gate      = W(ck, p + ".route_gate");
    w.context_w       = W(ck, p + ".context_out.weight");
    w.logits_w        = W(ck, p + ".logits.weight");
    return w;
}

MoEWeights load_moe(const Checkpoint& ck, const std::string& p) {
    MoEWeights w;
    w.router = load_router(ck, p + ".router");
    // Note: converter collapses ModuleList indices so these are stacked [E, H, D].
    w.experts_gate = W(ck, p + ".experts.gate_proj.weight");
    w.experts_up   = W(ck, p + ".experts.up_proj.weight");
    w.experts_down = W(ck, p + ".experts.down_proj.weight");
    w.shared_gate  = W(ck, p + ".shared.gate_proj.weight");
    w.shared_up    = W(ck, p + ".shared.up_proj.weight");
    w.shared_down  = W(ck, p + ".shared.down_proj.weight");
    return w;
}

CompressorWeights load_compressor(const Checkpoint& ck, const std::string& p) {
    CompressorWeights w;
    w.key_w   = W(ck, p + ".key.weight");
    w.value_w = W(ck, p + ".value.weight");
    return w;
}

CSAWeights load_csa(const Checkpoint& ck, const std::string& p) {
    CSAWeights w;
    w.query_w      = W(ck, p + ".query.weight");
    w.key_w        = W(ck, p + ".key.weight");
    w.value_w      = W(ck, p + ".value.weight");
    w.local_out_w  = W(ck, p + ".local_out.weight");
    w.remote_out_w = W(ck, p + ".remote_out.weight");
    w.out_w        = W(ck, p + ".out.weight");
    w.compressor   = load_compressor(ck, p + ".compressor");
    return w;
}

HCAWeights load_hca(const Checkpoint& ck, const std::string& p) {
    HCAWeights w;
    w.query_w      = W(ck, p + ".query.weight");
    w.key_w        = W(ck, p + ".key.weight");
    w.value_w      = W(ck, p + ".value.weight");
    w.local_out_w  = W(ck, p + ".local_out.weight");
    w.global_out_w = W(ck, p + ".global_out.weight");
    w.out_w        = W(ck, p + ".out.weight");
    w.compressor   = load_compressor(ck, p + ".compressor");
    return w;
}

}  // namespace

std::unique_ptr<Model> load_model(const std::string& path) {
    auto m = std::make_unique<Model>();
    m->ck = Checkpoint::open(path);
    m->ck.upload_to_device(m->device_weights);
    m->cfg = Config::parse(m->ck.config_text());

    // cuBLAS handle.
    if (cublasCreate(&m->blas) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasCreate failed");
    }

    // Global weights.
    m->embed_w        = W(m->ck, "embed_tokens.weight");
    m->stream_seed    = W(m->ck, "stream_seed");
    m->final_norm_w   = W(m->ck, "final_norm.weight");
    m->mhc_collapse_w = W(m->ck, "mhc_collapse.weight");

    // Loop-attention residual memory.
    m->loop_memory_enabled = false; // diagnostic: isolate prefill/decode state parity
    if (m->loop_memory_enabled) {
        m->loop_mem_compressor_down = W(m->ck, "loop_memory.compressor.down.weight");
        m->loop_mem_compressor_up   = W(m->ck, "loop_memory.compressor.up.weight");
        m->loop_mem_norm_w          = W(m->ck, "loop_memory.compressor.norm.weight");
        m->loop_mem_norm_b          = W(m->ck, "loop_memory.compressor.norm.bias");
        m->loop_mem_query_norm_w    = W(m->ck, "loop_memory.query_norm.weight");
        m->loop_mem_query_norm_b    = W(m->ck, "loop_memory.query_norm.bias");
        m->loop_mem_attention.q_w  = W(m->ck, "loop_memory.attention.q.weight");
        m->loop_mem_attention.k_w  = W(m->ck, "loop_memory.attention.k.weight");
        m->loop_mem_attention.v_w  = W(m->ck, "loop_memory.attention.v.weight");
        m->loop_mem_attention.o_w  = W(m->ck, "loop_memory.attention.o.weight");
        m->loop_mem_gating_logit    = W(m->ck, "loop_memory.gating_logit");
    }

    // Halting head is optional (legacy checkpoints omit it).
    if (m->ck.has("recurrent_halt_head.weight")) {
        m->halt_w = W(m->ck, "recurrent_halt_head.weight");
        m->halt_b = W(m->ck, "recurrent_halt_head.bias");
    }

    // Stem: GDN + dense ffn, no MoE.
    for (int i = 0; i < m->cfg.stem_blocks; ++i) {
        Block b;
        b.kind = "gdn";
        b.with_moe = false;
        std::string p = "stem." + std::to_string(i);
        b.mhc = load_mhc(m->ck, p);
        b.mixer_norm_w = W(m->ck, p + ".mixer_norm.weight");
        b.ffn_norm_w   = W(m->ck, p + ".ffn_norm.weight");
        b.gdn = load_gdn(m->ck, p + ".mixer");
        b.ffn = load_dffn(m->ck, p);
        m->stem.push_back(b);
    }

    // Recurrent core: alternate gdn/csa/hca, all with MoE.
    for (int i = 0; i < (int)m->cfg.recurrent_layout.size(); ++i) {
        Block b;
        b.kind = m->cfg.recurrent_layout[i].substr(0, 3);  // "gdn"|"csa"|"hca"
        b.with_moe = true;
        std::string p = "recurrent_core." + std::to_string(i);
        b.mhc = load_mhc(m->ck, p);
        b.mixer_norm_w = W(m->ck, p + ".mixer_norm.weight");
        b.ffn_norm_w   = W(m->ck, p + ".ffn_norm.weight");
        if (b.kind == "gdn") b.gdn = load_gdn(m->ck, p + ".mixer");
        else if (b.kind == "csa") b.csa = load_csa(m->ck, p + ".mixer");
        else if (b.kind == "hca") b.hca = load_hca(m->ck, p + ".mixer");
        else throw std::runtime_error("unknown block kind: " + b.kind);
        b.moe = load_moe(m->ck, p + ".moe");
        // HCA has its own latent memory.
        if (b.kind == "hca") {
            b.hca_latent = load_latent_memory(m->ck, p + ".hca_latent");
        }
        m->core.push_back(b);
    }

    // Tail: same structure as stem.
    for (int i = 0; i < m->cfg.tail_blocks; ++i) {
        Block b;
        b.kind = "gdn";
        b.with_moe = false;
        std::string p = "tail." + std::to_string(i);
        b.mhc = load_mhc(m->ck, p);
        b.mixer_norm_w = W(m->ck, p + ".mixer_norm.weight");
        b.ffn_norm_w   = W(m->ck, p + ".ffn_norm.weight");
        b.gdn = load_gdn(m->ck, p + ".mixer");
        b.ffn = load_dffn(m->ck, p);
        m->tail.push_back(b);
    }

    return m;
}

Model::~Model() {
    for (void* p : device_weights) cudaFree(p);
    device_weights.clear();
}

// ===========================================================================
// State allocation
// ===========================================================================

namespace {

float* dev_alloc_f(size_t n) {
    float* p = nullptr;
    if (cudaMalloc((void**)&p, n * sizeof(float)) != cudaSuccess) {
        throw std::runtime_error("cudaMalloc (float) failed");
    }
    cudaMemset(p, 0, n * sizeof(float));
    return p;
}

int32_t* dev_alloc_i32(size_t n) {
    int32_t* p = nullptr;
    if (cudaMalloc((void**)&p, n * sizeof(int32_t)) != cudaSuccess) {
        throw std::runtime_error("cudaMalloc (int32) failed");
    }
    cudaMemset(p, 0, n * sizeof(int32_t));
    return p;
}

uint8_t* dev_alloc_u8(size_t n) {
    uint8_t* p = nullptr;
    if (cudaMalloc((void**)&p, n * sizeof(uint8_t)) != cudaSuccess) {
        throw std::runtime_error("cudaMalloc (uint8) failed");
    }
    cudaMemset(p, 0, n * sizeof(uint8_t));
    return p;
}

MemoryCache make_memory_cache(int B, int capacity, int dk, int dv, int latent) {
    MemoryCache c;
    c.capacity = capacity;
    c.dk = dk;
    c.dv = dv;
    c.latent = latent;
    c.length = 0;
    c.pending_count_host = 0;
    c.pending_seg_host = 0;
    c.keys      = dev_alloc_f((size_t)B * capacity * dk);
    c.values    = dev_alloc_f((size_t)B * capacity * dv);
    c.positions = dev_alloc_i32((size_t)B * capacity);
    c.segment_ids = dev_alloc_i32((size_t)B * capacity);
    if (latent > 0) c.latents = dev_alloc_f((size_t)B * capacity * latent);
    return c;
}

}  // namespace

std::unique_ptr<State> alloc_state(Model& m, int batch) {
    auto s = std::make_unique<State>();
    const Config& cfg = m.cfg;
    const int B = batch;
    const int D = cfg.d_model;
    const int S = cfg.mhc_streams;
    const int H = cfg.gdn_qk_heads;
    const int K = cfg.gdn_qk_head_dim;
    const int V = cfg.gdn_value_head_dim;
    const int SD = S * D;
    const int SS = S * S;
    const int E = cfg.moe_routed_experts;
    const int top_k = cfg.moe_routed_top_k;

    s->batch = B;

    // mHC streams, initialized to zero.
    s->streams = dev_alloc_f((size_t)B * SD);
    s->loop_boundary_streams = dev_alloc_f((size_t)B * SD);

    // The Python model namespaces recurrent-core state by loop index. Stem
    // and tail state is shared, but every recurrent pass owns independent
    // GDN, attention, and router state.
    const int stem_blocks = (int)m.stem.size();
    const int core_blocks = (int)m.core.size();
    const int tail_blocks = (int)m.tail.size();
    const int loop_slots = std::max(1, cfg.num_loops);
    const int n_blocks = stem_blocks + loop_slots * core_blocks + tail_blocks;
    // Layout: stem, loop0 core, loop1 core, ..., tail.
    s->gdn_states.resize(n_blocks);
    for (int i = 0; i < n_blocks; ++i) {
        s->gdn_states[i] = dev_alloc_f((size_t)B * H * K * V);
    }

    // Memory caches.
    s->csa_memories.resize((size_t)loop_slots * core_blocks);
    s->hca_memories.resize((size_t)loop_slots * core_blocks);
    // The compressed memories are fixed-size decode caches.  Keep the
    // capacity bounded: csa_remote_attn_decode uses a score workspace
    // proportional to H*capacity in shared memory, and the reference format
    // uses 256 compressed rows rather than the full context length.
    int decode_cap = 256;

    for (int loop = 0; loop < loop_slots; ++loop) {
      for (int bi = 0; bi < core_blocks; ++bi) {
        const Block& b = m.core[bi];
        const size_t i = (size_t)loop * core_blocks + bi;
        if (b.kind == "csa") {
            int dk = cfg.csa_index_heads * cfg.csa_index_head_dim;
            s->csa_memories[i] = make_memory_cache(B, decode_cap, dk, D, 0);
            // Pending bucket.
            s->csa_memories[i].pending_sum = dev_alloc_f((size_t)B * D);
            s->csa_memories[i].pending_count = dev_alloc_i32((size_t)B);
            s->csa_memories[i].pending_seg = dev_alloc_i32((size_t)B);
        } else if (b.kind == "hca") {
            s->hca_memories[i] = make_memory_cache(B, decode_cap, D, D, cfg.mla_latent_dim);
            s->hca_memories[i].pending_sum = dev_alloc_f((size_t)B * D);
            s->hca_memories[i].pending_count = dev_alloc_i32((size_t)B);
            s->hca_memories[i].pending_seg = dev_alloc_i32((size_t)B);
        }
      }
    }

    // Per-block history.
    s->histories.resize(n_blocks);
    for (int i = 0; i < n_blocks; ++i) {
        BlockHistory h;
        // GDN conv history.
        h.conv_hist = dev_alloc_f((size_t)B * (cfg.gdn_conv_kernel - 1) * D);
        // Router latent history (only used for MoE blocks; allocate anyway).
        h.router_latent_hist = dev_alloc_f((size_t)B * cfg.router_local_context * cfg.mla_latent_dim);
        h.router_hist_valid = dev_alloc_i32((size_t)B * cfg.router_local_context);
        // Local K/V caches for CSA/HCA blocks.
        const Block* core_block = nullptr;
        if (i >= stem_blocks && i < stem_blocks + loop_slots * core_blocks) {
            core_block = &m.core[(i - stem_blocks) % core_blocks];
        }
        if (core_block == nullptr) {
            s->histories[i] = h;  // tail has no attention, but GDN needs conv history
            continue;
        }
        if (core_block->kind == "csa") {
            int dk = cfg.csa_index_heads * cfg.csa_index_head_dim;
            const int width = std::max(1, cfg.csa_local_window - 1);
            h.local_k_cache = dev_alloc_f((size_t)B * width * dk);
            h.local_v_cache = dev_alloc_f((size_t)B * width * D);
            h.local_valid   = dev_alloc_u8((size_t)B * width);
        } else if (core_block->kind == "hca") {
            const int width = std::max(1, cfg.hca_local_window - 1);
            h.local_k_cache = dev_alloc_f((size_t)B * width * D);
            h.local_v_cache = dev_alloc_f((size_t)B * width * D);
            h.local_valid   = dev_alloc_u8((size_t)B * width);
        }
        s->histories[i] = h;
    }

    // Loop endpoints (only for loop attention residual; disabled by default).
    s->loop_endpoints.reserve(cfg.num_loops);
    s->loop_endpoint_storage = dev_alloc_f(
        (size_t)B * cfg.num_loops * cfg.loop_attnres_latent_dim);
    for (int i = 0; i < cfg.num_loops; ++i)
        s->loop_endpoints.push_back(
            s->loop_endpoint_storage + (size_t)i * B * cfg.loop_attnres_latent_dim);

    // Scratch buffers.
    s->scratch_hidden      = dev_alloc_f((size_t)B * D);
    s->scratch_hidden2     = dev_alloc_f((size_t)B * D);
    s->scratch_update      = dev_alloc_f((size_t)B * D);
    int mix_width = std::max(H * K, H * V);
    s->scratch_mixed       = dev_alloc_f((size_t)B * mix_width);
    s->scratch_q           = dev_alloc_f((size_t)B * H * K);
    s->scratch_k           = dev_alloc_f((size_t)B * H * K);
    s->scratch_v           = dev_alloc_f((size_t)B * H * V);
    s->scratch_gate        = dev_alloc_f((size_t)B * H * V);
    s->scratch_beta        = dev_alloc_f((size_t)B * H);
    s->scratch_decay       = dev_alloc_f((size_t)H);
    s->scratch_mhc_pre     = dev_alloc_f((size_t)B * S);
    s->scratch_mhc_post    = dev_alloc_f((size_t)B * S);
    s->scratch_mhc_residual= dev_alloc_f((size_t)B * SS);
    s->scratch_mhc_logits  = dev_alloc_f((size_t)B * SS);
    s->scratch_router_logits = dev_alloc_f((size_t)B * E);
    s->scratch_topk_idx    = dev_alloc_i32((size_t)B * top_k);
    s->scratch_topk_prob   = dev_alloc_f((size_t)B * top_k);
    s->scratch_logits      = dev_alloc_f((size_t)cfg.vocab_size);
    // mHC mapping RMSNorm scratch.
    // (Reuse scratch_mixed if it's big enough, but for safety allocate a
    // dedicated buffer.)
    // We'll store it in scratch_hidden2 as long as SD <= D — which isn't
    // true. So allocate fresh.
    // TODO: track it as an explicit field rather than reusing.
    // Actually we do need a [B, S*D] buffer. Add it via a static cast hack:
    // we can't easily extend State now, so allocate a fresh buffer each call
    // for the first iteration and cache it. For simplicity, allocate here
    // through a global scratch owned by the model.
    // (This is not ideal — a proper fix adds a field to State.)

    // Allocate the [B, S*D] mHC mapping scratch inline. It'll be freed by
    // the OS at process exit; not ideal but acceptable for a single-session
    // sampler.
    extern float* g_mhc_mapping_scratch;
    if (g_mhc_mapping_scratch == nullptr) {
        g_mhc_mapping_scratch = dev_alloc_f((size_t)4096 * 4096);
    }

    return s;
}

float* g_mhc_mapping_scratch = nullptr;

void free_state(State& s) {
    auto free_f = [](float* p) { if (p) cudaFree(p); };
    auto free_i = [](int32_t* p) { if (p) cudaFree(p); };
    auto free_u = [](uint8_t* p) { if (p) cudaFree(p); };

    free_f(s.streams);
    free_f(s.loop_boundary_streams);
    for (auto* p : s.gdn_states) free_f(p);
    for (auto& c : s.csa_memories) {
        free_f(c.keys); free_f(c.values);
        free_i(c.positions); free_i(c.segment_ids);
        free_f(c.latents);
        free_f(c.pending_sum);
        free_i(c.pending_count); free_i(c.pending_seg);
    }
    for (auto& c : s.hca_memories) {
        free_f(c.keys); free_f(c.values);
        free_i(c.positions); free_i(c.segment_ids);
        free_f(c.latents);
        free_f(c.pending_sum);
        free_i(c.pending_count); free_i(c.pending_seg);
    }
    for (auto& h : s.histories) {
        free_f(h.conv_hist);
        free_f(h.router_latent_hist);
        free_i(h.router_hist_valid);
        free_f(h.local_k_cache);
        free_f(h.local_v_cache);
        free_u(h.local_valid);
    }
    free_f(s.loop_endpoint_storage);

    free_f(s.scratch_hidden);
    free_f(s.scratch_hidden2);
    free_f(s.scratch_update);
    free_f(s.scratch_mixed);
    free_f(s.scratch_q);
    free_f(s.scratch_k);
    free_f(s.scratch_v);
    free_f(s.scratch_gate);
    free_f(s.scratch_beta);
    free_f(s.scratch_decay);
    free_f(s.scratch_mhc_pre);
    free_f(s.scratch_mhc_post);
    free_f(s.scratch_mhc_residual);
    free_f(s.scratch_mhc_logits);
    free_f(s.scratch_router_logits);
    free_i(s.scratch_topk_idx);
    free_f(s.scratch_topk_prob);
    free_f(s.scratch_logits);
}

// ===========================================================================
// Internal forward helpers
// ===========================================================================

// ---------------------- mHC read/write composites -------------------------

// Full mHC "read": compute_mappings + read + LayerNorm.
// Writes hidden to out [B, D].
void mhc_read_full(const MHCWeights& mw,
                   const float* streams, int B, int S, int D,
                   float* pre_out, float* post_out, float* resid_out,
                   float* scratch_norm,         // [B, S*D]
                   float* hidden_out,           // [B, D]
                   int sinkhorn_iters,
                   cudaStream_t st) {
    mhc_compute_mappings(
        streams, mw.mapping_norm_w,
        mw.pre_w, mw.post_w, mw.residual_w,
        mw.pre_bias, mw.post_bias, mw.residual_bias,
        mw.alpha_pre, mw.alpha_post, mw.alpha_residual,
        pre_out, post_out, resid_out,
        B, S, D, sinkhorn_iters,
        scratch_norm, st);
    mhc_read(streams, pre_out,
             mw.stream_norm_w, mw.stream_norm_b,
             hidden_out, B, S, D, st);
}

// Full mHC "write". In-place update of streams.
void mhc_write_full(const MHCWeights& mw,
                    float* streams, const float* update,
                    const float* resid, const float* post,
                    int B, int S, int D,
                    cudaStream_t st) {
    mhc_write(streams, update, resid, post,
              mw.stream_scale, mw.gating_logit, B, S, D, st);
}

// ---------------------- GDN one-token path --------------------------------

// Applies q/k/v/beta/gate/decay projections, runs the recurrence step,
// applies the conv gate, projects back to D. Writes to out [B, D].
//
// All hidden states are [B, D]; our B=1 in decode.
void apply_gdn_block(const Block& b, const Config& cfg,
                     const float* hidden,           // [B, D]
                     float* gdn_state,              // [B, H, K, V]
                     float* conv_hist,              // [B, K-1, D]
                     // scratch
                     float* q_buf, float* k_buf, float* v_buf,
                     float* gate_buf, float* beta_buf, float* decay_buf,
                     float* mixed_out,               // [B, H*V]
                     float* update_out,              // [B, D]
                     int B,
                     cudaStream_t st) {
    const int D = cfg.d_model;
    const int H = cfg.gdn_qk_heads;
    const int K = cfg.gdn_qk_head_dim;
    const int V = cfg.gdn_value_head_dim;

    // q = W_q @ hidden, k = W_k @ hidden, v = W_v @ hidden
    // For B=1, use cuBLAS with a shared handle. But this file doesn't own
    // the handle; we cheat by using cudaMemcpy + a small GEMM via the
    // device's `__ldg`-friendly direct kernel. Actually, use plain matmul
    // via a tiny helper kernel launched per projection.
    //
    // For simplicity here we use a generic "linear" kernel defined below.

    // (Implementation of `linear_forward` is below in this file; forward
    // declaration here.)
    extern void linear_forward(const float* x, const float* w, float* y,
                               int rows, int in_features, int out_features,
                               cudaStream_t st);

    linear_forward(hidden, b.gdn.q_w, q_buf, B, D, H * K, st);
    linear_forward(hidden, b.gdn.k_w, k_buf, B, D, H * K, st);
    linear_forward(hidden, b.gdn.v_w, v_buf, B, D, H * V, st);
    linear_forward(hidden, b.gdn.beta_w, beta_buf, B, D, H, st);

    // beta = sigmoid(beta + bias); must add bias then sigmoid.
    // For B=1, we handle this with a tiny elementwise pass:
    //   beta[i] = sigmoid(beta[i] + beta_b[i % H])
    // We do it in a fused kernel below.
    extern void sigmoid_with_bias_(float* x, const float* bias,
                                   int B, int H, cudaStream_t st);
    sigmoid_with_bias_(beta_buf, b.gdn.beta_b, B, H, st);

    // gate = sigmoid(W_gate @ hidden)
    linear_forward(hidden, b.gdn.gate_w, gate_buf, B, D, H * V, st);
    extern void sigmoid_(float* x, int64_t n, cudaStream_t st);
    sigmoid_(gate_buf, (int64_t)B * H * V, st);

    // decay = sigmoid(decay_logit)  [H]
    extern void sigmoid_(const float*, int64_t, cudaStream_t);  // no
    extern void sigmoid_into(const float* src, float* dst, int n, cudaStream_t st);
    sigmoid_into(b.gdn.decay_logit, decay_buf, H, st);

    // Beta needs a final unsqueeze? No — beta_buf is already [B, H], and
    // gdn_step expects [H] for K and V, so we flatten.
    // For B=1: beta_buf has shape [H].
    //
    // Also, q/k need L2 normalization per head. The reference:
    //   q = q / max(||q||_2, eps)   per head over K dim
    // Same for k.
    extern void l2_normalize_heads_(float* q, int B, int H, int K, cudaStream_t st);
    l2_normalize_heads_(q_buf, B, H, K, st);
    l2_normalize_heads_(k_buf, B, H, K, st);

    // Run the recurrence step.
    // Our gdn_step kernel expects B=1 shape [H, K, V] and q [H, K] etc.
    // For general B, we need a loop. Since decode uses B=1, we handle B=1
    // here and bail for B>1 (fall back to host loop).
    if (B != 1) {
        throw std::runtime_error("apply_gdn_block: batch>1 not yet supported");
    }

    extern void gdn_step(const float*, const float*, const float*,
                         const float*, const float*, const float*,
                         float*, float*, int, int, int, cudaStream_t);
    gdn_step(q_buf, k_buf, v_buf, beta_buf, gate_buf, decay_buf,
             gdn_state, mixed_out, H, K, V, st);

    // Conv gate: 1-D depthwise convolution over the last (K_conv - 1) hidden
    // states plus the current. Then multiply into mixed_out.
    // Reference computes:
    //   conv_gate = self.conv(conv_input)            # [B, D, T]
    //   output = output * sigmoid(conv_gate)
    // We compute the gate on the [D] hidden vector then multiply.
    extern void gdn_conv_gate(const float*, const float*, const float*,
                              float*, int, int, cudaStream_t);
    // Note: conv.weight has shape [D, 1, Kconv]; our kernel expects [D, Kconv].
    // So use conv_w directly (the layout is contiguous [D, Kconv]).
    float* conv_gate_buf = v_buf;   // reuse as scratch
    gdn_conv_gate(conv_hist, hidden, b.gdn.conv_w, conv_gate_buf,
                  D, cfg.gdn_conv_kernel, st);
    extern void sigmoid_(float*, int64_t, cudaStream_t);
    sigmoid_(conv_gate_buf, D, st);

    // Roll conv history (must happen BEFORE we do out projection? Reference
    // rolls after computing the gate, which uses the current history. So we
    // roll AFTER the gate.)
    extern void gdn_conv_roll(float*, const float*, int, int, cudaStream_t);
    gdn_conv_roll(conv_hist, hidden, D, cfg.gdn_conv_kernel, st);

    // out_proj: [H*V] -> [D]. mixed_out currently holds the raw [H*V] output.
    linear_forward(mixed_out, b.gdn.out_w, update_out, B, H * V, D, st);

    // Apply gate: update_out[d] *= conv_gate[d]
    extern void mul_(float* x, const float* g, int64_t n, cudaStream_t st);
    mul_(update_out, conv_gate_buf, D, st);
}

// ---------------------- CSA one-token path --------------------------------

void apply_csa_block(const Block& b, const Config& cfg,
                     const float* hidden,           // [B, D]
                     MemoryCache& mem,
                     BlockHistory& hist,
                     // scratch
                     float* q_buf,                  // [B, H*Dh]
                     float* mixed_out,              // [B, D]
                     float* update_out,             // [B, D]
                     int B, int position, int32_t segment,
                     cudaStream_t st) {
    if (B != 1) throw std::runtime_error("apply_csa_block: batch>1 unsupported");

    const int D = cfg.d_model;
    const int H = cfg.csa_index_heads;
    const int Dh = cfg.csa_index_head_dim;
    const int Dk = H * Dh;
    const int W = std::max(1, cfg.csa_local_window - 1);
    const int top_k = cfg.csa_index_top_k;

    // q = W_q @ hidden
    extern void linear_forward(const float*, const float*, float*,
                               int, int, int, cudaStream_t);
    linear_forward(hidden, b.csa.query_w, q_buf, B, D, Dk, st);

    // --- Local attention ---
    // Compute the current token's k and v: k_new = W_k @ hidden, v_new = W_v @ hidden.
    extern void linear_forward(const float*, const float*, float*,
                               int, int, int, cudaStream_t);
    // Instead: allocate on the fly from a dedicated scratch. Since we don't
    // have one in the state, use a static buffer.
    extern float* g_csa_k_scratch;
    extern float* g_csa_v_scratch;
    extern float* g_csa_local_out_scratch;
    extern float* g_csa_remote_out_scratch;
    if (!g_csa_k_scratch) {
        cudaMalloc((void**)&g_csa_k_scratch, sizeof(float) * Dk);
        cudaMalloc((void**)&g_csa_v_scratch, sizeof(float) * D);
        cudaMalloc((void**)&g_csa_local_out_scratch, sizeof(float) * D);
        cudaMalloc((void**)&g_csa_remote_out_scratch, sizeof(float) * D);
    }
    linear_forward(hidden, b.csa.key_w, g_csa_k_scratch, B, D, Dk, st);
    linear_forward(hidden, b.csa.value_w, g_csa_v_scratch, B, D, D, st);

    // Roll the local K/V caches: shift left by one, write new at end.
    // Cache layout: [W, Dk] and [W, D].
    {
        extern void cache_roll_(float* cache, const float* new_row,
                                int W, int D_row, cudaStream_t st);
        cache_roll_(hist.local_k_cache, g_csa_k_scratch, W, Dk, st);
        cache_roll_(hist.local_v_cache, g_csa_v_scratch, W, D, st);
        // The cache contains the current token in its last row.  Only the
        // most recent `valid_count` rows are real; the leading rows are
        // padding until the local window fills.
        extern void local_valid_roll(uint8_t*, int, int, cudaStream_t);
        hist.local_valid_count = std::min(W, hist.local_valid_count + 1);
        local_valid_roll(hist.local_valid, W, hist.local_valid_count, st);
    }

    // Local attention: q [H, Dh] over cache [H, W, Dh] → out [D].
    // We need the cache to be laid out as [H, W, Dh], but our buffer is
    // [W, H*Dh]. That's actually the SAME layout if we interpret the second
    // dimension as interleaved heads. To make it [H, W, Dh], we'd need a
    // transpose. The CSA local path only needs q_h dotted with k_h[w];
    // we can implement it directly on the [W, H*Dh] layout.
    //
    // Reinterpret: k_cache [W, H*Dh] where the H*Dh is (h, d) packed.
    // q is [H*Dh]. score[h, w] = sum_d q[h*Dh + d] * k_cache[w, h*Dh + d].
    extern void csa_local_attn_decode(const float*, const float*, const float*,
                                      const uint8_t*, float*,
                                      int, int, int, int, cudaStream_t);
    csa_local_attn_decode(q_buf, hist.local_k_cache, hist.local_v_cache,
                          hist.local_valid, g_csa_local_out_scratch,
                          H, W, Dh, D, st);

    // --- Temporal compression step ---
    if (mem.pending_seg_host != segment) {
        mem.pending_seg_host = segment;
        mem.pending_count_host = 0;
    }
    ++mem.pending_count_host;
    const bool emitted = mem.pending_count_host >= cfg.csa_temporal_compression;
    const int32_t slot_length = (mem.length < mem.capacity)
        ? mem.length : mem.write_index;
    extern void csa_compress_step(const float*, float*, int32_t*, int32_t*,
                                  const float*, const float*,
                                  float*, float*, int32_t*, int32_t*, int32_t,
                                  int, int, int, int,
                                  int32_t, int32_t, int, cudaStream_t);
    csa_compress_step(hidden,
                      mem.pending_sum, mem.pending_count, mem.pending_seg,
                      b.csa.compressor.key_w, b.csa.compressor.value_w,
                      mem.keys, mem.values, mem.positions, mem.segment_ids,
                      slot_length,
                      D, mem.dk, mem.dv, mem.capacity,
                      position, segment, cfg.csa_temporal_compression, st);
    if (emitted) {
        mem.pending_count_host = 0;
        if (mem.length < mem.capacity) ++mem.length;
        else mem.write_index = (mem.write_index + 1) % mem.capacity;
    }
    // --- Remote attention ---
    extern void csa_remote_attn_decode(const float*, const float*, const float*,
                                       const int32_t*, int32_t, int32_t,
                                       float*, int, int, int, int, int,
                                       cudaStream_t);
    // The reference uses k=min(configured_top_k, visible_memory_length).
    // Passing the configured k when the cache has fewer rows makes the CUDA
    // selector manufacture -1 indices and then read them back as addresses.
    const int remote_k = std::min(top_k, mem.length);
    if (remote_k > 0) {
        csa_remote_attn_decode(q_buf, mem.keys, mem.values, mem.positions,
                               position, mem.length, g_csa_remote_out_scratch,
                               H, Dh, D, mem.capacity, remote_k, st);
    } else {
        extern void zero_(float*, int64_t, cudaStream_t);
        zero_(g_csa_remote_out_scratch, D, st);
    }

    // --- Combine local and remote outputs ---
    // local_out = W_local @ local_result
    // remote_out = W_remote @ remote_result
    // out = W_out @ (local_out + remote_out)
    // We do this in two steps: project each, add, project.
    float* local_proj = g_csa_local_out_scratch + 0;  // reuse
    float* remote_proj = g_csa_remote_out_scratch + 0;
    extern void linear_forward(const float*, const float*, float*,
                               int, int, int, cudaStream_t);
    // Note: we must NOT overwrite the input to linear_forward in place.
    // Use two extra scratch buffers.
    extern float* g_csa_local_proj_scratch;
    extern float* g_csa_remote_proj_scratch;
    if (!g_csa_local_proj_scratch) {
        cudaMalloc((void**)&g_csa_local_proj_scratch, sizeof(float) * D);
        cudaMalloc((void**)&g_csa_remote_proj_scratch, sizeof(float) * D);
    }
    linear_forward(local_proj, b.csa.local_out_w, g_csa_local_proj_scratch,
                   B, D, D, st);
    linear_forward(remote_proj, b.csa.remote_out_w, g_csa_remote_proj_scratch,
                   B, D, D, st);
    extern void add_(float* y, const float* x, int64_t n, cudaStream_t st);
    add_(g_csa_local_proj_scratch, g_csa_remote_proj_scratch, D, st);
    linear_forward(g_csa_local_proj_scratch, b.csa.out_w, update_out, B, D, D, st);

    (void)mixed_out;
}

// ---------------------- HCA one-token path --------------------------------

void apply_hca_block(const Block& b, const Config& cfg,
                     const float* hidden,
                     MemoryCache& mem,
                     BlockHistory& hist,
                     float* q_buf,
                     float* mixed_out,
                     float* update_out,
                     int B, int position, int32_t segment,
                     cudaStream_t st) {
    if (B != 1) throw std::runtime_error("apply_hca_block: batch>1 unsupported");

    const int D = cfg.d_model;
    const int W = std::max(1, cfg.hca_local_window - 1);
    const int latent = cfg.mla_latent_dim;

    extern void linear_forward(const float*, const float*, float*,
                               int, int, int, cudaStream_t);
    extern void cache_roll_(float*, const float*, int, int, cudaStream_t);
    extern void local_valid_roll(uint8_t*, int, int, cudaStream_t);
    extern void hca_local_attn_decode(const float*, const float*, const float*,
                                      const uint8_t*, float*, int, int,
                                      cudaStream_t);
    extern void hca_compress_step(const float*, float*, int32_t*, int32_t*,
                                  const float*, const float*, float*, float*,
                                  int32_t*, int32_t*, int32_t, int, int,
                                  int32_t, int32_t, int, cudaStream_t);
    extern void hca_global_attn_decode(const float*, const float*, const float*,
                                       const int32_t*, int32_t, int32_t,
                                       float*, int, int, cudaStream_t);
    extern void hca_global_topk_attn_decode(const float*, const float*, const float*,
                                            const int32_t*, int32_t, int32_t,
                                            float*, int, int, int, cudaStream_t);
    extern void layernorm_fwd(float*, const float*, const float*, int, int,
                              float, cudaStream_t);
    extern void copy_(float*, const float*, int64_t, cudaStream_t);
    extern void add_(float*, const float*, int64_t, cudaStream_t);

    extern float* g_hca_k_scratch;
    extern float* g_hca_v_scratch;
    extern float* g_hca_local_scratch;
    extern float* g_hca_global_scratch;
    extern float* g_hca_local_proj_scratch;
    extern float* g_hca_global_proj_scratch;
    extern float* g_hca_latent_scratch;
    if (!g_hca_k_scratch) {
        cudaMalloc((void**)&g_hca_k_scratch, sizeof(float) * D);
        cudaMalloc((void**)&g_hca_v_scratch, sizeof(float) * D);
        cudaMalloc((void**)&g_hca_local_scratch, sizeof(float) * D);
        cudaMalloc((void**)&g_hca_global_scratch, sizeof(float) * D);
        cudaMalloc((void**)&g_hca_local_proj_scratch, sizeof(float) * D);
        cudaMalloc((void**)&g_hca_global_proj_scratch, sizeof(float) * D);
        cudaMalloc((void**)&g_hca_latent_scratch, sizeof(float) * latent);
    }

    linear_forward(hidden, b.hca.query_w, q_buf, 1, D, D, st);
    linear_forward(hidden, b.hca.key_w, g_hca_k_scratch, 1, D, D, st);
    linear_forward(hidden, b.hca.value_w, g_hca_v_scratch, 1, D, D, st);

    cache_roll_(hist.local_k_cache, g_hca_k_scratch, W, D, st);
    cache_roll_(hist.local_v_cache, g_hca_v_scratch, W, D, st);
    hist.local_valid_count = std::min(W, hist.local_valid_count + 1);
    local_valid_roll(hist.local_valid, W, hist.local_valid_count, st);
    hca_local_attn_decode(q_buf, hist.local_k_cache, hist.local_v_cache,
                          hist.local_valid, g_hca_local_scratch, W, D, st);

    if (mem.pending_seg_host != segment) {
        mem.pending_seg_host = segment;
        mem.pending_count_host = 0;
    }
    ++mem.pending_count_host;
    const bool emitted = mem.pending_count_host >= cfg.hca_temporal_compression;
    const int32_t slot = (mem.length < mem.capacity)
        ? mem.length : mem.write_index;
    hca_compress_step(hidden,
                      mem.pending_sum, mem.pending_count, mem.pending_seg,
                      b.hca.compressor.key_w, b.hca.compressor.value_w,
                      mem.keys, mem.values, mem.positions, mem.segment_ids,
                      slot, D, mem.capacity, position, segment,
                      cfg.hca_temporal_compression, st);
    if (emitted) {
        mem.pending_count_host = 0;
        if (mem.length < mem.capacity) ++mem.length;
        else mem.write_index = (mem.write_index + 1) % mem.capacity;
    }

    // Encode the newest compressed-value slot for the following MoE router.
    linear_forward(mem.values + (size_t)slot * D, b.hca_latent.down_w,
                   g_hca_latent_scratch, 1, D, latent, st);
    layernorm_fwd(g_hca_latent_scratch, b.hca_latent.norm_w,
                  b.hca_latent.norm_b, 1, latent, 1e-5f, st);
    copy_(mem.latents + (size_t)slot * latent, g_hca_latent_scratch,
           latent, st);

    if (cfg.hca_dense_global_attention) {
        hca_global_attn_decode(q_buf, mem.keys, mem.values, mem.positions,
                               position, mem.length, g_hca_global_scratch,
                               D, mem.capacity, st);
    } else {
        hca_global_topk_attn_decode(q_buf, mem.keys, mem.values, mem.positions,
                                    position, mem.length, g_hca_global_scratch,
                                    D, mem.capacity, cfg.hca_global_top_k, st);
    }

    linear_forward(g_hca_local_scratch, b.hca.local_out_w,
                   g_hca_local_proj_scratch, 1, D, D, st);
    linear_forward(g_hca_global_scratch, b.hca.global_out_w,
                   g_hca_global_proj_scratch, 1, D, D, st);
    add_(g_hca_local_proj_scratch, g_hca_global_proj_scratch, D, st);
    linear_forward(g_hca_local_proj_scratch, b.hca.out_w,
                   update_out, 1, D, D, st);
    (void)mixed_out;
}

float* g_csa_k_scratch = nullptr;
float* g_csa_v_scratch = nullptr;
float* g_csa_local_out_scratch = nullptr;
float* g_csa_remote_out_scratch = nullptr;
float* g_csa_local_proj_scratch = nullptr;
float* g_csa_remote_proj_scratch = nullptr;
float* g_hca_k_scratch = nullptr;
float* g_hca_v_scratch = nullptr;
float* g_hca_local_scratch = nullptr;
float* g_hca_global_scratch = nullptr;
float* g_hca_local_proj_scratch = nullptr;
float* g_hca_global_proj_scratch = nullptr;
float* g_hca_latent_scratch = nullptr;

// ---------------------- Dense FFN one-token path --------------------------

void apply_dense_ffn(const DenseFFNWeights& w, const Config& cfg,
                     const float* hidden, float* out, int B, cudaStream_t st) {
    const int D = cfg.d_model;
    const int H = cfg.dense_ffn_hidden;
    // gate = W_gate @ hidden [B, H]
    // up   = W_up @ hidden
    // act  = silu(gate) * up
    // out  = W_down @ act
    extern void linear_forward(const float*, const float*, float*,
                               int, int, int, cudaStream_t);
    extern float* g_ffn_gate_scratch;
    extern float* g_ffn_up_scratch;
    if (!g_ffn_gate_scratch) {
        cudaMalloc((void**)&g_ffn_gate_scratch, sizeof(float) * 4096);
        cudaMalloc((void**)&g_ffn_up_scratch, sizeof(float) * 4096);
    }
    linear_forward(hidden, w.gate_w, g_ffn_gate_scratch, B, D, H, st);
    linear_forward(hidden, w.up_w,   g_ffn_up_scratch,   B, D, H, st);
    extern void silu_mul(const float*, const float*, float*, int64_t, cudaStream_t);
    silu_mul(g_ffn_gate_scratch, g_ffn_up_scratch, g_ffn_gate_scratch,
             (int64_t)B * H, st);
    linear_forward(g_ffn_gate_scratch, w.down_w, out, B, H, D, st);
}

float* g_ffn_gate_scratch = nullptr;
float* g_ffn_up_scratch = nullptr;

__global__ void k_router_hca_latent_attention(
    const float* q, const float* k, const float* v,
    const int32_t* positions, int32_t current_position,
    float* context, int heads, int head_dim, int capacity,
    int memory_length) {
    extern __shared__ float scores[];
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int width = heads * head_dim;
    const float* qh = q + head * head_dim;

    for (int i = tid; i < capacity; i += blockDim.x) {
        float score = -FLT_MAX;
        if (i < memory_length) {
            const int pos = positions[i];
            if (pos >= 0 && pos <= current_position) {
                const float* kh = k + (size_t)i * width + head * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) dot += qh[d] * kh[d];
                score = dot * rsqrtf((float)head_dim);
            }
        }
        scores[i] = score;
    }
    __syncthreads();

    if (tid == 0) {
        float max_score = -FLT_MAX;
        for (int i = 0; i < capacity; ++i) max_score = fmaxf(max_score, scores[i]);
        float sum = 0.0f;
        for (int i = 0; i < capacity; ++i) {
            scores[i] = __expf(scores[i] - max_score);
            sum += scores[i];
        }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int i = 0; i < capacity; ++i) scores[i] *= inv;
    }
    __syncthreads();

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < capacity; ++i)
            acc += scores[i] * v[(size_t)i * width + head * head_dim + d];
        context[head * head_dim + d] = acc;
    }
}

void router_hca_latent_attention(const float* query,
                                 const MemoryCache& memory,
                                 const LatentAttentionWeights& weights,
                                 const Config& cfg,
                                 float* out,
                                 cudaStream_t st) {
    if (!memory.latents || memory.capacity <= 0) {
        extern void zero_(float*, int64_t, cudaStream_t);
        zero_(out, cfg.d_model, st);
        return;
    }
    const int D = cfg.d_model;
    const int heads = cfg.router_query_heads;
    const int head_dim = cfg.router_query_head_dim;
    const int width = heads * head_dim;
    const int capacity = memory.capacity;
    extern float* g_moe_hca_q_scratch;
    extern float* g_moe_hca_k_scratch;
    extern float* g_moe_hca_v_scratch;
    extern float* g_moe_hca_context_scratch;
    if (!g_moe_hca_q_scratch) {
        cudaMalloc((void**)&g_moe_hca_q_scratch, sizeof(float) * width);
        cudaMalloc((void**)&g_moe_hca_k_scratch,
                   sizeof(float) * (size_t)capacity * width);
        cudaMalloc((void**)&g_moe_hca_v_scratch,
                   sizeof(float) * (size_t)capacity * width);
        cudaMalloc((void**)&g_moe_hca_context_scratch, sizeof(float) * width);
    }
    extern void linear_forward(const float*, const float*, float*,
                               int, int, int, cudaStream_t);
    linear_forward(query, weights.q_w, g_moe_hca_q_scratch,
                   1, D, width, st);
    linear_forward(memory.latents, weights.k_w, g_moe_hca_k_scratch,
                   capacity, cfg.mla_latent_dim, width, st);
    linear_forward(memory.latents, weights.v_w, g_moe_hca_v_scratch,
                   capacity, cfg.mla_latent_dim, width, st);
    k_router_hca_latent_attention<<<heads, 128,
                                    sizeof(float) * capacity, st>>>(
        g_moe_hca_q_scratch, g_moe_hca_k_scratch, g_moe_hca_v_scratch,
        memory.positions, 0x7fffffff, g_moe_hca_context_scratch,
        heads, head_dim, capacity, memory.length);
    linear_forward(g_moe_hca_context_scratch, weights.o_w, out,
                   1, width, D, st);
}

float* g_moe_hca_q_scratch = nullptr;
float* g_moe_hca_k_scratch = nullptr;
float* g_moe_hca_v_scratch = nullptr;
float* g_moe_hca_context_scratch = nullptr;

// ---------------------- MoE one-token path --------------------------------

void apply_moe(const MoEWeights& w, const Config& cfg,
               const float* hidden, float* out, int B,
               BlockHistory* hist, MemoryCache* hca_mem,
               cudaStream_t st) {
    if (B != 1) throw std::runtime_error("apply_moe: batch>1 unsupported");

    const int D = cfg.d_model;
    const int E = cfg.moe_routed_experts;
    const int top_k = cfg.moe_routed_top_k;
    const int H = cfg.moe_expert_hidden;

    extern float* g_moe_router_local;
    extern float* g_moe_router_latent_scratch;   // [latent]
    extern float* g_moe_router_context_latent_scratch; // [latent]
    extern float* g_moe_router_context_scratch;  // [D]
    extern float* g_moe_router_logits_scratch;   // [E]
    extern float* g_moe_expert_out_scratch;      // [D]
    extern float* g_moe_shared_out_scratch;      // [D]
    extern float* g_moe_routed_out_scratch;      // [D]
    extern int32_t* g_moe_topk_idx;
    extern float* g_moe_topk_prob;

    if (!g_moe_router_local) {
        cudaMalloc((void**)&g_moe_router_local, sizeof(float) * D);
        cudaMalloc((void**)&g_moe_router_latent_scratch, sizeof(float) * cfg.mla_latent_dim);
        cudaMalloc((void**)&g_moe_router_context_latent_scratch, sizeof(float) * cfg.mla_latent_dim);
        cudaMalloc((void**)&g_moe_router_context_scratch, sizeof(float) * D);
        cudaMalloc((void**)&g_moe_router_logits_scratch, sizeof(float) * E);
        cudaMalloc((void**)&g_moe_expert_out_scratch, sizeof(float) * D);
        cudaMalloc((void**)&g_moe_shared_out_scratch, sizeof(float) * D);
        cudaMalloc((void**)&g_moe_routed_out_scratch, sizeof(float) * D);
        cudaMalloc((void**)&g_moe_topk_idx, sizeof(int32_t) * top_k);
        cudaMalloc((void**)&g_moe_topk_prob, sizeof(float) * top_k);
    }

    extern void rmsnorm_fwd(const float*, const float*, float*,
                            int, int, float, cudaStream_t);
    extern void linear_forward(const float*, const float*, float*,
                               int, int, int, cudaStream_t);

    // local = RMSNorm(hidden)
    rmsnorm_fwd(hidden, w.router.norm_w, g_moe_router_local, 1, D, 1e-6f, st);

    // current_latent = norm(down(local))
    linear_forward(g_moe_router_local, w.router.latent_memory.down_w,
                   g_moe_router_latent_scratch, 1, D, cfg.mla_latent_dim, st);
    extern void layernorm_fwd(float*, const float*, const float*,
                              int, int, float, cudaStream_t);
    layernorm_fwd(g_moe_router_latent_scratch,
                  w.router.latent_memory.norm_w,
                  w.router.latent_memory.norm_b,
                  1, cfg.mla_latent_dim, 1e-5f, st);

    // The Python decode path averages the current latent with the valid
    // router-latent history, then appends the current latent to that history.
    // Keep the same ordering here.  This is not optional: using only the
    // current latent changes every recurrent MoE decision after token 1.
    extern void router_history_update_mean(const float*, float*, int32_t*,
                                           float*, int, int, cudaStream_t);
    router_history_update_mean(
        g_moe_router_latent_scratch, hist->router_latent_hist,
        hist->router_hist_valid, g_moe_router_context_latent_scratch,
        cfg.router_local_context, cfg.mla_latent_dim, st);

    linear_forward(g_moe_router_context_latent_scratch, w.router.local_ctx_w,
                   g_moe_router_context_scratch, 1, cfg.mla_latent_dim, D, st);

    // route_input = local + sigmoid(route_gate) * context_out(context)
    extern void scale_and_add_(float* y, const float* x, const float* scale,
                               int64_t n, cudaStream_t st);
    // First compute context_out(context)
    extern float* g_moe_router_ctxout_scratch;
    if (!g_moe_router_ctxout_scratch) {
        cudaMalloc((void**)&g_moe_router_ctxout_scratch, sizeof(float) * D);
    }
    linear_forward(g_moe_router_context_scratch, w.router.context_w,
                   g_moe_router_ctxout_scratch, 1, D, D, st);

    if (hca_mem != nullptr && hca_mem->latents != nullptr) {
        extern void add_(float* y, const float* x, int64_t n,
                         cudaStream_t st);
        extern void router_hca_latent_attention(
            const float*, const MemoryCache&, const LatentAttentionWeights&,
            const Config&, float*, cudaStream_t);
        extern float* g_moe_hca_router_out_scratch;
        if (!g_moe_hca_router_out_scratch) {
            cudaMalloc((void**)&g_moe_hca_router_out_scratch,
                       sizeof(float) * D);
        }
        router_hca_latent_attention(
            g_moe_router_local, *hca_mem, w.router.latent_attention,
            cfg, g_moe_hca_router_out_scratch, st);
        add_(g_moe_router_context_scratch,
             g_moe_hca_router_out_scratch, D, st);
        linear_forward(g_moe_router_context_scratch, w.router.context_w,
                       g_moe_router_ctxout_scratch, 1, D, D, st);
    }

    // We need a "route_gate" sigmoid applied. Let's compute it inline:
    extern void sigmoid_into(const float*, float*, int, cudaStream_t);
    extern float* g_moe_route_gate_sigmoid;
    if (!g_moe_route_gate_sigmoid) {
        cudaMalloc((void**)&g_moe_route_gate_sigmoid, sizeof(float));
    }
    sigmoid_into(w.router.route_gate, g_moe_route_gate_sigmoid, 1, st);

    // route_input = local + sigmoid(route_gate) * ctxout
    // We'll do: scale ctxout by sigmoid(gate), then add to local.
    extern void scale_(float* y, const float* x, const float* scale,
                       int64_t n, cudaStream_t);
    extern void add_(float* y, const float* x, int64_t n, cudaStream_t);
    scale_(g_moe_router_ctxout_scratch, g_moe_router_ctxout_scratch,
           g_moe_route_gate_sigmoid, D, st);
    add_(g_moe_router_local, g_moe_router_ctxout_scratch, D, st);
    // Now g_moe_router_local = route_input.

    // route_logits = logits_w @ route_input
    linear_forward(g_moe_router_local, w.router.logits_w,
                   g_moe_router_logits_scratch, 1, D, E, st);

    // top-k + softmax
    extern void router_topk(const float*, int, int, int32_t*, float*, cudaStream_t);
    router_topk(g_moe_router_logits_scratch, E, top_k,
                g_moe_topk_idx, g_moe_topk_prob, st);

    // Expert dispatch.
    extern void moe_route(const float*, const float*, const float*, const float*,
                          const int32_t*, const float*, float*, float*,
                          int, int, int, int, cudaStream_t);
    moe_route(hidden, w.experts_gate, w.experts_up, w.experts_down,
              g_moe_topk_idx, g_moe_topk_prob,
              g_moe_routed_out_scratch, g_moe_expert_out_scratch,
              D, H, E, top_k, st);

    // Shared experts: sum their forward passes.
    const int Ss = cfg.moe_shared_experts;
    // Zero the shared output first.
    extern void zero_(float*, int64_t, cudaStream_t);
    zero_(g_moe_shared_out_scratch, D, st);
    for (int s = 0; s < Ss; ++s) {
        const float* sg = w.shared_gate + (size_t)s * H * D;
        const float* su = w.shared_up   + (size_t)s * H * D;
        const float* sd = w.shared_down + (size_t)s * D * H;
        extern void moe_expert_fwd(const float*, const float*, const float*,
                                   const float*, float*, int, int, cudaStream_t);
        moe_expert_fwd(hidden, sg, su, sd, g_moe_expert_out_scratch, D, H, st);
        extern void add_(float* y, const float* x, int64_t n, cudaStream_t);
        add_(g_moe_shared_out_scratch, g_moe_expert_out_scratch, D, st);
    }
    // Divide by sqrt(shared_experts).
    extern void scale_scalar_(float* x, float s, int64_t n, cudaStream_t st);
    float inv_sqrt_shared = 1.0f / std::sqrt((float)Ss);
    scale_scalar_(g_moe_shared_out_scratch, inv_sqrt_shared, D, st);

    // Sum routed + shared.
    add_(g_moe_routed_out_scratch, g_moe_shared_out_scratch, D, st);
    extern void copy_(float* dst, const float* src, int64_t n, cudaStream_t st);
    copy_(out, g_moe_routed_out_scratch, D, st);
}

float* g_moe_router_local = nullptr;
float* g_moe_router_latent_scratch = nullptr;
float* g_moe_router_context_latent_scratch = nullptr;
float* g_moe_router_context_scratch = nullptr;
float* g_moe_router_logits_scratch = nullptr;
float* g_moe_router_ctxout_scratch = nullptr;
float* g_moe_route_gate_sigmoid = nullptr;
float* g_moe_hca_router_out_scratch = nullptr;
float* g_moe_expert_out_scratch = nullptr;
float* g_moe_shared_out_scratch = nullptr;
float* g_moe_routed_out_scratch = nullptr;
int32_t* g_moe_topk_idx = nullptr;
float* g_moe_topk_prob = nullptr;

// ---------------------- Small elementwise helpers -------------------------

// Roll the fixed-size router history and form the mean of the valid entries
// plus the current latent.  The reference implementation concatenates the
// old history and current_latent, takes the last local_context entries, masks
// invalid entries, and then averages.  In the native B=1 decode path the
// history is already a fixed ring-like array, so this kernel performs the
// equivalent operation without allocating or synchronizing with the host.
__global__ void k_router_history_update_mean(
    const float* current, float* history, int32_t* valid, float* mean,
    int context, int latent) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < latent) {
        float sum = current[d];
        int count = 1;
        for (int t = 0; t < context; ++t) {
            if (valid[t]) {
                sum += history[(size_t)t * latent + d];
                ++count;
            }
        }
        mean[d] = sum / (float)count;
    }
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int t = 0; t + 1 < context; ++t) {
            for (int d2 = 0; d2 < latent; ++d2)
                history[(size_t)t * latent + d2] =
                    history[(size_t)(t + 1) * latent + d2];
            valid[t] = valid[t + 1];
        }
        for (int d2 = 0; d2 < latent; ++d2)
            history[(size_t)(context - 1) * latent + d2] = current[d2];
        valid[context - 1] = 1;
    }
}

void router_history_update_mean(const float* current, float* history,
                                int32_t* valid, float* mean,
                                int context, int latent, cudaStream_t st) {
    int threads = 256;
    // The history update is deliberately one block: its barrier separates
    // the mean read from the in-place roll.
    k_router_history_update_mean<<<1, threads, 0, st>>>(
        current, history, valid, mean, context, latent);
}

// These are tiny scalar ops used above. Each is a single kernel launch.

__global__ void k_local_valid_roll(uint8_t* valid, int width, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) valid[i] = (i >= width - count) ? 1 : 0;
}

void local_valid_roll(uint8_t* valid, int width, int count, cudaStream_t st) {
    int threads = 128;
    int blocks = (width + threads - 1) / threads;
    k_local_valid_roll<<<blocks, threads, 0, st>>>(valid, width, count);
}

__global__ void k_zero(float* p, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.0f;
}

__global__ void k_copy(float* dst, const float* src, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

__global__ void k_add(float* y, const float* x, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

__global__ void k_mul(float* x, const float* g, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= g[i];
}

__global__ void k_scale(float* y, const float* x, const float* s, int64_t n) {
    float sv = s[0];
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * sv;
}

__global__ void k_scale_scalar(float* x, float s, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

__global__ void k_sigmoid(float* x, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 1.0f / (1.0f + __expf(-x[i]));
}

__global__ void k_sigmoid_into(const float* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = 1.0f / (1.0f + __expf(-src[i]));
}

__global__ void k_sigmoid_with_bias(float* x, const float* bias, int B, int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * H) return;
    int h = i % H;
    x[i] = 1.0f / (1.0f + __expf(-(x[i] + bias[h])));
}

__global__ void k_fill_u8(uint8_t* p, uint8_t v, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

void zero_(float* p, int64_t n, cudaStream_t st) {
    int t = 256;
    int64_t b = (n + t - 1) / t;
    if (b > 65535) b = 65535;
    k_zero<<<(int)b, t, 0, st>>>(p, n);
}
void copy_(float* dst, const float* src, int64_t n, cudaStream_t st) {
    int t = 256;
    int64_t b = (n + t - 1) / t;
    if (b > 65535) b = 65535;
    k_copy<<<(int)b, t, 0, st>>>(dst, src, n);
}
void add_(float* y, const float* x, int64_t n, cudaStream_t st) {
    int t = 256;
    int64_t b = (n + t - 1) / t;
    if (b > 65535) b = 65535;
    k_add<<<(int)b, t, 0, st>>>(y, x, n);
}
void mul_(float* x, const float* g, int64_t n, cudaStream_t st) {
    int t = 256;
    int64_t b = (n + t - 1) / t;
    if (b > 65535) b = 65535;
    k_mul<<<(int)b, t, 0, st>>>(x, g, n);
}
void sigmoid_(float* x, int64_t n, cudaStream_t st) {
    int t = 256;
    int64_t b = (n + t - 1) / t;
    if (b > 65535) b = 65535;
    k_sigmoid<<<(int)b, t, 0, st>>>(x, n);
}
void sigmoid_into(const float* src, float* dst, int n, cudaStream_t st) {
    int t = 128;
    int b = (n + t - 1) / t;
    k_sigmoid_into<<<b, t, 0, st>>>(src, dst, n);
}
void sigmoid_with_bias_(float* x, const float* bias, int B, int H, cudaStream_t st) {
    int total = B * H;
    int t = 128;
    int b = (total + t - 1) / t;
    k_sigmoid_with_bias<<<b, t, 0, st>>>(x, bias, B, H);
}
void fill_u8_(uint8_t* p, uint8_t v, int64_t n, cudaStream_t st) {
    int t = 256;
    int64_t b = (n + t - 1) / t;
    if (b > 65535) b = 65535;
    k_fill_u8<<<(int)b, t, 0, st>>>(p, v, n);
}
void scale_scalar_(float* x, float s, int64_t n, cudaStream_t st) {
    int t = 256;
    int64_t b = (n + t - 1) / t;
    if (b > 65535) b = 65535;
    k_scale_scalar<<<(int)b, t, 0, st>>>(x, s, n);
}

// L2 normalize q/k per head.
__global__ void k_l2_normalize_heads(float* x, int H, int K) {
    int h = blockIdx.x;
    extern __shared__ float sm[];
    float* sh = sm;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float v = x[h * K + k];
        sh[k] = v * v;
    }
    if (threadIdx.x >= K) sh[threadIdx.x] = 0.0f;
    __syncthreads();
    // reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    __shared__ float norm_val;
    if (threadIdx.x == 0) {
        norm_val = rsqrtf(fmaxf(sh[0], 1e-12f));
    }
    __syncthreads();
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        x[h * K + k] *= norm_val;
    }
}

void l2_normalize_heads_(float* q, int B, int H, int K, cudaStream_t st) {
    if (B != 1) return;  // only B=1 supported
    int threads = 128;
    if (K < 128) threads = 64;
    size_t shmem = sizeof(float) * threads;
    k_l2_normalize_heads<<<H, threads, shmem, st>>>(q, H, K);
}

// cache_roll_: shift cache left by one, write new_row at the end.
__global__ void k_cache_roll(float* cache, const float* new_row,
                             int W, int D_row) {
    // Each thread handles a (d) element across all W rows.
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D_row) return;
    for (int w = 0; w < W - 1; ++w) {
        cache[(size_t)w * D_row + d] = cache[(size_t)(w + 1) * D_row + d];
    }
    cache[(size_t)(W - 1) * D_row + d] = new_row[d];
}

void cache_roll_(float* cache, const float* new_row, int W, int D_row,
                 cudaStream_t st) {
    int t = 256;
    int b = (D_row + t - 1) / t;
    k_cache_roll<<<b, t, 0, st>>>(cache, new_row, W, D_row);
}

// Generic linear: y = W @ x. W is [out, in], x is [rows, in], y is [rows, out].
// For rows > 32, calls cuBLAS. For small rows (decode), uses a custom kernel.
__global__ void k_linear_gemv(const float* x, const float* W, float* y,
                              int in_features, int out_features) {
    // One block per output row chunk; each warp handles one output row.
    int o = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (o >= out_features) return;
    int lane = threadIdx.x & 31;
    const float* wo = W + (size_t)o * in_features;
    float acc = 0.0f;
    for (int i = lane; i < in_features; i += 32) {
        acc += wo[i] * x[i];
    }
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[o] = acc;
}

__global__ void k_linear_rows(const float* x, const float* W, float* y,
                              int rows, int in_features, int out_features) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * out_features;
    if (index >= total) return;
    const int row = index / out_features;
    const int out = index - row * out_features;
    const float* xr = x + (size_t)row * in_features;
    const float* wr = W + (size_t)out * in_features;
    float acc = 0.0f;
    for (int i = 0; i < in_features; ++i) acc += xr[i] * wr[i];
    y[(size_t)row * out_features + out] = acc;
}

void linear_forward(const float* x, const float* w, float* y,
                    int rows, int in_features, int out_features,
                    cudaStream_t st) {
    if (rows == 1) {
        int warps = 4;
        int threads = warps * 32;
        int blocks = (out_features + warps - 1) / warps;
        k_linear_gemv<<<blocks, threads, 0, st>>>(x, w, y, in_features, out_features);
    } else {
        // Decode uses tiny row counts (for example the fixed HCA latent
        // cache). Keep this path allocation-free and correct for arbitrary
        // row counts; large prefill GEMMs are outside the one-token engine.
        const int total = rows * out_features;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        k_linear_rows<<<blocks, threads, 0, st>>>(
            x, w, y, rows, in_features, out_features);
    }
}

// ---------------------- Halting head --------------------------------------

// halt_probabilities: [B, 1] = sigmoid(halt_w @ collapse(streams) + halt_b)
// For B=1 this is a scalar.
__global__ void k_halt(const float* collapsed,  // [D]
                       const float* halt_w,     // [1, D]
                       const float* halt_b,     // [1]
                       float* prob_out,         // [1]
                       int D, float temperature) {
    extern __shared__ float sm[];
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        sm[d] = halt_w[d] * collapsed[d];
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    __shared__ float total;
    if (threadIdx.x == 0) {
        float v = sm[0] + halt_b[0];
        total = 1.0f / (1.0f + __expf(-v / temperature));
    }
    __syncthreads();
    if (threadIdx.x == 0) prob_out[0] = total;
}

void compute_halt_probability(const Model& m, const float* collapsed,
                              float* prob_out, cudaStream_t st) {
    if (m.halt_w == nullptr) { prob_out[0] = 1.0f; return; }
    const int D = m.cfg.d_model;
    // The reduction buffer holds one value per hidden dimension, not one
    // value per thread.  Use a power-of-two block large enough for D and
    // allocate the full D-element shared buffer.
    int threads = 1;
    while (threads < D) threads <<= 1;
    if (threads > 1024) throw std::runtime_error("halting head dimension exceeds CUDA block size");
    size_t shmem = sizeof(float) * (size_t)D;
    k_halt<<<1, threads, shmem, st>>>(
        collapsed, m.halt_w, m.halt_b, prob_out, D,
        m.cfg.adaptive_halt_temperature);
}

// ---------------------------------------------------------------------------
// Loop-attention residual memory.  The reference model records a latent
// endpoint after each recurrent loop and injects attention over prior
// endpoints before the next loop.  These kernels are intentionally small:
// decode has at most four endpoints, so a single block is sufficient.
// ---------------------------------------------------------------------------

__global__ void loop_mean_kernel(const float* streams, float* mean,
                                 int S, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float v = 0.0f;
    for (int s = 0; s < S; ++s) v += streams[s * D + d];
    mean[d] = v / (float)S;
}

__global__ void loop_inject_kernel(float* streams, const float* endpoints,
                                   int endpoint_count, const float* q_w,
                                   const float* k_w, const float* v_w,
                                   const float* o_w, const float* norm_w,
                                   const float* norm_b, const float* gate_logit,
                                   int S, int D, int L, int heads, int head_dim) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int HD = heads * head_dim;
    float q[2048];
    float attended[2048];
    float normed[2048];
    float mean = 0.0f, var = 0.0f;
    for (int d = 0; d < D; ++d) {
        float x = 0.0f;
        for (int s = 0; s < S; ++s) x += streams[s * D + d];
        normed[d] = x / (float)S;
        mean += normed[d];
    }
    mean /= (float)D;
    for (int d = 0; d < D; ++d) { float x = normed[d] - mean; var += x * x; }
    float inv = rsqrtf(var / (float)D + 1e-5f);
    for (int d = 0; d < D; ++d) normed[d] = (normed[d] - mean) * inv * norm_w[d] + norm_b[d];
    for (int o = 0; o < HD; ++o) {
        float x = 0.0f;
        for (int d = 0; d < D; ++d) x += q_w[o * D + d] * normed[d];
        q[o] = x;
        attended[o] = 0.0f;
    }
    for (int h = 0; h < heads; ++h) {
        float weights[16];
        float denom = 0.0f;
        for (int e = 0; e < endpoint_count; ++e) {
            float score = 0.0f;
            const float* ep = endpoints + e * L;
            for (int d = 0; d < head_dim; ++d) {
                float kv = 0.0f;
                for (int z = 0; z < L; ++z) kv += k_w[(h * head_dim + d) * L + z] * ep[z];
                score += q[h * head_dim + d] * kv;
            }
            weights[e] = expf(score * rsqrtf((float)head_dim));
            denom += weights[e];
        }
        for (int e = 0; e < endpoint_count; ++e) {
            float w = weights[e] / fmaxf(denom, 1e-20f);
            const float* ep = endpoints + e * L;
            for (int d = 0; d < head_dim; ++d) {
                float vv = 0.0f;
                for (int z = 0; z < L; ++z) vv += v_w[(h * head_dim + d) * L + z] * ep[z];
                attended[h * head_dim + d] += w * vv;
            }
        }
    }
    for (int d = 0; d < D; ++d) {
        float x = 0.0f;
        for (int i = 0; i < HD; ++i) x += o_w[d * HD + i] * attended[i];
        float gate = 1.0f / (1.0f + expf(-gate_logit[0]));
        for (int s = 0; s < S; ++s) streams[s * D + d] += gate * x;
    }
}

extern void linear_forward(const float*, const float*, float*, int, int, int,
                           cudaStream_t);
extern void layernorm_fwd(float*, const float*, const float*, int, int,
                          float, cudaStream_t);

void loop_memory_record(const Model& m, State& s, int endpoint_index,
                        cudaStream_t st) {
    if (!m.loop_memory_enabled || endpoint_index >= (int)s.loop_endpoints.size()) return;
    const int D = m.cfg.d_model;
    loop_mean_kernel<<<(D + 255) / 256, 256, 0, st>>>(s.streams, s.scratch_hidden,
                                        m.cfg.mhc_streams, D);
    linear_forward(s.scratch_hidden, m.loop_mem_compressor_down,
                   s.loop_endpoints[endpoint_index], 1, D,
                   m.cfg.loop_attnres_latent_dim, st);
    layernorm_fwd(s.loop_endpoints[endpoint_index], m.loop_mem_norm_w,
                  m.loop_mem_norm_b, 1, m.cfg.loop_attnres_latent_dim,
                  1e-5f, st);
    if (std::getenv("LLMY_DEBUG_LOOP_MEMORY")) {
        cudaStreamSynchronize(st);
        float sample[8] = {};
        cudaMemcpy(sample, s.loop_endpoints[endpoint_index], sizeof(sample),
                   cudaMemcpyDeviceToHost);
        std::printf("loop endpoint %d:", endpoint_index);
        for (float x : sample) std::printf(" %.9g", x);
        std::printf("\n");
    }
}

void loop_memory_inject(const Model& m, State& s, cudaStream_t st) {
    if (!m.loop_memory_enabled || s.loop_endpoint_count <= 0) return;
    if (std::getenv("LLMY_DEBUG_LOOP_MEMORY")) {
        cudaStreamSynchronize(st);
        float sample[8] = {};
        cudaMemcpy(sample, s.streams, sizeof(sample), cudaMemcpyDeviceToHost);
        std::printf("streams before loop-memory inject:");
        for (float x : sample) std::printf(" %.9g", x);
        std::printf("\n");
    }
    loop_inject_kernel<<<1, 1, 0, st>>>(
        s.streams, s.loop_endpoint_storage, s.loop_endpoint_count,
        m.loop_mem_attention.q_w, m.loop_mem_attention.k_w,
        m.loop_mem_attention.v_w, m.loop_mem_attention.o_w,
        m.loop_mem_query_norm_w, m.loop_mem_query_norm_b,
        m.loop_mem_gating_logit, m.cfg.mhc_streams, m.cfg.d_model,
        m.cfg.loop_attnres_latent_dim, m.cfg.loop_attnres_heads,
        m.cfg.loop_attnres_head_dim);
    if (std::getenv("LLMY_DEBUG_LOOP_MEMORY")) {
        cudaStreamSynchronize(st);
        float sample[8] = {};
        cudaMemcpy(sample, s.streams, sizeof(sample), cudaMemcpyDeviceToHost);
        std::printf("streams after loop-memory inject:");
        for (float x : sample) std::printf(" %.9g", x);
        std::printf("\n");
    }
}

// ---------------------- Block dispatcher ----------------------------------

// Runs one block for one token. Updates streams in place.
// `state_key_index` is the index into gdn_states, csa_memories, histories.
void run_block(const Block& b, const Config& cfg,
               float* streams,                    // [B, S, D]
               int B,
               float* gdn_state,                  // [B, H, K, V]
               MemoryCache* csa_mem,
               MemoryCache* hca_mem,
               BlockHistory* hist,
               int position, int32_t segment,
               // big scratch block
               float* scratch_pre, float* scratch_post, float* scratch_resid,
               float* scratch_norm,               // [B, S*D]
               float* scratch_hidden, float* scratch_hidden2,
               float* scratch_mixed, float* scratch_update,
               float* q_buf, float* k_buf, float* v_buf,
               float* gate_buf, float* beta_buf, float* decay_buf,
               cudaStream_t st) {
    const int D = cfg.d_model;
    const int S = cfg.mhc_streams;

    // ---- First sublayer: mixer ----

    // mHC read + LayerNorm
    mhc_read_full(b.mhc, streams, B, S, D,
                  scratch_pre, scratch_post, scratch_resid,
                   scratch_norm, scratch_hidden,
                   cfg.mhc_sinkhorn_iterations, st);

    // mixer_norm
    extern void rmsnorm_fwd(const float*, const float*, float*,
                            int, int, float, cudaStream_t);
    rmsnorm_fwd(scratch_hidden, b.mixer_norm_w, scratch_hidden2, B, D, 1e-6f, st);

    // Dispatch by kind.
    if (b.kind == "gdn") {
        apply_gdn_block(b, cfg, scratch_hidden2, gdn_state, hist->conv_hist,
                        q_buf, k_buf, v_buf, gate_buf, beta_buf, decay_buf,
                        scratch_mixed, scratch_update, B, st);
    } else if (b.kind == "csa") {
        apply_csa_block(b, cfg, scratch_hidden2, *csa_mem, *hist,
                        q_buf, scratch_mixed, scratch_update,
                        B, position, segment, st);
    } else if (b.kind == "hca") {
        apply_hca_block(b, cfg, scratch_hidden2, *hca_mem, *hist,
                        q_buf, scratch_mixed, scratch_update,
                        B, position, segment, st);
    } else {
        throw std::runtime_error("unknown block kind: " + b.kind);
    }

    // mHC write for the mixer.
    mhc_write_full(b.mhc, streams, scratch_update,
                   scratch_resid, scratch_post, B, S, D, st);

    // ---- Second sublayer: FFN (MoE or dense) ----

    mhc_read_full(b.mhc, streams, B, S, D,
                  scratch_pre, scratch_post, scratch_resid,
                  scratch_norm, scratch_hidden, 5, st);

    rmsnorm_fwd(scratch_hidden, b.ffn_norm_w, scratch_hidden2, B, D, 1e-6f, st);

    if (b.with_moe) {
        apply_moe(b.moe, cfg, scratch_hidden2, scratch_update, B,
                  hist, hca_mem, st);
    } else {
        apply_dense_ffn(b.ffn, cfg, scratch_hidden2, scratch_update, B, st);
    }

    mhc_write_full(b.mhc, streams, scratch_update,
                   scratch_resid, scratch_post, B, S, D, st);

}

// ===========================================================================
// Public forward API
// ===========================================================================

// Runs the full forward pass for one token. This is the single-token path
// used by both prefill (looped) and decode.
void model_step(Model& m, State& s,
                int32_t token, int32_t position,
                int32_t num_loops,
                float* logits_out,
                bool write_logits) {
    const Config& cfg = m.cfg;
    const int D = cfg.d_model;
    const int S = cfg.mhc_streams;
    const int B = s.batch;
    const int SD = S * D;
    const int V = cfg.vocab_size;

    cudaStream_t st = 0;
    s.position = position;
    s.loop_endpoint_count = 0;
    // ---- 1. Embedding ----
    // We embed into a [D] buffer then broadcast into S streams + stream_seed.
    extern void embedding_lookup(const float*, int32_t, float*, int, cudaStream_t);
    embedding_lookup(m.embed_w, token, s.scratch_hidden, D, st);

    // streams[b, s, :] = hidden[b, :] + stream_seed[s, :]
    extern void broadcast_streams_(float* streams,
                                   const float* hidden,
                                   const float* seed,
                                   int B, int S, int D, cudaStream_t);
    broadcast_streams_(s.streams, s.scratch_hidden, m.stream_seed, B, S, D, st);

    // ---- 2. Stem stack ----
    for (size_t i = 0; i < m.stem.size(); ++i) {
        run_block(m.stem[i], cfg,
                  s.streams, B,
                  s.gdn_states[i],
                  nullptr, nullptr, &s.histories[i],
                  position, 0,
                  s.scratch_mhc_pre, s.scratch_mhc_post, s.scratch_mhc_residual,
                  g_mhc_mapping_scratch,
                  s.scratch_hidden, s.scratch_hidden2,
                  s.scratch_mixed, s.scratch_update,
                  s.scratch_q, s.scratch_k, s.scratch_v,
                  s.scratch_gate, s.scratch_beta, s.scratch_decay,
                  st);
    }

    // Preserve the post-stem boundary.  Unless explicitly configured
    // otherwise, the Python implementation starts every recurrent pass from
    // this same stream value.
    extern void copy_(float*, const float*, int64_t, cudaStream_t);
    copy_(s.loop_boundary_streams, s.streams, (int64_t)B * SD, st);

    // ---- 3. Recurrent loops ----
    int loops = num_loops > 0 ? num_loops : cfg.num_loops;
    if (loops > cfg.num_loops) loops = cfg.num_loops;
    const bool learned_halting = cfg.adaptive_enabled &&
                                  cfg.adaptive_learned_halting &&
                                  m.halt_w != nullptr && m.halt_b != nullptr;
    const int minimum_loops = learned_halting
        ? std::min(loops, std::max(1, cfg.adaptive_min_loops))
        : loops;
    if (learned_halting) {
        loops = std::min(loops, std::max(1, cfg.adaptive_max_loops));
    }
    int executed_loops = 0;
    for (int loop = 0; loop < loops; ++loop) {
        if (loop > 0 && !cfg.mhc_preserve_streams_across_loop_boundary) {
            copy_(s.streams, s.loop_boundary_streams, (int64_t)B * SD, st);
        }
        loop_memory_inject(m, s, st);
        for (size_t bi = 0; bi < m.core.size(); ++bi) {
            // Match Python's state namespace: recurrent_core block state is
            // distinct for each loop pass.
            size_t state_idx = m.stem.size() +
                               (size_t)loop * m.core.size() + bi;
            size_t cache_idx = (size_t)loop * m.core.size() + bi;
            MemoryCache* csa = (m.core[bi].kind == "csa")
                               ? &s.csa_memories[cache_idx] : nullptr;
            MemoryCache* hca = (m.core[bi].kind == "hca")
                               ? &s.hca_memories[cache_idx] : nullptr;
            run_block(m.core[bi], cfg,
                      s.streams, B,
                      s.gdn_states[state_idx],
                      csa, hca, &s.histories[state_idx],
                      position, 0,
                      s.scratch_mhc_pre, s.scratch_mhc_post, s.scratch_mhc_residual,
                      g_mhc_mapping_scratch,
                      s.scratch_hidden, s.scratch_hidden2,
                      s.scratch_mixed, s.scratch_update,
                      s.scratch_q, s.scratch_k, s.scratch_v,
                      s.scratch_gate, s.scratch_beta, s.scratch_decay,
                      st);
        }
        ++executed_loops;
        if (m.loop_memory_enabled) {
            loop_memory_record(m, s, s.loop_endpoint_count, st);
            ++s.loop_endpoint_count;
        }

        // The native B=1 path can make the same eager halting decision as the
        // reference model after each completed loop. The decision is made
        // from the post-loop streams, so no state rollback is needed: the
        // current iteration is retained and only later iterations are skipped.
        if (learned_halting && loop + 1 < loops && loop + 1 >= minimum_loops) {
            linear_forward(s.streams, m.mhc_collapse_w, s.scratch_hidden,
                           B, SD, D, st);
            float* halt_prob_dev = s.scratch_beta;
            compute_halt_probability(m, s.scratch_hidden, halt_prob_dev, st);
            float halt_prob = 1.0f;
            cudaMemcpyAsync(&halt_prob, halt_prob_dev, sizeof(float),
                            cudaMemcpyDeviceToHost, st);
            cudaStreamSynchronize(st);
            if (!(halt_prob >= cfg.adaptive_halt_threshold)) break;
        }
    }
    s.last_executed_loops = executed_loops;

    // ---- 4. Tail stack ----
    for (size_t i = 0; i < m.tail.size(); ++i) {
        size_t state_idx = m.stem.size() +
                           (size_t)std::max(1, cfg.num_loops) * m.core.size() + i;
        run_block(m.tail[i], cfg,
                  s.streams, B,
                  s.gdn_states[state_idx],
                  nullptr, nullptr, &s.histories[state_idx],
                  position, 0,
                  s.scratch_mhc_pre, s.scratch_mhc_post, s.scratch_mhc_residual,
                  g_mhc_mapping_scratch,
                  s.scratch_hidden, s.scratch_hidden2,
                  s.scratch_mixed, s.scratch_update,
                  s.scratch_q, s.scratch_k, s.scratch_v,
                  s.scratch_gate, s.scratch_beta, s.scratch_decay,
                  st);
    }

    // ---- 5. Collapse streams -> hidden ----
    // collapsed[b, d] = sum_{s, d'} mhc_collapse_w[d, s*D + d'] * streams[b, s, d']
    extern void linear_forward(const float*, const float*, float*,
                               int, int, int, cudaStream_t);
    // streams has shape [B, S*D]; mhc_collapse_w is [D, S*D].
    linear_forward(s.streams, m.mhc_collapse_w, s.scratch_hidden,
                   B, SD, D, st);

    // ---- 6. Final norm ----
    extern void rmsnorm_fwd(const float*, const float*, float*,
                            int, int, float, cudaStream_t);
    rmsnorm_fwd(s.scratch_hidden, m.final_norm_w, s.scratch_hidden2,
                B, D, 1e-6f, st);
    if (!write_logits) {
        s.position = position;
        return;
    }

    // ---- 7. LM head ----
    extern void lm_head(const float*, const float*, float*, int, int,
                        cublasHandle_t, cudaStream_t);
    lm_head(s.scratch_hidden2, m.embed_w, s.scratch_logits, V, D, m.blas, st);

    // Copy logits back to host.
    cudaMemcpyAsync(logits_out, s.scratch_logits, V * sizeof(float),
                    cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);

    s.position = position;
}

// broadcast_streams kernel: streams[b,s,:] = hidden[b,:] + seed[s,:]
__global__ void k_broadcast_streams(float* streams,
                                    const float* hidden,
                                    const float* seed,
                                    int B, int S, int D) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int d = blockIdx.z * blockDim.x + threadIdx.x;
    if (d >= D) return;
    streams[((size_t)b * S + s) * D + d] = hidden[b * D + d] + seed[s * D + d];
}

void broadcast_streams_(float* streams, const float* hidden, const float* seed,
                        int B, int S, int D, cudaStream_t st) {
    dim3 grid(B, S, (D + 255) / 256);
    k_broadcast_streams<<<grid, 256, 0, st>>>(streams, hidden, seed, B, S, D);
}

void model_prefill(Model& m, State& s,
                   const int32_t* tokens, int n_tokens,
                   float* logits_out) {
    // Token-by-token prefill. This is correct but slower than a batched
    // prefill; the batched version would use the T>1 kernels from Part 2.
    // For the sampler use case (short prompts), the difference is small.
    if (n_tokens <= 0) return;
    for (int i = 0; i < n_tokens; ++i) {
        bool last = (i == n_tokens - 1);
        model_step(m, s, tokens[i], i, m.cfg.num_loops, logits_out, last);
    }
}

void model_decode(Model& m, State& s,
                  int32_t token, int32_t position, int32_t num_loops,
                  float* logits_out) {
    model_step(m, s, token, position, num_loops, logits_out, true);
}

}  // namespace llmy
