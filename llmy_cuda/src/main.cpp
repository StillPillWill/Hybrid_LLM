// Standalone CLI for the llmy engine. Uses a minimal built-in BPE
// tokenizer that reads a Qwen-style tokenizer.json. If no tokenizer is
// given, falls back to a "space-split" toy tokenizer for smoke testing.

#include "llmy.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

// ---------------------------------------------------------------------------
// Minimal CLI tokenizer: splits on whitespace and maps each word to its
// byte-sum hash mod vocab. This is NOT the real Qwen tokenizer — it's a
// placeholder so the CLI can run without a tokenizer file. For actual use
// call the engine from Python (see python/sample.py).
// ---------------------------------------------------------------------------

struct ToyTokenizer {
    int vocab;
    std::vector<int32_t> encode(const std::string& text) const {
        std::vector<int32_t> ids;
        std::istringstream in(text);
        std::string w;
        while (in >> w) {
            uint32_t h = 2166136261u;
            for (char c : w) {
                h ^= (uint8_t)c;
                h *= 16777619u;
            }
            ids.push_back((int32_t)(h % (uint32_t)vocab));
        }
        return ids;
    }
    std::string decode(const std::vector<int32_t>& ids) const {
        std::ostringstream out;
        for (auto id : ids) out << id << " ";
        return out.str();
    }
    int32_t eos = -1;
};

static void usage(const char* argv0) {
    fprintf(stderr,
        "usage: %s --model MODEL.llmy [options]\n"
        "  -m, --model PATH     path to converted .llmy checkpoint\n"
        "  -p, --prompt TEXT    prompt to generate from\n"
        "      --token-ids CSV  pre-tokenized IDs from the real tokenizer\n"
        "  -n, --tokens N       max tokens to generate (default 32)\n"
        "      --temp F         sampling temperature (default 0.7)\n"
        "      --top-k N        top-k filter (default 50)\n"
        "      --top-p F        top-p (default 0.9)\n"
        "      --rep F          repetition penalty (default 1.1)\n"
        "      --loops N        num recurrent loops (0 = model default)\n"
        "      --fixed-loops N  disable learned halting and run exactly N loops\n"
        "      --greedy         choose the highest-logit token\n"
        "      --top-logits K   print the K largest final logits\n"
        "      --seed N         rng seed\n",
        argv0);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string model_path;
    std::string prompt;
    std::string token_ids_csv;
    int n_tokens = 32;
    float temp = 0.7f;
    int top_k = 50;
    float top_p = 0.9f;
    float rep = 1.1f;
    int loops = 0;
    int fixed_loops = 0;
    int top_logits = 0;
    bool greedy = false;
    uint64_t seed = 1337;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { usage(argv[0]); exit(1); }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") model_path = next();
        else if (a == "-p" || a == "--prompt") prompt = next();
        else if (a == "--token-ids") token_ids_csv = next();
        else if (a == "-n" || a == "--tokens") n_tokens = atoi(next());
        else if (a == "--temp") temp = atof(next());
        else if (a == "--top-k") top_k = atoi(next());
        else if (a == "--top-p") top_p = atof(next());
        else if (a == "--rep") rep = atof(next());
        else if (a == "--loops") loops = atoi(next());
        else if (a == "--fixed-loops") fixed_loops = atoi(next());
        else if (a == "--greedy") greedy = true;
        else if (a == "--top-logits") top_logits = atoi(next());
        else if (a == "--seed") seed = (uint64_t)strtoull(next(), nullptr, 10);
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return 1; }
    }

    if (model_path.empty()) {
        fprintf(stderr, "error: --model required\n");
        usage(argv[0]);
        return 1;
    }

    printf("loading %s ...\n", model_path.c_str());
    llmy_model* model = llmy_load(model_path.c_str());
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    int vocab = llmy_vocab_size(model);
    int d_model = llmy_d_model(model);
    int default_loops = llmy_num_loops(model);
    int max_loops = llmy_max_loops(model);
    printf("model loaded: vocab=%d, d_model=%d, num_loops=%d, max_loops=%d\n",
           vocab, d_model, default_loops, max_loops);
    if (loops > 0) llmy_set_num_loops(model, loops);
    if (fixed_loops > 0) llmy_set_fixed_loops(model, fixed_loops);
    if (greedy) {
        temp = 0.0f;
        top_k = 0;
        top_p = 1.0f;
        rep = 1.0f;
    }

    if (prompt.empty() && token_ids_csv.empty()) prompt = "Once upon a time";

    ToyTokenizer tok;
    tok.vocab = vocab;
    std::vector<int32_t> ids;
    if (!token_ids_csv.empty()) {
        std::stringstream ss(token_ids_csv);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) ids.push_back((int32_t)std::stol(item));
        }
    } else {
        ids = tok.encode(prompt);
    }
    printf("prompt tokens (%zu): ", ids.size());
    for (auto id : ids) printf("%d ", id);
    printf("\n");

    llmy_state* state = llmy_state_new(model);
    if (!state) { fprintf(stderr, "state allocation failed\n"); return 1; }

    std::vector<float> logits(vocab);

    // Prefill.
    printf("prefilling %zu tokens ...\n", ids.size());
    auto t0 = std::chrono::steady_clock::now();
    llmy_prefill(model, state, ids.data(), (int)ids.size(), logits.data());
    int finite_logits = 0;
    float logit_min = 0.0f, logit_max = 0.0f;
    for (float v : logits) {
        if (std::isfinite(v)) {
            ++finite_logits;
            if (finite_logits == 1) logit_min = logit_max = v;
            else { logit_min = std::min(logit_min, v); logit_max = std::max(logit_max, v); }
        }
    }
    printf("logits: finite=%d/%d range=[%.5g, %.5g]\n",
           finite_logits, vocab, logit_min, logit_max);
    printf("adaptive loops: %d\n", llmy_last_executed_loops(state));
    if (top_logits > 0) {
        std::vector<int> order(vocab);
        std::iota(order.begin(), order.end(), 0);
        top_logits = std::min(top_logits, vocab);
        std::partial_sort(order.begin(), order.begin() + top_logits, order.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });
        printf("top logits:");
        for (int j = 0; j < top_logits; ++j)
            printf(" %d=%.9g", order[j], logits[order[j]]);
        printf("\n");
    }
    auto t1 = std::chrono::steady_clock::now();
    double prefill_s = std::chrono::duration<double>(t1 - t0).count();
    printf("prefill: %.2fs (%.2f tok/s)\n", prefill_s,
           ids.size() / std::max(prefill_s, 1e-6));

    // Sampling loop.
    printf("\ngenerating %d tokens ...\n", n_tokens);
    std::vector<int32_t> generated;
    auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < n_tokens; ++i) {
        int32_t next = llmy_sample(logits.data(), vocab,
                                   temp, top_k, top_p, rep,
                                   generated.data(), (int)generated.size(),
                                   &seed);
        generated.push_back(next);
        printf("%d ", next);
        fflush(stdout);

        if (i + 1 < n_tokens) {
            int pos = (int)ids.size() + i;
            llmy_decode(model, state, next, pos, loops, logits.data());
            printf(" [loops=%d]", llmy_last_executed_loops(state));
            if (top_logits > 0) {
                std::vector<int> order(vocab);
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(order.begin(), order.begin() + top_logits, order.end(),
                                  [&](int a, int b) { return logits[a] > logits[b]; });
                printf("\nnext top logits:");
                for (int j = 0; j < top_logits; ++j)
                    printf(" %d=%.9g", order[j], logits[order[j]]);
            }
        }
    }
    auto t3 = std::chrono::steady_clock::now();
    double gen_s = std::chrono::duration<double>(t3 - t2).count();
    printf("\n");

    printf("generation: %d tokens in %.2fs (%.2f tok/s)\n",
           n_tokens, gen_s, n_tokens / std::max(gen_s, 1e-6));

    printf("GENERATED_IDS:");
    for (auto id : generated) printf(" %d", id);
    printf("\n");

    llmy_state_free(state);
    llmy_free(model);
    return 0;
}
