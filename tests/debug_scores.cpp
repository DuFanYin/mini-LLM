#include "model/kv_cache.h"
#include "model/model_types.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <print>
#include <vector>

using namespace model;

int main() {
    ModelConfig cfg;
    cfg.d_model = 4;
    cfg.num_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.head_dim = 2;
    cfg.d_ff = 8;

    model::KVCacheConfig kcfg;
    kcfg.num_layers = 1;
    kcfg.num_kv_heads = 1;
    kcfg.max_seq_len = 16;
    kcfg.head_dim = 2;
    kcfg.page_size = 4;
    model::KVCache cache(kcfg);

    ForwardInput in;
    in.seq_len = 2;
    in.layer_id = 0;
    in.use_cache = false;
    in.causal = true;

    std::vector<float> q = {
        1.0f, 0.0f, 0.8f, 0.2f,
        0.6f, 0.4f, 0.5f, 0.5f
    };
    std::vector<float> k = {
        1.0f, 0.0f,
        0.0f, 1.0f
    };
    std::vector<float> v = {
        2.0f, 1.0f,
        -1.0f, 3.0f
    };

    const size_t S = in.seq_len;
    const size_t T = 2;
    const size_t Hq = cfg.num_heads;
    const size_t Hkv = cfg.num_kv_heads;
    const size_t D = cfg.head_dim;
    const size_t group = Hq / Hkv;
    const float inv = 1.0f / std::sqrt(static_cast<float>(D));

    float score_min = std::numeric_limits<float>::infinity();
    float score_max = -std::numeric_limits<float>::infinity();
    float prob_min = std::numeric_limits<float>::infinity();
    float prob_max = -std::numeric_limits<float>::infinity();

    auto idx3 = [](size_t a, size_t b, size_t c, size_t B, size_t C) {
        return (a * B + b) * C + c;
    };

    std::vector<float> logits(T, 0.0f), probs(T, 0.0f);
    for (size_t si = 0; si < S; ++si) {
        size_t visible = std::min(si + 1, T);
        for (size_t qh = 0; qh < Hq; ++qh) {
            size_t kh = qh / group;
            float max_l = -std::numeric_limits<float>::infinity();
            for (size_t ti = 0; ti < T; ++ti) {
                float l = -std::numeric_limits<float>::infinity();
                if (ti < visible) {
                    float dot = 0.0f;
                    for (size_t d = 0; d < D; ++d) {
                        dot += q[idx3(si, qh, d, Hq, D)] * k[idx3(ti, kh, d, Hkv, D)];
                    }
                    l = dot * inv;
                }
                logits[ti] = l;
                if (l > max_l) max_l = l;
                score_min = std::min(score_min, l);
                score_max = std::max(score_max, l);
            }

            float denom = 0.0f;
            for (size_t ti = 0; ti < T; ++ti) {
                float e = std::exp(logits[ti] - max_l);
                probs[ti] = e;
                denom += e;
            }
            for (size_t ti = 0; ti < T; ++ti) {
                float p = probs[ti] / denom;
                prob_min = std::min(prob_min, p);
                prob_max = std::max(prob_max, p);
            }
        }
    }

    std::println("attention score shape: [{}, {}, {}]", Hq, S, T);
    std::println("attention prob  shape: [{}, {}, {}]", Hq, S, T);
    std::println("score range: [{}, {}]", score_min, score_max);
    std::println("prob  range: [{}, {}]", prob_min, prob_max);
    return 0;
}
