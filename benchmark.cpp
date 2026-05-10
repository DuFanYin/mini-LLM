// Micro-benchmarks grouped by inference/train with sub-sections.

#include "engine/decode.h"
#include "engine/embedding.h"
#include "kernel/kernel.h"
#include "model/mini_llm.h"
#include "task/task.h"
#include "train/train.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <print>
#include <span>
#include <vector>

using namespace engine;
using namespace train;

namespace {

using steady_clock = std::chrono::steady_clock;

template <class Fn>
double mean_ns(Fn&& fn, size_t warmup, size_t iters) {
    for (size_t i = 0; i < warmup; ++i) {
        fn();
    }
    const auto t0 = steady_clock::now();
    for (size_t i = 0; i < iters; ++i) {
        fn();
    }
    const auto t1 = steady_clock::now();
    const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return static_cast<double>(ns) / static_cast<double>(iters);
}

template <class SetupFn, class Fn>
double mean_ns_with_setup(SetupFn&& setup, Fn&& fn, size_t warmup, size_t iters) {
    for (size_t i = 0; i < warmup; ++i) {
        setup();
        fn();
    }
    int64_t total_ns = 0;
    for (size_t i = 0; i < iters; ++i) {
        setup();
        const auto t0 = steady_clock::now();
        fn();
        const auto t1 = steady_clock::now();
        total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    }
    return static_cast<double>(total_ns) / static_cast<double>(iters);
}

void print_section(const char* title) {
    std::println("");
    std::println("=== {} ===", title);
}

void print_subsection(const char* title) {
    std::println("-- {} --", title);
}

void print_metric(const char* name, double ns_per_iter) {
    const double us_per_iter = ns_per_iter / 1000.0;
    std::println("{:<44} {:>12.2f} us/iter", name, us_per_iter);
}

void print_metric_per_token(const char* name, double ns_per_token) {
    const double us_per_token = ns_per_token / 1000.0;
    std::println("{:<44} {:>12.2f} us/token", name, us_per_token);
}

size_t parse_size(const char* s, size_t fallback) {
    if (s == nullptr || s[0] == '\0') return fallback;
    return static_cast<size_t>(std::strtoull(s, nullptr, 10));
}

} // namespace

int main(int argc, char** argv) {
    size_t warmup = 8;
    size_t iters = 50;
    size_t train_iters = 30;
    size_t train_warmup = 1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = parse_size(argv[++i], warmup);
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = parse_size(argv[++i], iters);
        } else if (std::strcmp(argv[i], "--train-iters") == 0 && i + 1 < argc) {
            train_iters = parse_size(argv[++i], train_iters);
        } else if (std::strcmp(argv[i], "--train-warmup") == 0 && i + 1 < argc) {
            train_warmup = parse_size(argv[++i], train_warmup);
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::println("usage: benchmark [--warmup N] [--iters N] [--train-iters N] [--train-warmup N]");
            std::println("  Prints grouped benchmark report: Inference + Train.");
            return 0;
        }
    }

    std::println("benchmark config: warmup={} iters={} train_warmup={} train_iters={}", warmup, iters, train_warmup,
                 train_iters);

    constexpr size_t k_vocab = 32;
    constexpr size_t k_num_layers = 4;
    constexpr uint32_t k_model_seed = 42;
    std::unique_ptr<model::MiniLlm> model = std::make_unique<model::MiniLlm>(
        model::MiniLlm::build(/*vocab_size=*/k_vocab, /*seed=*/k_model_seed, /*num_layers=*/k_num_layers));
    const model::ModelConfig& cfg = model->config();
    model::ModelWeights adam_m = clone_model(model->weights());
    model::ModelWeights adam_v = clone_model(model->weights());

    constexpr size_t k_prefill_len = 128;
    constexpr size_t k_decode_prompt_len = 127;
    constexpr size_t k_decode_steps = 32;
    constexpr size_t k_bench_cache_len = 4096;
    model->configure_cache(k_bench_cache_len);
    std::vector<uint32_t> tok_prefill(k_prefill_len);
    std::vector<uint32_t> tok_decode_prompt(k_decode_prompt_len);
    const std::vector<uint32_t> tok_train = {
        0u, 1u, 26u, 3u, 4u, 27u, 5u, 6u, 28u, 7u, 8u, 29u, 9u, 10u, 30u, 3u, 4u};
    for (size_t i = 0; i < tok_prefill.size(); ++i) {
        tok_prefill[i] = static_cast<uint32_t>((i * 7 + 3) % k_vocab);
    }
    for (size_t i = 0; i < tok_decode_prompt.size(); ++i) {
        tok_decode_prompt[i] = tok_prefill[i];
    }
    const uint32_t decode_token = tok_prefill.back();

    const size_t d_model = model->d_model();
    const size_t seq_len_prefill = tok_prefill.size();
    const size_t seq_len_decode_prompt = tok_decode_prompt.size();
    std::println(
        "benchmark context: vocab={} layers={} d_model={} d_ff={} heads={} kv_heads={} head_dim={} prefill_len={} "
        "decode_prompt_len={}",
        k_vocab, model->num_layers(), cfg.d_model, cfg.d_ff, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim,
        seq_len_prefill, seq_len_decode_prompt);
    const model::DecoderLayerWeights& w0 = model->weights().layers[0];
    std::vector<float> x7(seq_len_prefill * cfg.d_model, 0.02f);
    std::vector<float> x7_norm(seq_len_prefill * cfg.d_model, 0.0f);
    const kernel::RmsNormParams norm_params{seq_len_prefill, cfg.d_model};
    kernel::rms_norm(x7.data(), w0.norm1.weight.data(), w0.norm1.eps, x7_norm.data(), norm_params);

    const size_t q_proj_dim = cfg.num_heads * cfg.head_dim;
    const kernel::LinearParams q_linear_params{seq_len_prefill, cfg.d_model, q_proj_dim};
    std::vector<float> q_bench_out(seq_len_prefill * q_proj_dim, 0.0f);

    const size_t hq = cfg.num_heads;
    const size_t hkv = cfg.num_kv_heads;
    const size_t dh = cfg.head_dim;
    std::vector<float> q(seq_len_prefill * hq * dh, 0.03f);
    std::vector<float> q_rope(seq_len_prefill * hq * dh, 0.03f);
    std::vector<size_t> rope_positions(seq_len_prefill, 0);
    for (size_t i = 0; i < rope_positions.size(); ++i) {
        rope_positions[i] = i;
    }
    kernel::RopeCache rope_cache(cfg.rope_base, cfg.head_dim, cfg.rope_dim);
    const kernel::RopeParams rope_params{seq_len_prefill, hq, dh};
    std::vector<float> k_all(seq_len_prefill * hkv * dh, 0.04f);
    std::vector<float> v_all(seq_len_prefill * hkv * dh, 0.05f);
    std::vector<float> ctx_bench(seq_len_prefill * hq * dh, 0.0f);
    std::vector<float> add_a(seq_len_prefill * cfg.d_model, 0.02f);
    std::vector<float> add_b(seq_len_prefill * cfg.d_model, 0.03f);
    std::vector<float> add_out(seq_len_prefill * cfg.d_model, 0.0f);
    const kernel::AddParams add_params{add_out.size()};
    kernel::AttentionParams attention_bench_params;
    attention_bench_params.seq_len = seq_len_prefill;
    attention_bench_params.num_heads = hq;
    attention_bench_params.num_kv_heads = hkv;
    attention_bench_params.head_dim = dh;
    attention_bench_params.past_len = 0;
    attention_bench_params.total_kv_len = seq_len_prefill;
    attention_bench_params.causal = true;
    attention_bench_params.use_cache = true;
    std::vector<float> q_decode(hq * dh, 0.03f);
    std::vector<float> k_decode_all(seq_len_prefill * hkv * dh, 0.04f);
    std::vector<float> v_decode_all(seq_len_prefill * hkv * dh, 0.05f);
    std::vector<float> ctx_decode(hq * dh, 0.0f);
    kernel::AttentionParams attention_decode_params;
    attention_decode_params.seq_len = 1;
    attention_decode_params.num_heads = hq;
    attention_decode_params.num_kv_heads = hkv;
    attention_decode_params.head_dim = dh;
    attention_decode_params.past_len = seq_len_prefill - 1;
    attention_decode_params.total_kv_len = seq_len_prefill;
    attention_decode_params.causal = true;
    attention_decode_params.use_cache = true;
    std::vector<float> softmax_logits(seq_len_prefill, 0.01f);
    std::vector<float> softmax_probs(seq_len_prefill, 0.0f);
    const kernel::SoftmaxParams softmax_params{seq_len_prefill};

    std::vector<float> mlp_down_out(seq_len_prefill * cfg.d_model, 0.0f);
    std::vector<float> gate_buf(seq_len_prefill * cfg.d_ff, 0.0f);
    std::vector<float> up_buf(seq_len_prefill * cfg.d_ff, 0.0f);
    std::vector<float> hidden_buf(seq_len_prefill * cfg.d_ff, 0.0f);
    std::vector<float> row_buf(d_model, 0.0f);
    const kernel::LinearParams gate_params{seq_len_prefill, cfg.d_model, cfg.d_ff};
    const kernel::LinearParams up_params{seq_len_prefill, cfg.d_model, cfg.d_ff};
    const kernel::LinearParams down_params{seq_len_prefill, cfg.d_ff, cfg.d_model};
    const kernel::SiluMulParams silu_mul_params{hidden_buf.size()};

    // ---------- Inference ----------
    print_section("Inference");

    print_subsection("Kernel Components");
    print_metric("inference.kernel.rms_norm",
                 mean_ns([&] { kernel::rms_norm(x7.data(), w0.norm1.weight.data(), w0.norm1.eps, x7_norm.data(), norm_params); },
                         warmup, iters));
    print_metric("inference.kernel.add",
                 mean_ns([&] { kernel::add(add_a.data(), add_b.data(), add_out.data(), add_params); }, warmup, iters));
    print_metric("inference.kernel.linear_q",
                 mean_ns([&] {
                     kernel::linear(x7_norm.data(), w0.attention.q_proj.weight.data(),
                                    w0.attention.q_proj.bias.empty() ? nullptr : w0.attention.q_proj.bias.data(),
                                    q_bench_out.data(), q_linear_params);
                 }, warmup, iters));
    print_metric("inference.kernel.apply_rope",
                 mean_ns([&] {
                     std::copy(q.begin(), q.end(), q_rope.begin());
                     kernel::apply_rope(q_rope.data(), rope_positions.data(), rope_cache, rope_params);
                 }, warmup, iters));
    print_metric("inference.kernel.softmax_stable",
                 mean_ns([&] { kernel::softmax_stable(softmax_logits.data(), softmax_probs.data(), softmax_params); },
                         warmup, iters));
    print_metric("inference.kernel.attention_prefill",
                 mean_ns([&] {
                     kernel::gqa_attention_prefill(q.data(), k_all.data(), v_all.data(), nullptr, ctx_bench.data(), nullptr,
                                                   attention_bench_params);
                 }, warmup, iters));
    print_metric("inference.kernel.attention_decode",
                 mean_ns([&] {
                     kernel::gqa_attention_decode(q_decode.data(), k_decode_all.data(), v_decode_all.data(), nullptr,
                                                  ctx_decode.data(), attention_decode_params);
                 }, warmup, iters));
    print_metric("inference.kernel.mlp_gate_linear",
                 mean_ns([&] {
                     kernel::linear(x7_norm.data(), w0.mlp.gate.weight.data(),
                                    w0.mlp.gate.bias.empty() ? nullptr : w0.mlp.gate.bias.data(), gate_buf.data(),
                                    gate_params);
                 }, warmup, iters));
    print_metric("inference.kernel.mlp_up_linear",
                 mean_ns([&] {
                     kernel::linear(x7_norm.data(), w0.mlp.up.weight.data(),
                                    w0.mlp.up.bias.empty() ? nullptr : w0.mlp.up.bias.data(), up_buf.data(), up_params);
                 }, warmup, iters));
    print_metric("inference.kernel.mlp_silu_mul",
                 mean_ns([&] {
                     kernel::silu_mul(gate_buf.data(), up_buf.data(), hidden_buf.data(), silu_mul_params);
                 }, warmup, iters));
    print_metric("inference.kernel.mlp_down_linear",
                 mean_ns([&] {
                     kernel::linear(hidden_buf.data(), w0.mlp.down.weight.data(),
                                    w0.mlp.down.bias.empty() ? nullptr : w0.mlp.down.bias.data(), mlp_down_out.data(),
                                    down_params);
                 }, warmup, iters));

    print_subsection("Pipeline Components");
    print_metric("inference.pipeline.embed",
                 mean_ns([&] {
                     std::vector<float> hidden;
                     embed_token_ids_into(tok_prefill, model->weights().token_embedding, d_model, hidden);
                 }, warmup, iters));
    print_metric("inference.pipeline.prefill",
                 mean_ns([&] {
                     std::vector<float> hidden(tok_prefill.size() * d_model, 0.0f);
                     model->prefill(tok_prefill, hidden.data());
                 }, warmup, iters));
    print_metric("inference.pipeline.decode_first_token",
                 mean_ns([&] {
                     model->reset_cache();
                    std::vector<float> hidden_prefill(tok_decode_prompt.size() * d_model, 0.0f);
                    model->prefill(tok_decode_prompt, hidden_prefill.data());
                    std::vector<float> hidden_decode(d_model, 0.0f);
                    model->decode(decode_token, hidden_decode.data());
                 }, warmup, iters));
    print_metric("inference.pipeline.decode_steady_state",
                 mean_ns_with_setup(
                     [&] {
                         model->reset_cache();
                        std::vector<float> hidden_prefill(tok_decode_prompt.size() * d_model, 0.0f);
                        model->prefill(tok_decode_prompt, hidden_prefill.data());
                     },
                    [&] {
                        std::vector<float> hidden_decode(d_model, 0.0f);
                        model->decode(decode_token, hidden_decode.data());
                    }, warmup, iters));

    std::vector<float> h_fwd7(tok_prefill.size() * d_model, 0.0f);
    model->prefill(tok_prefill, h_fwd7.data());
    LogitsOutput logits_frozen;
    project_logits_into(h_fwd7, seq_len_prefill, d_model, model->weights().lm_head, k_vocab, logits_frozen);
    print_metric("inference.pipeline.project_logits",
                 mean_ns([&] {
                     LogitsOutput logits;
                     project_logits_into(h_fwd7, seq_len_prefill, d_model, model->weights().lm_head, k_vocab, logits);
                 }, warmup,
                         iters));
    print_metric("inference.pipeline.argmax_from_hidden_row",
                 mean_ns([&] {
                     const size_t last = seq_len_prefill - 1;
                     for (size_t d = 0; d < d_model; ++d) {
                         row_buf[d] = h_fwd7[last * d_model + d];
                     }
                     (void)argmax_from_hidden(std::span<const float>(row_buf.data(), row_buf.size()),
                                              model->weights().lm_head, k_vocab, d_model);
                 }, warmup, iters));

    print_subsection("End-to-End");
    print_metric("inference.e2e.last_argmax",
                 mean_ns([&] { (void)task::last_argmax(*model, tok_decode_prompt); }, warmup, iters));
    print_metric("inference.e2e.first_token_argmax",
                 mean_ns([&] {
                     model->reset_cache();
                     std::vector<float> hidden_prefill(tok_decode_prompt.size() * d_model, 0.0f);
                     model->prefill(tok_decode_prompt, hidden_prefill.data());
                     model->decode(decode_token, row_buf.data());
                     (void)argmax_from_hidden(std::span<const float>(row_buf.data(), row_buf.size()),
                                              model->weights().lm_head, k_vocab, d_model);
                 }, warmup, iters));
    const double decode_xn_ns = mean_ns_with_setup(
        [&] {
            model->reset_cache();
            std::vector<float> hidden_prefill(tok_decode_prompt.size() * d_model, 0.0f);
            model->prefill(tok_decode_prompt, hidden_prefill.data());
        },
        [&] {
            uint32_t token = decode_token;
            for (size_t step = 0; step < k_decode_steps; ++step) {
                model->decode(token, row_buf.data());
                token = argmax_from_hidden(std::span<const float>(row_buf.data(), row_buf.size()), model->weights().lm_head,
                                           k_vocab, d_model);
            }
        },
        warmup, iters);
    print_metric_per_token("inference.e2e.decode_x32_tokens", decode_xn_ns / static_cast<double>(k_decode_steps));

    // ---------- Train ----------
    print_section("Train");

    std::vector<model::BlockForwardTape> tapes(model->num_layers());
    std::vector<float> hidden_train(tok_train.size() * d_model, 0.0f);
    model->forward_for_training(tok_train, tapes, hidden_train.data());
    LogitsOutput logits_train;
    project_logits_into(hidden_train, tok_train.size(), d_model, model->weights().lm_head, k_vocab, logits_train);
    CrossEntropyResult ce_train;
    const std::vector<size_t> train_prediction_steps = task::answer_prediction_steps(tok_train);
    cross_entropy_steps_into(logits_train, tok_train, train_prediction_steps, ce_train);
    std::vector<float> grad_hidden_out(tok_train.size() * d_model, 0.0f);
    std::vector<float> grad_lm_head(model->weights().lm_head.size(), 0.0f);
    std::vector<float> grad_embed(model->weights().token_embedding.size(), 0.0f);
    model::ModelWeights grad_model_backward = clone_model(model->weights());
    const model::BlockForwardTape& tape0 = tapes[0];

    std::vector<float> dy_linear(tape0.q_rope.size(), 0.01f);
    std::vector<float> dx_linear(tape0.norm1_out.size(), 0.0f);
    std::vector<float> grad_w_linear(w0.attention.q_proj.weight.size(), 0.0f);
    std::vector<float> grad_b_linear(w0.attention.q_proj.bias.size(), 0.0f);
    const kernel::LinearParams linear_bwd_params{tok_train.size(), cfg.d_model, cfg.num_heads * cfg.head_dim};

    std::vector<float> dy_norm(tape0.hidden_after_attn.size(), 0.01f);
    std::vector<float> dx_norm(tape0.hidden_after_attn.size(), 0.0f);
    std::vector<float> grad_norm_weight(w0.norm2.weight.size(), 0.0f);
    const kernel::RmsNormParams norm_bwd_params{tok_train.size(), cfg.d_model};

    std::vector<float> grad_hidden_mid(tape0.hidden_mid.size(), 0.01f);
    std::vector<float> grad_gate(tape0.gate.size(), 0.0f);
    std::vector<float> grad_up(tape0.up.size(), 0.0f);
    const kernel::SiluMulParams silu_mul_bwd_params{tape0.hidden_mid.size()};

    std::vector<float> grad_q_rope(tape0.q_rope.size(), 0.01f);
    std::vector<float> grad_q_pre_rope(tape0.q_rope.size(), 0.0f);
    kernel::RopeCache rope_cache_bwd(cfg.rope_base, cfg.head_dim, cfg.rope_dim);
    const kernel::RopeParams rope_bwd_params{tok_train.size(), cfg.num_heads, cfg.head_dim};
    if (!tape0.positions.empty()) {
        rope_cache_bwd.ensure(tape0.positions.back() + 1);
    }

    std::vector<float> grad_attn_ctx(tape0.attn_ctx.size(), 0.01f);
    std::vector<float> grad_q_attn(tape0.q_rope.size(), 0.0f);
    const std::vector<float>& tape0_k_all = tape0.k_all.empty() ? tape0.k_rope : tape0.k_all;
    const std::vector<float>& tape0_v_all = tape0.v_all.empty() ? tape0.v_proj : tape0.v_all;
    std::vector<float> grad_k_attn(tape0_k_all.size(), 0.0f);
    std::vector<float> grad_v_attn(tape0_v_all.size(), 0.0f);
    const size_t total_kv_len = tape0_k_all.size() / (cfg.num_kv_heads * cfg.head_dim);
    const kernel::AttentionParams attn_bwd_params{
        tok_train.size(), cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, tape0.past_len, total_kv_len, true, true};

    const model::ModelWeights weights_ref = model->weights();
    const model::ModelWeights adam_m_ref = adam_m;
    const model::ModelWeights adam_v_ref = adam_v;
    model::ModelWeights weights_opt = clone_model(weights_ref);
    model::ModelWeights adam_m_opt = clone_model(adam_m_ref);
    model::ModelWeights adam_v_opt = clone_model(adam_v_ref);

    print_subsection("Kernel Components");
    print_metric("train.kernel.backward_lm_head",
                 mean_ns([&] {
                     std::fill(grad_lm_head.begin(), grad_lm_head.end(), 0.0f);
                     kernel::backward_lm_head(hidden_train.data(), ce_train.probs.data(), ce_train.targets.data(),
                                              ce_train.steps.data(), ce_train.valid_steps, d_model, k_vocab,
                                              grad_lm_head.data());
                 }, warmup, iters));
    print_metric("train.kernel.backward_hidden",
                 mean_ns([&] {
                    kernel::backward_hidden(ce_train.probs.data(), ce_train.targets.data(), ce_train.steps.data(),
                                            ce_train.valid_steps, model->weights().lm_head.data(), tok_train.size(),
                                            d_model, k_vocab, grad_hidden_out.data());
                 }, warmup, iters));
    print_metric("train.kernel.backward_embedding",
                 mean_ns([&] {
                     std::fill(grad_embed.begin(), grad_embed.end(), 0.0f);
                    kernel::backward_embedding(tok_train.data(), tok_train.size(), hidden_train.data(), d_model,
                                                grad_embed.data());
                 }, warmup, iters));
    print_metric("train.kernel.linear_backward",
                 mean_ns([&] {
                     kernel::linear_backward(tape0.norm1_out.data(), w0.attention.q_proj.weight.data(),
                                             !w0.attention.q_proj.bias.empty(), dy_linear.data(), dx_linear.data(),
                                             grad_w_linear.data(), grad_b_linear.data(), linear_bwd_params);
                 }, warmup, iters));
    print_metric("train.kernel.rms_norm_backward",
                 mean_ns([&] {
                     kernel::rms_norm_backward(tape0.hidden_after_attn.data(), w0.norm2.weight.data(), w0.norm2.eps,
                                               dy_norm.data(), dx_norm.data(), grad_norm_weight.data(), norm_bwd_params);
                 }, warmup, iters));
    print_metric("train.kernel.silu_mul_backward",
                 mean_ns([&] {
                     kernel::silu_mul_backward(tape0.gate.data(), tape0.up.data(), grad_hidden_mid.data(), grad_gate.data(),
                                               grad_up.data(), silu_mul_bwd_params);
                 }, warmup, iters));
    print_metric("train.kernel.apply_rope_backward",
                 mean_ns([&] {
                     kernel::apply_rope_backward(grad_q_rope.data(), grad_q_pre_rope.data(), tape0.positions.data(),
                                                 rope_cache_bwd, rope_bwd_params);
                 }, warmup, iters));
    print_metric("train.kernel.attention_prefill_backward",
                 mean_ns([&] {
                     kernel::gqa_attention_prefill_backward(
                        tape0.q_rope.data(), tape0_k_all.data(), tape0_v_all.data(), nullptr, tape0.attn_probs.data(),
                         grad_attn_ctx.data(), grad_q_attn.data(), grad_k_attn.data(), grad_v_attn.data(), attn_bwd_params);
                 }, warmup, iters));

    print_subsection("Pipeline Components");
    print_metric("train.pipeline.forward_for_training",
                 mean_ns([&] {
                     std::vector<float> hidden(tok_train.size() * d_model, 0.0f);
                     model->forward_for_training(tok_train, tapes, hidden.data());
                 }, warmup, iters));
    print_metric("train.pipeline.cross_entropy",
                 mean_ns([&] {
                     CrossEntropyResult ce;
                     cross_entropy_steps_into(logits_train, tok_train, train_prediction_steps, ce);
                 }, warmup, iters));
    print_metric("train.pipeline.model_backward",
                 mean_ns([&] {
                     clear(grad_model_backward);
                     model->backward(tapes, grad_hidden_out, grad_model_backward, grad_hidden_out);
                 }, warmup, iters));
    print_metric("train.pipeline.adamw_update_model",
                 mean_ns_with_setup(
                     [&] {
                         weights_opt = weights_ref;
                         adam_m_opt = adam_m_ref;
                         adam_v_opt = adam_v_ref;
                     },
                     [&] {
                         adamw_update_model(weights_opt, grad_model_backward, adam_m_opt, adam_v_opt,
                                            /*t=*/1, 0.004f, 0.9f, 0.95f, 1e-8f, 0.02f);
                     },
                     warmup, iters));

    print_subsection("End-to-End");
    TrainWorkspace workspace{};
    size_t adam_step = 1;
    print_metric("train.e2e.train_step",
                 mean_ns([&] {
                     (void)train_step(*model, tok_train, train_prediction_steps, adam_step, 0.004f, 0.9f, 0.95f,
                                      1e-8f, 0.02f, 1.0f, adam_m, adam_v, workspace);
                     ++adam_step;
                 }, train_warmup, train_iters));
    const double train_step_ns = mean_ns([&] {
        (void)train_step(*model, tok_train, train_prediction_steps, adam_step, 0.004f, 0.9f, 0.95f, 1e-8f, 0.02f, 1.0f,
                         adam_m, adam_v, workspace);
        ++adam_step;
    }, train_warmup, train_iters);
    print_metric_per_token("train.e2e.train_step_per_token", train_step_ns / static_cast<double>(tok_train.size()));

    std::println("");
    std::println("done.");
    return 0;
}
