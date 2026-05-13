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
    std::println("{:<52} {:>12.2f} us/iter", name, us_per_iter);
}

void print_metric_per_token(const char* name, double ns_per_token) {
    const double us_per_token = ns_per_token / 1000.0;
    std::println("{:<52} {:>12.2f} us/token", name, us_per_token);
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
    constexpr uint32_t k_model_seed = 42;
    std::unique_ptr<model::MiniLlm> model =
        std::make_unique<model::MiniLlm>(model::MiniLlm::init_random(/*vocab_size=*/k_vocab, k_model_seed));
    const model::ModelConfig& cfg = model->config();
    model::ModelWeights adam_m = train::zeros_like(model->weights());
    model::ModelWeights adam_v = train::zeros_like(model->weights());

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
    model::RopeCache rope_cache(cfg.rope_base, cfg.head_dim, cfg.rope_dim);
    rope_cache.ensure(seq_len_prefill);
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

    print_subsection("Individual Kernels");
    print_metric("kernel: RMSNorm",
                 mean_ns([&] { kernel::rms_norm(x7.data(), w0.norm1.weight.data(), w0.norm1.eps, x7_norm.data(), norm_params); },
                         warmup, iters));
    print_metric("kernel: residual add",
                 mean_ns([&] { kernel::add(add_a.data(), add_b.data(), add_out.data(), add_params); }, warmup, iters));
    print_metric("kernel: Q projection",
                 mean_ns([&] {
                     kernel::linear(x7_norm.data(), w0.attention.q_proj.weight.data(),
                                    w0.attention.q_proj.bias.empty() ? nullptr : w0.attention.q_proj.bias.data(),
                                    q_bench_out.data(), q_linear_params);
                 }, warmup, iters));
    print_metric("kernel: RoPE rotation",
                 mean_ns([&] {
                     std::copy(q.begin(), q.end(), q_rope.begin());
                     kernel::apply_rope(q_rope.data(), rope_positions.data(), rope_cache.cos_data(),
                                        rope_cache.sin_data(), rope_cache.rot_dim(), rope_params);
                 }, warmup, iters));
    print_metric("kernel: vocab softmax",
                 mean_ns([&] { kernel::softmax(softmax_logits.data(), softmax_probs.data(), softmax_params); },
                         warmup, iters));
    print_metric("kernel: attention prefill",
                 mean_ns([&] {
                     kernel::gqa_attention_forward(q.data(), k_all.data(), v_all.data(), nullptr, ctx_bench.data(), nullptr,
                                                   attention_bench_params);
                 }, warmup, iters));
    print_metric("kernel: attention decode",
                 mean_ns([&] {
                     kernel::gqa_attention_forward(q_decode.data(), k_decode_all.data(), v_decode_all.data(), nullptr,
                                                   ctx_decode.data(), nullptr, attention_decode_params);
                 }, warmup, iters));
    print_metric("kernel: MLP gate projection",
                 mean_ns([&] {
                     kernel::linear(x7_norm.data(), w0.mlp.gate.weight.data(),
                                    w0.mlp.gate.bias.empty() ? nullptr : w0.mlp.gate.bias.data(), gate_buf.data(),
                                    gate_params);
                 }, warmup, iters));
    print_metric("kernel: MLP up projection",
                 mean_ns([&] {
                     kernel::linear(x7_norm.data(), w0.mlp.up.weight.data(),
                                    w0.mlp.up.bias.empty() ? nullptr : w0.mlp.up.bias.data(), up_buf.data(), up_params);
                 }, warmup, iters));
    print_metric("kernel: MLP SiLU gate multiply",
                 mean_ns([&] {
                     kernel::silu_mul(gate_buf.data(), up_buf.data(), hidden_buf.data(), silu_mul_params);
                 }, warmup, iters));
    print_metric("kernel: MLP down projection",
                 mean_ns([&] {
                     kernel::linear(hidden_buf.data(), w0.mlp.down.weight.data(),
                                    w0.mlp.down.bias.empty() ? nullptr : w0.mlp.down.bias.data(), mlp_down_out.data(),
                                    down_params);
                 }, warmup, iters));

    print_subsection("Model Stages");
    print_metric("stage: token embedding",
                 mean_ns([&] {
                     std::vector<float> hidden;
                     embed_tokens(tok_prefill, model->weights().token_embedding, d_model, hidden);
                 }, warmup, iters));
    print_metric("stage: prefill whole prompt",
                 mean_ns([&] {
                     std::vector<float> hidden(tok_prefill.size() * d_model, 0.0f);
                     model->prefill(tok_prefill, hidden.data());
                 }, warmup, iters));
    print_metric("stage: decode first generated token",
                 mean_ns([&] {
                     model->reset_cache();
                    std::vector<float> hidden_prefill(tok_decode_prompt.size() * d_model, 0.0f);
                    model->prefill(tok_decode_prompt, hidden_prefill.data());
                    std::vector<float> hidden_decode(d_model, 0.0f);
                    model->decode(decode_token, hidden_decode.data());
                 }, warmup, iters));
    print_metric("stage: decode next generated token",
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
    project_logits(h_fwd7, seq_len_prefill, d_model, model->weights().output_projection, k_vocab, logits_frozen);
    print_metric("stage: project hidden to vocab logits",
                 mean_ns([&] {
                     LogitsOutput logits;
                     project_logits(h_fwd7, seq_len_prefill, d_model, model->weights().output_projection, k_vocab, logits);
                 }, warmup,
                         iters));
    std::vector<float> logits_row(k_vocab, 0.0f);
    print_metric("stage: choose argmax token",
                 mean_ns([&] {
                     const size_t last = seq_len_prefill - 1;
                     for (size_t d = 0; d < d_model; ++d) {
                         row_buf[d] = h_fwd7[last * d_model + d];
                     }
                     kernel::gemm_nt(row_buf.data(), model->weights().output_projection.data(), logits_row.data(), 1,
                                     k_vocab, d_model);
                     (void)argmax(logits_row);
                 }, warmup, iters));

    print_subsection("End-to-End Paths");
    print_metric("path: prefill then argmax last row",
                 mean_ns([&] { (void)task::last_argmax(*model, tok_decode_prompt); }, warmup, iters));
    print_metric("path: prefill then decode one token",
                 mean_ns([&] {
                     model->reset_cache();
                     std::vector<float> hidden_prefill(tok_decode_prompt.size() * d_model, 0.0f);
                     model->prefill(tok_decode_prompt, hidden_prefill.data());
                     model->decode(decode_token, row_buf.data());
                     kernel::gemm_nt(row_buf.data(), model->weights().output_projection.data(), logits_row.data(), 1,
                                     k_vocab, d_model);
                     (void)argmax(logits_row);
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
                kernel::gemm_nt(row_buf.data(), model->weights().output_projection.data(), logits_row.data(), 1, k_vocab,
                                d_model);
                token = argmax(logits_row);
            }
        },
        warmup, iters);
    print_metric_per_token("path: decode 32 generated tokens", decode_xn_ns / static_cast<double>(k_decode_steps));

    // ---------- Train ----------
    print_section("Train");

    std::vector<model::BlockForwardTape> tapes(model->num_layers());
    std::vector<float> hidden_train(tok_train.size() * d_model, 0.0f);
    model->forward_train(tok_train, tapes, hidden_train.data());
    LogitsOutput logits_train;
    project_logits(hidden_train, tok_train.size(), d_model, model->weights().output_projection, k_vocab, logits_train);
    train::CrossEntropyResult ce_train;
    const std::vector<size_t> train_prediction_steps = task::answer_steps(tok_train);
    train::cross_entropy(logits_train, tok_train, train_prediction_steps, ce_train);
    std::vector<float> grad_hidden_out(tok_train.size() * d_model, 0.0f);
    std::vector<float> grad_output_projection(model->weights().output_projection.size(), 0.0f);
    std::vector<float> grad_embed(model->weights().token_embedding.size(), 0.0f);
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
    model::RopeCache rope_cache_bwd(cfg.rope_base, cfg.head_dim, cfg.rope_dim);
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

    train::cross_entropy_grad_hidden(ce_train.probs.data(), ce_train.targets.data(), ce_train.steps.data(),
                                     ce_train.valid_steps, model->weights().output_projection.data(),
                                     tok_train.size(), d_model, k_vocab, grad_hidden_out.data());

    print_subsection("Individual Kernels");
    print_metric("train: grad final vocab projection",
                 mean_ns([&] {
                     std::fill(grad_output_projection.begin(), grad_output_projection.end(), 0.0f);
                     train::cross_entropy_grad_weight(hidden_train.data(), ce_train.probs.data(),
                                                     ce_train.targets.data(), ce_train.steps.data(),
                                                     ce_train.valid_steps, d_model, k_vocab,
                                                     grad_output_projection.data());
                 }, warmup, iters));
    print_metric("train: grad hidden from logits",
                 mean_ns([&] {
                     train::cross_entropy_grad_hidden(ce_train.probs.data(), ce_train.targets.data(),
                                                     ce_train.steps.data(), ce_train.valid_steps,
                                                     model->weights().output_projection.data(),
                                                     tok_train.size(), d_model, k_vocab, grad_hidden_out.data());
                 }, warmup, iters));
    print_metric("engine: grad token embedding",
                 mean_ns([&] {
                     std::fill(grad_embed.begin(), grad_embed.end(), 0.0f);
                     engine::embed_tokens_grad(tok_train, hidden_train, d_model, grad_embed);
                 }, warmup, iters));
    print_metric("kernel: linear layer backward",
                 mean_ns([&] {
                     kernel::linear_backward(tape0.norm1_out.data(), w0.attention.q_proj.weight.data(),
                                             !w0.attention.q_proj.bias.empty(), dy_linear.data(), dx_linear.data(),
                                             grad_w_linear.data(), grad_b_linear.data(), linear_bwd_params);
                 }, warmup, iters));
    print_metric("kernel: RMSNorm backward",
                 mean_ns([&] {
                     kernel::rms_norm_backward(tape0.hidden_after_attn.data(), w0.norm2.weight.data(), w0.norm2.eps,
                                               dy_norm.data(), dx_norm.data(), grad_norm_weight.data(), norm_bwd_params);
                 }, warmup, iters));
    print_metric("kernel: SiLU gate backward",
                 mean_ns([&] {
                     kernel::silu_mul_backward(tape0.gate.data(), tape0.up.data(), grad_hidden_mid.data(), grad_gate.data(),
                                               grad_up.data(), silu_mul_bwd_params);
                 }, warmup, iters));
    print_metric("kernel: RoPE backward",
                 mean_ns([&] {
                     kernel::apply_rope_backward(grad_q_rope.data(), grad_q_pre_rope.data(), tape0.positions.data(),
                                                 rope_cache_bwd.cos_data(), rope_cache_bwd.sin_data(),
                                                 rope_cache_bwd.rot_dim(), rope_bwd_params);
                 }, warmup, iters));
    print_metric("kernel: attention backward",
                 mean_ns([&] {
                     kernel::gqa_attention_backward(
                        tape0.q_rope.data(), tape0_k_all.data(), tape0_v_all.data(), nullptr, tape0.attn_probs.data(),
                         grad_attn_ctx.data(), grad_q_attn.data(), grad_k_attn.data(), grad_v_attn.data(), attn_bwd_params);
                 }, warmup, iters));

    // Time each phase of train::step inline. This avoids the deep-copy "setup" that
    // mean_ns_with_setup forced (and which thrashed cache between iterations), and
    // exposes each phase under the exact same cache state it sees in train::step().
    print_subsection("Training Step Breakdown");
    train::Workspace pipeline_ws{};
    size_t breakdown_step = 1;
    int64_t t_prepare = 0, t_fwd = 0, t_proj = 0, t_ce = 0;
    int64_t t_bwd_lm = 0, t_bwd_h = 0, t_model_bwd = 0, t_bwd_emb = 0;
    int64_t t_clip = 0, t_adamw = 0;

    auto run_breakdown_step = [&](bool measure) {
        const auto t0 = steady_clock::now();
        pipeline_ws.prepare(*model);
        const auto t1 = steady_clock::now();

        pipeline_ws.hidden_states.resize(tok_train.size() * d_model);
        model->forward_train(tok_train, pipeline_ws.tapes, pipeline_ws.hidden_states.data());
        const auto t2 = steady_clock::now();

        project_logits(pipeline_ws.hidden_states.data(), tok_train.size(), d_model,
                       std::span<const size_t>(train_prediction_steps.data(),
                                               train_prediction_steps.size()),
                       model->weights().output_projection, k_vocab,
                       pipeline_ws.logits, pipeline_ws.gathered_hidden);
        const auto t3 = steady_clock::now();

        train::cross_entropy(pipeline_ws.logits, tok_train, train_prediction_steps,
                             pipeline_ws.cross_entropy);
        const auto t4 = steady_clock::now();

        const train::CrossEntropyResult& ce = pipeline_ws.cross_entropy;
        train::cross_entropy_grad_weight(pipeline_ws.hidden_states.data(), ce.probs.data(), ce.targets.data(),
                                         ce.steps.data(), ce.valid_steps, d_model, k_vocab,
                                         pipeline_ws.grad.output_projection.data());
        const auto t5 = steady_clock::now();

        pipeline_ws.grad_hidden_out.resize(tok_train.size() * d_model);
        train::cross_entropy_grad_hidden(ce.probs.data(), ce.targets.data(), ce.steps.data(), ce.valid_steps,
                                         model->weights().output_projection.data(), tok_train.size(), d_model,
                                         k_vocab, pipeline_ws.grad_hidden_out.data());
        const auto t6 = steady_clock::now();

        model->backward(pipeline_ws.tapes, pipeline_ws.grad_hidden_out, pipeline_ws.grad,
                        pipeline_ws.grad_hidden_in);
        const auto t7 = steady_clock::now();

        engine::embed_tokens_grad(tok_train, pipeline_ws.grad_hidden_in, d_model,
                                  pipeline_ws.grad.token_embedding);
        const auto t8 = steady_clock::now();

        train::clip_grad(pipeline_ws.grad, 1.0f);
        const auto t9 = steady_clock::now();

        train::adamw_step(model->mutable_weights(), pipeline_ws.grad, adam_m, adam_v, breakdown_step,
                          0.004f, 0.9f, 0.95f, 1e-8f, 0.02f);
        const auto t10 = steady_clock::now();

        if (measure) {
            using std::chrono::duration_cast;
            using std::chrono::nanoseconds;
            t_prepare   += duration_cast<nanoseconds>(t1  - t0).count();
            t_fwd       += duration_cast<nanoseconds>(t2  - t1).count();
            t_proj      += duration_cast<nanoseconds>(t3  - t2).count();
            t_ce        += duration_cast<nanoseconds>(t4  - t3).count();
            t_bwd_lm    += duration_cast<nanoseconds>(t5  - t4).count();
            t_bwd_h     += duration_cast<nanoseconds>(t6  - t5).count();
            t_model_bwd += duration_cast<nanoseconds>(t7  - t6).count();
            t_bwd_emb   += duration_cast<nanoseconds>(t8  - t7).count();
            t_clip      += duration_cast<nanoseconds>(t9  - t8).count();
            t_adamw     += duration_cast<nanoseconds>(t10 - t9).count();
        }
        ++breakdown_step;
    };

    for (size_t i = 0; i < train_warmup; ++i) run_breakdown_step(false);
    for (size_t i = 0; i < train_iters; ++i) run_breakdown_step(true);

    const double inv_iters = 1.0 / static_cast<double>(train_iters);
    const double prepare_avg     = static_cast<double>(t_prepare)   * inv_iters;
    const double fwd_avg         = static_cast<double>(t_fwd)       * inv_iters;
    const double proj_avg        = static_cast<double>(t_proj)      * inv_iters;
    const double ce_avg          = static_cast<double>(t_ce)        * inv_iters;
    const double bwd_lm_avg      = static_cast<double>(t_bwd_lm)    * inv_iters;
    const double bwd_h_avg       = static_cast<double>(t_bwd_h)     * inv_iters;
    const double model_bwd_avg   = static_cast<double>(t_model_bwd) * inv_iters;
    const double bwd_emb_avg     = static_cast<double>(t_bwd_emb)   * inv_iters;
    const double clip_avg        = static_cast<double>(t_clip)      * inv_iters;
    const double adamw_avg       = static_cast<double>(t_adamw)     * inv_iters;

    print_metric("step: reset gradient buffers", prepare_avg);
    print_metric("step: forward training pass", fwd_avg);
    print_metric("step: project selected logits", proj_avg);
    print_metric("step: cross entropy loss", ce_avg);
    print_metric("step: grad final vocab projection", bwd_lm_avg);
    print_metric("step: grad hidden from logits", bwd_h_avg);
    print_metric("step: transformer backward pass", model_bwd_avg);
    print_metric("step: grad token embedding", bwd_emb_avg);
    print_metric("step: gradient clipping", clip_avg);
    print_metric("step: AdamW optimizer update", adamw_avg);

    const double train_phase_sum_ns = prepare_avg + fwd_avg + proj_avg + ce_avg + bwd_lm_avg + bwd_h_avg +
                                      model_bwd_avg + bwd_emb_avg + clip_avg + adamw_avg;
    print_metric("check: sum of listed train steps", train_phase_sum_ns);

    print_subsection("End-to-End Training");
    train::Workspace workspace{};
    size_t adam_step = breakdown_step;
    const double train_step_ns = mean_ns([&] {
        (void)train::step(*model, tok_train, train_prediction_steps, adam_step, 0.004f, 0.9f, 0.95f, 1e-8f, 0.02f, 1.0f,
                          adam_m, adam_v, workspace);
        ++adam_step;
    }, train_warmup, train_iters);
    print_metric("full train::step()", train_step_ns);
    print_metric_per_token("full train::step() per token", train_step_ns / static_cast<double>(tok_train.size()));
    print_metric("check: unlisted train::step overhead", train_step_ns - train_phase_sum_ns);

    std::println("");
    std::println("done.");
    return 0;
}
