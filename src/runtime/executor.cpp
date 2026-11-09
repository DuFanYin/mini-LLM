#include "runtime/executor.h"

#include "core/cuda_check.h"
#include "kernels/ops.h"

#include <format>
#include <stdexcept>
#include <variant>

namespace runtime {

namespace {

using core::DType;
using kernels::bf16;

const bf16* w16(const core::Tensor& t) {
    return t.data<bf16>();
}

} // namespace

Executor::Executor(const model::Config& config, const model::Weights& weights, Options options)
    : config_(config), weights_(weights), options_(options) {
    const auto& c = config_;
    const auto& a = c.attn;
    const auto& g = c.gdn;
    const std::size_t T = options_.max_chunk;
    auto f32 = [&](std::vector<std::size_t> shape) { return core::Tensor(std::move(shape), DType::f32, alloc_); };

    layers_.resize(c.num_layers());
    for (std::size_t i = 0; i < c.num_layers(); ++i) {
        LayerState& st = layers_[i];
        if (c.layers[i] == model::LayerKind::full_attention) {
            st.k_cache = f32({a.num_kv_heads, options_.max_seq, a.head_dim});
            st.v_cache = f32({a.num_kv_heads, options_.max_seq, a.head_dim});
        } else {
            st.conv = f32({g.conv_kernel - 1, g.conv_dim()});
            st.s = f32({g.num_v_heads, g.k_head_dim, g.v_head_dim});
        }
    }

    ids_ = core::Tensor({T}, DType::i32, alloc_);
    next_ = core::Tensor({1}, DType::i32, alloc_);
    hidden_ = f32({T, c.hidden_size});
    normed_ = f32({T, c.hidden_size});
    mixed_ = f32({T, c.hidden_size});
    qg_ = f32({T, a.num_heads * 2 * a.head_dim});
    k_ = f32({T, a.num_kv_heads * a.head_dim});
    v_ = f32({T, a.num_kv_heads * a.head_dim});
    attn_ = f32({T, a.num_heads * a.head_dim});
    qkv_ = f32({T, g.conv_dim()});
    conv_ = f32({T, g.conv_dim()});
    z_ = f32({T, g.value_dim()});
    a_ = f32({T, g.num_v_heads});
    b_ = f32({T, g.num_v_heads});
    g_ = f32({T, g.num_v_heads});
    beta_ = f32({T, g.num_v_heads});
    core_ = f32({T, g.value_dim()});
    gate_ = f32({T, c.intermediate_size});
    up_ = f32({T, c.intermediate_size});
    logits_ = f32({c.vocab_size});
    reset();
}

void Executor::reset() {
    for (LayerState& st : layers_) {
        st.conv.zero();
        st.s.zero();
    }
    position_ = 0;
}

std::int32_t Executor::forward(std::span<const std::int32_t> ids, std::vector<float>* last_logits, const Trace* trace) {
    if (ids.empty()) {
        throw std::invalid_argument("forward: no tokens");
    }
    if (position_ + ids.size() > options_.max_seq) {
        throw std::length_error(
            std::format("forward: {} + {} tokens exceeds max_seq {}", position_, ids.size(), options_.max_seq));
    }
    for (std::int32_t id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= config_.vocab_size) {
            throw std::out_of_range(std::format("forward: token id {} is outside the vocabulary", id));
        }
    }
    std::int32_t next = -1;
    for (std::size_t off = 0; off < ids.size(); off += options_.max_chunk) {
        const auto chunk = ids.subspan(off, std::min(options_.max_chunk, ids.size() - off));
        const bool last = off + chunk.size() == ids.size();
        next = run_chunk(chunk, last ? last_logits : nullptr, trace);
    }
    return next;
}

std::int32_t Executor::run_chunk(std::span<const std::int32_t> ids, std::vector<float>* last_logits,
                                 const Trace* trace) {
    const auto& c = config_;
    const cudaStream_t s = stream_.get();
    const std::size_t T = ids.size();
    const std::size_t H = c.hidden_size;
    const auto eps = static_cast<float>(c.rms_norm_eps);

    core::cuda_check(cudaMemcpyAsync(ids_.raw(), ids.data(), T * sizeof(std::int32_t), cudaMemcpyHostToDevice, s));
    kernels::embedding(ids_.data<std::int32_t>(), T, w16(weights_.embed_tokens()), H, hidden_.data<float>(), s);
    emit(trace, "embed", hidden_, T, H);

    for (std::size_t i = 0; i < c.num_layers(); ++i) {
        const model::LayerWeights& lw = weights_.layers()[i];
        kernels::rms_norm(hidden_.data<float>(), T, H, w16(lw.input_norm), eps, normed_.data<float>(), s);
        if (const auto* attn = std::get_if<model::FullAttentionWeights>(&lw.mixer)) {
            attention_layer(*attn, layers_[i], T);
        } else {
            delta_net_layer(std::get<model::GatedDeltaNetWeights>(lw.mixer), layers_[i], T);
        }
        kernels::add_inplace(hidden_.data<float>(), mixed_.data<float>(), T * H, s);
        emit(trace, std::format("layer{:02}.mixer", i), hidden_, T, H);

        const std::size_t I = c.intermediate_size;
        kernels::rms_norm(hidden_.data<float>(), T, H, w16(lw.post_attention_norm), eps, normed_.data<float>(), s);
        kernels::linear(normed_.data<float>(), T, H, w16(lw.mlp.gate_proj), I, gate_.data<float>(), s);
        kernels::linear(normed_.data<float>(), T, H, w16(lw.mlp.up_proj), I, up_.data<float>(), s);
        kernels::silu_mul(gate_.data<float>(), up_.data<float>(), gate_.data<float>(), T * I, s);
        kernels::linear(gate_.data<float>(), T, I, w16(lw.mlp.down_proj), H, mixed_.data<float>(), s);
        kernels::add_inplace(hidden_.data<float>(), mixed_.data<float>(), T * H, s);
        emit(trace, std::format("layer{:02}.out", i), hidden_, T, H);
    }

    kernels::rms_norm(hidden_.data<float>(), T, H, w16(weights_.final_norm()), eps, normed_.data<float>(), s);
    emit(trace, "final_norm", normed_, T, H);
    kernels::linear(normed_.data<float>() + (T - 1) * H, 1, H, w16(weights_.output_projection()), c.vocab_size,
                    logits_.data<float>(), s);
    kernels::argmax(logits_.data<float>(), c.vocab_size, next_.data<std::int32_t>(), s);

    std::int32_t next = -1;
    core::cuda_check(cudaMemcpyAsync(&next, next_.raw(), sizeof(next), cudaMemcpyDeviceToHost, s));
    if (last_logits != nullptr) {
        last_logits->resize(c.vocab_size);
        core::cuda_check(
            cudaMemcpyAsync(last_logits->data(), logits_.raw(), logits_.bytes(), cudaMemcpyDeviceToHost, s));
    }
    stream_.synchronize();
    emit(trace, "logits", logits_, 1, c.vocab_size);
    position_ += T;
    return next;
}

void Executor::attention_layer(const model::FullAttentionWeights& w, LayerState& st, std::size_t T) {
    const auto& c = config_;
    const auto& a = c.attn;
    const cudaStream_t s = stream_.get();
    const std::size_t H = c.hidden_size;
    const std::size_t q_row = a.num_heads * 2 * a.head_dim;
    const std::size_t kv_row = a.num_kv_heads * a.head_dim;
    const auto eps = static_cast<float>(c.rms_norm_eps);
    const float* x = normed_.data<float>();

    kernels::linear(x, T, H, w16(w.q_proj), q_row, qg_.data<float>(), s);
    kernels::linear(x, T, H, w16(w.k_proj), kv_row, k_.data<float>(), s);
    kernels::linear(x, T, H, w16(w.v_proj), kv_row, v_.data<float>(), s);

    // q_proj packs [query | gate] per head.
    const kernels::HeadView q{qg_.data<float>(), q_row, 2 * a.head_dim};
    const kernels::HeadView gate{qg_.data<float>() + a.head_dim, q_row, 2 * a.head_dim};
    const kernels::HeadView k{k_.data<float>(), kv_row, a.head_dim};
    kernels::head_norm_rope(q, T, a.num_heads, a.head_dim, w16(w.q_norm), eps, a.rotary_dim, a.rope_theta, position_,
                            s);
    kernels::head_norm_rope(k, T, a.num_kv_heads, a.head_dim, w16(w.k_norm), eps, a.rotary_dim, a.rope_theta, position_,
                            s);
    kernels::kv_append(k_.data<float>(), v_.data<float>(), st.k_cache.data<float>(), st.v_cache.data<float>(), T,
                       a.num_kv_heads, a.head_dim, position_, options_.max_seq, s);
    kernels::attention(q, st.k_cache.data<float>(), st.v_cache.data<float>(), options_.max_seq, attn_.data<float>(), T,
                       a.num_heads, a.num_kv_heads, a.head_dim, position_, s);
    kernels::sigmoid_gate(attn_.data<float>(), gate, T, a.num_heads, a.head_dim, s);
    kernels::linear(attn_.data<float>(), T, a.num_heads * a.head_dim, w16(w.o_proj), H, mixed_.data<float>(), s);
}

void Executor::delta_net_layer(const model::GatedDeltaNetWeights& w, LayerState& st, std::size_t T) {
    const auto& c = config_;
    const auto& g = c.gdn;
    const cudaStream_t s = stream_.get();
    const std::size_t H = c.hidden_size;
    const float* x = normed_.data<float>();

    kernels::linear(x, T, H, w16(w.in_proj_qkv), g.conv_dim(), qkv_.data<float>(), s);
    kernels::linear(x, T, H, w16(w.in_proj_z), g.value_dim(), z_.data<float>(), s);
    kernels::linear(x, T, H, w16(w.in_proj_a), g.num_v_heads, a_.data<float>(), s);
    kernels::linear(x, T, H, w16(w.in_proj_b), g.num_v_heads, b_.data<float>(), s);
    kernels::causal_conv1d_silu(qkv_.data<float>(), st.conv.data<float>(), w16(w.conv1d), T, g.conv_dim(),
                                g.conv_kernel, conv_.data<float>(), s);
    kernels::gdn_gates(a_.data<float>(), b_.data<float>(), w.a_log.data<float>(), w16(w.dt_bias), T, g.num_v_heads,
                       g_.data<float>(), beta_.data<float>(), s);
    kernels::gdn_recurrence(conv_.data<float>(), g_.data<float>(), beta_.data<float>(), st.s.data<float>(),
                            core_.data<float>(), T, g.num_k_heads, g.num_v_heads, g.k_head_dim, g.v_head_dim, s);
    // Gated RMSNorm per head, written over the projection input it no longer needs.
    kernels::gated_rms_norm(core_.data<float>(), z_.data<float>(), w.norm.data<float>(),
                            static_cast<float>(c.rms_norm_eps), T * g.num_v_heads, g.v_head_dim, conv_.data<float>(),
                            s);
    kernels::linear(conv_.data<float>(), T, g.value_dim(), w16(w.out_proj), H, mixed_.data<float>(), s);
}

void Executor::emit(const Trace* trace, std::string_view name, const core::Tensor& t, std::size_t rows,
                    std::size_t cols) const {
    if (trace == nullptr) {
        return;
    }
    std::vector<float> host(rows * cols);
    stream_.synchronize();
    core::cuda_check(cudaMemcpy(host.data(), t.raw(), host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    (*trace)(name, host, rows, cols);
}

} // namespace runtime
