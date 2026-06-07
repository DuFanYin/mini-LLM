#include "model/device_executor.h"

#include "kernel/cuda/launch.h"
#include "model/cache.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace model {

namespace {
core::Tensor upload_vec(const std::vector<float>& src, std::vector<std::size_t> shape, core::PoolAllocator& alloc) {
    core::Tensor t(std::move(shape), alloc);
    if (t.numel() != 0) {
        t.copy_from_host(src.data());
    }
    return t;
}
} // namespace

DeviceModel::DeviceLinear DeviceModel::upload_linear(const LinearWeights& src) {
    DeviceLinear out;
    out.weight = upload_vec(src.weight, {src.out_dim, src.in_dim}, alloc_);
    if (!src.bias.empty()) {
        out.bias = upload_vec(src.bias, {src.out_dim}, alloc_);
        out.has_bias = true;
    }
    return out;
}

DeviceModel::DeviceModel(const ModelConfig& config, const ModelWeights& weights, std::size_t max_seq_len)
    : config_(config), max_seq_len_(max_seq_len) {
    // RoPE tables: reuse the host RopeCache so device values match the scalar path exactly.
    RopeCache rope(config_.rope_base, config_.head_dim, config_.rope_dim);
    rope.ensure(max_seq_len_);
    rot_ = rope.rot_dim();
    half_ = rot_ / 2;
    eff_rot_ = std::min(rot_, config_.head_dim - (config_.head_dim % 2));
    if (half_ != 0) {
        cos_ = core::Tensor({max_seq_len_, half_}, alloc_);
        sin_ = core::Tensor({max_seq_len_, half_}, alloc_);
        cos_.copy_from_host(rope.cos_data());
        sin_.copy_from_host(rope.sin_data());
    }

    layers_.reserve(weights.layers.size());
    for (const DecoderLayerWeights& lw : weights.layers) {
        DeviceLayer dl;
        dl.norm1 = upload_vec(lw.norm1.weight, {config_.d_model}, alloc_);
        dl.q = upload_linear(lw.attention.q_proj);
        dl.k = upload_linear(lw.attention.k_proj);
        dl.v = upload_linear(lw.attention.v_proj);
        dl.o = upload_linear(lw.attention.o_proj);
        dl.norm2 = upload_vec(lw.norm2.weight, {config_.d_model}, alloc_);
        dl.gate = upload_linear(lw.mlp.gate);
        dl.up = upload_linear(lw.mlp.up);
        dl.down = upload_linear(lw.mlp.down);
        layers_.push_back(std::move(dl));
    }

    const std::size_t kv_elems = config_.num_kv_heads * max_seq_len_ * config_.head_dim;
    k_cache_.reserve(weights.layers.size());
    v_cache_.reserve(weights.layers.size());
    for (std::size_t i = 0; i < weights.layers.size(); ++i) {
        k_cache_.emplace_back(std::vector<std::size_t>{kv_elems}, alloc_);
        v_cache_.emplace_back(std::vector<std::size_t>{kv_elems}, alloc_);
    }
}

void DeviceModel::forward(const float* hidden_in, std::size_t seq_len, float* hidden_out) {
    if (seq_len == 0) {
        return;
    }
    const std::size_t d_model = config_.d_model;
    const std::size_t nh = config_.num_heads;
    const std::size_t hkv = config_.num_kv_heads;
    const std::size_t hd = config_.head_dim;
    const std::size_t d_ff = config_.d_ff;
    const std::size_t q_dim = nh * hd;
    const std::size_t kv_dim = hkv * hd;
    const float eps = config_.rms_norm_eps;
    const std::size_t past_len = current_len_;
    const std::size_t total_kv = current_len_ + seq_len;
    const float* dcos = (half_ != 0) ? cos_.data() : nullptr;
    const float* dsin = (half_ != 0) ? sin_.data() : nullptr;

    // Ping-pong hidden buffers + per-layer scratch (allocated once, reused across layers).
    core::Tensor buf_a({seq_len, d_model}, alloc_);
    core::Tensor buf_b({seq_len, d_model}, alloc_);
    core::Tensor norm1_out({seq_len, d_model}, alloc_);
    core::Tensor q({seq_len, q_dim}, alloc_);
    core::Tensor k({seq_len, kv_dim}, alloc_);
    core::Tensor v({seq_len, kv_dim}, alloc_);
    core::Tensor ctx({seq_len, q_dim}, alloc_);
    core::Tensor attn_out({seq_len, d_model}, alloc_);
    core::Tensor hidden_after_attn({seq_len, d_model}, alloc_);
    core::Tensor norm2_out({seq_len, d_model}, alloc_);
    core::Tensor gate({seq_len, d_ff}, alloc_);
    core::Tensor up({seq_len, d_ff}, alloc_);
    core::Tensor hidden_mid({seq_len, d_ff}, alloc_);
    core::Tensor mlp_out({seq_len, d_model}, alloc_);

    buf_a.copy_from_host(hidden_in);
    float* cur = buf_a.data();
    float* nxt = buf_b.data();

    for (std::size_t layer = 0; layer < layers_.size(); ++layer) {
        const DeviceLayer& L = layers_[layer];

        // --- attention block ---
        kernel::cuda::rms_norm_device(cur, L.norm1.data(), eps, norm1_out.data(), seq_len, d_model);
        kernel::cuda::linear_device(norm1_out.data(), L.q.weight.data(), L.q.has_bias ? L.q.bias.data() : nullptr,
                                    q.data(), seq_len, q_dim, d_model);
        kernel::cuda::linear_device(norm1_out.data(), L.k.weight.data(), L.k.has_bias ? L.k.bias.data() : nullptr,
                                    k.data(), seq_len, kv_dim, d_model);
        kernel::cuda::linear_device(norm1_out.data(), L.v.weight.data(), L.v.has_bias ? L.v.bias.data() : nullptr,
                                    v.data(), seq_len, kv_dim, d_model);

        kernel::cuda::apply_rope_device(q.data(), past_len, dcos, dsin, half_, eff_rot_, seq_len, nh, hd);
        kernel::cuda::apply_rope_device(k.data(), past_len, dcos, dsin, half_, eff_rot_, seq_len, hkv, hd);

        kernel::cuda::kv_append_device(k.data(), v.data(), k_cache_[layer].data(), v_cache_[layer].data(), seq_len,
                                       hkv, hd, past_len, max_seq_len_);
        kernel::cuda::gqa_attention_forward_device(q.data(), k_cache_[layer].data(), v_cache_[layer].data(), nullptr,
                                                   ctx.data(), seq_len, nh, hkv, hd, past_len, total_kv, max_seq_len_,
                                                   /*causal=*/true);

        kernel::cuda::linear_device(ctx.data(), L.o.weight.data(), L.o.has_bias ? L.o.bias.data() : nullptr,
                                    attn_out.data(), seq_len, d_model, q_dim);
        kernel::cuda::add_device(cur, attn_out.data(), hidden_after_attn.data(), seq_len * d_model);

        // --- MLP block ---
        kernel::cuda::rms_norm_device(hidden_after_attn.data(), L.norm2.data(), eps, norm2_out.data(), seq_len,
                                      d_model);
        kernel::cuda::linear_device(norm2_out.data(), L.gate.weight.data(),
                                    L.gate.has_bias ? L.gate.bias.data() : nullptr, gate.data(), seq_len, d_ff,
                                    d_model);
        kernel::cuda::linear_device(norm2_out.data(), L.up.weight.data(), L.up.has_bias ? L.up.bias.data() : nullptr,
                                    up.data(), seq_len, d_ff, d_model);
        kernel::cuda::silu_mul_device(gate.data(), up.data(), hidden_mid.data(), seq_len * d_ff);
        kernel::cuda::linear_device(hidden_mid.data(), L.down.weight.data(),
                                    L.down.has_bias ? L.down.bias.data() : nullptr, mlp_out.data(), seq_len, d_model,
                                    d_ff);
        kernel::cuda::add_device(hidden_after_attn.data(), mlp_out.data(), nxt, seq_len * d_model);

        std::swap(cur, nxt);
    }

    // `cur` holds the final hidden after the last swap.
    core::Tensor& final_buf = (cur == buf_a.data()) ? buf_a : buf_b;
    final_buf.copy_to_host(hidden_out);
    current_len_ += seq_len;
}

} // namespace model
