#include "model/weights.h"

#include "io/safetensors.h"

#include <chrono>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>

namespace model {

namespace {

constexpr std::string_view kTextPrefix = "model.language_model.";
constexpr std::string_view kSkippedPrefixes[] = {"model.visual.", "mtp."};

using core::DType;
using Shape = std::vector<std::size_t>;

// One tensor the engine expects: where it comes from and where it goes.
struct Spec {
    std::string name;
    Shape shape;
    DType dtype;
    Part part;
    core::Tensor* dest;
};

std::string shape_str(const Shape& s) {
    std::string out = "[";
    for (std::size_t i = 0; i < s.size(); ++i) {
        out += std::format("{}{}", i ? ", " : "", s[i]);
    }
    return out + "]";
}

void add_full_attention(std::vector<Spec>& specs, const std::string& p, const Config& c, FullAttentionWeights& w) {
    const auto& a = c.attn;
    const std::size_t h = c.hidden_size;
    specs.push_back(
        {p + "q_proj.weight", {a.num_heads * a.head_dim * 2, h}, DType::bf16, Part::full_attention, &w.q_proj});
    specs.push_back(
        {p + "k_proj.weight", {a.num_kv_heads * a.head_dim, h}, DType::bf16, Part::full_attention, &w.k_proj});
    specs.push_back(
        {p + "v_proj.weight", {a.num_kv_heads * a.head_dim, h}, DType::bf16, Part::full_attention, &w.v_proj});
    specs.push_back({p + "o_proj.weight", {h, a.num_heads * a.head_dim}, DType::bf16, Part::full_attention, &w.o_proj});
    specs.push_back({p + "q_norm.weight", {a.head_dim}, DType::bf16, Part::full_attention, &w.q_norm});
    specs.push_back({p + "k_norm.weight", {a.head_dim}, DType::bf16, Part::full_attention, &w.k_norm});
}

void add_gated_delta_net(std::vector<Spec>& specs, const std::string& p, const Config& c, GatedDeltaNetWeights& w) {
    const auto& g = c.gdn;
    const std::size_t h = c.hidden_size;
    specs.push_back({p + "in_proj_qkv.weight", {g.conv_dim(), h}, DType::bf16, Part::gated_delta_net, &w.in_proj_qkv});
    specs.push_back({p + "in_proj_z.weight", {g.value_dim(), h}, DType::bf16, Part::gated_delta_net, &w.in_proj_z});
    specs.push_back({p + "in_proj_a.weight", {g.num_v_heads, h}, DType::bf16, Part::gated_delta_net, &w.in_proj_a});
    specs.push_back({p + "in_proj_b.weight", {g.num_v_heads, h}, DType::bf16, Part::gated_delta_net, &w.in_proj_b});
    specs.push_back(
        {p + "conv1d.weight", {g.conv_dim(), 1, g.conv_kernel}, DType::bf16, Part::gated_delta_net, &w.conv1d});
    specs.push_back({p + "dt_bias", {g.num_v_heads}, DType::bf16, Part::gated_delta_net, &w.dt_bias});
    specs.push_back({p + "A_log", {g.num_v_heads}, DType::f32, Part::gated_delta_net, &w.a_log});
    specs.push_back({p + "norm.weight", {g.v_head_dim}, DType::f32, Part::gated_delta_net, &w.norm});
    specs.push_back({p + "out_proj.weight", {h, g.value_dim()}, DType::bf16, Part::gated_delta_net, &w.out_proj});
}

bool is_skipped(std::string_view name) {
    for (std::string_view prefix : kSkippedPrefixes) {
        if (name.starts_with(prefix)) {
            return true;
        }
    }
    return false;
}

} // namespace

Weights Weights::load(const Config& c, const io::Checkpoint& ckpt) {
    const auto t0 = std::chrono::steady_clock::now();
    Weights w;
    const std::size_t h = c.hidden_size;
    const std::string root(kTextPrefix);

    // Size the containers first so the Spec destinations stay valid.
    w.layers_.resize(c.num_layers());
    std::vector<Spec> specs;
    specs.push_back({root + "embed_tokens.weight", {c.vocab_size, h}, DType::bf16, Part::embedding, &w.embed_tokens_});
    specs.push_back({root + "norm.weight", {h}, DType::bf16, Part::norm, &w.final_norm_});
    if (!c.tie_word_embeddings) {
        specs.push_back({"lm_head.weight", {c.vocab_size, h}, DType::bf16, Part::embedding, &w.lm_head_});
    }
    for (std::size_t i = 0; i < c.num_layers(); ++i) {
        LayerWeights& layer = w.layers_[i];
        const std::string p = std::format("{}layers.{}.", root, i);
        specs.push_back({p + "input_layernorm.weight", {h}, DType::bf16, Part::norm, &layer.input_norm});
        specs.push_back(
            {p + "post_attention_layernorm.weight", {h}, DType::bf16, Part::norm, &layer.post_attention_norm});
        if (c.layers[i] == LayerKind::full_attention) {
            add_full_attention(specs, p + "self_attn.", c, layer.mixer.emplace<FullAttentionWeights>());
        } else {
            add_gated_delta_net(specs, p + "linear_attn.", c, layer.mixer.emplace<GatedDeltaNetWeights>());
        }
        specs.push_back(
            {p + "mlp.gate_proj.weight", {c.intermediate_size, h}, DType::bf16, Part::mlp, &layer.mlp.gate_proj});
        specs.push_back(
            {p + "mlp.up_proj.weight", {c.intermediate_size, h}, DType::bf16, Part::mlp, &layer.mlp.up_proj});
        specs.push_back(
            {p + "mlp.down_proj.weight", {h, c.intermediate_size}, DType::bf16, Part::mlp, &layer.mlp.down_proj});
    }

    // Validate everything before touching the device.
    std::vector<const io::TensorEntry*> sources;
    std::unordered_set<std::string_view> consumed;
    std::size_t arena_bytes = 0;
    for (const Spec& s : specs) {
        const io::TensorEntry& e = ckpt.at(s.name);
        if (e.shape != s.shape) {
            throw std::runtime_error(
                std::format("weights: {} has shape {}, expected {}", s.name, shape_str(e.shape), shape_str(s.shape)));
        }
        if (e.dtype != s.dtype) {
            throw std::runtime_error(std::format("weights: {} is {}, expected {}", s.name, core::dtype_name(e.dtype),
                                                 core::dtype_name(s.dtype)));
        }
        sources.push_back(&e);
        consumed.insert(e.name);
        arena_bytes += core::Arena::aligned(e.bytes);
    }
    for (const io::TensorEntry* e : ckpt.entries()) {
        if (consumed.contains(e->name)) {
            continue;
        }
        if (!is_skipped(e->name)) {
            throw std::runtime_error(std::format("weights: checkpoint tensor {} is not used by the engine", e->name));
        }
        ++w.report_.skipped_tensors;
        w.report_.skipped_params += e->bytes / core::dtype_size(e->dtype);
    }

    w.arena_ = std::make_unique<core::Arena>(arena_bytes);
    for (std::size_t i = 0; i < specs.size(); ++i) {
        const io::TensorEntry& e = *sources[i];
        *specs[i].dest = core::Tensor(e.shape, e.dtype, *w.arena_);
        specs[i].dest->copy_from_host(e.data);
        ++w.report_.tensors;
        w.report_.params += specs[i].dest->numel();
        w.report_.params_by_part[static_cast<std::size_t>(specs[i].part)] += specs[i].dest->numel();
        w.report_.bytes += e.bytes;
    }
    w.report_.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return w;
}

} // namespace model
