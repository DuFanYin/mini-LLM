#include "engine/io.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace engine {
namespace detail {

void write_u64(std::ofstream& out, uint64_t v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

void write_f32(std::ofstream& out, float v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

uint64_t read_u64(std::ifstream& in) {
    uint64_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

float read_f32(std::ifstream& in) {
    float v = 0.0f;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

void write_vec_f32(std::ofstream& out, const std::vector<float>& v) {
    write_u64(out, static_cast<uint64_t>(v.size()));
    if (!v.empty()) {
        out.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    }
}

std::vector<float> read_vec_f32(std::ifstream& in) {
    const uint64_t n = read_u64(in);
    std::vector<float> v(static_cast<size_t>(n), 0.0f);
    if (!v.empty()) {
        in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    }
    return v;
}

void write_linear(std::ofstream& out, const model::LinearWeights& l) {
    write_u64(out, static_cast<uint64_t>(l.in_dim));
    write_u64(out, static_cast<uint64_t>(l.out_dim));
    write_vec_f32(out, l.weight);
    write_vec_f32(out, l.bias);
}

model::LinearWeights read_linear(std::ifstream& in) {
    model::LinearWeights l;
    l.in_dim = static_cast<size_t>(read_u64(in));
    l.out_dim = static_cast<size_t>(read_u64(in));
    l.weight = read_vec_f32(in);
    l.bias = read_vec_f32(in);
    return l;
}

void write_rms(std::ofstream& out, const model::RMSNormWeights& n) {
    write_vec_f32(out, n.weight);
    write_f32(out, n.eps);
}

model::RMSNormWeights read_rms(std::ifstream& in) {
    model::RMSNormWeights n;
    n.weight = read_vec_f32(in);
    n.eps = read_f32(in);
    return n;
}

void write_decoder_layer(std::ofstream& out, const model::DecoderLayerWeights& b) {
    write_rms(out, b.norm1);
    write_linear(out, b.attention.q_proj);
    write_linear(out, b.attention.k_proj);
    write_linear(out, b.attention.v_proj);
    write_linear(out, b.attention.o_proj);
    write_rms(out, b.norm2);
    write_linear(out, b.mlp.gate);
    write_linear(out, b.mlp.up);
    write_linear(out, b.mlp.down);
}

model::DecoderLayerWeights read_decoder_layer(std::ifstream& in) {
    model::DecoderLayerWeights b;
    b.norm1 = read_rms(in);
    b.attention.q_proj = read_linear(in);
    b.attention.k_proj = read_linear(in);
    b.attention.v_proj = read_linear(in);
    b.attention.o_proj = read_linear(in);
    b.norm2 = read_rms(in);
    b.mlp.gate = read_linear(in);
    b.mlp.up = read_linear(in);
    b.mlp.down = read_linear(in);
    return b;
}

} // namespace detail

void save_model(const std::string& path, const model::ModelConfig& config, const model::ModelWeights& weights) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("save_model: cannot open file for writing: " + path);
    }

    detail::write_u64(out, 0x4D54524B43485054ULL);
    detail::write_u64(out, 1);

    detail::write_u64(out, static_cast<uint64_t>(config.d_model));
    detail::write_u64(out, static_cast<uint64_t>(config.num_heads));
    detail::write_u64(out, static_cast<uint64_t>(config.num_kv_heads));
    detail::write_u64(out, static_cast<uint64_t>(config.head_dim));
    detail::write_u64(out, static_cast<uint64_t>(config.d_ff));
    detail::write_f32(out, config.rope_base);
    detail::write_u64(out, static_cast<uint64_t>(config.rope_dim));
    detail::write_f32(out, config.rms_norm_eps);

    detail::write_u64(out, static_cast<uint64_t>(weights.vocab_size));
    detail::write_u64(out, static_cast<uint64_t>(weights.layers.size()));
    for (const auto& layer : weights.layers) {
        detail::write_decoder_layer(out, layer);
    }
    detail::write_vec_f32(out, weights.token_embedding);
    detail::write_vec_f32(out, weights.lm_head);

    if (!out.good()) {
        throw std::runtime_error("save_model: failed while writing: " + path);
    }
}

SavedModel load_model(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("load_model: cannot open file for reading: " + path);
    }

    const uint64_t magic = detail::read_u64(in);
    const uint64_t version = detail::read_u64(in);
    if (magic != 0x4D54524B43485054ULL) {
        throw std::runtime_error("load_model: invalid model file magic");
    }
    if (version != 1) {
        throw std::runtime_error("load_model: unsupported model file version");
    }

    SavedModel cp;
    cp.config.d_model = static_cast<size_t>(detail::read_u64(in));
    cp.config.num_heads = static_cast<size_t>(detail::read_u64(in));
    cp.config.num_kv_heads = static_cast<size_t>(detail::read_u64(in));
    cp.config.head_dim = static_cast<size_t>(detail::read_u64(in));
    cp.config.d_ff = static_cast<size_t>(detail::read_u64(in));
    cp.config.rope_base = detail::read_f32(in);
    cp.config.rope_dim = static_cast<size_t>(detail::read_u64(in));
    cp.config.rms_norm_eps = detail::read_f32(in);

    cp.weights.vocab_size = static_cast<size_t>(detail::read_u64(in));
    const size_t num_layers = static_cast<size_t>(detail::read_u64(in));
    cp.weights.layers.resize(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        cp.weights.layers[i] = detail::read_decoder_layer(in);
    }
    cp.weights.token_embedding = detail::read_vec_f32(in);
    cp.weights.lm_head = detail::read_vec_f32(in);

    if (!in.good() && !in.eof()) {
        throw std::runtime_error("load_model: failed while reading: " + path);
    }
    return cp;
}

} // namespace engine
