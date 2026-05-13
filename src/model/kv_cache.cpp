#include "model/kv_cache.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace model {

// Construction / lifecycle -----------------------------------------------------

KVCache::KVCache(const KVCacheConfig& cfg) : cfg_(cfg) {
    if (cfg_.num_layers == 0 || cfg_.num_kv_heads == 0 || cfg_.max_seq_len == 0 || cfg_.head_dim == 0 ||
        cfg_.page_size == 0) {
        throw std::invalid_argument("KVCache: invalid config");
    }
    pages_per_layer_ = ceil_div(cfg_.max_seq_len, cfg_.page_size);
    const size_t padded_seq_len = pages_per_layer_ * cfg_.page_size;
    elems_per_page_ = cfg_.num_kv_heads * padded_seq_len * cfg_.head_dim;
    k_pages_.assign(cfg_.num_layers * elems_per_page_, 0.0f);
    v_pages_.assign(cfg_.num_layers * elems_per_page_, 0.0f);
    seq_lens_.assign(cfg_.num_layers, 0);
}

void KVCache::reset() {
    std::fill(k_pages_.begin(), k_pages_.end(), 0.0f);
    std::fill(v_pages_.begin(), v_pages_.end(), 0.0f);
    std::fill(seq_lens_.begin(), seq_lens_.end(), 0);
}

void KVCache::reset_layer(size_t layer_id) {
    seq_lens_[layer_id] = 0;
}

// Shape / state queries --------------------------------------------------------

size_t KVCache::seq_len(size_t layer_id) const {
    return seq_lens_[layer_id];
}

size_t KVCache::max_seq_len() const noexcept {
    return cfg_.max_seq_len;
}

size_t KVCache::num_layers() const noexcept {
    return cfg_.num_layers;
}

size_t KVCache::num_kv_heads() const noexcept {
    return cfg_.num_kv_heads;
}

size_t KVCache::head_dim() const noexcept {
    return cfg_.head_dim;
}

size_t KVCache::page_size() const noexcept {
    return cfg_.page_size;
}

size_t KVCache::pages_per_layer() const noexcept {
    return pages_per_layer_;
}

// Append ----------------------------------------------------------------------

void KVCache::append(size_t layer_id, const float* k_new, const float* v_new, size_t seq_len) {
    const size_t hkv = cfg_.num_kv_heads;
    const size_t d = cfg_.head_dim;

    const size_t past = seq_lens_[layer_id];
    if (past + seq_len > cfg_.max_seq_len) {
        throw std::out_of_range("KVCache::append: cache overflow");
    }

    for (size_t si = 0; si < seq_len; ++si) {
        const size_t pos = past + si;
        for (size_t h = 0; h < hkv; ++h) {
            for (size_t di = 0; di < d; ++di) {
                const size_t src = idx3(si, h, di, hkv, d);
                k_pages_[idx4(layer_id, h, pos, di)] = k_new[src];
                v_pages_[idx4(layer_id, h, pos, di)] = v_new[src];
            }
        }
    }
    seq_lens_[layer_id] = past + seq_len;
}

// Read / view -----------------------------------------------------------------

float KVCache::k_at(size_t layer, size_t hkv, size_t pos, size_t d) const {
    return k_pages_[idx4(layer, hkv, pos, d)];
}

float KVCache::v_at(size_t layer, size_t hkv, size_t pos, size_t d) const {
    return v_pages_[idx4(layer, hkv, pos, d)];
}

const float* KVCache::k_layer_data(size_t layer_id) const {
    return k_pages_.data() + layer_id * elems_per_page_;
}

const float* KVCache::v_layer_data(size_t layer_id) const {
    return v_pages_.data() + layer_id * elems_per_page_;
}

CacheView KVCache::view(size_t layer_id) const {
    CacheView v;
    v.layer_id = layer_id;
    const size_t total_kv_len = seq_lens_[layer_id];
    v.past_len = total_kv_len;
    v.total_kv_len = total_kv_len;
    v.kv_stride = pages_per_layer_ * cfg_.page_size;
    v.k_cache = k_pages_.data() + layer_id * elems_per_page_;
    v.v_cache = v_pages_.data() + layer_id * elems_per_page_;
    return v;
}

// Index helpers ---------------------------------------------------------------

size_t KVCache::idx3(size_t a, size_t b, size_t c, size_t B, size_t C) {
    return (a * B + b) * C + c;
}

size_t KVCache::ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

size_t KVCache::idx4(size_t layer, size_t hkv, size_t pos, size_t d) const {
    const size_t padded_seq_len = pages_per_layer_ * cfg_.page_size;
    return layer * elems_per_page_ + (hkv * padded_seq_len + pos) * cfg_.head_dim + d;
}

} // namespace model
