#include "engine/kv_cache.h"
#include "engine/validation.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace engine {

KVCache::KVCache(const KVCacheConfig& cfg) : cfg_(cfg) {
    validate_kv_cache_config(cfg_);
    pages_per_layer_ = ceil_div(cfg_.max_seq_len, cfg_.page_size);
    elems_per_page_ = cfg_.num_kv_heads * cfg_.page_size * cfg_.head_dim;
    const size_t total_pages = cfg_.num_layers * pages_per_layer_;
    k_pages_.assign(total_pages * elems_per_page_, 0.0f);
    v_pages_.assign(total_pages * elems_per_page_, 0.0f);
    seq_lens_.assign(cfg_.num_layers, 0);
}

void KVCache::reset() {
    std::fill(k_pages_.begin(), k_pages_.end(), 0.0f);
    std::fill(v_pages_.begin(), v_pages_.end(), 0.0f);
    std::fill(seq_lens_.begin(), seq_lens_.end(), 0);
}

void KVCache::reset_layer(size_t layer_id) {
    validate_kv_cache_layer(cfg_, layer_id);
    seq_lens_[layer_id] = 0;
}

size_t KVCache::seq_len(size_t layer_id) const {
    validate_kv_cache_layer(cfg_, layer_id);
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

void KVCache::append_raw(size_t layer_id, const float* k_new, const float* v_new, size_t seq_len) {
    validate_kv_cache_layer(cfg_, layer_id);
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

void KVCache::append(size_t layer_id, const std::vector<float>& k_new, const std::vector<float>& v_new, size_t seq_len) {
    validate_kv_cache_layer(cfg_, layer_id);
    validate_kv_cache_append_input(cfg_, seq_len, k_new, v_new);
    append_raw(layer_id, k_new.data(), v_new.data(), seq_len);
}

float KVCache::k_at(size_t layer, size_t hkv, size_t pos, size_t d) const {
    validate_kv_cache_access(cfg_, layer, hkv, pos, d);
    return k_pages_[idx4(layer, hkv, pos, d)];
}

float KVCache::v_at(size_t layer, size_t hkv, size_t pos, size_t d) const {
    validate_kv_cache_access(cfg_, layer, hkv, pos, d);
    return v_pages_[idx4(layer, hkv, pos, d)];
}

const float* KVCache::k_layer_data(size_t layer_id) const {
    validate_kv_cache_layer(cfg_, layer_id);
    const size_t page_slot = layer_id * pages_per_layer_;
    return k_pages_.data() + page_slot * elems_per_page_;
}

const float* KVCache::v_layer_data(size_t layer_id) const {
    validate_kv_cache_layer(cfg_, layer_id);
    const size_t page_slot = layer_id * pages_per_layer_;
    return v_pages_.data() + page_slot * elems_per_page_;
}

size_t KVCache::idx3(size_t a, size_t b, size_t c, size_t B, size_t C) {
    return (a * B + b) * C + c;
}

size_t KVCache::ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

size_t KVCache::idx4(size_t layer, size_t hkv, size_t pos, size_t d) const {
    const size_t page_id = pos / cfg_.page_size;
    const size_t in_page = pos % cfg_.page_size;

    const size_t page_slot = layer * pages_per_layer_ + page_id;
    const size_t token_stride = cfg_.num_kv_heads * cfg_.head_dim;

    return page_slot * elems_per_page_ + in_page * token_stride + hkv * cfg_.head_dim + d;
}

model::CacheBridge make_cache_bridge(KVCache& cache) {
    model::CacheBridge bridge;
    bridge.num_layers = cache.num_layers();
    bridge.num_kv_heads = cache.num_kv_heads();
    bridge.head_dim = cache.head_dim();
    bridge.opaque_cache = &cache;
    return bridge;
}

size_t cache_seq_len(const model::CacheBridge& cache, size_t layer_id) {
    const auto* kv = static_cast<const KVCache*>(cache.opaque_cache);
    return kv->seq_len(layer_id);
}

void reset_cache(const model::CacheBridge& cache) {
    auto* kv = static_cast<KVCache*>(cache.opaque_cache);
    kv->reset();
}

void append_cache(const model::CacheBridge& cache, size_t layer_id, const float* k_new, const float* v_new, size_t seq_len) {
    auto* kv = static_cast<KVCache*>(cache.opaque_cache);
    kv->append_raw(layer_id, k_new, v_new, seq_len);
}

model::CacheView build_cache_view(const model::CacheBridge& cache, size_t layer_id) {
    model::CacheView view;
    view.layer_id = layer_id;
    const auto* kv = static_cast<const KVCache*>(cache.opaque_cache);
    const size_t total_kv_len = kv->seq_len(layer_id);
    view.total_kv_len = total_kv_len;
    view.past_len = total_kv_len;
    view.k_cache = kv->k_layer_data(layer_id);
    view.v_cache = kv->v_layer_data(layer_id);
    return view;
}

} // namespace engine
