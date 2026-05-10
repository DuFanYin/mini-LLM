#pragma once

#include "model/model_types.h"

#include <cstddef>
#include <vector>

namespace engine {

struct KVCacheConfig {
    size_t num_layers = 0;
    size_t num_kv_heads = 0;
    size_t max_seq_len = 0;
    size_t head_dim = 0;
    size_t page_size = 16;
};

class KVCache {
public:
    explicit KVCache(const KVCacheConfig& cfg);
    void reset();
    void reset_layer(size_t layer_id);
    [[nodiscard]] size_t seq_len(size_t layer_id) const;
    [[nodiscard]] size_t max_seq_len() const noexcept;
    [[nodiscard]] size_t num_layers() const noexcept;
    [[nodiscard]] size_t num_kv_heads() const noexcept;
    [[nodiscard]] size_t head_dim() const noexcept;
    [[nodiscard]] size_t page_size() const noexcept;
    [[nodiscard]] size_t pages_per_layer() const noexcept;
    void append_raw(size_t layer_id, const float* k_new, const float* v_new, size_t seq_len);
    void append(size_t layer_id, const std::vector<float>& k_new, const std::vector<float>& v_new, size_t seq_len);
    [[nodiscard]] float k_at(size_t layer, size_t hkv, size_t pos, size_t d) const;
    [[nodiscard]] float v_at(size_t layer, size_t hkv, size_t pos, size_t d) const;
    [[nodiscard]] const float* k_layer_data(size_t layer_id) const;
    [[nodiscard]] const float* v_layer_data(size_t layer_id) const;

private:
    static size_t idx3(size_t a, size_t b, size_t c, size_t B, size_t C);
    static size_t ceil_div(size_t a, size_t b);
    size_t idx4(size_t layer, size_t hkv, size_t pos, size_t d) const;

    KVCacheConfig cfg_;
    size_t pages_per_layer_ = 0;
    size_t elems_per_page_ = 0;
    std::vector<float> k_pages_;
    std::vector<float> v_pages_;
    std::vector<size_t> seq_lens_;
};

[[nodiscard]] model::CacheBridge make_cache_bridge(KVCache& cache);
[[nodiscard]] size_t cache_seq_len(const model::CacheBridge& cache, size_t layer_id);
void reset_cache(const model::CacheBridge& cache);
void append_cache(const model::CacheBridge& cache, size_t layer_id, const float* k_new, const float* v_new, size_t seq_len);
[[nodiscard]] model::CacheView build_cache_view(const model::CacheBridge& cache, size_t layer_id);

} // namespace engine
