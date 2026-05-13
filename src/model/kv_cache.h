#pragma once

// Single owner of all KV-cache concerns: storage layout, append/lookup, per-layer
// view, and the validation helpers that protect those operations. Executor and
// mini_llm code talk to KVCache directly; no opaque-pointer bridge layer exists.

#include <cstddef>
#include <vector>

namespace model {

// Static allocation parameters for the cache backing store.
struct KVCacheConfig {
    size_t num_layers = 0;
    size_t num_kv_heads = 0;
    size_t max_seq_len = 0;
    size_t head_dim = 0;
    size_t page_size = 16;
};

// Per-layer slice of the KV cache as used by the attention kernels. The pointers
// alias storage owned by the originating KVCache instance and stay valid until
// the cache is reset, reconfigured, or destroyed.
struct CacheView {
    size_t layer_id = 0;
    size_t past_len = 0;
    size_t total_kv_len = 0;
    size_t kv_stride = 0;
    const float* k_cache = nullptr;
    const float* v_cache = nullptr;
};

class KVCache {
public:
    // Construction / lifecycle.
    explicit KVCache(const KVCacheConfig& cfg);
    void reset();
    void reset_layer(size_t layer_id);

    // Shape / state queries.
    [[nodiscard]] size_t seq_len(size_t layer_id) const;
    [[nodiscard]] size_t max_seq_len() const noexcept;
    [[nodiscard]] size_t num_layers() const noexcept;
    [[nodiscard]] size_t num_kv_heads() const noexcept;
    [[nodiscard]] size_t head_dim() const noexcept;
    [[nodiscard]] size_t page_size() const noexcept;
    [[nodiscard]] size_t pages_per_layer() const noexcept;

    // Append K/V rows in [seq_len, num_kv_heads, head_dim] order.
    void append(size_t layer_id, const float* k_new, const float* v_new, size_t seq_len);

    // Read individual entries or layer-major contiguous backing spans.
    [[nodiscard]] float k_at(size_t layer, size_t hkv, size_t pos, size_t d) const;
    [[nodiscard]] float v_at(size_t layer, size_t hkv, size_t pos, size_t d) const;
    [[nodiscard]] const float* k_layer_data(size_t layer_id) const;
    [[nodiscard]] const float* v_layer_data(size_t layer_id) const;

    // Snapshot of the current K/V span for one layer (covers prefix already in cache;
    // call again after append to obtain a span that includes new tokens).
    [[nodiscard]] CacheView view(size_t layer_id) const;

private:
    // Index helpers.
    static size_t idx3(size_t a, size_t b, size_t c, size_t B, size_t C);
    static size_t ceil_div(size_t a, size_t b);
    size_t idx4(size_t layer, size_t hkv, size_t pos, size_t d) const;

    // Layout metadata.
    KVCacheConfig cfg_;
    size_t pages_per_layer_ = 0;
    size_t elems_per_page_ = 0;

    // Backing store: [layer, hkv, padded_pos, head_dim].
    std::vector<float> k_pages_;
    std::vector<float> v_pages_;
    std::vector<size_t> seq_lens_;
};

} // namespace model
