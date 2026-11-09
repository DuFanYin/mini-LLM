#pragma once

#include "core/dtype.h"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace io {

struct TensorEntry {
    std::string name;
    core::DType dtype = core::DType::f32;
    std::vector<std::size_t> shape;
    const std::byte* data = nullptr; // points into the mapped file
    std::size_t bytes = 0;
};

// One read-only, memory-mapped .safetensors file. F32, BF16 and I32 tensors are
// accepted. Errors throw std::runtime_error.
class SafetensorsFile {
public:
    explicit SafetensorsFile(const std::string& path);
    ~SafetensorsFile();
    SafetensorsFile(const SafetensorsFile&) = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;

    const std::vector<TensorEntry>& entries() const { return entries_; }

private:
    void* map_ = nullptr;
    std::size_t size_ = 0;
    std::vector<TensorEntry> entries_;
};

// All shards of a Hugging Face checkpoint directory, found through
// model.safetensors.index.json, or a lone model.safetensors.
class Checkpoint {
public:
    explicit Checkpoint(const std::string& dir);

    const TensorEntry* find(std::string_view name) const;
    const TensorEntry& at(std::string_view name) const; // throws if absent
    std::vector<const TensorEntry*> entries() const;    // sorted by name

private:
    std::vector<std::unique_ptr<SafetensorsFile>> files_;
    std::unordered_map<std::string_view, const TensorEntry*> by_name_;
};

} // namespace io
