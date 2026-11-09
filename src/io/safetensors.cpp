#include "io/safetensors.h"

#include "io/json.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <set>
#include <stdexcept>
#include <string_view>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace io {

namespace {

core::DType parse_dtype(const std::string& s, const std::string& name) {
    if (s == "F32") {
        return core::DType::f32;
    }
    if (s == "BF16") {
        return core::DType::bf16;
    }
    if (s == "I32") {
        return core::DType::i32;
    }
    throw std::runtime_error(std::format("safetensors: {} has unsupported dtype {}", name, s));
}

} // namespace

SafetensorsFile::SafetensorsFile(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(std::format("safetensors: cannot open {}", path));
    }
    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        throw std::runtime_error(std::format("safetensors: cannot stat {}", path));
    }
    size_ = static_cast<std::size_t>(st.st_size);
    if (size_ < 8) {
        ::close(fd);
        throw std::runtime_error(std::format("safetensors: {} is too small", path));
    }
    map_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (map_ == MAP_FAILED) {
        map_ = nullptr;
        throw std::runtime_error(std::format("safetensors: cannot mmap {}", path));
    }

    try {
        const auto* base = static_cast<const std::byte*>(map_);
        std::uint64_t header_len = 0;
        std::memcpy(&header_len, base, 8); // little-endian on every supported host
        if (header_len > size_ - 8) {
            throw std::runtime_error("header length exceeds file size");
        }
        const Json header = Json::parse(
            std::string_view(reinterpret_cast<const char*>(base + 8), static_cast<std::size_t>(header_len)));
        const std::byte* data_start = base + 8 + header_len;
        const std::size_t data_size = size_ - 8 - static_cast<std::size_t>(header_len);

        for (const auto& [name, info] : header.as_object()) {
            if (name == "__metadata__") {
                continue;
            }
            TensorEntry e;
            e.name = name;
            e.dtype = parse_dtype(info["dtype"].as_string(), name);
            std::size_t numel = 1;
            for (const Json& d : info["shape"].as_array()) {
                e.shape.push_back(static_cast<std::size_t>(d.as_int()));
                numel *= e.shape.back();
            }
            const auto& offsets = info["data_offsets"].as_array();
            if (offsets.size() != 2) {
                throw std::runtime_error(std::format("{}: data_offsets must have 2 entries", name));
            }
            const auto begin = static_cast<std::size_t>(offsets[0].as_int());
            const auto end = static_cast<std::size_t>(offsets[1].as_int());
            if (end < begin || end > data_size) {
                throw std::runtime_error(std::format("{}: data_offsets out of range", name));
            }
            e.bytes = end - begin;
            if (e.bytes != numel * core::dtype_size(e.dtype)) {
                throw std::runtime_error(std::format("{}: byte size does not match shape", name));
            }
            e.data = data_start + begin;
            entries_.push_back(std::move(e));
        }
    } catch (const std::runtime_error& e) {
        ::munmap(map_, size_);
        map_ = nullptr;
        throw std::runtime_error(std::format("safetensors: {}: {}", path, e.what()));
    }
}

SafetensorsFile::~SafetensorsFile() {
    if (map_ != nullptr) {
        ::munmap(map_, size_);
    }
}

Checkpoint::Checkpoint(const std::string& dir) {
    namespace fs = std::filesystem;
    const fs::path root(dir);
    std::set<std::string> shards;
    const fs::path index = root / "model.safetensors.index.json";
    if (fs::exists(index)) {
        for (const auto& [name, file] : Json::parse_file(index.string())["weight_map"].as_object()) {
            shards.insert(file.as_string());
        }
    } else {
        shards.insert("model.safetensors");
    }

    for (const std::string& shard : shards) {
        files_.push_back(std::make_unique<SafetensorsFile>((root / shard).string()));
        for (const TensorEntry& e : files_.back()->entries()) {
            if (!by_name_.emplace(e.name, &e).second) {
                throw std::runtime_error(std::format("checkpoint: tensor {} appears in more than one shard", e.name));
            }
        }
    }
}

const TensorEntry* Checkpoint::find(std::string_view name) const {
    auto it = by_name_.find(name);
    return it == by_name_.end() ? nullptr : it->second;
}

const TensorEntry& Checkpoint::at(std::string_view name) const {
    const TensorEntry* e = find(name);
    if (e == nullptr) {
        throw std::runtime_error(std::format("checkpoint: missing tensor {}", name));
    }
    return *e;
}

std::vector<const TensorEntry*> Checkpoint::entries() const {
    std::vector<const TensorEntry*> out;
    out.reserve(by_name_.size());
    for (const auto& [name, e] : by_name_) {
        out.push_back(e);
    }
    std::ranges::sort(out, {}, &TensorEntry::name);
    return out;
}

} // namespace io
