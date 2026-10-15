#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace core {

enum class DType : std::uint8_t { f32, bf16 };

constexpr std::size_t dtype_size(DType t) noexcept {
    return t == DType::f32 ? 4u : 2u;
}

constexpr std::string_view dtype_name(DType t) noexcept {
    return t == DType::f32 ? "f32" : "bf16";
}

} // namespace core
