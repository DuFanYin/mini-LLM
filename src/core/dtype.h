#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace core {

enum class DType : std::uint8_t { f32, bf16, i32 };

constexpr std::size_t dtype_size(DType t) noexcept {
    switch (t) {
    case DType::bf16:
        return 2;
    case DType::f32:
    case DType::i32:
        return 4;
    }
    return 0;
}

constexpr std::string_view dtype_name(DType t) noexcept {
    switch (t) {
    case DType::f32:
        return "f32";
    case DType::bf16:
        return "bf16";
    case DType::i32:
        return "i32";
    }
    return "?";
}

} // namespace core
