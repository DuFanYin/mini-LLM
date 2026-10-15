#pragma once

#include <cstddef>
#include <string>

namespace core {

struct DeviceInfo {
    int ordinal = 0;
    std::string name;
    int cc_major = 0;
    int cc_minor = 0;
    int sm_count = 0;
    std::size_t total_bytes = 0;
    std::size_t free_bytes = 0;
};

DeviceInfo current_device();

} // namespace core
